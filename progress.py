"""Progress, percent done and ETA for one "Process meeting" run.

A run has a few stages (transcribe or read, build the Q&A index, write bullet points).
Each stage starts with an estimate in seconds, from measured speeds on this laptop.
While a stage runs it reports a fraction done (Whisper's position in the audio,
embedding batches finished, tokens written). The remaining time for the active stage
blends the estimate with the measured rate, trusting the measurement more as the
stage advances. Pending stages count at their estimate.

    percent = elapsed / (elapsed + remaining)

Percent is kept non-decreasing and never reaches 100% until every stage has finished.

Thread-safe: a worker thread calls update(), and the UI thread calls snapshot()."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Stage:
    key: str
    label: str
    estimate: float              # seconds
    state: str = "pending"       # pending | active | done
    frac: float = 0.0
    detail: str = ""
    started: Optional[float] = None
    ended: Optional[float] = None
    rate_t: Optional[float] = None   # where the measured rate is counted from (see rebase)
    rate_f: float = 0.0


@dataclass
class Tracker:
    stages: list[Stage]
    clock: Callable[[], float] = time.perf_counter
    t0: float = field(init=False)
    _last_pct: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.t0 = self.clock()

    def _get(self, key: str) -> Stage:
        for s in self.stages:
            if s.key == key:
                return s
        raise KeyError(key)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            s = self._get(key)
            s.state, s.started, s.frac, s.detail = "active", self.clock(), 0.0, detail
            s.rate_t, s.rate_f = s.started, 0.0

    def set_estimate(self, key: str, seconds: float) -> None:
        with self._lock:
            self._get(key).estimate = max(0.1, float(seconds))

    def update(self, key: str, frac: float, detail: Optional[str] = None) -> None:
        with self._lock:
            s = self._get(key)
            if s.state == "pending":
                s.state, s.started = "active", self.clock()
                s.rate_t, s.rate_f = s.started, 0.0
            s.frac = min(max(float(frac), s.frac), 1.0)  # never moves backwards
            if detail is not None:
                s.detail = detail

    def restart(self, key: str, detail: str = "") -> None:
        """Start a stage over, e.g. when the GPU fails and Whisper retries on the CPU."""
        with self._lock:
            s = self._get(key)
            s.frac, s.detail, s.started = 0.0, detail, self.clock()
            s.rate_t, s.rate_f = s.started, 0.0

    def rebase(self, key: str) -> None:
        """Measure the rate from here on. Used when a stage changes pace part-way, e.g. the
        language model reads the transcript (no progress) and then starts writing tokens:
        counting the reading time as writing time would make the ETA too pessimistic."""
        with self._lock:
            s = self._get(key)
            s.rate_t, s.rate_f = self.clock(), s.frac

    def finish(self, key: str) -> None:
        with self._lock:
            s = self._get(key)
            if s.started is None:
                s.started = self.clock()
            s.state, s.frac, s.ended = "done", 1.0, self.clock()

    # ------------------------------------------------------------------ reading
    def _remaining(self, s: Stage, now: float) -> float:
        if s.state == "done":
            return 0.0
        if s.state == "pending":
            return s.estimate
        spent = now - (now if s.started is None else s.started)
        by_estimate = max(s.estimate - spent, 0.0)
        if self._measured(s, now):
            dt, df = now - s.rate_t, s.frac - s.rate_f
            by_rate = dt * (1 - s.frac) / df
            w = s.frac  # trust the measured rate more as the stage advances
            left = w * by_rate + (1 - w) * by_estimate
        else:
            left = by_estimate
        # Running past the estimate with no new progress: assume a little is left
        # rather than showing 0 s while work is clearly still going on.
        return max(left, 1.0 - s.frac) if s.frac < 1 else 0.0

    @staticmethod
    def _measured(s: Stage, now: float) -> bool:
        """Enough real progress since rate_t to trust the measured rate."""
        return s.rate_t is not None and s.frac - s.rate_f >= 0.03 and now - s.rate_t >= 0.5

    def snapshot(self) -> dict:
        with self._lock:
            now = self.clock()
            elapsed = now - self.t0
            remaining = sum(self._remaining(s, now) for s in self.stages)
            all_done = all(s.state == "done" for s in self.stages)
            pct = 1.0 if all_done else min(elapsed / max(elapsed + remaining, 1e-6), 0.99)
            self._last_pct = pct = max(pct, self._last_pct)
            active = next((s for s in self.stages if s.state == "active"), None)
            # The ETA is "measured" once the active stage has reported real progress.
            measured = active is None or self._measured(active, now)
            return {
                "percent": pct,
                "eta": 0.0 if all_done else remaining,
                "eta_measured": measured,
                "elapsed": elapsed,
                "current": active.key if active else None,
                "label": active.label if active else ("Done" if all_done else ""),
                "detail": active.detail if active else "",
                "done": all_done,
                "stages": [{"key": s.key, "label": s.label, "state": s.state, "frac": s.frac,
                            "seconds": (now if s.ended is None else s.ended) - s.started
                            if s.started is not None else None}
                           for s in self.stages],
            }


def fmt_duration(seconds: float) -> str:
    """12 -> '12 s', 75 -> '1 min 15 s'."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    m, s = divmod(seconds, 60)
    return f"{m} min {s:02d} s"
