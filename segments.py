"""The numbered transcript that every input becomes, and its download formats."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    id: int
    start: Optional[float]  # seconds; None when the source has no times (e.g. notes)
    end: Optional[float]
    text: str


def fmt_ts(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def has_times(segments: list[Segment]) -> bool:
    return any(s.start is not None for s in segments)


def transcript_txt(segments: list[Segment]) -> str:
    """Plain-text download: one numbered line each, with its time when known."""
    out = []
    for s in segments:
        ts = f"[{fmt_ts(s.start)}] " if s.start is not None else ""
        out.append(f"[{s.id}] {ts}{s.text}")
    return "\n".join(out) + "\n"


def transcript_md(segments: list[Segment], title: str = "Transcript") -> str:
    """Markdown download, readable in any editor."""
    out = [f"# {title}", ""]
    for s in segments:
        ts = f"`{fmt_ts(s.start)}` " if s.start is not None else ""
        out.append(f"{s.id}. {ts}{s.text}")
    out += ["", "_Created on this device by SNAP Minutes. Nothing was uploaded._"]
    return "\n".join(out) + "\n"
