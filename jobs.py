"""Processing one meeting: read or transcribe, build the Q&A index, write the bullet points.

Shared by the local app (app.py, which runs it in a background thread) and the PC service
(server.py, which runs it for the hosted Streamlit Cloud page). It reports progress to a
progress.Tracker and never touches Streamlit, so it can run on any thread.

Also here: the starting speeds the ETA uses, how they are updated from each run, and the
conversion of results to and from JSON for the hosted page."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pipeline as P
import rag
import readers
from progress import Stage, Tracker
from segments import Segment, fmt_ts

SAMPLE = Path(__file__).parent / "sample" / "sample_transcript.json"

# Starting speeds for the ETA, measured on the reference laptop (RTX 3060). After each run
# they are replaced with what was measured, so later estimates fit the machine.
DEFAULT_RATES = {
    "whisper_rtf": {"cuda": 0.10, "cpu": 0.45},   # seconds of work per second of audio
    "whisper_load": 4.0,                          # seconds to load Whisper
    "tok_per_s": {"qwen3:4b-instruct": 80.0, "qwen3:1.7b": 140.0, "phi4-mini": 45.0},
    "prompt_chars_per_s": 8000.0,                 # how fast the model reads the transcript
    "llm_load": 16.0,                             # seconds to load a model that isn't in memory
    "embed_batch_s": 0.4,
}


def new_rates() -> dict:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_RATES.items()}


def make_tracker(kind: str, size_bytes: int, rates: dict, upload: bool = False) -> Tracker:
    """The stages for one run. `upload` adds a first stage for sending the file to the PC."""
    video = kind == "video"
    stages = [Stage("upload", "Sending to the PC", 1.0 + size_bytes / 4e6)] if upload else []
    stages += [
        Stage("input", "Transcribing audio" if video else "Reading the input",
              # before Whisper reports the audio length, guess from the file size
              (rates["whisper_load"] + size_bytes / 1e6 * 1.5) if video else 0.4),
        Stage("index", "Building the Q&A index", 1.0),
        Stage("notes", "Writing bullet points", 15.0),
    ]
    return Tracker(stages)


def process(job: dict, tracker: Tracker, rates: dict, llm: str, whisper_size: str) -> dict:
    """Run one meeting. job: {"kind": video|file|sample|text, "name", "data" (bytes), "text"}.
    Returns segments, source, index, notes, timings and measured speeds ("stats")."""
    stats: dict = {}
    t_in = time.perf_counter()
    tracker.begin("input")
    if job["kind"] == "video":
        tracker.update("input", 0.0, "Loading Whisper")

        def on_info(info):  # once per attempt: the audio length is now known
            dev = info["device"]
            stats.update(duration=info["duration"], device=dev, t_audio=time.perf_counter())
            tracker.restart("input", f"Whisper on {'GPU' if dev == 'cuda' else 'CPU'} · "
                                     f"{fmt_ts(info['duration'])} of audio")
            tracker.set_estimate("input", info["duration"] * rates["whisper_rtf"][dev])

        def on_progress(p):
            if stats.get("duration"):
                tracker.update("input", p, f"Whisper on {'GPU' if stats['device'] == 'cuda' else 'CPU'} · "
                                           f"{fmt_ts(p * stats['duration'])} of {fmt_ts(stats['duration'])}")

        segments, meta = P.transcribe(io.BytesIO(job["data"]), model_size=whisper_size,
                                      on_progress=on_progress, on_info=on_info)
        source = f"{job['name']} (Whisper on {'GPU' if meta['device'] == 'cuda' else 'CPU'})"
        stats["gpu_error"] = meta.get("gpu_error")
        stats["vad_off"] = meta.get("vad") is False
    elif job["kind"] == "file":
        tracker.update("input", 0.3, f"Reading {job['name']}")
        segments, source = readers.read_file(job["name"], job["data"]), job["name"]
    elif job["kind"] == "sample":
        tracker.update("input", 0.3, "Reading the sample transcript")
        segments, source = readers.read_file(SAMPLE.name, SAMPLE.read_bytes()), "sample transcript"
    else:
        tracker.update("input", 0.3, "Reading the pasted text")
        segments, source = readers.read_text(job["text"]), "pasted text"
    tracker.finish("input")
    t_idx = stats["t_input_end"] = time.perf_counter()
    stats["input_s"] = t_idx - t_in

    # --- Q&A index
    n_passages = len(rag.make_passages(segments))
    batches = -(-n_passages // 64)
    tracker.set_estimate("index", 0.3 + batches * rates["embed_batch_s"]
                         + (0 if P.model_loaded(rag.EMBED_MODEL) else 2.0))
    tracker.begin("index", f"Embedding {n_passages} passages in memory")
    index = rag.build_index(segments, on_progress=lambda p: tracker.update("index", p))
    tracker.finish("index")
    t_notes = time.perf_counter()

    # --- bullet points
    sizes = P.note_calls(segments)
    loaded = P.model_loaded(llm)
    tok_s = rates["tok_per_s"].get(llm, 60.0)
    tracker.set_estimate("notes", (0 if loaded else rates["llm_load"])
                         + sum(c / rates["prompt_chars_per_s"] + P.expected_note_tokens(c) / tok_s for c in sizes))
    tracker.begin("notes", "Reading the transcript" if loaded else f"Loading {llm} into memory")
    first = {}

    def on_notes(frac, info):
        part = f" (part {info['call']} of {info['calls']})" if info["calls"] > 1 else ""
        if info["tokens"] == 0:
            tracker.update("notes", frac, f"Reading the transcript{part}")
        elif info["tokens"]:
            if info["call"] not in first:  # writing has started: measure speed from here
                first[info["call"]] = (time.perf_counter(), info["tokens"])
                tracker.rebase("notes")
            stats["tokens_last_call"] = info["tokens"]
            tracker.update("notes", frac, f"Writing with {llm} · {info['tokens']} tokens{part}")

    notes = P.extract_notes(segments, model=llm, on_progress=on_notes)
    tracker.finish("notes")
    t_end = time.perf_counter()
    if first:  # measured generation speed for the last call, for future estimates
        t_first, _ = first[max(first)]
        stats["first_token_s"] = min(t for t, _ in first.values()) - t_notes
        stats["loaded"] = loaded
        stats["prompt_chars"] = sizes[0]
        stats["gen_window"] = t_end - t_first
    stats["notes_s"] = t_end - t_notes
    stats["index_s"] = t_notes - t_idx
    return {"segments": segments, "source": source, "index": index, "notes": notes, "stats": stats,
            "timings": {"read": stats["input_s"], "index": stats["index_s"], "notes": stats["notes_s"]}}


def learn_rates(rates: dict, stats: dict, llm: str) -> None:
    """Fold this run's measured speeds into the estimates for the next run (kept in memory only)."""
    notes_tokens = stats.get("tokens_last_call")
    if stats.get("duration") and stats.get("t_audio"):
        audio_s = stats["t_input_end"] - stats["t_audio"]
        rates["whisper_rtf"][stats["device"]] = max(0.02, audio_s / stats["duration"])
    if stats.get("gen_window") and notes_tokens and stats["gen_window"] > 0.5:
        rates["tok_per_s"][llm] = max(5.0, notes_tokens / stats["gen_window"])
    if stats.get("loaded") and stats.get("first_token_s", 0) > 0.05:
        rates["prompt_chars_per_s"] = max(500.0, stats["prompt_chars"] / stats["first_token_s"])


# ------------------------------------------------------------------ JSON for the hosted page
def result_to_json(res: dict) -> dict:
    """Everything the page needs to show results. The Q&A index stays on the PC."""
    return {"segments": [[s.id, s.start, s.end, s.text] for s in res["segments"]],
            "notes": res["notes"], "source": res["source"], "timings": res["timings"],
            "notices": {"gpu_error": bool(res["stats"].get("gpu_error")),
                        "vad_off": bool(res["stats"].get("vad_off"))}}


def segments_from_json(rows: list) -> list[Segment]:
    return [Segment(int(i), start, end, text) for i, start, end, text in rows]
