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
import speakers as S
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
    "speakers_rtf": 0.16,                         # telling voices apart, per second of audio
    "speakers_ocr_s": 18.0,                       # reading names on screen
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
    ]
    if video and S.available()[0]:
        stages.append(Stage("speakers", "Identifying speakers", 25.0))
    stages += [
        Stage("index", "Building the Q&A index", 1.0),
        Stage("notes", "Writing bullet points", 15.0),
    ]
    return Tracker(stages)


def speakers_estimate(duration_s: float, rates: dict) -> float:
    """Measured on the RTX 3060 laptop: telling voices apart ~0.15 s per second of audio
    (CPU, 4 threads), reading names ~18 s for a 4-person meeting."""
    return 3.0 + duration_s * rates["speakers_rtf"] + rates["speakers_ocr_s"]


def process(job: dict, tracker: Tracker, rates: dict, llm: str, whisper_size: str) -> dict:
    """Run one meeting. job: {"kind": video|file|sample|text, "name", "data" (bytes), "text"}.
    Returns segments, source, index, notes, timings and measured speeds ("stats")."""
    stats: dict = {}
    speakers = None  # who's who, for video (see identify_speakers)
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
        tracker.finish("input")
        stats["t_input_end"] = time.perf_counter()
        stats["input_s"] = stats["t_input_end"] - t_in
        speakers = identify_speakers(job["data"], segments, tracker, rates, stats)
    elif job["kind"] == "file":
        tracker.update("input", 0.3, f"Reading {job['name']}")
        segments, source = readers.read_file(job["name"], job["data"]), job["name"]
    elif job["kind"] == "sample":
        tracker.update("input", 0.3, "Reading the sample transcript")
        segments, source = readers.read_file(SAMPLE.name, SAMPLE.read_bytes()), "sample transcript"
    else:
        tracker.update("input", 0.3, "Reading the pasted text")
        segments, source = readers.read_text(job["text"]), "pasted text"
    if "t_input_end" not in stats:  # text inputs: reading is the whole first step
        tracker.finish("input")
        stats["t_input_end"] = time.perf_counter()
        stats["input_s"] = stats["t_input_end"] - t_in
    t_idx = time.perf_counter()

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
    timings = {"read": stats["input_s"], "index": stats["index_s"], "notes": stats["notes_s"]}
    if "speakers_s" in stats:
        timings["speakers"] = stats["speakers_s"]
    return {"segments": segments, "source": source, "index": index, "notes": notes, "stats": stats,
            "speakers": speakers, "timings": timings}


def identify_speakers(video: bytes, segments, tracker: Tracker, rates: dict, stats: dict):
    """Label each transcript line with who said it (speakers.py). Returns the speaker list for
    the app's Speakers panel, or None when the speaker models aren't installed or it fails;
    the meeting is still processed, just without names."""
    ok, why = S.available()
    if not ok:
        stats["speakers_off"] = why
        return None
    t0 = time.perf_counter()
    tracker.set_estimate("speakers", speakers_estimate(stats.get("duration") or 60.0, rates))
    tracker.begin("speakers", "Listening for different voices")
    try:
        info = S.identify(video, segments, on_progress=lambda f, d: tracker.update("speakers", f, d))
    except Exception as e:  # noqa: BLE001  (names are a bonus; never lose the meeting over them)
        stats["speakers_off"] = f"Speaker identification failed: {e}"[:300]
        for seg in segments:
            seg.speaker = seg.speaker_id = None
        tracker.finish("speakers")
        return None
    tracker.finish("speakers")
    stats["speakers_s"] = time.perf_counter() - t0
    return info["speakers"]


def rewrite(segments, names: dict[str, str], tracker: Tracker, rates: dict, llm: str) -> dict:
    """After the user corrects speaker names: relabel the lines, then rebuild the Q&A index and
    the bullet points so owners use the corrected names. Tracker stages: index, notes."""
    S.apply_names(segments, names)
    t0 = time.perf_counter()
    tracker.begin("index", "Updating the Q&A index with the names")
    index = rag.build_index(segments, on_progress=lambda p: tracker.update("index", p))
    tracker.finish("index")
    t1 = time.perf_counter()
    tok_s = rates["tok_per_s"].get(llm, 60.0)
    tracker.set_estimate("notes", sum(c / rates["prompt_chars_per_s"] + P.expected_note_tokens(c) / tok_s
                                      for c in P.note_calls(segments)))
    tracker.begin("notes", "Rewriting bullet points with the names")
    started = {}

    def on_notes(frac, info):
        if info["tokens"] and not started:
            started["t"] = True
            tracker.rebase("notes")
        tracker.update("notes", frac, f"Writing with {llm} · {info['tokens']} tokens" if info["tokens"]
                       else "Reading the transcript")

    notes = P.extract_notes(segments, model=llm, on_progress=on_notes)
    tracker.finish("notes")
    return {"segments": segments, "index": index, "notes": notes,
            "timings": {"index": t1 - t0, "notes": time.perf_counter() - t1}}


def learn_rates(rates: dict, stats: dict, llm: str) -> None:
    """Fold this run's measured speeds into the estimates for the next run (kept in memory only)."""
    notes_tokens = stats.get("tokens_last_call")
    if stats.get("duration") and stats.get("t_audio"):
        audio_s = stats["t_input_end"] - stats["t_audio"]
        rates["whisper_rtf"][stats["device"]] = max(0.02, audio_s / stats["duration"])
    if stats.get("speakers_s") and stats.get("duration"):
        # attribute the measured time to the voice part, keeping the name-reading estimate
        voice_s = max(1.0, stats["speakers_s"] - 3.0 - rates["speakers_ocr_s"])
        rates["speakers_rtf"] = max(0.02, voice_s / stats["duration"])
    if stats.get("gen_window") and notes_tokens and stats["gen_window"] > 0.5:
        rates["tok_per_s"][llm] = max(5.0, notes_tokens / stats["gen_window"])
    if stats.get("loaded") and stats.get("first_token_s", 0) > 0.05:
        rates["prompt_chars_per_s"] = max(500.0, stats["prompt_chars"] / stats["first_token_s"])


# ------------------------------------------------------------------ JSON for the hosted page
def result_to_json(res: dict) -> dict:
    """Everything the page needs to show results. The Q&A index stays on the PC."""
    return {"segments": [[s.id, s.start, s.end, s.text, s.speaker, s.speaker_id] for s in res["segments"]],
            "notes": res["notes"], "source": res["source"], "timings": res["timings"],
            "speakers": res.get("speakers"), "notices": notices(res["stats"])}


def notices(stats: dict) -> dict:
    return {"gpu_error": bool(stats.get("gpu_error")), "vad_off": bool(stats.get("vad_off")),
            "speakers_off": stats.get("speakers_off") or ""}


def segments_from_json(rows: list) -> list[Segment]:
    """Rows of [id, start, end, text] or [id, start, end, text, speaker, speaker_id]."""
    return [Segment(int(r[0]), r[1], r[2], r[3], *(r[4:6])) for r in rows]
