"""Who said what: tell speakers apart by voice, then read their names from the video.

Why: a transcript without speakers can't tell who "I'll send it Monday" belongs to, so action
items end up "Unassigned" or on the wrong person. With every line labelled by speaker, the
language model can assign owners correctly.

1. Diarization (sherpa-onnx, on this PC): the pyannote segmentation model finds where speech
   changes hands, and a 3D-Speaker voice model groups the pieces by voice into
   "Speaker 1", "Speaker 2", ...
2. Names from the screen (RapidOCR, on this PC). Teams, Zoom and Meet recordings show each
   person's name on their video tile. Frames are read while each speaker is talking. Names on
   tiles that are always visible appear for every speaker, so they say nothing; the name that
   appears in the same screen position *only while one speaker talks* is that speaker's name.
   So every (name, screen position) pair is scored by how specific it is to one speaker.
3. Two voice groups that read the same name are the same person, and are merged.
Anything not resolved keeps its "Speaker N" label, which the user can rename in the app.

The models are downloaded once by the setup script into models/; nothing is downloaded at run
time, and video frames are decoded in memory (never written to disk)."""
from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

MODELS = Path(__file__).parent / "models"
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
SAMPLE_RATE = 16000
THRESHOLD = 0.6        # voice clustering: lower splits one person into two (fixed by the names
                       # step), higher can merge two people into one (which nothing can fix)
THREADS = 4            # measured: 4 threads ~3x faster than 1; 8 is no faster
FRAMES_PER_SPEAKER = 5
MIN_SPECIFICITY = 0.5  # how much more often a name shows while this speaker talks than others

# Meeting-app words that are never a person's name
UI_WORDS = {
    "recording", "recorded", "live", "mute", "muted", "unmute", "share", "sharing", "screen", "chat",
    "leave", "end", "people", "participants", "raise", "hand", "more", "camera", "mic", "microphone",
    "reactions", "react", "view", "apps", "you", "me", "host", "co-host", "presenting", "presenter",
    "meeting", "join", "joined", "waiting", "room", "lobby", "rooms", "breakout", "captions", "transcript",
    "speaker", "gallery", "together", "mode", "notes", "whiteboard", "start", "stop", "video", "audio",
    "settings", "security", "polls", "invite", "details", "guest", "external", "organizer", "teams",
    "zoom", "google", "meet", "microsoft", "connecting", "reconnecting", "unstable", "connection",
}


@dataclass
class Turn:
    start: float
    end: float
    label: str


def available() -> tuple[bool, str]:
    """True when the speaker models and libraries are installed (the setup script does this)."""
    missing = [p.name for p in (SEG_MODEL, EMB_MODEL) if not p.exists()]
    if missing:
        return False, "Speaker models not downloaded (run the setup script once while online)."
    try:  # import (not just look up) so a library that fails to load counts as missing
        import importlib
        for lib in ("sherpa_onnx", "rapidocr_onnxruntime"):
            importlib.import_module(lib)
    except Exception:
        return False, "Speaker libraries not installed (run the setup script once while online)."
    return True, "Speaker identification ready."


# ------------------------------------------------------------------ 1. voices
def decode_audio(data: bytes):
    from faster_whisper import decode_audio as fw_decode
    return fw_decode(io.BytesIO(data), sampling_rate=SAMPLE_RATE)


def diarize(samples, on_progress: Optional[Callable[[float], None]] = None) -> list[Turn]:
    """Speaker turns, labelled "Speaker 1", "Speaker 2", ... in order of first appearance."""
    import sherpa_onnx as so
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=THREADS),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=THREADS),
        clustering=so.FastClusteringConfig(num_clusters=-1, threshold=THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = so.OfflineSpeakerDiarization(cfg)

    def cb(done, total):
        if on_progress and total:
            on_progress(done / total)
        return 0  # 0 = keep going

    result = sd.process(samples, callback=cb).sort_by_start_time()
    order: dict[int, str] = {}
    turns = []
    for s in result:
        if s.speaker not in order:
            order[s.speaker] = f"Speaker {len(order) + 1}"
        turns.append(Turn(float(s.start), float(s.end), order[s.speaker]))
    return turns


def label_segments(segments, turns: list[Turn]) -> None:
    """Give each transcript line the speaker who talks most during it."""
    for seg in segments:
        if seg.start is None:
            continue
        end = seg.end if seg.end is not None else seg.start + 2.0
        overlap: dict[str, float] = defaultdict(float)
        for t in turns:
            ov = min(end, t.end) - max(seg.start, t.start)
            if ov > 0:
                overlap[t.label] += ov
        if overlap:
            seg.speaker = max(overlap, key=overlap.get)
        else:  # in a pause: take the nearest turn
            near = min(turns, key=lambda t: min(abs(t.start - seg.start), abs(t.end - seg.start)), default=None)
            seg.speaker = near.label if near else None


# ------------------------------------------------------------------ 2. names on screen
def clean_name(text: str) -> Optional[str]:
    """OCR text -> a plausible display name, or None. Fixes OCR's usual slips on name labels."""
    t = re.sub(r"\((guest|external|unverified|host|co-host|organizer|he/him|she/her|they/them)\)", "",
               text, flags=re.I)
    t = re.split(r"[|•·]", t)[0]
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)          # "PreciousMensah" -> "Precious Mensah"
    t = re.sub(r"[^A-Za-zÀ-ɏ' .-]", " ", t)
    words = [w.strip(".-'") for w in t.split()]
    words = [w for w in words if w]
    if not 1 <= len(words) <= 4:
        return None
    # A capital I read as a lowercase l ("Niharika lyer" -> "Niharika Iyer"): in a name whose
    # other words are capitalized, a word doesn't start with a lowercase l.
    def capital_i(i, w):
        others_capitalized = any(o[0].isupper() for j, o in enumerate(words) if j != i)
        return "I" + w[1:] if others_capitalized and w[0] == "l" and len(w) > 1 and w[1:].islower() else w
    words = [capital_i(i, w) for i, w in enumerate(words)]
    if any(w.lower() in UI_WORDS for w in words) or not words[0][0].isupper():
        return None
    name = " ".join(words)
    return name if 2 <= len(name) <= 40 else None


def _frame_times(turns: list[Turn], per_speaker: int) -> list[tuple[float, str]]:
    """A few moments inside each speaker's longest turns, away from the turn edges (where
    meeting apps are still switching the big tile)."""
    by_label: dict[str, list[Turn]] = defaultdict(list)
    for t in turns:
        by_label[t.label].append(t)
    picks: dict[float, str] = {}
    for label, ts in by_label.items():
        chosen = []
        for t in sorted(ts, key=lambda t: t.end - t.start, reverse=True):
            if t.end - t.start < 1.2:
                break  # the rest are shorter still
            lo, hi = t.start + 0.8, max(t.start + 0.8, t.end - 0.4)
            for f in (0.5, 0.85, 0.15):  # middle first, then late, then early in the turn
                chosen.append(round(lo + f * (hi - lo), 2))
                if len(chosen) >= per_speaker:
                    break
            if len(chosen) >= per_speaker:
                break
        for x in chosen:
            picks.setdefault(x, label)
    return sorted(picks.items())


def _frames(video: bytes, times: list[float]):
    """Decode the frames nearest to `times` (sorted), in memory. Yields (time, BGR array)."""
    import av
    with av.open(io.BytesIO(video)) as c:
        if not c.streams.video:
            return
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for want in times:
            c.seek(int(max(0.0, want - 1.0) / vs.time_base), stream=vs, backward=True)
            for fr in c.decode(vs):
                if fr.time is not None and fr.time >= want:
                    yield want, fr.to_ndarray(format="bgr24")
                    break


_OCR = None


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def read_names(img) -> list[tuple[str, tuple[int, int]]]:
    """Names on one frame with a coarse screen position (6x6 grid cell)."""
    import cv2
    h, w = img.shape[:2]
    scale = 960 / w if w > 960 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img
    found, _ = _ocr()(small)
    out = []
    for box, text, score in found or []:
        if float(score) < 0.6 or not clean_name(text) and not re.search(r"[a-z][A-Z]", text):
            continue
        xs, ys = [p[0] / scale for p in box], [p[1] / scale for p in box]
        x0, x1, y0, y1 = int(min(xs)) - 6, int(max(xs)) + 6, int(min(ys)) - 6, int(max(ys)) + 6
        crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if crop.size == 0:
            continue
        # Read the label again at 3x: OCR then keeps the space between first and last name.
        big = cv2.copyMakeBorder(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                                 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        again, _ = _ocr()(big, use_det=False, use_cls=False, use_rec=True)
        name = clean_name(again[0][0]) if again else None
        name = name or clean_name(text)
        if name:
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            out.append((name, (min(int(cx * 6), 5), min(int(cy * 6), 5))))
    return out


def name_key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _fold_spellings(counts: Counter, similarity: float = 0.85) -> dict[str, str]:
    """key -> canonical key: each key joins the most frequent similar key (difflib ratio)."""
    from difflib import SequenceMatcher
    canon: dict[str, str] = {}
    heads: list[str] = []
    for key, _ in counts.most_common():
        head = next((h for h in heads if SequenceMatcher(None, key, h).ratio() >= similarity), None)
        if head is None:
            heads.append(key)
            head = key
        canon[key] = head
    return canon


def names_from_video(video: bytes, turns: list[Turn],
                     on_progress: Optional[Callable[[float], None]] = None) -> dict[str, dict]:
    """{"Speaker 1": {"name": "Precious Mensah", "score": 0.9}, ...} for speakers whose name could
    be read with confidence. Speakers without a confident name are left out."""
    picks = _frame_times(turns, FRAMES_PER_SPEAKER)
    if not picks:
        return {}
    label_at = dict(picks)
    observed = []                                       # (frame number, label, name, cell)
    frames_of: Counter = Counter()
    for i, (t, img) in enumerate(_frames(video, [p[0] for p in picks])):
        label = label_at[t]
        frames_of[label] += 1
        observed += [(i, label, name, cell) for name, cell in read_names(img)]
        if on_progress:
            on_progress((i + 1) / len(picks))
    return _match_names(observed, frames_of)


def _match_names(observed: list[tuple], frames_of: Counter) -> dict[str, dict]:
    """observed: (frame number, voice label, name read, screen cell). See names_from_video."""
    # OCR misreads a letter now and then ("Kulkarni" / "Kuikarni"). A misspelling seen only in
    # a few frames would look like a separate name that appears "only while X talks", so
    # near-identical spellings are folded into the most common one before counting.
    canon = _fold_spellings(Counter(name_key(n) for _, _, n, _ in observed))
    per_frame = {(i, label, canon[name_key(name)], cell) for i, label, name, cell in observed}
    seen: dict[str, Counter] = defaultdict(Counter)     # label -> (key, cell) -> frames
    for _, label, key, cell in per_frame:
        seen[label][(key, cell)] += 1
    # Spellings per (key, position): the name is spelled as read where it was matched, which
    # is the big tile, whose larger text OCR reads better than the small tiles'.
    spelled: dict[tuple, Counter] = defaultdict(Counter)
    for _, _, name, cell in observed:
        spelled[(canon[name_key(name)], cell)][name] += 1

    # For every (name, screen position): which voice groups show it in most of their frames
    # ("holders"), and how clearly that separates them from the other groups. One person's
    # voice can be split into two groups, so a name may legitimately have several holders.
    labels = list(frames_of)
    rate = {(lab, k): seen[lab][k] / frames_of[lab] for lab in labels for k in seen[lab]}
    candidates = []
    for k in {k for lab in labels for k in seen[lab]}:
        holders = [lab for lab in labels if rate.get((lab, k), 0) >= 0.5]
        others = [rate.get((lab, k), 0) for lab in labels if lab not in holders] or [0.0]
        if holders:
            spec = min(rate[(lab, k)] for lab in holders) - max(others)
            candidates.append((len(holders), -spec, k, holders))

    # The big "who is talking" tile is the position held by the fewest groups. Small tiles
    # showing everyone *else* are held by almost every group; a name shown for every group
    # (someone who never speaks) says nothing about who talks.
    max_holders = max(1, len(labels) - 2)
    named: dict[str, dict] = {}
    used: set[str] = set()
    for n_holders, neg_spec, (key, cell), holders in sorted(candidates):
        if (-neg_spec < MIN_SPECIFICITY or n_holders > max_holders or key in used
                or any(lab in named for lab in holders)):
            continue
        used.add(key)
        for lab in holders:
            named[lab] = {"name": spelled[(key, cell)].most_common(1)[0][0], "score": round(-neg_spec, 2)}
    return named


# ------------------------------------------------------------------ 3. putting it together
def identify(video: bytes, segments, on_progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Label every line of `segments` with a speaker, named from the video where possible.
    Returns {"speakers": [{label, name, source, lines}], "turns": n}."""
    say = on_progress or (lambda f, d: None)
    say(0.0, "Listening for different voices")
    samples = decode_audio(video)
    turns = diarize(samples, on_progress=lambda f: say(0.65 * f, "Telling voices apart"))
    label_segments(segments, turns)
    say(0.65, "Reading names on screen")
    names = names_from_video(video, turns, on_progress=lambda f: say(0.65 + 0.35 * f, "Reading names on screen"))

    # Two voice groups showing the same on-screen name are one person.
    canonical: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for label in sorted(names, key=lambda x: int(x.split()[-1])):
        key = name_key(names[label]["name"])
        canonical[label] = by_key.setdefault(key, label)
    for seg in segments:
        if seg.speaker in canonical:
            seg.speaker = canonical[seg.speaker]
    for t in turns:
        t.label = canonical.get(t.label, t.label)

    # Number the remaining voices 1, 2, 3... again (a merge would otherwise leave a gap,
    # "Speaker 1, 3, 4", which looks like a mistake).
    counts = Counter(s.speaker for s in segments if s.speaker)
    old = sorted(counts, key=lambda x: int(x.split()[-1]))
    renumber = {lab: f"Speaker {i + 1}" for i, lab in enumerate(old)}
    for seg in segments:
        if seg.speaker:
            seg.speaker = renumber[seg.speaker]
    speakers = []
    for label in old:
        info = names.get(label)
        speakers.append({"label": renumber[label], "name": info["name"] if info else "",
                         "source": "screen" if info else "", "lines": counts[label]})
    apply_names(segments, {s["label"]: s["name"] for s in speakers})
    say(1.0, f"{len(speakers)} speakers, {sum(1 for s in speakers if s['name'])} named from the video")
    return {"speakers": speakers, "turns": len(turns)}


def apply_names(segments, mapping: dict[str, str]) -> None:
    """mapping: label -> name ("" keeps the label). Each line keeps its label in `speaker_id`
    so names can be changed again later."""
    for seg in segments:
        label = getattr(seg, "speaker_id", None) or seg.speaker
        if not label:
            continue
        seg.speaker_id = label
        seg.speaker = (mapping.get(label) or "").strip() or label
