"""Read meeting transcripts and notes into numbered segments, entirely locally.

Supported: .txt, .md, .vtt, .srt, .docx, .pdf, and SNAP's own .json transcripts.
Every reader returns a list of pipeline.Segment, numbered from 1, with start/end
times in seconds when the file has them and None when it doesn't.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from segments import Segment

TEXT_TYPES = {".txt", ".md", ".vtt", ".srt", ".docx", ".pdf", ".json"}
MAX_LINE_CHARS = 400  # long paragraphs are split at sentence ends so citations stay precise

# "00:01:02.500 --> 00:01:05.000" (VTT) or "00:01:02,500 --> 00:01:05,000" (SRT)
_CUE = re.compile(r"(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}:)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
# "[00:12]", "(1:02:03)", "00:12 -" at the start of a line in plain-text transcripts
_LEAD_TS = re.compile(r"^\s*[\[(]?((?:\d{1,2}:)?\d{1,2}:\d{2})(?:[.,]\d+)?[\])]?\s*[-–:]?\s*")
_VTT_VOICE = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>")
_TAGS = re.compile(r"<[^>]+>")


def _hms(h: Optional[str], m: str, s: str, frac: str = "0") -> float:
    hours = int(h.rstrip(":")) if h else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")[:3]) / 1000


def _ts_to_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _split_long(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= MAX_LINE_CHARS:
        return [text] if text else []
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(sent) + 1 > MAX_LINE_CHARS:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def _number(rows: list[tuple[Optional[float], Optional[float], str]]) -> list[Segment]:
    segs: list[Segment] = []
    for start, end, text in rows:
        for piece in _split_long(text):
            segs.append(Segment(len(segs) + 1, start, end, piece))
    return segs


# --------------------------------------------------------------------------- readers
def parse_cues(text: str) -> list[Segment]:
    """WebVTT or SRT subtitles. Keeps the speaker from <v Name> tags."""
    rows, lines = [], text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _CUE.search(lines[i])
        if not m:
            i += 1
            continue
        start = _hms(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _hms(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        cue = " ".join(body)
        voice = _VTT_VOICE.search(cue)
        cue = _TAGS.sub("", cue).strip()
        if voice and cue:
            cue = f"{voice.group(1).strip()}: {cue}"
        if cue:
            rows.append((start, end, cue))
    return _number(rows)


def parse_plain(text: str) -> list[Segment]:
    """Plain text or Markdown notes. A leading timestamp like [00:12] is kept as the
    line's time; Markdown bullets and heading marks are removed."""
    rows, para = [], []

    def flush():
        if para:
            rows.append((None, None, " ".join(para)))
            para.clear()

    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        line = re.sub(r"^(#{1,6}\s+|[-*+]\s+(\[[ xX]\]\s+)?|\d+[.)]\s+|>\s*)", "", line)
        m = _LEAD_TS.match(line)
        if m and len(m.group(0)) < len(line):
            flush()
            t = _ts_to_seconds(m.group(1))
            rows.append((t, None, line[m.end():]))
        elif re.match(r"^[A-Z][\w .'-]{0,40}:\s", line) or raw.lstrip()[:1] in "-*+#>" or re.match(r"^\d+[.)]\s", raw.lstrip()):
            flush()  # a speaker turn, bullet or heading starts its own line
            rows.append((None, None, line))
        else:
            para.append(line)
    flush()
    return _number(rows)


def parse_docx(data: bytes) -> list[Segment]:
    """Word documents, read straight from the .docx XML (no extra dependency)."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{{{ns['w']}}}p"):
        txt = "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t")).strip()
        if txt:
            paras.append(txt)
    return parse_plain("\n\n".join(paras))


def parse_pdf(data: bytes) -> list[Segment]:
    """Text-based PDFs (scanned PDFs need OCR, planned for a later phase)."""
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Reading PDFs needs pdfplumber. Run the setup script.") from e
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        text = "\n\n".join((page.extract_text() or "") for page in pdf.pages)
    if not text.strip():
        raise ValueError("This PDF has no readable text (it may be a scan).")
    return parse_plain(text)


def parse_json(text: str) -> list[Segment]:
    data = json.loads(text)
    rows = data["segments"] if isinstance(data, dict) else data
    return _number([(r.get("start"), r.get("end"), r["text"]) for r in rows])


def read_file(name: str, data: bytes) -> list[Segment]:
    """Pick a reader by file extension."""
    ext = Path(name).suffix.lower()
    if ext not in TEXT_TYPES:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(TEXT_TYPES))}")
    if ext == ".docx":
        segs = parse_docx(data)
    elif ext == ".pdf":
        segs = parse_pdf(data)
    else:
        text = data.decode("utf-8-sig", errors="replace")
        if ext == ".json":
            segs = parse_json(text)
        elif ext in {".vtt", ".srt"}:
            segs = parse_cues(text)
        else:
            segs = parse_plain(text)
    if not segs:
        raise ValueError("No text was found in this file.")
    return segs


def read_text(text: str) -> list[Segment]:
    """Pasted text: subtitles if it has cue times, otherwise plain notes."""
    segs = parse_cues(text) if _CUE.search(text) else parse_plain(text)
    if not segs:
        raise ValueError("Paste some meeting text first.")
    return segs
