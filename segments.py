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
    # Who said it, for video (speakers.py): the display name, or "Speaker 2" when unknown.
    # speaker_id keeps the voice group's label, so names can be corrected later.
    # Transcript files keep speakers inside `text` ("Sai: ...") as before.
    speaker: Optional[str] = None
    speaker_id: Optional[str] = None


def said(s: Segment) -> str:
    """The line as shown everywhere: "Name: text" when the speaker is known."""
    return f"{s.speaker}: {s.text}" if s.speaker else s.text


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
        out.append(f"[{s.id}] {ts}{said(s)}")
    return "\n".join(out) + "\n"


def transcript_md(segments: list[Segment], title: str = "Transcript") -> str:
    """Markdown download, readable in any editor."""
    out = [f"# {title}", ""]
    for s in segments:
        ts = f"`{fmt_ts(s.start)}` " if s.start is not None else ""
        out.append(f"{s.id}. {ts}{said(s)}")
    out += ["", "_Created on this device by SNAP Assist. Nothing was uploaded._"]
    return "\n".join(out) + "\n"
