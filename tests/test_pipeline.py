"""Offline tests. Ollama and Whisper are faked, so these run anywhere:
    python -m pytest tests -q      (or)      python tests/test_pipeline.py
"""
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import pipeline as P  # noqa: E402
import rag  # noqa: E402
import readers  # noqa: E402
from segments import Segment, fmt_ts, transcript_md, transcript_txt  # noqa: E402

SAMPLES = ROOT / "sample"
JSON_SAMPLE = SAMPLES / "sample_transcript.json"


# ----------------------------------------------------------------- a fake local Ollama
class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


VOCAB = ["pilot", "riverbend", "december", "model", "3b", "1b", "ocr", "scan", "tilted", "progress",
         "bar", "privacy", "summary", "monday", "friday", "test", "aarya", "sai", "niharika", "precious"]


def bag_of_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [float(sum(w.startswith(v) for w in words)) + 0.01 for v in VOCAB]


class FakeOllama:
    """Answers /api/embed with bag-of-words vectors and /api/chat with a reply that
    depends on which schema was requested."""

    def __init__(self, notes=None, answer=None):
        self.notes = notes or {"title": "t", "summary": "s",
                               "key_points": [{"point": "p", "segments": [1]}], "action_items": []}
        self.answer = answer
        self.calls = []

    def __call__(self, url, json=None, timeout=None):  # noqa: A002
        self.calls.append((url, json))
        if url.endswith("/api/embed"):
            return FakeResp({"embeddings": [bag_of_words(t) for t in json["input"]]})
        if url.endswith("/api/chat"):
            if json["format"] is rag.ANSWER_SCHEMA:
                ctx = json["messages"][1]["content"]
                ids = [int(i) for i in re.findall(r"^\[(\d+)\]", ctx, re.M)]
                reply = self.answer(ids) if self.answer else {"found": True, "answer": "a", "lines": ids[:1]}
                return FakeResp({"message": {"content": __import__("json").dumps(reply)}})
            return FakeResp({"message": {"content": __import__("json").dumps(self.notes)}})
        return FakeResp({})


# ----------------------------------------------------------------- segments and readers
def test_fmt_ts_and_none():
    assert fmt_ts(5) == "00:05" and fmt_ts(125.9) == "02:05" and fmt_ts(3725) == "1:02:05" and fmt_ts(None) == ""


def test_read_json_sample():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    assert len(segs) > 20 and segs[0].id == 1 and segs[0].start == 0.0 and all(s.text for s in segs)


def test_read_vtt_sample_keeps_times():
    segs = readers.read_file("m.vtt", (SAMPLES / "sample_meeting.vtt").read_bytes())
    assert len(segs) == len(readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes()))
    assert segs[2].start == 5.3 and "Riverbend" in segs[2].text


def test_read_srt_and_vtt_voice_tags():
    srt = "1\n00:00:01,000 --> 00:00:03,500\nHello team.\n\n2\n00:01:02,250 --> 00:01:04,000\n<i>Pilot</i> starts soon.\n"
    segs = readers.read_file("x.srt", srt.encode())
    assert [s.text for s in segs] == ["Hello team.", "Pilot starts soon."] and segs[1].start == 62.25
    vtt = "WEBVTT\n\n00:00.000 --> 00:02.000\n<v Precious>Let's begin.\n"
    assert readers.read_text(vtt)[0].text == "Precious: Let's begin."


def test_read_markdown_notes():
    segs = readers.read_file("n.md", (SAMPLES / "sample_notes.md").read_bytes())
    texts = [s.text for s in segs]
    assert "Riverbend Legal pilot" in texts and all(s.start is None for s in segs)
    assert any(t.startswith("Aarya: add automatic page straightening") for t in texts)


def test_read_plain_text_with_timestamps_and_speakers():
    text = "[00:12] Precious: Let's start.\n(01:02:03) Sai: Progress bar is next.\nNo time here, just a note\nthat wraps."
    segs = readers.read_text(text)
    assert segs[0].start == 12 and segs[0].text == "Precious: Let's start."
    assert segs[1].start == 3723 and segs[2].text == "No time here, just a note that wraps."


def test_long_paragraph_is_split_for_precise_citations():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    segs = readers.read_text(para)
    assert len(segs) > 1 and all(len(s.text) <= readers.MAX_LINE_CHARS for s in segs)


def test_read_docx_without_extra_dependencies():
    body = ("<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body>"
            "<w:p><w:r><w:t>Decision: ship the 3B model.</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Sai: progress bar by </w:t></w:r><w:r><w:t>December 10.</w:t></w:r></w:p>"
            "</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", body)
    segs = readers.read_file("notes.docx", buf.getvalue())
    assert [s.text for s in segs] == ["Decision: ship the 3B model.", "Sai: progress bar by December 10."]


def test_read_pdf():
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        return  # reportlab is only used to build a test PDF; skip if absent
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Pilot with Riverbend Legal starts December 15.")
    c.save()
    segs = readers.read_file("n.pdf", buf.getvalue())
    assert "Riverbend" in segs[0].text


def test_unsupported_and_empty_inputs():
    for name, data, msg in [("x.exe", b"", "Unsupported"), ("x.txt", b"   \n", "No text")]:
        try:
            readers.read_file(name, data)
            raise AssertionError("expected an error")
        except ValueError as e:
            assert msg in str(e)


def test_transcript_downloads():
    segs = [Segment(1, 12.0, 14.0, "Hello."), Segment(2, None, None, "A note.")]
    assert transcript_txt(segs) == "[1] [00:12] Hello.\n[2] A note.\n"
    md = transcript_md(segs, "Sync")
    assert md.startswith("# Sync") and "1. `00:12` Hello." in md and "2. A note." in md and "Nothing was uploaded" in md


# ----------------------------------------------------------------- bullet points
def test_validate_drops_bad_ids_and_flags_unverified():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    raw = {"title": "Sync", "summary": "s",
           "key_points": [{"point": "Pilot starts Dec 15", "segments": [1, 999]},
                          {"point": "Made-up claim", "segments": [999]},
                          {"point": "  ", "segments": [2]}],
           "action_items": [{"owner": "", "task": "Add page straightening", "due": "", "segments": [5]}]}
    notes = P.validate_notes(raw, segs)
    assert len(notes["key_points"]) == 2
    assert notes["key_points"][0]["segments"] == [1] and notes["key_points"][0]["verified"]
    assert notes["key_points"][1]["verified"] is False and notes["key_points"][1]["start"] is None
    ai = notes["action_items"][0]
    assert ai["owner"] == "Unassigned" and ai["due"] == "Not stated" and ai["start"] == segs[4].start


def test_extract_uses_local_ollama_qwen3_and_schema():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    fake = FakeOllama(notes={"title": "t", "summary": "s",
                             "key_points": [{"point": "p", "segments": [3]}], "action_items": []})
    with mock.patch.object(P.requests, "post", side_effect=fake):
        notes = P.extract_notes(segs)
    url, payload = fake.calls[0]
    assert url.startswith("http://127.0.0.1:11434") and payload["model"] == "qwen3:4b-instruct"
    assert payload["format"] == P.NOTES_SCHEMA and payload["options"]["temperature"] == 0
    assert "think" not in payload  # instruct model: no thinking flag needed
    assert "[1] (00:00)" in payload["messages"][1]["content"]
    assert notes["key_points"][0]["start"] == segs[2].start


def test_hybrid_qwen3_gets_thinking_turned_off():
    fake = FakeOllama()
    with mock.patch.object(P.requests, "post", side_effect=fake):
        P.extract_notes([Segment(1, None, None, "Hello.")], model="qwen3:1.7b")
    assert fake.calls[0][1]["think"] is False
    assert P._thinking_off("phi4-mini") == {}


def test_notes_without_times_render_line_numbers():
    segs = readers.read_file("n.md", (SAMPLES / "sample_notes.md").read_bytes())
    fake = FakeOllama(notes={"title": "T", "summary": "S",
                             "key_points": [{"point": "p", "segments": [4]}],
                             "action_items": [{"owner": "Sai", "task": "x", "due": "Dec 10", "segments": [12]}]})
    with mock.patch.object(P.requests, "post", side_effect=fake):
        notes = P.extract_notes(segs)
    prompt = fake.calls[0][1]["messages"][1]["content"]
    assert "[4] " in prompt and "(" not in prompt.split("\n")[1][:6]
    md = P.to_markdown(notes, segs)
    assert "- p (line 4)" in md and "| Sai | x | Dec 10 | (line 12) |" in md


def test_long_transcript_is_chunked_then_merged():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    fake = FakeOllama()
    with mock.patch.object(P.requests, "post", side_effect=fake):
        P.extract_notes(segs, max_chars=500)
    n_chunks = len(P._chunks(P.lines_for_prompt(segs), 500))
    assert n_chunks > 1 and len(fake.calls) == n_chunks + 1
    assert fake.calls[-1][1]["messages"][0]["content"] == P.MERGE_PROMPT


def test_check_ollama_messages():
    with mock.patch.object(P.requests, "get", side_effect=P.requests.ConnectionError()):
        ok, msg = P.check_ollama()
    assert not ok and "not running" in msg
    tags = FakeResp({"models": [{"name": "qwen3:4b-instruct"}, {"name": "all-minilm:latest"}]})
    with mock.patch.object(P.requests, "get", return_value=tags):
        assert P.check_ollama("qwen3:4b-instruct")[0] and P.check_ollama("all-minilm")[0]
        ok, msg = P.check_ollama("qwen3:1.7b")
    assert not ok and "ollama pull qwen3:1.7b" in msg


# ----------------------------------------------------------------- transcription
def test_transcribe_with_fake_whisper_frees_model_and_skips_empty():
    seg = lambda a, b, t: types.SimpleNamespace(start=a, end=b, text=t)  # noqa: E731
    freed = []

    class FakeModel:
        def __init__(self, *a, **kw):
            assert kw["local_files_only"] is True and kw["device"] == "cpu"

        def transcribe(self, audio, **kw):
            assert kw["vad_filter"] is True and isinstance(audio, io.BytesIO)
            return iter([seg(0, 2, " Hello team. "), seg(2, 3, "  "), seg(3, 6, "Pilot starts soon.")]), \
                types.SimpleNamespace(language="en", duration=6.0)

        def __del__(self):
            freed.append(True)

    progress = []
    with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeModel)}):
        segs, info = P.transcribe(io.BytesIO(b"video bytes"), on_progress=progress.append)
    assert [s.text for s in segs] == ["Hello team.", "Pilot starts soon."] and [s.id for s in segs] == [1, 2]
    assert info["language"] == "en" and progress[-1] == 1.0 and freed


def test_gpu_first_then_cpu_fallback():
    """With an NVIDIA GPU visible but CUDA libraries missing, transcription must still
    succeed on the CPU and report why."""
    seg = types.SimpleNamespace(start=0, end=2, text="Hello team.")
    tried = []

    class FakeModel:
        def __init__(self, size, device, compute_type, local_files_only):
            tried.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")

        def transcribe(self, audio, **kw):
            return iter([seg]), types.SimpleNamespace(language="en", duration=2.0)

    fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeModel),
                                       "ctranslate2": fake_ct2}):
        segs, meta = P.transcribe(io.BytesIO(b"x"))
    assert tried == [("cuda", "float16"), ("cpu", "int8")]
    assert meta["device"] == "cpu" and "cublas" in meta["gpu_error"] and segs[0].text == "Hello team."


def test_gpu_used_when_it_works():
    seg = types.SimpleNamespace(start=0, end=2, text="Hi.")

    class FakeModel:
        def __init__(self, size, device, compute_type, local_files_only):
            self.device = device

        def transcribe(self, audio, **kw):
            return iter([seg]), types.SimpleNamespace(language="en", duration=2.0)

    fake_ct2 = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeModel),
                                       "ctranslate2": fake_ct2}):
        _, meta = P.transcribe(io.BytesIO(b"x"))
    assert meta["device"] == "cuda" and "gpu_error" not in meta


# ----------------------------------------------------------------- Q&A (RAG)
def test_passages_overlap_and_cover_every_line():
    segs = [Segment(i, None, None, f"line {i}") for i in range(1, 10)]
    ps = rag.make_passages(segs)
    covered = sorted({i for p in ps for i in p.ids})
    assert covered == list(range(1, 10)) and ps[0].ids == [1, 2, 3, 4] and ps[1].ids == [3, 4, 5, 6]


def test_index_is_in_memory_and_normalized():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    with mock.patch.object(rag.requests, "post", side_effect=FakeOllama()):
        idx = rag.build_index(segs)
    assert isinstance(idx.vectors, np.ndarray) and idx.vectors.shape[0] == len(idx.passages)
    assert np.allclose(np.linalg.norm(idx.vectors, axis=1), 1.0, atol=1e-5)


def test_retrieval_finds_the_relevant_passage():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    fake = FakeOllama()
    with mock.patch.object(rag.requests, "post", side_effect=fake):
        idx = rag.build_index(segs)
        hits = rag.retrieve("Who will build the progress bar?", idx, k=2)
    top_text = hits[0][0].text.lower()
    assert "progress bar" in top_text


def test_answer_keeps_only_retrieved_citations():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    fake = FakeOllama(answer=lambda ids: {"found": True, "answer": "Sai builds it by December 10.",
                                          "lines": [ids[0], 999]})
    with mock.patch.object(rag.requests, "post", side_effect=fake), \
            mock.patch.object(P.requests, "post", side_effect=fake):
        idx = rag.build_index(segs)
        res = rag.answer("Who will build the progress bar?", idx, segs)
    assert res["found"] and res["segments"] and 999 not in res["segments"]
    assert set(res["segments"]) <= set(res["sources"]) and res["start"] is not None
    chat = [c for c in fake.calls if c[0].endswith("/api/chat")][0][1]
    assert chat["format"] is rag.ANSWER_SCHEMA and "Question: Who will build" in chat["messages"][1]["content"]


def test_answer_not_found_or_uncited_becomes_not_in_transcript():
    segs = readers.read_file(JSON_SAMPLE.name, JSON_SAMPLE.read_bytes())
    for reply in [{"found": False, "answer": "Not in the transcript.", "lines": []},
                  {"found": True, "answer": "The budget is $5M.", "lines": [999]}]:
        fake = FakeOllama(answer=lambda ids, r=reply: r)
        with mock.patch.object(rag.requests, "post", side_effect=fake), \
                mock.patch.object(P.requests, "post", side_effect=fake):
            idx = rag.build_index(segs)
            res = rag.answer("What is the budget?", idx, segs)
        assert res == {**res, "answer": rag.NOT_FOUND, "found": False, "segments": [], "start": None}


# ----------------------------------------------------------------- privacy
def test_privacy_receipt_shape():
    r = P.external_connections()
    assert set(r) == {"checked", "external", "notes"} and r["checked"]


def test_nothing_written_to_disk_by_readers_and_rag(tmp_path=None):
    import os
    before = set(os.listdir(ROOT))
    segs = readers.read_file("n.md", (SAMPLES / "sample_notes.md").read_bytes())
    with mock.patch.object(rag.requests, "post", side_effect=FakeOllama()):
        rag.build_index(segs)
    assert set(os.listdir(ROOT)) - before <= {"__pycache__"}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
