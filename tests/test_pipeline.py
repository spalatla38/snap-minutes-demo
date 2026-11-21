"""Offline tests for pipeline.py. Ollama and Whisper are mocked, so these run anywhere:
    python -m pytest tests -q      (or)      python tests/test_pipeline.py
"""
import json
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as P  # noqa: E402

SAMPLE = ROOT / "sample" / "sample_transcript.json"


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def fake_model_reply(raw: dict):
    return FakeResp({"message": {"content": json.dumps(raw)}})


def test_load_sample_transcript():
    segs = P.load_transcript(str(SAMPLE))
    assert len(segs) > 20 and segs[0].id == 1 and segs[0].start == 0.0
    assert all(s.text for s in segs)


def test_fmt_ts():
    assert P.fmt_ts(5) == "00:05" and P.fmt_ts(125.9) == "02:05" and P.fmt_ts(3725) == "1:02:05"


def test_validate_drops_bad_ids_and_flags_unverified():
    segs = P.load_transcript(str(SAMPLE))
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


def test_extract_single_call_uses_local_ollama_with_schema():
    segs = P.load_transcript(str(SAMPLE))
    raw = {"title": "t", "summary": "s", "key_points": [{"point": "p", "segments": [3]}], "action_items": []}
    with mock.patch.object(P.requests, "post", return_value=fake_model_reply(raw)) as post:
        notes = P.extract_notes(segs, model="llama3.2:3b")
    assert post.call_count == 1
    url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url.startswith("http://127.0.0.1:11434")
    assert payload["format"] == P.NOTES_SCHEMA and payload["options"]["temperature"] == 0
    assert "[1] (00:00)" in payload["messages"][1]["content"]
    assert notes["key_points"][0]["start"] == segs[2].start


def test_extract_long_transcript_is_chunked_then_merged():
    segs = P.load_transcript(str(SAMPLE))
    raw = {"title": "t", "summary": "s", "key_points": [{"point": "p", "segments": [1]}], "action_items": []}
    with mock.patch.object(P.requests, "post", return_value=fake_model_reply(raw)) as post:
        P.extract_notes(segs, max_chars=500)
    n_chunks = len(P._chunks(P._lines(segs), 500))
    assert n_chunks > 1 and post.call_count == n_chunks + 1
    assert post.call_args.kwargs["json"]["messages"][0]["content"] == P.MERGE_PROMPT


def test_check_ollama_messages():
    with mock.patch.object(P.requests, "get", side_effect=P.requests.ConnectionError()):
        ok, msg = P.check_ollama()
    assert not ok and "not running" in msg
    tags = FakeResp({"models": [{"name": "llama3.2:3b"}]})
    with mock.patch.object(P.requests, "get", return_value=tags):
        assert P.check_ollama("llama3.2:3b")[0]
        ok, msg = P.check_ollama("llama3.2:1b")
    assert not ok and "ollama pull llama3.2:1b" in msg


def test_transcribe_with_fake_whisper_skips_empty_segments():
    seg = lambda a, b, t: types.SimpleNamespace(start=a, end=b, text=t)  # noqa: E731

    class FakeModel:
        def __init__(self, *a, **kw):
            assert kw["local_files_only"] is True and kw["device"] == "cpu"

        def transcribe(self, path, **kw):
            assert kw["vad_filter"] is True
            return iter([seg(0, 2, " Hello team. "), seg(2, 3, "  "), seg(3, 6, "Pilot starts soon.")]), \
                types.SimpleNamespace(language="en", duration=6.0)

    progress = []
    with mock.patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeModel)}):
        segs, info = P.transcribe("x.mp4", on_progress=progress.append)
    assert [s.text for s in segs] == ["Hello team.", "Pilot starts soon."] and [s.id for s in segs] == [1, 2]
    assert info["language"] == "en" and progress[-1] == 1.0


def test_privacy_receipt_and_markdown():
    r = P.external_connections()
    assert set(r) == {"checked", "external", "notes"} and r["checked"]
    segs = P.load_transcript(str(SAMPLE))
    notes = P.validate_notes({"title": "T", "summary": "S",
                              "key_points": [{"point": "p", "segments": [1]}],
                              "action_items": [{"owner": "Sai", "task": "x", "due": "Dec 10", "segments": [2]}]}, segs)
    md = P.to_markdown(notes, segs)
    assert md.startswith("# T") and "| Sai | x | Dec 10 |" in md and "No data left the machine" in md


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
