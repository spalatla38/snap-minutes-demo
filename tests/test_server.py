"""Offline tests for the PC service (server.py) and the hosted page's client (backend.RemoteBackend).

A real server.py runs on a free 127.0.0.1 port in a background thread, and RemoteBackend talks
to it over HTTP exactly as the hosted page would through the tunnel. Ollama is faked.
    python tests/test_server.py      (or)      python -m pytest tests -q
"""
import re
import socket
import sys
import threading
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import uvicorn  # noqa: E402

import backend as B  # noqa: E402
import pipeline as P  # noqa: E402
import rag  # noqa: E402
import server as S  # noqa: E402
from test_pipeline import FakeResp, FakeStream, bag_of_words  # noqa: E402

TOKEN = "t" * 40
NOTES = {"title": "Pilot", "summary": "s", "action_items": [{"owner": "Sai", "task": "Build the bar",
         "due": "Dec 10", "segments": [2]}], "key_points": [{"point": "Pilot starts", "segments": [1]}]}


def fake_post(url, json=None, timeout=None, stream=False):  # noqa: A002
    """Fake Ollama: embeddings, streamed notes, and answers."""
    if url.endswith("/api/embed"):
        return FakeResp({"embeddings": [bag_of_words(t) for t in json["input"]]})
    if url.endswith("/api/generate"):
        return FakeResp({})
    if json["format"] is rag.ANSWER_SCHEMA:  # cite the first line the server retrieved
        first = int(re.search(r"^\[(\d+)\]", json["messages"][1]["content"], re.M).group(1))
        return FakeResp({"message": {"content": __import__("json").dumps(
            {"found": True, "answer": "Sai builds it.", "lines": [first]})}})
    body = __import__("json").dumps(NOTES)
    time.sleep(0.3)  # long enough for a second job to queue behind this one
    return FakeStream([body[i:i + 7] for i in range(0, len(body), 7)])


def fake_get(url, timeout=None):
    if url.endswith("/api/tags"):
        return FakeResp({"models": [{"name": m} for m in P.LLM_CHOICES] + [{"name": "all-minilm:latest"}]})
    return FakeResp({"models": []})  # /api/ps: nothing loaded


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
URL = f"http://127.0.0.1:{PORT}"
_patches = [mock.patch.object(P.requests, "post", side_effect=fake_post),
            mock.patch.object(P.requests, "get", side_effect=fake_get),
            mock.patch.object(S, "TOKEN", TOKEN), mock.patch.object(S, "MAX_SESSIONS", 2)]


def setup_module(module=None):
    for p in _patches:
        p.start()
    threading.Thread(target=S.worker, daemon=True).start()
    server = uvicorn.Server(uvicorn.Config(S.app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return
        time.sleep(0.05)
    raise RuntimeError("server did not start")


def _clear():
    with S.LOCK:
        S.JOBS.clear()


def run_to_end(run, limit=20.0):
    states, t0 = [], time.time()
    while time.time() - t0 < limit:
        out = run.poll()
        states.append(out["state"])
        if out["state"] in ("done", "error"):
            return out, states
        time.sleep(0.05)
    raise AssertionError(f"timed out: {states[-5:]}")


def test_requests_without_the_token_are_refused():
    _clear()
    import urllib.error
    import urllib.request
    try:  # urllib, because requests.get is replaced by the fake Ollama in these tests
        urllib.request.urlopen(URL + "/health", timeout=5)
        raise AssertionError("no token, but the request was accepted")
    except urllib.error.HTTPError as e:
        assert e.code == 401
    bad = B.RemoteBackend(URL, "wrong" * 10)
    try:
        bad.start({"kind": "sample"}, P.DEFAULT_LLM, "small")
        raise AssertionError("should have been refused")
    except B.RemoteError as e:
        assert "token" in str(e)
    assert not S.JOBS  # nothing was created


def test_health_reports_models_and_sessions():
    _clear()
    st = B.RemoteBackend(URL, TOKEN).status(P.DEFAULT_LLM)
    assert st["ok_llm"] and st["ok_emb"] and st["sessions"] == 0 and st["max_sessions"] == 2


def test_sample_runs_end_to_end_and_answers_questions():
    _clear()
    be = B.RemoteBackend(URL, TOKEN)
    out, states = run_to_end(be.start({"kind": "sample"}, P.DEFAULT_LLM, "small"))
    res = out["result"]
    assert out["state"] == "done" and res["notes"]["title"] == "Pilot" and res["segments"]
    assert res["segments"][0].id == 1 and res["notes"]["action_items"][0]["verified"]
    assert out["snapshot"]["done"] and out["snapshot"]["percent"] == 1.0
    ans = be.ask(res["ref"], res["segments"], "Who will build the progress bar?", P.DEFAULT_LLM)
    assert ans["found"] and len(ans["segments"]) == 1 and set(ans["segments"]) <= set(ans["sources"])
    be.end(res["ref"])
    assert res["ref"] not in S.JOBS


def test_upload_in_pieces_reports_upload_progress():
    _clear()
    be = B.RemoteBackend(URL, TOKEN)
    text = (ROOT / "sample" / "sample_notes.md").read_bytes()
    with mock.patch.object(S, "UPLOAD_CHUNK", 100):  # force several pieces
        run = be.start({"kind": "file", "name": "notes.md", "data": text}, P.DEFAULT_LLM, "small")
        run._chunk = 100
        out, states = run_to_end(run)
    assert states.count("uploading") == -(-len(text) // 100)
    assert out["state"] == "done" and out["result"]["source"] == "notes.md"
    labels = [s["key"] for s in out["snapshot"]["stages"]]
    assert labels[0] == "upload"


def test_only_two_sessions_and_the_second_waits_in_line():
    _clear()
    be = B.RemoteBackend(URL, TOKEN)
    first = be.start({"kind": "sample"}, P.DEFAULT_LLM, "small")
    second = be.start({"kind": "text", "text": "[00:01] Sai: Hello team."}, P.DEFAULT_LLM, "small")
    try:
        be.start({"kind": "sample"}, P.DEFAULT_LLM, "small")
        raise AssertionError("a third session should be refused")
    except B.Busy:
        pass
    first.poll()                    # first starts processing (the fake model takes 0.3 s)
    time.sleep(0.05)
    out2 = second.poll()
    assert out2["state"] == "queued" and out2["position"] == 1
    a, _ = run_to_end(first)
    b, _ = run_to_end(second)
    assert a["state"] == b["state"] == "done"
    be.end(a["result"]["ref"])      # ending a session frees a slot
    run_to_end(be.start({"kind": "sample"}, P.DEFAULT_LLM, "small"))


def test_bad_input_is_rejected():
    _clear()
    be = B.RemoteBackend(URL, TOKEN)
    for body in [{"kind": "exe"}, {"kind": "sample", "llm": "gpt-4"}]:
        try:
            be._call("POST", "/jobs", json={"size": 0, "llm": P.DEFAULT_LLM, "whisper": "small", **body})
            raise AssertionError(f"accepted {body}")
        except B.RemoteError:
            pass
    created = be._call("POST", "/jobs", json={"kind": "file", "name": "x.md", "size": 5,
                                              "llm": P.DEFAULT_LLM, "whisper": "small"}).json()
    try:  # more bytes than announced
        be._call("POST", f"/jobs/{created['job_id']}/chunk", data=b"0123456789")
        raise AssertionError("oversized upload accepted")
    except B.RemoteError:
        pass


def test_idle_sessions_expire():
    _clear()
    be = B.RemoteBackend(URL, TOKEN)
    out, _ = run_to_end(be.start({"kind": "sample"}, P.DEFAULT_LLM, "small"))
    ref = out["result"]["ref"]
    S.JOBS[ref].last_seen -= S.SESSION_IDLE_S + 1
    assert be.status(P.DEFAULT_LLM)["sessions"] == 0  # health runs the cleanup
    try:
        be.ask(ref, [], "anything?", P.DEFAULT_LLM)
        raise AssertionError("expired session answered")
    except B.RemoteError as e:
        assert "ended" in str(e)


def test_unreachable_pc_is_reported_not_raised():
    st = B.RemoteBackend("http://127.0.0.1:9", TOKEN).status(P.DEFAULT_LLM)
    assert st.get("unreachable") and "Can't reach" in st["msg_llm"]


if __name__ == "__main__":
    setup_module()
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
