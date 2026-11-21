"""Where the work happens, as seen by the page (app.py).

LocalBackend   runs everything inside this Streamlit process (run_demo.ps1, run_lan.ps1).
RemoteBackend  sends the work to the PC service (server.py) over HTTPS, for the hosted
               Streamlit Cloud page. Only `requests` is needed on that side; no models.

Both offer the same calls, so app.py does not care which one it has:
  status(llm), warm_up(llm), start(job, llm, whisper) -> run, run.poll(), ask(ref, ...),
  receipt(), end(ref)
run.poll() advances the run one step (for a remote upload: sends the next piece) and returns
{"state", "snapshot", "position"?, "result"?, "error"?}; "result" holds segments, notes,
source, timings, notices and "ref" (what Q&A needs: the in-memory index, or the job id)."""
from __future__ import annotations

import threading
from typing import Optional

import requests

import pipeline as P
from segments import Segment


class Busy(Exception):
    """The PC is already serving the maximum number of sessions."""


# ------------------------------------------------------------------ local
class LocalRun:
    """Runs `work(tracker)` on a background thread; poll() reports progress, then the result."""

    def __init__(self, work, tracker, finish, lock: threading.Lock):
        self.tracker, self._finish = tracker, finish
        self._out: dict = {}

        def target():
            try:
                self._out["res"] = work(tracker)
            except Exception as e:  # noqa: BLE001  (shown on the page)
                self._out["error"] = str(e)
            finally:
                # Released by the worker, not the page: if the tab closes mid-run, the work
                # still finishes before the next person can start.
                lock.release()

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def poll(self) -> dict:
        snap = self.tracker.snapshot()
        if self._thread.is_alive():
            return {"state": "running", "snapshot": snap}
        if "error" in self._out:
            return {"state": "error", "snapshot": snap, "error": self._out["error"]}
        return {"state": "done", "snapshot": self.tracker.snapshot(), "result": self._finish(self._out["res"])}


class LocalBackend:
    remote = False

    def __init__(self, lock: threading.Lock, serving_port: Optional[int] = None):
        self._lock, self._port = lock, serving_port

    def status(self, llm: str) -> dict:
        import rag
        ok_llm, msg_llm = P.check_ollama(llm)
        ok_emb, msg_emb = P.check_ollama(rag.EMBED_MODEL)
        return {"ok_llm": ok_llm, "msg_llm": msg_llm, "ok_emb": ok_emb, "msg_emb": msg_emb,
                "cuda": P.cuda_available(), "loaded": P.model_loaded(llm) if ok_llm else False,
                "processing": self._lock.locked()}

    def warm_up(self, llm: str) -> None:
        P.warm_up(llm)

    can_rename = True

    def _run(self, work, tracker, finish) -> LocalRun:
        if not self._lock.acquire(blocking=False):
            raise Busy()
        try:
            return LocalRun(work, tracker, finish, self._lock)
        except Exception:
            self._lock.release()
            raise

    def start(self, job: dict, llm: str, whisper: str, rates: dict) -> LocalRun:
        import jobs as J
        tracker = J.make_tracker(job["kind"], len(job.get("data") or b""), rates)

        def finish(res):
            J.learn_rates(rates, res["stats"], llm)
            return {"segments": res["segments"], "notes": res["notes"], "source": res["source"],
                    "timings": res["timings"], "ref": res["index"], "speakers": res["speakers"],
                    "notices": J.notices(res["stats"])}

        return self._run(lambda tr: J.process(job, tr, rates, llm, whisper), tracker, finish)

    def rename(self, segments, names: dict, llm: str, rates: dict) -> LocalRun:
        """Apply corrected speaker names, then rebuild the index and bullet points."""
        import jobs as J
        from progress import Stage, Tracker
        tracker = Tracker([Stage("index", "Updating the Q&A index", 1.0),
                           Stage("notes", "Rewriting bullet points", 12.0)])

        def finish(res):
            return {"segments": res["segments"], "notes": res["notes"], "ref": res["index"],
                    "timings": res["timings"]}

        return self._run(lambda tr: J.rewrite(segments, names, tr, rates, llm), tracker, finish)

    def ask(self, ref, segments: list[Segment], question: str, llm: str) -> dict:
        import rag
        return rag.answer(question, ref, segments, model=llm)

    def receipt(self) -> dict:
        return P.external_connections(serving_port=self._port)

    def end(self, ref) -> None:
        pass  # the index lives in this session's memory and goes with it


# ------------------------------------------------------------------ remote
class RemoteError(Exception):
    pass


class RemoteBackend:
    remote = True
    can_rename = False  # renaming speakers is only built for the local app

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self._s = requests.Session()
        self._s.headers.update({
            "Authorization": f"Bearer {token}",
            "ngrok-skip-browser-warning": "1",   # ngrok's free plan shows a warning page otherwise
            "User-Agent": "snap-assist-cloud",
        })

    def _call(self, method: str, path: str, timeout: float = 15, **kw) -> requests.Response:
        try:
            r = self._s.request(method, self.url + path, timeout=timeout, **kw)
        except requests.RequestException as e:
            raise RemoteError("Can't reach the SNAP Assist PC. It may be off, asleep, or the "
                              "tunnel isn't running (run_server.ps1).") from e
        if r.status_code == 401:
            raise RemoteError("The PC refused the connection: the server token in the app's "
                              "secrets doesn't match the PC's.")
        if r.status_code == 503 and r.headers.get("content-type", "").startswith("application/json") \
                and r.json().get("error") == "busy":
            raise Busy()
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", r.text)
            except ValueError:
                msg = r.text[:200]
            raise RemoteError(msg)
        return r

    def status(self, llm: str) -> dict:
        try:
            return self._call("GET", "/health", params={"llm": llm}, timeout=8).json()
        except RemoteError as e:
            return {"ok_llm": False, "msg_llm": str(e), "ok_emb": True, "msg_emb": "",
                    "cuda": False, "loaded": False, "unreachable": True}

    def warm_up(self, llm: str) -> None:
        self._call("POST", "/warmup", params={"llm": llm}, timeout=300)

    def start(self, job: dict, llm: str, whisper: str, rates: dict = None) -> "RemoteRun":
        data = job.get("data") or b""
        body = {"kind": job["kind"], "name": job.get("name", ""), "size": len(data),
                "text": job.get("text", ""), "llm": llm, "whisper": whisper}
        created = self._call("POST", "/jobs", json=body).json()
        return RemoteRun(self, created["job_id"], data if job["kind"] in ("video", "file") else b"",
                         created["chunk_size"])

    def ask(self, ref, segments, question: str, llm: str) -> dict:
        return self._call("POST", f"/jobs/{ref}/ask", json={"question": question, "llm": llm},
                          timeout=180).json()

    def receipt(self) -> dict:
        try:
            return self._call("GET", "/receipt").json()
        except RemoteError:
            return None

    def end(self, ref) -> None:
        if ref:
            try:
                self._call("DELETE", f"/jobs/{ref}", timeout=5)
            except (RemoteError, Busy):
                pass  # the PC forgets idle sessions on its own


class RemoteRun:
    def __init__(self, backend: RemoteBackend, job_id: str, data: bytes, chunk_size: int):
        self.b, self.job_id, self._data, self._chunk = backend, job_id, data, chunk_size
        self._sent, self._started = 0, False

    def poll(self) -> dict:
        b = self.b
        if self._sent < len(self._data):  # still uploading: send the next piece
            piece = self._data[self._sent:self._sent + self._chunk]
            r = b._call("POST", f"/jobs/{self.job_id}/chunk", data=piece, timeout=300,
                        headers={"Content-Type": "application/octet-stream"}).json()
            self._sent = r["received"]
            return {"state": "uploading", "snapshot": r["snapshot"]}
        if not self._started:
            b._call("POST", f"/jobs/{self.job_id}/start")
            self._started = True
        out = b._call("GET", f"/jobs/{self.job_id}").json()
        if out["state"] == "done":
            res = out["result"]
            # rows: [id, start, end, text] or [id, start, end, text, speaker, speaker_id]
            res["segments"] = [Segment(int(r[0]), r[1], r[2], r[3], *r[4:6]) for r in res["segments"]]
            res["ref"] = self.job_id
        return out
