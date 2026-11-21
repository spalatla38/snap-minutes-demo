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
    def __init__(self, job: dict, llm: str, whisper: str, rates: dict, lock: threading.Lock):
        import jobs as J
        self._J = J
        self.tracker = J.make_tracker(job["kind"], len(job.get("data") or b""), rates)
        self._out: dict = {}
        self._rates, self._llm = rates, llm

        def target():
            try:
                self._out["res"] = J.process(job, self.tracker, rates, llm, whisper)
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
        res = self._out["res"]
        self._J.learn_rates(self._rates, res["stats"], self._llm)
        return {"state": "done", "snapshot": self.tracker.snapshot(), "result": {
            "segments": res["segments"], "notes": res["notes"], "source": res["source"],
            "timings": res["timings"], "ref": res["index"],
            "notices": {"gpu_error": bool(res["stats"].get("gpu_error")),
                        "vad_off": bool(res["stats"].get("vad_off"))}}}


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

    def start(self, job: dict, llm: str, whisper: str, rates: dict) -> LocalRun:
        if not self._lock.acquire(blocking=False):
            raise Busy()
        try:
            return LocalRun(job, llm, whisper, rates, self._lock)
        except Exception:
            self._lock.release()
            raise

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
            res["segments"] = [Segment(int(i), s, e, t) for i, s, e, t in res["segments"]]
            res["ref"] = self.job_id
        return out
