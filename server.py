"""SNAP Assist PC service: does the processing for the hosted (Streamlit Cloud) page.

    hosted page (any device) --HTTPS--> ngrok tunnel --> this service on 127.0.0.1:8600 --> Whisper / Ollama

Start it with run_server.ps1, which also starts the tunnel. The models, the transcripts and
the Q&A indexes only ever exist on this PC, in memory; nothing is written to disk.

Rules:
- Every request needs "Authorization: Bearer <SNAP_SERVER_TOKEN>"; the hosted page has the
  token in its Streamlit secrets. Without it the service answers 401 and does nothing else.
- At most SNAP_MAX_SESSIONS (default 2) meetings are held at once. A session ends when the
  page deletes it, or after SESSION_IDLE_S without any request (closed tabs).
- One meeting is processed at a time (one GPU); the next waits in line.
- Uploads arrive in pieces (UPLOAD_CHUNK bytes each), because tunnels limit request size.

Endpoints (JSON):
  GET    /health?llm=...          models, GPU, and how many sessions are in use
  POST   /warmup?llm=...          load the language model into memory
  POST   /jobs                    {kind, name, size, text, llm, whisper} -> {job_id}
  POST   /jobs/{id}/chunk         raw bytes (in order)                   -> {received, snapshot}
  POST   /jobs/{id}/start         queue it                              -> {position}
  GET    /jobs/{id}               {state, position, snapshot, result?, error?}
  POST   /jobs/{id}/ask           {question, llm} -> answer with cited lines
  DELETE /jobs/{id}               forget it now
  GET    /receipt                 this PC's non-local network connections (the privacy receipt)
"""
from __future__ import annotations

import hmac
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import jobs as J
import pipeline as P
import rag
from progress import Tracker

TOKEN = os.environ.get("SNAP_SERVER_TOKEN", "")
MAX_SESSIONS = int(os.environ.get("SNAP_MAX_SESSIONS", "2"))
PORT = int(os.environ.get("SNAP_SERVER_PORT", "8600"))
SESSION_IDLE_S = 30 * 60        # a finished meeting is kept this long after the last request
UPLOAD_IDLE_S = 10 * 60         # an unfinished upload is dropped after this
UPLOAD_CHUNK = 32 * 1024 * 1024
MAX_UPLOAD = 500 * 1024 * 1024
KINDS = {"video", "file", "text", "sample"}


@dataclass
class Job:
    id: str
    kind: str
    name: str
    size: int
    llm: str
    whisper: str
    text: str = ""
    data: bytearray = field(default_factory=bytearray)
    state: str = "uploading"      # uploading | queued | running | done | error
    tracker: Optional[Tracker] = None
    result: Optional[dict] = None
    index: object = None
    segments: list = field(default_factory=list)
    error: str = ""
    last_seen: float = field(default_factory=time.time)
    last_queued: float = 0.0


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()           # guards JOBS
LINE: "queue.Queue[str]" = queue.Queue()
RATES = J.new_rates()             # measured speeds, shared by every session


def _expired(job: Job, now: float) -> bool:
    idle = now - job.last_seen
    if job.state == "uploading":
        return idle > UPLOAD_IDLE_S
    return job.state in ("done", "error") and idle > SESSION_IDLE_S


def _cleanup() -> None:
    now = time.time()
    with LOCK:
        for jid in [j.id for j in JOBS.values() if _expired(j, now)]:
            del JOBS[jid]


def _position(job: Job) -> int:
    """How many meetings are ahead of this one (0 = being processed or next)."""
    with LOCK:
        ahead = [j for j in JOBS.values() if j.state in ("queued", "running") and j.id != job.id]
        if job.state == "running":
            return 0
        return sum(1 for j in ahead if j.state == "running" or j.last_queued < job.last_queued)


def worker() -> None:
    """The only thread that runs models for meetings: one at a time, in arrival order."""
    while True:
        jid = LINE.get()
        with LOCK:
            job = JOBS.get(jid)
        if job is None:  # deleted while waiting
            continue
        job.state = "running"
        try:
            payload = {"kind": job.kind, "name": job.name, "data": bytes(job.data), "text": job.text}
            res = J.process(payload, job.tracker, RATES, job.llm, job.whisper)
            J.learn_rates(RATES, res["stats"], job.llm)
            job.index, job.segments = res["index"], res["segments"]
            job.result = J.result_to_json(res)
            job.state = "done"
        except Exception as e:  # noqa: BLE001  (reported to the page)
            job.error, job.state = str(e)[:500], "error"
        finally:
            job.data = bytearray()  # the upload is not needed any more
            job.last_seen = time.time()


# ------------------------------------------------------------------ HTTP
def _authorized(request: Request) -> bool:
    given = request.headers.get("authorization", "")
    return bool(TOKEN) and hmac.compare_digest(given.encode(), f"Bearer {TOKEN}".encode())


def _job(request: Request) -> Optional[Job]:
    with LOCK:
        job = JOBS.get(request.path_params["job_id"])
    if job:
        job.last_seen = time.time()
    return job


def _err(status: int, msg: str, **extra) -> JSONResponse:
    return JSONResponse({"error": msg, **extra}, status_code=status)


async def health(request: Request):
    _cleanup()
    llm = request.query_params.get("llm", P.DEFAULT_LLM)

    def check():  # blocking HTTP calls to Ollama: run off the event loop
        ok_llm, msg_llm = P.check_ollama(llm)
        ok_emb, msg_emb = P.check_ollama(rag.EMBED_MODEL)
        return {"ok_llm": ok_llm, "msg_llm": msg_llm, "ok_emb": ok_emb, "msg_emb": msg_emb,
                "cuda": P.cuda_available(), "loaded": P.model_loaded(llm) if ok_llm else False}

    out = await run_in_threadpool(check)
    with LOCK:
        out.update(sessions=len(JOBS), max_sessions=MAX_SESSIONS,
                   processing=any(j.state == "running" for j in JOBS.values()))
    return JSONResponse(out)


async def warmup(request: Request):
    await run_in_threadpool(P.warm_up, request.query_params.get("llm", P.DEFAULT_LLM))
    return JSONResponse({"ok": True})


async def create_job(request: Request):
    _cleanup()
    body = await request.json()
    kind, size = body.get("kind"), int(body.get("size") or 0)
    if kind not in KINDS:
        return _err(400, "Unknown input kind.")
    if size > MAX_UPLOAD:
        return _err(413, f"Files up to {MAX_UPLOAD // 2**20} MB are supported.")
    if body.get("llm") not in P.LLM_CHOICES or body.get("whisper") not in ("base", "small"):
        return _err(400, "Unknown model choice.")
    with LOCK:
        if len(JOBS) >= MAX_SESSIONS:
            return _err(503, "busy", sessions=len(JOBS), max_sessions=MAX_SESSIONS)
        job = Job(uuid.uuid4().hex, kind, str(body.get("name") or "")[:200], size,
                  body["llm"], body["whisper"], text=str(body.get("text") or ""))
        job.tracker = J.make_tracker(kind, size, RATES, upload=kind in ("video", "file"))
        JOBS[job.id] = job
    return JSONResponse({"job_id": job.id, "chunk_size": UPLOAD_CHUNK}, status_code=201)


async def chunk(request: Request):
    job = _job(request)
    if job is None:
        return _err(404, "This session has ended. Process the meeting again.")
    if job.state != "uploading":
        return _err(409, "Upload already finished.")
    data = await request.body()
    if len(data) > UPLOAD_CHUNK or len(job.data) + len(data) > max(job.size, 1):
        return _err(413, "Upload is larger than announced.")
    job.data.extend(data)
    if job.tracker.snapshot()["current"] != "upload":
        job.tracker.begin("upload", "Sending the file over the encrypted tunnel")
    job.tracker.update("upload", len(job.data) / max(job.size, 1),
                       f"{len(job.data) / 1e6:.1f} of {job.size / 1e6:.1f} MB received")
    return JSONResponse({"received": len(job.data), "snapshot": job.tracker.snapshot()})


async def start(request: Request):
    job = _job(request)
    if job is None:
        return _err(404, "This session has ended.")
    if job.state != "uploading":
        return _err(409, "Already started.")
    if job.kind in ("video", "file"):
        if len(job.data) != job.size:
            return _err(400, "The upload is incomplete.")
        job.tracker.finish("upload")
    job.state = "queued"
    job.last_queued = time.time()
    LINE.put(job.id)
    return JSONResponse({"position": _position(job)})


async def status(request: Request):
    job = _job(request)
    if job is None:
        return _err(404, "This session has ended. Process the meeting again.")
    out = {"state": job.state, "snapshot": job.tracker.snapshot()}
    if job.state == "queued":
        out["position"] = _position(job)
    if job.state == "done":
        out["result"] = job.result
    if job.state == "error":
        out["error"] = job.error
    return JSONResponse(out)


async def ask(request: Request):
    job = _job(request)
    if job is None or job.state != "done":
        return _err(404, "This session has ended. Process the meeting again.")
    body = await request.json()
    llm = body.get("llm") if body.get("llm") in P.LLM_CHOICES else P.DEFAULT_LLM
    try:
        res = await run_in_threadpool(rag.answer, str(body.get("question", ""))[:1000],
                                      job.index, job.segments, model=llm)
    except Exception as e:  # noqa: BLE001
        return _err(500, str(e)[:300])
    return JSONResponse(res)


async def delete(request: Request):
    with LOCK:
        JOBS.pop(request.path_params["job_id"], None)
    return JSONResponse({"ok": True})


async def receipt(request: Request):
    return JSONResponse(await run_in_threadpool(P.external_connections))


ROUTES = [
    Route("/health", health), Route("/warmup", warmup, methods=["POST"]),
    Route("/jobs", create_job, methods=["POST"]),
    Route("/jobs/{job_id}", status), Route("/jobs/{job_id}", delete, methods=["DELETE"]),
    Route("/jobs/{job_id}/chunk", chunk, methods=["POST"]),
    Route("/jobs/{job_id}/start", start, methods=["POST"]),
    Route("/jobs/{job_id}/ask", ask, methods=["POST"]),
    Route("/receipt", receipt),
]


class TokenGate:
    """Rejects every request without the right token before any route runs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not _authorized(Request(scope)):
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


app = TokenGate(Starlette(routes=ROUTES))

if __name__ == "__main__":
    if len(TOKEN) < 24:
        raise SystemExit("Set SNAP_SERVER_TOKEN (24+ characters). run_server.ps1 does this for you.")
    threading.Thread(target=worker, daemon=True).start()
    # 127.0.0.1 only: the tunnel connects from this PC, nothing on the network can reach it directly.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
