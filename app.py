"""SNAP Assist app. Run with:  streamlit run app.py

Input: a meeting video/audio file, or a transcript/notes file (or pasted text).
Output: 1) optional transcript download, 2) main bullet points, 3) Q&A over the transcript.
Everything runs on this computer, and nothing is written to disk: uploads, the
transcript and the Q&A index live in memory for this session only.

While a meeting is processed, the work runs in a background thread and the page redraws
an animated panel about five times a second with % done, time remaining and elapsed time
(see progress.py for how the ETA is worked out). The worker never touches Streamlit;
it only reports progress to the Tracker, which the page reads."""
from __future__ import annotations

import hmac
import io
import os
import socket
import threading
import time
from pathlib import Path

import streamlit as st

import pipeline as P
import rag
import readers
import ui
from progress import Stage, Tracker, fmt_duration
from segments import fmt_ts, has_times, transcript_md, transcript_txt

SAMPLE = Path(__file__).parent / "sample" / "sample_transcript.json"
VIDEO_EXT = sorted(e.lstrip(".") for e in P.VIDEO_TYPES)
TEXT_EXT = sorted(e.lstrip(".") for e in readers.TEXT_TYPES)
SUGGESTED = ["What was decided?", "What are the action items, and who owns them?",
             "What deadlines were mentioned?", "What risks or open issues came up?"]

# Starting speeds for the ETA, measured on the reference laptop (RTX 3060). After each run
# the app replaces them with what it measured, so later estimates fit this machine.
DEFAULT_RATES = {
    "whisper_rtf": {"cuda": 0.10, "cpu": 0.45},   # seconds of work per second of audio
    "whisper_load": 4.0,                          # seconds to load Whisper
    "tok_per_s": {"qwen3:4b-instruct": 80.0, "qwen3:1.7b": 140.0, "phi4-mini": 45.0},
    "prompt_chars_per_s": 8000.0,                 # how fast the model reads the transcript
    "llm_load": 16.0,                             # seconds to load a model that isn't in memory
    "embed_batch_s": 0.4,
}

st.set_page_config(page_title="SNAP Assist · on-device", page_icon="⚡", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
ss = st.session_state
for key, default in {"start": 0, "media": None, "media_name": None, "segments": None, "notes": None,
                     "index": None, "qa": [], "timings": {}, "receipt": None, "source": None,
                     "pending_q": None, "rates": None, "run_id": 0}.items():
    ss.setdefault(key, default)
if ss.rates is None:
    ss.rates = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_RATES.items()}

# ------------------------------------------------------------------ sharing on the local network
# run_lan.ps1 sets SNAP_LAN=1 and SNAP_ACCESS_CODE, and serves the app to the local network.
# The default (run_demo.ps1) listens on 127.0.0.1 only and needs no code.
LAN_MODE = os.environ.get("SNAP_LAN") == "1"
ACCESS_CODE = os.environ.get("SNAP_ACCESS_CODE", "")
SERVER_PORT = int(st.get_option("server.port") or 8501)

if ACCESS_CODE and not ss.get("authed"):
    st.markdown(ui.hero(), unsafe_allow_html=True)
    st.info(f"SNAP Assist is shared from **{socket.gethostname()}** on this network. "
            "Ask the host for the access code.")
    with st.form("gate"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(code.strip().encode(), ACCESS_CODE.encode()):
                ss.authed = True
                st.rerun()
            time.sleep(1.5)  # slows down guessing
            st.error("That code isn't right.")
    st.stop()


@st.cache_resource
def gpu_lock() -> threading.Lock:
    """One lock for the whole server (shared by every browser session): only one meeting is
    processed at a time, so two people never compete for the GPU and its memory."""
    return threading.Lock()


def privacy_receipt() -> dict:
    return P.external_connections(serving_port=SERVER_PORT if LAN_MODE else None)


def reset(media=None, media_name=None):
    ss.update(media=media, media_name=media_name, segments=None, notes=None, index=None,
              qa=[], timings={}, receipt=None, start=0, source=None, pending_q=None)


def jump(seconds):
    ss.start = int(seconds or 0)


def is_video(name) -> bool:
    return bool(name) and Path(name).suffix.lower() in P.VIDEO_TYPES


def line_tuples(segments, ids=None):
    by_id = {s.id: s for s in segments}
    pick = [by_id[i] for i in ids if i in by_id] if ids is not None else segments
    return [(s.id, fmt_ts(s.start) if s.start is not None else "", s.text) for s in pick]


def source_controls(item_start, ids, key):
    """A 'Source' popover with the cited lines, plus a timestamp jump button for video."""
    if not ids:
        return
    with st.popover("📍 Source", help="Show the transcript lines this came from"):
        st.markdown(ui.source_lines(line_tuples(ss.segments, ids)), unsafe_allow_html=True)
    if item_start is not None and ss.media is not None and is_video(ss.media_name):
        st.button(f"▶ {fmt_ts(item_start)}", key=key, on_click=jump, args=(item_start,),
                  help="Jump the video to this moment")


# ------------------------------------------------------------------ processing (worker thread)
def process(job: dict, tracker: Tracker, rates: dict, llm: str, whisper_size: str) -> dict:
    """Runs off the Streamlit thread: read or transcribe, build the index, write the bullets.
    Reports progress to `tracker` and returns results plus measured speeds."""
    stats: dict = {}
    t_in = time.perf_counter()
    tracker.begin("input")
    if job["kind"] == "video":
        tracker.update("input", 0.0, "Loading Whisper on this device")

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
    elif job["kind"] == "file":
        tracker.update("input", 0.3, f"Reading {job['name']}")
        segments, source = readers.read_file(job["name"], job["data"]), job["name"]
    elif job["kind"] == "sample":
        tracker.update("input", 0.3, "Reading the sample transcript")
        segments, source = readers.read_file(SAMPLE.name, SAMPLE.read_bytes()), "sample transcript"
    else:
        tracker.update("input", 0.3, "Reading the pasted text")
        segments, source = readers.read_text(job["text"]), "pasted text"
    tracker.finish("input")
    t_idx = stats["t_input_end"] = time.perf_counter()
    stats["input_s"] = t_idx - t_in

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
    return {"segments": segments, "source": source, "index": index, "notes": notes, "stats": stats,
            "timings": {"read": stats["input_s"], "index": stats["index_s"], "notes": stats["notes_s"]}}


def learn_rates(rates: dict, stats: dict, llm: str) -> None:
    """Fold this run's measured speeds into the estimates for the next run (kept in memory only)."""
    notes_tokens = stats.get("tokens_last_call")
    if stats.get("duration") and stats.get("t_audio"):
        audio_s = stats["t_input_end"] - stats["t_audio"]
        rates["whisper_rtf"][stats["device"]] = max(0.02, audio_s / stats["duration"])
    if stats.get("gen_window") and notes_tokens and stats["gen_window"] > 0.5:
        rates["tok_per_s"][llm] = max(5.0, notes_tokens / stats["gen_window"])
    if stats.get("loaded") and stats.get("first_token_s", 0) > 0.05:
        rates["prompt_chars_per_s"] = max(500.0, stats["prompt_chars"] / stats["first_token_s"])


def run_with_progress(job: dict, llm: str, whisper_size: str) -> None:
    video = job["kind"] == "video"
    tracker = Tracker([
        Stage("input", "Transcribing audio" if video else "Reading the input",
              # before Whisper reports the audio length, guess from the file size
              (ss.rates["whisper_load"] + len(job.get("data") or b"") / 1e6 * 1.5) if video else 0.4),
        Stage("index", "Building the Q&A index", 1.0),
        Stage("notes", "Writing bullet points", 15.0),
    ])
    out: dict = {}
    lock = gpu_lock()
    if not lock.acquire(blocking=False):
        st.warning("⏳ Someone else is processing a meeting on this PC right now. "
                   "Please try again in a minute.")
        return

    def target():
        try:
            out["result"] = process(job, tracker, ss_rates, llm, whisper_size)
        except Exception as e:  # noqa: BLE001  (shown to the user below)
            out["error"] = e
        finally:
            # Released by the worker, not the page: if the person closes the tab mid-run,
            # the work still finishes before the next person can start.
            lock.release()

    ss_rates = ss.rates  # the worker gets a plain dict, never session_state itself
    panel = st.empty()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    while worker.is_alive():
        panel.markdown(ui.progress_panel(tracker.snapshot(), video), unsafe_allow_html=True)
        time.sleep(0.2)
    worker.join()

    if "error" in out:
        panel.empty()
        st.error(f"Something went wrong: {out['error']}")
        return
    panel.markdown(ui.progress_panel(tracker.snapshot(), video), unsafe_allow_html=True)
    res = out["result"]
    learn_rates(ss.rates, res["stats"], llm)
    if res["stats"].get("vad_off"):
        st.toast("Silence filter (VAD) unavailable: onnxruntime did not load, so the whole audio was transcribed. "
                 "Run: pip install onnxruntime==1.20.1")
    if res["stats"].get("gpu_error"):
        st.toast("GPU transcription wasn't available, so the CPU was used. See README: GPU setup.")
    time.sleep(0.9)  # let the finished state (green check, 100%) show for a moment
    ss.update(segments=res["segments"], index=res["index"], notes=res["notes"], qa=[], start=0,
              source=res["source"], timings=res["timings"], receipt=privacy_receipt(),
              pending_q=None, run_id=ss.run_id + 1)
    st.rerun()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    llm = st.selectbox("Language model (open source)", P.LLM_CHOICES,
                       help="qwen3:4b-instruct is the default. qwen3:1.7b uses less memory.")
    whisper_size = st.selectbox("Speech-to-text for video (Whisper)", ["base", "small"], index=1,
                                help="'base' is fastest; 'small' is the best balance.")
    ok_llm, msg_llm = P.check_ollama(llm)
    ok_emb, msg_emb = P.check_ollama(rag.EMBED_MODEL)
    (st.success if ok_llm else st.error)(msg_llm)
    st.caption("Transcription: " + ("NVIDIA GPU available" if P.cuda_available() else "CPU (no usable NVIDIA GPU found)"))
    if not ok_emb:
        st.error(msg_emb)
    ready = ok_llm and ok_emb
    if ok_llm:
        warm = P.model_loaded(llm)
        st.caption("Model status: " + ("🟢 loaded in memory" if warm else "⚪ not loaded yet (first run adds ~15 s)"))
        if st.button("Warm up model", help="Loads the model into memory before you present.", disabled=warm):
            with st.spinner("Loading model..."):
                P.warm_up(llm)
            st.toast("Model is loaded and ready.")
            st.rerun()

    st.divider()
    st.header("🔒 Privacy")
    if LAN_MODE:
        st.caption(f"All processing happens on **{socket.gethostname()}**, the PC hosting this app on "
                   "your local network. Files you upload go only to that PC and are not saved.")
        if gpu_lock().locked():
            st.warning("⏳ A meeting is being processed right now.")
    else:
        st.caption("All processing happens on this computer. Nothing is saved after you close the app.")
        # Airplane mode would also cut the local network, so this check is for local-only mode.
        if st.button("Check: is this laptop offline?"):
            if P.internet_reachable():
                st.warning("Internet is reachable. Turn on airplane mode for the demo.")
            else:
                st.success("No internet connection. Everything runs on this device.")
    if ss.receipt:
        r = ss.receipt
        n = len(r["external"])
        (st.success if n == 0 else st.error)(f"Privacy receipt: {n} external connections")
        if r.get("viewers"):
            st.caption(f"Plus {len(r['viewers'])} browser connection(s) from the local network "
                       "(people using this app). These stay inside your network.")
        st.caption("Checked: " + ", ".join(r["checked"]))
        for line in r["external"] + r["notes"]:
            st.caption(line)

# ------------------------------------------------------------------ input
st.markdown(ui.hero(), unsafe_allow_html=True)
if LAN_MODE:
    st.caption(f"🖧 Shared from **{socket.gethostname()}** on your local network · one meeting is processed at a time")

tab_file, tab_paste = st.tabs(["📁 Upload a file", "📋 Paste text"])
pasted = None
with tab_file:
    up = st.file_uploader("Drop a meeting video or audio, or a transcript or notes file",
                          type=VIDEO_EXT + TEXT_EXT,
                          help="Video: " + ", ".join(VIDEO_EXT) + ". Text: " + ", ".join(TEXT_EXT))
    if up is not None and ss.media_name != up.name:
        reset(media=up.getvalue(), media_name=up.name)
    if up is not None:
        kind = "🎬 Video/audio: will be transcribed with Whisper" if is_video(up.name) else "📄 Text: read directly"
        st.caption(f"**{up.name}** · {len(ss.media or b'') / 1e6:.1f} MB · {kind}")
with tab_paste:
    pasted = st.text_area("Paste a transcript or meeting notes", height=160,
                          placeholder="[00:12] Precious: Let's start with the pilot...\n\nor plain notes")

c1, c2 = st.columns([2, 1])
has_input = ss.media is not None or bool((pasted or "").strip())
run = c1.button("▶ Process meeting", type="primary", width="stretch", disabled=not (ready and has_input))
sample = c2.button("✨ Use sample transcript", width="stretch", disabled=not ready,
                   help="A made-up 2-minute meeting, for rehearsing.")
if not ready:
    st.info("Start Ollama and download the models (see the sidebar) to enable processing.")

if run or sample:
    if sample:
        reset()
        job = {"kind": "sample"}
    elif ss.media is not None and is_video(ss.media_name):
        job = {"kind": "video", "name": ss.media_name, "data": ss.media}
    elif ss.media is not None:
        job = {"kind": "file", "name": ss.media_name, "data": ss.media}
    else:
        reset()
        job = {"kind": "text", "text": pasted}
    run_with_progress(job, llm, whisper_size)

# ------------------------------------------------------------------ outputs
if ss.segments:
    segments, notes = ss.segments, ss.notes
    t = ss.timings
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transcript lines", len(segments))
    m2.metric("Key points", len(notes["key_points"]))
    m3.metric("Action items", len(notes["action_items"]))
    m4.metric("Processed in", fmt_duration(sum(t.values())),
              help=f"Read {t.get('read', 0):.1f} s · index {t.get('index', 0):.1f} s · bullets {t.get('notes', 0):.1f} s")
    st.caption(f"Source: {ss.source}")

    if ss.media is not None and is_video(ss.media_name):
        st.video(ss.media, start_time=ss.start)

    tab_bul, tab_qa, tab_tx = st.tabs(["📝 Bullet points", "💬 Ask the transcript", "📄 Transcript"])

    with tab_bul:
        st.markdown(f"<div class='sa-card'><h3>{ui.html.escape(notes['title'])}</h3>"
                    f"<div>{ui.html.escape(notes['summary'])}</div></div>", unsafe_allow_html=True)
        st.markdown("#### Key points")
        for i, kp in enumerate(notes["key_points"]):
            a, b = st.columns([7, 2], vertical_alignment="center")
            a.markdown(f"**{i + 1}.** {kp['point']}"
                       + ("" if kp["verified"] else "  \n:orange[⚠ not linked to the transcript]"))
            with b:
                source_controls(kp["start"], kp["segments"], f"kp{i}")

        items = notes["action_items"]
        st.markdown("#### Action items")
        if items:
            keys = [f"done{ss.run_id}_{i}" for i in range(len(items))]
            n_done = sum(bool(ss.get(k)) for k in keys)
            st.progress(n_done / len(items), text=f"{n_done} of {len(items)} ticked off (this session only)")
            for i, ai in enumerate(items):
                a, b = st.columns([7, 2], vertical_alignment="center")
                with a:
                    st.checkbox(ai["task"], key=keys[i])
                    st.markdown(f"<span class='sa-chip owner'>👤 {ui.html.escape(ai['owner'])}</span>"
                                f"<span class='sa-chip'>📅 {ui.html.escape(ai['due'])}</span>"
                                + ("" if ai["verified"] else "<span class='sa-chip warn'>⚠ not linked</span>"),
                                unsafe_allow_html=True)
                with b:
                    source_controls(ai["start"], ai["segments"], f"ai{i}")
        else:
            st.caption("No action items were stated.")
        st.download_button("⬇ Download bullet points (Markdown)", P.to_markdown(notes, segments),
                           file_name="snap_assist_notes.md", mime="text/markdown")

    with tab_qa:
        st.caption("Answers come only from this transcript, with the lines they used. "
                   "If the answer isn't there, you'll see \"Not in the transcript.\"")

        def pick_suggestion():
            ss.pending_q = ss.suggest
            ss.suggest = None  # clear the pill so the same question can be picked again

        st.pills("Try a question", SUGGESTED, key="suggest", on_change=pick_suggestion)

        for i, item in enumerate(ss.qa):
            with st.chat_message("user"):
                st.markdown(item["q"])
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(item["answer"] if item["found"] else f":gray[{item['answer']}]")
                if item["segments"]:
                    with st.expander(f"Sources: lines {', '.join(str(s) for s in item['segments'])}"):
                        st.markdown(ui.source_lines(line_tuples(segments, item["segments"])),
                                    unsafe_allow_html=True)
                    if item["start"] is not None and ss.media is not None and is_video(ss.media_name):
                        st.button(f"▶ {fmt_ts(item['start'])}", key=f"qa{i}", on_click=jump,
                                  args=(item["start"],), help="Jump the video to this moment")

        q = st.chat_input("Ask about this meeting…", key="ask")
        question = (q or "").strip() or ss.pending_q
        if question:
            ss.pending_q = None
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="⚡"):
                thinking = st.empty()
                thinking.markdown(ui.typing(), unsafe_allow_html=True)
                try:
                    res = rag.answer(question, ss.index, segments, model=llm)
                    ss.qa.append({"q": question, **res})
                    ss.receipt = privacy_receipt()
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    thinking.error(f"Something went wrong: {e}")
        if ss.qa:
            st.button("🧹 Clear conversation", on_click=lambda: ss.update(qa=[]))

    with tab_tx:
        query = st.text_input("🔎 Search the transcript", placeholder="e.g. Riverbend, December, progress bar")
        rows = line_tuples(segments)
        if query.strip():
            rows = [r for r in rows if query.strip().lower() in r[2].lower()]
            st.caption(f"{len(rows)} of {len(segments)} lines match “{query.strip()}”")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download transcript (.txt)", transcript_txt(segments),
                           file_name="transcript.txt", mime="text/plain", width="stretch")
        d2.download_button("⬇ Download transcript (.md)", transcript_md(segments, notes["title"]),
                           file_name="transcript.md", mime="text/markdown", width="stretch")
        st.caption("Optional: the downloads include line numbers"
                   + (" and timestamps." if has_times(segments) else "."))
        st.markdown(ui.transcript_html(rows, query.strip()), unsafe_allow_html=True)
