"""SNAP Assist app. Run with:  streamlit run app.py   (or run_demo.ps1 / run_lan.ps1)

Input: a meeting video/audio file, or a transcript/notes file (or pasted text).
Output: 1) optional transcript download, 2) main bullet points, 3) Q&A over the transcript.

Three ways to run the same page (see backend.py):
- local   (run_demo.ps1)  everything on this computer, 127.0.0.1 only. Works in airplane mode.
- lan     (run_lan.ps1)   same, shared with the local network behind an access code.
- hosted  (Streamlit Cloud) the page runs in the cloud; the processing runs on the host PC
          (server.py) reached through an HTTPS tunnel. Selected by the SNAP_BACKEND_URL secret.
Nothing is written to disk in any mode: uploads, transcripts and Q&A indexes live in memory.

While a meeting is processed the page redraws an animated panel with % done, time remaining
and elapsed time (see progress.py for how the ETA is worked out)."""
from __future__ import annotations

import hmac
import os
import socket
import threading
import time
from pathlib import Path

import streamlit as st

import pipeline as P
import readers
import ui
from backend import Busy, LocalBackend, RemoteBackend, RemoteError
from jobs import new_rates
from progress import fmt_duration
from segments import fmt_ts, has_times, said, transcript_md, transcript_txt

VIDEO_EXT = sorted(e.lstrip(".") for e in P.VIDEO_TYPES)
TEXT_EXT = sorted(e.lstrip(".") for e in readers.TEXT_TYPES)
SUGGESTED = ["What was decided?", "What are the action items, and who owns them?",
             "What deadlines were mentioned?", "What risks or open issues came up?"]


def _setting(name: str) -> str:
    """Environment variable first (local runs), then Streamlit secrets (the hosted page)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # no secrets file at all: a normal local run
        return ""


BACKEND_URL = _setting("SNAP_BACKEND_URL")
MODE = "hosted" if BACKEND_URL else ("lan" if os.environ.get("SNAP_LAN") == "1" else "local")
ACCESS_CODE = _setting("SNAP_ACCESS_CODE")
SERVER_PORT = int(st.get_option("server.port") or 8501)
HOST = socket.gethostname()

st.set_page_config(page_title="SNAP Assist", page_icon="⚡", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
ss = st.session_state
for key, default in {"start": 0, "media": None, "media_name": None, "segments": None, "notes": None,
                     "ref": None, "qa": [], "timings": {}, "receipt": None, "source": None,
                     "pending_q": None, "rates": None, "run_id": 0, "speakers": None,
                     "speakers_off": ""}.items():
    ss.setdefault(key, default)
if ss.rates is None:
    ss.rates = new_rates()

if MODE == "local" and (st.get_option("server.address") or "") not in ("127.0.0.1", "localhost", "::1"):
    st.warning("This copy may be reachable from other devices on your network. For the private, "
               "this-computer-only demo, start it with run_demo.ps1 (or add --server.address 127.0.0.1).")

# ------------------------------------------------------------------ access code
if MODE == "hosted" and not ACCESS_CODE:
    st.markdown(ui.hero(MODE), unsafe_allow_html=True)
    st.error("This hosted copy has no access code configured, so it is locked. "
             "Add SNAP_ACCESS_CODE to the app's secrets.")
    st.stop()
if ACCESS_CODE and not ss.get("authed"):
    st.markdown(ui.hero(MODE), unsafe_allow_html=True)
    st.info("SNAP Assist is available to invited people. Enter the access code you were given."
            if MODE == "hosted" else f"SNAP Assist is shared from **{HOST}** on this network. "
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


# ------------------------------------------------------------------ backend
@st.cache_resource
def gpu_lock() -> threading.Lock:
    """One lock for the whole Streamlit server (every browser session): one meeting at a time."""
    return threading.Lock()


@st.cache_resource
def remote_backend(url: str, token: str) -> RemoteBackend:
    return RemoteBackend(url, token)


backend = (remote_backend(BACKEND_URL, _setting("SNAP_SERVER_TOKEN")) if MODE == "hosted"
           else LocalBackend(gpu_lock(), serving_port=SERVER_PORT if MODE == "lan" else None))


@st.cache_data(ttl=10, show_spinner=False)
def backend_status(llm: str, mode: str) -> dict:
    """Cached briefly: every click reruns the page, and in hosted mode this crosses the tunnel."""
    return backend.status(llm)


@st.cache_data(show_spinner=False)
def speakers_ready() -> tuple[bool, str]:
    import speakers
    return speakers.available()


def reset(media=None, media_name=None):
    if ss.get("ref") is not None:
        backend.end(ss.ref)  # hosted: frees this session's slot on the PC right away
    ss.update(media=media, media_name=media_name, segments=None, notes=None, ref=None,
              qa=[], timings={}, receipt=None, start=0, source=None, pending_q=None,
              speakers=None, speakers_off="")


def jump(seconds):
    ss.start = int(seconds or 0)


def is_video(name) -> bool:
    return bool(name) and Path(name).suffix.lower() in P.VIDEO_TYPES


def line_tuples(segments, ids=None):
    by_id = {s.id: s for s in segments}
    pick = [by_id[i] for i in ids if i in by_id] if ids is not None else segments
    return [(s.id, fmt_ts(s.start) if s.start is not None else "", said(s)) for s in pick]


def source_controls(item_start, ids, key):
    """A 'Source' popover with the cited lines, plus a timestamp jump button for video."""
    if not ids:
        return
    with st.popover("📍 Source", help="Show the transcript lines this came from"):
        st.markdown(ui.source_lines(line_tuples(ss.segments, ids)), unsafe_allow_html=True)
    if item_start is not None and ss.media is not None and is_video(ss.media_name):
        st.button(f"▶ {fmt_ts(item_start)}", key=key, on_click=jump, args=(item_start,),
                  help="Jump the video to this moment")


def busy_message() -> str:
    return ("⏳ The SNAP Assist PC is already serving the most people it allows at once. "
            "Please try again in a few minutes." if backend.remote else
            "⏳ Someone else is processing a meeting on this PC right now. Please try again in a minute.")


def run_with_progress(job: dict, llm: str, whisper_size: str) -> None:
    video = job["kind"] == "video"
    try:
        run = backend.start(job, llm, whisper_size, ss.rates)
    except Busy:
        st.warning(busy_message())
        return
    except RemoteError as e:
        st.error(str(e))
        return
    res = drive(run, video)
    if res is None:
        return
    notices = res["notices"]
    if notices.get("vad_off"):
        st.toast("Silence filter (VAD) unavailable: onnxruntime did not load, so the whole audio "
                 "was transcribed. Run: pip install onnxruntime==1.20.1")
    if notices.get("gpu_error"):
        st.toast("GPU transcription wasn't available, so the CPU was used. See README: GPU setup.")
    time.sleep(0.9)  # let the finished state (green check, 100%) show for a moment
    ss.update(segments=res["segments"], ref=res["ref"], notes=res["notes"], qa=[], start=0,
              source=res["source"], timings=res["timings"], receipt=backend.receipt(),
              speakers=res.get("speakers"), speakers_off=notices.get("speakers_off", ""),
              pending_q=None, run_id=ss.run_id + 1)
    st.rerun()


def rename_speakers(names: dict, llm: str) -> None:
    """Apply the names typed in the Speakers panel and rewrite the bullet points."""
    try:
        run = backend.rename(ss.segments, names, llm, ss.rates)
    except Busy:
        st.warning(busy_message())
        return
    res = drive(run, video=False)
    if res is None:
        return
    for s in ss.speakers:
        new = names.get(s["label"], "").strip()
        if new != s["name"]:  # only names the user actually changed count as theirs
            s["name"], s["source"] = new, ("you" if new else "")
    time.sleep(0.6)
    ss.update(segments=res["segments"], ref=res["ref"], notes=res["notes"], qa=[],
              timings={**ss.timings, **res["timings"]}, run_id=ss.run_id + 1)
    st.toast("Speaker names applied, and action items rewritten.")
    st.rerun()


def drive(run, video: bool):
    """Show the animated progress panel until `run` finishes. Returns its result, or None
    after showing an error."""
    panel = st.empty()
    failures = 0
    while True:
        try:
            out = run.poll()
            failures = 0
        except RemoteError as e:
            failures += 1  # a dropped request over the internet: retry a few times
            if failures >= 4:
                panel.empty()
                st.error(f"Lost the connection to the SNAP Assist PC: {e}")
                return None
            time.sleep(1.5)
            continue
        snap = out["snapshot"]
        if out["state"] == "queued":
            n = out.get("position", 0)
            snap = {**snap, "label": "Waiting for the PC",
                    "detail": f"{n} meeting{'s' if n != 1 else ''} ahead of you" if n else "You're next"}
        panel.markdown(ui.progress_panel(snap, video), unsafe_allow_html=True)
        if out["state"] in ("done", "error"):
            break
        if out["state"] != "uploading":  # uploads move as fast as the pieces go
            time.sleep(0.5 if backend.remote else 0.2)

    if out["state"] == "error":
        panel.empty()
        st.error(f"Something went wrong: {out.get('error')}")
        return None
    return out["result"]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    llm = st.selectbox("Language model (open source)", P.LLM_CHOICES,
                       help="qwen3:4b-instruct is the default. qwen3:1.7b uses less memory.")
    whisper_size = st.selectbox("Speech-to-text for video (Whisper)", ["base", "small"], index=1,
                                help="'base' is fastest; 'small' is the best balance.")
    stat = backend_status(llm, MODE)
    if stat.get("unreachable"):
        st.error(stat["msg_llm"])
        if st.button("↻ Check again"):
            backend_status.clear()
            st.rerun()
    else:
        (st.success if stat["ok_llm"] else st.error)(
            ("Host PC connected · " if MODE == "hosted" else "") + stat["msg_llm"])
        st.caption("Transcription: " + ("NVIDIA GPU available" if stat["cuda"] else "CPU (no usable NVIDIA GPU found)"))
        if MODE != "hosted":
            sp_ok, sp_msg = speakers_ready()
            st.caption("Speaker names from video: " + ("✅ ready" if sp_ok else f"⚠️ {sp_msg}"))
        if not stat["ok_emb"]:
            st.error(stat["msg_emb"])
        if MODE == "hosted":
            st.caption(f"People using the PC: {stat.get('sessions', 0)} of {stat.get('max_sessions', 2)}"
                       + (" · processing a meeting now" if stat.get("processing") else ""))
    ready = bool(stat["ok_llm"] and stat["ok_emb"])
    if stat["ok_llm"]:
        warm = stat["loaded"]
        st.caption("Model status: " + ("🟢 loaded in memory" if warm else "⚪ not loaded yet (first run adds ~15 s)"))
        if st.button("Warm up model", help="Loads the model into memory before you present.", disabled=warm):
            with st.spinner("Loading model..."):
                backend.warm_up(llm)
            backend_status.clear()
            st.toast("Model is loaded and ready.")
            st.rerun()

    st.divider()
    st.header("🔒 Privacy")
    if MODE == "hosted":
        st.caption("This page is hosted on Streamlit Community Cloud. Your file travels **encrypted** from "
                   "your browser through Streamlit Cloud and an ngrok tunnel to the **host PC**, where "
                   "open-source models process it. No cloud AI service sees it. **Nothing is saved**: the PC "
                   "forgets the meeting when you start another one or after 30 minutes without use.")
    elif MODE == "lan":
        st.caption(f"All processing happens on **{HOST}**, the PC hosting this app on "
                   "your local network. Files you upload go only to that PC and are not saved.")
        if stat.get("processing"):
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
        where = "the host PC's models" if MODE == "hosted" else "this app and its models"
        (st.success if n == 0 else st.error)(f"Privacy receipt: {n} external connections from {where}")
        if r.get("viewers"):
            st.caption(f"Plus {len(r['viewers'])} browser connection(s) from the local network "
                       "(people using this app). These stay inside your network.")
        st.caption("Checked: " + ", ".join(r["checked"]))
        for line in r["external"] + r["notes"]:
            st.caption(line)

# ------------------------------------------------------------------ input
st.markdown(ui.hero(MODE), unsafe_allow_html=True)
if MODE == "lan":
    st.caption(f"🖧 Shared from **{HOST}** on your local network · one meeting is processed at a time")

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
    st.info("The host PC isn't reachable right now. Try again shortly." if MODE == "hosted" else
            "Start Ollama and download the models (see the sidebar) to enable processing.")

if run or sample:
    if sample:
        reset()
        job = {"kind": "sample"}
    elif ss.media is not None and is_video(ss.media_name):
        backend.end(ss.ref)
        ss.ref = None
        job = {"kind": "video", "name": ss.media_name, "data": ss.media}
    elif ss.media is not None:
        backend.end(ss.ref)
        ss.ref = None
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
              help=f"Read {t.get('read', 0):.1f} s"
                   + (f" · speakers {t['speakers']:.1f} s" if "speakers" in t else "")
                   + f" · index {t.get('index', 0):.1f} s · bullets {t.get('notes', 0):.1f} s")
    st.caption(f"Source: {ss.source}")

    if ss.media is not None and is_video(ss.media_name):
        st.video(ss.media, start_time=ss.start)

    # ---------------- who's who (video only): names read from the meeting video, editable
    if ss.speakers:
        named = sum(1 for sp in ss.speakers if sp["name"])
        with st.expander(f"👥 Speakers · {len(ss.speakers)} voices, {named} named from the video",
                         expanded=named < len(ss.speakers)):
            st.caption("Each voice in the recording, with the name shown for it on screen in the meeting "
                       "video. Fix or fill in names, then rewrite: action items said as \"I'll …\" go to "
                       "whoever said them.")
            lines_of: dict = {}
            for s in segments:
                lines_of.setdefault(s.speaker_id, []).append(s)
            with st.form(f"speakers{ss.run_id}"):
                names = {}
                for sp in ss.speakers:
                    c1, c2 = st.columns([2, 5], vertical_alignment="center")
                    names[sp["label"]] = c1.text_input(sp["label"], value=sp["name"], placeholder="Type a name",
                                                       key=f"name{ss.run_id}_{sp['label']}")
                    source = {"screen": "📺 read from the video", "you": "✏️ set by you"}.get(
                        sp["source"], "❔ not identified: type a name")
                    example = max(lines_of.get(sp["label"], []), key=lambda s: len(s.text), default=None)
                    c2.markdown(f"<small>{source} · {sp['lines']} lines</small>"
                                + (f"<div class='sa-src'>“{ui.html.escape(example.text)}”</div>" if example else ""),
                                unsafe_allow_html=True)
                apply = st.form_submit_button("✍️ Apply names and rewrite action items", type="primary",
                                              disabled=not backend.can_rename)
            if apply:
                rename_speakers(names, llm)
    elif ss.speakers_off and ss.media is not None and is_video(ss.media_name):
        st.info(f"👥 Speakers weren't identified, so action items may not have owners. {ss.speakers_off}")

    tab_bul, tab_qa, tab_tx = st.tabs(["📝 Bullet points", "💬 Ask SNAP", "📄 Transcript"])

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
                thinking.markdown(ui.typing("Searching the transcript on the host PC" if MODE == "hosted"
                                            else "Searching the transcript on this device"),
                                  unsafe_allow_html=True)
                try:
                    res = backend.ask(ss.ref, segments, question, llm)
                    ss.qa.append({"q": question, **res})
                    ss.receipt = backend.receipt()
                    st.rerun()
                except RemoteError as e:
                    thinking.error(f"{e} If the session ended (30 minutes without use), process the meeting again.")
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
