"""SNAP Minutes demo app. Run with:  streamlit run app.py
Everything runs on this computer: faster-whisper for speech-to-text and a local
Ollama server for the language model. The app only listens on 127.0.0.1."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import pipeline as P

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SAMPLE = Path(__file__).parent / "sample" / "sample_transcript.json"

st.set_page_config(page_title="SNAP Minutes · on-device demo", page_icon="⚡", layout="wide")
ss = st.session_state
ss.setdefault("start", 0)
ss.setdefault("video_path", None)
ss.setdefault("segments", None)
ss.setdefault("notes", None)
ss.setdefault("timings", {})
ss.setdefault("receipt", None)


def jump(seconds):
    ss.start = int(seconds or 0)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    whisper_size = st.selectbox("Speech-to-text model (Whisper)", ["base", "small", "medium"], index=1,
                                help="'base' is fastest; 'small' is the best balance for a demo.")
    llm = st.selectbox("Language model (Ollama)", ["llama3.2:3b", "llama3.2:1b"],
                       help="Switch to 1b if the laptop is slow.")
    ok, msg = P.check_ollama(llm)
    (st.success if ok else st.error)(msg)
    if ok and st.button("Warm up model", help="Loads the model into memory before you present."):
        with st.spinner("Loading model..."):
            P.warm_up(llm)
        st.toast("Model is loaded and ready.")

    st.divider()
    st.header("🔒 Privacy")
    if st.button("Check: is this laptop offline?"):
        if P.internet_reachable():
            st.warning("Internet is reachable. Turn on airplane mode for the demo.")
        else:
            st.success("No internet connection. Everything below runs on this device.")
    if ss.receipt:
        r = ss.receipt
        n = len(r["external"])
        (st.success if n == 0 else st.error)(f"Privacy receipt: {n} external connections")
        st.caption("Checked: " + ", ".join(r["checked"]))
        for line in r["external"] + r["notes"]:
            st.caption(line)

# ------------------------------------------------------------------ header
st.title("⚡ SNAP Minutes")
st.caption("Upload a short meeting video. It is transcribed and summarized **on this device**; "
           "nothing is uploaded anywhere.")

left, right = st.columns([1, 1], gap="large")

with left:
    up = st.file_uploader("Meeting video or audio", type=["mp4", "mov", "mkv", "webm", "m4a", "mp3", "wav"])
    if up is not None:
        dest = DATA_DIR / up.name
        if ss.video_path != str(dest):
            dest.write_bytes(up.getbuffer())
            ss.update(video_path=str(dest), segments=None, notes=None, timings={}, receipt=None, start=0)
    c1, c2 = st.columns(2)
    run = c1.button("▶ Transcribe and summarize", type="primary",
                    disabled=not (ok and ss.video_path), use_container_width=True)
    sample = c2.button("Use sample transcript", disabled=not ok, use_container_width=True,
                       help="Skips the video; useful for rehearsing.")

    if ss.video_path and Path(ss.video_path).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
        st.video(ss.video_path, start_time=ss.start)
    elif ss.video_path:
        st.audio(ss.video_path, start_time=ss.start)

# ------------------------------------------------------------------ run
if run or sample:
    try:
        t0 = time.perf_counter()
        if sample:
            ss.video_path = None
            segments = P.load_transcript(str(SAMPLE))
        else:
            bar = st.progress(0.0, text="Step 1 of 2: transcribing on device...")
            segments, info = P.transcribe(ss.video_path, model_size=whisper_size,
                                          on_progress=lambda p: bar.progress(p, text=f"Step 1 of 2: transcribing on device... {p:.0%}"))
            bar.empty()
        t1 = time.perf_counter()
        with st.spinner(f"Step 2 of 2: finding key points with {llm} on device..."):
            notes = P.extract_notes(segments, model=llm)
        t2 = time.perf_counter()
        ss.update(segments=segments, notes=notes, start=0,
                  timings={"transcribe": t1 - t0, "notes": t2 - t1},
                  receipt=P.external_connections())
        st.rerun()
    except Exception as e:  # show a readable error during the demo
        st.error(f"Something went wrong: {e}")

# ------------------------------------------------------------------ results
if ss.notes:
    notes, segments = ss.notes, ss.segments
    with right:
        st.subheader(notes["title"])
        if notes["summary"]:
            st.write(notes["summary"])
        t = ss.timings
        st.caption(f"⏱ Transcription {t.get('transcribe', 0):.1f}s · Key points {t.get('notes', 0):.1f}s · "
                   f"{len(segments)} transcript lines")

        st.markdown("#### Key points")
        for i, kp in enumerate(notes["key_points"]):
            a, b = st.columns([5, 1])
            a.markdown(f"- {kp['point']}" + ("" if kp["verified"] else "  \n  :orange[⚠ not linked to the transcript]"))
            if kp["start"] is not None:
                b.button(P.fmt_ts(kp["start"]), key=f"kp{i}", on_click=jump, args=(kp["start"],),
                         help="Jump to this moment")

        st.markdown("#### Action items")
        if notes["action_items"]:
            for i, ai in enumerate(notes["action_items"]):
                a, b = st.columns([5, 1])
                a.markdown(f"**{ai['owner']}**: {ai['task']}  \n:gray[Due: {ai['due']}]")
                if ai["start"] is not None:
                    b.button(P.fmt_ts(ai["start"]), key=f"ai{i}", on_click=jump, args=(ai["start"],))
        else:
            st.caption("No action items were stated.")

        st.download_button("⬇ Download notes (Markdown)", P.to_markdown(notes, segments),
                           file_name="snap_minutes.md", mime="text/markdown")

    with left:
        with st.expander("Full transcript", expanded=False):
            for s in segments:
                st.markdown(f"`{P.fmt_ts(s.start)}` {s.text}")
