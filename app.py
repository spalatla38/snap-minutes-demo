"""SNAP Minutes app. Run with:  streamlit run app.py

Input: a meeting video/audio file, or a transcript/notes file (or pasted text).
Output: 1) optional transcript download, 2) main bullet points, 3) Q&A over the transcript.
Everything runs on this computer, and nothing is written to disk: uploads, the
transcript and the Q&A index live in memory for this session only."""
from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st

import pipeline as P
import rag
import readers
from segments import fmt_ts, has_times, transcript_md, transcript_txt

SAMPLE = Path(__file__).parent / "sample" / "sample_transcript.json"
VIDEO_EXT = sorted(e.lstrip(".") for e in P.VIDEO_TYPES)
TEXT_EXT = sorted(e.lstrip(".") for e in readers.TEXT_TYPES)

st.set_page_config(page_title="SNAP Minutes · on-device", page_icon="⚡", layout="wide")
ss = st.session_state
for key, default in {"start": 0, "media": None, "media_name": None, "segments": None, "notes": None,
                     "index": None, "qa": [], "timings": {}, "receipt": None, "source": None}.items():
    ss.setdefault(key, default)


def reset(media=None, media_name=None):
    ss.update(media=media, media_name=media_name, segments=None, notes=None, index=None,
              qa=[], timings={}, receipt=None, start=0, source=None)


def jump(seconds):
    ss.start = int(seconds or 0)


def is_video(name) -> bool:
    return bool(name) and Path(name).suffix.lower() in P.VIDEO_TYPES


def cite_button(item_start, ids, key):
    """A timestamp button when there is a video; otherwise the line numbers."""
    if item_start is not None and ss.media is not None:
        st.button(fmt_ts(item_start), key=key, on_click=jump, args=(item_start,), help="Jump to this moment")
    elif ids:
        st.caption("line " + ", ".join(str(i) for i in ids[:3]))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    llm = st.selectbox("Language model (open source)", P.LLM_CHOICES,
                       help="qwen3:4b-instruct is the default. qwen3:1.7b uses less memory.")
    whisper_size = st.selectbox("Speech-to-text for video (Whisper)", ["base", "small", "medium"], index=1,
                                help="'base' is fastest; 'small' is the best balance.")
    ok_llm, msg_llm = P.check_ollama(llm)
    ok_emb, msg_emb = P.check_ollama(rag.EMBED_MODEL)
    (st.success if ok_llm else st.error)(msg_llm)
    if not ok_emb:
        st.error(msg_emb)
    ready = ok_llm and ok_emb
    if ok_llm and st.button("Warm up model", help="Loads the model into memory before you present."):
        with st.spinner("Loading model..."):
            P.warm_up(llm)
        st.toast("Model is loaded and ready.")

    st.divider()
    st.header("🔒 Privacy")
    st.caption("All processing happens on this computer. Nothing is saved after you close the app.")
    if st.button("Check: is this laptop offline?"):
        if P.internet_reachable():
            st.warning("Internet is reachable. Turn on airplane mode for the demo.")
        else:
            st.success("No internet connection. Everything runs on this device.")
    if ss.receipt:
        r = ss.receipt
        n = len(r["external"])
        (st.success if n == 0 else st.error)(f"Privacy receipt: {n} external connections")
        st.caption("Checked: " + ", ".join(r["checked"]))
        for line in r["external"] + r["notes"]:
            st.caption(line)

# ------------------------------------------------------------------ input
st.title("⚡ SNAP Minutes")
st.caption("Meeting video, transcript or notes in. Transcript, bullet points and answers out. "
           "**Everything runs on this computer.**")

tab_file, tab_paste = st.tabs(["📁 Upload a file", "📋 Paste text"])
pasted = None
with tab_file:
    up = st.file_uploader("Meeting video or audio, or a transcript or notes file",
                          type=VIDEO_EXT + TEXT_EXT,
                          help="Video: " + ", ".join(VIDEO_EXT) + ". Text: " + ", ".join(TEXT_EXT))
    if up is not None and ss.media_name != up.name:
        reset(media=up.getvalue(), media_name=up.name)
with tab_paste:
    pasted = st.text_area("Paste a transcript or meeting notes", height=160,
                          placeholder="[00:12] Precious: Let's start with the pilot...\n\nor plain notes")

c1, c2 = st.columns([2, 1])
run = c1.button("▶ Process meeting", type="primary", use_container_width=True,
                disabled=not (ready and (ss.media is not None or (pasted or "").strip())))
sample = c2.button("Use sample transcript", use_container_width=True, disabled=not ready,
                   help="A made-up 2-minute meeting, for rehearsing.")

# ------------------------------------------------------------------ processing
if run or sample:
    try:
        t0 = time.perf_counter()
        if sample:
            reset()
            segments, source = readers.read_file(SAMPLE.name, SAMPLE.read_bytes()), "sample transcript"
        elif ss.media is not None and is_video(ss.media_name):
            bar = st.progress(0.0, text="Step 1 of 3: transcribing on this device...")
            segments, _ = P.transcribe(io.BytesIO(ss.media), model_size=whisper_size,
                                       on_progress=lambda p: bar.progress(p, text=f"Step 1 of 3: transcribing on this device... {p:.0%}"))
            bar.empty()
            source = ss.media_name
        elif ss.media is not None:
            segments, source = readers.read_file(ss.media_name, ss.media), ss.media_name
        else:
            reset()
            segments, source = readers.read_text(pasted), "pasted text"
        t1 = time.perf_counter()
        with st.spinner("Step 2 of 3: building the Q&A index in memory..."):
            index = rag.build_index(segments)
        t2 = time.perf_counter()
        with st.spinner(f"Step 3 of 3: writing bullet points with {llm}..."):
            notes = P.extract_notes(segments, model=llm)
        t3 = time.perf_counter()
        ss.update(segments=segments, index=index, notes=notes, qa=[], start=0, source=source,
                  timings={"read": t1 - t0, "index": t2 - t1, "notes": t3 - t2},
                  receipt=P.external_connections())
        st.rerun()
    except Exception as e:  # show a readable error during the demo
        st.error(f"Something went wrong: {e}")

# ------------------------------------------------------------------ outputs
if ss.segments:
    segments, notes = ss.segments, ss.notes
    t = ss.timings
    st.divider()
    st.caption(f"Source: {ss.source} · {len(segments)} lines · read {t.get('read', 0):.1f}s · "
               f"index {t.get('index', 0):.1f}s · bullets {t.get('notes', 0):.1f}s")

    if ss.media is not None and is_video(ss.media_name):
        st.video(ss.media, start_time=ss.start)

    tab_bul, tab_qa, tab_tx = st.tabs(["📝 Bullet points", "💬 Ask the transcript", "📄 Transcript"])

    with tab_bul:
        st.subheader(notes["title"])
        if notes["summary"]:
            st.write(notes["summary"])
        st.markdown("#### Key points")
        for i, kp in enumerate(notes["key_points"]):
            a, b = st.columns([6, 1])
            a.markdown(f"- {kp['point']}" + ("" if kp["verified"] else "  \n  :orange[⚠ not linked to the transcript]"))
            with b:
                cite_button(kp["start"], kp["segments"], f"kp{i}")
        st.markdown("#### Action items")
        if notes["action_items"]:
            for i, ai in enumerate(notes["action_items"]):
                a, b = st.columns([6, 1])
                a.markdown(f"**{ai['owner']}**: {ai['task']}  \n:gray[Due: {ai['due']}]"
                           + ("" if ai["verified"] else "  \n:orange[⚠ not linked to the transcript]"))
                with b:
                    cite_button(ai["start"], ai["segments"], f"ai{i}")
        else:
            st.caption("No action items were stated.")
        st.download_button("⬇ Download bullet points (Markdown)", P.to_markdown(notes, segments),
                           file_name="snap_minutes_notes.md", mime="text/markdown")

    with tab_qa:
        st.caption("Answers come only from this transcript, with the lines they used. "
                   "If the answer isn't there, you'll see \"Not in the transcript.\"")
        with st.form("ask", clear_on_submit=True):
            q = st.text_input("Your question", placeholder="What did we decide about the model?")
            asked = st.form_submit_button("Ask", type="primary")
        if asked and q.strip():
            with st.spinner("Searching the transcript on this device..."):
                try:
                    res = rag.answer(q, ss.index, segments, model=llm)
                    ss.qa.insert(0, {"q": q.strip(), **res})
                    ss.receipt = P.external_connections()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        by_id = {s.id: s for s in segments}
        for i, item in enumerate(ss.qa):
            st.markdown(f"**Q: {item['q']}**")
            a, b = st.columns([6, 1])
            a.markdown(item["answer"] if item["found"] else f":gray[{item['answer']}]")
            with b:
                cite_button(item["start"], item["segments"], f"qa{i}")
            if item["segments"]:
                with st.expander("Lines used"):
                    for sid in item["segments"]:
                        s = by_id[sid]
                        st.markdown(f"`[{sid}]` " + (f"`{fmt_ts(s.start)}` " if s.start is not None else "") + s.text)
            st.divider()

    with tab_tx:
        st.caption("Optional: download the transcript with line numbers"
                   + (" and timestamps." if has_times(segments) else "."))
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download transcript (.txt)", transcript_txt(segments),
                           file_name="transcript.txt", mime="text/plain", use_container_width=True)
        d2.download_button("⬇ Download transcript (.md)", transcript_md(segments, notes["title"]),
                           file_name="transcript.md", mime="text/markdown", use_container_width=True)
        for s in segments:
            st.markdown(f"`[{s.id}]` " + (f"`{fmt_ts(s.start)}` " if s.start is not None else "") + s.text)
