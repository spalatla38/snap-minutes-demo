"""Look and feel for the SNAP Assist app: CSS and small HTML builders.

Everything here is plain HTML and CSS rendered through st.markdown, with no external
fonts, scripts or images, so the app looks the same in airplane mode. Any text that
comes from a transcript or a model is passed through html.escape() before it goes into
HTML. Animations respect the operating system's "reduce motion" setting."""
from __future__ import annotations

import html
import re

from progress import fmt_duration

CSS = """
<style>
:root { --sa-a: #6366f1; --sa-b: #06b6d4; --sa-ok: #10b981; --sa-mute: rgba(128,128,128,.55);
        --sa-card: rgba(128,128,128,.08); --sa-line: rgba(128,128,128,.22); }

/* ---------- header ---------- */
.sa-hero { padding: .4rem 0 .2rem; }
.sa-hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.1;
  background: linear-gradient(90deg, var(--sa-a), var(--sa-b), var(--sa-a)); background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: sa-pan 6s linear infinite; }
.sa-hero p { margin: .35rem 0 .6rem; opacity: .8; }
.sa-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }
.sa-badge { font-size: .78rem; padding: .18rem .6rem; border-radius: 999px; border: 1px solid var(--sa-line);
  background: var(--sa-card); }
.sa-badge::before { content: "●"; color: var(--sa-ok); margin-right: .35rem; font-size: .65rem; }

/* ---------- processing panel ---------- */
.sa-proc { position: relative; border-radius: 18px; padding: 1.3rem 1.4rem 1.1rem; margin: .6rem 0 1rem;
  background: var(--sa-card); border: 1px solid var(--sa-line); overflow: hidden; }
.sa-proc::before { content: ""; position: absolute; inset: 0; border-radius: 18px; padding: 1.5px;
  background: linear-gradient(120deg, var(--sa-a), var(--sa-b), transparent 60%, var(--sa-a));
  background-size: 300% 300%; animation: sa-pan 4s linear infinite;
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none; }
.sa-proc.done::before { background: var(--sa-ok); animation: none; }
.sa-top { display: flex; gap: 1.4rem; align-items: center; }
.sa-headline { flex: 1; min-width: 0; }
.sa-headline .sa-label { font-size: 1.15rem; font-weight: 700; }
.sa-headline .sa-detail { opacity: .75; font-size: .9rem; min-height: 1.3em; }

/* the animated picture on the left changes with the stage */
.sa-visual { width: 132px; height: 72px; flex: none; position: relative; display: flex;
  align-items: center; justify-content: center; }
.sa-visual > div { display: none; }
.sa-visual.wave .sa-wave, .sa-visual.scan .sa-scan, .sa-visual.orbit .sa-orbit,
.sa-visual.done .sa-check { display: flex; }

.sa-wave { gap: 3px; align-items: center; height: 100%; }
.sa-wave span { width: 3px; height: 70%; border-radius: 3px; transform-origin: center;
  background: linear-gradient(var(--sa-b), var(--sa-a)); animation: sa-eq 1.1s ease-in-out infinite; }

.sa-scan { width: 100%; height: 100%; flex-direction: column; justify-content: center; gap: 7px;
  position: relative; overflow: hidden; }
.sa-scan i { display: block; height: 7px; border-radius: 4px; background: var(--sa-line); }
.sa-scan i:nth-child(2) { width: 80%; } .sa-scan i:nth-child(3) { width: 92%; } .sa-scan i:nth-child(4) { width: 60%; }
.sa-scan::after { content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
  background: linear-gradient(90deg, transparent, rgba(6,182,212,.45), transparent);
  animation: sa-sweep 1.4s ease-in-out infinite; }

.sa-orbit { width: 72px; height: 72px; position: relative; align-items: center; justify-content: center; }
.sa-orbit .core { width: 22px; height: 22px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #fff, var(--sa-b) 45%, var(--sa-a));
  box-shadow: 0 0 18px var(--sa-b); animation: sa-pulse 1.6s ease-in-out infinite; }
.sa-orbit .ring { position: absolute; inset: 0; border-radius: 50%; border: 1px dashed var(--sa-line);
  animation: sa-spin 3s linear infinite; }
.sa-orbit .ring:nth-child(3) { inset: 12px; animation-duration: 2s; animation-direction: reverse; }
.sa-orbit .ring::after { content: ""; position: absolute; top: -4px; left: calc(50% - 4px); width: 8px; height: 8px;
  border-radius: 50%; background: var(--sa-a); box-shadow: 0 0 8px var(--sa-a); }
.sa-orbit .ring:nth-child(3)::after { background: var(--sa-b); box-shadow: 0 0 8px var(--sa-b); }

.sa-check { width: 56px; height: 56px; border-radius: 50%; background: var(--sa-ok); color: #fff;
  font-size: 30px; font-weight: 800; align-items: center; justify-content: center;
  animation: sa-pop .45s cubic-bezier(.2,1.6,.4,1) both; box-shadow: 0 0 22px rgba(16,185,129,.55); }

/* big numbers */
.sa-stats { display: flex; gap: 1.6rem; text-align: right; }
.sa-stats div { display: flex; flex-direction: column; }
.sa-stats b { font-size: 1.5rem; font-variant-numeric: tabular-nums; line-height: 1.1; }
.sa-stats b.big { font-size: 2.3rem; background: linear-gradient(90deg, var(--sa-a), var(--sa-b));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.sa-stats small { opacity: .65; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; }

/* the bar */
.sa-bar { height: 14px; border-radius: 999px; background: var(--sa-line); margin: 1rem 0 1.1rem;
  overflow: hidden; position: relative; }
.sa-fill { height: 100%; border-radius: 999px; position: relative;
  background: linear-gradient(90deg, var(--sa-a), var(--sa-b)); box-shadow: 0 0 14px rgba(6,182,212,.55); }
/* No width transition: the width is redrawn 5x a second, and a transition restarted that often
   lags far behind (badly so in a tab the browser is not repainting often). */
.sa-fill::after { content: ""; position: absolute; inset: 0; border-radius: 999px;
  background: repeating-linear-gradient(45deg, rgba(255,255,255,.22) 0 10px, transparent 10px 20px);
  background-size: 28px 28px; animation: sa-stripes .8s linear infinite; }
.sa-proc.done .sa-fill { background: var(--sa-ok); box-shadow: 0 0 14px rgba(16,185,129,.5); }
.sa-proc.done .sa-fill::after { display: none; }

/* step tracker */
.sa-steps { display: flex; gap: .5rem; }
.sa-step { flex: 1; display: flex; gap: .55rem; align-items: flex-start; padding: .55rem .6rem;
  border-radius: 12px; border: 1px solid transparent; transition: all .3s ease; min-width: 0; }
.sa-step.active { border-color: var(--sa-line); background: var(--sa-card); }
.sa-step .dot { width: 22px; height: 22px; flex: none; border-radius: 50%; border: 2px solid var(--sa-line);
  display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; margin-top: 1px; }
.sa-step.active .dot { border-color: var(--sa-b); border-top-color: transparent; animation: sa-spin .8s linear infinite; }
.sa-step.done .dot { border-color: var(--sa-ok); background: var(--sa-ok); color: #fff; }
.sa-step.done .dot::after { content: "✓"; }
.sa-step .txt { display: flex; flex-direction: column; min-width: 0; }
.sa-step .txt b { font-size: .88rem; }
.sa-step.pending .txt { opacity: .5; }
.sa-step .txt small { opacity: .7; font-size: .78rem; font-variant-numeric: tabular-nums;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sa-mini { height: 3px; border-radius: 3px; background: var(--sa-line); margin-top: .35rem; overflow: hidden; }
.sa-mini i { display: block; height: 100%; background: var(--sa-b); }
.sa-step.done .sa-mini i { background: var(--sa-ok); }

/* ---------- results ---------- */
.sa-card { border-radius: 14px; padding: 1rem 1.2rem; background: var(--sa-card); border: 1px solid var(--sa-line);
  margin-bottom: .8rem; animation: sa-rise .5s ease both; }
.sa-card h3 { margin: 0 0 .3rem; padding: 0; }
.sa-chip { display: inline-block; font-size: .78rem; padding: .1rem .55rem; border-radius: 999px; margin-right: .3rem;
  border: 1px solid var(--sa-line); background: var(--sa-card); white-space: nowrap; }
.sa-chip.owner { border-color: rgba(99,102,241,.5); }
.sa-chip.warn { border-color: rgba(245,158,11,.6); color: #d97706; }
.sa-done { text-decoration: line-through; opacity: .55; }
.sa-tx { font-size: .92rem; line-height: 1.55; }
.sa-tx div { padding: .15rem .4rem; border-radius: 6px; }
.sa-tx div:hover { background: var(--sa-card); }
.sa-tx code { opacity: .7; margin-right: .3rem; }
.sa-tx mark { background: rgba(6,182,212,.35); color: inherit; border-radius: 3px; padding: 0 2px; }
.sa-src { font-size: .88rem; border-left: 3px solid var(--sa-b); padding: .2rem .6rem; margin: .25rem 0; }

/* typing dots while a question is answered */
.sa-typing { display: inline-flex; gap: 5px; align-items: center; padding: .3rem 0; }
.sa-typing span { width: 8px; height: 8px; border-radius: 50%; background: var(--sa-b);
  animation: sa-bounce 1.2s ease-in-out infinite; }
.sa-typing span:nth-child(2) { animation-delay: .15s; } .sa-typing span:nth-child(3) { animation-delay: .3s; }
.sa-typing em { margin-left: .5rem; opacity: .7; font-size: .9rem; }

@keyframes sa-pan { to { background-position: 200% center; } }
@keyframes sa-eq { 0%,100% { transform: scaleY(.25); } 50% { transform: scaleY(1); } }
@keyframes sa-sweep { from { left: -40%; } to { left: 100%; } }
@keyframes sa-pulse { 0%,100% { transform: scale(.85); } 50% { transform: scale(1.15); } }
@keyframes sa-spin { to { transform: rotate(360deg); } }
@keyframes sa-pop { from { transform: scale(0); } to { transform: scale(1); } }
@keyframes sa-stripes { to { background-position: 28px 0; } }
@keyframes sa-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes sa-bounce { 0%,80%,100% { transform: translateY(0); opacity: .4; } 40% { transform: translateY(-6px); opacity: 1; } }

@media (max-width: 720px) {
  .sa-top { flex-wrap: wrap; } .sa-stats { text-align: left; } .sa-steps { flex-direction: column; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""

# Fixed pseudo-random delays so the waveform looks organic; the markup never changes
# between updates, so the browser keeps the same elements and the animation runs smoothly.
_WAVE = "".join(f'<span style="animation-delay:-{(i * 0.37) % 1.1:.2f}s;height:{40 + (i * 29) % 55}%"></span>'
                for i in range(30))


# The badges must stay true in each mode: the hosted page does not run on "this computer"
# and does not work offline, so it says what actually happens instead.
BADGES = {
    "local": ["Runs on this computer", "Works offline", "Nothing is saved", "Open-source models"],
    "lan": ["Runs on the host PC", "Stays on your network", "Nothing is saved", "Open-source models"],
    "hosted": ["Processed on the host PC, not cloud AI", "Encrypted in transit", "Nothing is saved",
               "Open-source models"],
}


def hero(mode: str = "local") -> str:
    badges = "".join(f"<span class='sa-badge'>{html.escape(b)}</span>" for b in BADGES[mode])
    return ("<div class='sa-hero'><h1>⚡ SNAP Assist</h1>"
            "<p>Meeting video, transcript or notes in. Transcript, bullet points and answers out.</p>"
            f"<div class='sa-badges'>{badges}</div></div>")


def progress_panel(snap: dict, video: bool) -> str:
    """The processing panel: animation, % done, ETA, elapsed, bar and step tracker.
    The HTML structure is identical on every update (only text, classes and widths change),
    so the browser updates it in place instead of restarting the animations."""
    cur = snap["current"]
    visual = "done" if snap["done"] else {"input": "wave" if video else "scan",
                                         "index": "scan", "notes": "orbit"}.get(cur, "scan")
    pct = snap["percent"]
    if snap["done"]:
        eta = "0 s"
    elif snap["eta_measured"]:
        eta = fmt_duration(snap["eta"])
    else:
        eta = "≈ " + fmt_duration(snap["eta"])  # still an estimate from earlier speeds
    steps = []
    for s in snap["stages"]:
        if s["state"] == "done":
            sub = f"done in {s['seconds']:.1f} s"
        elif s["state"] == "active":
            sub = f"{s['frac']:.0%}" + (f" · {s['seconds']:.0f} s" if s["seconds"] else "")
        else:
            sub = "waiting"
        steps.append(f"<div class='sa-step {s['state']}'><div class='dot'></div><div class='txt' style='flex:1'>"
                     f"<b>{html.escape(s['label'])}</b><small>{sub}</small>"
                     f"<div class='sa-mini'><i style='width:{s['frac'] * 100:.1f}%'></i></div></div></div>")
    label = "All done. Opening your results…" if snap["done"] else snap["label"] + "…"
    return (
        f"<div class='sa-proc {'done' if snap['done'] else ''}'>"
        f"<div class='sa-top'>"
        f"<div class='sa-visual {visual}'><div class='sa-wave'>{_WAVE}</div>"
        f"<div class='sa-scan'><i></i><i></i><i></i><i></i></div>"
        f"<div class='sa-orbit'><div class='core'></div><div class='ring'></div><div class='ring'></div></div>"
        f"<div class='sa-check'>✓</div></div>"
        f"<div class='sa-headline'><div class='sa-label'>{html.escape(label)}</div>"
        f"<div class='sa-detail'>{html.escape(snap['detail'] or '')}</div></div>"
        f"<div class='sa-stats'><div><b class='big'>{pct * 100:.0f}%</b><small>done</small></div>"
        f"<div><b>{eta}</b><small>remaining</small></div>"
        f"<div><b>{fmt_duration(snap['elapsed'])}</b><small>elapsed</small></div></div>"
        f"</div>"
        f"<div class='sa-bar'><div class='sa-fill' style='width:{pct * 100:.1f}%'></div></div>"
        f"<div class='sa-steps'>{''.join(steps)}</div>"
        f"</div>"
    )


def typing(text: str = "Searching the transcript on this device") -> str:
    return (f"<div class='sa-typing'><span></span><span></span><span></span>"
            f"<em>{html.escape(text)}…</em></div>")


def highlight(text: str, query: str) -> str:
    """Escape the text, then wrap case-insensitive matches of the query in <mark>."""
    safe = html.escape(text)
    if not query:
        return safe
    return re.sub("(" + re.escape(html.escape(query)) + ")", r"<mark>\1</mark>", safe, flags=re.I)


def transcript_html(lines: list[tuple[int, str, str]], query: str = "") -> str:
    """lines: (id, timestamp or '', text). Returns one escaped HTML block."""
    rows = [f"<div><code>[{i}]</code>" + (f"<code>{html.escape(ts)}</code>" if ts else "")
            + highlight(text, query) + "</div>" for i, ts, text in lines]
    return "<div class='sa-tx'>" + "".join(rows) + "</div>"


def source_lines(lines: list[tuple[int, str, str]]) -> str:
    return "".join(f"<div class='sa-src'><code>[{i}]</code> " + (f"<code>{html.escape(ts)}</code> " if ts else "")
                   + html.escape(text) + "</div>" for i, ts, text in lines)
