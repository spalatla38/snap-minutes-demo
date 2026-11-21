# SNAP Assist: on-device meeting notes

Give SNAP Assist a **meeting video**, or a **transcript or notes file**, and it produces three
things, **entirely on your computer**:

1. **Transcript download (optional):** a button that saves the transcript with line numbers and timestamps.
2. **Main bullet points:** decisions, key points and action items, each linked to the line it came from.
3. **Ask SNAP:** a question box that answers only from the transcript, cites the lines it
   used, and replies "Not in the transcript." when the answer isn't there.

Nothing is uploaded and nothing is saved: uploads, the transcript and the Q&A index live in memory
and disappear when you close the app. Every model and library is open source.

```
meeting video ──► faster-whisper (then freed) ─┐
transcript / notes file ──► readers.py ────────┴─► numbered transcript (in memory)
                                                       │
                     ┌─────────────────────────────────┼──────────────────────────────┐
                     ▼                                 ▼                              ▼
          1. transcript download          2. bullet points                3. Q&A (rag.py)
             (.txt / .md)                    Qwen3 via Ollama               all-minilm embeddings
                                             on 127.0.0.1                   in a NumPy index in RAM
                                                       │
                     validation: drop invented line ids, attach times, flag unlinked points
```

## Inputs

| Kind | Formats |
|---|---|
| Video or audio | `.mp4` `.mov` `.mkv` `.webm` `.m4a` `.mp3` `.wav` |
| Transcripts | `.vtt` `.srt` (Teams, Zoom and Meet exports keep their times and speakers), `.json` |
| Notes | `.txt` `.md` `.docx` `.pdf` (text PDFs; scanned PDFs are planned), or paste text into the app |

## Open-source stack

| Part | Choice | Licence |
|---|---|---|
| Speech-to-text | Whisper via faster-whisper | MIT |
| Who spoke when | sherpa-onnx with pyannote segmentation 3.0 + 3D-Speaker ERes2Net | Apache 2.0 / MIT / Apache 2.0 |
| Names on screen | RapidOCR (PaddleOCR models) | Apache 2.0 |
| Language model | `qwen3:4b-instruct` (default), `qwen3:1.7b` (low memory), `phi4-mini` | Apache 2.0 / MIT |
| Embeddings | `all-minilm` (all-MiniLM-L6-v2) | Apache 2.0 |
| Model runtime | Ollama | MIT |
| Search index | NumPy, in memory | BSD |
| PDF text | pdfplumber | MIT |
| App | Streamlit | Apache 2.0 |

## Requirements

- **Laptop:** Windows 10/11 or an Apple Silicon Mac. **16 GB RAM** recommended. With 8 GB, use
  `qwen3:1.7b` and Whisper `base`. About **6 GB** of free disk for the models (plus about 1 GB of
  CUDA libraries on Windows laptops with an NVIDIA GPU).
- **Reference laptop:** Windows, 16 GB RAM, NVIDIA RTX 3060. Defaults: `qwen3:4b-instruct` and Whisper `small`.
- **Software:** Python 3.10 or later and [Ollama](https://ollama.com/download).

## Setup (once, while online)

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```
**Mac**
```bash
./setup_mac.sh
```
This installs the Python packages and downloads the models. After this, **no internet is needed**.

## GPU setup (NVIDIA)

- **Language model:** Ollama uses an NVIDIA GPU automatically. Nothing to configure.
- **Transcription:** `setup_windows.ps1` installs the CUDA libraries Whisper needs (`nvidia-cublas-cu12`,
  `nvidia-cudnn-cu12`) when it finds an NVIDIA GPU. The app then transcribes on the GPU.
- **If the GPU can't be used** (missing libraries, old driver), transcription falls back to the CPU on its
  own and the app shows a notice. The sidebar says which one is available.
- **Check:** run `nvidia-smi` in a terminal. It should list the GPU and a driver version.

## Run

1. Make sure Ollama is running (system tray on Windows, menu bar on Mac).
2. Turn on airplane mode.
3. Start the app: `.\run_demo.ps1` (Windows) or `./run_demo.sh` (Mac). It opens at http://127.0.0.1:8501.
4. Upload a video, transcript or notes file (or paste text, or click **Use sample transcript**), then **Process meeting**.
   While it works, an animated panel shows **% done, time remaining and elapsed time**, plus a step
   tracker (transcribe or read → Q&A index → bullet points). The percentage follows real progress:
   Whisper's position in the audio, embedding batches, and tokens the model has written. The ETA starts
   from speeds measured on the reference laptop, shows "≈" until the current step reports progress,
   and adapts to this laptop after the first run.
5. Use the three tabs: **Bullet points** (each with a 📍 Source popover; tick off action items),
   **Ask SNAP** (chat, with suggested questions), and **Transcript** (search and downloads).

### Who owns each action item (video)
"I'll send it Monday" only has an owner if we know who said it. For a meeting video, SNAP Assist:
1. **Tells voices apart** (speaker diarization), so every transcript line gets "Speaker 1", "Speaker 2"…
2. **Reads the names on screen** in Teams/Zoom/Meet recordings: frames are read while each voice is
   talking, and the name that shows on the big tile only while that voice talks is its name. Small
   tiles (always visible) and meeting UI text are ignored; OCR misspellings are folded together.
3. **Writes the action items** with every line labelled `Name: …`, so "I'll…", "Will do" and
   "Can you…? / Yes" go to the right person.

The **👥 Speakers** panel shows each voice, where its name came from, and a sample line. Fix or fill in
a name and click **Apply names and rewrite action items**. On a 2-minute, 4-person meeting this step
takes about 35 s (CPU) and names every speaker; the progress bar shows it as "Identifying speakers".

### Share with people on the same network (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File run_lan.ps1
```
The models still run only on this PC; others open the printed `http://<this-PC's-IP>:8501` link.
- **Access code:** a random 6-digit code is printed at each start (or set `$env:SNAP_ACCESS_CODE` first).
- **One at a time:** while a meeting is processing, others see "busy, try again".
- **Privacy receipt:** browsers on the local network are listed as viewers; anything else still counts
  as an external connection.
- **Network:** allow Python through Windows Firewall on **private** networks. Guest Wi-Fi often blocks
  device-to-device traffic; use the office network or a phone hotspot instead.
- Airplane mode turns off the local network too, so use `run_demo.ps1` for the fully offline demo.

### Hosted on Streamlit Cloud, processed on this PC
Anyone with the link and access code can use SNAP Assist from any device. The page runs on
Streamlit Community Cloud, and the processing runs on this PC:

```
browser (any device) ─HTTPS─► Streamlit Cloud (page only) ─HTTPS─► ngrok tunnel ─► server.py on this PC ─► Whisper / Ollama
```
- **What travels where:** uploads go encrypted through Streamlit Cloud and ngrok to this PC, and
  are processed only here by the open-source models. Nothing is saved: the PC forgets a meeting
  when its user starts another one or after 30 idle minutes. The hosted page says so; for
  confidential meetings use `run_demo.ps1` (fully offline).
- **Limits:** 2 people at a time (`SNAP_MAX_SESSIONS`); one meeting is processed at a time and the
  next waits in line. Uploads up to 500 MB, sent in 32 MB pieces.
- **Security:** visitors need the access code; the PC service only accepts requests carrying the
  server token, and listens only on 127.0.0.1 (the tunnel is its only way in).

**One-time setup**
1. Create a free account at ngrok.com. In the ngrok dashboard, copy your **authtoken**, and under
   **Domains** claim your free **static domain** (like `something.ngrok-free.app`).
2. In a terminal: `ngrok config add-authtoken <your authtoken>`
3. Start the PC side: `powershell -ExecutionPolicy Bypass -File run_server.ps1`. It asks for the
   static domain once, creates the server token in `.snap_server_token`, and starts the tunnel.
4. At share.streamlit.io, sign in with GitHub and click **Create app**: this repository, the branch
   you want, and main file path **`cloud/streamlit_app.py`**. Under **Advanced settings → Secrets**:
   ```toml
   SNAP_BACKEND_URL = "https://something.ngrok-free.app"
   SNAP_SERVER_TOKEN = "contents of .snap_server_token"
   SNAP_ACCESS_CODE = "a code you give to the people allowed to use it"
   ```
5. Deploy, open the app's link, and enter the access code.

**Every day:** keep this PC on and awake, with Ollama running and `run_server.ps1` started. When the
PC is off, the hosted page says it can't reach the PC.

### Command line
```bash
python pipeline.py sample/sample_notes.md --ask "Who builds the progress bar?" --ask "What is the budget?"
python pipeline.py sample/sample_meeting.vtt --save-transcript transcript.md --out notes.md
python pipeline.py my_meeting.mp4 --whisper base --llm qwen3:1.7b
```

### Tests
```bash
python tests/test_pipeline.py
python tests/test_server.py      # the PC service and the hosted page's client, over real HTTP
python tests/test_speakers.py    # speaker naming logic and how names reach the action items
```
The tests fake Ollama and Whisper, so they run anywhere without models.

## How "local only" is enforced

- **No cloud calls:** the language model and embeddings are served by Ollama on `127.0.0.1:11434`.
  Whisper runs inside the app.
- **No downloads at run time:** `HF_HUB_OFFLINE=1` and `local_files_only=True`, so a missing model fails
  loudly instead of quietly downloading.
- **Nothing saved:** uploaded files stay in memory, and the video is passed to Whisper as bytes, never
  written to disk. The Q&A index is a NumPy array in RAM.
- **Memory:** Whisper is released before the language model runs, so the two are never loaded together.
- **Local-only app:** Streamlit listens only on `127.0.0.1`, with usage statistics turned off.
- **Privacy receipt:** after each run and each question, the sidebar lists any non-local network
  connections held by the app or Ollama. The expected result is **0**.
- **One deliberate exception:** the **"Check: is this laptop offline?"** button tries one connection to
  `1.1.1.1` to prove there's no internet.

## Settings

| Environment variable | Default | Use |
|---|---|---|
| `SNAP_LLM` | `qwen3:4b-instruct` | `qwen3:1.7b` on 8 GB laptops |
| `SNAP_EMBED` | `all-minilm` | Embedding model for Q&A |
| `SNAP_WHISPER` | `small` | `base` is faster |
| `SNAP_DEVICE` | `auto` | `auto` tries the NVIDIA GPU, then falls back to the CPU; force with `cpu` or `cuda` |
| `SNAP_OLLAMA_URL` | `http://127.0.0.1:11434` | Change only if Ollama uses another port |

## Known limitations (Phase 0)

- **Speaker names need names on screen:** for video, voices are told apart in any recording, but
  names are read from Teams/Zoom/Meet name labels, which works best in **speaker view** (the person
  talking gets the big tile). In gallery view, or in a camera recording of a room, voices show as
  "Speaker 1, 2…" to name in the 👥 Speakers panel. Transcript files without speaker names leave
  "I'll …" commitments **Unassigned** rather than guessing.
- **Small models:** answers and bullets can miss things. Every item links to its source lines, and
  unlinked items are flagged, so review before sharing.
- **Q&A is one question at a time:** follow-up questions don't yet use earlier answers.
- **Scanned PDFs** have no text layer and aren't supported yet.
