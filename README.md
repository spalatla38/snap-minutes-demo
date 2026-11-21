# SNAP Minutes: on-device demo starter kit

Upload a short meeting video. SNAP Minutes **transcribes it and pulls out the key points and
action items entirely on the laptop**. Nothing is uploaded, and it works in airplane mode.
Every point links to its moment in the video: click the timestamp and the video jumps there.

```
video.mp4 ──► faster-whisper (speech-to-text, on CPU/GPU) ──► timestamped transcript
                                                                      │
                     Llama 3.2 3B via Ollama on 127.0.0.1 ◄───────────┘
                                      │  (JSON: key points, action items, supporting line ids)
                                      ▼
            validation: drop invented line ids, attach timestamps, flag unverified points
                                      ▼
            Streamlit app on 127.0.0.1 ──► notes + clickable timestamps + privacy receipt
```

## What's in the kit

| File | Purpose |
|---|---|
| `app.py` | The demo app (Streamlit, local only) |
| `pipeline.py` | Transcription, key-point extraction, validation, privacy receipt. Also a command-line tool |
| `setup_mac.sh` / `setup_windows.ps1` | One-time setup: installs packages and downloads all models |
| `run_demo.sh` / `run_demo.ps1` | Starts the app |
| `sample/sample_meeting_script.md` | A made-up 2-minute meeting to record as your demo video, with the expected output |
| `sample/sample_transcript.json` | The same meeting as a transcript, for rehearsing without a video |
| `tests/test_pipeline.py` | Offline tests (models are mocked) |
| `DEMO_CHECKLIST.md` | Step-by-step plan for demo day |
| `.streamlit/config.toml` | Turns off Streamlit usage statistics and binds the app to 127.0.0.1 |

## Requirements

- **Laptop:** Apple Silicon Mac or a recent Windows laptop with **16 GB RAM** (8 GB works with `llama3.2:1b` and Whisper `base`). About **10 GB** of free disk space.
- **Software:** Python 3.10 or later and [Ollama](https://ollama.com/download).

## Setup (once, while online)

**Mac**
```bash
./setup_mac.sh
```
**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```
This installs the Python packages and downloads Llama 3.2 3B and 1B plus Whisper `base` and `small`.
After this, **the internet is never needed again**.

## Run the demo

1. Make sure the Ollama app is running (menu bar on Mac, system tray on Windows).
2. Turn on airplane mode.
3. Start the app: `./run_demo.sh` (Mac) or `.\run_demo.ps1` (Windows). It opens at http://127.0.0.1:8501.
4. In the sidebar, click **Warm up model**, then **Check: is this laptop offline?**
5. Upload your video and click **Transcribe and summarize**.

To rehearse without a video, click **Use sample transcript**.

### Command line (no UI)
```bash
source .venv/bin/activate
python pipeline.py my_meeting.mp4 --out notes.md
python pipeline.py --transcript sample/sample_transcript.json      # skip transcription
python pipeline.py my_meeting.mp4 --whisper base --llm llama3.2:1b # faster on slow laptops
```

### Tests
```bash
python tests/test_pipeline.py
```

## How privacy is enforced

- **Models run locally:** Whisper runs in-process, and the language model is served by Ollama on `127.0.0.1:11434`.
- **No model downloads at run time:** `HF_HUB_OFFLINE=1` and `local_files_only=True`, so a missing model fails loudly instead of quietly downloading.
- **Local-only web app:** Streamlit listens only on `127.0.0.1`, with usage statistics turned off.
- **Privacy receipt:** after each run, the app lists any non-local network connections held by the app or by Ollama. The expected result is **0**.
- **Command-line proof for the audience:** `lsof -i -P | grep -iE "ollama|python"` (Mac) or `netstat -ano | findstr 11434` (Windows) should show only `127.0.0.1`.
- **One exception:** the sidebar's **"Check: is this laptop offline?"** button deliberately tries a single connection to `1.1.1.1` to prove there's no internet. In airplane mode it fails immediately.

## Settings

| Environment variable | Default | Use |
|---|---|---|
| `SNAP_LLM` | `llama3.2:3b` | Use `llama3.2:1b` on slower laptops |
| `SNAP_WHISPER` | `small` | `base` is faster; `medium` is more accurate but slow |
| `SNAP_DEVICE` | `cpu` | Set to `cuda` on an NVIDIA laptop with CUDA 12 and cuDNN 9 installed |
| `SNAP_OLLAMA_URL` | `http://127.0.0.1:11434` | Change only if Ollama runs on another port |

## Known limitations

- **No speaker labels:** Whisper doesn't say who is talking, so owners come from names spoken aloud. The sample script ends with a recap that names every owner for this reason.
- **Small model:** a 3B model can miss implied tasks or misread dates. Points it can't tie to the transcript are marked **"not linked to the transcript"**, so always review before sharing.
- **Speed:** Whisper `small` on a CPU takes roughly as long as a fraction of the video itself, depending on hardware. Time your own laptop with `DEMO_CHECKLIST.md`.
- **Scope:** this is a demo, not production software. It has no login, encryption at rest, or multi-user support.
