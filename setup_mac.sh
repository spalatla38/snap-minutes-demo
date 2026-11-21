#!/usr/bin/env bash
# One-time setup (run while ONLINE). Afterwards the demo runs fully offline.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Download it from https://ollama.com/download, open it, then rerun this script."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Downloading language models (about 3.5 GB)..."
ollama pull llama3.2:3b
ollama pull llama3.2:1b

echo "Downloading Whisper speech-to-text models..."
HF_HUB_OFFLINE=0 python -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('base', 'small')]; print('Whisper models ready')"

echo
echo "Setup complete. You can now go offline and start the demo with: ./run_demo.sh"
