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

echo "Downloading open-source models (about 4 GB)..."
ollama pull qwen3:4b-instruct   # default language model (Apache 2.0)
ollama pull qwen3:1.7b          # low-memory option (Apache 2.0)
ollama pull all-minilm          # embeddings for Q&A (Apache 2.0)

echo "Downloading Whisper speech-to-text models..."
HF_HUB_OFFLINE=0 python -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('base', 'small')]; print('Whisper models ready')"

echo "Downloading speaker models (about 33 MB) for naming speakers in videos..."
# pyannote segmentation 3.0 (MIT) and 3D-Speaker ERes2Net English (Apache 2.0), packaged by
# sherpa-onnx. If this fails, videos are still processed, just without speaker names.
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p models
if [ ! -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  curl -fsSL "$REL/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" | tar -xj -C models \
    || echo "Speaker segmentation model not downloaded; videos will be processed without speaker names."
fi
if [ ! -f models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx ]; then
  curl -fsSL -o models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx \
    "$REL/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" \
    || echo "Speaker voice model not downloaded; videos will be processed without speaker names."
fi

echo
echo "Setup complete. You can now go offline and start the demo with: ./run_demo.sh"
