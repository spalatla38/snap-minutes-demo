# One-time setup (run while ONLINE) in PowerShell:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Ollama is not installed. Download it from https://ollama.com/download, open it, then rerun this script."
  exit 1
}

# Prefer the Python launcher (py), but many installs (e.g. from python.org without the
# launcher, or Anaconda) only have python on PATH.
if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
elseif (Get-Command python -ErrorAction SilentlyContinue) { python -m venv .venv }
else { Write-Host "Python 3.10+ is not installed. Install it from https://www.python.org/downloads/ and rerun."; exit 1 }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Optional: CUDA libraries so Whisper can transcribe on an NVIDIA GPU (e.g. RTX 3060).
# Ollama uses the GPU on its own. If this step fails, transcription simply uses the CPU.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host "NVIDIA GPU found. Installing CUDA libraries for GPU transcription (about 1 GB)..."
  .\.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
  if ($LASTEXITCODE -ne 0) { Write-Host "GPU libraries not installed; transcription will use the CPU." }
}

Write-Host "Downloading open-source models (about 4 GB)..."
ollama pull qwen3:4b-instruct   # default language model (Apache 2.0)
ollama pull qwen3:1.7b          # low-memory option (Apache 2.0)
ollama pull all-minilm          # embeddings for Q&A (Apache 2.0)

Write-Host "Downloading Whisper speech-to-text models..."
$env:HF_HUB_OFFLINE = "0"
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('base', 'small')]; print('Whisper models ready')"

Write-Host "Downloading speaker models (about 33 MB) for naming speakers in videos..."
# pyannote segmentation 3.0 (MIT) and 3D-Speaker ERes2Net English (Apache 2.0), as packaged
# by the sherpa-onnx project. If this fails, videos are still processed, just without names.
$rel = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
New-Item -ItemType Directory -Force models | Out-Null
try {
  if (-not (Test-Path "models\sherpa-onnx-pyannote-segmentation-3-0\model.onnx")) {
    Invoke-WebRequest "$rel/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" -OutFile "models\seg.tar.bz2"
    tar -xjf "models\seg.tar.bz2" -C models
    Remove-Item "models\seg.tar.bz2"
  }
  if (-not (Test-Path "models\3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx")) {
    Invoke-WebRequest "$rel/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" -OutFile "models\3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"
  }
  Write-Host "Speaker models ready"
} catch { Write-Host "Speaker models not downloaded ($($_.Exception.Message)); videos will be processed without speaker names." }

Write-Host ""
Write-Host "Setup complete. You can now go offline and start the demo with: .\run_demo.ps1"
