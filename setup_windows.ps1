# One-time setup (run while ONLINE) in PowerShell:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  Write-Host "Ollama is not installed. Download it from https://ollama.com/download, open it, then rerun this script."
  exit 1
}

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Downloading language models (about 3.5 GB)..."
ollama pull llama3.2:3b
ollama pull llama3.2:1b

Write-Host "Downloading Whisper speech-to-text models..."
$env:HF_HUB_OFFLINE = "0"
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; [WhisperModel(m, device='cpu', compute_type='int8') for m in ('base', 'small')]; print('Whisper models ready')"

Write-Host ""
Write-Host "Setup complete. You can now go offline and start the demo with: .\run_demo.ps1"
