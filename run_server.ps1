# Serve the hosted SNAP Assist page (Streamlit Cloud) from THIS PC.
#   powershell -ExecutionPolicy Bypass -File run_server.ps1
#
# Starts two things and keeps them running until you press Ctrl+C:
#   1. server.py on 127.0.0.1:8600: the models (Whisper, Ollama) and the processing.
#   2. an ngrok tunnel from your free static ngrok domain to that port, so the hosted page
#      can reach this PC over HTTPS. Nothing else on your network can reach server.py.
#
# One-time setup (see README: "Hosted on Streamlit Cloud"):
#   - ngrok installed and signed in:   ngrok config add-authtoken <your ngrok authtoken>
#   - your static domain saved:        it is asked for the first time and kept in .snap_ngrok_domain
# The server token is created once and kept in .snap_server_token (both files are gitignored);
# paste it into the Streamlit app's secrets as SNAP_SERVER_TOKEN.
# Keep this PC awake and Ollama running while people use the hosted page.
Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
  Write-Host "ngrok is not installed. Install it with:  winget install --id Ngrok.Ngrok -e"; exit 1
}

# --- server token: 64 hex characters from the OS's secure random generator, created once
$tokenFile = Join-Path $PSScriptRoot ".snap_server_token"
if (-not (Test-Path $tokenFile)) {
  $bytes = New-Object byte[] 32
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  -join ($bytes | ForEach-Object { $_.ToString("x2") }) | Set-Content -Path $tokenFile -NoNewline -Encoding ascii
  Write-Host "Created a new server token in .snap_server_token. Paste it into the Streamlit secrets."
}
$env:SNAP_SERVER_TOKEN = (Get-Content $tokenFile -Raw).Trim()

# --- the static ngrok domain (from the ngrok dashboard: Domains), asked once
$domainFile = Join-Path $PSScriptRoot ".snap_ngrok_domain"
if ($env:SNAP_NGROK_DOMAIN) { $domain = $env:SNAP_NGROK_DOMAIN }
elseif (Test-Path $domainFile) { $domain = (Get-Content $domainFile -Raw).Trim() }
else {
  $domain = (Read-Host "Your ngrok static domain (e.g. something.ngrok-free.app)").Trim()
}
$domain = $domain -replace "^https?://", "" -replace "/.*$", ""
if (-not $domain) { Write-Host "No ngrok domain given."; exit 1 }
Set-Content -Path $domainFile -Value $domain -NoNewline -Encoding ascii

$port = if ($env:SNAP_SERVER_PORT) { $env:SNAP_SERVER_PORT } else { "8600" }
$env:SNAP_SERVER_PORT = $port

Write-Host ""
Write-Host "SNAP Assist PC service" -ForegroundColor Cyan
Write-Host "  Tunnel URL (secret SNAP_BACKEND_URL):  https://$domain"
Write-Host "  Server token (secret SNAP_SERVER_TOKEN): in .snap_server_token"
Write-Host "  Sessions at once: $(if ($env:SNAP_MAX_SESSIONS) { $env:SNAP_MAX_SESSIONS } else { 2 })"
Write-Host "Press Ctrl+C to stop. The hosted page shows 'can't reach the PC' while this is stopped."
Write-Host ""

$tunnel = Start-Process ngrok -ArgumentList @("http", "--url=$domain", "--log=stdout", "--log-level=warn", $port) `
  -PassThru -WindowStyle Hidden
try {
  .\.venv\Scripts\python.exe server.py
}
finally {
  if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
}
