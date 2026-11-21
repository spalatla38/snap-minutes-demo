# Share SNAP Assist with people on the SAME Wi-Fi / office network.
#   powershell -ExecutionPolicy Bypass -File run_lan.ps1
#
# - The models still run only on this PC. Other people's browsers send their uploads here,
#   over your local network, and nothing goes to the internet.
# - Everyone needs the access code printed below. To choose your own, set it first:
#       $env:SNAP_ACCESS_CODE = "your-code"; .\run_lan.ps1
# - One meeting is processed at a time; others see "busy, try again".
# - The first time, Windows asks whether Python may accept connections: allow it on
#   PRIVATE networks only. If the Wi-Fi is marked "Public", Windows blocks other devices.
# - For the fully offline demo (airplane mode), use run_demo.ps1 instead: airplane mode
#   also turns off the local network.
Set-Location $PSScriptRoot

if (-not $env:SNAP_ACCESS_CODE) {
  # Random 6-digit code from the OS's secure random generator, new for every start.
  $bytes = New-Object byte[] 4
  [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $env:SNAP_ACCESS_CODE = "{0:D6}" -f ([BitConverter]::ToUInt32($bytes, 0) % 1000000)
}
$env:SNAP_LAN = "1"
$port = 8501

$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
  Select-Object -ExpandProperty IPAddress

Write-Host ""
Write-Host "SNAP Assist is shared on your local network." -ForegroundColor Cyan
foreach ($ip in $ips) { Write-Host "  Open on another device:  http://${ip}:$port" }
Write-Host "  On this PC:               http://127.0.0.1:$port"
Write-Host "  Access code:              $env:SNAP_ACCESS_CODE" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop sharing."
Write-Host ""

.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port $port
