Set-Location $PSScriptRoot
# 127.0.0.1: only reachable from this computer, not the network.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
