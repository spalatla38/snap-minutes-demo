"""Entry point for the hosted copy on Streamlit Community Cloud.

On Streamlit Cloud, set "Main file path" to cloud/streamlit_app.py. Streamlit then installs
cloud/requirements.txt (small: no models), and this file runs the normal app.py, which sees
the SNAP_BACKEND_URL secret and sends all processing to the host PC (server.py).

Secrets to set in the app's settings on Streamlit Cloud (see README: "Hosted on Streamlit Cloud"):
    SNAP_BACKEND_URL  = "https://<your-static-domain>.ngrok-free.app"
    SNAP_SERVER_TOKEN = "<the token run_server.ps1 printed>"
    SNAP_ACCESS_CODE  = "<a code you give to the people who may use it>"
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so app.py's own imports (ui, backend, ...) resolve
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
