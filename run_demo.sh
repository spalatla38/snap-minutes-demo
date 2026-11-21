#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
# 127.0.0.1: only reachable from this computer, not the network.
python -m streamlit run app.py --server.address 127.0.0.1
