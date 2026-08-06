#!/usr/bin/env bash
set -e
echo "==> Installing system packages (needs sudo)..."
sudo apt update
sudo apt install -y python3-venv python3-pip libatlas-base-dev libopenjp2-7 v4l-utils
echo "==> Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate
echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Done. Next: set camera in config.yaml, then:"
echo "  source .venv/bin/activate"
echo "  python src/roi_tool.py   (draw boxes, needs ssh -X)"
echo "  python src/main.py       (run)"
