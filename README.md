# lab-display-reader

Standalone, single-Pi program: reads numeric digital displays via PaddleOCR
(onnx, CPU-only), stays locked onto the display through camera shake via
ORB+RANSAC tracking, and lets you draw / redraw the reading boxes live over SSH.

No web server, no MQTT, no multi-device stuff — just SSH in and run it.
The OCR model is already included in this repo (`models/`), so there is
nothing extra to download for the reading itself.

## Quick start (Raspberry Pi 5, Raspberry Pi OS Bookworm or later)

```bash
git clone https://github.com/AnasAlanqar/lab-display-reader.git
cd lab-display-reader
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs the system libraries, creates a Python virtual environment,
and installs the Python dependencies. The Pi needs internet the first time.
After it finishes, set the camera port and run.

## Set the camera port

Edit `config.yaml` — the only setting you change to switch USB <-> CSI camera:

```yaml
camera:
  mode: usb          # or: csi
  device: /dev/video0 # only used for usb — find yours with: v4l2-ctl --list-devices
  csi_index: 0        # only used for csi
```

If you use the CSI ribbon camera, also run once: `sudo apt install -y rpicam-apps`

## Draw the reading boxes

Opens a live OpenCV window, so SSH in with X11 forwarding:

```bash
ssh -X ubuntu@anaspi     # -Y instead of -X on macOS if -X is slow
cd lab-display-reader
source .venv/bin/activate
python src/roi_tool.py
```

(You need an X server on your own machine: XQuartz on macOS, VcXsrv on Windows,
built-in on Linux.)

- **left-click + drag** — draw a box   - **u** — undo   - **c** — clear
- **s** — save to rois.json   - **q** — quit

Re-run any time to redraw; it loads your existing rois.json so you fix just the
box that's off.

## Run it

```bash
source .venv/bin/activate
python src/main.py                 # headless, prints readings
python src/main.py --show          # + live window (needs ssh -X)
```

Output per frame: `{'field_1': {'raw': '23.4', 'value': 23.4, 'conf': 0.981}}`
Ctrl+C to stop.

## Notes

- Small camera bumps are absorbed automatically by the ORB+RANSAC tracker.
- If a big bump drifts the boxes off, just re-run roi_tool.py and redraw.
- The box tool needs a GUI, so `opencv-python` (not headless) is required —
  already pinned in requirements.txt.
