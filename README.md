# Lab Display Reader 📸📊

A standalone, multi-camera OCR digit reader system designed for Raspberry Pi and Linux workstations. 

It reads numeric digital displays in real time via **PaddleOCR (ONNX Runtime, CPU-optimized)**, handles minor camera vibration/shake via **ORB + RANSAC feature stabilization**, and presents a live multi-feed analytical dashboard over SSH.

---

## 🌟 Key Features

* **Multi-Camera Support:** Simultaneously capture, process, and analyze readings from multiple USB or CSI camera feeds in real time.
* **Camera Stabilization:** Automatically stays locked onto ROI display screens during physical bumps using ORB feature detection and RANSAC homography.
* **Live Multi-Feed Dashboard:** Stacks active camera streams side-by-side alongside live OCR sparklines, confidence scoring, and majority voting logic.
* **Lightweight & Self-Contained:** Runs entirely CPU-bound using pre-packaged ONNX models without needing external web servers, MQTT, or heavy dependencies.
* **Flexible ROI Configuration:** Draw and save Regions of Interest (ROIs) for each camera independently.
* **Data Logging:** Optionally export clean, confidence-scored reading values straight to a `.csv` log file.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository & Run Setup

```bash
git clone [https://github.com/SalahaldenAhmed/Lab-display-reader-.git](https://github.com/SalahaldenAhmed/Lab-display-reader-.git)
cd Lab-display-reader-
chmod +x setup.sh
./setup.sh
```
setup.sh installs required system packages, builds the Python virtual environment (.venv), and installs all core dependencies.
2. Configure Your Cameras (config.yaml)
Edit config.yaml to set up your individual cameras, model choices, and data fields:
---
# Default single camera (Used by roi_tool.py)
```bash
camera:
  mode: "usb"
  device: 2                      # Default USB index (/dev/video2)
  width: 1280
  height: 720

# Multi-camera configuration (Used during runtime in main.py)
cameras:
  cam_front:
    mode: "usb"
    device: 2                    # Camera 1 (/dev/video2)
    width: 1280
    height: 720
  cam_side:
    mode: "usb"
    device: 4                    # Camera 2 (/dev/video4)
    width: 1280
    height: 720

# OCR engine configuration
ocr:
  model_path: "models/en_PP-OCRv5_mobile_rec.onnx"
  dict_path: "models/ppocrv5_dict.txt"

# Field metadata & UI mapping
fields:
  temp_a:
    camera: "cam_front"
    label: "Temperature A"
    unit: "°C"
  pressure_b:
    camera: "cam_side"
    label: "Display B"
    unit: ""

```
🎯 Step-by-Step ROI Calibration
Because roi_tool.py calibrates one camera stream at a time, follow this workflow to configure ROIs for multiple cameras:
1. Calibrate Camera 1 (cam_front)
1\ Set camera.device: ? in config.yaml.
2\ Run the ROI tool over SSH:
```bash
source .venv/bin/activate
python src/roi_tool.py
```
3\ Click + drag to draw a box around the first screen \rightarrow Press s to save \rightarrow Press q to exit.

4\ Rename the output file:

mv rois.json rois_front.json

2. Calibrate Camera 2 (cam_side)
1\ Change camera.device: ? in config.yaml.
2\ Run the ROI tool again:
```bash
python src/roi_tool.py
```

3. Merge ROIs into rois.json
Combine both calibration files into your primary rois.json file:


Running the Reader
Activate your virtual environment and launch main.py:
```bash
source .venv/bin/activate
```
# 1. Launch with GUI Dashboard (Requires SSH X11 forwarding: ssh -X user@host)
```bash
python src/main.py
```
# 2. Run in Headless Mode (Console output only)
```bash
python src/main.py --headless
```
# 3. Save readings directly to CSV
```bash
python src/main.py --log readings.csv
```

Dashboard GUI Hotkeys
q : Quit application cleanly.
p : Pause / Unpause video stream.
r : Toggle raw OCR string vs. cleaned numeric value.
s : Save a frame snapshot PNG to disk.

