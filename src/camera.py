"""
Unified camera capture for Raspberry Pi.

Supports two sources, picked purely by config:
  - "usb"  -> standard UVC webcam via OpenCV / V4L2
  - "csi"  -> Raspberry Pi camera module via `rpicam-vid` (libcamera stack),
              piped in as MJPEG and decoded frame by frame.

Switch by editing camera.mode in config.yaml. No code changes needed
when swapping a USB webcam for a CSI ribbon camera or vice versa.
"""

import shutil
import subprocess
import numpy as np
import cv2

# Raspberry Pi OS (recent) ships the binary as `rpicam-vid`.
# Some Ubuntu builds still ship the older libcamera-apps name `libcamera-vid`.
# Same tool, same flags — just try both so this works on either OS.
_CSI_BIN_CANDIDATES = ["rpicam-vid", "libcamera-vid"]


def _find_csi_binary():
    for name in _CSI_BIN_CANDIDATES:
        if shutil.which(name):
            return name
    raise RuntimeError(
        "No CSI camera tool found (looked for: " + ", ".join(_CSI_BIN_CANDIDATES) + "). "
        "Install rpicam-apps (Raspberry Pi OS) or libcamera-apps (Ubuntu)."
    )


class Camera:
    def __init__(self, mode="usb", device="/dev/video0", width=1280, height=720,
                 fps=30, csi_camera_index=0):
        self.mode = mode
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None
        self.cap = None
        self._buf = b""

        if mode == "usb":
            self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open USB camera at {device}. "
                    f"Check `v4l2-ctl --list-devices` for the correct path."
                )

        elif mode == "csi":
            csi_bin = _find_csi_binary()
            cmd = [
                csi_bin,
                "--camera", str(csi_camera_index),
                "-t", "0",
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(fps),
                "--codec", "mjpeg",
                "--nopreview",
                "-o", "-",
            ]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)

        else:
            raise ValueError("camera.mode must be 'usb' or 'csi'")

    def read(self):
        if self.mode == "usb":
            ok, frame = self.cap.read()
            return frame if ok else None
        return self._read_mjpeg_frame()

    def _read_mjpeg_frame(self):
        # Scan the raw MJPEG stream for SOI/EOI markers and decode one JPEG frame.
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                return None
            self._buf += chunk
            start = self._buf.find(b"\xff\xd8")
            end = self._buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self._buf[start:end + 2]
                self._buf = self._buf[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                # corrupted frame boundary, keep scanning

    def release(self):
        if self.mode == "usb" and self.cap is not None:
            self.cap.release()
        elif self.mode == "csi" and self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=2)
