"""
Main reading loop (Multi-Camera Supported).

Run this after rois.json has been created with roi_tool.py.

  python src/main.py                       # GUI dashboard
  python src/main.py --headless           # no window, prints readings
  python src/main.py --log readings.csv   # append every accepted reading to CSV

Keys in the GUI window:
  q  quit        p  pause        r  toggle raw OCR text        s  save snapshot PNG

Ctrl+C also stops it cleanly.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import yaml

from camera import Camera
from stabilizer import Stabilizer
from ocr_engine import PaddleOCRRecognizer
from cleaners import clean_smart
from voter import TrackerBank
from dashboard import Dashboard


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def clamp_box(box, w, h):
    """Clamp into the frame AND keep x2>=x1, y2>=y1 so an off-screen box yields
    an empty crop, never a negative-index (wrong) slice."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = min(max(0, x1), w)
    y1 = min(max(0, y1), h)
    x2 = min(max(x1, x2), w)
    y2 = min(max(y1, y2), h)
    return x1, y1, x2, y2


def gui_available():
    if os.name == "nt" or sys.platform == "darwin":
        return True
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    return hasattr(cv2, "imshow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rois", default="rois.json")
    parser.add_argument("--headless", action="store_true",
                        help="no window at all — just print readings")
    parser.add_argument("--show", action="store_true",
                        help=argparse.SUPPRESS)  # kept for backwards compatibility
    parser.add_argument("--log", default=None,
                        help="append accepted readings to this CSV file")
    parser.add_argument("--print-every", type=float, default=1.0,
                        help="seconds between console prints (0 = every frame)")
    parser.add_argument("--scale", type=float, default=None,
                        help="scale the video pane in the GUI (e.g. 0.75 for slow SSH)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ocr_cfg = cfg["ocr"]
    stab_cfg = cfg.get("stabilizer", {})
    disp_cfg = cfg.get("display", {}) or {}
    field_meta = cfg.get("fields", {}) or {}

    with open(args.rois) as f:
        rois = json.load(f)
    if not rois:
        print("No ROIs defined yet. Run roi_tool.py first.")
        return

    use_gui = not args.headless
    if use_gui and not gui_available():
        print("[gui] No display found (no $DISPLAY). Falling back to headless.\n"
              "      Reconnect with `ssh -X` (or `ssh -Y`) to get the dashboard.")
        use_gui = False

    # --------------------------------------------------------------------------
    # Initialize Multi-Camera or Single-Camera Fallback
    # --------------------------------------------------------------------------
    cameras = {}
    stabilizers = {}
    
    if "cameras" in cfg:
        camera_configs = cfg["cameras"]
    else:
        # Fallback to single camera block if 'cameras' dictionary is missing
        camera_configs = {"cam_front": cfg["camera"]}

    for cam_name, cam_cfg in camera_configs.items():
        print(f"Initializing camera [{cam_name}] on device {cam_cfg.get('device')}...")
        cam = Camera(
            mode=cam_cfg.get("mode", "usb"),
            device=cam_cfg.get("device", "/dev/video0"),
            width=cam_cfg.get("width", 1280),
            height=cam_cfg.get("height", 720),
            fps=cam_cfg.get("fps", 30),
            csi_camera_index=cam_cfg.get("csi_index", 0),
        )
        cameras[cam_name] = cam
        stabilizers[cam_name] = Stabilizer(alpha=stab_cfg.get("alpha", 0.35))

    ocr = PaddleOCRRecognizer(ocr_cfg["model_path"], ocr_cfg["dict_path"])

    bank = TrackerBank(
        rois,
        field_meta=field_meta,
        vote_window=disp_cfg.get("vote_window", 15),
        history=disp_cfg.get("history", 120),
        min_conf=disp_cfg.get("min_conf", 0.40),
    )

    dash = None
    if use_gui:
        dash = Dashboard(
            panel_width=disp_cfg.get("panel_width", 430),
            scale=args.scale if args.scale is not None else disp_cfg.get("scale", 1.0),
        )

    log_writer = log_file = None
    if args.log:
        new = not os.path.exists(args.log)
        log_file = open(args.log, "a", newline="")
        log_writer = csv.writer(log_file)
        if new:
            log_writer.writerow(["timestamp", "field", "value", "raw", "conf", "stability", "status"])

    # Establish stabilization reference frames for each camera
    active_frames = {}
    for cam_name, cam in list(cameras.items()):
        first_frame = cam.read()
        if first_frame is None:
            print(f"Warning: Failed to read initial frame from camera [{cam_name}].")
        else:
            stabilizers[cam_name].set_reference(first_frame)
            active_frames[cam_name] = first_frame

    if not active_frames:
        print("Error: Could not read frames from any configured camera. Check camera devices in config.yaml.")
        for cam in cameras.values():
            cam.release()
        return

    print(f"Running with {len(rois)} ROI(s) across {len(cameras)} camera(s). "
          f"{'GUI window open — press q to quit.' if use_gui else 'Headless — Ctrl+C to stop.'}")

    last_print = 0.0
    canvas = None

    try:
        while True:
            # Handle pause state in GUI
            if dash is not None and dash.paused:
                if canvas is not None:
                    dash.show(canvas)
                if not dash.handle_key(cv2.waitKey(30) & 0xFF, canvas):
                    break
                continue

            frames = {}
            all_boxes = {}
            cam_order = list(cameras.keys())

            # Read and process frames from all active cameras
            for cam_name, cam in cameras.items():
                frame = cam.read()
                if frame is None:
                    continue

                frames[cam_name] = frame
                stab = stabilizers[cam_name]
                H = stab.update(frame)
                fh, fw = frame.shape[:2]

                # Compute horizontal offset for GUI display when cameras are combined side-by-side
                cam_idx = cam_order.index(cam_name) if cam_name in cam_order else 0
                x_offset = cam_idx * fw

                # Filter ROIs assigned to this camera (checks ROI JSON first, then config.yaml fields)
                cam_rois = []
                for r in rois:
                    roi_cam = r.get("camera") or field_meta.get(r["name"], {}).get("camera", "cam_front")
                    if roi_cam == cam_name:
                        cam_rois.append(r)

                for r in cam_rois:
                    x1, y1, x2, y2 = clamp_box(stab.warp_box(r["box"], H), fw, fh)
                    
                    crop = frame[y1:y2, x1:x2]

                    # Offset bounding box coordinates for multi-camera concatenation view
                    all_boxes[r["name"]] = (x1 + x_offset, y1, x2 + x_offset, y2)

                    if crop.size == 0:
                        continue
                        
                    text, conf = ocr.recognize(crop)
                    value = clean_smart(text)
                    bank.update(r["name"], text, value, conf)

            if not frames:
                continue

            # Output logs/console readings
            now = time.time()
            if args.print_every <= 0 or now - last_print >= args.print_every:
                print(bank.snapshot())
                last_print = now

            if log_writer:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for t in bank:
                    log_writer.writerow([stamp, t.name, t.value, t.raw,
                                         round(t.conf, 3), round(t.stability, 3), t.status])
                log_file.flush()

            # Render Multi-Camera GUI Dashboard
            if dash is not None:
                dash.tick()
                
                # Combine frames horizontally into a single composite frame for display
                frame_list = list(frames.values())
                if len(frame_list) > 1:
                    ref_h = frame_list[0].shape[0]
                    resized_frames = []
                    for f in frame_list:
                        h, w = f.shape[:2]
                        if h != ref_h:
                            new_w = int(w * (ref_h / h))
                            resized_frames.append(cv2.resize(f, (new_w, ref_h)))
                        else:
                            resized_frames.append(f)
                    display_frame = cv2.hconcat(resized_frames)
                else:
                    display_frame = frame_list[0]

                source_label = f"MULTI-CAM ({len(frames)} active)"
                canvas = dash.render(display_frame, all_boxes, bank, source_label=source_label)
                dash.show(canvas)
                if not dash.handle_key(cv2.waitKey(1) & 0xFF, canvas):
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for cam in cameras.values():
            cam.release()
        if log_file:
            log_file.close()
        if dash is not None:
            dash.close()


if __name__ == "__main__":
    main()
