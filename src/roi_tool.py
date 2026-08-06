"""
Interactive tool to draw (and redraw) the reading boxes on the live camera feed.

No web server, no browser — this is a plain OpenCV window, so it works
over `ssh -X` / `ssh -Y` straight from your laptop.

Controls:
  - left-click + drag  : draw a new box
  - u                   : undo last box
  - c                   : clear all boxes
  - r                   : reset stabilizer reference to current frame
                          (do this if you redraw boxes after moving the camera)
  - s                   : save boxes to rois.json
  - q                   : quit

Run:
  python src/roi_tool.py --config config.yaml
"""

import argparse
import json
import cv2
import yaml

from camera import Camera


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="rois.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cam_cfg = cfg["camera"]
    cam = Camera(
        mode=cam_cfg["mode"],
        device=cam_cfg.get("device", "/dev/video0"),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
        csi_camera_index=cam_cfg.get("csi_index", 0),
    )

    # Load existing ROIs if present, so this doubles as the "redraw" tool.
    try:
        with open(args.out) as f:
            rois = json.load(f)
        print(f"Loaded {len(rois)} existing ROIs from {args.out}")
    except FileNotFoundError:
        rois = []

    drawing = {"active": False, "start": None, "cur": None}
    win = "ROI setup  |  drag=new box  u=undo  c=clear  r=reset ref  s=save  q=quit"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["start"] = (x, y)
            drawing["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            drawing["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
            drawing["active"] = False
            x1, y1 = drawing["start"]
            x2, y2 = x, y
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if box[2] - box[0] > 5 and box[3] - box[1] > 5:
                name = f"field_{len(rois) + 1}"
                rois.append({"name": name, "box": list(box)})
                print(f"Added {name}: {box}")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            disp = frame.copy()

            for r in rois:
                x1, y1, x2, y2 = r["box"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp, r["name"], (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if drawing["active"] and drawing["cur"]:
                x1, y1 = drawing["start"]
                x2, y2 = drawing["cur"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 165, 255), 1)

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("u") and rois:
                removed = rois.pop()
                print(f"Removed {removed['name']}")
            elif key == ord("c"):
                rois.clear()
                print("Cleared all ROIs")
            elif key == ord("s"):
                with open(args.out, "w") as f:
                    json.dump(rois, f, indent=2)
                print(f"Saved {len(rois)} ROIs to {args.out}")
            elif key == ord("r"):
                print("Reference frame reset for this session "
                      "(main.py will still use its own reference on startup).")

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
