"""
Camera-shake compensation via ORB feature matching + RANSAC.

Instead of re-detecting the display every frame, we lock the ROIs once
(via roi_tool.py) against a reference frame, then track how much the
scene has shifted/rotated relative to that reference on every new frame.
The estimated transform is used to warp the ROI boxes so they stay
locked onto the display even if the camera / tripod gets bumped.

- estimateAffinePartial2D + RANSAC for a robust rigid transform
- exponential smoothing (alpha) so small frame-to-frame jitter doesn't
  make the boxes visibly shake
- last-good-hold: if a frame has too few matches (motion blur, glare,
  someone's hand in frame), keep using the last good transform instead
  of snapping to a bad one
"""

import cv2
import numpy as np


class Stabilizer:
    def __init__(self, alpha=0.35, min_matches=15):
        self.orb = cv2.ORB_create(nfeatures=800)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.alpha = alpha
        self.min_matches = min_matches

        self.ref_kp = None
        self.ref_des = None
        self.smoothed_H = np.eye(3, dtype=np.float32)
        self.last_good_H = np.eye(3, dtype=np.float32)

    def set_reference(self, frame):
        """Call this once, right after ROIs are drawn against `frame`."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)
        self.smoothed_H = np.eye(3, dtype=np.float32)
        self.last_good_H = np.eye(3, dtype=np.float32)

    def update(self, frame):
        """Returns a 3x3 transform mapping reference-frame coords -> current-frame coords."""
        if self.ref_des is None:
            self.set_reference(frame)
            return self.smoothed_H.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(des) < self.min_matches:
            return self.last_good_H.copy()

        matches = self.bf.match(self.ref_des, des)
        if len(matches) < self.min_matches:
            return self.last_good_H.copy()

        matches = sorted(matches, key=lambda m: m.distance)[:200]
        ref_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        cur_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # We draw boxes in the REFERENCE frame and want to place them in the
        # CURRENT frame, i.e. reference -> current. estimateAffinePartial2D(from, to)
        # returns M with  to ~= M * from, so from=reference, to=current.
        H, mask = cv2.estimateAffinePartial2D(ref_pts, cur_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
        if H is None:
            return self.last_good_H.copy()

        # A transform built on almost no inliers is unreliable — hold last good.
        if mask is not None and int(mask.sum()) < self.min_matches:
            return self.last_good_H.copy()

        H3 = np.eye(3, dtype=np.float32)
        H3[:2, :] = H

        self.smoothed_H = self.alpha * H3 + (1 - self.alpha) * self.smoothed_H
        self.last_good_H = self.smoothed_H.copy()
        return self.smoothed_H.copy()

    @staticmethod
    def warp_point(pt, H):
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        wp = H @ p
        return (wp[0] / wp[2], wp[1] / wp[2])

    def warp_box(self, box, H):
        """box = (x1, y1, x2, y2) in reference-frame coords -> warped box in current frame."""
        x1, y1, x2, y2 = box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        warped = [self.warp_point(p, H) for p in corners]
        xs = [p[0] for p in warped]
        ys = [p[1] for p in warped]
        return (min(xs), min(ys), max(xs), max(ys))
