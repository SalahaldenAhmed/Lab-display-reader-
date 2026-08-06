"""
The GUI: a single OpenCV window showing the live camera feed on the left and
a readings panel on the right.

Deliberately still "no web server, no browser" — it is drawn with plain cv2
primitives onto one canvas and shown with cv2.imshow, so it works over
`ssh -X` exactly like roi_tool.py does, with no extra dependencies beyond
what the reader already needs (opencv + numpy).

Left  : camera frame, ROI boxes tinted by field status, small name chips.
Right : one card per field — label, big value + unit, confidence bar,
        stability bar, sparkline of recent values, status dot.

Everything scales to the frame size, and the panel degrades gracefully
(cards shrink, sparklines drop out, then rows go compact) as the number of
fields grows.
"""

import time

import cv2
import numpy as np

# ---- theme (BGR) --------------------------------------------------------
BG          = (26, 24, 22)
PANEL_BG    = (20, 19, 18)
CARD_BG     = (43, 39, 36)
CARD_EDGE   = (62, 57, 53)
TEXT        = (238, 238, 236)
TEXT_DIM    = (150, 146, 142)
TEXT_FAINT  = (105, 102, 99)
ACCENT      = (196, 152, 78)          # muted blue-steel accent

STATUS_COLORS = {
    "good":  (126, 214, 132),
    "weak":  (70, 196, 246),
    "stale": (86, 152, 246),
    "lost":  (92, 92, 236),
}
STATUS_LABEL = {"good": "LOCKED", "weak": "NOISY", "stale": "HOLD", "lost": "NO READ"}

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


# ---- small drawing helpers ---------------------------------------------
def _deg_radius(scale):
    return max(2, int(round(scale * 4)))


def _measure(s, scale, thick=1, font=F):
    """Width/height of a string, counting '°' as the little circle we draw for it.

    cv2's Hershey fonts are ASCII-only — '°C', 'µ', etc. would come out as
    garbage or nothing — so units are measured and drawn through here.
    """
    s = str(s)
    parts = s.split("°")
    w = 0
    h = 0
    for i, part in enumerate(parts):
        part = part.encode("ascii", "ignore").decode()
        if part:
            (pw, ph), _ = cv2.getTextSize(part, font, scale, thick)
            w += pw
            h = max(h, ph)
        if i < len(parts) - 1:
            w += 2 * _deg_radius(scale) + 5
    if h == 0:
        (_, h), _ = cv2.getTextSize("0", font, scale, thick)
    return w, h


def _text(img, s, org, scale=0.5, color=TEXT, thick=1, font=F):
    x, y = int(org[0]), int(org[1])
    parts = str(s).split("°")
    for i, part in enumerate(parts):
        part = part.encode("ascii", "ignore").decode()
        if part:
            cv2.putText(img, part, (x, y), font, scale, color, thick, cv2.LINE_AA)
            (pw, _), _ = cv2.getTextSize(part, font, scale, thick)
            x += pw
        if i < len(parts) - 1:
            r = _deg_radius(scale)
            (_, ch), _ = cv2.getTextSize("O", font, scale, thick)
            cv2.circle(img, (x + r + 2, y - ch + r), r, color, max(1, thick - 1), cv2.LINE_AA)
            x += 2 * r + 5


def _text_r(img, s, right_x, y, scale=0.5, color=TEXT, thick=1, font=F):
    w, _ = _measure(s, scale, thick, font)
    _text(img, s, (int(right_x - w), y), scale, color, thick, font)


def _fit_scale(s, max_w, start, font=FD, thick=2, floor=0.4):
    """Shrink the font scale until the string fits in max_w pixels."""
    scale = start
    while scale > floor:
        w, _ = _measure(s, scale, thick, font)
        if w <= max_w:
            break
        scale -= 0.05
    return scale


def _panel_rect(img, x1, y1, x2, y2, fill, edge=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    if edge is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), edge, 1, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, track=(58, 54, 50)):
    frac = 0.0 if frac is None else max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x + w, y + h), track, -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)


def _sparkline(img, x, y, w, h, values, color):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        _text(img, "collecting…", (x, y + h - 2), 0.36, TEXT_FAINT)
        return
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i * (w - 1) / (n - 1))
        py = y + h - 1 - int((v - vmin) / span * (h - 2))
        pts.append((px, py))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.circle(img, pts[-1], 2, color, -1, cv2.LINE_AA)


# ---- video overlay ------------------------------------------------------
def draw_overlay(frame, boxes, trackers, show_raw=False):
    """Draw ROI rectangles + name chips on a copy of the camera frame."""
    out = frame.copy()
    for t in trackers:
        box = boxes.get(t.name)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = STATUS_COLORS[t.status]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        chip = t.label if not show_raw else f"{t.label}  raw:{t.raw or '-'}"
        (tw, th), _ = cv2.getTextSize(chip, F, 0.45, 1)
        cy = y1 - 8 if y1 - 8 - th > 0 else y2 + th + 8
        _panel_rect(out, x1, cy - th - 5, x1 + tw + 10, cy + 5, (0, 0, 0))
        _text(out, chip, (x1 + 5, cy), 0.45, color)
    return out


# ---- the dashboard ------------------------------------------------------
class Dashboard:
    def __init__(self, window="Lab Display Reader", panel_width=430, scale=1.0):
        self.window = window
        self.panel_width = int(panel_width)
        self.scale = float(scale)
        self.t0 = time.time()
        self._fps_t = time.time()
        self._fps_n = 0
        self.fps = 0.0
        self.show_raw = False
        self.paused = False
        self._created = False

    # -- fps bookkeeping ---------------------------------------------------
    def tick(self):
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 0.5:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_t = now
            self._fps_n = 0

    # -- composition -------------------------------------------------------
    def render(self, frame, boxes, trackers, source_label=""):
        video = draw_overlay(frame, boxes, trackers, self.show_raw)
        if self.scale != 1.0:
            video = cv2.resize(video, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)

        vh, vw = video.shape[:2]
        H = max(vh, 360)
        W = vw + self.panel_width
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas[0:vh, 0:vw] = video

        px = vw
        _panel_rect(canvas, px, 0, W - 1, H - 1, PANEL_BG)
        cv2.line(canvas, (px, 0), (px, H), (58, 54, 50), 1)

        self._draw_header(canvas, px, source_label, len(trackers))
        self._draw_fields(canvas, px, H, trackers)
        self._draw_footer(canvas, px, W, H)

        if self.paused:
            _panel_rect(canvas, 12, 12, 132, 44, (0, 0, 0))
            _text(canvas, "PAUSED", (24, 36), 0.7, (70, 196, 246), 2)
        return canvas

    def _draw_header(self, c, px, source_label, n):
        pad = 16
        _text(c, "LAB DISPLAY READER", (px + pad, 30), 0.62, TEXT, 1, FD)
        cv2.line(c, (px + pad, 40), (px + pad + 60, 40), ACCENT, 2)
        elapsed = time.time() - self.t0
        sub = f"{n} field{'s' if n != 1 else ''}   {self.fps:4.1f} fps   {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        _text(c, sub, (px + pad, 58), 0.42, TEXT_DIM)
        if source_label:
            _text_r(c, source_label, c.shape[1] - pad, 30, 0.42, TEXT_FAINT)

    def _draw_fields(self, c, px, H, trackers):
        pad = 16
        top, bottom = 74, H - 36
        avail = bottom - top
        trackers = list(trackers)
        n = max(1, len(trackers))
        gap = 10

        card_h = int((avail - gap * (n - 1)) / n)
        card_h = min(card_h, 190)          # don't blow one field up to full height
        hidden = 0
        if card_h < 44:                    # more fields than the window can hold
            card_h = 44
            n_fit = max(1, int((avail + gap) // (card_h + gap)))
            hidden = len(trackers) - n_fit
            trackers = trackers[:n_fit]

        block_h = card_h * len(trackers) + gap * (len(trackers) - 1)
        y = top + max(0, (avail - block_h) // 2) if hidden == 0 else top

        x1 = px + pad
        x2 = c.shape[1] - pad
        for t in trackers:
            self._draw_card(c, x1, y, x2, y + card_h, t)
            y += card_h + gap

        if hidden > 0:
            _text(c, f"+{hidden} more field(s) - enlarge the window",
                  (x1, min(y + 14, bottom + 12)), 0.38, TEXT_FAINT)

    def _draw_card(self, c, x1, y1, x2, y2, t):
        """Card content adapts to its own height:
             >=118px  label + big value + conf + stab + sparkline
             >= 86px  label + big value + conf + stab
             >= 62px  label + big value + conf
              < 62px  single row: label left, value right, thin conf bar
        """
        color = STATUS_COLORS[t.status]
        h = y2 - y1
        w = x2 - x1
        _panel_rect(c, x1, y1, x2, y2, CARD_BG, CARD_EDGE)
        cv2.rectangle(c, (x1, y1), (x1 + 3, y2), color, -1)   # status spine
        inner = x1 + 14

        # ---- one-row compact card ----
        if h < 62:
            mid = y1 + h // 2 + 2
            _text(c, t.label.upper(), (inner, mid), 0.42, TEXT_DIM)
            s = t.text + (f" {t.unit}" if t.unit else "")
            sc = _fit_scale(s, int(w * 0.45), 0.72)
            _text_r(c, s, x2 - 26, mid + 4, sc, TEXT, 2, FD)
            cv2.circle(c, (x2 - 13, mid - 4), 4, color, -1, cv2.LINE_AA)
            _bar(c, inner, y2 - 8, w - 28, 3, t.conf, color)
            return

        # ---- header row ----
        _text(c, t.label.upper(), (inner, y1 + 20), 0.44, TEXT_DIM)
        cv2.circle(c, (x2 - 14, y1 + 15), 4, color, -1, cv2.LINE_AA)
        _text_r(c, STATUS_LABEL[t.status], x2 - 24, y1 + 19, 0.36, color)

        # ---- big value ----
        vs = _fit_scale(t.text, w - 120, 1.30 if h >= 86 else 0.95)
        vw_, vh_ = _measure(t.text, vs, 2, FD)
        vy = y1 + 26 + vh_
        _text(c, t.text, (inner, vy), vs, TEXT, 2, FD)
        if t.unit:
            _text(c, t.unit, (inner + vw_ + 10, vy), 0.5, TEXT_DIM)

        # ---- bars ----
        bar_w = int(w * 0.34)
        by = vy + 12
        _text(c, "conf", (inner, by + 8), 0.36, TEXT_FAINT)
        _bar(c, inner + 34, by + 1, bar_w, 7, t.conf, color)
        _text(c, f"{t.conf * 100:3.0f}%", (inner + 40 + bar_w, by + 8), 0.36, TEXT_DIM)

        if h >= 86:
            by += 15
            _text(c, "stab", (inner, by + 8), 0.36, TEXT_FAINT)
            _bar(c, inner + 34, by + 1, bar_w, 7, t.stability, ACCENT)
            _text(c, f"{t.stability * 100:3.0f}%", (inner + 40 + bar_w, by + 8), 0.36, TEXT_DIM)

        # ---- sparkline, only if there is real room left ----
        spark_top = by + 16
        spark_h = y2 - spark_top - 8
        if spark_h >= 20:
            _sparkline(c, inner, spark_top, w - 28, spark_h, list(t.history), color)

        if self.show_raw:
            _text_r(c, f"raw:{t.raw or '-'}", x2 - 14, vy, 0.36, TEXT_FAINT)

    def _draw_footer(self, c, px, W, H):
        _text(c, "q quit   p pause   r raw   s snapshot", (px + 16, H - 12), 0.38, TEXT_FAINT)

    # -- window ------------------------------------------------------------
    def show(self, canvas):
        if not self._created:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, canvas.shape[1], canvas.shape[0])
            self._created = True
        cv2.imshow(self.window, canvas)

    def handle_key(self, key, canvas=None):
        """Returns False when the user asked to quit."""
        if key == 255 or key == -1:
            return True
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == "q":
            return False
        if ch == "p":
            self.paused = not self.paused
        elif ch == "r":
            self.show_raw = not self.show_raw
        elif ch == "s" and canvas is not None:
            name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, canvas)
            print(f"[gui] saved {name}")
        return True

    def close(self):
        cv2.destroyAllWindows()
