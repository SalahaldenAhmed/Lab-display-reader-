"""
Temporal smoothing of per-field readings.

Raw frame-by-frame OCR on a live camera flickers: one frame reads 23.4,
the next 234, the next nothing at all (glare, refresh flicker of a 7-seg
display, motion blur). Showing that straight on a GUI is unreadable and
untrustworthy.

FieldTracker keeps a short rolling window per field and reports the value
that WON the majority vote in that window, plus:

  - stability : how much of the window agreed with the winner (0..1)
  - conf      : mean OCR confidence of the frames that agreed
  - history   : longer deque of accepted values, for the sparkline
  - age       : how many frames since the display value last changed

If the current frame produces nothing usable, the last accepted value is
held (with a rising `stale` counter) instead of blanking the GUI.
"""

from collections import Counter, deque


class FieldTracker:
    def __init__(self, name, label=None, unit="", decimals=None,
                 vote_window=15, history=120, min_conf=0.40, stale_frames=45):
        self.name = name
        self.label = label or name.replace("_", " ").title()
        self.unit = unit or ""
        self.decimals = decimals
        self.min_conf = min_conf
        self.stale_frames = stale_frames

        self.window = deque(maxlen=max(1, int(vote_window)))
        self.history = deque(maxlen=max(2, int(history)))

        self.value = None          # accepted (voted) numeric value
        self.text = "--"           # what the GUI prints
        self.raw = ""              # last raw OCR string, for debugging
        self.conf = 0.0
        self.stability = 0.0
        self.stale = 0             # frames since we last accepted a fresh value
        self.age = 0               # frames the displayed value has been unchanged

    # -- formatting -------------------------------------------------------
    def format(self, value):
        if value is None:
            return "--"
        if self.decimals is not None:
            return f"{value:.{self.decimals}f}"
        # keep the display tidy without inventing precision
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    # -- main update ------------------------------------------------------
    def update(self, raw, value, conf):
        """Feed one frame's OCR result. Returns self (for chaining/reading)."""
        self.raw = raw or ""

        usable = value is not None and conf >= self.min_conf
        # A vote is cast per frame either way — a bad frame votes "None" so a
        # display that genuinely went blank eventually wins the vote too.
        self.window.append((self.format(value) if usable else None,
                            value if usable else None,
                            conf if usable else 0.0))

        keys = [w[0] for w in self.window]
        counts = Counter(k for k in keys if k is not None)

        if counts:
            winner, n = counts.most_common(1)[0]
            self.stability = n / len(self.window)
            agreeing = [w for w in self.window if w[0] == winner]
            self.conf = sum(w[2] for w in agreeing) / len(agreeing)
            new_value = agreeing[-1][1]

            changed = (self.text != winner)
            self.text = winner
            self.value = new_value
            self.history.append(new_value)
            self.stale = 0
            self.age = 0 if changed else self.age + 1
        else:
            # nothing readable anywhere in the window
            self.stability = 0.0
            self.conf = 0.0
            self.stale += 1
            self.age += 1
            if self.stale > self.stale_frames:
                self.value = None
                self.text = "--"

        return self

    # -- status for the GUI -----------------------------------------------
    @property
    def status(self):
        """'good' | 'weak' | 'stale' | 'lost' — drives the colour on the panel."""
        if self.value is None:
            return "lost"
        if self.stale > 0:
            return "stale"
        if self.stability < 0.6 or self.conf < 0.7:
            return "weak"
        return "good"

    def as_dict(self):
        return {
            "raw": self.raw,
            "value": self.value,
            "text": self.text,
            "unit": self.unit,
            "conf": round(self.conf, 3),
            "stability": round(self.stability, 3),
            "status": self.status,
        }


class TrackerBank:
    """One FieldTracker per ROI, built from rois.json + optional config metadata."""

    def __init__(self, rois, field_meta=None, vote_window=15, history=120, min_conf=0.40):
        field_meta = field_meta or {}
        self.trackers = []
        for r in rois:
            meta = field_meta.get(r["name"], {}) or {}
            # metadata may also live directly in rois.json
            self.trackers.append(FieldTracker(
                name=r["name"],
                label=r.get("label") or meta.get("label"),
                unit=r.get("unit") or meta.get("unit", ""),
                decimals=r.get("decimals", meta.get("decimals")),
                vote_window=vote_window,
                history=history,
                min_conf=min_conf,
            ))
        self.by_name = {t.name: t for t in self.trackers}

    def update(self, name, raw, value, conf):
        return self.by_name[name].update(raw, value, conf)

    def snapshot(self):
        return {t.name: t.as_dict() for t in self.trackers}

    def __iter__(self):
        return iter(self.trackers)

    def __len__(self):
        return len(self.trackers)
