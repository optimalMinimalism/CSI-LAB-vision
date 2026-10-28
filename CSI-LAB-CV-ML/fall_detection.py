"""Experimental single-person temporal heuristic; scores are not probabilities."""
import math


class FallDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.upright = None
        self.candidate = None

    def predict(self, people, postures, metadata):
        result = dict(enabled=True, experimental=True, state="UNKNOWN")
        stamp = metadata.get("monotonic_ns")
        source = (metadata.get("source_id"), metadata.get("experiment_id"))
        if type(stamp) is not int or len(people) != 1:
            self.reset()
            return result
        if self.previous and (source != self.previous[0] or not 0 < stamp-self.previous[1] <= 1_000_000_000):
            self.reset()
        self.previous = source, stamp
        state = postures[0]["posture"]
        box = people[0]["bbox"]
        x1, y1, x2, y2 = box
        if not all(math.isfinite(value) for value in box) or x2 <= x1 or y2 <= y1 or state == "UNKNOWN":
            self.upright = self.candidate = None
            return result
        center = ((x1+x2)/2, (y1+y2)/2)
        if state in ("STANDING", "SITTING"):
            self.upright = stamp, center, y2-y1
            self.candidate = None
            result["state"] = "MONITORING"
        elif state == "LYING_DOWN":
            if self.upright:
                elapsed = stamp-self.upright[0]
                origin, height = self.upright[1:]
                if (0 < elapsed <= 1_500_000_000 and
                        center[1]-origin[1] >= .2*height and
                        abs(center[0]-origin[0]) <= height):
                    if self.candidate is None:
                        self.candidate = stamp
                    if stamp-self.candidate >= 300_000_000:
                        result["state"] = "SUSPECTED_FALL"
                else:
                    self.upright = self.candidate = None
            # Already lying down is never evidence of a fall.
        return result
