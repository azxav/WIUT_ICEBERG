"""Inspect the median vehicle signal near phase transitions found in survey_videos."""
import json
from pathlib import Path

import cv2
import numpy as np


def state(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s > 50) & (v > 60)
    red = int(((((h < 12) | (h > 168)) & bright)).sum())
    green = int(((((h >= 35) & (h < 105)) & bright)).sum())
    amber = int(((((h >= 12) & (h < 35)) & bright)).sum())
    if red >= 25 and red > green * 1.5:
        return "R"
    if green >= 25 and green > red * 1.5:
        return "G"
    if amber >= 18 and amber > max(red, green):
        return "Y"
    return "?"


for name in ("C3896", "C3897", "C3902", "C3905"):
    data = json.loads(Path(f"cache/{name}_signal_samples.json").read_text())
    prior = "?"
    left_changes = []
    for row in data["samples"]:
        left = row["left"]
        phase = max(("R", "G"), key=lambda k: left[k])
        if left[phase] < 12:
            continue
        if phase != prior:
            left_changes.append((row["t"], phase))
            prior = phase
    cap = cv2.VideoCapture(f"samples/{name}.MP4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    x1, y1, x2, y2 = data["rois"]["main"]
    results = []
    for change, left in left_changes:
        start = max(0, change - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start*fps))
        index = round(start*fps)
        samples = []
        while index/fps <= min(data["duration"], change+7) and cap.grab():
            if index % 3 == 0:
                ok, frame = cap.retrieve()
                if ok:
                    samples.append((round(index/fps, 3), state(frame[y1:y2, x1:x2])))
            index += 1
        results.append({"left_change": change, "left_state": left, "median_samples": samples})
    cap.release()
    Path(f"cache/{name}_phase_refinement.json").write_text(json.dumps(results))
    print(name, [(round(x["left_change"], 1), x["left_state"],
                  [(t,s) for t,s in x["median_samples"] if s != "?"][:2]) for x in results])
