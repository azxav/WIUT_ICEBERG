"""Build manually reviewed dev GT, notes, and three-frame evidence strips."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# Seconds refer to visible event onset/clearance, not YOLO track lifespan.
# Each row: start, end, class, relevant track IDs, rationale, confidence.
EVENTS = {
    "C3896": [
        (0.0, 34.0, "congestion", [], "Near approach stopped across lanes at opening; clears after green", "medium"),
        (78.9, 111.3, "red_light", [1754], "White SUV crosses near stop line with median head red; downstream queue delays frame exit", "high"),
        (79.5, 80.9, "failure_to_yield", [1754, 844, 1772, 1661], "White SUV enters crossing A while people occupy it", "medium"),
        (87.0, 111.0, "congestion", [], "Near approach queue reaches all lanes and clears after green", "medium"),
        (146.0, 184.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (164.5, 170.8, "jaywalking", [3563], "Person leaves crossing A and traverses open carriageway toward lower refuge", "high"),
        (230.0, 258.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (237.0, 251.5, "jaywalking", [5456, 5818], "Pedestrians traverse road between crossing A and lower marked crossing", "high"),
        (298.0, 330.0, "congestion", [], "Near approach stopped across lanes until green clears it", "medium"),
        (326.9, 332.6, "red_light", [7582], "White sedan starts through stop line before median green", "medium"),
    ],
    "C3897": [
        (0.0, 27.0, "congestion", [], "Near approach queue already stopped at opening and clears after green", "medium"),
        (1.5, 20.0, "stop_line", [4], "Black sedan creeps past near stop line then waits for green", "high"),
        (11.0, 20.5, "jaywalking", [81, 49, 179], "People cross open road from crossing A to lower refuge", "high"),
        (57.3, 66.7, "red_light", [1400], "Bus crosses near stop line on median red and leaves frame", "high"),
        (63.0, 101.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (78.0, 88.5, "jaywalking", [1659, 2002, 1131], "People leave crossing A and walk diagonally to lower refuge", "high"),
        (94.3, 101.0, "red_light", [808], "White sedan starts over line before median green", "medium"),
        (130.8, 137.6, "red_light", [3459, 3575], "Two white sedans cross the stop line on the same red phase", "high"),
        (140.0, 175.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (230.0, 250.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (238.0, 245.0, "jaywalking", [6166, 6065], "People cross open space from crossing A to lower refuge", "high"),
        (244.3, 252.1, "red_light", [5731], "Black SUV begins crossing just before median green", "medium"),
        (281.6, 284.9, "red_light", [7616], "White sedan crosses stop line after median red", "high"),
        (294.0, 317.7, "congestion", [], "Near approach stops across lanes; video ends while queue remains", "medium"),
        (307.0, 315.4, "jaywalking", [8129, 8116], "Person crosses open road into lower crossing", "medium"),
    ],
    "C3902": [
        (4.0, 43.0, "congestion", [], "Near approach stopped across lanes until green discharge", "medium"),
        (82.0, 121.0, "congestion", [], "Near approach stopped across lanes until green discharge", "medium"),
        (83.0, 106.0, "jaywalking", [2533, 2536, 2493], "Three people leave lower crossing and traverse open junction toward far crossing", "high"),
        (168.0, 201.0, "congestion", [], "Near approach stopped across lanes", "medium"),
        (178.0, 190.5, "jaywalking", [5258, 5176, 4848], "Pedestrians leave crossing A and cut diagonally toward lower crossing", "high"),
        (219.4, 230.5, "jaywalking", [7269], "Person enters lower roadway outside marking, reaches triangular refuge", "high"),
        (237.5, 245.8, "jaywalking", [7269], "Person leaves triangular refuge, crosses open road to crossing A", "high"),
        (254.2, 277.2, "stop_line", [8338], "Dark SUV stops with front bumper beyond near line on red; moves on green", "medium"),
        (257.0, 269.8, "jaywalking", [7815, 5837], "People leave crossing A and cross open roadway toward lower refuge", "high"),
        (258.0, 281.0, "congestion", [], "Near approach stopped across lanes until green discharge", "medium"),
    ],
    "C3905": [
        (13.5, 17.5, "jaywalking", [13], "Person leaves crossing A diagonally through open carriageway onto refuge", "high"),
        (23.0, 39.0, "congestion", [], "Near approach stops across lanes then clears after green", "medium"),
        (78.0, 120.0, "congestion", [], "Near approach stopped across lanes until green discharge", "medium"),
    ],
}

PHASES = {
    "C3896": "R 0–27.2; G 27.2–63.1; R 63.1–102.1; G 102.1–138.1; R 138.1–177.1; G 177.1–213.1; R 213.1–252.2; G 252.2–288.3; R 288.3–327.3; G 327.3–end",
    "C3897": "R 0–20.0; G 20.0–55.8; R 55.8–94.8; G 94.8–130.7; R 130.7–169.9; G 169.9–205.9; R 205.9–244.9; G 244.9–281.0; R 281.0–end",
    "C3902": "R 0–37.3; G 37.3–75.2; R 75.2–117.1; G 117.1–155.2; R 155.2–197.2; G 197.2–235.4; R 235.4–277.2; G 277.2–315.3; R 315.3–end (last three transitions inferred from 2-lamp head)",
    "C3905": "R 0–34.5; G 34.5–72.7; R 72.7–114.5; G 114.5–end",
}

UNCERTAIN = [
    "C3896 86–99: SUV 1754 stands downstream after red-light entry; it is within a traffic queue, so no separate stopped_vehicle label.",
    "C3896 61.9/212.3/251.8, C3897 54.2/204–205, C3902 75/154–156/313, C3905 72: stop-line candidates fall in median green/amber or before confirmed red; no red_light label.",
    "C3897 0.5: sedan 4 creeps over the line and stops. Marked stop_line; whether to also count red_light is ambiguous.",
    "C3897 94.3 and 244.3; C3896 326.9: pre-green line crossings are only 0.4–0.7 s before green. Red head visible, but label confidence medium.",
    "C3902 254.2: SUV 8338 is only a few pixels beyond the stop line; stop_line boundary confidence medium.",
    "Crosswalk overlap leads, especially C_lower and B_far, contain projection errors and queueing vehicles; additional failure_to_yield cases need 4K frame-by-frame review.",
    "No collision, evasive near miss, wrong-way drive, prohibited turn/U-turn, solid-line crossing, obstacle, or fire/smoke was confirmed in overview scan. Turn permissions are not legible.",
]


def frame_strip(cap, fps, duration, name, idx, event, tracks):
    start, end, label, ids, _, _ = event
    times = [start, (start + end) / 2, min(end, duration - 1 / fps)]
    panels = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError((name, t))
        if label == "congestion":
            crop = (0, 0, 3840, 2160)
        elif label == "jaywalking":
            crop = (260, 850, 3400, 2160)
        else:
            crop = (200, 620, 3840, 2160)
        x0, y0, x1, y1 = crop
        panel = frame[y0:y1, x0:x1].copy()
        for tid in ids:
            g = tracks[(tracks.id == tid) & tracks.t.between(t - .18, t + .18)]
            if len(g):
                r = g.iloc[(g.t-t).abs().argmin()]
                cv2.rectangle(panel, (int(r.x1)-x0, int(r.y1)-y0),
                              (int(r.x2)-x0, int(r.y2)-y0), (0, 0, 255), 8)
                cv2.putText(panel, f"id {tid}", (int(r.x1)-x0, max(40, int(r.y1)-y0-10)),
                            0, 1.5, (0, 0, 255), 4)
        width = 1600
        panel = cv2.resize(panel, (width, round(panel.shape[0]*width/panel.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0,0), (width,58), (0,0,0), -1)
        cv2.putText(panel, f"{name} {label} {t:.1f}s ({['start','middle','end'][len(panels)]})",
                    (18,42),0,1.15,(255,255,255),3)
        panels.append(panel)
    strip = np.vstack(panels)
    dest = Path("labels/evidence") / f"{name}_{idx:02d}_{label}.jpg"
    cv2.imwrite(str(dest), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    Path("labels/evidence").mkdir(parents=True, exist_ok=True)
    for stale in Path("labels/evidence").glob("C*_*.jpg"):
        stale.unlink()
    gt = {}
    notes = ["# Dev label notes", "", "Times are seconds. Event intervals follow the official PDF; video boundary truncates open intervals. Evidence strips show start, middle, end. Tracking was a candidate finder; labels were reviewed against 4K crops and 5 s overview contact sheets. In phase timelines, G covers green plus the final amber clearance interval; R begins only when median red is visible.", ""]
    for name, rows in EVENTS.items():
        cap = cv2.VideoCapture(f"samples/{name}.MP4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        ordered = sorted(rows, key=lambda x: (x[0], x[2]))
        gt[f"{name}.MP4"] = {"duration": round(duration, 3), "fps": round(fps, 3),
                             "events": [[round(a,1), round(b,1), c] for a,b,c,*_ in ordered]}
        tracks = pd.read_parquet(f"cache/{name}_s3_960_yolo11s.parquet")
        notes += [f"## {name}.MP4", "", "| # | Start | End | Label | Track IDs | Why | Confidence |", "|---:|---:|---:|---|---|---|---|"]
        for idx, event in enumerate(ordered, 1):
            a,b,c,ids,why,confidence = event
            assert 0 <= a < b <= duration + .01, (name,event,duration)
            frame_strip(cap, fps, duration, name, idx, event, tracks)
            notes.append(f"| {idx} | {a:.1f} | {b:.1f} | {c} | {', '.join(map(str,ids)) or '—'} | {why} | {confidence} |")
        cap.release()
        notes += ["", "Median vehicle-head phase timeline: " + PHASES[name] + ". Transition estimates ±0.2 s where the head is occluded; the left 2-lamp head turns red ~3 s earlier during amber.", ""]
    notes += ["## Uncertain candidates", ""] + [f"- {s}" for s in UNCERTAIN] + [""]
    Path("labels/dev_labels.json").write_text(json.dumps(gt, indent=2) + "\n")
    Path("labels/dev_labels_notes.md").write_text("\n".join(notes), encoding="utf-8")
    for name, item in gt.items():
        print(name, dict(Counter(row[2] for row in item["events"])))


if __name__ == "__main__":
    main()
