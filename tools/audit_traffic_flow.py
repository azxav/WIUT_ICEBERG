"""List candidate vehicle movements through the fixed camera view."""
from pathlib import Path
import json

import pandas as pd
import cv2
import numpy as np


for path in sorted(Path("cache").glob("C*_tracks.parquet")):
    frame = pd.read_parquet(path)
    frame = frame[frame.cls.isin([2, 3, 5, 7])].copy()
    frame["x"] = (frame.x1 + frame.x2) / 4
    frame["y"] = frame.y2 / 2
    print(f"\n{path.stem}")
    rows = []
    for ident, track in frame.groupby("id"):
        track = track.sort_values("t")
        if len(track) < 12 or track.t.iloc[-1] - track.t.iloc[0] < 2:
            continue
        first = track.iloc[0]
        last = track.iloc[-1]
        # Near-side approach entering the junction, with enough observed travel.
        if first.x > 600 or first.y > 650 or last.y - first.y < 100:
            continue
        samples = track.iloc[[0, len(track)//4, len(track)//2, 3*len(track)//4, -1]]
        pts = " -> ".join(f"({r.x:.0f},{r.y:.0f})" for r in samples.itertuples())
        rows.append((float(first.t), f"id={ident} {first.t:.1f}-{last.t:.1f} cls={int(first.cls)} {pts}"))
    for _, row in sorted(rows)[:8]:
        print(row)
    if path.stem != "C3905_tracks":
        continue
    print("C3905 stop-line crossings: x at crossing, exit")
    crossing_rows = []
    for ident, track in frame.groupby("id"):
        track = track.sort_values("t")
        if len(track) < 15:
            continue
        start, end = track.iloc[0], track.iloc[-1]
        if start.x > 600 or start.y > 540:
            continue
        line_y = 545 - (track.x - 289) * 79 / 645
        signed = track.y - line_y
        crossing = track[(signed >= 0) & track.x.between(289, 934)]
        if crossing.empty or signed.iloc[0] > 0:
            continue
        cross = crossing.iloc[0]
        exit_name = "lower-left turn" if end.x < 160 and end.y > 830 else "down-right" if end.x > 1400 and end.y > 750 else "other/partial"
        crossing_rows.append((cross.x, f"id={ident} t={cross.t:.1f} x={cross.x:.0f} y={cross.y:.0f} -> {exit_name} ({end.x:.0f},{end.y:.0f})"))
    for _, row in sorted(crossing_rows):
        print(row)


def movement_sheet(video, ident, times):
    source = Path("samples/proxy") / f"{video}_1080p.mp4"
    cap = cv2.VideoCapture(str(source))
    track = pd.read_parquet(Path("cache") / f"{video}_tracks.parquet")
    track = track[track.id == ident].sort_values("t")
    panels = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (960, 540))
        upto = track[track.t <= t]
        if len(upto) >= 2:
            pts = np.c_[((upto.x1 + upto.x2) / 8).to_numpy(), (upto.y2 / 4).to_numpy()].astype(np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 3)
        one = track.iloc[(track.t - t).abs().argmin()]
        box = tuple(int(v / 4) for v in (one.x1, one.y1, one.x2, one.y2))
        cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255), 3)
        cv2.putText(frame, f"{video} id {ident}  {t:.1f}s", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 3)
        panels.append(frame)
    cap.release()
    if panels:
        while len(panels) < 6:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:6])])
        output = Path("eda/traffic_audit")
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / f"{video}_turn_{ident}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


for video, ident, times in [
    ("C3896", 118, [29, 31, 33, 34.5, 35.5]),
    ("C3897", 1329, [53, 55, 57, 59, 61]),
    ("C3902", 1076, [45, 48, 51, 54, 56]),
    ("C3905", 1328, [38, 40, 42, 44, 47]),
]:
    movement_sheet(video, ident, times)


def movement_map():
    canvas = cv2.imread("src/scene_ref.jpg")
    canvas = cv2.convertScaleAbs(canvas, alpha=1.55, beta=8)
    scene = json.loads(Path("src/scene.json").read_text())
    for name in ("A_near", "B_far", "C_lower"):
        poly = np.array(scene["crosswalks"][name]["polygon"], np.int32)
        cv2.polylines(canvas, [poly], True, (50, 210, 80), 3)
    stop = np.array(scene["stop_lines"]["near"]["line"], np.int32)
    cv2.polylines(canvas, [stop], False, (50, 50, 255), 4)
    tracks = pd.read_parquet("cache/C3905_tracks.parquet")
    for ident, color, label, at in [
        (1328, (0, 225, 255), "RIGHT TURN TO LOWER-LEFT BRANCH", (420, 735)),
        (1359, (255, 255, 0), "STRAIGHT", (850, 695)),
    ]:
        track = tracks[tracks.id == ident].sort_values("t")
        pts = np.c_[((track.x1 + track.x2) / 4).to_numpy(), (track.y2 / 2).to_numpy()].astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, 5, cv2.LINE_AA)
        for i in range(12, len(pts)-8, max(12, len(pts)//8)):
            cv2.arrowedLine(canvas, tuple(pts[i]), tuple(pts[i+8]), color, 4, tipLength=.45)
        cv2.putText(canvas, label, at, cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(canvas, label, at, cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    for key, color, caption in [
        ("veh_main", (70, 70, 255), "VEHICLE SIGNAL"),
        ("left_red_green", (255, 80, 200), "PEDESTRIAN SIGNAL"),
    ]:
        x1, y1, x2, y2 = scene["signals"][key]["roi"]
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 4)
        cv2.putText(canvas, caption, (x1-80, y1-15), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Observed tracks; scene boundaries still provisional", (900, 1020),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite("eda/traffic_audit/movement_map.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])


movement_map()
