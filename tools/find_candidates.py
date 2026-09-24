"""Loose track-based event leads for manual visual verification; no automatic labels."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene import load_scene

VEHICLES = {1, 2, 3, 5, 7}


def inside(poly, x, y):
    return cv2.pointPolygonTest(np.float32(poly), (float(x), float(y)), False) >= 0


def segments(t, flags, max_gap=.4, min_duration=.2):
    hits = np.asarray(t)[np.asarray(flags)]
    if not len(hits):
        return []
    splits = np.where(np.diff(hits) > max_gap)[0] + 1
    return [(round(float(a[0]), 2), round(float(a[-1]), 2)) for a in np.split(hits, splits)
            if a[-1] - a[0] >= min_duration]


for name in sys.argv[1:] or ["C3905"]:
    file = Path(f"cache/{name}_s3_960_yolo11s.parquet")
    if not file.exists():
        print("missing", file)
        continue
    scene, n = load_scene(cv2.imread(f"eda/frames/{name}_background.jpg"))
    if scene is None:
        raise RuntimeError((name, n))
    df = pd.read_parquet(file).sort_values(["id", "t"])
    df["cx"] = (df.x1 + df.x2) / 4
    df["foot"] = df.y2 / 2
    df["front_x"] = df.x2 / 2
    df["front_y"] = df.y2 / 2
    rows = []
    crosswalks = {k: v["polygon"] for k, v in scene["crosswalks"].items()}
    roads = {k: v["polygon"] for k, v in scene["carriageways"].items()}
    line = scene["stop_lines"]["near"]["line"]
    (ax, ay), (bx, by) = line
    for tid, g in df.groupby("id"):
        if len(g) < 3:
            continue
        cls = int(g.cls.mode().iloc[0])
        t, x, y = g.t.to_numpy(), g.cx.to_numpy(), g.foot.to_numpy()
        positions = np.stack((x, y), axis=1)
        in_cw = {key: np.array([inside(poly, *p) for p in positions]) for key, poly in crosswalks.items()}
        in_road = {key: np.array([inside(poly, *p) for p in positions]) for key, poly in roads.items()}
        if cls == 0:
            for key, flag in in_cw.items():
                for a, b in segments(t, flag, .5, .3):
                    rows.append(("crosswalk_person", a, b, tid, cls, key))
            flag = (in_road["near"] | in_road["far"]) & ~np.logical_or.reduce(list(in_cw.values()))
            for a, b in segments(t, flag, .6, 1):
                rows.append(("jaywalking", a, b, tid, cls, "road outside crossing"))
        if cls in VEHICLES:
            for key, flag in in_cw.items():
                for a, b in segments(t, flag, .5, .25):
                    rows.append(("crosswalk_vehicle", a, b, tid, cls, key))
            side = (bx-ax)*(g.front_y.to_numpy()-ay) - (by-ay)*(g.front_x.to_numpy()-ax)
            side /= np.hypot(bx-ax, by-ay)
            valid = (g.front_x.to_numpy() >= min(ax,bx)-5) & (g.front_x.to_numpy() <= max(ax,bx)+10)
            crosses = np.where((side[1:] >= 0) & (side[:-1] < 0) & valid[1:])[0] + 1
            for i in crosses:
                rows.append(("stop_crossing", round(float(t[i]), 2), round(float(t[i]), 2), tid, cls, "near"))
            # Slow/stopped candidate uses median positions over 1 s, robust to small box jitter.
            tx = pd.DataFrame({"t": t, "x": x, "y": y})
            tx["vx"] = tx.x.diff(10).abs() / tx.t.diff(10)
            tx["vy"] = tx.y.diff(10).abs() / tx.t.diff(10)
            speed = np.hypot(tx.vx.fillna(999), tx.vy.fillna(999)).to_numpy()
            for a, b in segments(t, speed < 5, .5, 8):
                rows.append(("stopped_or_queue", a, b, tid, cls, "<5 px/s for >=8s"))
            # Sustained motion against the local carriageway vector.
            if len(g) > 10:
                for key, flag in in_road.items():
                    direction = np.array(scene["carriageways"][key]["direction"])
                    delta = positions[10:] - positions[:-10]
                    vel = (delta @ direction) / (t[10:] - t[:-10])
                    reverse = np.r_[np.zeros(10, bool), vel < -12] & flag
                    for a, b in segments(t, reverse, .5, 1):
                        rows.append(("wrong_way", a, b, tid, cls, key))
            # Large change of direction, useful for illegal turn/U-turn review.
            if len(g) >= 30:
                dx = x[10:] - x[:-10]
                dy = y[10:] - y[:-10]
                angles = np.unwrap(np.arctan2(dy, dx))
                turn = np.abs(angles[10:] - angles[:-10])
                for a, b in segments(t[20:], turn > .65, .6, .3):
                    rows.append(("turn_or_swerve", a, b, tid, cls, f"{np.rad2deg(turn.max()):.0f}deg max"))
    out = pd.DataFrame(rows, columns=["kind", "start", "end", "track", "cls", "detail"])
    out = out.sort_values(["start", "kind", "track"])
    dest = Path(f"cache/{name}_candidates.csv")
    out.to_csv(dest, index=False)
    print(name, len(out), "candidates", out.kind.value_counts().to_dict(), "->", dest)
