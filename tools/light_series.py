"""HSV lamp state per sampled frame for the visible lights; compare with stop-line crossings."""
import sys

import cv2
import numpy as np
import pandas as pd

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "C3905"
ROIS = {"ped": (490, 970, 600, 1110), "veh": (2230, 690, 2380, 870)}  # 4K px, C3905 framing
HUE = {"red": [(0, 10), (170, 180)], "amber": [(11, 30)], "green": [(60, 95)]}


def lamp_counts(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 2] > 150) & (hsv[..., 1] > 90)
    out = {}
    for name, ranges in HUE.items():
        m = np.zeros(bright.shape, bool)
        for lo, hi in ranges:
            m |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
        out[name] = int((m & bright).sum())
    return out


cap = cv2.VideoCapture(f"samples/{VIDEO}.MP4")
fps = cap.get(cv2.CAP_PROP_FPS)
rows, idx = [], 0
while cap.grab():
    if idx % 15 == 0:
        _, f = cap.retrieve()
        r = {"t": idx / fps}
        for k, (x1, y1, x2, y2) in ROIS.items():
            for c, n in lamp_counts(f[y1:y2, x1:x2]).items():
                r[f"{k}_{c}"] = n
        rows.append(r)
    idx += 1
df = pd.DataFrame(rows)
df["ped"] = np.where(df.ped_green > df.ped_red, "G", "R")
df.to_parquet(f"cache/{VIDEO}_lights.parquet", index=False)

# stop-line crossings of near-carriageway vehicles (line in 1080p coords, C3905 framing)
(ax, ay), (bx, by) = (290, 545), (935, 468)
tr = pd.read_parquet(f"cache/{VIDEO}_s3_960_yolo11s.parquet")
tr = tr[tr.cls.isin([2, 3, 5, 7])].sort_values(["id", "frame"])
tr["px"], tr["py"] = (tr.x1 + tr.x2) / 4, tr.y2 / 2
side = np.sign((bx - ax) * (tr.py - ay) - (by - ay) * (tr.px - ax))
tr["side"] = side
tr["prev"] = tr.groupby("id").side.shift()
within = (tr.px > ax - 20) & (tr.px < bx + 60)
cross = tr[within & (tr.prev < 0) & (tr.side > 0)]
state = np.interp(cross.t, df.t, (df.ped == "G").astype(float)) > 0.5
print("ped phase timeline (s):")
chg = df[df.ped != df.ped.shift()]
print("  ", "  ".join(f"{t:.0f}:{s}" for t, s in zip(chg.t, chg.ped)))
print(f"crossings total {len(cross)}: while ped GREEN {int(state.sum())}, while ped RED {int((~state).sum())}")
print("crossing times while ped green:", np.round(cross.t[state].to_numpy(), 1))
