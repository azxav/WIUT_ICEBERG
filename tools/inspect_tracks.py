"""Save 4K crops around requested time:track pairs for visual candidate review."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

name = sys.argv[1]
pairs = [(float(x.split(':')[0]), int(x.split(':')[1])) for x in sys.argv[2:]]
tracks = pd.read_parquet(f"cache/{name}_s3_960_yolo11s.parquet")
cap = cv2.VideoCapture(f"samples/{name}.MP4")
fps = cap.get(cv2.CAP_PROP_FPS)
out = Path("eda/candidate_review")
out.mkdir(exist_ok=True)
for t, tid in pairs:
    ims = []
    for tt in [max(0, t-1), t, min(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps-.1, t+1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(tt*fps))
        ok, frame = cap.read()
        if not ok:
            continue
        # 4K crop: controlled approach and crossing A; extend to right edge for vehicle exit.
        crop = frame[670:1400, 350:2550].copy()
        match = tracks[(tracks.id == tid) & (tracks.t.between(tt-.11, tt+.11))]
        if len(match):
            row = match.iloc[(match.t-tt).abs().argmin()]
            x1, y1, x2, y2 = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
            cv2.rectangle(crop, (x1-350, y1-670), (x2-350, y2-670), (0, 0, 255), 5)
            cv2.putText(crop, f"id={tid}", (x1-350, max(35,y1-680)), 0, 1.2, (0,0,255), 3)
        cv2.putText(crop, f"{name} {tt:.2f}s", (20, 50), 0, 1.3, (255,255,255), 3)
        ims.append(cv2.resize(crop, (1100,365)))
    if len(ims) == 3:
        cv2.imwrite(str(out/f"{name}_{t:.1f}_id{tid}.jpg"), np.vstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 92])
