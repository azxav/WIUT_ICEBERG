"""Draw track paths on the background image; colour = direction of motion."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

tracks = Path(sys.argv[1])
video = tracks.stem.split("_")[0]
bg = cv2.imread(f"eda/frames/{video}_background.jpg")
bg = (bg * 0.55).astype(np.uint8)
df = pd.read_parquet(tracks)
df["bx"] = (df.x1 + df.x2) / 4  # bottom-centre in 1920x1080 image
df["by"] = df.y2 / 2
for kind, classes in {"vehicles": [2, 3, 5, 7], "persons": [0, 1]}.items():
    img = bg.copy()
    for tid, g in df[df.cls.isin(classes)].groupby("id"):
        pts = g[["bx", "by"]].to_numpy()
        if len(pts) < 5 or np.linalg.norm(pts[-1] - pts[0]) < 80:
            continue
        d = pts[-1] - pts[0]
        hue = int((math.degrees(math.atan2(d[1], d[0])) % 360) / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [pts.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.arrowedLine(img, tuple(pts[-4].astype(int)), tuple(pts[-1].astype(int)), col, 3, tipLength=1.5)
    # legend: colour wheel for direction
    c = (1820, 100)
    for a in range(0, 360, 5):
        col = cv2.cvtColor(np.uint8([[[a // 2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        e = (int(c[0] + 70 * math.cos(math.radians(a))), int(c[1] + 70 * math.sin(math.radians(a))))
        cv2.line(img, c, e, col, 4)
    cv2.imwrite(f"eda/{video}_traj_{kind}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"eda/{video}_traj_{kind}.jpg")
