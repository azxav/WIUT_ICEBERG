"""Quick YOLO11 + ByteTrack run: track table to Parquet and timing per stage."""
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

VIDEO = Path(sys.argv[1])
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3
IMGSZ = int(sys.argv[3]) if len(sys.argv) > 3 else 960
WEIGHTS = sys.argv[4] if len(sys.argv) > 4 else "weights/yolo11s.pt"
ROAD_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
SCALE = 2  # 4K frame resized to 1920x1080 before inference

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
rows, idx = [], 0
t_dec = t_inf = 0.0
t_all = time.perf_counter()
while True:
    t = time.perf_counter()
    if not cap.grab():
        break
    if idx % STRIDE:
        t_dec += time.perf_counter() - t
        idx += 1
        continue
    _, frame = cap.retrieve()
    small = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    t_dec += time.perf_counter() - t
    t = time.perf_counter()
    r = model.track(small, imgsz=IMGSZ, conf=0.2, classes=ROAD_CLASSES, persist=True,
                    tracker="bytetrack.yaml", device=0, half=True, verbose=False)[0]
    t_inf += time.perf_counter() - t
    b = r.boxes
    if b.id is not None:
        for tid, c, s, (x1, y1, x2, y2) in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                                b.conf.tolist(), b.xyxy.tolist()):
            rows.append((idx, idx / fps, tid, c, s, x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE))
    idx += 1
wall = time.perf_counter() - t_all
dur = idx / fps
df = pd.DataFrame(rows, columns=["frame", "t", "id", "cls", "conf", "x1", "y1", "x2", "y2"])
out = Path("cache") / f"{VIDEO.stem}_s{STRIDE}_{IMGSZ}_{Path(WEIGHTS).stem}.parquet"
df.to_parquet(out, index=False)
print(f"{VIDEO.stem}: {dur:.1f}s video, wall {wall:.1f}s = {wall / dur:.2f}x | decode {t_dec:.1f}s, "
      f"infer+track {t_inf:.1f}s | {df.id.nunique()} tracks, {len(df)} rows -> {out}")
