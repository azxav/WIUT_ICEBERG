"""One grab-skip pass per video: signal samples and overview contact sheets."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene import register

OUT = Path("eda/survey")
OUT.mkdir(parents=True, exist_ok=True)
ref = cv2.imread("src/scene_ref.jpg")


def count_lamps(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s > 95) & (v > 145)
    return {"R": int(((((h < 12) | (h > 168)) & bright)).sum()),
            "Y": int((((h >= 12) & (h < 35)) & bright).sum()),
            "G": int((((h >= 38) & (h < 100)) & bright).sum())}


for name in ("C3896", "C3897", "C3902", "C3905"):
    bg = cv2.imread(f"eda/frames/{name}_background.jpg")
    H, n = register(ref, bg)
    if H is None:
        raise RuntimeError(f"registration failed for {name}: {n}")
    rois = {}
    for key, box in {"main": (1143, 365, 1172, 421), "left": (254, 508, 283, 549)}.items():
        pts = cv2.perspectiveTransform(np.float32(box).reshape(2, 1, 2), H).reshape(2, 2)
        rois[key] = np.rint(pts * 2).astype(int).ravel().tolist()
    cap = cv2.VideoCapture(f"samples/{name}.MP4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = round(fps * .5)
    sheet_stride = round(fps * 5)
    rows, thumbs = [], []
    idx = 0
    while cap.grab():
        if idx % stride == 0 or idx % sheet_stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if idx % stride == 0:
                row = {"t": round(idx / fps, 3)}
                for key, (x1, y1, x2, y2) in rois.items():
                    row[key] = count_lamps(frame[y1:y2, x1:x2])
                rows.append(row)
            if idx % sheet_stride == 0:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.rectangle(small, (0, 0), (135, 33), (0, 0, 0), -1)
                cv2.putText(small, f"{idx/fps:.1f}s", (5, 25), 0, .75, (255, 255, 255), 2)
                thumbs.append(small)
        idx += 1
    cap.release()
    for start in range(0, len(thumbs), 12):
        page = thumbs[start:start+12]
        canvas = np.zeros((3 * 360, 4 * 640, 3), np.uint8)
        for i, im in enumerate(page):
            y, x = divmod(i, 4)
            canvas[y*360:(y+1)*360, x*640:(x+1)*640] = im
        cv2.imwrite(str(OUT / f"{name}_{start//12:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    Path(f"cache/{name}_signal_samples.json").write_text(json.dumps({"fps": fps, "duration": idx/fps,
                                                                    "rois": rois, "samples": rows}))
    print(name, round(idx/fps, 1), "s", len(rows), "light samples", len(thumbs), "overview frames")
