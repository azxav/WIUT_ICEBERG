"""Save reference frames and a median background image per sample video."""
from pathlib import Path

import cv2
import numpy as np

OUT = Path("eda/frames")
OUT.mkdir(parents=True, exist_ok=True)

for p in sorted(Path("samples").glob("*.MP4")):
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stack = []
    for k, idx in enumerate(np.linspace(0, n - 1, 41).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            continue
        small = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
        stack.append(small)
        if k in (0, 20, 40):
            cv2.imwrite(str(OUT / f"{p.stem}_f{idx:05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{p.stem}_background.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(p.stem, len(stack), "frames used")
