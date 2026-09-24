"""Draw src/scene.json over a background image for visual check."""
import json
import sys

import cv2
import numpy as np

scene = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "src/scene.json"))
img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "src/scene_ref.jpg")
img = cv2.convertScaleAbs(img, alpha=1.6, beta=10)
over = img.copy()
P = lambda pts: np.int32(pts).reshape(-1, 1, 2)
for name, c in scene["carriageways"].items():
    col = (255, 140, 0) if name == "far" else (0, 140, 255)
    cv2.fillPoly(over, [P(c["polygon"])], col)
for name, z in scene["zones"].items():
    cv2.fillPoly(over, [P(z["polygon"])], (0, 200, 255) if name == "queue_near" else (180, 0, 180))
img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
for name, c in scene["carriageways"].items():
    pts = np.array(c["polygon"])
    ctr = pts.mean(0).astype(int)
    d = np.array(c["direction"]) * 120
    cv2.arrowedLine(img, tuple(ctr), tuple((ctr + d).astype(int)), (255, 255, 255), 5, tipLength=0.3)
    cv2.putText(img, f"carriageway {name}", tuple(ctr + [10, -15]), 0, 0.9, (255, 255, 255), 2)
for name, cw in scene["crosswalks"].items():
    cv2.polylines(img, [P(cw["polygon"])], True, (0, 255, 0), 3)
    cv2.putText(img, f"crosswalk {name}", tuple(np.int32(cw["polygon"][0]) + [0, -8]), 0, 0.7, (0, 255, 0), 2)
for name, s in scene["stop_lines"].items():
    a, b = [tuple(map(int, p)) for p in s["line"]]
    cv2.line(img, a, b, (0, 0, 255), 4)
    cv2.putText(img, f"stop line {name}", (a[0], a[1] + 30), 0, 0.8, (0, 0, 255), 2)
a, b = [tuple(map(int, p)) for p in scene["median"]["polyline"]]
cv2.line(img, a, b, (0, 255, 255), 3)
for name, s in scene["signals"].items():
    x1, y1, x2, y2 = map(int, s["roi"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.putText(img, name, (x1, y1 - 8), 0, 0.7, (255, 0, 255), 2)
for name, z in scene["zones"].items():
    cv2.putText(img, f"zone {name}", tuple(np.int32(z["polygon"]).mean(0).astype(int)), 0, 0.8, (255, 255, 0), 2)
out = sys.argv[2] if len(sys.argv) > 2 else "eda/scene_overlay.jpg"
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out)
