"""Render a registered scene overlay. Usage: draw_scene.py [background] [output]."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene import _warp, register


def render(image, scene):
    image = cv2.convertScaleAbs(image, alpha=1.35, beta=5)
    overlay = image.copy()

    def points(pts):
        return np.int32(pts).reshape(-1, 1, 2)

    def line(pts, color, label, width=2):
        cv2.polylines(image, [points(pts)], False, color, width)
        x, y = np.int32(pts[-1])
        cv2.putText(image, label, (int(x) + 4, int(y) - 4), 0, .45, color, 1)

    def dashed(pts, color, label):
        for p, q in zip(pts[:-1], pts[1:]):
            p, q = np.asarray(p, float), np.asarray(q, float)
            length = np.linalg.norm(q - p)
            for offset in np.arange(0, length, 24):
                a = p + (q-p) * (offset / length)
                b = p + (q-p) * (min(offset+13, length) / length)
                cv2.line(image, tuple(np.int32(a)), tuple(np.int32(b)), color, 2)
        x, y = np.int32(pts[-1])
        cv2.putText(image, label, (int(x) + 4, int(y) - 4), 0, .45, color, 1)

    for name, item in scene.get("carriageways", {}).items():
        cv2.fillPoly(overlay, [points(item["polygon"])], (255, 140, 0) if name == "far" else (0, 140, 255))
    for name, item in scene.get("zones", {}).items():
        cv2.fillPoly(overlay, [points(item["polygon"])], (0, 200, 255) if name == "queue_near" else (180, 0, 180))
    image = cv2.addWeighted(overlay, .15, image, .85, 0)
    for name, item in scene.get("crosswalks", {}).items():
        cv2.polylines(image, [points(item["polygon"])], True, (0, 255, 0), 2)
        x, y = np.int32(item["polygon"][0])
        cv2.putText(image, name, (int(x), int(y) - 6), 0, .55, (0, 255, 0), 2)
    for name, item in scene.get("lanes", {}).items():
        cv2.polylines(image, [points(item["polygon"])], True, (160, 80, 255), 1)
        x, y = np.int32(item["polygon"][0])
        cv2.putText(image, "lane:" + name, (int(x), int(y) - 5), 0, .4, (160, 80, 255), 1)
    for key, color in (("stop_lines", (0, 0, 255)), ("solid_lines", (0, 165, 255)),
                       ("dashed_lines", (255, 255, 255)), ("curbs", (255, 255, 0))):
        for name, item in scene.get(key, {}).items():
            pts = item.get("line", item.get("polyline"))
            if key == "dashed_lines":
                dashed(pts, color, key + ":" + name)
            else:
                line(pts, color, key + ":" + name, 3 if key == "stop_lines" else 2)
    if "polyline" in scene.get("median", {}):
        line(scene["median"]["polyline"], (0, 255, 255), "median", 3)
    for name, item in scene.get("no_stopping", {}).items():
        cv2.polylines(image, [points(item["polygon"])], True, (0, 0, 200), 2)
        x, y = np.int32(item["polygon"][0])
        cv2.putText(image, "no stop:" + name, (int(x), int(y) - 4), 0, .45, (0, 0, 200), 1)
    for name, item in scene.get("signals", {}).items():
        x1, y1, x2, y2 = np.int32(item["roi"])
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(image, name, (int(x1), int(y1) - 5), 0, .55, (255, 0, 255), 2)
    return image


def main():
    background = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/scene_ref.jpg")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eda/scene_overlay.jpg")
    scene_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("src/scene.json")
    scene = json.loads(scene_file.read_text())
    image = cv2.imread(str(background))
    image = cv2.resize(image, tuple(scene["ref_size"]), interpolation=cv2.INTER_AREA)
    if background.resolve() != (scene_file.parent / scene["ref_image"]).resolve():
        ref = cv2.imread(str(scene_file.parent / scene["ref_image"]))
        H, count = register(ref, image)
        if H is None:
            raise RuntimeError(f"scene registration failed: {background}, {count} inliers")
        for key in ("carriageways", "median", "stop_lines", "crosswalks", "signals", "zones",
                    "solid_lines", "dashed_lines", "curbs", "no_stopping", "lanes"):
            if key in scene:
                scene[key] = _warp(scene[key], H)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render(image, scene), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(output)


if __name__ == "__main__":
    main()
