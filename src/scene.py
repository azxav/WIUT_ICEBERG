"""Scene map: load src/scene.json and map its geometry into a video's frame via homography."""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MIN_INLIERS = 60


def _features(gray):
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    return cv2.SIFT_create(4000).detectAndCompute(gray, None)


def register(ref_bgr, frame_bgr):
    """Homography ref -> frame (both 1920x1080), or None if the scene does not match."""
    k1, d1 = _features(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
    k2, d2 = _features(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    if d1 is None or d2 is None:
        return None, 0
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if n >= MIN_INLIERS else None), n


def _warp(obj, H):
    if isinstance(obj, dict):
        return {k: (_warp(v, H) if k in ("polygon", "polyline", "line", "roi") or isinstance(v, dict) else v)
                for k, v in obj.items()}
    pts = np.float32(obj)
    if pts.shape == (4,):  # roi x1,y1,x2,y2
        c = cv2.perspectiveTransform(pts.reshape(2, 1, 2), H).reshape(-1)
        return [float(v) for v in c]
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2).tolist()


def load_scene(background_bgr=None):
    """Scene geometry in 1920x1080 frame coords of the given video background.

    Returns (scene, n_inliers). scene is None if the video does not show the reference junction.
    """
    scene = json.loads((HERE / "scene.json").read_text())
    if background_bgr is None:
        return scene, -1
    ref = cv2.imread(str(HERE / scene["ref_image"]))
    bg = cv2.resize(background_bgr, tuple(scene["ref_size"]), interpolation=cv2.INTER_AREA)
    H, n = register(ref, bg)
    if H is None:
        return None, n
    for key in ("carriageways", "median", "stop_lines", "crosswalks", "signals", "zones"):
        scene[key] = _warp(scene[key], H)
    scene["H"] = H.tolist()
    return scene, n
