"""Shared helpers: background loading, paint detection, registration, warping."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene import register  # noqa: E402

VIDEOS = ["C3896", "C3897", "C3902", "C3905"]
REF_IMAGE = ROOT / "src" / "scene_ref.jpg"


def bg_path(video):
    return ROOT / "eda" / "frames" / f"{video}_background.jpg"


def load_bg(video):
    return cv2.imread(str(bg_path(video)))


def load_ref():
    return cv2.imread(str(REF_IMAGE))


def to_ref_H(video):
    """Homography video-background -> reference (1920x1080)."""
    ref = load_ref()
    bg = load_bg(video)
    H, n = register(ref, bg)
    if H is None:
        raise RuntimeError(f"registration failed for {video}: {n} inliers")
    return np.linalg.inv(H), n


def paint_mask(bgr, tophat_thr=22, sat_max=130):
    """Binary mask (255) of white/yellow road paint, robust to the dark evening frames."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    m = ((tophat > tophat_thr) & (hsv[..., 1] < sat_max)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m


def warp_to_ref(mask, video):
    H, _ = to_ref_H(video)
    ref = load_ref()
    h, w = ref.shape[:2]
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)


def thin_paint_mask(bgr, thick_kernel=31, tophat_thr=22, sat_max=130, min_area=25):
    """Paint mask that keeps only thin (stripe-like) structures: large bright blobs are dropped.

    Removes vehicles and over-exposed pavement, keeps lane paint and zebra stripes.
    """
    m = paint_mask(bgr, tophat_thr, sat_max)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick_kernel,) * 2)
    thick = cv2.erode(m, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    bad = np.zeros(n, bool)
    ys, xs = np.where(thick > 0)
    if len(ys):
        bad[np.unique(lab[ys, xs])] = True
    bad[0] = False
    out = m.copy()
    out[bad[lab]] = 0
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(out, 8)
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] < min_area:
            out[lab2 == i] = 0
    return out


def poly_mask(shape, poly, dilate=0):
    m = np.zeros(shape[:2], np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(m, [np.int32(poly)], 255)
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
        m = cv2.dilate(m, k)
    return m


def band_mask(shape, pts, width):
    m = np.zeros(shape[:2], np.uint8)
    cv2.polylines(m, [np.int32(pts)], False, 255, max(1, int(width * 2)))
    return m
