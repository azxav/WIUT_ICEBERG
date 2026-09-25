"""OpenCV drawing shared by the dev scripts: scene overlay, labels, event timeline."""
from __future__ import annotations

import cv2
import numpy as np

import _common  # noqa: F401  (puts the repo root on sys.path)
from evaluate import OFFICIAL_CLASSES
from src.scene import Scene

# BGR colours, one per official class (stable across scripts and renders)
CLASS_COLORS: dict[str, tuple[int, int, int]] = dict(zip(OFFICIAL_CLASSES, [
    (40, 40, 230),    # accident
    (40, 140, 255),   # near_miss
    (80, 80, 255),    # red_light
    (200, 60, 200),   # wrong_way
    (255, 120, 60),   # illegal_u_turn
    (0, 200, 255),    # stopped_vehicle
    (80, 220, 80),    # jaywalking
    (170, 230, 60),   # failure_to_yield
    (255, 200, 80),   # illegal_turn
    (230, 230, 230),  # solid_line_crossing
    (60, 60, 170),    # stop_line
    (0, 140, 200),    # congestion
    (140, 100, 60),   # road_obstacle
    (30, 90, 255),    # fire_smoke
]))
CATEGORY_COLORS = {"vehicle": (255, 180, 60), "person": (60, 220, 60), "bicycle": (0, 210, 255),
                   "animal": (220, 90, 255), "other": (170, 170, 170)}
LIGHT_COLORS = {"red": (0, 0, 255), "yellow": (0, 215, 255), "green": (0, 200, 0), "unknown": (140, 140, 140)}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pts(p: np.ndarray, s: float) -> np.ndarray:
    return np.round(np.asarray(p, float) * s).astype(np.int32)


def text(img: np.ndarray, s: str, org: tuple[int, int], color=(255, 255, 255), scale: float = 0.5,
         thick: int = 1, bg: tuple[int, int, int] | None = (0, 0, 0)) -> int:
    """Text with a filled background box; returns the box width."""
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return w + 4


def arrow(img: np.ndarray, origin, direction, length: float, color, thick: int = 2) -> None:
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n == 0:
        return
    tip = o + d / n * length
    cv2.arrowedLine(img, tuple(map(int, o)), tuple(map(int, tip)), color, thick, cv2.LINE_AA, tipLength=0.3)


def draw_scene(img: np.ndarray, scene: Scene, s: float = 1.0, alpha: float = 0.22,
               labels: bool = True) -> np.ndarray:
    """Scene layout over ``img`` (scene in video pixels, ``img`` scaled by ``s``)."""
    th = max(1, int(round(img.shape[1] / 900)))
    fs = 0.4 * max(1.0, img.shape[1] / 1280)
    over = img.copy()
    for p in scene.carriageway:
        cv2.fillPoly(over, [_pts(p, s)], (120, 120, 120))
    for p in scene.intersection:
        cv2.fillPoly(over, [_pts(p, s)], (0, 180, 220))
    for cw in scene.crosswalks:
        cv2.fillPoly(over, [_pts(cw.polygon, s)], (255, 255, 255))
    out = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)

    for p in scene.carriageway:
        cv2.polylines(out, [_pts(p, s)], True, (160, 160, 160), th, cv2.LINE_AA)
    for p in scene.intersection:
        cv2.polylines(out, [_pts(p, s)], True, (0, 200, 240), th, cv2.LINE_AA)
    for cw in scene.crosswalks:
        q = _pts(cw.polygon, s)
        cv2.polylines(out, [q], True, (255, 255, 255), th, cv2.LINE_AA)
        if labels:
            text(out, cw.id, q.min(axis=0), (255, 255, 255), fs)
    for z in scene.zones:
        q = _pts(z.polygon, s)
        cv2.polylines(out, [q], True, (255, 120, 255), th, cv2.LINE_AA)
        if labels:
            text(out, f"zone {z.id}", q.mean(axis=0).astype(int), (255, 120, 255), fs)
    for lane in scene.lanes:
        q = _pts(lane.polygon, s)
        cv2.polylines(out, [q], True, (255, 200, 0), th, cv2.LINE_AA)
        c = q.mean(axis=0)
        span = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))
        arrow(out, c, lane.direction, float(np.clip(0.25 * span, 20.0, 0.06 * img.shape[1])), (255, 200, 0), th + 1)
        if labels:
            text(out, lane.id, c.astype(int) + np.array([5, -5]), (255, 200, 0), fs)
    for sl in scene.stop_lines:
        a, b = _pts(sl.line, s)
        cv2.line(out, tuple(a), tuple(b), (0, 0, 255), th + 2, cv2.LINE_AA)
        mid = (a + b) / 2
        arrow(out, mid, sl.direction, 30 * max(1.0, s * 2), (0, 0, 255), th + 1)
        if labels:
            text(out, f"{sl.id}" + (f" <{sl.light}>" if sl.light else ""), mid.astype(int) + np.array([6, 16]),
                 (80, 80, 255), fs)
    for sl in scene.solid_lines:
        cv2.polylines(out, [_pts(sl.points, s)], False, (0, 255, 255), th + 1, cv2.LINE_AA)
    for light in scene.lights:
        for name, (x1, y1, x2, y2) in light.rois.items():
            p1, p2 = _pts([x1, y1], s), _pts([x2, y2], s)
            cv2.rectangle(out, tuple(p1), tuple(p2), (0, 255, 0), th)
            if labels:
                text(out, f"{light.id}:{name}", (p1[0], p1[1] - 4), (0, 255, 0), fs * 0.9)
    return out


def draw_light_states(img: np.ndarray, scene: Scene, states: dict[str, str], s: float = 1.0) -> None:
    """A filled dot next to each light's ROIs in the colour of its current state."""
    r = max(5, int(round(img.shape[1] / 160)))
    for light in scene.lights:
        if not light.rois:
            continue
        boxes = np.array(list(light.rois.values()), float)
        x, y = _pts([boxes[:, 2].max(), boxes[:, 1].min()], s)
        color = LIGHT_COLORS.get(states.get(light.id, "unknown"), LIGHT_COLORS["unknown"])
        cv2.circle(img, (int(x) + r + 4, int(y) + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x) + r + 4, int(y) + r), r, (0, 0, 0), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ timeline
LABEL_W = 150


def timeline(width: int, duration: float, rows: list[dict], row_h: int = 14) -> np.ndarray:
    """Event strip: one row per class with ``pred`` (filled) and optional ``gt`` (white) intervals.

    ``rows`` items: ``{"name": str, "color": bgr, "pred": [(s, e)], "gt": [(s, e)] | None}``.
    A 12 px time axis sits at the bottom. Use ``draw_playhead`` for the cursor.
    """
    axis_h = 14
    h = max(1, len(rows)) * row_h + axis_h
    img = np.full((h, width, 3), 24, np.uint8)
    span = max(1, width - LABEL_W - 4)

    def x(t: float) -> int:
        return LABEL_W + int(round(t / max(duration, 1e-6) * span))

    for k, row in enumerate(rows):
        y0 = k * row_h
        if k % 2:
            img[y0:y0 + row_h] = 32
        text(img, row["name"][:20], (4, y0 + row_h - 3), row["color"], 0.38, 1, None)
        gt = row.get("gt")
        ph = row_h - 2 if gt is None else int(row_h * 0.55)
        for s_, e_ in row.get("pred", []):
            cv2.rectangle(img, (x(s_), y0 + 1), (max(x(s_) + 1, x(e_)), y0 + ph), row["color"], -1)
        for s_, e_ in gt or []:
            cv2.rectangle(img, (x(s_), y0 + ph + 1), (max(x(s_) + 1, x(e_)), y0 + row_h - 2), (235, 235, 235), -1)
    y_axis = h - axis_h
    step = next((c for c in (5, 10, 15, 30, 60, 120, 300, 600) if duration / c <= 12), 1200)
    for t in np.arange(0, duration + 1e-6, step):
        cv2.line(img, (x(t), y_axis), (x(t), y_axis + 4), (160, 160, 160), 1)
        text(img, f"{int(t // 60)}:{int(t % 60):02d}", (x(t) + 2, h - 2), (160, 160, 160), 0.33, 1, None)
    return img


def draw_playhead(strip: np.ndarray, t: float, duration: float, color=(255, 255, 255)) -> None:
    span = max(1, strip.shape[1] - LABEL_W - 4)
    x = LABEL_W + int(round(t / max(duration, 1e-6) * span))
    cv2.line(strip, (x, 0), (x, strip.shape[0] - 1), color, 1)


def timeline_time(x: int, width: int, duration: float) -> float:
    """Inverse of the timeline x mapping (for click-to-seek)."""
    span = max(1, width - LABEL_W - 4)
    return float(np.clip((x - LABEL_W) / span, 0.0, 1.0) * duration)
