"""Map tracked detections to smoothed motion and scene features."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


VEHICLE_CLASSES = {1, 2, 3, 5, 7}


def _inside(point: tuple[float, float], polygon: Sequence, margin: float = 0.0) -> bool:
    """Single-point polygon check used by event rules."""
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return False
    x, y = point
    inside = False
    minimum = float("inf")
    for i in range(len(points)):
        ax, ay = points[i - 1]
        bx, by = points[i]
        if ((ay > y) != (by > y)) and x < (bx - ax) * (y - ay) / (by - ay + 1e-12) + ax:
            inside = not inside
        if margin > 0:
            abx, aby = float(bx - ax), float(by - ay)
            denom = abx * abx + aby * aby
            u = 0.0 if denom == 0 else max(0.0, min(1.0, ((x-ax)*abx + (y-ay)*aby) / denom))
            dx, dy = x - (ax + u*abx), y - (ay + u*aby)
            minimum = min(minimum, dx*dx + dy*dy)
    return inside or (margin > 0 and minimum <= margin * margin)


def _inside_many(points: np.ndarray, polygon: Sequence, margin: float = 0.0) -> np.ndarray:
    """Vectorized polygon membership, with an optional edge-distance margin."""
    vertices = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(vertices) < 3 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    min_d2 = np.full(len(points), np.inf, dtype=float) if margin > 0 else None
    previous = vertices[-1]
    for current in vertices:
        ax, ay = previous
        bx, by = current
        crosses = (ay > y) != (by > y)
        x_at = (bx - ax) * (y - ay) / (by - ay + 1e-12) + ax
        inside ^= crosses & (x < x_at)
        if min_d2 is not None:
            ab = current - previous
            denom = float(np.dot(ab, ab))
            if denom == 0:
                delta = points - previous
            else:
                u = np.clip(((points - previous) @ ab) / denom, 0.0, 1.0)
                delta = points - (previous + u[:, None] * ab)
            min_d2 = np.minimum(min_d2, np.einsum("ij,ij->i", delta, delta))
        previous = current
    return inside if min_d2 is None else inside | (min_d2 <= margin * margin)


def signed_distance_to_polyline(
    point: tuple[float, float], polyline: Sequence, direction: Sequence[float] | None = None
) -> float:
    """Signed distance to the closest polyline segment."""
    points = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return math.nan
    p = np.asarray(point, dtype=float)
    best_q, best_d2, best_tangent = None, float("inf"), None
    for a, b in zip(points[:-1], points[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        q = a if denom == 0 else a + np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0) * ab
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_d2:
            best_q, best_d2, best_tangent = q, d2, ab
    delta = p - best_q
    if direction is not None:
        unit = np.asarray(direction, dtype=float)
        unit /= max(float(np.linalg.norm(unit)), 1e-9)
        return float(np.dot(delta, unit))
    cross = best_tangent[0] * delta[1] - best_tangent[1] * delta[0]
    return float(np.sign(cross) * np.sqrt(best_d2))


def _signed_distances_many(points: np.ndarray, polyline: Sequence, direction: Sequence[float]) -> np.ndarray:
    vertices = np.asarray(polyline, dtype=float).reshape(-1, 2)
    if len(vertices) < 2:
        return np.full(len(points), np.nan)
    best_d2 = np.full(len(points), np.inf, dtype=float)
    best_q = np.zeros_like(points, dtype=float)
    for a, b in zip(vertices[:-1], vertices[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        u = np.zeros(len(points)) if denom == 0 else np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
        q = a + u[:, None] * ab
        delta = points - q
        d2 = np.einsum("ij,ij->i", delta, delta)
        choose = d2 < best_d2
        best_d2[choose] = d2[choose]
        best_q[choose] = q[choose]
    unit = np.asarray(direction, dtype=float)
    unit /= max(float(np.linalg.norm(unit)), 1e-9)
    return (points - best_q) @ unit


def _polygon_groups(scene: Mapping, key: str) -> dict[str, list]:
    return {name: shape["polygon"] for name, shape in scene.get(key, {}).items() if "polygon" in shape}


def _lane_ids(points: np.ndarray, scene: Mapping, roads: dict[str, np.ndarray]) -> np.ndarray:
    lanes = np.full(len(points), None, dtype=object)
    for name, lane in scene.get("lanes", {}).items():
        polygon = lane.get("polygon", [])
        hit = _inside_many(points, polygon)
        lanes[hit & pd.isna(lanes)] = name

    far_mask = roads.get("far", np.zeros(len(points), dtype=bool)) & pd.isna(lanes)
    if np.any(far_mask):
        direction = np.asarray(scene.get("carriageways", {}).get("far", {}).get("direction", [-0.92, -0.39]), dtype=float)
        lateral = direction[0] * points[:, 1] - direction[1] * points[:, 0]
        boundaries = []
        for name, shape in scene.get("dashed_lines", {}).items():
            if not name.startswith("far_lane_"):
                continue
            vertices = np.asarray(shape.get("polyline", []), dtype=float).reshape(-1, 2)
            if len(vertices):
                boundaries.append(float(np.median(direction[0] * vertices[:, 1] - direction[1] * vertices[:, 0])))
        boundaries.sort()
        if len(boundaries) == 3:
            indices = np.sum(lateral[far_mask, None] > np.asarray(boundaries)[None, :], axis=1)
            lanes[far_mask] = [f"far_through_{int(index)+1}" for index in indices]
    return lanes


def extract_features(
    tracks: pd.DataFrame,
    scene: Mapping,
    source_size: tuple[int, int],
    smooth_window: int = 3,
) -> pd.DataFrame:
    """Add map-space positions, speed, heading, lane and zone membership.

    `source_size` is `(width, height)` for the original video. Scene geometry is
    authored in the scene's reference size (normally 1920x1080).
    """
    feature_columns = [
        "cx", "cy", "foot_x", "foot_y", "front_x", "front_y", "rear_x", "rear_y", "speed",
        "heading_rad", "lane_id", "road_id", "crosswalk_id", "in_queue",
        "in_junction", "in_road", "stopline_signed", "front_stopline_signed", "scale_x", "scale_y",
    ]
    if tracks.empty:
        return pd.DataFrame(columns=list(dict.fromkeys([*tracks.columns, *feature_columns])))

    width, height = source_size
    ref_width, ref_height = scene.get("ref_size", [1920, 1080])
    scale_x, scale_y = float(ref_width) / width, float(ref_height) / height
    out = tracks.copy().sort_values(["id", "t"], kind="stable").reset_index(drop=True)
    out["scale_x"], out["scale_y"] = scale_x, scale_y
    out["cx"] = (out["x1"] + out["x2"]) * 0.5 * scale_x
    out["cy"] = (out["y1"] + out["y2"]) * 0.5 * scale_y
    out["foot_x"], out["foot_y"] = out["cx"], out["y2"] * scale_y
    foot_points = np.column_stack([out["foot_x"].to_numpy(), out["foot_y"].to_numpy()])

    road_polys = _polygon_groups(scene, "carriageways")
    road_masks = {name: _inside_many(foot_points, polygon, margin=8.0) for name, polygon in road_polys.items()}
    road_ids = np.full(len(out), None, dtype=object)
    for name, mask in road_masks.items():
        road_ids[mask & pd.isna(road_ids)] = name
    out["road_id"] = road_ids
    out["lane_id"] = _lane_ids(foot_points, scene, road_masks)

    crosswalk_ids = np.full(len(out), None, dtype=object)
    for name, polygon in _polygon_groups(scene, "crosswalks").items():
        hit = _inside_many(foot_points, polygon, margin=10.0)
        crosswalk_ids[hit & pd.isna(crosswalk_ids)] = name
    out["crosswalk_id"] = crosswalk_ids

    queue_poly = scene.get("zones", {}).get("queue_near", {}).get("polygon", [])
    junction_poly = scene.get("zones", {}).get("junction", {}).get("polygon", [])
    queue_mask = _inside_many(foot_points, queue_poly, margin=12.0)
    junction_mask = _inside_many(foot_points, junction_poly, margin=8.0)
    island_mask = np.zeros(len(out), dtype=bool)
    for name, shape in scene.get("curbs", {}).items():
        if name.startswith("island_") and len(shape.get("polyline", [])) >= 3:
            island_mask |= _inside_many(foot_points, shape["polyline"], margin=8.0)
    out["in_queue"] = queue_mask
    out["in_junction"] = junction_mask
    # The junction polygon includes sidewalks, refuge areas and signal
    # islands. Use mapped carriageway polygons for road-user violations so
    # pedestrians waiting at the curb are not labelled as jaywalking.
    out["in_road"] = pd.notna(out["road_id"]).to_numpy() & ~island_mask

    stop_line = scene.get("stop_lines", {}).get("near", {}).get("line", [])
    direction_near = scene.get("carriageways", {}).get("near", {}).get("direction", [0.92, 0.39])
    out["stopline_signed"] = _signed_distances_many(foot_points, stop_line, direction_near)

    cls = out["cls"].to_numpy(dtype=int)
    box_w = (out["x2"].to_numpy(dtype=float) - out["x1"].to_numpy(dtype=float)) * scale_x
    box_h = (out["y2"].to_numpy(dtype=float) - out["y1"].to_numpy(dtype=float)) * scale_y
    near = np.asarray([not (isinstance(lane, str) and lane.startswith("far_")) and road != "far"
                       for lane, road in zip(out["lane_id"], out["road_id"])])
    near_dir = np.asarray(direction_near, dtype=float)
    far_dir = np.asarray(scene.get("carriageways", {}).get("far", {}).get("direction", [-0.92, -0.39]), dtype=float)
    directions = np.where(near[:, None], near_dir[None, :], far_dir[None, :])
    norms = np.maximum(np.linalg.norm(directions, axis=1), 1e-9)
    directions = directions / norms[:, None]
    half_extent = 0.5 * (np.abs(directions[:, 0]) * box_w + np.abs(directions[:, 1]) * box_h)
    vehicle = np.isin(cls, list(VEHICLE_CLASSES))
    directions[~vehicle] = 0.0
    half_extent[~vehicle] = 0.0
    out["front_x"] = out["foot_x"].to_numpy() + directions[:, 0] * half_extent
    out["front_y"] = out["foot_y"].to_numpy() + directions[:, 1] * half_extent
    out["rear_x"] = out["foot_x"].to_numpy() - directions[:, 0] * half_extent
    out["rear_y"] = out["foot_y"].to_numpy() - directions[:, 1] * half_extent
    out["front_stopline_signed"] = _signed_distances_many(
        np.column_stack([out["front_x"].to_numpy(), out["front_y"].to_numpy()]),
        stop_line,
        direction_near,
    )

    grouped = out.groupby("id", sort=False)
    previous_t, next_t = grouped["t"].shift(1), grouped["t"].shift(-1)
    previous_x, next_x = grouped["cx"].shift(1), grouped["cx"].shift(-1)
    previous_y, next_y = grouped["cy"].shift(1), grouped["cy"].shift(-1)
    current_t = out["t"].to_numpy(dtype=float)
    current_x, current_y = out["cx"].to_numpy(dtype=float), out["cy"].to_numpy(dtype=float)
    px, py, pt = previous_x.to_numpy(), previous_y.to_numpy(), previous_t.to_numpy()
    nx, ny, nt = next_x.to_numpy(), next_y.to_numpy(), next_t.to_numpy()
    dx = np.where(np.isfinite(px) & np.isfinite(nx), nx - px, np.where(np.isfinite(nx), nx-current_x, current_x-px))
    dy = np.where(np.isfinite(py) & np.isfinite(ny), ny - py, np.where(np.isfinite(ny), ny-current_y, current_y-py))
    dt = np.where(np.isfinite(pt) & np.isfinite(nt), nt - pt, np.where(np.isfinite(nt), nt-current_t, current_t-pt))
    dt = np.maximum(np.nan_to_num(dt, nan=0.1), 1e-3)
    out["speed"] = np.nan_to_num(np.hypot(dx, dy) / dt, nan=0.0, posinf=0.0)
    out["heading_rad"] = np.nan_to_num(np.arctan2(dy, dx), nan=0.0)
    if smooth_window > 1:
        out["speed"] = out.groupby("id", sort=False)["speed"].transform(
            lambda series: series.rolling(int(smooth_window), center=True, min_periods=1).median()
        )
    return out.sort_values(["t", "id"], kind="stable").reset_index(drop=True)
