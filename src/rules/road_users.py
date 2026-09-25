"""Rules for pedestrians on the road: jaywalking and failure_to_yield.

Also home of the shared helpers ``carriageway_mask`` (where the road is),
``near_polygon`` and ``union_events`` used by the other stationary/road rules.
"""
from __future__ import annotations

import numpy as np

from ..scene import points_in_polygon
from ..segments import flags_to_intervals, merge_intervals
from . import Context, Event, rule

DEFAULTS = {
    "jaywalking": {
        "road_margin": 0.15,       # foot must be this many box heights inside the road (erosion)
        "crosswalk_margin": 0.5,   # box heights around a crosswalk that still count as on it
        "min_run_sec": 1.0,        # per person, time deep on the road
        "gap_sec": 0.5,            # merge per-person runs split by short dropouts
    },
    "failure_to_yield": {
        "crosswalk_margin": 0.3,   # pedestrian box heights: "stepping on" the crossing
        "edge_frac": 0.3,          # vehicle bottom edge probed at centre +- edge_frac * box width
        "max_ped_dist": 6.0,       # pedestrian within this many vehicle heights of the vehicle foot
        "min_conflict_samples": 2, # samples with vehicle moving + pedestrian on the crossing
        "gap_sec": 0.5,
    },
}

# probe offsets (unit steps) used to erode / dilate point-in-region tests
_PROBES = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], float)


def rule_cfg(ctx: Context, label: str, defaults: dict) -> dict:
    """Module defaults overridden by ``params.classes.<label>``."""
    return {**defaults, **ctx.params.get("classes", {}).get(label, {})}


def _probe_points(pts: np.ndarray, margin) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts, float).reshape(-1, 2)
    m = np.broadcast_to(np.asarray(margin, float), (len(pts),))
    return pts, m


def carriageway_mask(ctx: Context, pts: np.ndarray, margin=0.0) -> np.ndarray:
    """True where a point is on the road, at least ``margin`` px (scalar or per point) inside it.

    Uses ``scene.carriageway`` polygons when annotated, else the flow field's
    road mask (grid cells vehicles actually drive through).
    """
    pts, m = _probe_points(pts, margin)
    if ctx.scene.carriageway:
        test = ctx.scene.on_carriageway
    else:
        min_count = float(ctx.params["flow"]["min_count"])
        test = lambda p: ctx.flow.road_mask(p, min_count)  # noqa: E731
    out = np.ones(len(pts), bool)
    for d in _PROBES:
        out &= test(pts + d * m[:, None])
    return out


def near_polygon(poly: np.ndarray, pts: np.ndarray, margin=0.0) -> np.ndarray:
    """True where a point lies inside ``poly`` dilated by ``margin`` px (probe approximation)."""
    pts, m = _probe_points(pts, margin)
    out = np.zeros(len(pts), bool)
    for d in _PROBES:
        out |= points_in_polygon(poly, pts + d * m[:, None])
    return out


def sample_index(times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the nearest sample time for each ``t``."""
    t = np.asarray(t, float)
    if len(times) < 2:
        return np.zeros(len(t), int)
    i = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    return np.where(t - times[i - 1] <= times[i] - t, i - 1, i)


def union_events(items: list[tuple[float, float, int, float]], label: str) -> list[Event]:
    """Merge overlapping per-object runs ``(start, end, track_id, score)`` into single events.

    Same-class simultaneous events count as one segment in scoring. A negative
    ``track_id`` marks a run without an object (e.g. an image blob).
    """
    out: list[Event] = []
    for s, e in merge_intervals([(a, b) for a, b, _, _ in items]):
        inside = [it for it in items if it[0] >= s and it[1] <= e]
        ids = tuple(sorted({it[2] for it in inside if it[2] >= 0}))
        out.append(Event(s, e, label, max(it[3] for it in inside), ids))
    return out


@rule("jaywalking")
def jaywalking(ctx: Context) -> list[Event]:
    """Pedestrian (not a rider) on the carriageway outside any crosswalk.

    A person counts when their foot is ``road_margin`` box heights deep in the
    road for ``min_run_sec`` (this rejects sidewalk pixels at the kerb); the
    event boundaries come from the un-eroded test so the start is the moment
    they step onto the road and the end when they leave it.
    """
    cfg = rule_cfg(ctx, "jaywalking", DEFAULTS["jaywalking"])
    items: list[tuple[float, float, int, float]] = []
    for p in ctx.pedestrians:
        in_cw = np.zeros(len(p), bool)
        for cw in ctx.scene.crosswalks:
            in_cw |= near_polygon(cw.polygon, p.foot, cfg["crosswalk_margin"] * p.height)
        loose = carriageway_mask(ctx, p.foot) & ~in_cw
        if not loose.any():
            continue
        deep = carriageway_mask(ctx, p.foot, cfg["road_margin"] * p.height) & ~in_cw
        strict = flags_to_intervals(p.t, deep, cfg["gap_sec"], cfg["min_run_sec"])
        for s, e in flags_to_intervals(p.t, loose, cfg["gap_sec"]):
            if any(a >= s and b <= e for a, b in strict):
                m = (p.t >= s) & (p.t <= e)
                items.append((s, e, p.id, float(p.conf[m].mean())))
    return union_events(items, "jaywalking")


@rule("failure_to_yield")
def failure_to_yield(ctx: Context) -> list[Event]:
    """Vehicle drives through a crosswalk while a pedestrian is on it (or stepping on).

    Event = the vehicle's stay in the crosswalk (bottom edge inside the
    polygon), kept when for ``min_conflict_samples`` samples the vehicle was
    moving and a pedestrian stood on the same crossing within ``max_ped_dist``
    vehicle heights.
    """
    if not ctx.scene.crosswalks:
        return []
    cfg = rule_cfg(ctx, "failure_to_yield", DEFAULTS["failure_to_yield"])
    moving_speed = float(ctx.params["tracks"]["moving_speed"])

    # crosswalk id -> sample index -> pedestrian feet on it
    on_cw: dict[str, dict[int, list[np.ndarray]]] = {cw.id: {} for cw in ctx.scene.crosswalks}
    for p in ctx.pedestrians:
        idx = sample_index(ctx.times, p.t)
        for cw in ctx.scene.crosswalks:
            inside = near_polygon(cw.polygon, p.foot, cfg["crosswalk_margin"] * p.height)
            for i in np.flatnonzero(inside):
                on_cw[cw.id].setdefault(int(idx[i]), []).append(p.foot[i])
    if not any(on_cw.values()):
        return []

    events: list[Event] = []
    for v in ctx.vehicles:
        idx = sample_index(ctx.times, v.t)
        half = cfg["edge_frac"] * (v.box[:, 2] - v.box[:, 0])
        edge = [v.foot + np.column_stack([k * half, np.zeros(len(v))]) for k in (-1, 0, 1)]
        moving = v.speed > moving_speed
        for cw in ctx.scene.crosswalks:
            inside = np.zeros(len(v), bool)
            for pts in edge:
                inside |= points_in_polygon(cw.polygon, pts)
            if not inside.any():
                continue
            conflict = np.zeros(len(v), bool)
            for i in np.flatnonzero(inside & moving):
                feet = on_cw[cw.id].get(int(idx[i]))
                if feet:
                    d = np.linalg.norm(np.asarray(feet) - v.foot[i], axis=1).min() / v.height[i]
                    conflict[i] = d <= cfg["max_ped_dist"]
            for s, e in flags_to_intervals(v.t, inside, cfg["gap_sec"]):
                m = (v.t >= s) & (v.t <= e)
                n = int(conflict[m].sum())
                if n >= cfg["min_conflict_samples"]:
                    events.append(Event(s, e, "failure_to_yield", float(n / m.sum()), (v.id,),
                                        {"crosswalk": cw.id, "conflict_samples": n}))
    return events
