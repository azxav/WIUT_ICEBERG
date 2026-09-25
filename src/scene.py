"""Static scene layout of the fixed camera, plus vectorised geometry helpers.

``configs/scene.json`` stores every shape in normalised image coordinates
(x / width, y / height), so the file is independent of the video resolution.
Every key is optional: a rule whose scene elements are missing simply emits
nothing. Schema (all points are ``[x, y]`` in 0..1)::

    {
      "carriageway":  [[[x, y], ...], ...],            # road polygons
      "intersection": [[[x, y], ...], ...],            # box beyond the stop lines
      "crosswalks":   [{"id": "cw_n", "polygon": [...]}],
      "lanes":        [{"id": "n_in", "polygon": [...], "direction": [dx, dy],
                        "approach": "north"}],
      "stop_lines":   [{"id": "sl_n", "line": [[x, y], [x, y]], "direction": [dx, dy],
                        "light": "tl_n", "approach": "north"}],
      "solid_lines":  [{"id": "solid_1", "polyline": [[x, y], ...]}],
      "lights":       [{"id": "tl_n", "rois": {"red": [x1, y1, x2, y2], ...}}],
      "zones":        [{"id": "A", "polygon": [...]}],   # entry/exit zones
      "allowed_movements": [["A", "C"], ...],            # [entry, exit]
      "u_turn_allowed": false
    }

``direction`` is the legal travel direction as an image-space vector. A light's
``rois`` hold either per-lamp boxes (``red``/``yellow``/``green``) or one
``head`` box around the whole signal head.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Region:
    id: str
    polygon: np.ndarray  # (K, 2) pixels


@dataclass
class Lane:
    id: str
    polygon: np.ndarray
    direction: np.ndarray  # unit vector, pixels
    approach: str = ""


@dataclass
class StopLine:
    id: str
    line: np.ndarray  # (2, 2) pixels
    direction: np.ndarray  # unit vector of legal travel across the line
    light: str | None = None
    approach: str = ""


@dataclass
class Polyline:
    id: str
    points: np.ndarray  # (K, 2)


@dataclass
class Light:
    id: str
    rois: dict[str, tuple[int, int, int, int]]  # name -> x1, y1, x2, y2 (pixels)


@dataclass
class Scene:
    width: int
    height: int
    carriageway: list[np.ndarray] = field(default_factory=list)
    intersection: list[np.ndarray] = field(default_factory=list)
    crosswalks: list[Region] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    stop_lines: list[StopLine] = field(default_factory=list)
    solid_lines: list[Polyline] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    zones: list[Region] = field(default_factory=list)
    allowed_movements: set[tuple[str, str]] | None = None
    u_turn_allowed: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, width: int, height: int) -> "Scene":
        """Load a normalised scene file; a missing file gives an empty scene."""
        path = Path(path)
        if not path.exists():
            return cls(width, height)
        return cls.from_dict(json.loads(path.read_text()), width, height)

    @classmethod
    def from_dict(cls, d: dict, width: int, height: int) -> "Scene":
        scale = np.array([width, height], dtype=float)

        def pts(p) -> np.ndarray:
            return np.asarray(p, dtype=float).reshape(-1, 2) * scale

        def unit(v) -> np.ndarray:
            v = np.asarray(v, dtype=float) * scale  # normalised -> pixel direction
            n = np.linalg.norm(v)
            return v / n if n else v

        def box(b) -> tuple[int, int, int, int]:
            x1, y1, x2, y2 = b
            return (int(x1 * width), int(y1 * height), int(round(x2 * width)), int(round(y2 * height)))

        allowed = d.get("allowed_movements")
        return cls(
            width=width,
            height=height,
            carriageway=[pts(p) for p in d.get("carriageway", [])],
            intersection=[pts(p) for p in d.get("intersection", [])],
            crosswalks=[Region(c["id"], pts(c["polygon"])) for c in d.get("crosswalks", [])],
            lanes=[Lane(l["id"], pts(l["polygon"]), unit(l["direction"]), l.get("approach", ""))
                   for l in d.get("lanes", [])],
            stop_lines=[StopLine(s["id"], pts(s["line"]), unit(s["direction"]), s.get("light"),
                                 s.get("approach", "")) for s in d.get("stop_lines", [])],
            solid_lines=[Polyline(s["id"], pts(s["polyline"])) for s in d.get("solid_lines", [])],
            lights=[Light(l["id"], {k: box(v) for k, v in l["rois"].items()}) for l in d.get("lights", [])],
            zones=[Region(z["id"], pts(z["polygon"])) for z in d.get("zones", [])],
            allowed_movements={tuple(m) for m in allowed} if allowed is not None else None,
            u_turn_allowed=bool(d.get("u_turn_allowed", False)),
        )

    # ------------------------------------------------------------ queries
    def on_carriageway(self, pts: np.ndarray) -> np.ndarray:
        return any_polygon(self.carriageway, pts)

    def in_intersection(self, pts: np.ndarray) -> np.ndarray:
        return any_polygon(self.intersection, pts)

    def crosswalk_of(self, pts: np.ndarray) -> list[str | None]:
        """Id of the crosswalk containing each point, or None."""
        out: list[str | None] = [None] * len(pts)
        for cw in self.crosswalks:
            inside = points_in_polygon(cw.polygon, pts)
            for i in np.flatnonzero(inside):
                out[i] = out[i] or cw.id
        return out

    def lane_of(self, pts: np.ndarray) -> list[Lane | None]:
        out: list[Lane | None] = [None] * len(pts)
        for lane in self.lanes:
            inside = points_in_polygon(lane.polygon, pts)
            for i in np.flatnonzero(inside):
                out[i] = out[i] or lane
        return out

    def zone_of(self, pts: np.ndarray) -> list[str | None]:
        out: list[str | None] = [None] * len(pts)
        for z in self.zones:
            inside = points_in_polygon(z.polygon, pts)
            for i in np.flatnonzero(inside):
                out[i] = out[i] or z.id
        return out

    def light(self, light_id: str | None) -> Light | None:
        return next((l for l in self.lights if l.id == light_id), None)


# ---------------------------------------------------------------- geometry
def points_in_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. ``poly`` (K, 2), ``pts`` (N, 2) -> bool (N,)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(poly) < 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddle = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hits = straddle & (x < x_cross)
    return (hits.sum(axis=1) % 2).astype(bool)


def any_polygon(polys: list[np.ndarray], pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = np.zeros(len(pts), dtype=bool)
    for p in polys:
        out |= points_in_polygon(p, pts)
    return out


def side_of_line(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed cross product: >0 on the left of ``line[0] -> line[1]``, <0 on the right."""
    a, b = line[0], line[1]
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])


def segment_crossings(path: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[tuple[int, float]]:
    """Where the polyline ``path`` (N, 2) crosses segment ``a-b``.

    Returns ``(i, u)`` pairs: the crossing happens between ``path[i]`` and
    ``path[i+1]`` at fraction ``u`` in [0, 1] of that step.
    """
    if len(path) < 2:
        return []
    p, r = path[:-1], path[1:] - path[:-1]
    s = b - a
    denom = r[:, 0] * s[1] - r[:, 1] * s[0]
    qp = a - p
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (qp[:, 0] * s[1] - qp[:, 1] * s[0]) / denom        # along the path step
        v = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / denom  # along the segment
    ok = (denom != 0) & (u >= 0) & (u < 1) & (v >= 0) & (v <= 1)
    return [(int(i), float(u[i])) for i in np.flatnonzero(ok)]


def polyline_crossings(path: np.ndarray, polyline: np.ndarray) -> list[tuple[int, float]]:
    """Crossings of ``path`` with any segment of ``polyline``, sorted by time."""
    hits: list[tuple[int, float]] = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        hits += segment_crossings(path, a, b)
    return sorted(set(hits))
