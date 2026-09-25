"""Image-plane surrogate safety measures: 2D time-to-collision, closing speed, deceleration.

Objects are axis-aligned boxes moving at constant velocity in the image. Time
to collision is unit free, so pixels work as well as metres; closing speeds
and decelerations are divided by box height by the callers (the same
perspective correction as ``src.tracks``).

The 2D TTC below follows the idea of Yiru Jiao's "Two-Dimensional
Time-To-Collision" (TU Delft, github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision):
TTC is the first time two moving footprints overlap under constant relative
velocity, not a 1D gap / speed ratio. Jiao works with rotated rectangles in
world coordinates; here we use image-plane AABBs, for which the "swept box"
slab test gives a closed form. All functions broadcast over leading axes, so
a (N, 1, 2) vs (1, N, 2) call evaluates every pair at once.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def footprint(xyxy: np.ndarray, frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Ground-contact part of boxes: the bottom ``frac`` of each box -> (centre, half size).

    Seen from a pole camera, two vehicles at different depths overlap in the
    image while their ground footprints (bottom strip of the box) do not,
    which removes most occlusion-induced "contacts".
    """
    b = np.asarray(xyxy, float)
    w = b[..., 2] - b[..., 0]
    h = (b[..., 3] - b[..., 1]) * frac
    centre = np.stack([(b[..., 0] + b[..., 2]) / 2, b[..., 3] - h / 2], axis=-1)
    return centre, np.stack([w / 2, h / 2], axis=-1)


def box_gap(c1: np.ndarray, h1: np.ndarray, c2: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Chebyshev gap between AABBs (centre, half size): > 0 apart, <= 0 overlapping."""
    sep = np.abs(np.asarray(c2, float) - c1) - (np.asarray(h1, float) + h2)
    return sep.max(axis=-1)


def ttc_boxes(c1: np.ndarray, h1: np.ndarray, v1: np.ndarray,
              c2: np.ndarray, h2: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Time until two constant-velocity boxes first overlap (0 if they already do, inf if never).

    Per axis the boxes overlap while ``|d + v t| <= w`` (d relative centre,
    v relative velocity, w sum of half sizes); the collision time is the start
    of the intersection of the two per-axis intervals.
    """
    d = np.asarray(c2, float) - np.asarray(c1, float)
    v = np.asarray(v2, float) - np.asarray(v1, float)
    w = np.asarray(h1, float) + np.asarray(h2, float)
    d, v, w = np.broadcast_arrays(d, v, w)
    moving = np.abs(v) > _EPS
    safe_v = np.where(moving, v, 1.0)
    ta, tb = (-w - d) / safe_v, (w - d) / safe_v
    inside = np.abs(d) <= w
    # a still axis overlaps forever or never
    t_in = np.where(moving, np.minimum(ta, tb), np.where(inside, -np.inf, np.inf))
    t_out = np.where(moving, np.maximum(ta, tb), np.where(inside, np.inf, -np.inf))
    enter, leave = t_in.max(axis=-1), t_out.min(axis=-1)
    hit = (enter <= leave) & (leave >= 0)
    return np.where(hit, np.maximum(enter, 0.0), np.inf)


def closing_speed(c1: np.ndarray, v1: np.ndarray, c2: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Rate at which the centre distance shrinks (> 0 approaching), same units as the velocities."""
    d = np.asarray(c2, float) - np.asarray(c1, float)
    v = np.asarray(v2, float) - np.asarray(v1, float)
    dist = np.linalg.norm(d, axis=-1)
    return np.where(dist > _EPS, -(d * v).sum(axis=-1) / np.maximum(dist, _EPS), 0.0)


def pairwise_ttc(centre: np.ndarray, half: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """(N, N) TTC between every pair of N boxes; inf on the diagonal."""
    c, h, v = (np.asarray(a, float) for a in (centre, half, vel))
    out = ttc_boxes(c[:, None], h[:, None], v[:, None], c[None], h[None], v[None])
    np.fill_diagonal(out, np.inf)
    return out


def pairwise_closing(centre: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """(N, N) closing speed between every pair; 0 on the diagonal."""
    c, v = np.asarray(centre, float), np.asarray(vel, float)
    return closing_speed(c[:, None], v[:, None], c[None], v[None])


def speed_drop(t: np.ndarray, speed: np.ndarray, window: float) -> np.ndarray:
    """Causal speed drop: max speed over ``[t - window, t]`` minus the current speed.

    With speeds in box heights / s, ``speed_drop / window`` is a deceleration
    in heights / s^2. NaN speeds are ignored.
    """
    t, speed = np.asarray(t, float), np.asarray(speed, float)
    lo = np.searchsorted(t, t - window, side="left")
    out = np.zeros(len(t))
    for i, j in enumerate(lo):
        seg = speed[j:i + 1]
        if np.isfinite(seg).any() and np.isfinite(speed[i]):
            out[i] = np.nanmax(seg) - speed[i]
    return out


def heading_change(t: np.ndarray, heading: np.ndarray, window: float) -> np.ndarray:
    """|heading(t) - heading(t - window)| wrapped to [0, pi]; NaN when either end is NaN (slow)."""
    t, heading = np.asarray(t, float), np.asarray(heading, float)
    j = np.clip(np.searchsorted(t, t - window, side="left"), 0, max(len(t) - 1, 0))
    diff = heading - heading[j]
    return np.abs((diff + np.pi) % (2 * np.pi) - np.pi)
