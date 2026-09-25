"""Synthetic observations for rule tests: objects moving along waypoints.

    obs = make_obs([obj(1, CAR, [(0, 100, 500), (10, 1800, 500)])], duration=12)

Each waypoint is ``(t_sec, foot_x, foot_y)``; positions are linearly
interpolated and sampled every ``stride`` frames, like the real tracker.
"""
from __future__ import annotations

import numpy as np

from src.observe import Observations
from src.video import VideoMeta

W, H = 1920, 1080


def obj(track_id: int, cls: int, waypoints: list[tuple[float, float, float]],
        size: tuple[float, float] = (120, 80), conf: float = 0.9) -> dict:
    return {"id": track_id, "cls": cls, "wp": np.asarray(waypoints, float), "size": size, "conf": conf}


def make_obs(objects: list[dict], duration: float, fps: float = 25.0, stride: int = 2,
             width: int = W, height: int = H,
             light_features: dict[str, np.ndarray] | None = None) -> Observations:
    times = np.arange(0, int(duration * fps), stride) / fps
    rows = []
    for o in objects:
        wp = o["wp"]
        w, h = o["size"]
        mask = (times >= wp[0, 0]) & (times <= wp[-1, 0])
        ts = times[mask]
        x = np.interp(ts, wp[:, 0], wp[:, 1])
        y = np.interp(ts, wp[:, 0], wp[:, 2])
        for t, fx, fy in zip(ts, x, y):
            rows.append([o["id"], t, fx - w / 2, fy - h, fx + w / 2, fy, o["conf"], o["cls"]])
    meta = VideoMeta("synthetic.mp4", fps, int(duration * fps), width, height)
    return Observations(meta, times, np.asarray(rows, float).reshape(-1, 8), light_features or {})


def light_features_for(times: np.ndarray, schedule: list[tuple[float, str]]) -> np.ndarray:
    """ROI features for a single 'head' ROI following ``[(t_from, 'red'|'yellow'|'green'), ...]``."""
    out = np.zeros((len(times), 4), np.float32)
    for i, t in enumerate(times):
        state = [s for t0, s in schedule if t0 <= t][-1]
        out[i] = [0.8, state == "red", state == "yellow", state == "green"]
    return out
