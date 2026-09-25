"""Turning per-sample flags and raw intervals into clean event segments.

Score A matches segments by temporal IoU, so this module is where most of the
boundary accuracy is won or lost: merge fragments, drop blips, apply
per-class boundary offsets fitted on the dev labels, and never emit two
overlapping segments of the same class.
"""
from __future__ import annotations

import numpy as np

Interval = tuple[float, float]


def flags_to_intervals(times: np.ndarray, flags: np.ndarray, max_gap: float = 0.0,
                       min_len: float = 0.0) -> list[Interval]:
    """Runs of True in ``flags`` sampled at ``times`` -> [(start, end)].

    A run ends at the timestamp of its last True sample. Runs separated by at
    most ``max_gap`` seconds are merged; runs shorter than ``min_len`` dropped.
    """
    times = np.asarray(times, float)
    flags = np.asarray(flags, bool)
    if len(times) == 0 or not flags.any():
        return []
    padded = np.concatenate([[False], flags, [False]])
    d = np.diff(padded.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    runs = [(float(times[s]), float(times[e])) for s, e in zip(starts, ends)]
    return [r for r in merge_intervals(runs, max_gap) if r[1] - r[0] >= min_len]


def hysteresis(values: np.ndarray, high: float, low: float) -> np.ndarray:
    """Boolean mask that switches on at ``>= high`` and off at ``< low``."""
    out = np.zeros(len(values), bool)
    on = False
    for i, v in enumerate(values):
        if not on and v >= high:
            on = True
        elif on and v < low:
            on = False
        out[i] = on
    return out


def merge_intervals(intervals: list[Interval], gap: float = 0.0) -> list[Interval]:
    """Union of intervals, also joining ones separated by at most ``gap``."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def finalize(intervals: list[Interval], duration: float, cfg: dict) -> list[Interval]:
    """Apply per-class offsets, merge, drop short segments, clip to the video."""
    so, eo = float(cfg.get("start_offset", 0.0)), float(cfg.get("end_offset", 0.0))
    shifted = [(s + so, e + eo) for s, e in intervals]
    merged = merge_intervals(shifted, float(cfg.get("merge_gap", 0.0)))
    out = []
    for s, e in merged:
        s, e = max(0.0, s), min(duration, e)
        if e - s >= float(cfg.get("min_len", 0.0)) and e > s:
            out.append((round(s, 3), round(e, 3)))
    return out
