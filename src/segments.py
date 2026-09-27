"""Temporal cleanup shared by every hand-written event rule."""
from __future__ import annotations

from collections.abc import Iterable


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    max_gap: float = 1.0,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Merge overlaps and short gaps, then discard sub-threshold fragments."""
    valid = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not valid:
        return []
    merged: list[list[float]] = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        if start - merged[-1][1] < max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged if b - a >= min_duration]


def merge_events(
    events: Iterable[list],
    *,
    max_gap: float = 1.0,
    min_duration: float = 0.5,
    duration: float | None = None,
) -> list[list]:
    """Clean event fragments by class without joining different event types."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for event in events:
        if len(event) != 3:
            continue
        start, end, label = float(event[0]), float(event[1]), str(event[2])
        if duration is not None:
            start, end = max(0.0, start), min(float(duration), end)
        if start < end:
            by_label.setdefault(label, []).append((start, end))
    output = []
    for label, intervals in by_label.items():
        output.extend([[start, end, label] for start, end in merge_intervals(
            intervals, max_gap=max_gap, min_duration=min_duration
        )])
    return sorted(output, key=lambda event: (event[0], event[1], event[2]))
