"""HSV traffic-signal state sampling for the mapped near-approach signal."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .tracking import sampled_frames


SIGNAL_COLUMNS = ("frame", "t", "state", "red", "amber", "green")


def count_signal_colors(crop: np.ndarray) -> dict[str, int]:
    """Count saturated, illuminated red/amber/green pixels in a signal crop."""
    if crop is None or crop.size == 0:
        return {"R": 0, "Y": 0, "G": 0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lit = (saturation >= 50) & (value >= 60)
    masks = {
        "R": ((hue <= 12) | (hue >= 168)) & lit,
        "Y": (hue >= 13) & (hue <= 35) & lit,
        "G": (hue >= 38) & (hue <= 100) & lit,
    }
    return {name: int(mask.sum()) for name, mask in masks.items()}


def classify_counts(
    counts: dict[str, int],
    min_pixels: int = 5,
    dominance: float = 1.5,
) -> str:
    """Return R/Y/G when one lamp colour clearly dominates, else '?'."""
    ranked = sorted(((int(counts.get(state, 0)), state) for state in ("R", "Y", "G")), reverse=True)
    best, state = ranked[0]
    second = ranked[1][0]
    if best < min_pixels or (second > 0 and best < second * dominance):
        return "?"
    return state


def classify_signal(
    crop: np.ndarray,
    min_pixels: int = 5,
    dominance: float = 1.5,
) -> tuple[str, dict[str, int]]:
    """Classify one BGR ROI and return its raw color counts for diagnostics."""
    counts = count_signal_colors(crop)
    return classify_counts(counts, min_pixels=min_pixels, dominance=dominance), counts


def extract_signal_states(
    video_path: str | Path,
    roi: tuple[float, float, float, float] | list[float],
    sample_stride: int = 3,
    scene_size: tuple[int, int] = (1920, 1080),
    min_pixels: int = 5,
    dominance: float = 1.5,
) -> pd.DataFrame:
    """Sample a mapped signal ROI on the detector stride and return states."""
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if scene_size[0] < 1 or scene_size[1] < 1:
        raise ValueError("scene_size must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"video has invalid FPS: {video_path}")

    x1, y1, x2, y2 = (float(value) for value in roi)
    rows: list[dict[str, int | float | str]] = []
    try:
        for frame_index, frame in sampled_frames(capture, sample_stride):
            source_height, source_width = frame.shape[:2]
            scale_x = source_width / scene_size[0]
            scale_y = source_height / scene_size[1]
            crop = frame[
                max(0, round(y1 * scale_y)):min(source_height, round(y2 * scale_y)),
                max(0, round(x1 * scale_x)):min(source_width, round(x2 * scale_x)),
            ]
            state, counts = classify_signal(crop, min_pixels=min_pixels, dominance=dominance)
            rows.append({
                "frame": frame_index,
                "t": frame_index / fps,
                "state": state,
                "red": counts["R"],
                "amber": counts["Y"],
                "green": counts["G"],
            })
    finally:
        capture.release()

    return pd.DataFrame.from_records(rows, columns=SIGNAL_COLUMNS)
