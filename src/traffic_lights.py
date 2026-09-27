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


def classify_vehicle_signal(
    crop: np.ndarray,
    min_pixels: int = 5,
    dominance: float = 2.0,
) -> tuple[str, dict[str, int]]:
    """Classify a vertical three-lamp vehicle signal by lamp position.

    The mapped ROI contains all three lamps. Position matters: amber can be
    red-orange in HSV, so a whole-ROI colour vote confuses the two phases.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 3:
        return "?", {"R": 0, "Y": 0, "G": 0}
    top, middle, bottom = np.array_split(crop, 3, axis=0)
    red = count_signal_colors(top)["R"]
    amber = count_signal_colors(middle)["Y"]
    green = count_signal_colors(bottom)["G"]
    counts = {"R": red, "Y": amber, "G": green}
    red_min = max(1, min(min_pixels, int(np.ceil(top.size * 0.01))))
    amber_min = max(1, min(min_pixels, int(np.ceil(middle.size * 0.01))))
    green_min = max(1, min(min_pixels, int(np.ceil(bottom.size * 0.01))))
    minima = {"R": red_min, "Y": amber_min, "G": green_min}
    ranked = sorted(((counts[state], state) for state in ("R", "Y", "G")), reverse=True)
    best, state = ranked[0]
    second = ranked[1][0]
    # Road users and reflections can place red/green pixels inside an adjacent
    # lamp slice. Accept a phase only when its own lamp dominates; mixed crops
    # stay unknown instead of turning a green phase into a red-light event.
    if best < minima[state] or (second >= min(minima.values()) and best < second * dominance):
        return "?", counts
    return state, counts


def classify_vehicle_signal_frame(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | list[float],
    scene_size: tuple[int, int] = (1920, 1080),
    min_pixels: int = 5,
) -> tuple[str, dict[str, int]]:
    """Crop a scene-map ROI from a source frame and classify its three lamps."""
    source_height, source_width = frame.shape[:2]
    scale_x = source_width / float(scene_size[0])
    scale_y = source_height / float(scene_size[1])
    x1, y1, x2, y2 = (float(value) for value in roi)
    crop = frame[
        max(0, round(y1 * scale_y)):min(source_height, round(y2 * scale_y)),
        max(0, round(x1 * scale_x)):min(source_width, round(x2 * scale_x)),
    ]
    return classify_vehicle_signal(crop, min_pixels=min_pixels)


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

    rows: list[dict[str, int | float | str]] = []
    try:
        for frame_index, frame in sampled_frames(capture, sample_stride):
            state, counts = classify_vehicle_signal_frame(frame, roi, scene_size, min_pixels=min_pixels)
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
