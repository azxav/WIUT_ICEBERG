"""
solution.py — entry point imported by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The pipeline lives in src/ (see README). Models are loaded and warmed up at
import time, which the harness does once before it starts timing videos.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from src.config import load_params
from src.perception import Detector, seed_everything
from src.pipeline import analyze
from src.risk import CausalRisk

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0


_PARAMS = load_params()
_DETECTOR: Detector | None = None


def _detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        p = _PARAMS["perception"]
        _DETECTOR = Detector(p["detector"], p["imgsz"], p["conf"], p["iou"], p["half"])
    return _DETECTOR


def _warmup() -> None:
    seed_everything()
    if os.environ.get("ICEBERG_NO_WARMUP"):
        return
    try:
        _detector().warmup()
        CausalRisk.warmup(_PARAMS)
    except Exception as exc:  # a missing weight must not break the import; the error resurfaces per video
        print(f"[solution] warm-up skipped: {exc}", file=sys.stderr)


def detect_events(video_path: str) -> list[list]:
    """Part A: [[start_sec, end_sec, label], ...] for one .mp4 (labels from CLASSES)."""
    return analyze(video_path, CLASSES, _PARAMS, detector=_detector()).as_lists()


class RiskEstimator:
    """Part B: causal P(accident starts within RISK_HORIZON_SEC). Sees frames in order only."""

    def __init__(self) -> None:
        self._engine = CausalRisk(_PARAMS)

    def reset(self, meta: dict) -> None:
        self._engine.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return float(self._engine.step(frame, t_sec))


_warmup()
