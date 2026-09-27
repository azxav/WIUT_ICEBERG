"""Part A orchestration for a single fixed-camera video."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from .features import extract_features
from .rules import RuleConfig, generate_events
from .scene import load_scene
from .traffic_lights import SIGNAL_COLUMNS, classify_vehicle_signal_frame
from .tracking import TrackConfig, track_video


def video_info(path: str | Path) -> tuple[float, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata for {path}")
    return fps, width, height, n_frames / fps


def registered_scene(video_path: str | Path) -> tuple[dict, int]:
    """Register the reference scene map to this video's first frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video for scene registration: {video_path}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not read first frame for scene registration: {video_path}")
    scene, inliers = load_scene(frame)
    if scene is None:
        raise RuntimeError(f"scene registration failed for {video_path} ({inliers} matches)")
    return scene, inliers


def detect_video_events(video_path: str | Path) -> list[list]:
    """Run tracking, traffic-signal sampling, features and validated rules."""
    fps, width, height, duration = video_info(video_path)
    scene, _ = registered_scene(video_path)

    import torch
    from ultralytics import YOLO

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(TrackConfig().weights))
    config = TrackConfig(device=device, quantize=16 if device != "cpu" else None)
    signal_roi = scene["signals"]["veh_main"]["roi"]
    signal_rows = []

    def sample_signal(frame_index: int, time_sec: float, frame) -> None:
        state, counts = classify_vehicle_signal_frame(
            frame, signal_roi, tuple(scene.get("ref_size", (1920, 1080)))
        )
        signal_rows.append({
            "frame": frame_index, "t": time_sec, "state": state,
            "red": counts["R"], "amber": counts["Y"], "green": counts["G"],
        })

    tracks = track_video(video_path, config=config, model=model, frame_callback=sample_signal)
    signals = pd.DataFrame.from_records(signal_rows, columns=SIGNAL_COLUMNS)
    features = extract_features(tracks, scene, (width, height))
    return generate_events(features, signals, scene, duration, RuleConfig())
