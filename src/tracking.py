"""YOLO11 + ByteTrack inference and sampled-frame track cache generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import pandas as pd


ROAD_CLASSES = (0, 1, 2, 3, 5, 7)  # person, bicycle, car, motorcycle, bus, truck
TRACK_COLUMNS = ("frame", "t", "id", "cls", "conf", "x1", "y1", "x2", "y2")
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "yolo11s.pt"


def find_videos(directory: str | Path) -> list[Path]:
    """Find MP4 files once across case-sensitive and case-insensitive filesystems."""
    folder = Path(directory)
    return sorted({*folder.glob("*.mp4"), *folder.glob("*.MP4")}, key=lambda path: path.name.lower())


@dataclass(frozen=True)
class TrackConfig:
    """Settings selected from the sample-video timing and visual checks."""

    stride: int = 3
    imgsz: int = 960
    conf: float = 0.2
    max_frame_width: int = 1920
    weights: str = str(DEFAULT_WEIGHTS)
    device: int | str | None = 0
    quantize: int | str | None = 16
    tracker: str = "bytetrack.yaml"
    classes: tuple[int, ...] = ROAD_CLASSES

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.imgsz < 1 or self.max_frame_width < 1:
            raise ValueError("imgsz and max_frame_width must be positive")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be between 0 and 1")


def sampled_frames(capture: Any, stride: int) -> Iterator[tuple[int, Any]]:
    """Yield only every `stride`th decoded frame, retaining its source index."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    frame_index = 0
    while capture.grab():
        if frame_index % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f"failed to retrieve sampled frame {frame_index}")
            yield frame_index, frame
        frame_index += 1


def _as_list(values: Any) -> list[Any] | None:
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def result_rows(
    result: Any,
    frame_index: int,
    fps: float,
    coordinate_scale: float = 1.0,
) -> list[dict[str, int | float]]:
    """Convert one Ultralytics result to rows in original-video pixel units."""
    if fps <= 0 or coordinate_scale <= 0:
        raise ValueError("fps and coordinate_scale must be positive")
    boxes = getattr(result, "boxes", None)
    ids = _as_list(getattr(boxes, "id", None)) if boxes is not None else None
    if ids is None:
        return []
    classes = _as_list(boxes.cls)
    confidences = _as_list(boxes.conf)
    coordinates = _as_list(boxes.xyxy)
    rows = []
    for track_id, cls, confidence, xyxy in zip(ids, classes, confidences, coordinates):
        x1, y1, x2, y2 = (float(value) * coordinate_scale for value in xyxy)
        rows.append({
            "frame": int(frame_index),
            "t": float(frame_index / fps),
            "id": int(track_id),
            "cls": int(cls),
            "conf": float(confidence),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return rows


def track_video(
    video_path: str | Path,
    config: TrackConfig | None = None,
    model: Any | None = None,
    cache_path: str | Path | None = None,
    frame_callback: Callable[[int, float, Any], None] | None = None,
) -> pd.DataFrame:
    """Track road users; optionally persist a Parquet cache.

    Videos wider than 1920 px are reduced to that width before inference.
    Returned boxes stay in the source video's pixel coordinates.
    """
    cfg = config or TrackConfig()
    if model is None:
        from ultralytics import YOLO

        model = YOLO(cfg.weights)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"video has invalid FPS: {video_path}")

    rows: list[dict[str, int | float]] = []
    try:
        for frame_index, frame in sampled_frames(capture, cfg.stride):
            if frame_callback is not None:
                frame_callback(frame_index, frame_index / fps, frame)
            source_width = frame.shape[1]
            target_width = min(source_width, cfg.max_frame_width)
            if target_width != source_width:
                target_height = max(1, round(frame.shape[0] * target_width / source_width))
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            coordinate_scale = source_width / target_width
            results = model.track(
                frame,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                classes=list(cfg.classes),
                persist=True,
                tracker=cfg.tracker,
                device=cfg.device,
                quantize=cfg.quantize,
                verbose=False,
            )
            result = results[0] if isinstance(results, (list, tuple)) else results
            rows.extend(result_rows(result, frame_index, fps, coordinate_scale))
    finally:
        capture.release()

    tracks = pd.DataFrame.from_records(rows, columns=TRACK_COLUMNS)
    if cache_path is not None:
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tracks.to_parquet(destination, index=False)
    return tracks
