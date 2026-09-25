"""Object detection (Ultralytics YOLO) and multi-object tracking (ByteTrack).

Models are cached per (weights, device) so that ``solution.py`` can load and
warm them up at import time, outside the harness's per-video time budget.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DETECT_CLASSES, SEED, WEIGHTS_DIR


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def weights_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else WEIGHTS_DIR / p


@dataclass
class Detections:
    xyxy: np.ndarray   # (N, 4) float32 pixels
    conf: np.ndarray   # (N,)
    cls: np.ndarray    # (N,) int COCO id

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))

    def __len__(self) -> int:
        return len(self.conf)


@lru_cache(maxsize=4)
def _load_yolo(path: str, dev: str):
    from ultralytics import YOLO

    model = YOLO(path)
    model.to(dev)
    return model


class Detector:
    """Batched YOLO inference restricted to the traffic-relevant COCO classes."""

    def __init__(self, weights: str, imgsz: int = 960, conf: float = 0.2, iou: float = 0.6,
                 half: bool = True, classes: list[int] | None = DETECT_CLASSES):
        self.device = device()
        path = weights_path(weights)
        if not path.exists():
            raise FileNotFoundError(f"missing weights {path}; run weights/download.sh")
        self.model = _load_yolo(str(path), self.device)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.half = half and self.device.startswith("cuda")
        self.classes = classes

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        if not frames:
            return []
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     half=self.half, classes=self.classes, device=self.device,
                                     verbose=False)
        out = []
        for r in results:
            b = r.boxes
            out.append(Detections(b.xyxy.cpu().numpy().astype(np.float32),
                                  b.conf.cpu().numpy().astype(np.float32),
                                  b.cls.cpu().numpy().astype(int)))
        return out

    def warmup(self, width: int = 1920, height: int = 1080) -> None:
        self([np.zeros((height, width, 3), np.uint8)])


class Tracker:
    """ByteTrack (roboflow/supervision) over per-frame detections."""

    def __init__(self, frame_rate: float, lost_track_sec: float = 2.0,
                 activation: float = 0.3, matching: float = 0.8):
        import supervision as sv

        self._sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=activation,
            lost_track_buffer=max(1, int(round(lost_track_sec * frame_rate))),
            minimum_matching_threshold=matching,
            frame_rate=max(1, int(round(frame_rate))),
        )

    def update(self, det: Detections) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (track_ids, xyxy, conf, cls) for the confirmed tracks of this frame."""
        sv = self._sv
        d = sv.Detections(xyxy=det.xyxy.astype(np.float32), confidence=det.conf.astype(np.float32),
                          class_id=det.cls.astype(int))
        t = self.tracker.update_with_detections(d)
        if len(t) == 0:
            return np.zeros(0, int), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        return (t.tracker_id.astype(int), t.xyxy.astype(np.float32),
                t.confidence.astype(np.float32), t.class_id.astype(int))
