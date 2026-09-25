"""Video probing and strided frame iteration (OpenCV)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    name: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    meta = VideoMeta(
        name=Path(path).name,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, t_sec, bgr_frame)`` for every ``stride``-th frame.

    Skipped frames are only grabbed (no colour conversion), which is cheaper
    than a full read. Timestamps are ``idx / fps``, matching the harness.
    """
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
