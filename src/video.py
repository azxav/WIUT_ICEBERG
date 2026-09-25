"""Video probing and strided frame iteration.

The camera delivers 4K H.264 4:2:2 10-bit. Decoding is cheap; converting a
full 4K 10-bit frame to BGR is not (OpenCV manages ~30 fps for that on a
laptop). ``iter_frames`` therefore decodes with PyAV using frame threading
and lets swscale convert *and* downscale in one step, only for the frames we
keep: ~4x faster than ``cv2.VideoCapture.read`` on this footage. OpenCV is
the fallback when PyAV is unavailable or cannot open the file.
"""
from __future__ import annotations

import queue
import threading
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
    """Metadata as the harness sees it (OpenCV), so timestamps agree with it."""
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


def output_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """Frame size after downscaling to ``max_width`` (never upscales; even dimensions)."""
    if not max_width or width <= max_width:
        return width, height
    return max_width, int(round(height * max_width / width / 2)) * 2


class FrameReader:
    """Decode on a background thread (PyAV/swscale release the GIL) so decoding
    overlaps GPU inference. ``stride`` may be raised while iterating, which is
    how the pipeline sheds load when it falls behind its time budget.
    """

    def __init__(self, path: str | Path, stride: int = 1, max_width: int | None = None, prefetch: int = 16):
        self.path, self.stride, self.max_width = path, max(1, int(stride)), max_width
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()

    def _produce(self) -> None:
        try:
            next_idx = 0
            for idx, t, frame in iter_frames(self.path, 1, self.max_width, keep=lambda i: i >= next_idx):
                if self._stop.is_set():
                    return
                next_idx = idx + self.stride
                self._queue.put((idx, t, frame))
        except BaseException as exc:  # re-raised in the consumer
            self._queue.put(exc)
        finally:
            self._queue.put(None)

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        worker = threading.Thread(target=self._produce, daemon=True)
        worker.start()
        try:
            while (item := self._queue.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stop.set()
            while worker.is_alive():  # unblock a producer waiting on a full queue
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    worker.join(0.05)


def iter_frames(path: str | Path, stride: int = 1,
                max_width: int | None = None, keep=None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, t_sec, bgr_frame)`` for every ``stride``-th frame.

    ``keep(idx) -> bool`` replaces the stride test when given. Frames wider
    than ``max_width`` are downscaled (aspect kept). Timestamps are
    ``idx / fps`` with the OpenCV frame rate, matching the harness.
    """
    meta = probe(path)
    keep = keep or (lambda i: i % stride == 0)
    try:
        import av  # noqa: F401
    except ImportError:
        yield from _iter_cv2(path, keep, max_width, meta.fps)
        return
    yield from _iter_pyav(path, keep, max_width, meta)


def _iter_pyav(path, keep, max_width, meta: VideoMeta):
    import av

    w, h = output_size(meta.width, meta.height, max_width)
    try:
        container = av.open(str(path))
    except Exception:
        yield from _iter_cv2(path, keep, max_width, meta.fps)
        return
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # frame + slice threading; output order is deterministic
        for idx, frame in enumerate(container.decode(stream)):
            if keep(idx):
                yield idx, idx / meta.fps, frame.to_ndarray(width=w, height=h, format="bgr24")
    finally:
        container.close()


def _iter_cv2(path, keep, max_width, fps):
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while True:
            if keep(idx):
                ok, frame = cap.read()
                if not ok:
                    break
                w, h = output_size(frame.shape[1], frame.shape[0], max_width)
                if w != frame.shape[1]:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                yield idx, idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
