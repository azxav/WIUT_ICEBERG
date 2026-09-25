"""One decode pass over a video -> everything the rules need, as plain arrays.

Rules never touch pixels: they read ``Observations``. That keeps them fast
and lets ``scripts/tune.py`` re-run every rule from a cached ``.npz`` in
seconds. Setting ``ICEBERG_CACHE_DIR`` enables that cache (dev only; the
official run leaves it unset).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .perception import Detections, Detector, Tracker, weights_path
from .scene import Scene
from .video import FrameReader, VideoMeta, output_size, probe

# columns of Observations.tracks
T_ID, T_TIME, T_X1, T_Y1, T_X2, T_Y2, T_CONF, T_CLS = range(8)
# columns of Observations.fire
F_TIME, F_CONF, F_CLS, F_X1, F_Y1, F_X2, F_Y2 = range(7)


@dataclass
class Observations:
    meta: VideoMeta
    sample_times: np.ndarray                       # (S,) timestamps of processed frames
    tracks: np.ndarray                             # (M, 8) id, t, x1, y1, x2, y2, conf, cls
    light_features: dict[str, np.ndarray] = field(default_factory=dict)  # "light:roi" -> (S, 4)
    fire: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))     # (F, 7)
    thumb_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thumbs: np.ndarray = field(default_factory=lambda: np.zeros((0, 1, 1), np.uint8))  # grey

    # -------------------------------------------------------------- cache
    def save(self, path: Path) -> None:
        arrays = {
            "meta": np.array(json.dumps(self.meta.__dict__)),
            "sample_times": self.sample_times,
            "tracks": self.tracks,
            "fire": self.fire,
            "thumb_times": self.thumb_times,
            "thumbs": self.thumbs,
        }
        arrays.update({f"light::{k}": v for k, v in self.light_features.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Observations":
        z = np.load(path, allow_pickle=False)
        return cls(
            meta=VideoMeta(**json.loads(str(z["meta"]))),
            sample_times=z["sample_times"],
            tracks=z["tracks"],
            light_features={k[len("light::"):]: z[k] for k in z.files if k.startswith("light::")},
            fire=z["fire"],
            thumb_times=z["thumb_times"],
            thumbs=z["thumbs"],
        )


def light_roi_features(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """(mean V, red frac, yellow frac, green frac) of the bright pixels in a box."""
    x1, y1, x2, y2 = box
    roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if roi.size == 0:
        return np.full(4, np.nan, np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    bright = (v > 150) & (s > 80)
    n = max(1, int(bright.sum()))
    red = bright & ((h <= 10) | (h >= 160))
    yellow = bright & (h > 10) & (h <= 35)
    green = bright & (h >= 40) & (h <= 95)
    return np.array([v.mean() / 255.0, red.sum() / n, yellow.sum() / n, green.sum() / n], np.float32)


def _cache_key(path: Path, params: dict, scene: Scene) -> str:
    stat = path.stat()
    blob = json.dumps({"p": params["perception"], "size": stat.st_size, "mtime": int(stat.st_mtime),
                       "lights": [(l.id, l.rois) for l in scene.lights]}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def observe(video_path: str | Path, params: dict, scene: Scene | None = None,
            detector: Detector | None = None, fire_detector: Detector | None = None) -> Observations:
    """Run detection + tracking (+ light / fire / thumbnail sampling) once over a video."""
    video_path = Path(video_path)
    meta = probe(video_path)
    scene = scene or Scene(meta.width, meta.height)
    cache_dir = os.environ.get("ICEBERG_CACHE_DIR")
    if cache_dir:
        cache = Path(cache_dir) / f"{video_path.stem}_{_cache_key(video_path, params, scene)}.npz"
        if cache.exists():
            return Observations.load(cache)

    p = params["perception"]
    stride = int(p["stride"])
    detector = detector or Detector(p["detector"], p["imgsz"], p["conf"], p["iou"], p["half"])
    if fire_detector is None and weights_path(p["fire_detector"]).exists():
        fire_detector = Detector(p["fire_detector"], 640, p["fire_conf"], 0.5, p["half"], classes=None)
    tracker = Tracker(meta.fps / stride, p["lost_track_sec"], p["track_activation_threshold"],
                      p["minimum_matching_threshold"])

    # frames are decoded at a working resolution; everything stored stays in native pixels
    fw, fh = output_size(meta.width, meta.height, p.get("decode_width"))
    scale = np.array([meta.width / fw, meta.height / fh] * 2, np.float32)
    rois = [(f"{l.id}:{name}", tuple(int(round(v / s)) for v, s in zip(box, scale)))
            for l in scene.lights for name, box in l.rois.items()]
    times: list[float] = []
    rows: list[np.ndarray] = []
    lights: dict[str, list[np.ndarray]] = {k: [] for k, _ in rois}
    fire_rows: list[list[float]] = []
    thumb_t: list[float] = []
    thumbs: list[np.ndarray] = []
    next_thumb, next_fire = 0.0, 0.0
    thumb_w = int(p["thumb_width"])
    thumb_h = max(1, int(round(thumb_w * meta.height / max(1, meta.width))))

    def flush(batch: list[tuple[float, np.ndarray]]) -> None:
        for (t, _), det in zip(batch, detector([f for _, f in batch])):
            ids, xyxy, conf, cls = tracker.update(det)
            if len(ids):
                block = np.column_stack([ids, np.full(len(ids), t), xyxy * scale, conf, cls])
                rows.append(block)

    # Time guard: Part A must leave room for Part B inside the harness budget
    # (3x duration for both; the harness's own 4K decode for Part B is ~1x).
    # If we run slower than time_ratio x real time, process fewer frames. On
    # normal hardware this never fires, so output stays deterministic.
    reader = FrameReader(video_path, stride, p.get("decode_width"))
    max_stride = stride * int(p.get("max_stride_factor", 4))
    ratio = float(p.get("time_ratio", 1.0))
    t0 = time.perf_counter()
    batch: list[tuple[float, np.ndarray]] = []
    for _, t, frame in reader:
        times.append(t)
        if t > 5.0 and reader.stride < max_stride and time.perf_counter() - t0 > ratio * t:
            reader.stride *= 2
            print(f"[observe] {meta.name}: behind budget at t={t:.0f}s, stride -> {reader.stride}", file=sys.stderr)
        for key, box in rois:
            lights[key].append(light_roi_features(frame, box))
        if t >= next_thumb:
            thumbs.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (thumb_w, thumb_h),
                                     interpolation=cv2.INTER_AREA))
            thumb_t.append(t)
            next_thumb = t + float(p["thumb_every_sec"])
        if fire_detector is not None and t >= next_fire:
            for d in fire_detector([frame]):
                for b, c, k in zip(d.xyxy, d.conf, d.cls):
                    fire_rows.append([t, float(c), float(k), *map(float, b * scale)])
            next_fire = t + float(p["fire_every_sec"])
        batch.append((t, frame))
        if len(batch) >= int(p["batch"]):
            flush(batch)
            batch = []
    flush(batch)

    obs = Observations(
        meta=meta,
        sample_times=np.asarray(times, float),
        tracks=np.concatenate(rows) if rows else np.zeros((0, 8)),
        light_features={k: np.asarray(v, np.float32).reshape(-1, 4) for k, v in lights.items()},
        fire=np.asarray(fire_rows, float).reshape(-1, 7),
        thumb_times=np.asarray(thumb_t, float),
        thumbs=np.asarray(thumbs, np.uint8) if thumbs else np.zeros((0, thumb_h, thumb_w), np.uint8),
    )
    if cache_dir:
        obs.save(cache)
    return obs


__all__ = ["Observations", "observe", "Detections", "light_roi_features",
           "T_ID", "T_TIME", "T_X1", "T_Y1", "T_X2", "T_Y2", "T_CONF", "T_CLS",
           "F_TIME", "F_CONF", "F_CLS", "F_X1", "F_Y1", "F_X2", "F_Y2"]
