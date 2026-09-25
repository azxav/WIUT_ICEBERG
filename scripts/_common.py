"""Shared helpers for the dev scripts (never imported by the submission).

Scripts are run as ``python scripts/<name>.py``, which puts ``scripts/`` on
``sys.path``; importing this module also puts the repo root there so that
``src``, ``evaluate`` and ``solution`` resolve.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import CONFIG_DIR, load_params, scene_path  # noqa: E402
from src.observe import Observations, _cache_key, observe  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.video import probe  # noqa: E402

DEFAULT_VIDEOS = ROOT / "samples"
DEFAULT_CACHE = ROOT / ".cache"
RENDERS = ROOT / "renders"
VIDEO_EXTS = {".mp4", ".MP4"}

__all__ = ["ROOT", "CONFIG_DIR", "DEFAULT_VIDEOS", "DEFAULT_CACHE", "RENDERS", "list_videos",
           "cache_file", "iter_observations", "load_params", "scene_for", "solution_classes",
           "read_frame", "write_json", "json_default"]


def list_videos(src: str | Path) -> list[Path]:
    """A single .mp4, or every .mp4 in a folder (sorted); [] if nothing is there."""
    p = Path(src)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix in VIDEO_EXTS)
    return []


def scene_for(width: int, height: int) -> Scene:
    """The scene the pipeline would use (``ICEBERG_SCENE`` or configs/scene.json)."""
    return Scene.load(scene_path(), width, height)


def cache_file(video: Path, params: dict, cache_dir: Path) -> Path:
    """Path ``observe()`` reads / writes for this video under the current params and scene."""
    meta = probe(video)
    return Path(cache_dir) / f"{video.stem}_{_cache_key(video, params, scene_for(meta.width, meta.height))}.npz"


def _newest_caches(cache_dir: Path) -> list[Path]:
    """Newest cache file per video stem (file names are ``<stem>_<key>.npz``)."""
    newest: dict[str, Path] = {}
    for f in sorted(Path(cache_dir).glob("*.npz"), key=lambda p: p.stat().st_mtime):
        newest[f.stem.rsplit("_", 1)[0]] = f
    return [newest[k] for k in sorted(newest)]


def iter_observations(videos: str | Path, params: dict, cache_dir: Path,
                      compute: bool = True) -> Iterator[tuple[Path | None, Observations]]:
    """Yield ``(video path or None, Observations)`` per video.

    Uses the observation cache; missing entries are computed (runs the
    detector) unless ``compute`` is False. If no video files exist at all, falls
    back to the newest cache file per video so analysis works without samples.
    """
    os.environ["ICEBERG_CACHE_DIR"] = str(cache_dir)
    paths = list_videos(videos)
    if not paths:
        caches = _newest_caches(cache_dir)
        if not caches:
            print(f"no videos in {videos} and no cache files in {cache_dir}", file=sys.stderr)
        for f in caches:
            yield None, Observations.load(f)
        return
    for p in paths:
        if not compute and not cache_file(p, params, cache_dir).exists():
            print(f"[{p.name}] not cached, skipped (run scripts/extract.py)", file=sys.stderr)
            continue
        meta = probe(p)
        yield p, observe(p, params, scene_for(meta.width, meta.height))


def solution_classes() -> list[str]:
    """``solution.CLASSES`` without loading the models."""
    os.environ.setdefault("ICEBERG_NO_WARMUP", "1")
    import solution

    return list(solution.CLASSES)


def read_frame(video: str | Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {t:.2f}s")
    return frame


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path: Path, data, indent: int | None = 1) -> None:
    """Atomic JSON write (a crash mid-write never leaves a truncated file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, default=json_default))
    os.replace(tmp, path)
