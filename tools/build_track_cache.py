"""Build Parquet track caches for videos in a folder."""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tracking import DEFAULT_WEIGHTS, TrackConfig, find_videos, track_video  # noqa: E402


def build_config(args: argparse.Namespace) -> TrackConfig:
    """Translate CLI device settings into a safe inference precision."""
    if args.cpu or str(args.device).lower() == "cpu":
        device = "cpu"
    elif str(args.device).isdigit():
        device = int(args.device)
    else:
        device = args.device
    return TrackConfig(
        stride=args.stride,
        imgsz=args.imgsz,
        conf=args.conf,
        weights=str(args.weights),
        device=device,
        quantize=None if device == "cpu" else 16,
    )


def legacy_cache_path(video: Path, cache_dir: Path, config: TrackConfig) -> Path | None:
    """Return a compatible parameterized cache, if one is already present."""
    matches_legacy_settings = (
        config.conf == 0.2
        and config.max_frame_width == 1920
        and config.device == 0
        and config.quantize == 16
        and config.tracker == "bytetrack.yaml"
        and config.classes == (0, 1, 2, 3, 5, 7)
    )
    if not matches_legacy_settings:
        return None
    candidate = cache_dir / f"{video.stem}_s{config.stride}_{config.imgsz}_{Path(config.weights).stem}.parquet"
    return candidate if candidate.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=ROOT / "samples")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--cpu", action="store_true", help="Run full precision inference on CPU")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    config = build_config(args)
    videos = find_videos(args.videos)
    if not videos:
        raise FileNotFoundError(f"no MP4 videos in {args.videos}")

    args.cache.mkdir(parents=True, exist_ok=True)
    for video in videos:
        destination = args.cache / f"{video.stem}_tracks.parquet"
        if destination.exists() and not args.overwrite:
            print(f"skip existing {destination}")
            continue

        # Reuse an earlier named cache only when its full effective settings match.
        legacy = legacy_cache_path(video, args.cache, config)
        started = time.perf_counter()
        if legacy is not None and not args.overwrite:
            pd.read_parquet(legacy).to_parquet(destination, index=False)
            elapsed = time.perf_counter() - started
            print(f"{video.name}: reused {legacy.name} -> {destination.name} ({elapsed:.1f}s)")
            continue

        tracks = track_video(video, config=config, cache_path=destination)
        elapsed = time.perf_counter() - started
        import cv2

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        duration = frames / fps if fps > 0 else 0.0
        ratio = elapsed / duration if duration else float("inf")
        print(
            f"{video.name}: {duration:.1f}s video, wall {elapsed:.1f}s = {ratio:.2f}x; "
            f"{tracks.id.nunique() if len(tracks) else 0} tracks, {len(tracks)} rows -> {destination}"
        )


if __name__ == "__main__":
    main()
