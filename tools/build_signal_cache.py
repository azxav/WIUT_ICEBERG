"""Build HSV traffic-signal state caches for the sample videos."""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene import load_scene  # noqa: E402
from src.traffic_lights import extract_signal_states  # noqa: E402
from src.tracking import find_videos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=ROOT / "samples")
    parser.add_argument("--backgrounds", type=Path, default=ROOT / "eda" / "frames")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = find_videos(args.videos)
    if not videos:
        raise FileNotFoundError(f"no MP4 videos in {args.videos}")
    args.cache.mkdir(parents=True, exist_ok=True)

    for video in videos:
        destination = args.cache / f"{video.stem}_signals.parquet"
        if destination.exists() and not args.overwrite:
            print(f"skip existing {destination}")
            continue
        background = cv2.imread(str(args.backgrounds / f"{video.stem}_background.jpg"))
        if background is None:
            raise FileNotFoundError(f"missing background for {video.name}")
        scene, inliers = load_scene(background)
        if scene is None:
            raise RuntimeError(f"scene registration failed for {video.name}: {inliers} inliers")
        roi = scene["signals"]["veh_main"]["roi"]

        started = time.perf_counter()
        states = extract_signal_states(video, roi=roi, sample_stride=args.stride)
        states.to_parquet(destination, index=False)
        elapsed = time.perf_counter() - started
        print(
            f"{video.name}: {len(states)} samples, {states.state.value_counts().to_dict()}, "
            f"scene inliers={inliers}, wall={elapsed:.1f}s -> {destination}"
        )


if __name__ == "__main__":
    main()
