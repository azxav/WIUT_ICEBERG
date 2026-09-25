"""Run perception once per video and cache the observations (``.cache/*.npz``).

    python scripts/extract.py                      # every samples/*.mp4
    python scripts/extract.py --videos samples/a.mp4 --force

Every other dev script (render, tune, eda, build_flow_prior) re-uses this
cache, so rules can be re-run in seconds. The printed runtime is the Part A
perception cost; compare it with the official budget of 3x the video length
(shared with Part B's risk estimator).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from _common import DEFAULT_CACHE, DEFAULT_VIDEOS, cache_file, list_videos, load_params, scene_for
from src.perception import Detector, device, seed_everything
from src.observe import observe
from src.video import probe

BUDGET_FACTOR = 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="folder of .mp4 or one file")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="observation cache folder")
    ap.add_argument("--params", type=Path, help="params override JSON (default: ICEBERG_PARAMS or none)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    videos = list_videos(args.videos)
    if not videos:
        print(f"no .mp4 files in {args.videos}", file=sys.stderr)
        return 2
    params = load_params(args.params)
    os.environ["ICEBERG_CACHE_DIR"] = str(args.cache)
    seed_everything()

    p = params["perception"]
    t0 = time.perf_counter()
    detector = Detector(p["detector"], p["imgsz"], p["conf"], p["iou"], p["half"])
    first = probe(videos[0])
    detector.warmup(first.width, first.height)
    print(f"detector {p['detector']} on {device()} (imgsz {p['imgsz']}, stride {p['stride']}, batch {p['batch']}) "
          f"loaded in {time.perf_counter() - t0:.1f}s\n")

    hdr = f"{'video':<28}{'dur s':>8}{'run s':>8}{'fps':>8}{'x real':>8}{'% 3x':>7}{'tracks':>8}  note"
    print(hdr)
    print("-" * len(hdr))
    tot_dur = tot_run = 0.0
    for v in videos:
        meta = probe(v)
        cache = cache_file(v, params, args.cache)
        cached = cache.exists()
        if cached and args.force:
            cache.unlink()
            cached = False
        t1 = time.perf_counter()
        obs = observe(v, params, scene_for(meta.width, meta.height), detector)
        run = time.perf_counter() - t1
        n_tracks = len(set(obs.tracks[:, 0].astype(int).tolist())) if len(obs.tracks) else 0
        note = "cached (use --force to time)" if cached else ""
        if not cached:
            tot_dur += meta.duration
            tot_run += run
        print(f"{v.name[:27]:<28}{meta.duration:>8.1f}{run:>8.1f}{meta.n_frames / max(run, 1e-9):>8.1f}"
              f"{run / max(meta.duration, 1e-9):>8.2f}{100 * run / (BUDGET_FACTOR * meta.duration):>6.0f}%"
              f"{n_tracks:>8}  {note}")
    if tot_run:
        print(f"\ncomputed: {tot_run:.0f}s for {tot_dur:.0f}s of video -> {tot_run / tot_dur:.2f}x real time, "
              f"{100 * tot_run / (BUDGET_FACTOR * tot_dur):.0f}% of the 3x budget (before Part B)")
    print(f"cache: {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
