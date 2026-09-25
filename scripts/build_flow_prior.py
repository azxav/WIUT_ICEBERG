"""Aggregate the learned flow field over all sample videos -> configs/flow_prior.npz.

    python scripts/build_flow_prior.py

``src.pipeline`` adds this prior to the flow field it learns from each test
video, so ``wrong_way`` knows the legal direction even in cells the test clip
barely covers. The histogram is averaged per video (``--weight`` scales it),
so the prior counts about as much as one clip's own traffic.

Only meaningful when all clips come from the same fixed camera: the script
compares the median thumbnails and warns when they look different.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import CONFIG_DIR, DEFAULT_CACHE, DEFAULT_VIDEOS, iter_observations, load_params
from src.flowfield import FlowField
from src.tracks import build_tracks


def backdrop(obs) -> np.ndarray | None:
    if not len(obs.thumbs):
        return None
    img = np.median(obs.thumbs[:: max(1, len(obs.thumbs) // 20)], axis=0).astype(np.float32)
    return cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "flow_prior.npz")
    ap.add_argument("--weight", type=float, default=1.0, help="prior weight in 'videos worth of traffic'")
    ap.add_argument("--min-similarity", type=float, default=0.6,
                    help="warn when a clip's backdrop correlates less than this with the first one")
    args = ap.parse_args()

    params = load_params(args.params)
    grid = tuple(params["flow"]["grid"])
    total: FlowField | None = None
    ref = None
    n = 0
    for _, obs in iter_observations(args.videos, params, args.cache, compute=False):
        meta = obs.meta
        ff = FlowField.from_tracks(build_tracks(obs, params), meta.width, meta.height, grid)
        if total is None:
            total = FlowField(meta.width, meta.height, grid)
        total.hist += ff.hist  # grid cells are resolution independent
        n += 1
        bd = backdrop(obs)
        note = ""
        if bd is not None:
            if ref is None:
                ref = bd
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sim = float(np.corrcoef(ref.ravel(), bd.ravel())[0, 1])
                if np.isfinite(sim):
                    note = f"  backdrop similarity {sim:.2f}" + (
                        "  <-- different camera view?" if sim < args.min_similarity else "")
        print(f"[{meta.name}] {int(ff.hist.sum())} heading samples, "
              f"{int((ff.hist.sum(axis=2) >= params['flow']['min_count']).sum())} cells covered{note}")
    if total is None:
        print("no cached observations found (run scripts/extract.py first)", file=sys.stderr)
        return 2
    total.hist *= args.weight / n
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total.save(args.out)
    print(f"wrote {args.out} from {n} video(s), weight {args.weight}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
