"""Exploratory plots and a JSON summary from cached observations.

    python scripts/eda.py                  # every samples/*.mp4 (or every cached video)

Per video, as ``renders/eda/<stem>_<kind>.png``: brightness over time, object counts
by category, a motion heatmap of foot points, trajectories, the learned
flow-direction field (src.flowfield) and per-lane vehicle density (if
configs/scene.json has lanes). ``renders/eda/summary.json`` lists the figures
with captions plus the numbers (the format site/build_data.py reads). Needs
matplotlib (requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import (DEFAULT_CACHE, DEFAULT_VIDEOS, RENDERS, iter_observations, load_params,  # noqa: E402
                     scene_for, write_json)
from src.config import CATEGORY_OF  # noqa: E402
from src.flowfield import FlowField  # noqa: E402
from src.observe import T_CLS, T_TIME, T_X1, T_X2, T_Y2, Observations  # noqa: E402
from src.scene import points_in_polygon  # noqa: E402
from src.tracks import Track, build_tracks  # noqa: E402

CATS = ["vehicle", "person", "bicycle", "animal", "traffic_light", "other"]
CAT_COLORS = {"vehicle": "#3c8cff", "person": "#2ca02c", "bicycle": "#ffb400", "animal": "#c050ff",
              "traffic_light": "#d62728", "other": "#888888"}


def background(ax, obs: Observations) -> None:
    """Median thumbnail as a grey backdrop in full-resolution pixel coordinates."""
    W, H = obs.meta.width, obs.meta.height
    if len(obs.thumbs):
        ax.imshow(np.median(obs.thumbs[:: max(1, len(obs.thumbs) // 30)], axis=0), cmap="gray",
                  extent=(0, W, H, 0), alpha=0.8)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])


class FigureSaver:
    """Saves ``<out>/<stem>_<kind>.png`` and records it for summary.json's figure list."""

    def __init__(self, out_dir: Path, stem: str):
        self.out_dir, self.stem = out_dir, stem
        self.figures: list[dict] = []

    def __call__(self, fig, kind: str, caption: str) -> None:
        name = f"{self.stem}_{kind}.png"
        fig.tight_layout()
        fig.savefig(self.out_dir / name, dpi=110)
        plt.close(fig)
        self.figures.append({"src": name, "caption": f"{caption} ({self.stem})"})


def counts_per_sample(obs: Observations) -> dict[str, np.ndarray]:
    """Number of tracked objects of each category at every processed frame."""
    times = obs.sample_times
    out = {c: np.zeros(len(times), int) for c in CATS}
    if len(obs.tracks) and len(times):
        idx = np.clip(np.searchsorted(times, obs.tracks[:, T_TIME] - 1e-6), 0, len(times) - 1)
        cats = np.array([CATEGORY_OF.get(int(c), "other") for c in obs.tracks[:, T_CLS]])
        for c in CATS:
            np.add.at(out[c], idx[cats == c], 1)
    return out


def lane_density(obs: Observations, tracks: list[Track], scene) -> dict[str, dict]:
    times = obs.sample_times
    out = {}
    for lane in scene.lanes:
        occ = np.zeros(len(times), int)
        speeds = []
        for tr in tracks:
            if tr.category != "vehicle":
                continue
            inside = points_in_polygon(lane.polygon, tr.foot)
            if inside.any():
                np.add.at(occ, np.clip(np.searchsorted(times, tr.t[inside] - 1e-6), 0, len(times) - 1), 1)
                speeds.append(tr.speed[inside])
        sp = np.concatenate(speeds) if speeds else np.zeros(0)
        out[lane.id] = {"occupancy": occ, "mean_occupancy": float(occ.mean()) if len(occ) else 0.0,
                        "max_occupancy": int(occ.max()) if len(occ) else 0,
                        "mean_speed_heights_per_s": float(sp.mean()) if len(sp) else None,
                        "stationary_share": float((sp < 0.08).mean()) if len(sp) else None}
    return out


def analyse(obs: Observations, params: dict, save: FigureSaver) -> dict:
    meta = obs.meta
    W, H = meta.width, meta.height
    times = obs.sample_times
    tracks = build_tracks(obs, params)
    scene = scene_for(W, H)

    # brightness (from the 1 fps grey thumbnails)
    bright = obs.thumbs.reshape(len(obs.thumbs), -1).mean(axis=1) / 255 if len(obs.thumbs) else np.zeros(0)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(obs.thumb_times, bright, color="#444")
    ax.set(xlabel="time (s)", ylabel="mean brightness", ylim=(0, 1), title=f"{meta.name}: brightness")
    save(fig, "brightness", "Mean frame brightness over time")

    # object counts by category
    counts = counts_per_sample(obs)
    fig, ax = plt.subplots(figsize=(10, 3))
    for c in CATS:
        if counts[c].any():
            ax.plot(times, counts[c], label=c, color=CAT_COLORS[c], lw=1)
    ax.set(xlabel="time (s)", ylabel="tracked objects", title=f"{meta.name}: objects per frame")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "counts", "Tracked objects per frame by category")

    # motion heatmap of raw foot points
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    background(ax, obs)
    if len(obs.tracks):
        fx = (obs.tracks[:, T_X1] + obs.tracks[:, T_X2]) / 2
        fy = obs.tracks[:, T_Y2]
        hist, xe, ye = np.histogram2d(fx, fy, bins=(96, 54), range=((0, W), (0, H)))
        ax.imshow(np.log1p(hist.T), extent=(0, W, H, 0), cmap="inferno", alpha=0.6)
    ax.set_title(f"{meta.name}: foot-point heatmap (log)")
    save(fig, "heatmap", "Where road users are: foot-point heatmap")

    # trajectories
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    background(ax, obs)
    for tr in tracks:
        ax.plot(tr.foot[:, 0], tr.foot[:, 1], lw=0.6, alpha=0.6, color=CAT_COLORS.get(tr.category, "#888"))
    ax.set_title(f"{meta.name}: {len(tracks)} trajectories")
    save(fig, "trajectories", "All trajectories, coloured by category")

    # learned flow field
    fp = params["flow"]
    ff = FlowField.from_tracks(tracks, W, H, tuple(fp["grid"]))
    gx, gy = np.meshgrid((np.arange(ff.gx) + 0.5) * W / ff.gx, (np.arange(ff.gy) + 0.5) * H / ff.gy)
    centres = np.column_stack([gx.ravel(), gy.ravel()])
    angle, share, total = ff.dominant(centres)
    ok = total >= fp["min_count"]
    fig, ax = plt.subplots(figsize=(9, 9 * H / W))
    background(ax, obs)
    if ok.any():
        L = 0.8 * W / ff.gx
        q = ax.quiver(centres[ok, 0], centres[ok, 1], np.cos(angle[ok]) * L, np.sin(angle[ok]) * L, share[ok],
                      angles="xy", scale_units="xy", scale=1, cmap="viridis", clim=(0.3, 1.0), width=0.002)
        fig.colorbar(q, ax=ax, fraction=0.03, label="dominant-direction share")
    ax.set_title(f"{meta.name}: learned flow (cells with >= {fp['min_count']} samples)")
    save(fig, "flow", "Learned traffic direction per grid cell")

    # per-lane density
    lanes = lane_density(obs, tracks, scene)
    if lanes:
        fig, ax = plt.subplots(figsize=(10, 3))
        for lid, d in lanes.items():
            ax.plot(times, d["occupancy"], lw=1, label=lid)
        ax.set(xlabel="time (s)", ylabel="vehicles in lane", title=f"{meta.name}: lane occupancy")
        ax.legend(fontsize=8, loc="upper right")
        save(fig, "lanes", "Vehicles per annotated lane over time")

    n_by_cat = {c: sum(t.category == c for t in tracks) for c in CATS if any(t.category == c for t in tracks)}
    return {
        "video": meta.name,
        "resolution": [W, H], "fps": meta.fps, "duration": round(meta.duration, 2),
        "samples": len(times),
        "sample_step_sec": round(float(np.median(np.diff(times))), 4) if len(times) > 1 else None,
        "brightness": {"mean": float(bright.mean()), "min": float(bright.min()), "max": float(bright.max())}
        if len(bright) else None,
        "tracks": n_by_cat,
        "raw_track_ids": int(len(np.unique(obs.tracks[:, 0]))) if len(obs.tracks) else 0,
        "per_frame": {c: {"mean": round(float(v.mean()), 2), "max": int(v.max())}
                      for c, v in counts.items() if v.any()},
        "track_length_sec": {"median": round(float(np.median([t.end - t.start for t in tracks])), 2),
                             "p90": round(float(np.percentile([t.end - t.start for t in tracks], 90)), 2)}
        if tracks else None,
        "flow_cells_covered": int(ok.sum()),
        "lanes": {k: {kk: vv for kk, vv in v.items() if kk != "occupancy"} for k, v in lanes.items()},
        "light_rois": sorted(obs.light_features),
        "fire_detections": int(len(obs.fire)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--out", type=Path, default=RENDERS / "eda")
    ap.add_argument("--compute", action="store_true", help="run perception for uncached videos")
    args = ap.parse_args()

    params = load_params(args.params)
    args.out.mkdir(parents=True, exist_ok=True)
    videos, figures = [], []
    for _, obs in iter_observations(args.videos, params, args.cache, compute=args.compute):
        save = FigureSaver(args.out, Path(obs.meta.name).stem)
        info = analyse(obs, params, save)
        videos.append(info)
        figures += save.figures
        print(f"[{obs.meta.name}] {info['duration']}s {info['resolution'][0]}x{info['resolution'][1]} "
              f"@{info['fps']:.2f}  tracks {info['tracks']}")
    if not videos:
        print("nothing analysed (run scripts/extract.py first)", file=sys.stderr)
        return 2
    write_json(args.out / "summary.json", {"figures": figures, "summary": overall(videos), "videos": videos},
               indent=2)
    print(f"wrote {len(figures)} figures and summary.json to {args.out}")
    return 0


def overall(videos: list[dict]) -> dict:
    """Flat key -> value numbers across all videos (the website's EDA table)."""
    dur = sum(v["duration"] for v in videos)
    tracks: dict[str, int] = {}
    for v in videos:
        for c, n in v["tracks"].items():
            tracks[c] = tracks.get(c, 0) + n
    veh = [v["per_frame"].get("vehicle", {}).get("mean", 0.0) * v["duration"] for v in videos]
    return {
        "videos": len(videos),
        "footage_min": round(dur / 60, 1),
        "resolutions": sorted({f"{v['resolution'][0]}x{v['resolution'][1]}" for v in videos}),
        "fps": sorted({round(float(v["fps"]), 2) for v in videos}),
        **{f"{c}_tracks": n for c, n in sorted(tracks.items())},
        "mean_vehicles_in_view": round(sum(veh) / max(dur, 1e-9), 2),
        "max_vehicles_in_view": max(v["per_frame"].get("vehicle", {}).get("max", 0) for v in videos),
        "mean_brightness": round(float(np.mean([v["brightness"]["mean"] for v in videos if v["brightness"]]
                                               or [0.0])), 3),
    }


if __name__ == "__main__":
    sys.exit(main())
