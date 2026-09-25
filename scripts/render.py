"""Annotated MP4 + events JSON per video, for debugging and the website.

    python scripts/render.py                                  # every samples/*.mp4
    python scripts/render.py --videos samples/a.mp4 --labels my_labels.json --start 60 --end 90

Uses cached observations (scripts/extract.py) and ``pipeline.analyze(obs=...)``,
so only decoding, drawing and encoding cost time. The video shows boxes with
track ids and trails (red = involved in an active event), the scene layout,
light states, a banner with active events and a per-class timeline strip
(filled = predicted, white = ``--labels`` ground truth). Encodes H.264 with
ffmpeg when available (plays in browsers), otherwise mp4v.

Outputs per video in ``renders/``: ``<stem>.mp4``, a poster ``<stem>.jpg``, a
few event snapshots ``<stem>_<label>_<k>.jpg`` and ``<stem>.json`` - the
per-video input of site/build_data.py (meta, events, stats, examples) plus
raw rule events, tracks and light states for error analysis.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (DEFAULT_CACHE, DEFAULT_VIDEOS, RENDERS, iter_observations, load_params,
                     solution_classes, write_json)
from _viz import (CATEGORY_COLORS, CLASS_COLORS, draw_light_states, draw_playhead, draw_scene, text,
                  timeline)
from evaluate import OFFICIAL_CLASSES
from src.config import TWO_WHEELERS
from src.pipeline import Analysis, analyze

TRAIL_SEC = 2.0
EVENT_COLOR = (0, 0, 255)


class VideoWriter:
    """H.264 through an ffmpeg pipe (browser-playable), else OpenCV mp4v."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.proc = self.cv = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found: writing mp4v (may not play in browsers)", file=sys.stderr)
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


# ---------------------------------------------------------------- JSON
def event_score(an: Analysis, s: float, e: float, label: str) -> float:
    """Best raw rule score behind a final segment."""
    scores = [r.score for r in an.raw_events if r.label == label and r.start < e and r.end > s]
    return round(float(max(scores)), 4) if scores else 1.0


def stats(an: Analysis) -> dict:
    tracks = an.ctx.tracks
    over_time = []
    for t in np.arange(0.0, an.ctx.meta.duration, 1.0):
        live = [tr for tr in tracks if tr.start <= t <= tr.end]
        over_time.append([float(t), sum(tr.category == "vehicle" for tr in live),
                          sum(tr.category == "person" and not tr.is_rider for tr in live)])
    return {
        "tracks": len(tracks),
        "vehicles": sum(t.category == "vehicle" for t in tracks),
        "pedestrians": sum(t.category == "person" and not t.is_rider for t in tracks),
        "two_wheelers": sum(int(np.bincount(t.cls).argmax()) in TWO_WHEELERS for t in tracks),
        "objects_over_time": over_time,
    }


def analysis_json(an: Analysis, gt: list | None, path_hz: float = 2.0) -> dict:
    ctx, meta = an.ctx, an.ctx.meta
    tracks = []
    for tr in ctx.tracks:
        keep = np.unique(np.searchsorted(tr.t, np.arange(tr.start, tr.end + 1e-6, 1.0 / path_hz)).clip(0, len(tr) - 1))
        path = [[round(float(tr.t[i]), 2), round(float(tr.foot[i, 0] / meta.width), 4),
                 round(float(tr.foot[i, 1] / meta.height), 4)] for i in keep]
        tracks.append({"id": tr.id, "category": tr.category, "cls": int(np.bincount(tr.cls).argmax()),
                       "is_rider": tr.is_rider, "start": round(tr.start, 2), "end": round(tr.end, 2),
                       "max_speed": round(float(np.nanmax(tr.speed)), 3) if len(tr) else 0.0, "path": path})
    lights = {}
    for lid, states in ctx.light_states.items():
        if len(states):
            change = [0] + [i for i in range(1, len(states)) if states[i] != states[i - 1]]
            lights[lid] = [[round(float(ctx.times[i]), 2), str(states[i])] for i in change]
    return {
        "video": meta.name,
        "duration": round(meta.duration, 3), "fps": meta.fps, "width": meta.width, "height": meta.height,
        "events": [{"start": e.start, "end": e.end, "label": e.label,
                    "score": event_score(an, e.start, e.end, e.label), "track_ids": list(e.track_ids)}
                   for e in sorted(an.events, key=lambda e: e.start)],
        "stats": stats(an),
        "examples": [],
        "raw_events": [{"start": round(e.start, 3), "end": round(e.end, 3), "label": e.label,
                        "score": round(float(e.score), 4), "track_ids": list(e.track_ids), "info": e.info}
                       for e in sorted(an.raw_events, key=lambda e: e.start)],
        "ground_truth": gt,
        "tracks": tracks,
        "light_states": lights,
    }


def snapshot_plan(an: Analysis, stem: str, per_class: int) -> dict[str, dict]:
    """file name -> {"t", "event"}: a poster plus up to ``per_class`` mid-event frames per class."""
    events = sorted(an.events, key=lambda e: e.start)
    t_poster = (events[0].start + events[0].end) / 2 if events else an.ctx.meta.duration / 2
    plan = {f"{stem}.jpg": {"t": t_poster, "event": None}}
    seen: dict[str, int] = {}
    for e in events:
        k = seen.get(e.label, 0)
        if k < per_class:
            seen[e.label] = k + 1
            plan[f"{stem}_{e.label}_{k}.jpg"] = {"t": (e.start + e.end) / 2, "event": e}
    return plan


# --------------------------------------------------------------- video
def render_video(video: Path, an: Analysis, out: Path, width: int, gt: list | None,
                 start: float, end: float | None, every: int, snapshots: dict[str, float]) -> list[str]:
    """Writes the MP4 and the ``snapshots`` (file name -> time) as JPEGs next to it; returns those written."""
    ctx, meta = an.ctx, an.ctx.meta
    s = min(1.0, width / meta.width)
    fw, fh = int(round(meta.width * s)) // 2 * 2, int(round(meta.height * s)) // 2 * 2

    labels = sorted({e.label for e in an.events} | {g[2] for g in gt or []}, key=OFFICIAL_CLASSES.index)
    rows = [{"name": c, "color": CLASS_COLORS[c], "pred": [(e.start, e.end) for e in an.events if e.label == c],
             "gt": [(g[0], g[1]) for g in gt if g[2] == c] if gt is not None else None} for c in labels]
    strip = timeline(fw, meta.duration, rows, row_h=16 if gt is not None else 14)
    out_h = (fh + strip.shape[0]) // 2 * 2
    strip = strip[:out_h - fh]

    tracks = ctx.tracks
    t_start = np.array([t.start for t in tracks])
    t_end = np.array([t.end for t in tracks])
    step = float(np.median(np.diff(ctx.times))) if len(ctx.times) > 1 else 1.0 / meta.fps
    light_states = ctx.light_states

    first = int(max(0.0, start) * meta.fps)
    last = int((end if end is not None else meta.duration) * meta.fps)
    pending = sorted((t, name) for name, t in snapshots.items() if t >= first / meta.fps)
    written: list[str] = []
    cap = cv2.VideoCapture(str(video))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = VideoWriter(out, fw, out_h, meta.fps / every)
    try:
        for idx in range(first, last + 1):
            if (idx - first) % every:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / meta.fps
            img = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
            img = draw_scene(img, ctx.scene, s, alpha=0.15, labels=False)
            active = [e for e in an.events if e.start <= t <= e.end]
            involved = {i for e in active for i in e.track_ids}

            if light_states:
                k = int(np.clip(np.searchsorted(ctx.times, t, side="right") - 1, 0, len(ctx.times) - 1))
                draw_light_states(img, ctx.scene, {lid: str(v[k]) for lid, v in light_states.items() if len(v)}, s)

            for j in np.flatnonzero((t_start <= t) & (t_end + step >= t)):
                tr = tracks[j]
                i = tr.index_at(t)
                hit = tr.id in involved
                color = EVENT_COLOR if hit else CATEGORY_COLORS.get(tr.category, CATEGORY_COLORS["other"])
                x1, y1, x2, y2 = (tr.box[i] * s).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if hit else 1)
                lo = np.searchsorted(tr.t, t - TRAIL_SEC)
                trail = (tr.foot[lo:i + 1] * s).astype(np.int32)
                if len(trail) > 1:
                    cv2.polylines(img, [trail], False, color, 1, cv2.LINE_AA)
                text(img, f"{tr.id}{'R' if tr.is_rider else ''}", (x1, y1 - 2), color, 0.38, 1, (0, 0, 0))

            y = 22
            for e in active:
                text(img, f"{e.label}  {e.start:.1f}-{e.end:.1f}s", (8, y), (255, 255, 255), 0.55, 1,
                     CLASS_COLORS[e.label])
                y += 22
            text(img, f"{int(t // 60)}:{t % 60:05.2f}", (fw - 90, 20), (255, 255, 255), 0.5)

            while pending and pending[0][0] <= t + 0.5 * every / meta.fps:
                name = pending.pop(0)[1]
                cv2.imwrite(str(out.parent / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                written.append(name)
            bar = strip.copy()
            draw_playhead(bar, t, meta.duration)
            writer.write(np.vstack([img, bar]))
    finally:
        writer.close()
        cap.release()
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="folder of .mp4 or one file")
    ap.add_argument("--out", type=Path, default=RENDERS, help="output folder")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--params", type=Path, help="params override JSON (e.g. configs/params.tuned.json)")
    ap.add_argument("--labels", type=Path, help="ground truth JSON to show in the timeline")
    ap.add_argument("--width", type=int, default=1280, help="output frame width")
    ap.add_argument("--every", type=int, default=1, help="render every N-th frame")
    ap.add_argument("--start", type=float, default=0.0, help="clip start, seconds")
    ap.add_argument("--end", type=float, help="clip end, seconds")
    ap.add_argument("--examples", type=int, default=2, help="event snapshots per class")
    ap.add_argument("--json-only", action="store_true", help="skip the MP4 and snapshots")
    args = ap.parse_args()

    params = load_params(args.params)
    classes = solution_classes()
    labels = json.loads(args.labels.read_text()) if args.labels else None
    args.out.mkdir(parents=True, exist_ok=True)
    for video, obs in iter_observations(args.videos, params, args.cache):
        name = obs.meta.name
        stem = Path(name).stem
        an = analyze(video or name, classes, params, obs=obs)
        gt = labels.get(name, {}).get("events", []) if labels is not None else None
        data = analysis_json(an, gt)
        counts = {c: sum(e.label == c for e in an.events) for c in classes}
        print(f"[{name}] {len(an.events)} events " + " ".join(f"{c}={n}" for c, n in counts.items() if n))
        if video is None:
            print(f"[{name}] video file not found, JSON only", file=sys.stderr)
        elif not args.json_only:
            plan = snapshot_plan(an, stem, args.examples)
            written = render_video(video, an, args.out / f"{stem}.mp4", args.width, gt, args.start, args.end,
                                   max(1, args.every), {f: p["t"] for f, p in plan.items()})
            data["annotated"] = f"{stem}.mp4"
            for f in written:
                ev = plan[f]["event"]
                if ev is None:
                    data["poster"] = f
                else:
                    data["examples"].append({"label": ev.label, "start": ev.start, "end": ev.end, "image": f,
                                             "caption": f"{ev.label.replace('_', ' ')}, {ev.start:.1f}-{ev.end:.1f} s"})
            print(f"[{name}] wrote {args.out / data['annotated']} + {len(written)} snapshot(s)")
        write_json(args.out / f"{stem}.json", data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
