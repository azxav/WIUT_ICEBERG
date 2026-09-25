"""Gradio demo (Hugging Face Space, CPU): upload a road-camera clip, get events back.

Runs the same pipeline as ``solution.py`` (``src.pipeline.analyze``) with a
lighter perception config so a 2-minute clip finishes on CPU, then makes one
more decode pass that feeds every frame to the causal risk model and draws the
annotated video. Layout on the Space: this file next to ``src/``, ``configs/``,
``weights/`` and ``solution.py`` (or one directory below them, as in the repo).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, HERE.parent) if (p / "src" / "pipeline.py").exists()), HERE.parent)
sys.path[:0] = [str(ROOT), str(HERE)]
os.environ.setdefault("ICEBERG_NO_WARMUP", "1")  # we warm up the light detector ourselves

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

import visualize as viz  # noqa: E402

MAX_SECONDS = 120.0
MAX_MB = 200
RENDER_FPS = 12.5  # annotated output frame rate (drawing every frame is wasted on CPU)
RENDER_WIDTH = 960

# CPU-friendly perception; everything else comes from configs/params.json.
DEMO_OVERRIDE = {
    "perception": {
        "detector": os.environ.get("DEMO_DETECTOR", "yolo11n.pt"),
        "imgsz": 640, "stride": 4, "half": False, "batch": 4,
    },
    "risk": {"stride": 5},
}

_STATE: dict = {}


def _pipeline():
    """Import the repo lazily so the UI still starts (and explains) if something is missing."""
    if "err" in _STATE:
        raise RuntimeError(_STATE["err"])
    if "params" not in _STATE:
        try:
            from solution import CLASSES
            from src.config import WEIGHTS_DIR, load_params
            from src.perception import Detector
            from src.pipeline import analyze
            from src.risk import CausalRisk
            from src.video import probe

            params = load_params(DEMO_OVERRIDE)
            p = params["perception"]
            weights = WEIGHTS_DIR / p["detector"]
            if not weights.exists():  # Ultralytics fetches official checkpoints by name
                from ultralytics import YOLO

                WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
                got = Path(getattr(YOLO(p["detector"]), "ckpt_path", None) or p["detector"])
                if got.exists() and got.resolve() != weights.resolve():
                    shutil.copy(got, weights)
            detector = Detector(p["detector"], p["imgsz"], p["conf"], p["iou"], False)
            detector.warmup(640, 360)
            _STATE.update(params=params, classes=CLASSES, analyze=analyze, detector=detector,
                          risk_cls=CausalRisk, probe=probe)
        except Exception as exc:
            _STATE["err"] = f"pipeline failed to load: {exc}"
            raise RuntimeError(_STATE["err"]) from exc
    return _STATE


class ProgressDetector:
    """Wraps the detector to report decode/detection progress to Gradio."""

    def __init__(self, detector, total: int, progress, lo: float, hi: float):
        self.detector, self.total, self.progress = detector, max(1, total), progress
        self.lo, self.hi, self.done = lo, hi, 0

    def __call__(self, frames):
        out = self.detector(frames)
        self.done += len(frames)
        frac = min(1.0, self.done / self.total)
        self.progress(self.lo + (self.hi - self.lo) * frac,
                      desc=f"Detecting and tracking ({self.done}/{self.total} frames)")
        return out


def _validate(path: str | None, probe):
    if not path:
        raise gr.Error("Upload an .mp4 clip first.")
    size_mb = os.path.getsize(path) / 2**20
    if size_mb > MAX_MB:
        raise gr.Error(f"The file is {size_mb:.0f} MB; the limit is {MAX_MB} MB.")
    try:
        meta = probe(path)
    except Exception:
        raise gr.Error("This file could not be decoded as video. Upload an H.264 .mp4.")
    if meta.n_frames <= 0 or meta.fps <= 0:
        raise gr.Error("The video has no readable frames.")
    if meta.duration > MAX_SECONDS + 0.5:
        raise gr.Error(f"The clip is {meta.duration:.0f} s long; trim it to {MAX_SECONDS:.0f} s or less.")
    return meta


def _render_and_risk(path: str, meta, analysis, st, progress, lo: float, hi: float):
    """One pass over every frame: causal risk step() on all, drawing on a subset."""
    risk = st["risk_cls"](st["params"])
    risk.reset({"video_id": meta.name, "fps": meta.fps, "width": meta.width,
                "height": meta.height, "n_frames": meta.n_frames})
    scale = min(1.0, RENDER_WIDTH / max(1, meta.width))
    out_w, out_h = int(meta.width * scale), int(meta.height * scale)
    every = max(1, int(round(meta.fps / RENDER_FPS)))
    stride = int(st["params"]["perception"]["stride"])
    index = viz.TrackIndex(analysis.ctx.tracks, tolerance=1.5 * stride / meta.fps)
    events = analysis.events

    out_path = str(Path(tempfile.mkdtemp(prefix="iceberg_")) / f"{Path(meta.name).stem}_annotated.mp4")
    sink = viz.VideoSink(out_path, meta.fps / every, out_w, out_h)
    times, scores = [], []
    cap = cv2.VideoCapture(path)
    idx, score = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / meta.fps
            try:
                score = float(risk.step(frame, t))
            except Exception as exc:  # a broken risk model must not kill the demo
                print(f"[demo] risk.step failed at {t:.2f}s: {exc}", file=sys.stderr)
            times.append(t)
            scores.append(score)
            if idx % every == 0:
                small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if scale < 1 else frame
                active = [e for e in events if e.start <= t <= e.end]
                sink.write(viz.draw_frame(small, t, index.at(t), active, score, scale))
            if idx % 25 == 0:
                progress(lo + (hi - lo) * min(1.0, idx / max(1, meta.n_frames)),
                         desc=f"Risk model and annotated video ({t:.0f}/{meta.duration:.0f} s)")
            idx += 1
    finally:
        cap.release()
        sink.close()
    return out_path, np.asarray(times), np.asarray(scores), sink.h264


def run(video_path: str | None, progress=gr.Progress()):
    empty = (None, None, None, None, None)
    try:
        progress(0.0, desc="Loading models")
        st = _pipeline()
    except Exception as exc:
        return (f"**The demo could not start.** {exc}",) + empty
    meta = _validate(video_path, st["probe"])  # raises gr.Error -> shown as a message
    t0 = time.perf_counter()
    try:
        p = st["params"]["perception"]
        total = int(np.ceil(meta.n_frames / int(p["stride"])))
        det = ProgressDetector(st["detector"], total, progress, 0.02, 0.70)
        analysis = st["analyze"](video_path, st["classes"], st["params"], detector=det)
        t_detect = time.perf_counter() - t0
        progress(0.72, desc="Risk model and annotated video")
        video_out, rt, rs, h264 = _render_and_risk(video_path, meta, analysis, st, progress, 0.72, 0.99)
        events = sorted(analysis.events, key=lambda e: (e.start, e.label))

        rows = [[round(e.start, 2), round(e.end, 2), round(e.end - e.start, 2), viz.pretty(e.label),
                 ", ".join(map(str, e.track_ids))] for e in events]
        timeline = viz.timeline_figure(events, meta.duration, st["classes"])
        step = max(1, len(rt) // 1500)
        risk_fig = viz.risk_figure(rt[::step], rs[::step], events)

        pred = {"events": analysis.as_lists(),
                "risk": [[round(float(t), 3), round(float(s), 4)] for t, s in zip(rt, rs)]}
        json_path = Path(video_out).with_suffix(".json")
        json_path.write_text(json.dumps({"team": "iceberg", "videos": {meta.name: pred}}))

        total_s = time.perf_counter() - t0
        n_tracks = len(analysis.ctx.tracks)
        by_class = {}
        for e in events:
            by_class[e.label] = by_class.get(e.label, 0) + 1
        found = ", ".join(f"{viz.pretty(k)} ({v})" for k, v in by_class.items()) or "no events"
        status = (f"**{len(events)} events** in {meta.duration:.0f} s of video: {found}.  \n"
                  f"{n_tracks} tracked objects. Detection and rules took {t_detect:.0f} s, "
                  f"total {total_s:.0f} s on CPU ({total_s / max(meta.duration, 1):.1f}x real time). "
                  f"Peak risk {float(rs.max()) if len(rs) else 0:.2f}.")
        if not h264:
            status += "  \nffmpeg was not found, so the video is MPEG-4 and may not play in every browser."
        return status, rows, timeline, video_out, risk_fig, str(json_path)
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        return (f"**Processing failed:** `{type(exc).__name__}: {exc}`. "
                "Try a shorter clip or another encoding (H.264 .mp4).",) + empty


INTRO = """
# Iceberg: traffic events from a fixed camera
Upload a road-camera clip (.mp4, up to 2 minutes and 200 MB). The app detects and tracks road
users, runs our 14 event rules, scores accident risk frame by frame and returns an annotated video.
This Space runs on CPU with a small detector (YOLO11n, 640 px, every 4th frame), so expect roughly
1-3x the clip length and slightly lower accuracy than the GPU submission.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Iceberg traffic events", theme=gr.themes.Soft(primary_hue="yellow")) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Video(label="Road-camera clip (.mp4, up to 2 min)", sources=["upload"])
                btn = gr.Button("Detect events", variant="primary")
                examples = sorted((HERE / "examples").glob("*.mp4"))
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=[inp])
                status = gr.Markdown()
            with gr.Column(scale=1):
                out_video = gr.Video(label="Annotated video", interactive=False)
                out_json = gr.File(label="Predictions (harness format)")
        timeline = gr.Plot(label="Event timeline")
        risk = gr.Plot(label="Accident risk")
        table = gr.Dataframe(headers=["start (s)", "end (s)", "duration (s)", "class", "track ids"],
                             label="Events", wrap=True)
        btn.click(run, inputs=[inp], outputs=[status, table, timeline, out_video, risk, out_json],
                  concurrency_limit=1)
    return demo


if __name__ == "__main__":
    build().queue(max_size=8).launch(max_file_size=f"{MAX_MB}mb")
