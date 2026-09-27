"""Render compact annotated, browser-friendly previews for every sample clip."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_SIZE = (1280, 720)
CLASS_COLOR = {
    "congestion": (223, 198, 84),
    "failure_to_yield": (58, 165, 231),
    "red_light": (91, 108, 239),
}
COCO_CLASS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def render(video_id: str, result: dict, fps: float) -> tuple[Path, int]:
    stem = Path(video_id).stem
    source = ROOT / "samples" / "proxy" / f"{stem}_1080p.mp4"
    if not source.exists():
        source = ROOT / "samples" / video_id
    tracks_path = ROOT / "cache" / f"{stem}_tracks.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)

    tracks = pd.read_parquet(tracks_path)
    by_frame = {int(frame): rows for frame, rows in tracks.groupby("frame", sort=False)}
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"cannot open sample proxy: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or fps
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = source_fps / 3.0

    original_cap = cv2.VideoCapture(str(ROOT / "samples" / video_id))
    original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or source_width
    original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or source_height
    original_cap.release()

    output_dir = ROOT / "website" / "public" / "media"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{stem.lower()}_annotated.mp4"
    container = av.open(str(output), mode="w", format="mp4", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=Fraction(out_fps).limit_denominator(1001))
    stream.width, stream.height = OUT_SIZE
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "27"}

    duration = source_frames / source_fps if source_fps else 0.0
    events = result.get("events", [])
    risk = result.get("risk", [])
    frame_index = 0
    written = 0
    track_scale_x = OUT_SIZE[0] / original_width
    track_scale_y = OUT_SIZE[1] / original_height
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % 3 != 0:
                frame_index += 1
                continue
            t_sec = frame_index / source_fps
            canvas = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
            rows = by_frame.get(frame_index)
            if rows is not None:
                for row in rows.itertuples(index=False):
                    cls = int(row.cls)
                    if cls not in COCO_CLASS:
                        continue
                    x1, y1 = round(row.x1 * track_scale_x), round(row.y1 * track_scale_y)
                    x2, y2 = round(row.x2 * track_scale_x), round(row.y2 * track_scale_y)
                    color = (83, 197, 223) if cls == 0 else (219, 177, 72)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"{COCO_CLASS[cls]} #{int(row.id)}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1, cv2.LINE_AA)

            active = [event[2] for event in events if event[0] <= t_sec <= event[1]]
            active = list(dict.fromkeys(active))
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (OUT_SIZE[0], 62), (12, 20, 24), -1)
            cv2.rectangle(overlay, (0, OUT_SIZE[1] - 11), (OUT_SIZE[0], OUT_SIZE[1]), (12, 20, 24), -1)
            cv2.addWeighted(overlay, .83, canvas, .17, 0, canvas)
            cv2.putText(canvas, f"{video_id}  |  {t_sec:06.1f}s", (22, 27), cv2.FONT_HERSHEY_SIMPLEX, .62,
                        (238, 244, 245), 1, cv2.LINE_AA)
            label = "EVENT: " + (", ".join(active) if active else "no emitted event")
            label_color = CLASS_COLOR.get(active[0], (205, 217, 220)) if active else (177, 191, 196)
            cv2.putText(canvas, label, (22, 51), cv2.FONT_HERSHEY_SIMPLEX, .53, label_color, 1, cv2.LINE_AA)
            progress = min(1.0, t_sec / max(duration, 1e-6))
            cv2.line(canvas, (0, OUT_SIZE[1] - 5), (OUT_SIZE[0], OUT_SIZE[1] - 5), (63, 79, 85), 3)
            cv2.line(canvas, (0, OUT_SIZE[1] - 5), (round(OUT_SIZE[0] * progress), OUT_SIZE[1] - 5), (80, 194, 220), 3)
            risk_index = min(len(risk) - 1, max(0, round(t_sec * fps))) if risk else -1
            if risk_index >= 0:
                cv2.putText(canvas, f"experimental risk {float(risk[risk_index][1]):.2f}",
                            (OUT_SIZE[0] - 275, 51), cv2.FONT_HERSHEY_SIMPLEX, .48, (173, 199, 205), 1, cv2.LINE_AA)
            video_frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    if written < 2 or not output.exists() or output.stat().st_size < 2048:
        raise RuntimeError(f"rendered video is incomplete: {output}")
    return output, written


def main() -> int:
    predictions = json.loads((ROOT / "predictions_samples.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "my_labels.json").read_text(encoding="utf-8"))
    for video_id, meta in labels.items():
        output, frames = render(video_id, predictions["videos"][video_id], float(meta["fps"]))
        print(f"{video_id}: {frames} frames -> {output.relative_to(ROOT)} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
