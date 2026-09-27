"""Render a short track-cache clip with detections and mapped scene zones."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scene import load_scene  # noqa: E402
from src.tracking import sampled_frames  # noqa: E402


CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {
    0: (255, 210, 60), 1: (255, 100, 210), 2: (60, 230, 60),
    3: (0, 210, 255), 5: (255, 150, 40), 7: (50, 170, 255),
}


def draw_scene(frame: np.ndarray, scene: dict) -> np.ndarray:
    """Draw lightly tinted zones and thin geometry over a small preview frame."""
    overlay = frame.copy()

    def points(values):
        return np.rint(np.asarray(values, dtype=np.float32) * 0.5).astype(np.int32).reshape(-1, 1, 2)

    for name, item in scene.get("zones", {}).items():
        polygon = points(item["polygon"])
        cv2.fillPoly(overlay, [polygon], (150, 30, 150) if name == "junction" else (30, 170, 220))
        cv2.polylines(frame, [polygon], True, (190, 100, 230), 1, cv2.LINE_AA)
        x, y = polygon.reshape(-1, 2)[0]
        cv2.putText(frame, name, (int(x) + 2, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 220, 255), 1)
    frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

    for name, item in scene.get("crosswalks", {}).items():
        polygon = points(item["polygon"])
        cv2.polylines(frame, [polygon], True, (40, 230, 90), 1, cv2.LINE_AA)
        x, y = polygon.reshape(-1, 2)[0]
        cv2.putText(frame, name, (int(x) + 2, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 250, 110), 1)
    for name, item in scene.get("lanes", {}).items():
        cv2.polylines(frame, [points(item["polygon"])], True, (210, 80, 245), 1, cv2.LINE_AA)
    for name, item in scene.get("turn_paths", {}).items():
        cv2.polylines(frame, [points(item["polyline"])], False, (255, 80, 220), 2, cv2.LINE_AA)
    for item in scene.get("stop_lines", {}).values():
        cv2.polylines(frame, [points(item["line"])], False, (20, 20, 250), 2, cv2.LINE_AA)
    for name, item in scene.get("signals", {}).items():
        x1, y1, x2, y2 = np.rint(np.asarray(item["roi"]) * 0.5).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 40, 230), 1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "eda" / "debug" / "tracks.mp4")
    parser.add_argument("--background", type=Path)
    args = parser.parse_args()
    if args.start < 0 or args.end <= args.start:
        parser.error("require 0 <= --start < --end")

    background_path = args.background or ROOT / "eda" / "frames" / f"{args.video.stem}_background.jpg"
    background = cv2.imread(str(background_path))
    if background is None:
        raise FileNotFoundError(f"missing scene background: {background_path}")
    scene, inliers = load_scene(background)
    if scene is None:
        raise RuntimeError(f"scene registration failed: {inliers} inliers")

    tracks = pd.read_parquet(args.tracks)
    by_frame = {int(index): group for index, group in tracks.groupby("frame")}
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise ValueError("video has invalid metadata")

    preview_size = (960, 540)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, preview_size
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"could not create debug video: {args.output}")

    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps)
    first_preview = None
    frame_count = 0
    try:
        for frame_index, frame in sampled_frames(capture, args.stride):
            if frame_index > end_frame:
                break
            if frame_index < start_frame:
                continue
            small = cv2.resize(frame, preview_size, interpolation=cv2.INTER_AREA)
            small = draw_scene(small, scene)
            group = by_frame.get(frame_index)
            if group is not None:
                for row in group.itertuples():
                    xscale = preview_size[0] / source_width
                    yscale = preview_size[1] / source_height
                    x1, y1, x2, y2 = (
                        round(row.x1 * xscale), round(row.y1 * yscale),
                        round(row.x2 * xscale), round(row.y2 * yscale),
                    )
                    color = COLORS.get(int(row.cls), (255, 255, 255))
                    cv2.rectangle(small, (x1, y1), (x2, y2), color, 1)
                    text = f"{CLASS_NAMES.get(int(row.cls), row.cls)} #{int(row.id)} {row.conf:.2f}"
                    cv2.putText(small, text, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
            cv2.rectangle(small, (0, 0), (150, 24), (0, 0, 0), -1)
            cv2.putText(small, f"{frame_index / fps:.1f}s", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if first_preview is None:
                first_preview = small.copy()
            writer.write(small)
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("requested clip contains no sampled frames")
    preview_path = args.output.with_suffix(".jpg")
    cv2.imwrite(str(preview_path), first_preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{frame_count} frames, {frame_count / (fps / args.stride):.1f}s -> {args.output}; preview {preview_path}")


if __name__ == "__main__":
    main()
