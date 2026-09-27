"""Build the static, truthful data payload used by the project website."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import extract_features


CLASS_COLORS = {
    "congestion": "#54c6df", "failure_to_yield": "#e7a53a",
    "red_light": "#ef6c5b", "jaywalking": "#a48be4", "stop_line": "#d7bc71",
}


def video_size(video: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def simplify(points: np.ndarray, maximum: int = 28) -> list[list[float]]:
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum).round().astype(int)
        points = points[indices]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in points]


def build(
    root: Path,
    predictions_path: Path | None,
    metrics_path: Path | None,
    ablation_path: Path | None = None,
) -> dict:
    labels = json.loads((root / "my_labels.json").read_text(encoding="utf-8"))
    scene = json.loads((root / "src" / "scene.json").read_text(encoding="utf-8"))
    predictions = {}
    evaluation = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
    if predictions_path and predictions_path.exists():
        raw = json.loads(predictions_path.read_text(encoding="utf-8"))
        predictions = raw.get("videos", {})
        logs = raw.get("log", {})
    else:
        logs = {}

    class_counts: dict[str, int] = {}
    videos: list[dict] = []
    trajectory_lines: list[dict] = []
    all_points: list[np.ndarray] = []
    lane_counts: dict[str, int] = {}
    queue_counts: dict[str, int] = {}
    for video_id, annotation in labels.items():
        video_path = root / "samples" / video_id
        duration = float(annotation["duration"])
        width, height = video_size(video_path) if video_path.exists() else (3840, 2160)
        events = annotation.get("events", [])
        per_class: dict[str, int] = {}
        for _, _, label in events:
            class_counts[label] = class_counts.get(label, 0) + 1
            per_class[label] = per_class.get(label, 0) + 1

        pred_entry = predictions.get(video_id, {})
        pred_events = pred_entry.get("events", [])
        risk = pred_entry.get("risk", [])
        if risk:
            stride = max(1, len(risk) // 700)
            risk = risk[::stride]
        log = logs.get(video_id, {})
        media_name = f"/media/{Path(video_id).stem.lower()}_annotated.mp4"
        media_ready = (root / "website" / "public" / media_name.lstrip("/")).exists()
        videos.append({
            "id": video_id,
            "duration": duration,
            "fps": float(annotation.get("fps", 0.0)),
            "width": width,
            "height": height,
            "events": events,
            "eventCounts": per_class,
            "predictions": pred_events,
            "risk": risk,
            "runtime": log.get("total_sec"),
            "partA": log.get("part_a_sec"),
            "partB": log.get("part_b_sec"),
            "media": media_name,
            "mediaReady": media_ready,
            "errors": log.get("errors", []),
        })

        cache = root / "cache" / f"{Path(video_id).stem}_tracks.parquet"
        if not cache.exists():
            continue
        tracks = pd.read_parquet(cache)
        if tracks.empty:
            continue
        scale_x = float(scene.get("ref_size", [1920, 1080])[0]) / width
        scale_y = float(scene.get("ref_size", [1920, 1080])[1]) / height
        x = (tracks["x1"].to_numpy(float) + tracks["x2"].to_numpy(float)) * 0.5 * scale_x
        y = tracks["y2"].to_numpy(float) * scale_y
        coords = np.column_stack([x, y])
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < 1920) & (coords[:, 1] >= 0) & (coords[:, 1] < 1080)
        all_points.append(coords[valid][::max(1, int(len(coords) / 18000))])
        features = extract_features(tracks, scene, (width, height))
        for lane, count in features["lane_id"].value_counts(dropna=True).items():
            lane_counts[str(lane)] = lane_counts.get(str(lane), 0) + int(count)
        queue_counts[video_id] = int(features["in_queue"].sum())

        # Use a deterministic subset of tracked trajectories so the map stays light.
        groups = list(tracks.groupby("id", sort=True))
        if groups and len(trajectory_lines) < 72:
            take = max(1, len(groups) // 32)
            for _, group in groups[::take][:32]:
                points = np.column_stack([
                    (group["x1"].to_numpy(float) + group["x2"].to_numpy(float)) * 0.5 * scale_x,
                    group["y2"].to_numpy(float) * scale_y,
                ])
                points = points[np.isfinite(points).all(axis=1)]
                points = points[(points[:, 0] >= 0) & (points[:, 0] <= 1920) & (points[:, 1] >= 0) & (points[:, 1] <= 1080)]
                if len(points) >= 3:
                    trajectory_lines.append({
                        "video": video_id,
                        "points": simplify(points, 24),
                        "color": CLASS_COLORS["congestion"] if len(trajectory_lines) % 3 == 0 else "#8fa0a6",
                    })
                if len(trajectory_lines) >= 72:
                    break

    total_counts = [{"label": label, "count": class_counts[label]} for label in CLASS_COLORS if class_counts.get(label, 0)]
    total_counts.sort(key=lambda item: (-item["count"], item["label"]))
    density = []
    if all_points:
        points = np.concatenate(all_points, axis=0)
        grid, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(18, 32), range=((0, 1080), (0, 1920)))
        maximum = max(1.0, float(grid.max()))
        density = np.round(grid / maximum, 3).tolist()

    metrics = []
    per_class = evaluation.get("part_a", {}).get("per_class", {})
    for label, values in per_class.items():
        row = values.get("0.5", {})
        if row:
            metrics.append({
                "label": label,
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
                "tp": row.get("tp"), "fp": row.get("fp"), "fn": row.get("fn"),
            })

    per_video_eval = {row.get("video"): row for row in evaluation.get("part_a", {}).get("per_video", [])}
    for video in videos:
        error_row = per_video_eval.get(video["id"], {})
        video["errorCounts"] = {
            label: [int(value) for value in error_row[label].split("/")]
            for label in per_class
            if isinstance(error_row.get(label), str) and len(error_row[label].split("/")) == 3
        }

    signal_ablation = None
    if ablation_path and ablation_path.exists() and evaluation:
        before_eval = json.loads(ablation_path.read_text(encoding="utf-8"))
        def red_summary(result: dict) -> dict:
            row = result.get("part_a", {}).get("per_class", {}).get("red_light", {}).get("0.5", {})
            return {
                "score": result.get("part_a", {}).get("score_a"),
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "tp": row.get("tp", 0), "fp": row.get("fp", 0), "fn": row.get("fn", 0),
            }
        signal_ablation = {
            "before": {"label": "Red-priority vote", **red_summary(before_eval)},
            "after": {"label": "Lamp position + dominance", **red_summary(evaluation)},
            "note": "Same detections and dev labels; mixed red/green samples stay unknown after the fix.",
        }

    lane_rows = [{"lane": lane, "count": count} for lane, count in sorted(lane_counts.items())]
    return {
        "title": "Traffic event detection",
        "runtimeNote": "Timings are from the last completed full run before the final signal-rule and seed changes. The current code still needs a successful all-video timing run.",
        "threshold": 0.7,
        "emittedClasses": ["congestion", "failure_to_yield", "red_light"],
        "classCounts": total_counts,
        "videos": videos,
        "density": density,
        "trajectories": trajectory_lines,
        "laneCounts": lane_rows,
        "queueCounts": queue_counts,
        "metrics": metrics,
        "signalAblation": signal_ablation,
        "scene": scene,
        "datasetNote": "Manually reviewed development labels for rule validation; not a held-out test set.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=Path("predictions_samples.json"))
    parser.add_argument("--metrics", type=Path, default=Path("cache/site_eval.json"))
    parser.add_argument("--ablation-metrics", type=Path, default=Path("cache/site_eval_before_signal_fix.json"))
    parser.add_argument("--out", type=Path, default=Path("website/public/site-data.json"))
    args = parser.parse_args()
    pred = args.pred if args.pred.is_absolute() else ROOT / args.pred
    metrics = args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    ablation = args.ablation_metrics if args.ablation_metrics.is_absolute() else ROOT / args.ablation_metrics
    out = args.out if args.out.is_absolute() else ROOT / args.out
    payload = build(ROOT, pred, metrics, ablation)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    public = ROOT / "website" / "public"
    if pred.exists():
        (public / "predictions_samples.json").write_bytes(pred.read_bytes())
    readme = ROOT / "README.md"
    if readme.exists():
        (public / "README.md").write_bytes(readme.read_bytes())
    weights = ROOT / "weights" / "yolo11s.pt"
    if weights.exists():
        destination = public / "weights" / weights.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size != weights.stat().st_size:
            destination.write_bytes(weights.read_bytes())
    print(f"wrote {out} ({len(payload['videos'])} videos, {sum(x['count'] for x in payload['classCounts'])} reviewed events, {len(payload['trajectories'])} trajectories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
