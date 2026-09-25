"""Build site/data/results.json (and copy media) from the pipeline's outputs.

Usage (from the repo root)::

    python site/build_data.py --pred predictions_samples.json --renders renders \
        [--gt my_labels.json] [--ablations ablations.json] [--failures failures.json]

Inputs
------
``--pred``      harness output: {"team", "videos": {name: {"events": [[s, e, label]], "risk": [[t, p]]}}}
``--renders``   folder written by scripts/render.py. Per video ``<stem>.json`` (every key optional)::

                  {"video": "sample_001.mp4", "duration": 184.0, "fps": 25, "width": 1920, "height": 1080,
                   "recorded_at": "2026-05-12T08:14:00",          # wall-clock start, for per-hour charts
                   "annotated": "sample_001.mp4",                   # rendered video, relative to renders/
                   "poster": "sample_001.jpg",
                   "events": [{"start", "end", "label", "score", "track_ids"}],  # overrides --pred events
                   "stats": {"tracks", "vehicles", "pedestrians", "two_wheelers", "mean_speed_kmh",
                             "objects_over_time": [[t_sec, n_vehicles, n_persons]], "runtime_x_realtime"},
                   "examples": [{"label", "start", "end", "image", "caption"}]}

                plus ``eda/summary.json`` ({"figures": [{"src": "x.png", "caption"}], "summary": {k: v}})
                and ``eda/*.png``. Images are copied to site/data/eda/, videos/images to site/media/.
``--gt``        optional labels in ground_truth.json shape. Enables metrics (via evaluate.py),
                per-event status, the confusion matrix and ground-truth lanes on the timeline.
                Without it, a render's ``"ground_truth": [[s, e, label]]`` (render.py --labels) is used.
``--ablations`` optional list of {"variant", "score_a", "f1_03", "f1_05", "f1_07", "score_b", "runtime_x", "note"}
``--failures``  optional list of {"video", "label", "start", "end", "kind", "why"} (hand-written analysis)

Output schema (site/data/results.json, version 1)
-------------------------------------------------
{
  "schema": 1, "generated_at": ISO-8601, "is_sample": false,
  "classes": [14 class ids, solution.CLASSES order],
  "summary": {"videos", "footage_sec", "events", "score_a", "score_b", "model_score",
              "runtime_x_realtime", "part_b": {"ap", "f1_alarm", "mtta_sec"} | null,
              "per_class": {label: {"f1_03", "f1_05", "f1_07", "f1_mean", "gt", "pred"}}},
  "videos": [{"id", "title", "duration", "fps", "width", "height", "recorded_at", "media", "poster",
              "events": [{"start", "end", "label", "score", "track_ids", "status": "tp"|"boundary"|"fp"|null}],
              "ground_truth": [[s, e, label]], "risk": [[t, p]] (<= 2 Hz), "stats": {...}, "notes"}],
  "examples": [{"label", "video", "start", "end", "image", "caption"}],
  "failures": [{"video", "label", "start", "end", "kind", "why"}],
  "eda": {"figures": [{"src", "caption"}], "summary": {label: value}},
  "ablations": [...], "confusion": {"labels": [..., "none"], "matrix": [[pred x gt]], "note"}
}
Missing inputs leave their sections empty; the page hides or explains empty sections.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RISK_HZ = 2.0          # risk curves are downsampled to this rate for the page
STATUS_TIOU = (0.5, 0.1)  # >= 0.5 same class: tp; >= 0.1: boundary error; else fp
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
           "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line",
           "congestion", "road_obstacle", "fire_smoke"]


def load(path: Path | None, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else default


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def downsample(curve: list, hz: float = RISK_HZ) -> list:
    """Keep the max score per 1/hz bin so short peaks survive."""
    out: dict[int, list] = {}
    for t, p in curve:
        k = int(float(t) * hz)
        if k not in out or p > out[k][1]:
            out[k] = [round(k / hz, 2), round(float(p), 3)]
    return [out[k] for k in sorted(out)]


def copy_to(src: Path, dest_dir: Path, rel_prefix: str) -> str | None:
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)
    return f"{rel_prefix}/{src.name}"


EDA_LABELS = {"videos": "Videos", "footage_min": "Total footage (min)", "resolutions": "Resolution",
              "fps": "Frame rate", "mean_vehicles_in_view": "Vehicles in view, mean",
              "max_vehicles_in_view": "Vehicles in view, max", "mean_brightness": "Mean brightness (0-1)"}


def humanize(summary: dict) -> dict:
    """scripts/eda.py keys (footage_min, vehicle_tracks, ...) -> table labels; lists joined."""
    out = {}
    for k, v in summary.items():
        label = EDA_LABELS.get(k) or (f"{k[:-7].replace('_', ' ').capitalize()} tracks" if k.endswith("_tracks")
                                      else k.replace("_", " ").capitalize())
        out[label] = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
    return out


def as_event(e) -> dict:
    if isinstance(e, dict):
        return {"start": float(e["start"]), "end": float(e["end"]), "label": e["label"],
                "score": e.get("score"), "track_ids": list(e.get("track_ids", []))}
    s, t, lab = e[:3]
    return {"start": float(s), "end": float(t), "label": lab, "score": None, "track_ids": []}


def event_status(ev: dict, gt_events: list) -> str:
    best_same = max((tiou((ev["start"], ev["end"]), (s, e)) for s, e, lab in gt_events if lab == ev["label"]),
                    default=0.0)
    if best_same >= STATUS_TIOU[0]:
        return "tp"
    return "boundary" if best_same >= STATUS_TIOU[1] else "fp"


def confusion(videos: list[dict], gt: dict) -> dict:
    """Rows = predicted class, columns = GT class, matched by best temporal overlap."""
    labels = [c for c in CLASSES if any(e["label"] == c for v in videos for e in v["events"])
              or any(lab == c for g in gt.values() for *_, lab in g["events"])] + ["none"]
    idx = {c: i for i, c in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for v in videos:
        g = gt.get(v["id"], {}).get("events", [])
        hit = set()
        for ev in v["events"]:
            scores = [(tiou((ev["start"], ev["end"]), (s, e)), j) for j, (s, e, _) in enumerate(g)]
            best = max(scores, default=(0.0, -1))
            if best[0] >= STATUS_TIOU[1]:
                m[idx[ev["label"]]][idx[g[best[1]][2]]] += 1
                hit.add(best[1])
            else:
                m[idx[ev["label"]]][idx["none"]] += 1
        for j, (_, _, lab) in enumerate(g):
            if j not in hit:
                m[idx["none"]][idx[lab]] += 1
    return {"labels": labels, "matrix": m,
            "note": "Rows: predicted class, columns: ground-truth class, matched on temporal overlap "
                    "(tIoU >= 0.1). 'none' = no overlapping event."}


def metrics(pred: dict, gt: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    import evaluate  # organizers' metric, imported read-only

    rep = evaluate.evaluate(gt, pred)
    a, b = rep["part_a"], rep["part_b"]
    per_class = {}
    for c in a["classes"]:
        pc = a["per_class"][c]
        per_class[c] = {"f1_03": round(pc["0.3"]["f1"], 3), "f1_05": round(pc["0.5"]["f1"], 3),
                        "f1_07": round(pc["0.7"]["f1"], 3), "f1_mean": round(pc["f1_mean"], 3),
                        "gt": pc["0.3"]["tp"] + pc["0.3"]["fn"], "pred": pc["0.3"]["tp"] + pc["0.3"]["fp"]}
    return {"score_a": round(a["score_a"], 4), "score_b": round(b["score_b"], 4) if b else None,
            "model_score": round(rep["model_score"], 4), "per_class": per_class,
            "part_b": {"ap": round(b["ap"], 3), "f1_alarm": round(b["f1_alarm"], 3),
                       "mtta_sec": round(b["mtta_sec"], 2)} if b else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", type=Path, default=ROOT / "predictions_samples.json")
    ap.add_argument("--renders", type=Path, default=ROOT / "renders")
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--ablations", type=Path)
    ap.add_argument("--failures", type=Path)
    ap.add_argument("--out", type=Path, default=SITE / "data" / "results.json")
    args = ap.parse_args()

    pred = load(args.pred, {"videos": {}})
    gt = load(args.gt, {}) or {}
    renders = {p.stem: load(p) for p in sorted(args.renders.glob("*.json"))} if args.renders.exists() else {}
    renders = {k: r for k, r in renders.items() if isinstance(r, dict) and "video" in r}  # per-video files only
    names = sorted(set(pred.get("videos", {})) | {r.get("video", f"{s}.mp4") for s, r in renders.items()})
    if not gt:  # fall back to labels embedded by render.py --labels
        for s, r in renders.items():
            if r.get("ground_truth"):
                gt[r.get("video", f"{s}.mp4")] = {"duration": r.get("duration"), "fps": r.get("fps"),
                                                   "events": r["ground_truth"]}
    scored = {"videos": {}}  # what evaluate.py sees: final events + full-rate risk

    videos, examples = [], []
    for name in names:
        stem = Path(name).stem
        r = renders.get(stem, {})
        p = pred.get("videos", {}).get(name, {})
        g = gt.get(name, {})
        events = [as_event(e) for e in (r.get("events") or p.get("events", []))]
        scored["videos"][name] = {"events": [[e["start"], e["end"], e["label"]] for e in events],
                                  "risk": p.get("risk") or r.get("risk") or []}
        if g:
            for ev in events:
                ev["status"] = event_status(ev, g["events"])
        media = copy_to(args.renders / r["annotated"], SITE / "media", "media") if r.get("annotated") else None
        poster = copy_to(args.renders / r["poster"], SITE / "media", "media") if r.get("poster") else None
        duration = r.get("duration") or g.get("duration") or max([e["end"] for e in events] + [0.0])
        videos.append({
            "id": name, "title": r.get("title", stem.replace("_", " ")), "duration": duration,
            "fps": r.get("fps") or g.get("fps", 25.0), "width": r.get("width"), "height": r.get("height"),
            "recorded_at": r.get("recorded_at"), "media": media, "poster": poster,
            "events": sorted(events, key=lambda e: e["start"]),
            "ground_truth": g.get("events", []), "risk": downsample(p.get("risk") or r.get("risk") or []),
            "stats": r.get("stats", {}), "notes": r.get("notes", ""),
        })
        for ex in r.get("examples", []):
            img = copy_to(args.renders / ex["image"], SITE / "media" / "examples", "media/examples") \
                if ex.get("image") else None
            examples.append({**ex, "video": name, "image": img})

    eda_dir = args.renders / "eda"
    eda = load(eda_dir / "summary.json", {}) or {}
    figures = []
    for fig in eda.get("figures", [{"src": p.name, "caption": p.stem.replace("_", " ")}
                                   for p in sorted(eda_dir.glob("*.png"))]):
        src = copy_to(eda_dir / fig["src"], SITE / "data" / "eda", "data/eda")
        if src:
            figures.append({"src": src, "caption": fig.get("caption", "")})

    runtimes = [v["stats"].get("runtime_x_realtime") for v in videos if v["stats"].get("runtime_x_realtime")]
    summary = {"videos": len(videos), "footage_sec": round(sum(v["duration"] for v in videos), 1),
               "events": sum(len(v["events"]) for v in videos), "score_a": None, "score_b": None,
               "model_score": None, "part_b": None, "per_class": {},
               "runtime_x_realtime": round(max(runtimes), 2) if runtimes else None}
    if gt:
        summary.update(metrics(scored, gt))

    out = {
        "schema": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_sample": False, "classes": CLASSES, "summary": summary, "videos": videos,
        "examples": examples, "failures": load(args.failures, []) or [],
        "eda": {"figures": figures, "summary": humanize(eda.get("summary", {}))},
        "ablations": load(args.ablations, []) or [],
        "confusion": confusion(videos, gt) if gt else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.pred.exists():  # published as a download in the Links section
        shutil.copy2(args.pred, args.out.parent / "predictions_samples.json")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(videos)} videos, {summary['events']} events, {len(figures)} EDA figures)")


if __name__ == "__main__":
    main()
