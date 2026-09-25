"""Fit per-class post-processing (and chosen rule parameters) to my_labels.json.

    python scripts/tune.py                                 # all labelled classes -> configs/params.tuned.json
    python scripts/tune.py --classes stopped_vehicle congestion --trials 500
    ICEBERG_PARAMS=configs/params.tuned.json python run_submission.py ...

Score A is a macro mean over classes, so each class is tuned on its own:
maximise that class's mean F1 over tIoU {0.3, 0.5, 0.7} (``evaluate.evaluate_part_a``)
over ``min_score``, ``start_offset``, ``end_offset``, ``merge_gap``, ``min_len``
plus the rule keys listed in ``RULE_SPACE``. Random search (the current
config is always trial 0) followed by a coordinate-descent polish. Rules are
re-run only when a rule key changes; post-processing trials reuse raw events.

With a handful of labelled events per class this overfits easily: a change
is kept only if it beats the current config by ``--min-gain``, and the report
shows TP/FP/FN so you can judge whether the gain is real.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from _common import CONFIG_DIR, DEFAULT_CACHE, DEFAULT_VIDEOS, ROOT, iter_observations, load_params, write_json
from evaluate import OFFICIAL_CLASSES, evaluate_part_a
from src.config import _deep_merge
from src.pipeline import build_context
from src.rules import RULES, Context, Event
from src.segments import finalize

OFFSETS = [round(x, 2) for x in np.arange(-2.0, 2.01, 0.25)]
POST_SPACE: dict[str, list] = {
    "min_score": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "start_offset": OFFSETS,
    "end_offset": OFFSETS,
    "merge_gap": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
    "min_len": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
}
# class-specific replacements of POST_SPACE entries
POST_OVERRIDES: dict[str, dict[str, list]] = {
    "stopped_vehicle": {"min_len": [8.0, 9.0, 10.0, 11.0, 12.0], "merge_gap": [0.0, 1.0, 2.0, 4.0]},
    "congestion": {"min_len": [5.0, 10.0, 15.0, 20.0, 30.0], "merge_gap": [2.0, 5.0, 10.0, 20.0],
                   "start_offset": [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
                   "end_offset": [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]},
    "road_obstacle": {"min_len": [2.0, 3.0, 5.0, 10.0], "merge_gap": [1.0, 2.0, 5.0, 10.0]},
    "fire_smoke": {"min_len": [2.0, 3.0, 5.0, 10.0], "merge_gap": [1.0, 3.0, 5.0, 10.0]},
}
# rule-specific keys (module DEFAULTS in src/rules/*, overridden by params.classes.<label>).
# Each new combination re-runs the rule on every video, so keep these grids small
# and include the default value.
RULE_SPACE: dict[str, dict[str, list]] = {
    "accident": {"contact_gap": [0.02, 0.05, 0.1], "min_drop": [1.0, 1.5, 2.0]},
    "near_miss": {"ttc_max": [1.0, 1.5, 2.0], "min_drop": [0.7, 1.0, 1.5]},
    "red_light": {"min_advance_heights": [0.5, 1.0, 1.5]},
    "wrong_way": {"max_cos": [-0.3, -0.5, -0.7], "min_run_sec": [0.5, 1.0, 2.0]},
    "illegal_u_turn": {"min_turn_deg": [130.0, 150.0, 165.0]},
    "stopped_vehicle": {"min_stop_sec": [8.0, 9.0, 10.0, 11.0], "queue_heights": [8.0, 12.0, 16.0]},
    "jaywalking": {"road_margin": [0.0, 0.15, 0.3], "min_run_sec": [0.5, 1.0, 1.5]},
    "failure_to_yield": {"crosswalk_margin": [0.2, 0.3, 0.5]},
    "solid_line_crossing": {"min_lateral_heights": [0.2, 0.3, 0.5]},
    "stop_line": {"min_stop_sec": [1.0, 1.5, 2.5]},
    "congestion": {"crawl_speed": [0.2, 0.3, 0.4], "min_vehicles": [3, 4, 5]},
    "fire_smoke": {"conf": [0.4, 0.5, 0.6], "min_hits": [2, 3, 4]},
}


def class_score(label: str, gt: dict, pred: dict[str, list]) -> dict:
    """Per-class mean F1 over tIoU thresholds (+ counts at 0.5) exactly as evaluate.py computes it."""
    g = {v: {"events": [e for e in d["events"] if e[2] == label]} for v, d in gt.items()}
    p = {v: {"events": pred.get(v, [])} for v in gt}
    rep = evaluate_part_a(g, p)
    if label not in rep["per_class"]:
        return {"f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    pc = rep["per_class"][label]
    return {"f1": pc["f1_mean"], "tp": pc["0.5"]["tp"], "fp": pc["0.5"]["fp"], "fn": pc["0.5"]["fn"]}


def post_process(raw: list[Event], label: str, cfg: dict, duration: float) -> list[list]:
    """Same thresholding + finalize as ``src.pipeline.run_rules``."""
    kept = [(e.start, e.end) for e in raw if e.score >= float(cfg.get("min_score", 0.0))]
    return [[s, e, label] for s, e in finalize(kept, duration, cfg)]


class ClassTuner:
    def __init__(self, label: str, contexts: dict[str, Context], gt: dict, base_cfg: dict):
        self.label, self.contexts, self.gt, self.base = label, contexts, gt, dict(base_cfg)
        self.space = {**POST_SPACE, **POST_OVERRIDES.get(label, {})}
        self.rule_space = RULE_SPACE.get(label, {})
        self._raw: dict[tuple, dict[str, list[Event]]] = {}

    def raw_events(self, cfg: dict) -> dict[str, list[Event]]:
        key = tuple(cfg.get(k) for k in sorted(self.rule_space))
        if key not in self._raw:
            out = {}
            for vid, ctx in self.contexts.items():
                saved = ctx.params["classes"][self.label]
                ctx.params["classes"][self.label] = cfg
                try:
                    out[vid] = RULES[self.label](ctx)
                finally:
                    ctx.params["classes"][self.label] = saved
            self._raw[key] = out
        return self._raw[key]

    def score(self, cfg: dict) -> dict:
        raw = self.raw_events(cfg)
        pred = {vid: post_process(raw[vid], self.label, cfg, ctx.meta.duration) for vid, ctx in self.contexts.items()}
        return class_score(self.label, self.gt, pred)

    def sample(self, rng: random.Random) -> dict:
        cfg = dict(self.base)
        for k, vals in itertools.chain(self.space.items(), self.rule_space.items()):
            cfg[k] = rng.choice(vals)
        return cfg

    def search(self, trials: int, seed: int, min_gain: float) -> tuple[dict, dict, dict]:
        """Returns (base score, best score, best cfg)."""
        rng = random.Random(seed)
        base_score = self.score(self.base)
        best_cfg, best = dict(self.base), base_score
        for _ in range(trials):
            cfg = self.sample(rng)
            sc = self.score(cfg)
            if sc["f1"] > best["f1"]:
                best_cfg, best = cfg, sc
        # coordinate descent from the best random point
        for _ in range(2):
            improved = False
            for k, vals in itertools.chain(self.space.items(), self.rule_space.items()):
                for v in vals:
                    cfg = {**best_cfg, k: v}
                    sc = self.score(cfg)
                    if sc["f1"] > best["f1"] + 1e-9:
                        best_cfg, best, improved = cfg, sc, True
            if not improved:
                break
        # revert changes that do not matter, so the override stays small and close to the defaults
        for k in [k for k in best_cfg if best_cfg[k] != self.base.get(k)]:
            cfg = {kk: vv for kk, vv in best_cfg.items() if kk != k}
            if k in self.base:
                cfg[k] = self.base[k]
            sc = self.score(cfg)
            if sc["f1"] >= best["f1"] - 1e-9:
                best_cfg, best = cfg, sc
        if best["f1"] < base_score["f1"] + min_gain:
            return base_score, base_score, dict(self.base)
        return base_score, best, best_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "my_labels.json")
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--params", type=Path, help="start from this override (e.g. a previous params.tuned.json)")
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "params.tuned.json")
    ap.add_argument("--classes", nargs="*", help="only these classes (default: every labelled class)")
    ap.add_argument("--trials", type=int, default=300, help="random trials per class")
    ap.add_argument("--min-gain", type=float, default=0.01, help="keep a change only if F1 improves by this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = json.loads(args.labels.read_text())
    if args.params is None and os.environ.get("ICEBERG_PARAMS"):
        args.params = Path(os.environ["ICEBERG_PARAMS"])
    params = load_params(args.params)
    contexts: dict[str, Context] = {}
    for _, obs in iter_observations(args.videos, params, args.cache, compute=False):
        if obs.meta.name in labels:
            contexts[obs.meta.name] = build_context(obs, params)
    gt = {v: labels[v] for v in contexts}
    missing = sorted(set(labels) - set(contexts))
    if missing:
        print(f"warning: no cached observations for {missing}; they are left out of tuning")
    if not gt:
        print("nothing to tune: no labelled video has cached observations", file=sys.stderr)
        return 2

    labelled = {e[2] for d in gt.values() for e in d["events"]}
    classes = args.classes or sorted(labelled, key=OFFICIAL_CLASSES.index)
    changes: dict[str, dict] = {}
    print(f"{len(gt)} video(s), {sum(len(d['events']) for d in gt.values())} labels\n")
    print(f"{'class':<20}{'n':>4}{'F1 before':>11}{'F1 after':>10}   TP/FP/FN@0.5 before -> after   changed")
    for label in classes:
        cfg = params["classes"].get(label, {})
        n = sum(e[2] == label for d in gt.values() for e in d["events"])
        if label not in RULES:
            print(f"{label:<20}{n:>4}   (no rule registered)")
            continue
        tuner = ClassTuner(label, contexts, gt, cfg)
        before, after, best = tuner.search(args.trials, args.seed, args.min_gain)
        diff = {k: v for k, v in best.items() if cfg.get(k) != v}
        if diff:
            changes[label] = diff
        fmt = lambda s: f"{s['tp']}/{s['fp']}/{s['fn']}"  # noqa: E731
        print(f"{label:<20}{n:>4}{before['f1']:>11.3f}{after['f1']:>10.3f}   {fmt(before):>9} -> {fmt(after):<9}"
              f"      {', '.join(f'{k}={v}' for k, v in diff.items()) or '-'}")
        if not cfg.get("enabled", False):
            print(f"{'':<20}    note: {label} is disabled in params; enable it to use these values")

    # classes predicted on the dev set but never labelled add a zero-F1 class to Score A
    for label in sorted(set(RULES) - labelled):
        cfg = params["classes"].get(label, {})
        if not cfg.get("enabled", False):
            continue
        n_pred = sum(len(post_process(RULES[label](ctx), label, cfg, ctx.meta.duration)) for ctx in contexts.values())
        if n_pred:
            print(f"warning: {n_pred} {label} prediction(s) but no {label} label: that class scores 0 on this set")

    base = json.loads(args.params.read_text()) if args.params else {}
    out = _deep_merge(base, {"classes": changes})
    write_json(args.out, out, indent=2)
    print(f"\nwrote {args.out} ({len(changes)} class(es) changed); use it with ICEBERG_PARAMS={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
