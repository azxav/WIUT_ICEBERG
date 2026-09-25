"""Part A end to end: video -> observations -> rules -> clean segments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, load_params, scene_path
from .flowfield import FlowField
from .observe import Observations, observe
from .perception import Detector
from .rules import RULES, Context, Event
from .scene import Scene
from .segments import finalize
from .tracks import build_tracks
from .video import probe

FLOW_PRIOR = CONFIG_DIR / "flow_prior.npz"


@dataclass
class Analysis:
    events: list[Event]        # final events (after thresholds, offsets, merging)
    raw_events: list[Event]    # what the rules emitted
    ctx: Context

    def as_lists(self) -> list[list]:
        return [[e.start, e.end, e.label] for e in sorted(self.events, key=lambda e: (e.start, e.label))]


def build_context(obs: Observations, params: dict, scene: Scene | None = None) -> Context:
    meta = obs.meta
    scene = scene or Scene.load(scene_path(), meta.width, meta.height)
    tracks = build_tracks(obs, params)
    prior = FlowField.load(FLOW_PRIOR, meta.width, meta.height)
    flow = FlowField.from_tracks(tracks, meta.width, meta.height, tuple(params["flow"]["grid"]), prior)
    return Context(obs, tracks, scene, flow, params)


def run_rules(ctx: Context, classes: list[str]) -> tuple[list[Event], list[Event]]:
    raw: list[Event] = []
    final: list[Event] = []
    duration = ctx.meta.duration
    for label in classes:
        cfg = ctx.params["classes"].get(label, {})
        if not cfg.get("enabled", False) or label not in RULES:
            continue
        events = RULES[label](ctx)
        raw += events
        kept = [e for e in events if e.score >= float(cfg.get("min_score", 0.0))]
        for s, e in finalize([(ev.start, ev.end) for ev in kept], duration, cfg):
            ids = tuple(sorted({i for ev in kept if ev.start < e and ev.end > s for i in ev.track_ids}))
            final.append(Event(s, e, label, 1.0, ids))
    return final, raw


def analyze(video_path: str | Path, classes: list[str], params: dict | None = None,
            detector: Detector | None = None, obs: Observations | None = None) -> Analysis:
    params = params or load_params()
    if obs is None:
        meta = probe(video_path)
        scene = Scene.load(scene_path(), meta.width, meta.height)
        obs = observe(video_path, params, scene, detector)
    ctx = build_context(obs, params)
    final, raw = run_rules(ctx, classes)
    return Analysis(final, raw, ctx)
