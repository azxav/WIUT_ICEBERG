"""Rule registry: one function per event class, turning a ``Context`` into events.

A rule is ``fn(ctx) -> list[Event]`` registered with ``@rule("label")``. Rules
read only arrays (tracks, scene, flow field, light states), never pixels, so
they re-run in seconds from cached observations while tuning. They return
raw events; ``src.pipeline`` applies score thresholds, offsets and merging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np

from ..flowfield import FlowField
from ..observe import Observations
from ..scene import Scene
from ..tracks import Track
from ..video import VideoMeta


@dataclass
class Event:
    start: float
    end: float
    label: str
    score: float = 1.0                        # rule confidence, thresholded by classes.<label>.min_score
    track_ids: tuple[int, ...] = ()           # objects involved (for rendering / error analysis)
    info: dict = field(default_factory=dict)  # free-form debug values


@dataclass
class Context:
    obs: Observations
    tracks: list[Track]
    scene: Scene
    flow: FlowField
    params: dict

    @property
    def meta(self) -> VideoMeta:
        return self.obs.meta

    @property
    def times(self) -> np.ndarray:
        return self.obs.sample_times

    def cfg(self, label: str) -> dict:
        return self.params["classes"][label]

    def at_border(self, boxes: np.ndarray, frac: float) -> np.ndarray:
        """True for xyxy boxes within ``frac`` of the frame width of an edge (truncated boxes)."""
        b = np.asarray(boxes, float).reshape(-1, 4)
        w, h = self.meta.width, self.meta.height
        m = frac * w
        return (b[:, 0] <= m) | (b[:, 1] <= m) | (b[:, 2] >= w - m) | (b[:, 3] >= h - m)

    @cached_property
    def vehicles(self) -> list[Track]:
        return [t for t in self.tracks if t.category == "vehicle"]

    @cached_property
    def pedestrians(self) -> list[Track]:
        return [t for t in self.tracks if t.category == "person" and not t.is_rider]

    @cached_property
    def light_states(self) -> dict[str, np.ndarray]:
        """light id -> per-sample state ('red' / 'yellow' / 'green' / 'unknown')."""
        from ..lights import classify_lights

        return classify_lights(self.obs, self.scene, self.params)

    def light_at(self, light_id: str | None, t: float) -> str:
        states = self.light_states.get(light_id) if light_id else None
        if states is None or len(states) == 0:
            return "unknown"
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(states) - 1))
        return str(states[i])


RuleFn = Callable[[Context], list[Event]]
RULES: dict[str, RuleFn] = {}


def rule(label: str) -> Callable[[RuleFn], RuleFn]:
    def deco(fn: RuleFn) -> RuleFn:
        RULES[label] = fn
        return fn
    return deco


# Import rule modules for their registration side effects.
from . import fire, interactions, road_users, stationary, violations  # noqa: E402,F401
