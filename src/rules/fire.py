"""Rule fire_smoke: persistent detections of the optional fire/smoke detector."""
from __future__ import annotations

import numpy as np

from ..observe import F_CLS, F_CONF, F_TIME
from ..segments import flags_to_intervals
from . import Context, Event, rule
from .road_users import rule_cfg, sample_index

DEFAULTS = {
    "fire_smoke": {
        "conf": 0.5,         # max detection confidence in a fire sample
        "window": 5,         # persistence: >= min_hits flagged samples ...
        "min_hits": 3,       # ... in some window of this many consecutive fire samples
        "gap_samples": 3,    # merge runs separated by up to this many fire samples
        "classes": None,     # detector class ids to keep; None = every class is fire/smoke
    },
}


def fire_sample_times(times: np.ndarray, every: float) -> np.ndarray:
    """Timestamps at which ``observe`` ran the fire detector (first sample >= each deadline)."""
    out, nxt = [], 0.0
    for t in times:
        if t >= nxt:
            out.append(float(t))
            nxt = float(t) + every
    return np.asarray(out, float)


@rule("fire_smoke")
def fire_smoke(ctx: Context) -> list[Event]:
    """Fire/smoke seen in >= ``min_hits`` of ``window`` consecutive detector samples.

    Start = first confirmed detection, end = last detection + one sample
    interval (the fire is visible until the next, empty, sample).
    """
    rows = ctx.obs.fire
    if not len(rows):
        return []
    cfg = rule_cfg(ctx, "fire_smoke", DEFAULTS["fire_smoke"])
    if cfg["classes"] is not None:
        rows = rows[np.isin(rows[:, F_CLS].astype(int), [int(c) for c in cfg["classes"]])]
    every = float(ctx.params["perception"].get("fire_every_sec", 1.0))
    st = fire_sample_times(ctx.times, every)
    if not len(rows) or not len(st):
        return []

    best = np.zeros(len(st))
    np.maximum.at(best, sample_index(st, rows[:, F_TIME]), rows[:, F_CONF])
    hit = best >= cfg["conf"]
    win = int(min(cfg["window"], len(st)))
    ok_win = np.convolve(hit, np.ones(win, int), mode="valid") >= min(cfg["min_hits"], win)
    covered = np.convolve(ok_win, np.ones(win, int))[:len(st)] > 0  # sample lies in a passing window
    confirmed = hit & covered

    events = []
    for s, e in flags_to_intervals(st, confirmed, cfg["gap_samples"] * every):
        m = (st >= s) & (st <= e) & hit
        events.append(Event(s, e + every, "fire_smoke", float(best[m].mean()), (),
                            {"hits": int(m.sum())}))
    return events
