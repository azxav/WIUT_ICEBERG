"""Rules: accident and near_miss, from pairwise interactions of road users.

Both rules look at pairs of tracks (at least one vehicle) over the samples
they share, using ground footprints (bottom strip of the box, ``src.ttc``) so
that image-space occlusion between objects at different depths is not
mistaken for contact.

* accident  = footprints touch AND at least one object brakes abruptly or
  jerks its heading around that moment AND afterwards the involved objects
  come to rest (or one's track is lost mid-frame). Two cars that merely pass behind each other
  in the image keep a steady velocity and are rejected.
* near_miss = constant-velocity TTC drops below ~1.5 s while approaching AND
  one of them brakes hard or swerves AND they never touch.

Speeds are in box heights / s, decelerations in heights / s^2 (see src.tracks).
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..segments import flags_to_intervals
from ..tracks import Track
from ..ttc import box_gap, closing_speed, footprint, heading_change, speed_drop, ttc_boxes
from . import Context, Event, rule

DEFAULTS: dict[str, dict] = {
    "accident": {
        "foot_frac": 0.4,         # bottom fraction of the box used as ground footprint
        "contact_gap": 0.05,      # footprints closer than this (box heights) touch
        "episode_gap": 0.5,       # contact samples closer than this (s) form one episode
        "min_pre_speed": 1.0,     # someone must be moving at least this fast just before contact
        "decel_window": 1.0,      # s, window of the abrupt-change tests
        "min_drop": 1.5,          # speed drop (heights/s) within decel_window counted as abrupt
        "heading_jump": 0.8,      # rad, heading change within decel_window counted as abrupt
        "stationary_speed": 0.15,  # heights/s
        "still_sec": 2.0,         # must stay stationary this long after the crash
        "stop_within": 3.0,       # ... starting at most this long after first contact
        "leave_within": 5.0,      # or its track ends at most this long after first contact
        "min_duration": 2.0,      # floor on the emitted segment length
        "border_frac": 0.02,      # ignore contacts with a box this close to the frame edge (truncated boxes)
        "moving_before_sec": 2.0,
        "min_impact_speed": 0.5,  # heights/s closing speed just before contact
        "impact_window": 0.5,     # s
        "max_jump": 6.0,          # heights/s of raw box motion beyond which a track is an ID swap  # only objects moving in this window before contact must stop / leave
    },
    "near_miss": {
        "foot_frac": 0.4,
        "contact_gap": 0.05,
        "ttc_max": 1.5,           # s, dangerous constant-velocity TTC
        "min_closing": 0.3,       # heights/s, pair must be approaching
        "episode_gap": 0.5,
        "min_pre_speed": 0.8,
        "decel_window": 1.0,
        "min_drop": 1.0,          # hard braking: speed drop (heights/s) within decel_window
        "swerve_window": 0.5,
        "swerve": 0.5,            # rad of heading change within swerve_window
        "search_sec": 1.0,        # evasive action may start this long before the danger episode
        "clear_sec": 3.0,         # give up looking for "clear" after this long past the episode
    },
}


def _cfg(ctx: Context, label: str) -> dict:
    user = ctx.params.get("classes", {}).get(label, {})
    return {k: user.get(k, v) for k, v in DEFAULTS[label].items()}


def _road_users(ctx: Context) -> list[Track]:
    return [t for t in ctx.tracks if t.category in ("vehicle", "bicycle")] + ctx.pedestrians


def _pairs(tracks: list[Track]) -> Iterator[tuple[Track, Track]]:
    """Pairs overlapping in time with at least one vehicle."""
    tracks = sorted(tracks, key=lambda t: t.start)
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            if b.start > a.end:
                break
            if "vehicle" in (a.category, b.category):
                yield a, b


class _Pair:
    """Two tracks restricted to their common samples, with footprints and relative kinematics."""

    def __init__(self, a: Track, b: Track, foot_frac: float):
        self.a, self.b = a, b
        self.t, self.ia, self.ib = np.intersect1d(a.t, b.t, assume_unique=True, return_indices=True)
        ca, ha = footprint(a.box[self.ia], foot_frac)
        cb, hb = footprint(b.box[self.ib], foot_frac)
        self.h = (a.height[self.ia] + b.height[self.ib]) / 2
        self.gap = box_gap(ca, ha, cb, hb) / self.h
        self.ttc = ttc_boxes(ca, ha, a.vel[self.ia], cb, hb, b.vel[self.ib])
        self.closing = closing_speed(ca, a.vel[self.ia], cb, b.vel[self.ib]) / self.h

    def __len__(self) -> int:
        return len(self.t)


def _window(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t >= t0) & (t <= t1)


def _max_in(t: np.ndarray, values: np.ndarray, t0: float, t1: float) -> float:
    v = values[_window(t, t0, t1)]
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else 0.0


def _local(tr: Track, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, speed, heading) of ``tr`` restricted to [t0, t1] (keeps windowed ops cheap on long tracks)."""
    lo, hi = np.searchsorted(tr.t, [t0, t1], side="left")
    sl = slice(lo, min(hi + 1, len(tr)))
    return tr.t[sl], tr.speed[sl], tr.heading[sl]


def _stop_time(tr: Track, t0: float, t1: float, still_sec: float, stationary: float) -> float | None:
    """First time in [t0, t1] after which ``tr`` stays (>= 90 % of samples) slow for ``still_sec``."""
    slow = tr.speed < stationary
    for i in np.flatnonzero(_window(tr.t, t0, t1) & slow):
        if tr.end < tr.t[i] + 0.9 * still_sec:
            return None
        j = np.searchsorted(tr.t, tr.t[i] + still_sec, side="right")
        if slow[i:j].mean() >= 0.9:
            return float(tr.t[i])
    return None


# ----------------------------------------------------------------------------- accident
def _abrupt(tr: Track, tc: float, c: dict) -> float:
    """Strength of the kinematic shock of ``tr`` around contact time ``tc`` (>= 1 means abrupt)."""
    w = c["decel_window"]
    if _max_in(tr.t, tr.speed, tc - w, tc + 0.2) < c["min_pre_speed"]:
        return 0.0
    t, speed, heading = _local(tr, tc - 0.3 - w, tc + 1.5)
    heading = np.where(speed >= c["min_pre_speed"], heading, np.nan)  # heading of a crawling box is noise
    drop = _max_in(t, speed_drop(t, speed, w), tc - 0.3, tc + 1.5)
    turn = _max_in(t, heading_change(t, heading, w), tc - 0.3, tc + 1.5)
    return max(drop / c["min_drop"], turn / c["heading_jump"])


def _accident_episode(ctx: Context, p: _Pair, s: float, c: dict) -> Event | None:
    if ctx.at_border([tr.box[tr.index_at(s)] for tr in (p.a, p.b)], c["border_frac"]).any():
        return None  # truncated boxes jump in size / position at the frame edge
    if _max_in(p.t, p.closing, s - c["impact_window"], s + 0.1) < c["min_impact_speed"]:
        return None  # crept up to touching: a queue, not an impact
    if max(tr.max_jump(s - 1.5, s + 1.5) for tr in (p.a, p.b)) > c["max_jump"]:
        return None  # a box teleported: ID swap between neighbours, not a real shock
    shocks = [_abrupt(tr, s, c) for tr in (p.a, p.b)]
    if max(shocks) < 1.0:
        return None  # steady velocities through the overlap: occlusion, not contact
    ends, stopped = [], False
    for tr in (p.a, p.b):
        if _max_in(tr.t, tr.speed, s - c["moving_before_sec"], s + 0.2) < 0.5 * c["min_pre_speed"]:
            continue  # was already standing (parked / queued): it is hit, not required to stop
        stop = _stop_time(tr, s - 0.5, s + c["stop_within"], c["still_sec"], c["stationary_speed"])
        if stop is not None:
            ends.append(stop)
            stopped = True
        elif tr.end <= s + c["leave_within"] and not ctx.at_border(tr.box[-1], c["border_frac"])[0]:
            ends.append(tr.end)  # track lost mid-frame (occluded / wreck misdetected)
        else:
            return None  # keeps driving normally, or simply drives out of view
    if not stopped:
        return None
    end = max(max(ends), s + c["min_duration"])
    score = float(np.clip(max(shocks) / 2.0, 0.0, 1.0))
    return Event(s, end, "accident", score, (p.a.id, p.b.id), {"shock": [round(x, 2) for x in shocks]})


@rule("accident")
def accident(ctx: Context) -> list[Event]:
    c = _cfg(ctx, "accident")
    events: list[Event] = []
    for a, b in _pairs(_road_users(ctx)):
        p = _Pair(a, b, c["foot_frac"])
        if len(p) < 2 or p.gap.min() > c["contact_gap"]:
            continue
        for s, _ in flags_to_intervals(p.t, p.gap <= c["contact_gap"], max_gap=c["episode_gap"]):
            ev = _accident_episode(ctx, p, s, c)
            if ev is not None:
                events.append(ev)
    return events


# ----------------------------------------------------------------------------- near_miss
def _evasion(tr: Track, t0: float, t1: float, c: dict) -> tuple[float, float] | None:
    """(onset time, strength >= 1) of the strongest hard braking / swerve of ``tr`` in [t0, t1]."""
    dw, sw = c["decel_window"], c["swerve_window"]
    t, speed, heading = _local(tr, t0 - dw, t1)
    win = _window(t, t0, t1)
    if not win.any() or _max_in(t, speed, t0 - dw, t1) < c["min_pre_speed"]:
        return None
    drop = speed_drop(t, speed, dw) / c["min_drop"]
    turn = np.nan_to_num(heading_change(t, heading, sw)) / c["swerve"]
    strength = np.where(win, np.maximum(drop, turn), 0.0)
    k = int(np.argmax(strength))
    if strength[k] < 1.0:
        return None
    if drop[k] >= turn[k]:
        # braking onset = when the speed was highest just before the drop
        lo = np.searchsorted(t, t[k] - dw, side="left")
        onset = t[lo + int(np.nanargmax(speed[lo:k + 1]))]
    else:
        # swerve onset = first sample of this manoeuvre turning at a third of the threshold, minus its window
        lo = np.searchsorted(t, t[k] - sw, side="left")
        onset = t[lo + int(np.argmax(turn[lo:k + 1] >= 1 / 3))] - sw
    return float(onset), float(strength[k])


@rule("near_miss")
def near_miss(ctx: Context) -> list[Event]:
    c = _cfg(ctx, "near_miss")
    events: list[Event] = []
    for a, b in _pairs(_road_users(ctx)):
        p = _Pair(a, b, c["foot_frac"])
        if len(p) < 2:
            continue
        contact = p.gap <= c["contact_gap"]
        danger = (p.ttc <= c["ttc_max"]) & (p.closing >= c["min_closing"]) & ~contact
        for ds, de in flags_to_intervals(p.t, danger, max_gap=c["episode_gap"]):
            near = (p.t >= ds - c["search_sec"]) & (p.t <= de + c["clear_sec"])
            if contact[near].any():
                continue  # they touched: that is the accident rule's business
            moves = [m for m in (_evasion(tr, ds - c["search_sec"], de + 0.5, c) for tr in (a, b)) if m]
            if not moves:
                continue
            onset, strength = min(moves)[0], max(m[1] for m in moves)
            after = np.flatnonzero((p.t > de) & near & np.isinf(p.ttc) & (p.closing <= 0))
            end = float(p.t[after[0]]) if len(after) else min(de + c["clear_sec"], float(p.t[-1]))
            if end <= onset:
                continue
            events.append(Event(onset, end, "near_miss", float(np.clip(strength / 2.0, 0.0, 1.0)),
                                (a.id, b.id), {"min_ttc": round(float(p.ttc[(p.t >= ds) & (p.t <= de)].min()), 2)}))
    return events
