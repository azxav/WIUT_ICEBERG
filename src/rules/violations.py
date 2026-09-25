"""Rules: traffic-signal and lane-discipline violations.

All rules work on the smoothed foot point of vehicle tracks (bottom centre of
the box, where the vehicle meets the road) against the scene layout. A rule
whose scene elements are missing returns no events. Thresholds live in
``DEFAULTS`` and can be overridden per class in ``params["classes"][label]``.

Turn boundaries (U-turns, illegal turns) are found on the smoothed, unwrapped
heading: the turn is detected where the heading crosses an onset / settle
threshold, then extrapolated back / forward at the local turn rate so the
boundaries estimate when the turn really started and ended.
"""
from __future__ import annotations

import numpy as np

from ..scene import StopLine, segment_crossings, side_of_line
from ..segments import flags_to_intervals
from ..tracks import Track, _smooth
from . import Context, Event, rule

DEFAULTS: dict[str, dict] = {
    "red_light": {
        "min_advance_heights": 1.0,   # must get this far past the line (else it is a stop_line case)
        "max_len": 10.0,              # cap when the track never leaves the intersection polygon
        "full_score_red_sec": 2.0,    # red for this long before crossing -> score 1
    },
    "stop_line": {
        "min_stop_sec": 1.5,          # stationary on red for at least this long
        "max_past_heights": 1.5,      # foot at most this far past the line (else inside the junction)
        "lateral_margin": 0.1,        # tolerance beyond the stop-line ends, fraction of its length
        "approach_sec": 2.0,          # look-back to check it arrived in the line's legal direction
        "min_approach_heights": 0.5,
    },
    "wrong_way": {
        "max_cos": -0.5,              # heading . lane direction below this -> against the lane
        "smooth_sec": 0.5,            # majority filter on per-sample flags
        "gap_sec": 0.5,
        "min_run_sec": 1.0,
    },
    "illegal_turn": {
        "onset_deg": 20.0,
        "settle_deg": 15.0,
        "max_extrap_sec": 1.0,
        "heading_smooth_sec": 0.5,
        "ref_sec": 1.0,               # initial / final heading = median over this much of moving track
        "min_zone_sec": 0.3,          # zone visits shorter than this are ignored
    },
    "illegal_u_turn": {
        "min_turn_deg": 150.0,
        "max_turn_sec": 20.0,
        "onset_deg": 20.0,
        "settle_deg": 15.0,
        "max_extrap_sec": 1.0,
        "heading_smooth_sec": 0.5,
        "min_path_heights": 1.5,      # rejects heading noise of near-stationary boxes
    },
    "solid_line_crossing": {
        "max_lead_sec": 1.0,          # wheel-to-centre lead / lag cap
        "min_lateral_heights": 0.3,   # must be this far off the line on both sides (jitter rejection)
        "window_sec": 3.0,            # ... within this long before / after the crossing
    },
}


# ------------------------------------------------------------------ helpers
def _cfg(ctx: Context, label: str) -> dict:
    return {**DEFAULTS[label], **ctx.params.get("classes", {}).get(label, {})}


def _cross_time(t: np.ndarray, i: int, u: float) -> float:
    return float(t[i] + u * (t[i + 1] - t[i]))


def _states_at(ctx: Context, light_id: str | None, ts: np.ndarray) -> np.ndarray:
    """Light state at each time in ``ts`` (vectorised ``ctx.light_at``)."""
    states = ctx.light_states.get(light_id) if light_id else None
    if states is None or len(states) == 0:
        return np.full(len(ts), "unknown", dtype=object)
    idx = np.clip(np.searchsorted(ctx.times, ts, side="right") - 1, 0, len(states) - 1)
    return np.asarray(states, dtype=object)[idx]


def _red_for(ctx: Context, light_id: str, t: float) -> float:
    """How long the light has been continuously red at time ``t``."""
    states = ctx.light_states[light_id]
    k = i = int(np.clip(np.searchsorted(ctx.times, t, side="right") - 1, 0, len(states) - 1))
    while k > 0 and states[k - 1] == "red":
        k -= 1
    return float(t - ctx.times[k]) if states[i] == "red" else 0.0


def _normal(sl: StopLine) -> np.ndarray:
    """Unit normal of the stop line pointing to its far (legal exit) side."""
    d = sl.line[1] - sl.line[0]
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) or 1.0)
    return n if n @ sl.direction >= 0 else -n


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of True runs."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _odd(sec: float, t: np.ndarray) -> int:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return max(1, int(round(sec / max(dt, 1e-6)))) | 1


def _heading(tr: Track, smooth_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """(indices of moving samples, their smoothed unwrapped heading in rad)."""
    idx = np.flatnonzero(~np.isnan(tr.heading))
    if len(idx) < 3:
        return idx[:0], np.zeros(0)
    return idx, _smooth(np.unwrap(tr.heading[idx]), _odd(smooth_sec, tr.t[idx]))


def _turn_bounds(t: np.ndarray, th: np.ndarray, base: float, final: float, i0: int, i1: int,
                 cfg: dict) -> tuple[float, float] | None:
    """(start, end) of a turn from heading ``base`` to ``final`` within samples ``i0..i1``."""
    sgn = np.sign(final - base)
    onset, settle = np.deg2rad(cfg["onset_deg"]), np.deg2rad(cfg["settle_deg"])
    dev = sgn * (th[i0:i1 + 1] - base)
    on = np.flatnonzero(dev > onset)
    if sgn == 0 or not on.size:
        return None
    on = i0 + on[0]
    done = np.flatnonzero(sgn * (final - th[on:i1 + 1]) <= settle)
    off = on + done[0] if done.size else i1
    rate = np.abs(np.gradient(th, t)) if len(t) > 1 else np.zeros(len(t))
    cap = cfg["max_extrap_sec"]
    lead = min(onset / rate[on], cap) if rate[on] > 1e-6 else 0.0
    lag = min(settle / rate[off], cap) if rate[off] > 1e-6 else 0.0
    return float(max(t[on] - lead, t[i0])), float(min(t[off] + lag, t[i1]))


def _u_turns(ctx: Context, cfg: dict) -> list[tuple[Track, float, float, float]]:
    """(track, start, end, turned rad) for every heading reversal of a moving vehicle."""
    out = []
    thr, max_sec = np.deg2rad(cfg["min_turn_deg"]), cfg["max_turn_sec"]
    for tr in ctx.vehicles:
        idx, th = _heading(tr, cfg["heading_smooth_sec"])
        t = tr.t[idx]
        first, k = 0, 0
        while k < len(t):
            lo = max(first, int(np.searchsorted(t, t[k] - max_sec)))
            dev = th[k] - th[lo:k + 1]
            r = int(np.argmax(np.abs(dev)))
            if abs(dev[r]) < thr:
                k += 1
                continue
            i, sgn = lo + r, np.sign(dev[r])
            hi = int(np.searchsorted(t, t[k] + max_sec, side="right"))
            peak = k + int(np.argmax(sgn * th[k:hi]))  # heading once the turn has finished
            bounds = _turn_bounds(t, th, th[i], th[peak], i, peak, cfg)
            first = k = peak + 1
            if bounds is None:
                continue
            seg = (tr.t >= bounds[0]) & (tr.t <= bounds[1])
            path = np.linalg.norm(np.diff(tr.foot[seg], axis=0), axis=1).sum()
            if path >= cfg["min_path_heights"] * float(np.median(tr.height[seg])):
                out.append((tr, bounds[0], bounds[1], float(abs(th[peak] - th[i]))))
    return out


def _zone_visits(zones: list[str | None], t: np.ndarray, min_sec: float) -> list[tuple[str, int, int]]:
    """(zone id, first index, last index) of successive zone visits lasting >= ``min_sec``."""
    visits: list[tuple[str, int, int]] = []
    s = 0
    for k in range(1, len(zones) + 1):
        if k < len(zones) and zones[k] == zones[s]:
            continue
        z = zones[s]
        if z is not None and t[k - 1] - t[s] >= min_sec:
            if visits and visits[-1][0] == z:
                visits[-1] = (z, visits[-1][1], k - 1)
            else:
                visits.append((z, s, k - 1))
        s = k
    return visits


# -------------------------------------------------------------------- rules
@rule("red_light")
def red_light(ctx: Context) -> list[Event]:
    """Foot crosses a stop line in its legal direction while its light is red.

    Start: interpolated crossing time. End: leaves the intersection polygon
    (or the frame; capped at ``max_len`` without an intersection polygon).
    """
    cfg = _cfg(ctx, "red_light")
    lines = [sl for sl in ctx.scene.stop_lines if sl.light]
    events: list[Event] = []
    for tr in ctx.vehicles if lines else []:
        for sl in lines:
            past = (tr.foot - sl.line[0]) @ _normal(sl)  # signed px beyond the line
            for i, u in segment_crossings(tr.foot, sl.line[0], sl.line[1]):
                if (tr.foot[i + 1] - tr.foot[i]) @ sl.direction <= 0:
                    continue
                if past[i + 1:].max() < cfg["min_advance_heights"] * tr.height[i]:
                    continue
                tc = _cross_time(tr.t, i, u)
                if ctx.light_at(sl.light, tc) != "red":
                    continue
                end = min(_leave_intersection(ctx, tr, i + 1), tc + cfg["max_len"])
                red_for = _red_for(ctx, sl.light, tc)
                score = float(np.clip(0.5 + 0.5 * red_for / cfg["full_score_red_sec"], 0.0, 1.0))
                events.append(Event(tc, end, "red_light", score, (tr.id,),
                                    {"stop_line": sl.id, "red_for": red_for}))
                break  # one event per vehicle and stop line
    return events


def _leave_intersection(ctx: Context, tr: Track, i0: int) -> float:
    """Time the track leaves the intersection after sample ``i0`` (track end if never)."""
    inside = ctx.scene.in_intersection(tr.foot[i0:])
    if inside.any():
        j = int(np.argmax(inside))
        out = np.flatnonzero(~inside[j:])
        if out.size:
            return float(tr.t[i0 + j + out[0]])
    return tr.end


@rule("stop_line")
def stop_line(ctx: Context) -> list[Event]:
    """Vehicle comes to rest just past a stop line while its light is red.

    Start: stop time (or red onset if it stopped earlier). End: light turns
    non-red, the vehicle moves off, or the track ends.
    """
    cfg = _cfg(ctx, "stop_line")
    lines = [sl for sl in ctx.scene.stop_lines if sl.light]
    still_speed = cfg.get("stationary_speed", ctx.params["tracks"]["stationary_speed"])
    events: list[Event] = []
    for tr in ctx.vehicles if lines else []:
        still = tr.speed < still_speed
        for sl in lines:
            a, d = sl.line[0], sl.line[1] - sl.line[0]
            along = ((tr.foot - a) @ d) / max(float(d @ d), 1e-9)
            past = (tr.foot - a) @ _normal(sl)
            m = cfg["lateral_margin"]
            near = (along >= -m) & (along <= 1 + m) & (past > 0) & (past <= cfg["max_past_heights"] * tr.height)
            red = _states_at(ctx, sl.light, tr.t) == "red"
            for s, e in _runs(still & near):
                # it must have driven up to the line in its legal direction
                k = tr.index_at(tr.t[s] - cfg["approach_sec"])
                if (tr.foot[s] - tr.foot[k]) @ sl.direction < cfg["min_approach_heights"] * tr.height[s]:
                    continue
                on = np.flatnonzero(red[s:e + 1])
                if not on.size:
                    continue
                k0 = s + on[0]
                off = np.flatnonzero(~red[k0:e + 1])
                start, end = float(tr.t[k0]), float(tr.t[k0 + off[0]] if off.size else tr.t[e])
                if end - start >= cfg["min_stop_sec"]:
                    events.append(Event(start, end, "stop_line", 1.0, (tr.id,),
                                        {"stop_line": sl.id, "past_px": float(past[k0])}))
    return events


@rule("wrong_way")
def wrong_way(ctx: Context) -> list[Event]:
    """Vehicle heads against its lane (or, without lanes, against the learned flow)."""
    cfg = _cfg(ctx, "wrong_way")
    fp = ctx.params["flow"]
    events: list[Event] = []
    for tr in ctx.vehicles:
        if len(tr) < 3:
            continue
        if ctx.scene.lanes:
            dirs = np.array([l.direction if l is not None else (np.nan, np.nan)
                             for l in ctx.scene.lane_of(tr.foot)], float)
            cos = np.cos(tr.heading) * dirs[:, 0] + np.sin(tr.heading) * dirs[:, 1]
            raw = cos < cfg["max_cos"]  # NaN (slow / outside lanes) compares False
        else:
            raw = ctx.flow.opposite(tr.foot, tr.heading, fp["min_count"], fp["dominance"])
        flags = _smooth(raw.astype(float), _odd(cfg["smooth_sec"], tr.t)) >= 0.5
        for s, e in flags_to_intervals(tr.t, flags, cfg["gap_sec"], cfg["min_run_sec"]):
            share = float(raw[(tr.t >= s) & (tr.t <= e)].mean())
            events.append(Event(s, e, "wrong_way", share, (tr.id,)))
    return events


@rule("illegal_u_turn")
def illegal_u_turn(ctx: Context) -> list[Event]:
    """Heading reverses (>= ~150 deg within ~20 s) where U-turns are not allowed."""
    if ctx.scene.u_turn_allowed:
        return []
    cfg = _cfg(ctx, "illegal_u_turn")
    return [Event(s, e, "illegal_u_turn", min(1.0, turned / np.pi), (tr.id,),
                  {"turned_deg": float(np.rad2deg(turned))})
            for tr, s, e, turned in _u_turns(ctx, cfg)]


@rule("illegal_turn")
def illegal_turn(ctx: Context) -> list[Event]:
    """Entry -> exit zone pair not in ``allowed_movements``; bounded by the turn itself.

    U-turning tracks are left to ``illegal_u_turn``.
    """
    sc = ctx.scene
    if not sc.zones or sc.allowed_movements is None:
        return []
    cfg = _cfg(ctx, "illegal_turn")
    u_ids = {tr.id for tr, *_ in _u_turns(ctx, _cfg(ctx, "illegal_u_turn"))}
    events: list[Event] = []
    for tr in ctx.vehicles:
        if tr.id in u_ids:
            continue
        visits = _zone_visits(sc.zone_of(tr.foot), tr.t, cfg["min_zone_sec"])
        if len(visits) < 2:
            continue
        (entry, e0, e1), (exit_, x0, x1) = visits[0], visits[-1]
        if entry == exit_ or (entry, exit_) in sc.allowed_movements:
            continue
        bounds = _movement_turn(tr, e0, x1, cfg)
        if bounds is not None:
            start, end, score = *bounds, 1.0
        else:  # no clear turn (e.g. prohibited straight-on): from leaving entry to reaching exit
            start, end, score = float(tr.t[e1]), float(tr.t[x0]), 0.6
        if end > start:
            events.append(Event(start, end, "illegal_turn", score, (tr.id,),
                                {"entry": entry, "exit": exit_}))
    return events


def _movement_turn(tr: Track, i0: int, i1: int, cfg: dict) -> tuple[float, float] | None:
    """Turn boundaries between track samples ``i0..i1`` from initial to final heading."""
    idx, th = _heading(tr, cfg["heading_smooth_sec"])
    keep = (idx >= i0) & (idx <= i1)
    idx, th = idx[keep], th[keep]
    if len(idx) < 3:
        return None
    t = tr.t[idx]
    base = float(np.median(th[t <= t[0] + cfg["ref_sec"]]))
    final = float(np.median(th[t >= t[-1] - cfg["ref_sec"]]))
    return _turn_bounds(t, th, base, final, 0, len(t) - 1, cfg)


@rule("solid_line_crossing")
def solid_line_crossing(ctx: Context) -> list[Event]:
    """Foot path crosses a solid lane marking, with a clear excursion on both sides.

    Start / end: crossing time -/+ the time half the box width needs to cross
    at the lateral speed (wheel touches the line -> fully in the new lane).
    """
    cfg = _cfg(ctx, "solid_line_crossing")
    moving = ctx.params["tracks"]["moving_speed"]
    win = cfg["window_sec"]
    events: list[Event] = []
    for tr in ctx.vehicles if ctx.scene.solid_lines else []:
        t = tr.t
        for pl in ctx.scene.solid_lines:
            for a, b in zip(pl.points[:-1], pl.points[1:]):
                seg_len = float(np.linalg.norm(b - a))
                if seg_len == 0:
                    continue
                dist = side_of_line(np.array([a, b]), tr.foot) / seg_len  # signed px from the line
                for i, u in segment_crossings(tr.foot, a, b):
                    if tr.speed[i] < moving:
                        continue
                    tc = _cross_time(t, i, u)
                    lo, hi = np.searchsorted(t, [tc - win, tc + win])
                    s0 = np.sign(dist[i]) or -np.sign(dist[i + 1])
                    before = float((s0 * dist[lo:i + 1]).max())
                    after = float((-s0 * dist[i + 1:max(hi, i + 2)]).max())
                    excursion = min(before, after) / tr.height[i]
                    if excursion < cfg["min_lateral_heights"]:
                        continue
                    k0, k1 = tr.index_at(tc - 0.25), tr.index_at(tc + 0.25)
                    v_lat = abs(dist[k1] - dist[k0]) / max(t[k1] - t[k0], 1e-6)
                    half_w = (tr.box[i, 2] - tr.box[i, 0]) / 2
                    lead = min(half_w / v_lat, cfg["max_lead_sec"]) if v_lat > 0 else cfg["max_lead_sec"]
                    events.append(Event(float(tc - lead), float(tc + lead), "solid_line_crossing",
                                        float(min(1.0, excursion)), (tr.id,), {"line": pl.id}))
    return events
