"""Rules for things that stand still on the road: stopped_vehicle, congestion, road_obstacle."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..observe import T_TIME, T_X1, T_X2, T_Y1, T_Y2
from ..segments import flags_to_intervals, hysteresis
from ..tracks import Track
from . import Context, Event, rule
from .road_users import carriageway_mask, rule_cfg, sample_index, union_events

DEFAULTS = {
    "stopped_vehicle": {
        "min_stop_sec": 10.0,          # non-excused stationary time needed
        "median_sec": 2.0,             # rolling median on speed before thresholding
        "gap_sec": 1.0,                # merge stationary runs split by jitter
        "require_arrival": True,       # must be seen moving before the stop (skips parked cars)
        "queue_heights": 12.0,         # queue reach behind a stop line, in box heights
        "queue_grace_sec": 5.0,        # queue still discharging this long after red ends
        "unknown_light_is_queue": True,  # near a stop line with unknown signal state -> queued
        "jam_neighbours": 3,           # >= this many other stationary vehicles nearby -> jam, not a stop
        "jam_radius_heights": 4.0,
        "rejoin_gap_sec": 5.0,         # join stationary runs of broken tracks at the same place
        "rejoin_iou": 0.5,
    },
    "congestion": {
        "min_vehicles": 4,             # vehicles present in the direction group
        "crawl_speed": 0.3,            # median speed below this (box heights / s)
        "smooth_sec": 4.0,             # moving average of the per-sample flag
        "on": 0.6,                     # hysteresis thresholds on the smoothed flag
        "off": 0.4,
        "min_dur_sec": 15.0,
        "gap_sec": 3.0,
        "exclude_signal_queue": True,  # drop short queues behind a red light of the same direction
        "max_signal_queue_sec": 120.0,
        "signal_red_share": 0.5,
        "parked_disp_heights": 1.0,    # tracks that never move this far ...
        "parked_min_sec": 60.0,        # ... over at least this long are parked cars, ignored
    },
    "road_obstacle": {
        "animal_classes": [15, 16, 17, 18, 19],  # COCO cat, dog, horse, sheep, cow (birds excluded)
        "road_margin": 0.0,
        "min_run_sec": 2.0,
        "gap_sec": 1.0,
        "static_blobs": False,         # thumbnail background-difference detector (experimental)
        "blob_diff": 25,               # grey-level difference from the median background
        "blob_persist_sec": 5.0,       # pixel must differ this long
        "blob_min_frac": 0.001,        # blob area as a share of the thumbnail
        "blob_box_pad": 0.1,           # track boxes (grown by this share) are not obstacles
    },
}
_REDDISH = ("red", "yellow")


def _rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    """Centred rolling median with edge padding (odd window ``n``)."""
    if n <= 1 or len(x) < 3:
        return x
    n = min(n | 1, len(x) if len(x) % 2 else len(x) - 1)
    xp = np.pad(x, n // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, n), axis=1)


def _dt(t: np.ndarray) -> float:
    return float(np.median(np.diff(t))) if len(t) > 1 else 1.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- stopped
@dataclass
class _Stop:
    start: float
    end: float
    box_end: np.ndarray  # last box of the run (the vehicle is still, so also where it stood)
    free: float        # seconds not explained by a signal queue or a jam
    arrived: bool      # vehicle was seen moving before it stopped
    ids: tuple[int, ...]
    score: float


def _stationary(ctx: Context, v: Track, cfg: dict) -> np.ndarray:
    n = int(round(cfg["median_sec"] / _dt(v.t)))
    slow = _rolling_median(v.speed, n) < float(ctx.params["tracks"]["stationary_speed"])
    return slow & carriageway_mask(ctx, v.foot)


def _queued(ctx: Context, v: Track, cfg: dict) -> np.ndarray:
    """Samples where the vehicle waits behind a stop line whose signal is red (or unknown)."""
    out = np.zeros(len(v), bool)
    for sl in ctx.scene.stop_lines:
        a, b = sl.line
        seg = b - a
        u = (v.foot - a) @ seg / max(float(seg @ seg), 1e-9)
        past = (v.foot - (a + b) / 2) @ sl.direction  # > 0 beyond the line
        near = (u > -0.25) & (u < 1.25) & (past <= 0.5 * v.height) & (past >= -cfg["queue_heights"] * v.height)
        for i in np.flatnonzero(near & ~out):
            state = ctx.light_at(sl.light, float(v.t[i]))
            out[i] = state in _REDDISH or (state == "unknown" and cfg["unknown_light_is_queue"])
    if out.any() and cfg["queue_grace_sec"] > 0:  # queue discharges a few seconds after green
        last = np.maximum.accumulate(np.where(out, v.t, -np.inf))
        out = v.t - last <= cfg["queue_grace_sec"]
    return out


@rule("stopped_vehicle")
def stopped_vehicle(ctx: Context) -> list[Event]:
    """Vehicle stationary on the carriageway for >= ``min_stop_sec``, not queued at a signal or in a jam.

    Start = first stationary sample (the moment it stopped), end = last one
    (it moves again or disappears). A stop is kept when its time not explained
    by a red-light queue (behind a stop line, within ``queue_heights``) or by
    a jam (``jam_neighbours`` other stationary vehicles close by) is still
    ``min_stop_sec``. Runs of broken tracks at the same place are joined.
    """
    cfg = rule_cfg(ctx, "stopped_vehicle", DEFAULTS["stopped_vehicle"])
    moving_speed = float(ctx.params["tracks"]["moving_speed"])
    stat = {v.id: _stationary(ctx, v, cfg) for v in ctx.vehicles}

    # sample index -> (track ids, feet) of stationary vehicles, for the jam test
    by_sample: dict[int, list[tuple[int, np.ndarray]]] = {}
    for v in ctx.vehicles:
        idx = sample_index(ctx.times, v.t)
        for i in np.flatnonzero(stat[v.id]):
            by_sample.setdefault(int(idx[i]), []).append((v.id, v.foot[i]))

    stops: list[_Stop] = []
    for v in ctx.vehicles:
        runs = flags_to_intervals(v.t, stat[v.id], cfg["gap_sec"], min_len=1.0)
        if not runs:
            continue
        idx = sample_index(ctx.times, v.t)
        excused = _queued(ctx, v, cfg)
        dt = np.diff(v.t, append=v.t[-1] + _dt(v.t))
        for s, e in runs:
            m = (v.t >= s) & (v.t <= e)
            for i in np.flatnonzero(m & ~excused):
                near = [f for tid, f in by_sample.get(int(idx[i]), []) if tid != v.id]
                if len(near) >= cfg["jam_neighbours"]:
                    d = np.linalg.norm(np.asarray(near) - v.foot[i], axis=1) / v.height[i]
                    excused[i] = (d <= cfg["jam_radius_heights"]).sum() >= cfg["jam_neighbours"]
            last = np.flatnonzero(m)[-1]
            stops.append(_Stop(s, e, v.box[last], float(dt[m & ~excused].sum()),
                               bool((v.speed[v.t < s] > moving_speed).any()), (v.id,),
                               float(v.conf[m].mean())))

    # join a stop that continues on a new track id (tracker lost it during the long stop)
    joined: list[_Stop] = []
    for st in sorted(stops, key=lambda x: x.start):
        prev = next((j for j in joined if -0.5 <= st.start - j.end <= cfg["rejoin_gap_sec"]
                     and _iou(j.box_end, st.box_end) >= cfg["rejoin_iou"]), None)
        if prev is None:
            joined.append(st)
        else:
            prev.end, prev.box_end = max(prev.end, st.end), st.box_end
            prev.free += st.free
            prev.ids += st.ids
            prev.score = max(prev.score, st.score)

    return [Event(j.start, j.end, "stopped_vehicle", j.score, j.ids, {"free_sec": round(j.free, 2)})
            for j in joined
            if j.free >= cfg["min_stop_sec"] and j.end - j.start >= cfg["min_stop_sec"]
            and (j.arrived or not cfg["require_arrival"])]


# -------------------------------------------------------------- congestion
def _direction_groups(ctx: Context, v: Track) -> tuple[list[str | None], dict[str, np.ndarray]]:
    """Per-sample direction group of a vehicle, and each group's travel direction.

    Groups are lane approaches when lanes are annotated, else four heading
    bins of the flow field's dominant direction at the vehicle's cell.
    """
    if ctx.scene.lanes:
        lanes = ctx.scene.lane_of(v.foot)
        keys = [(l.approach or l.id) if l is not None else None for l in lanes]
        dirs = {(l.approach or l.id): l.direction for l in ctx.scene.lanes}
        return keys, dirs
    angle, _, total = ctx.flow.dominant(v.foot)
    b = np.floor((angle + np.pi / 4) / (np.pi / 2)).astype(int) % 4
    ok = total >= float(ctx.params["flow"]["min_count"])
    keys = [f"dir{k}" if o else None for k, o in zip(b, ok)]
    dirs = {f"dir{k}": np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)]) for k in range(4)}
    return keys, dirs


def _group_light(ctx: Context, direction: np.ndarray) -> str | None:
    """Light of the stop line whose legal direction matches the group's (within 45 deg)."""
    best, best_dot = None, np.cos(np.pi / 4)
    for sl in ctx.scene.stop_lines:
        d = float(sl.direction @ direction)
        if sl.light and d >= best_dot:
            best, best_dot = sl.light, d
    return best


def _parked(v: Track, cfg: dict) -> bool:
    disp = np.linalg.norm(v.foot - v.foot[0], axis=1).max() / max(float(np.median(v.height)), 1.0)
    return disp < cfg["parked_disp_heights"] and v.end - v.start >= cfg["parked_min_sec"]


@rule("congestion")
def congestion(ctx: Context) -> list[Event]:
    """Crawling traffic across a direction: >= ``min_vehicles`` present with median speed < ``crawl_speed``.

    The per-sample flag is smoothed over ``smooth_sec`` with hysteresis, and
    runs of ``min_dur_sec`` become events. With ``exclude_signal_queue``, a
    run no longer than ``max_signal_queue_sec`` during which the group's
    light (stop line of the same direction) is red for ``signal_red_share``
    of the time is an ordinary red-light queue and is dropped.
    """
    cfg = rule_cfg(ctx, "congestion", DEFAULTS["congestion"])
    times = ctx.times
    if len(times) < 2:
        return []
    idx_all, grp_all, spd_all = [], [], []
    group_dirs: dict[str, np.ndarray] = {}
    for v in ctx.vehicles:
        if _parked(v, cfg):
            continue
        keys, dirs = _direction_groups(ctx, v)
        group_dirs.update(dirs)
        ok = np.array([k is not None for k in keys])
        idx_all.append(sample_index(times, v.t)[ok])
        grp_all += [k for k in keys if k is not None]
        spd_all.append(v.speed[ok])
    if not grp_all:
        return []
    idx, spd, grp = np.concatenate(idx_all), np.concatenate(spd_all), np.asarray(grp_all)

    n_smooth = int(np.clip(round(cfg["smooth_sec"] / _dt(times)), 1, len(times)))
    events: list[Event] = []
    for g in np.unique(grp):
        m = grp == g
        gi, gs = idx[m], spd[m]
        count = np.bincount(gi, minlength=len(times))
        med = np.full(len(times), np.inf)
        order = np.argsort(gi, kind="stable")
        samples, starts = np.unique(gi[order], return_index=True)
        for s, block in zip(samples, np.split(gs[order], starts[1:])):
            med[s] = np.median(block)
        flag = (count >= cfg["min_vehicles"]) & (med < cfg["crawl_speed"])
        smooth = np.convolve(flag.astype(float), np.ones(n_smooth) / n_smooth, mode="same")
        on = hysteresis(smooth, cfg["on"], cfg["off"])
        light = _group_light(ctx, group_dirs[g]) if cfg["exclude_signal_queue"] else None
        for s, e in flags_to_intervals(times, on, cfg["gap_sec"], cfg["min_dur_sec"]):
            span = (times >= s) & (times <= e)
            if light and e - s <= cfg["max_signal_queue_sec"]:
                red = np.mean([ctx.light_at(light, float(t)) in _REDDISH for t in times[span]])
                if red >= cfg["signal_red_share"]:
                    continue
            events.append(Event(s, e, "congestion", float(smooth[span].mean()), (),
                                {"group": str(g), "max_vehicles": int(count[span].max())}))
    return events


# ------------------------------------------------------------ road_obstacle
def _static_blobs(ctx: Context, cfg: dict) -> list[tuple[float, float, int, float]]:
    """Persistent foreground on the road in the 1 fps thumbnails, not covered by any track box.

    Background = per-pixel median over the whole clip, so an obstacle that
    stays for more than half the video is absorbed into it and missed.
    """
    obs = ctx.obs
    tt, th = obs.thumb_times, obs.thumbs
    if len(tt) < 3 or th.ndim != 3:
        return []
    T, h, w = th.shape
    sx, sy = w / ctx.meta.width, h / ctx.meta.height
    fg = np.abs(th.astype(np.int16) - np.median(th, axis=0).astype(np.int16)) > cfg["blob_diff"]

    ys, xs = np.mgrid[0:h, 0:w]
    centres = np.column_stack([(xs.ravel() + 0.5) / sx, (ys.ravel() + 0.5) / sy])
    fg &= carriageway_mask(ctx, centres).reshape(h, w)[None]

    rows = obs.tracks[np.argsort(obs.tracks[:, T_TIME])] if len(obs.tracks) else obs.tracks
    lo, hi = np.searchsorted(rows[:, T_TIME], tt - 0.25), np.searchsorted(rows[:, T_TIME], tt + 0.25)
    for k in range(T):
        for r in rows[lo[k]:hi[k]]:
            px, py = cfg["blob_box_pad"] * (r[T_X2] - r[T_X1]), cfg["blob_box_pad"] * (r[T_Y2] - r[T_Y1])
            x1, x2 = int((r[T_X1] - px) * sx), int(np.ceil((r[T_X2] + px) * sx))
            y1, y2 = int((r[T_Y1] - py) * sy), int(np.ceil((r[T_Y2] + py) * sy))
            fg[k, max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = False

    k = max(2, int(round(cfg["blob_persist_sec"] / _dt(tt))))
    if T < k:
        return []
    cs = np.concatenate([np.zeros((1, h, w), np.int32), np.cumsum(fg, axis=0, dtype=np.int32)])
    persistent = (cs[k:] - cs[:-k]) == k  # window starting at thumb j: foreground throughout
    min_area = cfg["blob_min_frac"] * h * w
    win_ok = np.zeros(len(persistent), bool)
    for j, mask in enumerate(persistent):
        if mask.sum() >= min_area:
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            win_ok[j] = stats[1:, cv2.CC_STAT_AREA].max(initial=0) >= min_area
    flags = np.convolve(win_ok, np.ones(k, int))[:T] > 0  # thumbs covered by a persistent window
    return [(s, e, -1, 1.0) for s, e in flags_to_intervals(tt, flags, cfg["gap_sec"])]


@rule("road_obstacle")
def road_obstacle(ctx: Context) -> list[Event]:
    """Animals on the carriageway for >= ``min_run_sec`` (plus optional static blobs)."""
    cfg = rule_cfg(ctx, "road_obstacle", DEFAULTS["road_obstacle"])
    classes = set(int(c) for c in cfg["animal_classes"])
    items: list[tuple[float, float, int, float]] = []
    for a in ctx.tracks:
        if a.category != "animal" or int(np.bincount(a.cls.astype(int)).argmax()) not in classes:
            continue
        on_road = carriageway_mask(ctx, a.foot, cfg["road_margin"] * a.height)
        for s, e in flags_to_intervals(a.t, on_road, cfg["gap_sec"], cfg["min_run_sec"]):
            m = (a.t >= s) & (a.t <= e)
            items.append((s, e, a.id, float(a.conf[m].mean())))
    if cfg["static_blobs"]:
        items += _static_blobs(ctx, cfg)
    return union_events(items, "road_obstacle")
