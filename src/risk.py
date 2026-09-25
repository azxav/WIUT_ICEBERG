"""Part B: causal accident-risk estimator.

``CausalRisk.step(frame, t)`` sees frames strictly in order. Every
``risk.stride``-th frame (by time) it runs its own detector (small input size) + ByteTrack
and ``update()``s per-track short histories; other calls return the last
score in O(1). The pipeline per detection step:

1. causal kinematics per track: least-squares velocity over the last
   ``vel_sec`` of foot positions, speed in box heights / s, deceleration as
   the speed drop over the last ``decel_sec``;
2. ``score_features`` (pure, no pixels): snapshot features -> logistic score.
   Main cue is the minimum 2D TTC over approaching footprint pairs
   (``src.ttc``), boosted by hard braking;
3. temporal shaping for the alarm metric: EMA smoothing, then a small state
   machine that turns each risky episode into ONE sharp alarm (score >= 0.5
   only on a rising edge, capped at ``max_alarm_sec``, then a refractory
   period). Outside alarms the score is squashed below 0.5 but keeps its
   ordering, so frame-level AP still sees the graded risk.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import BICYCLE, PERSON, VEHICLES
from .ttc import footprint, pairwise_closing, pairwise_ttc

RISK_CLASSES = sorted(VEHICLES | {PERSON, BICYCLE})
FEATURES = ("ttc_margin", "decel_excess", "closing")

DEFAULTS: dict = {
    "stride": 3,             # run detection every N-th frame (by time, at the video fps)
    "imgsz": 640,
    "conf": 0.25,
    "foot_frac": 0.4,        # footprint = bottom fraction of the box (see src.ttc.footprint)
    "hist_sec": 1.5,         # per-track history kept
    "vel_sec": 0.6,          # least-squares velocity window
    "min_vel_samples": 3,
    "decel_sec": 1.0,        # deceleration = speed drop over this window
    "moving_speed": 0.5,     # heights/s; decel only counts for objects that were moving
    "min_closing": 0.3,      # heights/s; pairs closing slower than this are ignored
    "ttc0": 1.5,             # s, TTC at which the TTC term is neutral
    "ttc_cap": 5.0,          # s, TTC beyond this (or inf) counts as this
    "decel0": 1.0,           # heights/s^2 of "normal" braking
    "closing_cap": 5.0,
    "weights": [3.0, 0.5, 0.3],  # logistic weights for FEATURES (fit these on dev videos)
    "bias": 0.0,
    "ema_sec": 0.3,          # smoothing time constant of the raw score
    "alarm_on": 0.6,         # smoothed score that starts an alarm (rising edge only)
    "alarm_off": 0.35,       # ... falls below this to end it and re-arm
    "max_alarm_sec": 3.0,    # cap on the >= 0.5 plateau
    "refractory_sec": 6.0,   # no new alarm this long after one ends (one alarm per episode)
}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))


def risk_features(centre: np.ndarray, half: np.ndarray, vel: np.ndarray, height: np.ndarray,
                  decel: np.ndarray, is_vehicle: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Snapshot of N tracked objects -> feature vector ordered as ``FEATURES``.

    centre/half: (N, 2) footprint boxes (px); vel: (N, 2) px/s; height: (N,) px;
    decel: (N,) heights/s^2; is_vehicle: (N,) bool (pedestrian-only pairs are skipped).
    """
    c = {**DEFAULTS, **(cfg or {})}
    min_ttc, closing = math.inf, 0.0
    n = len(centre)
    if n >= 2:
        ttc = pairwise_ttc(centre, half, vel)
        pair_h = (height[:, None] + height[None]) / 2
        close = pairwise_closing(centre, vel) / pair_h
        valid = (close >= c["min_closing"]) & (is_vehicle[:, None] | is_vehicle[None])
        ttc = np.where(valid, ttc, np.inf)
        k = int(np.argmin(ttc))
        min_ttc = float(ttc.flat[k])
        closing = float(close.flat[k]) if np.isfinite(min_ttc) else 0.0
    max_decel = float(np.max(decel)) if n else 0.0
    return np.array([
        c["ttc0"] - min(min_ttc, c["ttc_cap"]),
        max(0.0, max_decel - c["decel0"]),
        min(closing, c["closing_cap"]),
    ])


def score_features(centre: np.ndarray, half: np.ndarray, vel: np.ndarray, height: np.ndarray,
                   decel: np.ndarray, is_vehicle: np.ndarray,
                   cfg: dict | None = None) -> tuple[float, np.ndarray]:
    """Instantaneous risk in [0, 1] = sigmoid(weights . features + bias); also returns the features."""
    c = {**DEFAULTS, **(cfg or {})}
    f = risk_features(centre, half, vel, height, decel, is_vehicle, c)
    return _sigmoid(float(np.dot(c["weights"], f)) + float(c["bias"])), f


class _TrackHist:
    """Short causal history of one track: foot positions, box sizes and speeds."""

    def __init__(self, is_vehicle: bool):
        self.is_vehicle = is_vehicle
        self.obs: deque[tuple[float, float, float, float, float]] = deque()  # t, foot x, foot y, w, h
        self.speeds: deque[tuple[float, float]] = deque()                    # t, heights/s

    def add(self, t: float, box: np.ndarray, hist_sec: float) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        self.obs.append((t, (x1 + x2) / 2, y2, x2 - x1, y2 - y1))
        while self.obs and self.obs[0][0] < t - hist_sec:
            self.obs.popleft()

    def kinematics(self, t: float, c: dict) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """(foot xy now, velocity px/s, height px, width px), least squares over the last ``vel_sec``."""
        a = np.array([o for o in self.obs if o[0] >= t - c["vel_sec"] - 1e-6])
        if len(a) < c["min_vel_samples"]:
            return None
        tt = a[:, 0] - a[:, 0].mean()
        denom = float((tt ** 2).sum())
        if denom <= 0:
            return None
        vel = (tt[:, None] * a[:, 1:3]).sum(axis=0) / denom
        foot = a[:, 1:3].mean(axis=0) + vel * (t - a[:, 0].mean())
        return foot, vel, float(np.median(a[:, 4])), float(np.median(a[:, 3]))

    def decel(self, t: float, speed: float, c: dict) -> float:
        """Speed drop over the last ``decel_sec`` / ``decel_sec`` (heights/s^2), 0 if it wasn't moving."""
        self.speeds.append((t, speed))
        while self.speeds and self.speeds[0][0] < t - c["decel_sec"]:
            self.speeds.popleft()
        peak = max(s for _, s in self.speeds)
        return (peak - speed) / c["decel_sec"] if peak >= c["moving_speed"] else 0.0


class CausalRisk:
    def __init__(self, params: dict):
        self.params = params
        self.cfg = {**DEFAULTS, **params.get("risk", {})}
        self._detector = None
        self._tracker = None
        self.reset({})

    # ------------------------------------------------------------------ lifecycle
    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.fps = float(meta.get("fps") or 25.0)
        self._interval = max(1, int(self.cfg["stride"])) / self.fps
        self._next_t = -math.inf
        self._hist: dict[int, _TrackHist] = {}
        self._last_seen: dict[int, float] = {}
        self._ema: float | None = None
        self._ema_t = 0.0
        self._alarm_start: float | None = None
        self._quiet_until = -math.inf
        self._armed = True
        self.last = 0.0
        self._tracker = None
        if meta:
            try:
                self._detector = self._detector or self._make_detector(self.params)
                self._tracker = self._make_tracker(self.fps / max(1, int(self.cfg["stride"])))
            except Exception:
                self._detector, self._tracker = None, None  # step() then returns 0

    @staticmethod
    def _make_detector(params: dict):
        from .perception import Detector

        p, c = params["perception"], {**DEFAULTS, **params.get("risk", {})}
        return Detector(p["detector"], int(c["imgsz"]), float(c["conf"]), p["iou"], p["half"],
                        classes=RISK_CLASSES)

    def _make_tracker(self, frame_rate: float):
        from .perception import Tracker

        p = self.params["perception"]
        return Tracker(frame_rate, p["lost_track_sec"], p["track_activation_threshold"],
                       p["minimum_matching_threshold"])

    @staticmethod
    def warmup(params: dict) -> None:
        """Load and warm up the risk detector (raises if its weights are missing)."""
        CausalRisk._make_detector(params).warmup()

    # ------------------------------------------------------------------ per frame
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if t_sec < self._next_t - 1e-6:
            return self.last
        self._next_t = t_sec + self._interval
        try:
            if self._detector is None or self._tracker is None:
                return self.last
            det = self._detector([frame])[0]
            ids, xyxy, _, cls = self._tracker.update(det)
            return self.update(t_sec, ids, xyxy, cls)
        except Exception:
            return self.last

    def update(self, t: float, ids: np.ndarray, xyxy: np.ndarray, cls: np.ndarray) -> float:
        """Feed the tracked boxes of one detection step; returns the shaped risk score."""
        c = self.cfg
        rows = []
        for tid, box, k in zip(ids, xyxy, cls):
            tid = int(tid)
            h = self._hist.setdefault(tid, _TrackHist(int(k) in VEHICLES))
            h.add(t, box, c["hist_sec"])
            self._last_seen[tid] = t
            kin = h.kinematics(t, c)
            if kin is None:
                continue
            foot, vel, height, width = kin
            speed = float(np.linalg.norm(vel)) / max(height, 1.0)
            rows.append((foot, vel, height, width, h.decel(t, speed, c), h.is_vehicle))
        for tid in [i for i, ts in self._last_seen.items() if ts < t - c["hist_sec"]]:
            del self._hist[tid], self._last_seen[tid]

        if rows:
            foot = np.array([r[0] for r in rows])
            height = np.array([r[2] for r in rows])
            width = np.array([r[3] for r in rows])
            boxes = np.column_stack([foot[:, 0] - width / 2, foot[:, 1] - height,
                                     foot[:, 0] + width / 2, foot[:, 1]])
            centre, half = footprint(boxes, c["foot_frac"])
            raw, _ = score_features(centre, half, np.array([r[1] for r in rows]), height,
                                    np.array([r[4] for r in rows]), np.array([r[5] for r in rows]), c)
        else:
            raw = 0.0
        self.last = self._shape(t, raw)
        return self.last

    def _shape(self, t: float, raw: float) -> float:
        """EMA + one-alarm-per-episode state machine; >= 0.5 only while an alarm is active."""
        c = self.cfg
        if self._ema is None:
            self._ema = raw
        else:
            alpha = 1.0 - math.exp(-max(t - self._ema_t, 0.0) / max(c["ema_sec"], 1e-6))
            self._ema += alpha * (raw - self._ema)
        self._ema_t = t
        p = self._ema
        if self._alarm_start is not None and (p < c["alarm_off"] or t - self._alarm_start >= c["max_alarm_sec"]):
            self._alarm_start = None
            self._quiet_until = t + c["refractory_sec"]
        if p < c["alarm_off"]:
            self._armed = True
        if self._alarm_start is None and self._armed and p >= c["alarm_on"] and t >= self._quiet_until:
            self._alarm_start, self._armed = t, False
        return 0.5 + 0.5 * p if self._alarm_start is not None else 0.49 * p
