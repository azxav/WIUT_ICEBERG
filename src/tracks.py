"""Tracks from raw tracker rows: grouping, ID-switch stitching, kinematics.

Speeds are expressed in *box heights per second* rather than pixels per
second. Without camera calibration this is a cheap perspective correction:
a car near the camera and a car far away moving at the same real speed get
similar values.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import CATEGORY_OF, TWO_WHEELERS
from .observe import T_CLS, T_CONF, T_ID, T_TIME, T_X1, T_X2, T_Y1, T_Y2, Observations


@dataclass
class Track:
    id: int
    t: np.ndarray           # (N,) seconds, strictly increasing
    box: np.ndarray         # (N, 4) xyxy pixels
    conf: np.ndarray        # (N,)
    cls: np.ndarray         # (N,) COCO ids
    category: str = "vehicle"
    is_rider: bool = False  # person riding a bicycle / motorcycle
    # kinematics (filled by compute_kinematics)
    foot: np.ndarray = field(default=None, repr=False)     # (N, 2) smoothed bottom-centre
    height: np.ndarray = field(default=None, repr=False)   # (N,) smoothed box height, px
    vel: np.ndarray = field(default=None, repr=False)      # (N, 2) px/s
    speed: np.ndarray = field(default=None, repr=False)    # (N,) box heights / s
    heading: np.ndarray = field(default=None, repr=False)  # (N,) radians, NaN when slow

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def __len__(self) -> int:
        return len(self.t)

    def index_at(self, t: float) -> int:
        """Index of the observation closest to time ``t``."""
        i = int(np.searchsorted(self.t, t))
        if i <= 0:
            return 0
        if i >= len(self.t):
            return len(self.t) - 1
        return i if self.t[i] - t < t - self.t[i - 1] else i - 1

    def raw_foot(self) -> np.ndarray:
        return np.column_stack([(self.box[:, 0] + self.box[:, 2]) / 2, self.box[:, 3]])


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average along axis 0 with edge padding."""
    if n <= 1 or len(x) < 3:
        return x.astype(float)
    n = min(n, len(x) if len(x) % 2 else len(x) - 1)
    pad = n // 2
    k = np.ones(n) / n
    xp = np.pad(x.astype(float), [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(xp, k, mode="valid")
    return np.column_stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])])


def compute_kinematics(tr: Track, smooth_sec: float, moving_speed: float) -> Track:
    dt = (tr.end - tr.start) / max(1, len(tr) - 1) if len(tr) > 1 else 1.0
    n = max(1, int(round(smooth_sec / max(dt, 1e-6))))
    if n % 2 == 0:
        n += 1
    tr.foot = _smooth(tr.raw_foot(), n)
    tr.height = np.maximum(_smooth(tr.box[:, 3] - tr.box[:, 1], n), 1.0)
    if len(tr) >= 2:
        tr.vel = np.gradient(tr.foot, tr.t, axis=0)
    else:
        tr.vel = np.zeros((len(tr), 2))
    tr.speed = np.linalg.norm(tr.vel, axis=1) / tr.height
    tr.heading = np.where(tr.speed > moving_speed, np.arctan2(tr.vel[:, 1], tr.vel[:, 0]), np.nan)
    return tr


def _category(cls: np.ndarray) -> tuple[str, int]:
    cats = Counter(CATEGORY_OF.get(int(c), "other") for c in cls)
    cat = cats.most_common(1)[0][0]
    top = Counter(int(c) for c in cls if CATEGORY_OF.get(int(c), "other") == cat).most_common(1)[0][0]
    return cat, top


def _stitch(tracks: list[Track], gap: float, dist_heights: float) -> list[Track]:
    """Join a track that ends with one of the same category that starts shortly after nearby."""
    tracks = sorted(tracks, key=lambda t: t.start)
    starts = np.array([t.start for t in tracks])
    removed: set[int] = set()
    for a in tracks:
        if id(a) in removed:
            continue
        while True:
            feet = a.raw_foot()
            k = min(len(a) - 1, 5)
            va = (feet[-1] - feet[-1 - k]) / max(a.t[-1] - a.t[-1 - k], 1e-6) if k else np.zeros(2)
            ha = max(float(a.box[-1, 3] - a.box[-1, 1]), 1.0)
            lo, hi = np.searchsorted(starts, [a.end, a.end + gap], side="right")
            best, best_d = None, np.inf
            for b in tracks[lo:hi]:
                if b is a or id(b) in removed or b.category != a.category or b.start <= a.end:
                    continue
                predicted = feet[-1] + va * (b.start - a.end)  # constant-velocity extrapolation
                d = float(np.linalg.norm(b.raw_foot()[0] - predicted))
                if d <= dist_heights * ha and d < best_d:
                    best, best_d = b, d
            if best is None:
                break
            a.t = np.concatenate([a.t, best.t])
            a.box = np.concatenate([a.box, best.box])
            a.conf = np.concatenate([a.conf, best.conf])
            a.cls = np.concatenate([a.cls, best.cls])
            removed.add(id(best))
    return [t for t in tracks if id(t) not in removed]


def _mark_riders(tracks: list[Track], min_share: float = 0.5) -> None:
    """A person whose box sits mostly inside a two-wheeler box is its rider."""
    wheels = [t for t in tracks if int(Counter(t.cls.tolist()).most_common(1)[0][0]) in TWO_WHEELERS]
    for p in (t for t in tracks if t.category == "person"):
        hits = 0
        for i, tp in enumerate(p.t):
            pb = p.box[i]
            for w in wheels:
                if not (w.start <= tp <= w.end):
                    continue
                wb = w.box[w.index_at(tp)]
                ix = max(0.0, min(pb[2], wb[2]) - max(pb[0], wb[0]))
                iy = max(0.0, min(pb[3], wb[3] + 0.5 * (wb[3] - wb[1])) - max(pb[1], wb[1] - (wb[3] - wb[1])))
                if ix * iy >= 0.3 * (pb[2] - pb[0]) * (pb[3] - pb[1]):
                    hits += 1
                    break
        p.is_rider = hits >= min_share * len(p)


def build_tracks(obs: Observations, params: dict) -> list[Track]:
    tp = params["tracks"]
    rows = obs.tracks
    tracks: list[Track] = []
    if len(rows):
        rows = rows[np.lexsort((rows[:, T_TIME], rows[:, T_ID]))]
        ids, starts = np.unique(rows[:, T_ID], return_index=True)
        for tid, block in zip(ids, np.split(rows, starts[1:])):
            _, keep = np.unique(block[:, T_TIME], return_index=True)
            block = block[keep]
            cls = block[:, T_CLS].astype(int)
            cat, _ = _category(cls)
            tracks.append(Track(int(tid), block[:, T_TIME].copy(), block[:, T_X1:T_Y2 + 1].copy(),
                                block[:, T_CONF].copy(), cls, cat))
    tracks = _stitch(tracks, tp["stitch_gap_sec"], tp["stitch_dist_heights"])
    tracks = [t for t in tracks if len(t) >= tp["min_obs"] and t.category != "traffic_light"]
    for t in tracks:
        compute_kinematics(t, tp["smooth_sec"], tp["moving_speed"])
    _mark_riders(tracks)
    return tracks
