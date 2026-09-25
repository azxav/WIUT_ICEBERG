"""Traffic-light state per sample from the ROI colour features in Observations.

Two ROI layouts are supported per light (see ``src.scene``):

* per-lamp boxes ``red`` / ``yellow`` / ``green``: the lit lamp is the one
  brightest relative to its own per-video range (so a dim green lamp and a
  glaring red one compare fairly) or, failing that, the one with the highest
  share of its own colour (covers a lamp that is lit for the whole clip);
* one ``head`` box around the signal: the dominant hue share among its
  bright pixels.

Raw per-sample states are then smoothed: a ~0.5 s majority filter removes
single-frame flicker, and runs shorter than a minimum dwell take the state of
their neighbours, so a change is only accepted once it persists. Red and
yellow lit together (the red+amber phase) counts as red.
"""
from __future__ import annotations

import numpy as np

from .observe import Observations
from .scene import Scene

STATES = np.array(["unknown", "red", "yellow", "green"], dtype=object)  # index = state code
LAMPS = ("red", "yellow", "green")  # code = LAMPS.index(name) + 1; feature column = code

DEFAULTS = {
    "lamp_level": 0.5,        # min lamp score (normalised brightness or own-colour share) to be lit
    "lamp_margin": 0.25,      # the lit lamp must beat the runner-up by this much
    "min_spread": 0.05,       # floor on a lamp's brightness range (mean V), keeps noise from amplifying
    "head_level": 0.3,        # min hue share of the winning colour in a 'head' ROI
    "head_margin": 0.15,
    "window_sec": 0.5,        # majority filter width
    "min_dwell_sec": 0.4,     # shorter coloured runs are absorbed by agreeing neighbours
    "fill_unknown_sec": 1.0,  # shorter unknown gaps too (e.g. the dark half of a flashing green)
}


def classify_lights(obs: Observations, scene: Scene, params: dict) -> dict[str, np.ndarray]:
    """light id -> array of 'red' / 'yellow' / 'green' / 'unknown', aligned to obs.sample_times."""
    cfg = {**DEFAULTS, **params.get("lights", {})}
    n = len(obs.sample_times)
    out: dict[str, np.ndarray] = {}
    for light in scene.lights:
        feats = {name: np.asarray(obs.light_features[f"{light.id}:{name}"], float)
                 for name in light.rois
                 if len(obs.light_features.get(f"{light.id}:{name}", ())) == n}
        lamps = {k: v for k, v in feats.items() if k in LAMPS}
        if lamps:
            codes = _lamp_codes(lamps, cfg)
        elif "head" in feats:
            codes = _decide(np.nan_to_num(feats["head"][:, 1:4]), [1, 2, 3],
                            cfg["head_level"], cfg["head_margin"])
        else:
            codes = np.zeros(n, int)
        out[light.id] = STATES[_smooth_codes(codes, obs.sample_times, cfg)]
    return out


def _lamp_codes(lamps: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """Per-sample state from per-lamp ROIs."""
    names = [k for k in LAMPS if k in lamps]
    scores = []
    for name in names:
        f = lamps[name]
        v = f[:, 0]
        lo, hi = np.nanpercentile(v, [5, 95]) if np.isfinite(v).any() else (0.0, 0.0)
        norm = (v - lo) / max(hi - lo, cfg["min_spread"])
        own = f[:, 1 + LAMPS.index(name)]  # share of the lamp's own colour among bright pixels
        scores.append(np.fmax(norm, own))
    s = np.nan_to_num(np.column_stack(scores), nan=0.0)
    return _decide(s, [LAMPS.index(k) + 1 for k in names], cfg["lamp_level"], cfg["lamp_margin"])


def _decide(scores: np.ndarray, codes: list[int], level: float, margin: float) -> np.ndarray:
    """Winning state code per row of ``scores`` (columns = ``codes``), 0 when ambiguous or dark."""
    n = len(scores)
    if scores.shape[1] == 0:
        return np.zeros(n, int)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(n)
    best = scores[rows, order[:, 0]]
    second = scores[rows, order[:, 1]] if scores.shape[1] > 1 else np.zeros(n)
    codes_arr = np.asarray(codes)
    out = np.where((best >= level) & (best - second >= margin), codes_arr[order[:, 0]], 0)
    if 1 in codes and 2 in codes:  # red + amber lit together: still red
        r, y = scores[:, codes.index(1)], scores[:, codes.index(2)]
        g = scores[:, codes.index(3)] if 3 in codes else np.zeros(n)
        out = np.where((r >= level) & (y >= level) & (g < level), 1, out)
    return out.astype(int)


def _smooth_codes(codes: np.ndarray, times: np.ndarray, cfg: dict) -> np.ndarray:
    """Majority filter, then absorb runs shorter than the dwell time into agreeing neighbours."""
    n = len(codes)
    if n < 3:
        return codes
    dt = float(np.median(np.diff(times)))
    w = max(1, int(round(cfg["window_sec"] / dt))) | 1  # odd width
    onehot = np.eye(len(STATES))[codes]
    pad = w // 2
    csum = np.cumsum(np.vstack([np.zeros((1, len(STATES))),
                                np.pad(onehot, ((pad, pad), (0, 0)), mode="edge")]), axis=0)
    counts = csum[w:] - csum[:-w]
    codes = np.argmax(counts + 0.5 * onehot, axis=1)  # ties keep the centre sample's state

    change = np.flatnonzero(np.diff(codes)) + 1
    starts, ends = np.r_[0, change], np.r_[change, n]  # ends exclusive
    out = codes.copy()
    for k, (s, e) in enumerate(zip(starts, ends)):
        dur = times[e - 1] - times[s] + dt
        if dur >= (cfg["fill_unknown_sec"] if codes[s] == 0 else cfg["min_dwell_sec"]):
            continue
        prev = out[s - 1] if s > 0 else None
        nxt = codes[e] if e < n else None
        if prev is not None and nxt is not None and prev != nxt:
            continue  # a genuine transition, e.g. green -> (dark) -> yellow
        fill = prev if prev is not None else nxt
        if fill is not None:
            out[s:e] = fill
    return out
