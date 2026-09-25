import numpy as np

from src.lights import classify_lights
from src.scene import Scene
from tests.synth import light_features_for, make_obs


def _times(duration: float, fps: float = 25.0, stride: int = 2) -> np.ndarray:
    return np.arange(0, int(duration * fps), stride) / fps


def _state_of(schedule, t):
    return [s for t0, s in schedule if t0 <= t][-1]


def _lamp_features(times, schedule, dark=(), seed=0):
    """Per-lamp ROI features: lit lamp V~0.9 with its colour share 0.8, unlit V~0.2."""
    rng = np.random.default_rng(seed)
    out = {}
    for col, lamp in enumerate(("red", "yellow", "green"), start=1):
        f = np.zeros((len(times), 4), np.float32)
        for i, t in enumerate(times):
            lit = _state_of(schedule, t) == lamp and not any(a <= t < b for a, b in dark)
            f[i, 0] = (0.9 if lit else 0.2) + rng.normal(0, 0.02)
            f[i, col] = 0.8 if lit else 0.0
        out[f"tl:{lamp}"] = f
    return out


def _scene(rois):
    return Scene.from_dict({"lights": [{"id": "tl", "rois": {k: [0.1, 0.1, 0.11, 0.12] for k in rois}}]},
                           1920, 1080)


def _agree_away_from_transitions(states, times, schedule, tol=0.3):
    edges = np.array([t0 for t0, _ in schedule[1:]])
    for s, t in zip(states, times):
        if edges.size and np.min(np.abs(edges - t)) <= tol:
            continue
        assert s == _state_of(schedule, t), (t, s)


def test_head_roi_with_flicker_is_smoothed():
    times = _times(20)
    schedule = [(0, "red"), (10, "green")]
    feats = light_features_for(times, schedule)
    flicker = int(np.searchsorted(times, 5.0))
    feats[flicker] = [0.8, 0, 0, 1]          # one green frame inside red
    feats[flicker + 30] = [0.8, 0, 0, 0]     # one dark frame
    obs = make_obs([], 20, light_features={"tl:head": feats})
    states = classify_lights(obs, _scene(["head"]), {})["tl"]
    assert len(states) == len(times)
    assert states[flicker] == "red" and states[flicker + 30] == "red"
    _agree_away_from_transitions(states, times, schedule)
    change = times[np.flatnonzero(states == "green")[0]]
    assert abs(change - 10.0) <= 0.2


def test_per_lamp_rois_cycle_and_dark_period():
    times = _times(30)
    schedule = [(0, "red"), (10, "green"), (20, "yellow"), (23, "red")]
    feats = _lamp_features(times, schedule, dark=[(25, 28)])
    obs = make_obs([], 30, light_features=feats)
    states = classify_lights(obs, _scene(["red", "yellow", "green"]), {})["tl"]
    ok = times < 24.7
    _agree_away_from_transitions(states[ok], times[ok], schedule)
    dark = (times > 25.5) & (times < 27.5)
    assert (states[dark] == "unknown").all()


def test_per_lamp_single_flicker_frame_ignored():
    times = _times(12)
    schedule = [(0, "green"), (6, "red")]
    feats = _lamp_features(times, schedule)
    i = int(np.searchsorted(times, 3.0))
    feats["tl:green"][i] = [0.2, 0, 0, 0]
    feats["tl:red"][i] = [0.9, 0.8, 0, 0]
    states = classify_lights(make_obs([], 12, light_features=feats), _scene(["red", "yellow", "green"]), {})["tl"]
    assert states[i] == "green"
    _agree_away_from_transitions(states, times, schedule)


def test_red_amber_counts_as_red():
    times = _times(6)
    feats = _lamp_features(times, [(0, "red")])
    feats["tl:yellow"][times >= 3] = [0.9, 0, 0.8, 0]
    states = classify_lights(make_obs([], 6, light_features=feats), _scene(["red", "yellow", "green"]), {})["tl"]
    assert (states == "red").all()


def test_missing_features_are_unknown():
    obs = make_obs([], 5)
    states = classify_lights(obs, _scene(["head"]), {})["tl"]
    assert len(states) == len(obs.sample_times) and (states == "unknown").all()
    assert classify_lights(obs, Scene(1920, 1080), {}) == {}
