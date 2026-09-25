import numpy as np

from evaluate import alarm_starts
from src.config import CAR, PERSON, load_params
from src.risk import CausalRisk, score_features
from src.ttc import footprint

SIZE = (120.0, 80.0)
FPS, STRIDE = 25.0, 3


def _snapshot(feet, vels, size=SIZE):
    feet, vels = np.asarray(feet, float), np.asarray(vels, float)
    w, h = size
    boxes = np.column_stack([feet[:, 0] - w / 2, feet[:, 1] - h, feet[:, 0] + w / 2, feet[:, 1]])
    centre, half = footprint(boxes, 0.4)
    n = len(feet)
    return centre, half, vels, np.full(n, h), np.zeros(n), np.ones(n, bool)


def test_score_features_ttc_and_parallel():
    # head-on, 400 px/s closing, 480 px apart -> TTC ~0.9 s
    p_close, f = score_features(*_snapshot([[700, 600], [1300, 600]], [[200, 0], [-200, 0]]))
    p_far, _ = score_features(*_snapshot([[100, 600], [1800, 600]], [[200, 0], [-200, 0]]))
    p_par, f_par = score_features(*_snapshot([[700, 500], [700, 600]], [[200, 0], [200, 0]]))
    assert p_close > 0.8 > 0.05 > p_far and p_par < 0.01
    assert f[0] > 0 and f_par[2] == 0.0
    # two pedestrians walking into each other are not a vehicle accident
    c, h, v, hh, d, _ = _snapshot([[700, 600], [800, 600]], [[50, 0], [-50, 0]])
    assert score_features(c, h, v, hh, d, np.zeros(2, bool))[0] < 0.01


def _run(objects, duration):
    """Feed synthetic tracked boxes through CausalRisk.update at the detection stride."""
    engine = CausalRisk(load_params())
    engine.reset({})
    curve = []
    for t in np.arange(0, duration, STRIDE / FPS):
        ids, boxes, cls = [], [], []
        for tid, k, wp in objects:
            wp = np.asarray(wp, float)
            if not wp[0, 0] <= t <= wp[-1, 0]:
                continue
            x, y = np.interp(t, wp[:, 0], wp[:, 1]), np.interp(t, wp[:, 0], wp[:, 2])
            ids.append(tid)
            boxes.append([x - SIZE[0] / 2, y - SIZE[1], x + SIZE[0] / 2, y])
            cls.append(k)
        curve.append([float(t), engine.update(float(t), np.array(ids), np.array(boxes).reshape(-1, 4),
                                              np.array(cls))])
    return np.array(curve)


def test_risk_rises_before_collision_single_alarm():
    # head-on at 200 px/s each; footprints touch at t=4 (edges meet), then both stop
    crash = [(1, CAR, [(0, 100, 600), (4, 900, 600), (12, 900, 600)]),
             (2, CAR, [(0, 1820, 600), (4, 1020, 600), (12, 1020, 600)])]
    curve = _run(crash, 12)
    t, s = curve[:, 0], curve[:, 1]
    assert s[t < 1.5].max() < 0.2
    starts = alarm_starts(curve.tolist())
    assert len(starts) == 1 and 2.0 <= starts[0] < 4.0
    assert s[(t > 6)].max() < 0.5  # wrecks at rest do not re-alarm


def test_risk_low_for_parallel_traffic():
    lanes = [(i, CAR, [(0.5 * i, 100 + 30 * i, y), (0.5 * i + 8, 1500 + 30 * i, y)])
             for i, y in enumerate([500, 600, 700, 500, 600, 700])]
    lanes.append((9, PERSON, [(0, 1700, 300), (10, 1700, 300)]))
    curve = _run(lanes, 12)
    assert curve[:, 1].max() < 0.1 and alarm_starts(curve.tolist()) == []


def test_step_skips_frames_and_never_raises():
    engine = CausalRisk(load_params())
    engine.reset({})  # no detector: step must fall back to the last score
    frame = np.zeros((4, 4, 3), np.uint8)
    assert [engine.step(frame, i / FPS) for i in range(6)] == [0.0] * 6
