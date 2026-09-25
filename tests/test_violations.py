import numpy as np

from src.config import CAR, load_params
from src.pipeline import build_context
from src.rules import RULES
from src.scene import Scene
from tests.synth import light_features_for, make_obs, obj

W, H = 1920, 1080
TOL = 0.3

# Stop line at x=960 for eastbound traffic, junction box beyond it (x 998..1536).
SIGNAL = {
    "stop_lines": [{"id": "sl", "line": [[0.5, 0.45], [0.5, 0.65]], "direction": [1, 0], "light": "tl"}],
    "intersection": [[[0.52, 0.3], [0.8, 0.3], [0.8, 0.8], [0.52, 0.8]]],
    "lights": [{"id": "tl", "rois": {"head": [0.1, 0.1, 0.11, 0.12]}}],
}
# Eastbound lane y 540..648, westbound lane y 648..756.
LANES = {"lanes": [{"id": "e", "polygon": [[0, 0.5], [1, 0.5], [1, 0.6], [0, 0.6]], "direction": [1, 0]},
                   {"id": "w", "polygon": [[0, 0.6], [1, 0.6], [1, 0.7], [0, 0.7]], "direction": [-1, 0]}]}
# West arm split into inbound A / outbound D, east arm B, south arm C.
ZONES = {"zones": [{"id": "A", "polygon": [[0, 0], [0.3, 0], [0.3, 0.6], [0, 0.6]]},
                   {"id": "D", "polygon": [[0, 0.6], [0.3, 0.6], [0.3, 1], [0, 1]]},
                   {"id": "B", "polygon": [[0.7, 0], [1, 0], [1, 1], [0.7, 1]]},
                   {"id": "C", "polygon": [[0.4, 0.8], [0.6, 0.8], [0.6, 1], [0.4, 1]]}],
         "allowed_movements": [["A", "B"]]}
SOLID = {"solid_lines": [{"id": "s1", "polyline": [[0.2, 0.6], [0.5, 0.6], [0.8, 0.6]]}]}


def _run(label, objects, duration, scene=None, schedule=None):
    times = np.arange(0, int(duration * 25), 2) / 25
    feats = {"tl:head": light_features_for(times, schedule)} if schedule else {}
    obs = make_obs(objects, duration, light_features=feats)
    ctx = build_context(obs, load_params(), Scene.from_dict(scene or {}, W, H))
    return sorted(RULES[label](ctx), key=lambda e: e.start)


def _close(ev, start, end, tol=TOL):
    assert abs(ev.start - start) <= tol and abs(ev.end - end) <= tol, (ev.start, ev.end, start, end)


def _arc(cx, cy, r, phi0, phi1, t0, t1, n=24):
    """Waypoints on a circle; heading follows phi when moving with increasing phi."""
    return [(t0 + (t1 - t0) * k / n, cx + r * np.sin(p), cy - r * np.cos(p))
            for k, p in enumerate(np.linspace(phi0, phi1, n + 1))]


# ---------------------------------------------------------------- red_light
EAST = [(0, 400, 600), (7, 1800, 600)]  # 200 px/s: crosses x=960 at 2.8 s, leaves junction at 5.68 s


def test_red_light_positive():
    ev = _run("red_light", [obj(1, CAR, EAST)], 9, SIGNAL, [(0, "red")])
    assert len(ev) == 1 and ev[0].track_ids == (1,)
    _close(ev[0], 2.8, 5.68, 0.15)
    assert ev[0].score == 1.0


def test_red_light_negatives():
    assert _run("red_light", [obj(1, CAR, EAST)], 9, SIGNAL, [(0, "green")]) == []
    # turns red 0.2 s after the crossing
    assert _run("red_light", [obj(1, CAR, EAST)], 9, SIGNAL, [(0, "green"), (2, "yellow"), (3, "red")]) == []
    # crossing against the stop line's direction
    assert _run("red_light", [obj(1, CAR, [(0, 1800, 600), (7, 400, 600)])], 9, SIGNAL, [(0, "red")]) == []
    # no scene / no light features
    assert _run("red_light", [obj(1, CAR, EAST)], 9, None, [(0, "red")]) == []
    assert _run("red_light", [obj(1, CAR, EAST)], 9, SIGNAL, None) == []


# ---------------------------------------------------------------- stop_line
def test_stop_line_positive_and_not_red_light():
    car = obj(1, CAR, [(0, 400, 600), (3, 1000, 600), (12, 1000, 600)])  # stops 0.5 box past the line
    sched = [(0, "red"), (8, "green")]
    ev = _run("stop_line", [car], 12, SIGNAL, sched)
    assert len(ev) == 1
    _close(ev[0], 3.0, 8.0, 0.4)  # start lags by ~half the 0.6 s kinematics smoothing window
    assert _run("red_light", [car], 12, SIGNAL, sched) == []


def test_stop_line_negatives():
    sched = [(0, "red"), (8, "green")]
    before = obj(1, CAR, [(0, 400, 600), (3, 940, 600), (12, 940, 600)])
    too_far = obj(2, CAR, [(0, 400, 600), (4, 1200, 600), (12, 1200, 600)])
    assert _run("stop_line", [before, too_far], 12, SIGNAL, sched) == []
    past = obj(3, CAR, [(0, 400, 600), (3, 1000, 600), (12, 1000, 600)])
    assert _run("stop_line", [past], 12, SIGNAL, [(0, "green")]) == []
    assert _run("stop_line", [past], 12, None, sched) == []


# ---------------------------------------------------------------- wrong_way
def test_wrong_way_overtake_into_opposing_lane():
    car = obj(1, CAR, [(0, 200, 600), (3, 800, 600), (4, 1000, 700), (7, 1600, 700), (8, 1800, 600)])
    ev = _run("wrong_way", [car], 9, LANES)
    assert len(ev) == 1
    _close(ev[0], 3.48, 7.52, 0.15)


def test_wrong_way_lane_negatives_and_full_reverse():
    good = obj(1, CAR, [(0, 200, 600), (8, 1800, 600)])
    assert _run("wrong_way", [good], 9, LANES) == []
    bad = obj(2, CAR, [(1, 1800, 600), (9, 200, 600)])
    ev = _run("wrong_way", [bad], 10, LANES)
    assert len(ev) == 1
    _close(ev[0], 1.0, 9.0, 0.15)


def test_wrong_way_flow_fallback_without_lanes():
    objs = [obj(i, CAR, [(i * 0.5, 100, 600), (i * 0.5 + 8, 1800, 600)]) for i in range(1, 12)]
    objs.append(obj(99, CAR, [(2, 1800, 600), (10, 100, 600)]))
    ev = _run("wrong_way", objs, 15)
    assert [e.track_ids for e in ev] == [(99,)]
    _close(ev[0], 2.0, 10.0, 0.5)


# ------------------------------------------------------- illegal_u_turn / turn
U_TURN = [(0, 400, 600), (4, 1200, 600)] + _arc(1200, 700, 100, 0, np.pi, 4, 7)[1:] + [(11, 400, 800)]


def test_u_turn_positive():
    ev = _run("illegal_u_turn", [obj(1, CAR, U_TURN)], 12)
    assert len(ev) == 1
    _close(ev[0], 4.0, 7.0)
    assert ev[0].info["turned_deg"] > 170


def test_u_turn_negatives():
    right = [(0, 200, 500), (5, 900, 500)] + _arc(900, 600, 100, 0, np.pi / 2, 5, 6.5)[1:] + [(10, 1000, 1050)]
    straight = [(0, 200, 400), (8, 1800, 400)]
    assert _run("illegal_u_turn", [obj(1, CAR, right), obj(2, CAR, straight)], 11) == []
    assert _run("illegal_u_turn", [obj(1, CAR, U_TURN)], 12, {"u_turn_allowed": True}) == []
    # a car jittering in place (heading swings around, but it goes nowhere)
    rng = np.random.default_rng(0)
    jitter = [(t, 900 + 60 * np.cos(t * 3) + rng.normal(0, 5), 600 + 40 * np.sin(t * 3)) for t in np.arange(0, 10, 0.1)]
    assert _run("illegal_u_turn", [obj(1, CAR, jitter)], 11) == []


RIGHT_TURN = [(0, 200, 500), (5, 900, 500)] + _arc(900, 600, 100, 0, np.pi / 2, 5, 6.5)[1:] + [(10, 1000, 1050)]


def test_illegal_turn_positive():
    ev = _run("illegal_turn", [obj(1, CAR, RIGHT_TURN)], 11, ZONES)
    assert len(ev) == 1 and ev[0].info == {"entry": "A", "exit": "C"}
    _close(ev[0], 5.0, 6.5)


def test_illegal_turn_negatives():
    allowed = obj(1, CAR, [(0, 200, 500), (8, 1800, 500)])  # A -> B
    u_turn = obj(2, CAR, [(0, 200, 500), (4, 900, 500)] + _arc(900, 600, 100, 0, np.pi, 4, 7)[1:]
                 + [(10, 200, 700)])  # A -> D, but it is a U-turn
    assert _run("illegal_turn", [allowed, u_turn], 11, ZONES) == []
    assert len(_run("illegal_u_turn", [u_turn], 11, ZONES)) == 1
    assert _run("illegal_turn", [obj(1, CAR, RIGHT_TURN)], 11, {"zones": ZONES["zones"]}) == []
    assert _run("illegal_turn", [obj(1, CAR, RIGHT_TURN)], 11) == []


# ------------------------------------------------------- solid_line_crossing
def test_solid_line_crossing_positive():
    # lateral 100 px/s, crosses y=648 at 4.48 s; half box width 60 px -> 0.6 s lead / lag
    car = obj(1, CAR, [(0, 400, 600), (4, 1200, 600), (5, 1400, 700), (8, 1900, 700)])
    ev = _run("solid_line_crossing", [car], 9, SOLID)
    assert len(ev) == 1
    _close(ev[0], 3.88, 5.08, 0.15)


def test_solid_line_jitter_and_missing_scene():
    ts = np.arange(0, 8.01, 0.1)
    jitter = obj(1, CAR, [(t, 400 + 150 * t, 648 + 15 * np.sin(2 * np.pi * t)) for t in ts])
    drift = obj(2, CAR, [(0, 400, 640), (8, 1500, 656)])
    assert _run("solid_line_crossing", [jitter, drift], 9, SOLID) == []
    car = obj(3, CAR, [(0, 400, 600), (4, 1200, 600), (5, 1400, 700), (8, 1900, 700)])
    assert _run("solid_line_crossing", [car], 9) == []
