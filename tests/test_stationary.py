import numpy as np

from src.config import CAR, PERSON, load_params
from src.pipeline import build_context
from src.rules import RULES
from src.rules.fire import fire_sample_times
from src.scene import Scene
from tests.synth import H, W, make_obs, obj

ROAD = [[[0, 0.4], [1, 0.4], [1, 0.8], [0, 0.8]]]            # y in [432, 864] px
LANE = {"id": "e1", "polygon": ROAD[0], "direction": [1, 0], "approach": "east"}
STOP_LINE = {"id": "sl", "line": [[0.3646, 0.4], [0.3646, 0.8]], "direction": [1, 0], "light": "tl"}  # x=700
DOG = 16


def ctx_for(objects, duration, scene: dict | None = None, params: dict | None = None, **obs_kw):
    obs = make_obs(objects, duration)
    for k, v in obs_kw.items():
        setattr(obs, k, v)
    return build_context(obs, params or load_params(), Scene.from_dict(scene or {}, W, H))


def set_light(ctx, schedule: list[tuple[float, str]]):
    """Force the per-sample state of light 'tl' (the light classifier is not under test)."""
    states = np.array([[s for t0, s in schedule if t0 <= t][-1] for t in ctx.times], dtype=object)
    ctx.__dict__["light_states"] = {"tl": states}


def run(label, ctx):
    return RULES[label](ctx)


def stopping_car(tid=1, stop_from=5.0, stop_to=25.0, x=600.0, y=700.0):
    return obj(tid, CAR, [(0, x - 500, y), (stop_from, x, y), (stop_to, x, y), (stop_to + 5, x + 1000, y)])


# ----------------------------------------------------------- stopped_vehicle
def test_stopped_vehicle_long_stop():
    ev = run("stopped_vehicle", ctx_for([stopping_car()], 32, {"carriageway": ROAD}))
    assert len(ev) == 1 and ev[0].track_ids == (1,)
    assert abs(ev[0].start - 5.0) < 0.6 and abs(ev[0].end - 25.0) < 0.6


def test_stopped_vehicle_short_stop_and_parked_ignored():
    short = stopping_car(stop_to=12.0)
    parked = obj(2, CAR, [(0, 1500, 800), (30, 1500, 800)])  # never seen arriving
    assert run("stopped_vehicle", ctx_for([short, parked], 32, {"carriageway": ROAD})) == []


def test_stopped_vehicle_queue_at_red_is_not_stopped():
    scene = {"carriageway": ROAD, "stop_lines": [STOP_LINE], "lights": [{"id": "tl", "rois": {"head": [0, 0, 0.01, 0.01]}}]}
    ctx = ctx_for([stopping_car()], 32, scene)
    set_light(ctx, [(0, "red"), (24, "green")])
    assert run("stopped_vehicle", ctx) == []
    # same stop behind the line while the light shows green -> a real stopped vehicle
    set_light(ctx, [(0, "green")])
    assert len(run("stopped_vehicle", ctx)) == 1


def test_stopped_vehicle_joins_broken_track():
    a = obj(1, CAR, [(0, 100, 700), (5, 600, 700), (12, 600, 700)])
    b = obj(2, CAR, [(16, 600, 700), (25, 600, 700), (30, 1600, 700)])  # re-acquired after 4 s
    ev = run("stopped_vehicle", ctx_for([a, b], 32, {"carriageway": ROAD}))
    assert len(ev) == 1 and ev[0].track_ids == (1, 2)
    assert abs(ev[0].start - 5.0) < 0.6 and abs(ev[0].end - 25.0) < 0.6


# ---------------------------------------------------------------- congestion
def crawling_queue(t0=10.0, t1=40.0, speed=8.0, n=5):
    return [obj(100 + i, CAR, [(t0, 200 + 150 * i, 700), (t1, 200 + 150 * i + speed * (t1 - t0), 700)])
            for i in range(n)]


def free_flow(n=8):
    return [obj(i + 1, CAR, [(i * 0.8, 100, 700), (i * 0.8 + 6, 1800, 700)]) for i in range(n)]


def test_congestion_with_lanes():
    ev = run("congestion", ctx_for(crawling_queue(), 45, {"carriageway": ROAD, "lanes": [LANE]}))
    assert len(ev) == 1 and abs(ev[0].start - 10) < 1.0 and abs(ev[0].end - 40) < 1.0


def test_congestion_with_flow_directions():
    ev = run("congestion", ctx_for(free_flow() + crawling_queue(), 45))
    assert len(ev) == 1 and abs(ev[0].start - 10) < 1.0 and abs(ev[0].end - 40) < 1.0


def test_congestion_negatives():
    scene = {"carriageway": ROAD, "lanes": [LANE]}
    # moving traffic, however dense, is not congestion
    assert run("congestion", ctx_for(crawling_queue(speed=60, t1=25), 45, scene)) == []
    # three crawling vehicles are too few
    assert run("congestion", ctx_for(crawling_queue(n=3), 45, scene)) == []
    # a queue that forms at red and clears on green is a signal queue
    scene = {**scene, "stop_lines": [STOP_LINE], "lights": [{"id": "tl", "rois": {"head": [0, 0, 0.01, 0.01]}}]}
    ctx = ctx_for(crawling_queue(), 45, scene)
    set_light(ctx, [(0, "green"), (9, "red"), (39, "green")])
    assert run("congestion", ctx) == []


# ------------------------------------------------------------- road_obstacle
def test_road_obstacle_animal_on_road_only():
    dog = obj(1, DOG, [(5, 900, 700), (15, 960, 700)], size=(60, 40))
    off_road = obj(2, DOG, [(5, 900, 300), (15, 960, 300)], size=(60, 40))
    ev = run("road_obstacle", ctx_for([dog, off_road], 20, {"carriageway": ROAD}))
    assert len(ev) == 1 and ev[0].track_ids == (1,)
    assert abs(ev[0].start - 5) < 0.2 and abs(ev[0].end - 15) < 0.2


def test_road_obstacle_static_blob_detector():
    tt = np.arange(0, 60.0)
    rng = np.random.default_rng(0)
    th = np.clip(100 + rng.normal(0, 3, (60, 180, 320)), 0, 255).astype(np.uint8)
    th[20:35, 120:130, 150:160] = 220  # object on the road (y ~ 0.7 H) for 15 s
    params = load_params({"classes": {"road_obstacle": {"static_blobs": True}}})
    walker = obj(1, PERSON, [(0, 100, 200), (59, 1800, 200)], size=(40, 100))
    ctx = ctx_for([walker], 60, {"carriageway": ROAD}, params, thumb_times=tt, thumbs=th)
    ev = run("road_obstacle", ctx)
    assert len(ev) == 1 and abs(ev[0].start - 20) <= 1 and abs(ev[0].end - 34) <= 1
    assert run("road_obstacle", ctx_for([walker], 60, {"carriageway": ROAD}, thumb_times=tt, thumbs=th)) == []


# ----------------------------------------------------------------- fire_smoke
FIRE_T = fire_sample_times(make_obs([], 30).sample_times, 1.0)  # when the fire detector ran


def fire_rows(t0, t1, conf=0.8, every=1):
    ts = FIRE_T[(FIRE_T >= t0) & (FIRE_T <= t1)][::every]
    return np.array([[t, conf, 0, 100, 100, 200, 200] for t in ts], float).reshape(-1, 7)


def test_fire_smoke_persistent_detections():
    ev = run("fire_smoke", ctx_for([], 30, fire=fire_rows(10, 20)))
    first, last = FIRE_T[FIRE_T >= 10][0], FIRE_T[FIRE_T <= 20][-1]
    assert len(ev) == 1 and abs(ev[0].start - first) < 1e-6 and abs(ev[0].end - (last + 1.0)) < 1e-6
    # a missed sample every other time still counts (3 of 5)
    assert len(run("fire_smoke", ctx_for([], 30, fire=fire_rows(10, 20, every=2)))) == 1


def test_fire_smoke_negatives():
    assert run("fire_smoke", ctx_for([], 30)) == []
    sporadic = np.concatenate([fire_rows(5, 5.5), fire_rows(12, 12.5), fire_rows(25, 25.5)])
    assert run("fire_smoke", ctx_for([], 30, fire=sporadic)) == []
    assert run("fire_smoke", ctx_for([], 30, fire=fire_rows(10, 20, conf=0.3))) == []  # weak
