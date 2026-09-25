from src.config import CAR, load_params
from src.pipeline import build_context
from src.rules.interactions import accident, near_miss
from src.scene import Scene
from tests.synth import make_obs, obj


def _ctx(objects, duration):
    return build_context(make_obs(objects, duration), load_params(), Scene(1920, 1080))


def test_accident_converge_touch_stop():
    # two cars drive head-on at 200 px/s (2.5 heights/s), touch at t=5 (edges meet at x=960), both stop
    a = obj(1, CAR, [(0, -100, 600), (5, 900, 600), (15, 900, 600)])
    b = obj(2, CAR, [(0, 2020, 600), (5, 1020, 600), (15, 1020, 600)])
    events = accident(_ctx([a, b], 16))
    assert len(events) == 1
    ev = events[0]
    assert abs(ev.start - 5.0) <= 0.3 and ev.end > ev.start
    assert set(ev.track_ids) == {1, 2} and ev.score > 0.5


def test_accident_guards_queue_and_frame_edge():
    # rear car brakes hard behind a standing car, then creeps the last metre into touching it: a queue
    lead = obj(1, CAR, [(0, 1000, 600), (15, 1000, 600)])
    rear = obj(2, CAR, [(0, -236, 600), (4.5, 664, 600), (5.5, 864, 600), (6, 872, 600), (7, 880, 600), (15, 880, 600)])
    assert accident(_ctx([lead, rear], 16)) == []
    # same head-on crash as the positive test, but at the right frame edge (boxes truncated there)
    a = obj(1, CAR, [(0, 900, 600), (5, 1720, 600), (15, 1720, 600)])
    b = obj(2, CAR, [(0, 2840, 600), (5, 1840, 600), (15, 1840, 600)])
    assert accident(_ctx([a, b], 16)) == []


def test_occlusion_pass_through_is_not_accident():
    # opposite directions in adjacent image rows: footprints overlap while passing, speeds stay constant
    a = obj(1, CAR, [(0, 100, 600), (10, 1800, 600)])
    b = obj(2, CAR, [(0, 1800, 630), (10, 100, 630)])
    # and a crossing pair passing through the same spot at constant speed
    c = obj(3, CAR, [(0, 100, 900), (10, 1800, 900)])
    d = obj(4, CAR, [(0, 950, 300), (10, 950, 1500)])
    assert accident(_ctx([a, b, c, d], 11)) == []


def test_parked_neighbours_are_not_accident():
    a = obj(1, CAR, [(0, 500, 600), (10, 500, 600)])
    b = obj(2, CAR, [(0, 615, 600), (10, 615, 600)])
    assert accident(_ctx([a, b], 11)) == []


def test_near_miss_hard_braking_no_contact():
    # A crosses east; B heads south towards A's path, brakes hard at t=4 and waits, then goes on
    a = obj(1, CAR, [(0, 200, 600), (10, 1700, 600)])
    b = obj(2, CAR, [(0, 950, 80), (4, 950, 480), (4.4, 950, 500), (7, 950, 500), (12, 950, 1000)])
    ctx = _ctx([a, b], 13)
    assert accident(ctx) == []
    events = near_miss(ctx)
    assert len(events) == 1
    ev = events[0]
    assert 3.4 <= ev.start <= 4.3 and ev.end > ev.start and ev.end <= 7.5
    assert set(ev.track_ids) == {1, 2}


def test_near_miss_needs_evasive_action():
    # same geometry but B keeps a steady speed and passes safely behind A: no evasive action
    a = obj(1, CAR, [(0, 200, 600), (10, 1700, 600)])
    b = obj(2, CAR, [(0, 1400, 80), (10, 1400, 1000)])
    assert near_miss(_ctx([a, b], 11)) == []
