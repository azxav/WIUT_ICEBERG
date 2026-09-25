import numpy as np

from src.config import CAR, PERSON, load_params
from src.pipeline import build_context
from src.scene import Scene, points_in_polygon, polyline_crossings, segment_crossings
from src.segments import finalize, flags_to_intervals, hysteresis, merge_intervals
from src.tracks import build_tracks
from tests.synth import make_obs, obj


def test_flags_to_intervals_merges_and_drops():
    t = np.arange(10.0)
    f = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0], bool)
    assert flags_to_intervals(t, f) == [(1.0, 2.0), (4.0, 4.0), (8.0, 8.0)]
    assert flags_to_intervals(t, f, max_gap=2.0) == [(1.0, 4.0), (8.0, 8.0)]
    assert flags_to_intervals(t, f, max_gap=2.0, min_len=1.0) == [(1.0, 4.0)]


def test_hysteresis():
    v = np.array([0, 0.6, 0.4, 0.2, 0.6])
    assert hysteresis(v, 0.5, 0.3).tolist() == [False, True, True, False, True]


def test_finalize_offsets_clip_and_merge():
    cfg = {"start_offset": -1.0, "end_offset": 1.0, "merge_gap": 0.5, "min_len": 3.0}
    assert finalize([(0.5, 2.0), (3.2, 4.0), (8.0, 8.5)], 9.0, cfg) == [(0.0, 5.0)]
    assert merge_intervals([(0, 1), (0.5, 2), (5, 6)]) == [(0, 2), (5, 6)]


def test_geometry():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert points_in_polygon(sq, np.array([[5, 5], [15, 5], [-1, 3]])).tolist() == [True, False, False]
    path = np.array([[0, 5], [4, 5], [8, 5]], float)
    hits = segment_crossings(path, np.array([6.0, 0]), np.array([6.0, 10]))
    assert hits == [(1, 0.5)]
    assert len(polyline_crossings(path, np.array([[2, 0], [2, 10], [7, 10], [7, 0]], float))) == 2


def test_scene_normalised_coordinates():
    s = Scene.from_dict({"carriageway": [[[0, 0.5], [1, 0.5], [1, 1], [0, 1]]],
                         "lanes": [{"id": "e", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]],
                                    "direction": [1, 0]}]}, 1920, 1080)
    assert s.on_carriageway(np.array([[100, 900], [100, 100]])).tolist() == [True, False]
    assert np.allclose(s.lanes[0].direction, [1, 0])


def test_tracks_kinematics_and_stitching():
    params = load_params()
    # one car with an ID switch at t=5 (gap of 0.5 s, same place) -> stitched into one track
    obs = make_obs([obj(1, CAR, [(0, 100, 600), (5, 600, 600)]),
                    obj(2, CAR, [(5.5, 650, 600), (10, 1100, 600)]),
                    obj(3, PERSON, [(0, 1500, 900), (10, 1500, 900)], size=(40, 100))], duration=11)
    tracks = build_tracks(obs, params)
    cars = [t for t in tracks if t.category == "vehicle"]
    assert len(cars) == 1 and cars[0].start == 0 and cars[0].end >= 9.9
    # 100 px/s over an 80 px tall box = 1.25 heights/s, heading east (0 rad)
    mid = cars[0].index_at(3.0)
    assert abs(cars[0].speed[mid] - 1.25) < 0.05
    assert abs(cars[0].heading[mid]) < 0.05
    person = next(t for t in tracks if t.category == "person")
    assert np.nanmax(person.speed) < 0.01 and np.isnan(person.heading).all()


def test_flow_field_flags_wrong_way():
    params = load_params()
    objs = [obj(i, CAR, [(i * 0.5, 100, 600), (i * 0.5 + 8, 1800, 600)]) for i in range(1, 12)]
    objs.append(obj(99, CAR, [(2, 1800, 600), (10, 100, 600)]))
    ctx = build_context(make_obs(objs, duration=15), params, Scene(1920, 1080))
    wrong = next(t for t in ctx.tracks if t.id == 99)
    flags = ctx.flow.opposite(wrong.foot, wrong.heading, 8, 0.7)
    assert flags.mean() > 0.8
    good = next(t for t in ctx.tracks if t.id == 1)
    assert not ctx.flow.opposite(good.foot, good.heading, 8, 0.7).any()
