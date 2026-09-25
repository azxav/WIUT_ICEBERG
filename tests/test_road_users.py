from src.config import CAR, MOTORCYCLE, PERSON, load_params
from src.pipeline import build_context
from src.rules import RULES
from src.scene import Scene
from tests.synth import H, W, make_obs, obj

PED = (40, 100)

# road: horizontal band y in [432, 864] px; crosswalk: x in [864, 1056] px across the road
ROAD = [[[0, 0.4], [1, 0.4], [1, 0.8], [0, 0.8]]]
CROSSWALK = {"id": "cw", "polygon": [[0.45, 0.4], [0.55, 0.4], [0.55, 0.8], [0.45, 0.8]]}


def ctx_for(objects, duration, scene: dict | None = None):
    params = load_params()
    return build_context(make_obs(objects, duration), params, Scene.from_dict(scene or {}, W, H))


def run(label, ctx):
    return RULES[label](ctx)


# ---------------------------------------------------------------- jaywalking
def test_jaywalking_crossing_outside_crosswalk():
    # walks from the sidewalk (y=380) across the road to y=920 at 54 px/s: on road t~1..9
    ped = obj(1, PERSON, [(0, 400, 380), (10, 400, 920)], size=PED)
    ev = run("jaywalking", ctx_for([ped], 12, {"carriageway": ROAD, "crosswalks": [CROSSWALK]}))
    assert len(ev) == 1
    assert abs(ev[0].start - 52 / 54) < 0.3 and abs(ev[0].end - 484 / 54) < 0.3
    assert ev[0].track_ids == (1,)


def test_jaywalking_ignores_crosswalk_rider_and_sidewalk():
    scene = {"carriageway": ROAD, "crosswalks": [CROSSWALK]}
    on_cw = obj(1, PERSON, [(0, 960, 380), (10, 960, 920)], size=PED)
    kerb = obj(2, PERSON, [(0, 200, 440), (10, 1500, 440)], size=PED)  # kerb: 8 px inside the road edge
    moto = obj(3, MOTORCYCLE, [(0, 100, 700), (10, 1800, 700)], size=(60, 80))
    rider = obj(4, PERSON, [(0, 100, 680), (10, 1800, 680)], size=PED)
    assert run("jaywalking", ctx_for([on_cw, kerb, moto, rider], 12, scene)) == []


def test_jaywalking_simultaneous_people_is_one_event():
    scene = {"carriageway": ROAD}
    peds = [obj(1, PERSON, [(0, 300, 380), (10, 300, 920)], size=PED),
            obj(2, PERSON, [(2, 1500, 380), (12, 1500, 920)], size=PED)]
    ev = run("jaywalking", ctx_for(peds, 14, scene))
    assert len(ev) == 1 and ev[0].track_ids == (1, 2)


def test_jaywalking_falls_back_to_flow_road_mask():
    # no scene: road = cells cars drive through (rows y=620..740)
    cars = [obj(10 + 6 * r + i, CAR, [(i * 1.5, 100, y), (i * 1.5 + 8, 1800, y)])
            for r, y in enumerate((620, 660, 700, 740)) for i in range(6)]
    ped = obj(1, PERSON, [(0, 960, 380), (10, 960, 920)], size=PED)
    ctx = ctx_for(cars + [ped], 20)
    ev = run("jaywalking", ctx)
    assert len(ev) == 1 and 3.5 < ev[0].start < 5.0 and 6.5 < ev[0].end < 8.0
    assert run("jaywalking", ctx_for(cars, 20)) == []


# ---------------------------------------------------------- failure_to_yield
def test_failure_to_yield_vehicle_passes_pedestrian_on_crosswalk():
    scene = {"carriageway": ROAD, "crosswalks": [CROSSWALK]}
    ped = obj(1, PERSON, [(0, 960, 400), (20, 960, 900)], size=PED)   # on the crossing t~1..19
    car = obj(2, CAR, [(0, 100, 700), (10, 1800, 700)])               # 170 px/s eastwards
    ev = run("failure_to_yield", ctx_for([ped, car], 22, scene))
    assert len(ev) == 1 and ev[0].track_ids == (2,)
    # bottom edge (+-36 px) inside x in [864, 1056]: t ~ (828-100)/170 .. (1092-100)/170
    assert abs(ev[0].start - 728 / 170) < 0.3 and abs(ev[0].end - 992 / 170) < 0.3


def test_failure_to_yield_negatives():
    scene = {"carriageway": ROAD, "crosswalks": [CROSSWALK]}
    car = obj(2, CAR, [(0, 100, 700), (10, 1800, 700)])
    crossed_before = obj(1, PERSON, [(0, 960, 400), (3, 960, 900)], size=PED)  # gone by t=3.5
    waiting_far = obj(3, PERSON, [(0, 400, 380), (10, 400, 380)], size=PED)   # sidewalk, off the crossing
    assert run("failure_to_yield", ctx_for([car, crossed_before, waiting_far], 12, scene)) == []
    # the car waits at the crossing (stationary) while the pedestrian crosses, then goes
    ped = obj(1, PERSON, [(0, 960, 400), (8, 960, 900)], size=PED)
    waiting_car = obj(2, CAR, [(0, 100, 700), (3, 800, 700), (9, 800, 700), (13, 1800, 700)])
    assert run("failure_to_yield", ctx_for([ped, waiting_car], 15, scene)) == []
    # no crosswalks in the scene -> rule is silent
    assert run("failure_to_yield", ctx_for([car, crossed_before], 12, {"carriageway": ROAD})) == []
