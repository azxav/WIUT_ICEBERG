import numpy as np

from src.ttc import (box_gap, closing_speed, footprint, heading_change, pairwise_ttc, speed_drop,
                     ttc_boxes)

HALF = np.array([10.0, 5.0])


def test_head_on():
    # centres 100 apart, half widths 10 + 10, closing at 20 px/s -> touch after (100 - 20) / 20 = 4 s
    t = ttc_boxes([0, 0], HALF, [10, 0], [100, 0], HALF, [-10, 0])
    assert np.isclose(t, 4.0)
    assert closing_speed([0, 0], [10, 0], [100, 0], [-10, 0]) == 20.0


def test_crossing_hit_and_miss():
    # A goes east, B goes south; both reach the origin region at the same time
    hit = ttc_boxes([-50, 0], HALF, [10, 0], [0, -50], HALF, [0, 10])
    assert np.isfinite(hit) and 2.5 < hit < 5.0
    # B arrives far too late: A has left the crossing box before B's y-interval opens
    miss = ttc_boxes([-50, 0], HALF, [10, 0], [0, -500], HALF, [0, 10])
    assert np.isinf(miss)


def test_parallel_and_overlapping():
    assert np.isinf(ttc_boxes([0, 0], HALF, [10, 0], [0, 30], HALF, [10, 0]))   # same velocity
    assert np.isinf(ttc_boxes([0, 0], HALF, [10, 0], [50, 30], HALF, [5, 0]))   # different lane
    assert ttc_boxes([0, 0], HALF, [0, 0], [5, 0], HALF, [0, 0]) == 0.0        # already overlap
    assert np.isinf(ttc_boxes([0, 0], HALF, [0, 0], [100, 0], HALF, [10, 0]))  # diverging


def test_pairwise_matches_scalar():
    c = np.array([[0, 0], [100, 0], [0, 300]], float)
    v = np.array([[10, 0], [-10, 0], [0, 0]], float)
    h = np.tile(HALF, (3, 1))
    m = pairwise_ttc(c, h, v)
    assert m.shape == (3, 3) and np.isinf(np.diag(m)).all()
    assert np.isclose(m[0, 1], 4.0) and np.isclose(m[1, 0], 4.0) and np.isinf(m[0, 2])


def test_footprint_and_gap():
    c, h = footprint(np.array([[0, 0, 20, 100]]), frac=0.4)
    assert np.allclose(c, [[10, 80]]) and np.allclose(h, [[10, 20]])
    assert box_gap([0, 0], HALF, [25, 0], HALF) == 5.0
    assert box_gap([0, 0], HALF, [15, 0], HALF) < 0


def test_speed_drop_and_heading_change():
    t = np.arange(0, 3, 0.1)
    speed = np.where(t < 1.5, 2.0, 0.0)
    drop = speed_drop(t, speed, 1.0)
    assert drop[t < 1.5].max() == 0.0 and np.isclose(drop[-1], 0.0) and drop.max() == 2.0
    heading = np.where(t < 1.5, 0.0, np.pi / 2)
    heading[5] = np.nan
    ch = heading_change(t, heading, 0.5)
    assert np.isclose(np.nanmax(ch), np.pi / 2) and np.isnan(ch[5])
    # wrap-around: from +3.1 to -3.1 rad is a small turn
    assert heading_change(np.array([0.0, 1.0]), np.array([3.1, -3.1]), 1.0)[1] < 0.1
