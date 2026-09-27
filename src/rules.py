"""Conservative traffic-event rules over mapped, smoothed track features."""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .features import VEHICLE_CLASSES, _inside, _inside_many
from .segments import merge_events, merge_intervals


@dataclass(frozen=True)
class RuleConfig:
    slow_speed_px_s: float = 8.0
    stop_speed_px_s: float = 4.0
    jaywalk_min_sec: float = 1.2
    jaywalk_min_displacement_px: float = 14.0
    fty_vehicle_min_speed_px_s: float = 12.0
    fty_pedestrian_margin_px: float = 12.0
    min_lanes_for_congestion: int = 4
    min_slow_vehicles_for_congestion: int = 5
    merge_gap_sec: float = 1.0
    min_event_sec: float = 0.5
    include_unvalidated_classes: bool = False


# Current manual GT supports these classes with positive examples. Jaywalking
# and stop_line stay in CLASSES, but emit no candidates in the current scored
# run; C11 keeps them suppressed until a rule demonstrates >=0.7 dev precision.
VALIDATED_CLASSES = {
    "congestion", "failure_to_yield", "red_light"
}
VEHICLE_TYPES = set(VEHICLE_CLASSES)


def _duration_step(times: np.ndarray, fallback: float = 0.1) -> float:
    diffs = np.diff(np.unique(times))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else fallback


def _mask_intervals(
    times: Iterable[float], mask: Iterable[bool], *, gap: float = 0.35, fallback_step: float = 0.1
) -> list[tuple[float, float]]:
    t = np.asarray(list(times), dtype=float)
    active = np.asarray(list(mask), dtype=bool)
    if not len(t) or len(t) != len(active):
        return []
    order = np.argsort(t, kind="stable")
    t, active = t[order], active[order]
    step = _duration_step(t, fallback_step)
    intervals: list[tuple[float, float]] = []
    starts: list[float] = []
    ends: list[float] = []
    for time, hit in zip(t, active):
        if not hit:
            continue
        if starts and time - ends[-1] <= gap:
            ends[-1] = time + step
        else:
            starts.append(float(time))
            ends.append(float(time + step))
    intervals.extend(zip(starts, ends))
    return intervals


def _track_intervals(group: pd.DataFrame, mask: pd.Series | np.ndarray, *, gap: float = 0.35) -> list[tuple[float, float]]:
    ordered = group.sort_values("t", kind="stable")
    return _mask_intervals(ordered["t"].to_numpy(), np.asarray(mask, dtype=bool), gap=gap)


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def _box_overlap_fraction(a, b) -> float:
    x1, y1 = max(float(a.x1), float(b.x1)), max(float(a.y1), float(b.y1))
    x2, y2 = min(float(a.x2), float(b.x2)), min(float(a.y2), float(b.y2))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, (float(a.x2) - float(a.x1)) * (float(a.y2) - float(a.y1)))
    area_b = max(1.0, (float(b.x2) - float(b.x1)) * (float(b.y2) - float(b.y1)))
    return intersection / min(area_a, area_b)


def _rider_flags(persons: pd.DataFrame, vehicle_rows: pd.DataFrame) -> np.ndarray:
    """Mark person detections that are duplicate rider boxes on a bike/motorcycle."""
    if persons.empty or vehicle_rows.empty:
        return np.zeros(len(persons), dtype=bool)
    frame_col = "frame" if "frame" in persons and "frame" in vehicle_rows else "t"
    vehicle_groups = {
        key: list(group.itertuples(index=False))
        for key, group in vehicle_rows.groupby(frame_col, sort=False)
    }
    flags = []
    for person in persons.itertuples(index=False):
        key = getattr(person, frame_col)
        nearby = vehicle_groups.get(key, ())
        flags.append(any(
            int(vehicle.cls) in (1, 3) and _box_overlap_fraction(person, vehicle) >= 0.24
            for vehicle in nearby
        ))
    return np.asarray(flags, dtype=bool)


def _person_road_intervals(features: pd.DataFrame, scene: Mapping, config: RuleConfig) -> list[tuple[float, float]]:
    persons = features[features["cls"] == 0]
    vehicles = features[features["cls"].isin((1, 3))]
    persons = persons.assign(_rider_duplicate=_rider_flags(persons, vehicles))
    raw: list[tuple[float, float]] = []
    crosswalks = scene.get("crosswalks", {})
    # Re-evaluate with a wider edge tolerance to avoid calling a person a
    # jaywalker because one foot sample falls just outside a zebra polygon.
    for _, group in persons.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable")
        rider_mask = group["_rider_duplicate"].to_numpy(dtype=bool)
        points = np.column_stack([group["foot_x"].to_numpy(), group["foot_y"].to_numpy()])
        on_crossing = np.zeros(len(group), dtype=bool)
        for shape in crosswalks.values():
            on_crossing |= _inside_many(points, shape.get("polygon", []), margin=20.0)
        mask = group["in_road"].to_numpy(dtype=bool) & ~on_crossing & ~rider_mask
        intervals = _track_intervals(group, mask, gap=0.45)
        for start, end in intervals:
            subset = group[group["t"].between(start - 0.2, end + 0.2)]
            if end - start < config.jaywalk_min_sec or len(subset) < 2:
                continue
            displacement = hypot(
                float(subset.iloc[-1].foot_x - subset.iloc[0].foot_x),
                float(subset.iloc[-1].foot_y - subset.iloc[0].foot_y),
            )
            if displacement >= config.jaywalk_min_displacement_px:
                raw.append((start, end))
    return merge_intervals(raw, max_gap=config.merge_gap_sec, min_duration=config.jaywalk_min_sec)


def _crosswalk_intervals(
    group: pd.DataFrame,
    polygon: list,
    *,
    pedestrian: bool,
    direction: list[float],
    config: RuleConfig,
) -> list[tuple[float, float]]:
    points = np.column_stack([group["foot_x"].to_numpy(), group["foot_y"].to_numpy()])
    if pedestrian:
        margin = config.fty_pedestrian_margin_px
        hits = _inside_many(points, polygon, margin=margin)
    else:
        # Approximate body occupancy with front, center and rear ground points.
        # This preserves crossing duration without the large radial padding
        # that would count vehicles merely travelling beside a zebra.
        hits = _inside_many(points, polygon, margin=8.0)
        front = np.column_stack([group["front_x"].to_numpy(), group["front_y"].to_numpy()])
        rear = np.column_stack([group["rear_x"].to_numpy(), group["rear_y"].to_numpy()])
        hits |= _inside_many(front, polygon, margin=8.0)
        hits |= _inside_many(rear, polygon, margin=8.0)
    intervals = _track_intervals(group, hits, gap=0.25)
    return merge_intervals(intervals, max_gap=0.25, min_duration=0.15)


def _failure_to_yield(features: pd.DataFrame, scene: Mapping, config: RuleConfig) -> list[list]:
    crossings = scene.get("crosswalks", {})
    pedestrians = features[features["cls"] == 0]
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    rider_boxes = features[features["cls"].isin((1, 3))]
    rider_flags = _rider_flags(pedestrians, rider_boxes)
    pedestrians = pedestrians.assign(_rider_duplicate=rider_flags)
    pedestrian_intervals: dict[str, list[tuple[float, float]]] = {name: [] for name in crossings}
    for _, group in pedestrians.groupby("id", sort=False):
        group = group[~group["_rider_duplicate"]]
        for name, shape in crossings.items():
            pedestrian_intervals[name].extend(_crosswalk_intervals(
                group, shape.get("polygon", []), pedestrian=True, direction=[0.92, 0.39], config=config
            ))

    output: list[list] = []
    for _, group in vehicles.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable")
        if len(group) < 2:
            continue
        road = str(group.iloc[len(group) // 2].get("road_id") or "near")
        direction = scene.get("carriageways", {}).get(road, {}).get("direction", [0.92, 0.39])
        if road not in ("near", "far"):
            direction = [0.92, 0.39]
        mean_speed = float(group["speed"].median())
        displacement = hypot(
            float(group.iloc[-1].cx - group.iloc[0].cx),
            float(group.iloc[-1].cy - group.iloc[0].cy),
        )
        group_min_speed = 3.0 if int(group.iloc[0].cls) == 1 else config.fty_vehicle_min_speed_px_s
        if mean_speed < group_min_speed or displacement < 8.0:
            continue
        for name, shape in crossings.items():
            vehicle_intervals = _crosswalk_intervals(
                group, shape.get("polygon", []), pedestrian=False, direction=direction, config=config
            )
            for interval in vehicle_intervals:
                local = group[group["t"].between(interval[0] - 0.12, interval[1] + 0.12)]
                if len(local) < 2:
                    continue
                local_min_speed = 3.0 if int(local.iloc[0].cls) == 1 else config.fty_vehicle_min_speed_px_s
                if float(local["speed"].median()) < local_min_speed:
                    continue
                travel = hypot(
                    float(local.iloc[-1].cx - local.iloc[0].cx),
                    float(local.iloc[-1].cy - local.iloc[0].cy),
                )
                if travel < 8.0:
                    continue
                if any(_overlaps(interval, ped) for ped in pedestrian_intervals[name]):
                    output.append([interval[0], interval[1], "failure_to_yield"])
    return output


def _congestion(features: pd.DataFrame, scene: Mapping, config: RuleConfig) -> list[list]:
    near_lanes = [name for name in scene.get("lanes", {}) if name.startswith("near_")]
    if not near_lanes:
        return []
    candidates = features[
        features["cls"].isin(VEHICLE_TYPES)
        & features["in_queue"].astype(bool)
        & (features["road_id"] == "near")
        & features["lane_id"].isin(near_lanes)
        & (features["speed"] <= config.slow_speed_px_s)
    ]
    if candidates.empty:
        return []
    samples: list[tuple[float, bool]] = []
    for time, frame in candidates.groupby(candidates["t"].round(2), sort=True):
        lanes = set(frame["lane_id"].dropna().astype(str))
        n_vehicles = int(frame["id"].nunique())
        samples.append((float(time), len(lanes) >= config.min_lanes_for_congestion
                        and n_vehicles >= config.min_slow_vehicles_for_congestion))
    times, mask = zip(*samples)
    intervals = _mask_intervals(times, mask, gap=0.45)
    intervals = merge_intervals(intervals, max_gap=config.merge_gap_sec, min_duration=3.0)
    return [[start, end, "congestion"] for start, end in intervals]


def _signal_at(signal_states: pd.DataFrame, time: float, max_distance: float = 0.22) -> str:
    if signal_states is None or signal_states.empty or "state" not in signal_states:
        return "?"
    distances = (signal_states["t"].astype(float) - float(time)).abs()
    index = distances.idxmin()
    if float(distances.loc[index]) > max_distance:
        return "?"
    return str(signal_states.loc[index, "state"])


def _stable_signal_at(signal_states: pd.DataFrame, time: float, lookaround: float = 0.18) -> str:
    if signal_states is None or signal_states.empty:
        return "?"
    nearby = signal_states[(signal_states["t"] >= time - lookaround) & (signal_states["t"] <= time + lookaround)]
    states = [state for state in nearby["state"].astype(str) if state in {"R", "Y", "G"}]
    if not states:
        return _signal_at(signal_states, time)
    counts = {state: states.count(state) for state in {"R", "Y", "G"}}
    state = max(counts, key=counts.get)
    second = max(value for key, value in counts.items() if key != state)
    return state if (len(states) == 1 or (counts[state] >= 2 and counts[state] > second)) else "?"


def _next_stable_green(signal_states: pd.DataFrame, time: float) -> float | None:
    """Find the start of a sustained green run after `time`."""
    if signal_states is None or signal_states.empty:
        return None
    later = signal_states[(signal_states["t"] >= time) & (signal_states["state"] == "G")].sort_values("t")
    times = later["t"].to_numpy(dtype=float)
    for index, start in enumerate(times):
        run = times[index:index + 3]
        if len(run) >= 2 and run[-1] - run[0] <= 0.35:
            return float(start)
    return None


def _red_and_stopline(
    features: pd.DataFrame,
    signal_states: pd.DataFrame,
    scene: Mapping,
    duration: float,
    config: RuleConfig,
) -> list[list]:
    line = scene.get("stop_lines", {}).get("near", {}).get("line", [])
    direction = scene.get("carriageways", {}).get("near", {}).get("direction", [0.92, 0.39])
    if not line:
        return []
    output: list[list] = []
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    for _, group in vehicles.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable").reset_index(drop=True)
        if len(group) < 2:
            continue
        first_road = str(group.iloc[0].get("road_id") or "")
        if first_road not in ("near", "") and not group["in_queue"].any():
            continue
        signed = group["front_stopline_signed"].to_numpy(dtype=float)
        times = group["t"].to_numpy(dtype=float)

        # A short positive-side crossing is noise from box jitter. Confirm that
        # the front advances through the line before treating it as a violation.
        red_crossings = []
        for index in range(1, len(group)):
            if signed[index - 1] >= 12.0 or signed[index] < 12.0:
                continue
            time = float(times[index])
            if _stable_signal_at(signal_states, time) != "R":
                continue
            future = group.iloc[index:]
            confirmed = future[
                (future["t"] <= time + 1.2)
                & (future["front_stopline_signed"] > 48.0)
                & (future["speed"] > 5.0)
            ]
            if len(confirmed):
                red_crossings.append(index)

        # A stop-line violation requires the front to be clearly past the line,
        # remain there on red, and not proceed into the intersection before green.
        stopped_past = group[
            (group["speed"] <= config.stop_speed_px_s)
            & (group["front_stopline_signed"] >= 10.0)
            & (group["front_stopline_signed"] <= 48.0)
        ]
        stops = _mask_intervals(stopped_past["t"].to_numpy(), np.ones(len(stopped_past), dtype=bool), gap=0.35)
        valid_stop = None
        for start, end in stops:
            if end - start < 0.65 or _stable_signal_at(signal_states, start) != "R":
                continue
            green_time = _next_stable_green(signal_states, start)
            horizon = green_time if green_time is not None else min(duration, float(group.iloc[-1].t) + 0.1)
            before_green = group[(group["t"] >= start) & (group["t"] < horizon)]
            proceeded_before_green = before_green[
                (before_green["front_stopline_signed"] > 48.0) & (before_green["speed"] > 5.0)
            ]
            if len(proceeded_before_green):
                continue
            valid_stop = (float(start), float(horizon))
            break

        if valid_stop is not None:
            start, end = valid_stop
            if end - start >= config.min_event_sec:
                output.append([start, end, "stop_line"])
            continue

        if not red_crossings:
            continue
        crossing_index = red_crossings[0]
        crossing_time = float(times[crossing_index])
        # The event lasts until the tracked vehicle leaves view/intersection.
        event_end = min(duration, float(group.iloc[-1].t) + 0.1)
        output.append([crossing_time, max(crossing_time + 0.1, event_end), "red_light"])
    return output


def _stopped_vehicle(features: pd.DataFrame, duration: float) -> list[list]:
    output = []
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    for _, group in vehicles.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable")
        if group["in_queue"].mean() > 0.15 or group["road_id"].eq("far").mean() > 0.5:
            continue
        mask = group["in_road"].to_numpy(dtype=bool) & (group["speed"].to_numpy() <= 2.5)
        for start, end in _track_intervals(group, mask, gap=0.5):
            if end - start >= 10.0:
                output.append([start, min(duration, end), "stopped_vehicle"])
    return output


def _wrong_way(features: pd.DataFrame, scene: Mapping) -> list[list]:
    output = []
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    for _, group in vehicles.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable")
        wrong = np.zeros(len(group), dtype=bool)
        for i, row in enumerate(group.itertuples(index=False)):
            road = str(row.road_id or "")
            lane_id = str(row.lane_id or "")
            if road not in ("near", "far") or not lane_id or row.in_junction or lane_id == "near_curb":
                continue
            direction = np.asarray(scene.get("carriageways", {}).get(road, {}).get("direction", [0.0, 0.0]), dtype=float)
            velocity = np.asarray([np.cos(float(row.heading_rad)), np.sin(float(row.heading_rad))])
            wrong[i] = float(np.dot(velocity, direction)) < -0.55 and float(row.speed) > 5.0
        for start, end in _track_intervals(group, wrong, gap=0.35):
            if end - start >= 1.0:
                output.append([start, end, "wrong_way"])
    return output


def _solid_line_crossing(features: pd.DataFrame, scene: Mapping) -> list[list]:
    output = []
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    for _, group in vehicles.groupby("id", sort=False):
        group = group.sort_values("t", kind="stable").reset_index(drop=True)
        for name, shape in scene.get("solid_lines", {}).items():
            line = shape.get("polyline", [])
            if len(line) < 2:
                continue
            points = list(zip(group["cx"].astype(float), group["foot_y"].astype(float)))
            signed = [signed_distance_to_polyline(point, line) for point in points]
            for i in range(1, len(group)):
                if signed[i - 1] * signed[i] >= 0 or min(abs(signed[i - 1]), abs(signed[i])) > 28.0:
                    continue
                if max(float(group.iloc[i - 1].speed), float(group.iloc[i].speed)) < 5.0:
                    continue
                output.append([float(group.iloc[i - 1].t), float(group.iloc[i].t), "solid_line_crossing"])
    return output


def _accidents_and_near_misses(features: pd.DataFrame) -> list[list]:
    """High-confidence box-contact/sudden-stop candidates; disabled by default."""
    output = []
    vehicles = features[features["cls"].isin(VEHICLE_TYPES)]
    times = sorted(vehicles["t"].round(2).unique())
    for time in times:
        frame = vehicles[(vehicles["t"] - time).abs() <= 0.02]
        rows = list(frame.itertuples(index=False))
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                overlap = _box_overlap_fraction(left, right)
                if overlap >= 0.35 and float(left.speed) < 3.0 and float(right.speed) < 3.0:
                    output.append([float(time) - 0.2, float(time) + 0.5, "accident"])
                elif overlap >= 0.12 and min(float(left.speed), float(right.speed)) < 4.0:
                    output.append([float(time) - 0.3, float(time) + 0.6, "near_miss"])
    return output


def generate_events(
    features: pd.DataFrame,
    signal_states: pd.DataFrame,
    scene: Mapping,
    duration: float,
    config: RuleConfig | None = None,
) -> list[list]:
    """Run event rules and apply the required 1 s merge / 0.5 s minimum."""
    cfg = config or RuleConfig()
    if features.empty:
        return []
    events: list[list] = []
    events.extend(_congestion(features, scene, cfg))
    events.extend([[start, end, "jaywalking"] for start, end in _person_road_intervals(features, scene, cfg)])
    events.extend(_failure_to_yield(features, scene, cfg))
    events.extend(_red_and_stopline(features, signal_states, scene, duration, cfg))

    if cfg.include_unvalidated_classes:
        events.extend(_stopped_vehicle(features, duration))
        events.extend(_wrong_way(features, scene))
        events.extend(_solid_line_crossing(features, scene))
        events.extend(_accidents_and_near_misses(features))

    allowed = VALIDATED_CLASSES | ({
        "stopped_vehicle", "wrong_way", "solid_line_crossing", "accident", "near_miss"
    } if cfg.include_unvalidated_classes else set())
    events = [event for event in events if event[2] in allowed]
    return merge_events(
        events,
        max_gap=cfg.merge_gap_sec,
        min_duration=cfg.min_event_sec,
        duration=duration,
    )
