"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import random
import numpy as np
from collections import defaultdict, deque
from math import exp, hypot

import cv2

from src.features import VEHICLE_CLASSES
from src.pipeline import detect_video_events
from src.scene import load_scene
from src.traffic_lights import classify_vehicle_signal
from src.tracking import DEFAULT_WEIGHTS, ROAD_CLASSES, result_rows

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0
INFERENCE_SEED = 0


def _seed_inference() -> None:
    """Fix RNG state before model/tracker initialization for repeatable runs."""
    random.seed(INFERENCE_SEED)
    np.random.seed(INFERENCE_SEED)
    try:
        import torch

        torch.manual_seed(INFERENCE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(INFERENCE_SEED)
    except Exception:
        # Part A will report missing inference dependencies; Part B is optional.
        pass


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file. You may open it any way you like
            (OpenCV, decord, PyAV, ffmpeg), read it several times, sample
            frames, run batched models — anything goes.

    Returns:
        A list of events, each ``[start_sec, end_sec, label]`` with
        ``0 <= start_sec < end_sec <= duration`` (floats, seconds from the
        first frame) and ``label in CLASSES``. Return ``[]`` if nothing
        happened. Segments of the same class must not overlap.

    A typical pipeline:
        1. sample frames (every 2nd–5th frame is usually enough),
        2. detect road users (YOLO / RT-DETR) and track them (ByteTrack),
        3. turn trajectories + scene layout (lanes, stop line, crossing)
           into per-frame flags for each class,
        4. merge consecutive flags into segments, drop blips < 0.5 s,
           merge gaps < 1 s,
        5. optionally re-score `accident` / `near_miss` candidates with a
           learned clip classifier.
    """
    _seed_inference()
    return detect_video_events(video_path)


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus).

    The harness calls ``reset(meta)`` once per video and then ``step`` for
    EVERY frame, in order. ``step`` must use only the frames it has seen so
    far: do not open the video file inside this class, and do not reuse
    Part A results that were computed with access to future frames.
    """

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        _seed_inference()
        self.meta = dict(meta)
        self.last_score = 0.0
        self.frame_count = 0
        self.sample_stride = 10  # about 0.33 s at 30 fps
        self.last_update = 0.0
        self.device = "cpu"
        self.model = None
        self.scene, _ = load_scene()
        self.scene_registered = False
        self.history = defaultdict(lambda: deque(maxlen=5))
        self.signal_state = "?"
        self.previous_state = "?"
        try:
            import torch
            from ultralytics import YOLO

            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(str(DEFAULT_WEIGHTS))
        except Exception:
            # Part B is optional. Keep the causal interface safe if optional
            # inference dependencies or weights are unavailable.
            self.model = None

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s).

        Args:
            frame: BGR uint8 array of shape (H, W, 3) — OpenCV convention.
            t_sec: timestamp of this frame in seconds.

        Returns:
            A float in [0, 1]. Skipping frames internally and returning the
            previous score is fine; the harness still expects a value for
            every call.
        """
        self.frame_count += 1
        if not self.scene_registered:
            registered, _ = load_scene(frame)
            if registered is not None:
                self.scene = registered
            self.scene_registered = True
        elapsed = max(0.0, float(t_sec) - self.last_update)
        self.last_score *= exp(-elapsed / 1.5)
        self.last_update = float(t_sec)
        if self.frame_count % self.sample_stride:
            return float(np.clip(self.last_score, 0.0, 1.0))
        if self.model is None:
            self.last_score = 0.0
            return 0.0

        height, width = frame.shape[:2]
        target_width = min(width, 1920)
        if target_width != width:
            target_height = max(1, round(height * target_width / width))
            infer_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            infer_frame = frame
        coordinate_scale = width / target_width
        results = self.model.track(
            infer_frame,
            imgsz=640,
            conf=0.25,
            classes=list(ROAD_CLASSES),
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            quantize=16 if self.device != "cpu" else None,
            verbose=False,
        )
        result = results[0] if isinstance(results, (list, tuple)) else results
        rows = result_rows(result, self.frame_count, float(self.meta.get("fps", 30.0)), coordinate_scale)
        ref_width, ref_height = self.scene.get("ref_size", (1920, 1080))
        scale_x, scale_y = ref_width / width, ref_height / height

        # Causal red indication from the correct vehicle lamp (top lamp of the
        # three-lamp head); the left pedestrian head is never consulted.
        roi = self.scene["signals"]["veh_main"]["roi"]
        x1, y1, x2, y2 = [float(value) for value in roi]
        crop = frame[
            max(0, round(y1 * scale_y)):min(height, round(y2 * scale_y)),
            max(0, round(x1 * scale_x)):min(width, round(x2 * scale_x)),
        ]
        self.previous_state = self.signal_state
        self.signal_state, _ = classify_vehicle_signal(crop)

        current = {}
        for row in rows:
            cls = int(row["cls"])
            if cls not in VEHICLE_CLASSES and cls != 0:
                continue
            x1p, y1p, x2p, y2p = row["x1"] * scale_x, row["y1"] * scale_y, row["x2"] * scale_x, row["y2"] * scale_y
            point = np.asarray([(x1p + x2p) * 0.5, y2p], dtype=float)
            size = hypot(x2p - x1p, y2p - y1p)
            obj_id = int(row["id"])
            history = self.history[obj_id]
            history.append((float(t_sec), point, max(6.0, size * 0.25), cls))
            velocity = np.zeros(2, dtype=float)
            speed = 0.0
            if len(history) >= 2:
                old_t, old_point, _, _ = history[-2]
                dt = max(1e-3, float(t_sec) - old_t)
                velocity = (point - old_point) / dt
                speed = float(np.linalg.norm(velocity))
            current[obj_id] = (point, velocity, max(6.0, size * 0.25), cls, speed)

        threat_score = self._ttc_risk(current, float(t_sec))
        braking_score = self._braking_risk(float(t_sec))
        rule_score = self._red_or_wrong_way_risk(current)
        self.last_score = max(threat_score, braking_score, rule_score)
        return float(np.clip(self.last_score, 0.0, 1.0))

    def _ttc_risk(self, objects: dict, t_sec: float) -> float:
        rows = list(objects.values())
        highest = 0.0
        for index, left in enumerate(rows):
            p1, v1, r1, cls1, s1 = left
            if s1 < 3.0:
                continue
            for right in rows[index + 1:]:
                p2, v2, r2, cls2, s2 = right
                if s2 < 3.0 or (cls1 == cls2 == 0):
                    continue
                relative_position = p2 - p1
                relative_velocity = v2 - v1
                denom = float(np.dot(relative_velocity, relative_velocity))
                if denom < 1e-5:
                    continue
                ttc = -float(np.dot(relative_position, relative_velocity)) / denom
                if not 0.0 < ttc <= RISK_HORIZON_SEC:
                    continue
                miss_distance = float(np.linalg.norm(relative_position + relative_velocity * ttc))
                clearance = miss_distance - (r1 + r2)
                if clearance <= 25.0:
                    proximity = 1.0 / (1.0 + exp(min(40.0, max(-40.0, (clearance - 4.0) / 5.0))))
                    urgency = exp(-ttc / 4.0)
                    highest = max(highest, 0.9 * proximity * urgency)
        return highest

    def _braking_risk(self, t_sec: float) -> float:
        highest = 0.0
        for history in self.history.values():
            if len(history) < 3:
                continue
            old = history[-3]
            previous = history[-2]
            current = history[-1]
            dt1 = max(0.1, previous[0] - old[0])
            dt2 = max(0.1, current[0] - previous[0])
            speed1 = float(np.linalg.norm(previous[1] - old[1])) / dt1
            speed2 = float(np.linalg.norm(current[1] - previous[1])) / dt2
            decel = (speed1 - speed2) / dt2
            if decel >= 30.0 and speed1 >= 12.0:
                highest = max(highest, min(0.8, 0.52 + (decel - 30.0) / 100.0))
        return highest

    def _red_or_wrong_way_risk(self, objects: dict) -> float:
        if self.signal_state != "R":
            return 0.0
        junction = self.scene.get("zones", {}).get("junction", {}).get("polygon", [])
        near = self.scene.get("carriageways", {}).get("near", {})
        direction = np.asarray(near.get("direction", [0.92, 0.39]), dtype=float)
        highest = 0.0
        for point, velocity, _, cls, speed in objects.values():
            if cls == 0 or speed < 3.0:
                continue
            if _point_in_polygon(point, junction):
                highest = max(highest, 0.58)
                continue
            approach = np.asarray(near.get("polygon", []), dtype=float)
            if len(approach) and _point_in_polygon(point, approach):
                alignment = float(np.dot(velocity / max(speed, 1e-6), direction))
                if alignment < -0.6:
                    highest = max(highest, 0.55)
        return highest


def _point_in_polygon(point: np.ndarray, polygon) -> bool:
    vertices = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(vertices) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = vertices[-1]
    for current in vertices:
        ax, ay = previous
        bx, by = current
        if (ay > y) != (by > y) and x < (bx-ax) * (y-ay) / (by-ay+1e-12) + ax:
            inside = not inside
        previous = current
    return inside
