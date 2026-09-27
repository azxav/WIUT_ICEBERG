import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src import traffic_lights


class FakeCapture:
    def __init__(self, frames, fps=10.0):
        self.frames = frames
        self.fps = fps
        self.index = -1
        self.retrieved = []
        self.released = False

    def grab(self):
        self.index += 1
        return self.index < len(self.frames)

    def retrieve(self):
        self.retrieved.append(self.index)
        return True, self.frames[self.index]

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: 8,
            cv2.CAP_PROP_FRAME_HEIGHT: 4,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
        }.get(prop, 0.0)

    def release(self):
        self.released = True

    def isOpened(self):
        return True


class TrafficLightTests(unittest.TestCase):
    def test_hsv_counts_and_classifies_dominant_red_green_and_amber(self):
        colors = {
            "R": (0, 0, 255),
            "Y": (0, 220, 255),
            "G": (0, 255, 0),
        }
        for expected, bgr in colors.items():
            with self.subTest(expected=expected):
                crop = np.zeros((12, 12, 3), dtype=np.uint8)
                crop[2:10, 2:10] = bgr
                state, counts = traffic_lights.classify_signal(crop)
                self.assertEqual(state, expected)
                self.assertGreaterEqual(counts[expected], 5)

    def test_low_or_ambiguous_color_counts_return_unknown(self):
        self.assertEqual(traffic_lights.classify_counts({"R": 3, "Y": 0, "G": 0}), "?")
        self.assertEqual(traffic_lights.classify_counts({"R": 12, "Y": 10, "G": 0}), "?")

    def test_vehicle_signal_uses_lamp_position_to_separate_amber_from_red(self):
        amber = np.zeros((90, 30, 3), dtype=np.uint8)
        amber[34:57, 5:25] = (0, 200, 255)
        state, counts = traffic_lights.classify_vehicle_signal(amber)
        self.assertEqual(state, "Y")
        self.assertGreater(counts["Y"], 0)

        red = np.zeros_like(amber)
        red[5:27, 5:25] = (0, 0, 255)
        self.assertEqual(traffic_lights.classify_vehicle_signal(red)[0], "R")

        green = np.zeros_like(amber)
        green[64:86, 5:25] = (0, 255, 0)
        self.assertEqual(traffic_lights.classify_vehicle_signal(green)[0], "G")

    def test_vehicle_signal_rejects_red_reflection_when_green_lamp_is_stronger(self):
        mixed = np.zeros((90, 30, 3), dtype=np.uint8)
        mixed[4:9, 4:10] = (0, 0, 255)  # 30 red pixels in the top lamp slice
        mixed[67:73, 4:14] = (0, 255, 0)  # 60 green pixels in the bottom slice
        state, counts = traffic_lights.classify_vehicle_signal(mixed)
        self.assertEqual(state, "G")
        self.assertGreater(counts["G"], counts["R"])

    def test_vehicle_signal_returns_unknown_when_two_lamp_slices_compete(self):
        mixed = np.zeros((90, 30, 3), dtype=np.uint8)
        mixed[4:14, 4:14] = (0, 0, 255)
        mixed[64:74, 4:14] = (0, 255, 0)
        self.assertEqual(traffic_lights.classify_vehicle_signal(mixed)[0], "?")

    def test_signal_cache_samples_every_stride_in_video_time(self):
        red = np.zeros((4, 8, 3), dtype=np.uint8)
        red[:, :] = (0, 0, 255)
        frames = [red.copy() for _ in range(7)]
        capture = FakeCapture(frames)

        with patch.object(traffic_lights.cv2, "VideoCapture", return_value=capture):
            states = traffic_lights.extract_signal_states(
                "clip.mp4", roi=(0, 0, 4, 2), sample_stride=3,
                scene_size=(4, 2),
            )

        self.assertEqual(states.frame.tolist(), [0, 3, 6])
        self.assertEqual(states.state.tolist(), ["R", "R", "R"])
        self.assertEqual(states.t.tolist(), [0.0, 0.3, 0.6])
        self.assertEqual(capture.retrieved, [0, 3, 6])
        self.assertTrue(capture.released)

    def test_small_signal_roi_is_cropped_at_source_resolution(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[:] = (0, 255, 0)
        frame[:, 2:4] = (0, 0, 255)
        capture = FakeCapture([frame])

        with patch.object(traffic_lights.cv2, "VideoCapture", return_value=capture):
            states = traffic_lights.extract_signal_states(
                "clip.mp4", roi=(1, 0, 2, 2), sample_stride=1,
                scene_size=(4, 2),
            )

        self.assertEqual(states.state.tolist(), ["R"])


if __name__ == "__main__":
    unittest.main()
