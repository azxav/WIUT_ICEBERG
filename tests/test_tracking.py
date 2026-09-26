import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src import tracking


class FakeCapture:
    def __init__(self, frame_count=7, width=4, height=2, fps=30.0):
        self.frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.width = width
        self.height = height
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
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
        }.get(prop, 0.0)

    def release(self):
        self.released = True

    def isOpened(self):
        return True


class FakeModel:
    def __init__(self):
        self.shapes = []
        self.calls = []

    def track(self, frame, **kwargs):
        self.shapes.append(frame.shape[:2])
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            id=np.array([7]),
            cls=np.array([2]),
            conf=np.array([0.93]),
            xyxy=np.array([[1.0, 2.0, 3.0, 4.0]]),
        )
        return [SimpleNamespace(boxes=boxes)]


class TrackingTests(unittest.TestCase):
    def test_default_inference_config_uses_fp16_quantize_precision(self):
        self.assertEqual(tracking.TrackConfig().quantize, 16)

    def test_video_discovery_deduplicates_case_insensitive_extension_matches(self):
        with TemporaryDirectory() as directory:
            expected = Path(directory) / "C3902.MP4"
            expected.touch()

            videos = tracking.find_videos(directory)

        self.assertEqual(videos, [expected])

    def test_stride_sampling_retrieves_only_selected_frames(self):
        capture = FakeCapture(frame_count=8)

        sampled = list(tracking.sampled_frames(capture, stride=3))

        self.assertEqual([index for index, _ in sampled], [0, 3, 6])
        self.assertEqual(capture.retrieved, [0, 3, 6])

    def test_detection_boxes_are_scaled_back_to_source_coordinates(self):
        result = SimpleNamespace(boxes=SimpleNamespace(
            id=np.array([7]), cls=np.array([2]), conf=np.array([0.93]),
            xyxy=np.array([[10.0, 20.0, 50.0, 60.0]]),
        ))

        rows = tracking.result_rows(result, frame_index=6, fps=30.0, coordinate_scale=2.0)

        self.assertEqual(rows, [{
            "frame": 6, "t": 0.2, "id": 7, "cls": 2, "conf": 0.93,
            "x1": 20.0, "y1": 40.0, "x2": 100.0, "y2": 120.0,
        }])

    def test_track_video_uses_stride_resizes_and_returns_source_scale_tracks(self):
        capture = FakeCapture(frame_count=7)
        model = FakeModel()
        config = tracking.TrackConfig(stride=3, imgsz=32, max_frame_width=2, device="cpu", quantize=None)

        with patch.object(tracking.cv2, "VideoCapture", return_value=capture):
            tracks = tracking.track_video("clip.mp4", config=config, model=model)

        self.assertEqual(tracks.frame.tolist(), [0, 3, 6])
        self.assertEqual(tracks.id.tolist(), [7, 7, 7])
        self.assertEqual(tracks.x1.tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(model.shapes, [(1, 2), (1, 2), (1, 2)])
        self.assertTrue(capture.released)
        self.assertTrue(all(call["persist"] for call in model.calls))
        self.assertTrue(all(call["quantize"] is None for call in model.calls))
        self.assertTrue(all(call["classes"] == [0, 1, 2, 3, 5, 7] for call in model.calls))


if __name__ == "__main__":
    unittest.main()
