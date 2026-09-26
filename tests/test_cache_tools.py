import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from src.tracking import TrackConfig
from tools import build_track_cache


class TrackCacheToolTests(unittest.TestCase):
    def test_cpu_device_builds_full_precision_config(self):
        args = SimpleNamespace(
            stride=3, imgsz=960, conf=0.2, weights=Path("weights/yolo11s.pt"),
            device="cpu", cpu=False,
        )

        config = build_track_cache.build_config(args)

        self.assertEqual(config.device, "cpu")
        self.assertIsNone(config.quantize)

    def test_legacy_cache_is_reused_only_for_its_recorded_settings(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            cached = cache_dir / "C3905_s3_960_yolo11s.parquet"
            cached.touch()
            video = Path("C3905.MP4")

            self.assertEqual(build_track_cache.legacy_cache_path(video, cache_dir, TrackConfig()), cached)
            self.assertIsNone(
                build_track_cache.legacy_cache_path(video, cache_dir, TrackConfig(conf=0.3),)
            )
            self.assertIsNone(
                build_track_cache.legacy_cache_path(video, cache_dir, TrackConfig(device="cpu", quantize=None))
            )


if __name__ == "__main__":
    unittest.main()
