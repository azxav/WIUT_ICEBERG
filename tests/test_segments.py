import unittest

from src.segments import merge_events, merge_intervals


class SegmentTests(unittest.TestCase):
    def test_merges_short_gaps_and_drops_subminimum_fragments(self):
        intervals = [(0.0, 0.2), (1.0, 2.0), (2.5, 3.5), (5.0, 5.3)]

        self.assertEqual(merge_intervals(intervals, max_gap=0.6, min_duration=0.5), [(1.0, 3.5)])

    def test_same_class_events_do_not_overlap_but_different_classes_can(self):
        events = [
            [1.0, 3.0, "red_light"],
            [2.0, 4.0, "red_light"],
            [2.5, 3.5, "jaywalking"],
        ]

        self.assertEqual(
            merge_events(events, max_gap=0.0, min_duration=0.5),
            [[1.0, 4.0, "red_light"], [2.5, 3.5, "jaywalking"]],
        )


if __name__ == "__main__":
    unittest.main()
