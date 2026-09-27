import unittest
from unittest.mock import patch

import pandas as pd

from src import rules


class RulePolicyTests(unittest.TestCase):
    def test_only_classes_meeting_dev_precision_gate_are_emitted(self):
        features = pd.DataFrame([{"cls": 2}])
        predicted = [
            [1.0, 2.0, "jaywalking"],
            [1.0, 2.0, "stop_line"],
            [3.0, 4.0, "red_light"],
        ]
        with (
            patch.object(rules, "_congestion", return_value=[]),
            patch.object(rules, "_person_road_intervals", return_value=[(1.0, 2.0)]),
            patch.object(rules, "_failure_to_yield", return_value=[]),
            patch.object(rules, "_red_and_stopline", return_value=predicted[1:]),
        ):
            events = rules.generate_events(features, pd.DataFrame(), {}, 10.0)

        self.assertEqual(events, [[3.0, 4.0, "red_light"]])


if __name__ == "__main__":
    unittest.main()
