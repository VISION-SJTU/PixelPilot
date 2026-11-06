# SPDX-License-Identifier: Apache-2.0
import unittest

from pixelpilot.evaluate import action_metrics, box_metrics
from pixelpilot.formats import parse_actions, parse_boxes, parse_waypoints
from pixelpilot.evaluate_planning import planning_metrics


class TextMetricsTests(unittest.TestCase):
    def test_action_metrics_are_repeatable_and_count_invalid_predictions(self):
        rows = [
            {"predict": "<action>FORWARD, KEEP</action>", "label": "<action>FORWARD, KEEP</action>"},
            {"predict": "malformed", "label": "<action>LEFT, STOP</action>"},
        ]
        first = action_metrics(rows)
        for _ in range(10):
            self.assertEqual(first, action_metrics(rows))
        self.assertEqual(first["joint_accuracy"], 0.5)
        self.assertEqual(first["malformed_predictions"], 1)
        self.assertEqual(first["per_class"]["LEFT"]["fn"], 1)
        self.assertEqual(first["micro"]["f1"], 2 / 3)

    def test_invalid_action_ground_truth_fails(self):
        with self.assertRaises(ValueError):
            action_metrics([{"predict": "", "label": "<action>FORWARD, UNKNOWN</action>"}])
        with self.assertRaises(ValueError):
            parse_actions("<action>FORWARD, LEFT, KEEP</action>")

    def test_legacy_perception_literals_and_duplicate_boxes(self):
        car = {"bbox_2d": [0, 0, 10, 10], "label": "car"}
        truth = f"<perception>{[car]!r}</perception>"
        pred = f"<perception>{[car, car]!r}</perception>"
        metrics = box_metrics([{"predict": pred, "label": truth}], 0.5)
        self.assertEqual(metrics["micro"]["tp"], 1)
        self.assertEqual(metrics["micro"]["fp"], 1)
        self.assertEqual(metrics["micro"]["fn"], 0)
        self.assertEqual(parse_boxes(truth), [car])

    def test_wrong_box_class_cannot_match(self):
        pred = '<perception>[{"bbox_2d":[0,0,10,10],"label":"truck"}]</perception>'
        truth = pred.replace("truck", "car")
        metrics = box_metrics([{"predict": pred, "label": truth}], 0.5)
        self.assertEqual(metrics["micro"]["tp"], 0)
        self.assertEqual(metrics["micro"]["fp"], 1)
        self.assertEqual(metrics["micro"]["fn"], 1)

    def test_waypoint_order_preserves_mixed_tokens(self):
        body = "<traj0.0>, [1, 2], <traj-1.25>, [3.5, 4], [5, 6], [7, 8]"
        result = parse_waypoints(f"<answer>{body}</answer>")
        self.assertEqual(result[:3], [{"traj": 0.0}, [1.0, 2.0], {"traj": -1.25}])
        for value in ("[1,2]" * 5, body + "[9,10]", body + " hidden text"):
            with self.assertRaises(ValueError):
                parse_waypoints(value, tagged=False)

    def test_duplicate_tags_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_actions("<action>FORWARD, KEEP</action> <action>LEFT, STOP</action>")


class PlanningTests(unittest.TestCase):
    def records(self):
        pred = [{"id": "a", "frame": "LIDAR_TOP", "trajectory": [[3, 4]] * 6}]
        gt = [{"id": "a", "frame": "LIDAR_TOP", "trajectory": [[0, 0]] * 6, "valid": [True] * 4 + [False] * 2}]
        return pred, gt

    def test_masked_horizons_and_known_distance(self):
        metrics = planning_metrics(*self.records())
        self.assertEqual(metrics["horizons"]["1s"]["prefix_mean_l2"], 5.0)
        self.assertEqual(metrics["horizons"]["2s"]["endpoint_l2"], 5.0)
        self.assertEqual(metrics["horizons"]["3s"]["valid_samples"], 0)
        self.assertIsNone(metrics["horizons"]["3s"]["endpoint_l2"])

    def test_alignment_frame_and_nonfinite_errors(self):
        for kind in ("id", "frame", "nan", "mask"):
            pred, gt = self.records()
            if kind == "id":
                pred[0]["id"] = "other"
            elif kind == "frame":
                pred[0]["frame"] = "EGO"
            elif kind == "nan":
                pred[0]["trajectory"][0] = [float("nan"), 0]
            else:
                gt[0]["valid"] = [1] * 6
            with self.assertRaises(ValueError):
                planning_metrics(pred, gt)


if __name__ == "__main__":
    unittest.main()
