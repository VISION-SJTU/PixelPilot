# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import unittest

from pixelpilot.benchmark.evaluate import align_predictions
from pixelpilot.reasoning import prompt_for
from pixelpilot.select_reasoning import attach_reasoning, select


class ReasoningBenchmarkTests(unittest.TestCase):
    def inputs(self):
        label = "<action>FORWARD, KEEP</action>"
        before = [{"id": "a", "label": label, "predict": "<action>LEFT, STOP</action>"}]
        after = [{"id": "a", "label": label, "predict": label}]
        a = [{"id": "a", "frame": "PLANNING", "trajectory": [[2, 0]] * 6}]
        b = [{"id": "a", "frame": "PLANNING", "trajectory": [[1, 0]] * 6}]
        gt = [{"id": "a", "frame": "PLANNING", "trajectory": [[0, 0]] * 6, "valid": [True] * 6}]
        return before, after, a, b, gt, {"a": {"reasoning": "A vehicle is ahead."}}

    def test_selection_requires_action_correction_and_metric_improvement(self):
        args = self.inputs()
        self.assertEqual(select(*args), (["a"], 0))
        args[0][0]["predict"] = args[1][0]["predict"]
        self.assertEqual(select(*args), ([], 0))
        args = self.inputs()
        args[3][0]["trajectory"] = [[3, 0]] * 6
        self.assertEqual(select(*args), ([], 0))

    def test_selection_rejects_id_frame_and_mask_mismatch(self):
        args = self.inputs()
        args[3][0]["frame"] = "LIDAR_TOP"
        with self.assertRaises(ValueError):
            select(*args)
        args = self.inputs()
        args[3][0]["id"] = "other"
        with self.assertRaises(ValueError):
            select(*args)

    def test_reasoning_attachment_preserves_original_labels_and_source_record(self):
        path = Path(__file__).resolve().parents[1] / "examples/annotations.json"
        row = json.loads(path.read_text())[0]
        row["messages"][1]["content"] = row["messages"][1]["content"].replace("<think>The car ahead is in the planned path.</think>", "")
        original = copy.deepcopy(row)
        result = attach_reasoning(row, "A car occupies the lane.")
        self.assertEqual(row, original)
        self.assertIn("</perception><think>A car occupies the lane.</think><action>", result["messages"][1]["content"])
        self.assertIn("Current Speed:", prompt_for(row))

    def test_benchmark_requires_frozen_split_and_planning_frame(self):
        predictions = self.inputs()[2]
        self.assertEqual(set(align_predictions(predictions, [{"id": "a"}])), {"a"})
        with self.assertRaises(ValueError):
            align_predictions(predictions, [{"id": "b"}])
        predictions[0]["frame"] = "LIDAR_TOP"
        with self.assertRaises(ValueError):
            align_predictions(predictions, [{"id": "a"}])


if __name__ == "__main__":
    unittest.main()
