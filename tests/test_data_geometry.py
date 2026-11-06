# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pixelpilot.collage import compose
from pixelpilot.geometry import lift_pixels
from pixelpilot.io import image_path, validate_annotations
from pixelpilot.lift import inverse_projection, unmap_pixel
from pixelpilot.prepare import grpo_rows, validate_legacy_grpo_targets, validate_targets

ROOT = Path(__file__).resolve().parents[1]


class DataGeometryTests(unittest.TestCase):
    def test_rear_rotation_and_dimensions(self):
        front = Image.new("RGB", (2, 2), "red")
        rear = Image.new("RGB", (2, 2), "blue")
        rear.putpixel((0, 0), (0, 255, 0))
        result = compose(front, rear, width=2, height=2)
        self.assertEqual(result.size, (2, 4))
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(result.getpixel((1, 3)), (0, 255, 0))
        self.assertEqual(unmap_pixel([0, 448]), ("CAM_BACK", [783, 447]))
        self.assertEqual(unmap_pixel([783, 895]), ("CAM_BACK", [0, 0]))

    def test_ground_intersection_and_invalid_rays(self):
        matrix = np.diag([1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(lift_pixels([[2, 3]], matrix, -2), [[4, 6, -2]])
        with self.assertRaises(ValueError):
            lift_pixels([[2, 3]], np.eye(4), -1)
        with self.assertRaises(ValueError):
            lift_pixels([[float("inf"), 3]], matrix, -1)

    def test_path_escape_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                image_path(folder, "../outside.jpg")
            with self.assertRaises(ValueError):
                image_path(folder, "/absolute/image.jpg")
        records = json.loads((ROOT / "examples/annotations.json").read_text())
        with self.assertRaises(ValueError):
            validate_annotations(records * 2)

    def test_export_uses_actual_image_dimensions(self):
        records = json.loads((ROOT / "examples/annotations.json").read_text())
        validate_targets(records)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "synthetic/example.jpg"
            image.parent.mkdir()
            Image.new("RGB", (784, 896)).save(image)
            validate_annotations(records, folder)
            rows = list(grpo_rows(records, folder))
        self.assertEqual((rows[0]["img_width"], rows[0]["img_height"]), (784, 896))
        self.assertTrue(rows[0]["solution"].startswith("[392, 400]"))
        self.assertNotIn("<perception>", rows[0]["solution"])

    def test_historical_grpo_parser_mismatch_is_rejected(self):
        records = json.loads((ROOT / "examples/annotations.json").read_text())
        validate_legacy_grpo_targets(records)
        content = records[0]["messages"][1]["content"]
        for replacement in ("[392.0, 400]", "[392,400]", "[-392, 400]"):
            records[0]["messages"][1]["content"] = content.replace("[392, 400]", replacement)
            with self.assertRaises(ValueError):
                validate_legacy_grpo_targets(records)

    def test_historical_reward_import_is_lazy_and_formula_retained(self):
        path = ROOT / "verl/utils/reward_score/nuscenes_reward.py"
        spec = importlib.util.spec_from_file_location("historical_nuscenes_reward", path)
        reward = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reward)
        self.assertEqual(reward.get_nuscenes.cache_info().currsize, 0)
        body = ", ".join(["[392, 300]"] * 6)
        text = f"<perception>[]</perception><action>FORWARD, KEEP</action><answer>{body}</answer>"
        with patch.object(reward, "get_nuscenes", return_value=object()), patch.object(
            reward, "obtain_sensor", return_value=np.diag([1.0, 1.0, -1.0, 1.0])
        ):
            self.assertEqual(reward.nuscenes_compute_score(text, body, "synthetic"), 5.0)
        # Document, rather than silently change, the historical think-tag mismatch.
        self.assertEqual(reward.nuscenes_format_reward(text.replace("<action>", "<think>x</think><action>")), 0.0)

    def test_projection_combines_asynchronous_sensor_poses(self):
        # Fake nuScenes tables; keep this test independent of nuScenes downloads.
        tables = {
            ("sample_data", "camera"): {"calibrated_sensor_token": "cam_cal", "ego_pose_token": "cam_pose", "width": 1600, "height": 900},
            ("sample_data", "lidar"): {"calibrated_sensor_token": "lidar_cal", "ego_pose_token": "lidar_pose"},
            ("calibrated_sensor", "cam_cal"): {"rotation": [1, 0, 0, 0], "translation": [1, 0, 0], "camera_intrinsic": np.eye(3).tolist()},
            ("calibrated_sensor", "lidar_cal"): {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
            ("ego_pose", "cam_pose"): {"rotation": [1, 0, 0, 0], "translation": [2, 0, 0]},
            ("ego_pose", "lidar_pose"): {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
        }
        class Nusc:
            def get(self, table, key):
                return tables[table, key]
        class Quaternion:
            def __init__(self, value):
                self.rotation_matrix = np.eye(3)
        import types
        with patch.dict("sys.modules", {"pyquaternion": types.SimpleNamespace(Quaternion=Quaternion)}):
            matrix, width, height = inverse_projection(Nusc(), {"data": {"CAM_FRONT": "camera", "LIDAR_TOP": "lidar"}}, "CAM_FRONT")
        self.assertEqual((width, height), (1600, 900))
        self.assertEqual(matrix[0, 3], 3.0)


if __name__ == "__main__":
    unittest.main()
