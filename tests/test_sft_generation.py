# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image

from pixelpilot.collage import compose_native
from pixelpilot.generate import frame_records, generate_split, info_labels
from pixelpilot.io import validate_annotations
from pixelpilot.prepare import validate_targets
from pixelpilot.sft import check_data, effective_config
from pixelpilot.data.projection import TrajectoryProjector
from pixelpilot.lift import inverse_projection

ROOT = Path(__file__).resolve().parents[1]


class Projector:
    def preprocess_trajectory(self, points):
        return points

    def project_trajectory_to_camera(self, points, token, camera):
        return np.tile([800.0, 600.0], (len(points), 1)), np.ones(len(points), dtype=bool)


def synthetic_info():
    bus = np.zeros(18)
    bus[-3] = 2.0
    return {"token": "synthetic-example-001", "can_bus": bus, "gt_planning_command": 2, "next": "",
            "gt_planning": np.array([[[i + 1.0, 0.0] for i in range(6)]]),
            "gt_planning_mask": np.ones((1, 6, 2)),
            "bboxes2d": [np.array([[0, 0, 100, 100], [0, 0, 1, 1]])], "labels2d": [np.array([0, 1])]}


class SFTGenerationTests(unittest.TestCase):
    def raw(self):
        rows = json.loads((ROOT / "examples/annotations.json").read_text())
        rows[0]["messages"][0]["content"] = "<image> History: (t-0.0s) [0.0, 0.0]"
        return rows

    def test_multitask_generation_preserves_frame_ids_and_repeats_deterministically(self):
        raw = self.raw()
        info = synthetic_info()
        args = (raw, {info["token"]: info}, Projector(), 42, "nusc_fb_train")
        a = generate_split(*args)
        b = generate_split(*args)
        self.assertEqual(a, b)
        mixed, integrated, metric = a
        self.assertGreaterEqual(len(mixed), 2)
        self.assertEqual(len(integrated), 1)
        self.assertEqual(metric[0]["frame"], "PLANNING")
        self.assertEqual(metric[0]["valid"], [True] * 6)
        validate_annotations(mixed, allow_repeated_ids=True)
        validate_targets(mixed, multitask=True)
        with self.assertRaises(ValueError):
            validate_annotations(mixed)
        with self.assertRaises(ValueError):
            validate_annotations(mixed + mixed[:1], allow_repeated_ids=True)

    def test_recovered_action_threshold_and_box_filtering(self):
        info = synthetic_info()
        successor = copy.deepcopy(info)
        successor["can_bus"][-3] = 3.0
        info["next"] = "next"
        labels = info_labels(info, {"next": successor})
        self.assertEqual(labels["meta_action"], "FORWARD, ACCELERATE")
        self.assertEqual(labels["percep"], [{"bbox_2d": [0, 0, 49, 50], "label": "car"}])
        info["gt_planning"][:] = 0
        self.assertEqual(info_labels(info, {})["meta_action"], "FORWARD, STOP")

    def test_single_task_targets_are_accepted_only_for_sft(self):
        rows = self.raw()
        rows[0]["messages"][1]["content"] = "<action>FORWARD, KEEP</action>"
        validate_targets(rows, multitask=True)
        with self.assertRaises(ValueError):
            validate_targets(rows)

    def test_native_collage_matches_recovered_camera_storage(self):
        front = Image.new("RGB", (1600, 900), "red")
        back = Image.new("RGB", (1600, 900), "blue")
        back.putpixel((0, 0), (0, 255, 0))
        result = compose_native(front, back)
        self.assertEqual(result.size, (1600, 1800))
        self.assertEqual(result.getpixel((1599, 1799)), (0, 255, 0))

    def test_recovered_projection_matches_explicit_calibration_transform(self):
        tables = {
            ("sample", "frame"): {"data": {"CAM_FRONT": "camera", "LIDAR_TOP": "lidar"}},
            ("sample_data", "camera"): {"calibrated_sensor_token": "cam_cal", "ego_pose_token": "cam_pose", "width": 1600, "height": 900},
            ("sample_data", "lidar"): {"calibrated_sensor_token": "lidar_cal", "ego_pose_token": "lidar_pose"},
            ("calibrated_sensor", "cam_cal"): {"rotation": [1, 0, 0, 0], "translation": [1, 0, 2], "camera_intrinsic": [[10, 0, 800], [0, 10, 450], [0, 0, 1]]},
            ("calibrated_sensor", "lidar_cal"): {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
            ("ego_pose", "cam_pose"): {"rotation": [1, 0, 0, 0], "translation": [2, 0, 0]},
            ("ego_pose", "lidar_pose"): {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
        }
        class Nusc:
            def get(self, table, token):
                return tables[table, token]
        nusc = Nusc()
        points = np.array([[3, 4, 10, 1], [5, 6, 10, 1]], dtype=float)
        actual, valid = TrajectoryProjector(nusc).project_trajectory_to_camera(points, "frame")
        matrix, _, _ = inverse_projection(nusc, tables["sample", "frame"], "CAM_FRONT")
        projected = (np.linalg.inv(matrix) @ points.T).T
        expected = projected[:, :2] / projected[:, 2:3]
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(valid.all())

    def test_recovered_sft_configuration_and_registered_multitask_data(self):
        config = effective_config(ROOT / "configs/sft_7b.yaml")
        self.assertEqual(config["learning_rate"], 5e-6)
        self.assertEqual(config["num_train_epochs"], 2)
        self.assertEqual(config["gradient_accumulation_steps"], 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            Image.new("RGB", (4, 4)).save(image)
            rows = self.raw()
            rows[0]["images"] = [str(image)]
            action = copy.deepcopy(rows[0])
            action["messages"][1]["content"] = "<action>FORWARD, KEEP</action>"
            (root / "train.json").write_text(json.dumps(rows + [action]))
            (root / "dataset_info.json").write_text(json.dumps({"pixelpilot_train": {"file_name": "train.json", "formatting": "sharegpt"}}))
            config["dataset_dir"] = directory
            self.assertEqual(check_data(config), {"training_records": 2, "unique_frames": 1})


if __name__ == "__main__":
    unittest.main()
