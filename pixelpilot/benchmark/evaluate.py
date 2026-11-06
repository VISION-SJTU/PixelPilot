# SPDX-License-Identifier: Apache-2.0
# Metric aggregation adapted from code Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
"""Run the recovered nuScenes planning metrics with explicit sample alignment."""

import argparse
import json

from ..evaluate_planning import by_id, trajectory
from ..io import read_records, write_json


def align_predictions(predictions, split_records):
    pred, split = by_id(predictions), by_id(split_records)
    if pred.keys() != split.keys():
        raise ValueError("Prediction ids must exactly match the frozen evaluation split")
    for row in pred.values():
        if row.get("frame") != "PLANNING":
            raise ValueError("Benchmark requires PLANNING coordinates: x forward, y left")
        trajectory(row)
    return pred


def score_sample(info, points, metric):
    import numpy as np
    from pyquaternion import Quaternion
    import torch

    target = np.asarray(info["gt_planning"])[0, :6, :2]
    points = np.asarray(points, dtype=np.float64)
    boxes = np.concatenate([info["gt_boxes"], info["gt_velocity"]], -1)
    features = np.concatenate([
        info["gt_fut_traj"][:, :6].reshape(-1, 12), info["gt_fut_traj_mask"][:, :6],
        info["gt_fut_yaw"][:, :6], info["gt_fut_idx"],
    ], -1)
    occupancy = metric.get_birds_eye_view_label(boxes, features, add_rec=True)
    rotation = Quaternion(info["ego2global_rotation"]).rotation_matrix
    drivable = metric.get_drivable_area(info["ego2global_translation"], rotation, info)
    directions = np.concatenate([points[:1], np.diff(points, axis=0)], axis=0)
    yaw = np.arctan2(directions[:, 1], directions[:, 0])
    ego_features = np.concatenate([points.reshape(1, -1), np.ones((1, 6)), yaw.reshape(1, -1)], axis=-1)
    ego_boxes = np.array([[0.5 + 0.985793, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])
    ego = metric.get_ego_seg(ego_boxes, ego_features, add_rec=True)
    p = torch.from_numpy(points).unsqueeze(0)
    g = torch.from_numpy(target).unsqueeze(0)
    distance = torch.sqrt(torch.sum((p - g) ** 2, dim=-1))
    result = {}
    starts_offroad = ((np.expand_dims(drivable, 0) == 0) & (ego[:1] == 1)).any()
    for seconds in (1, 2, 3):
        count = seconds * 2
        result[f"l2_{seconds}s"] = distance[:, :count].mean().item()
        _, collisions = metric.evaluate_coll(p[:, :count], g[:, :count], torch.from_numpy(occupancy[1:]).unsqueeze(0))
        result[f"plan_obj_box_col_{seconds}s"] = collisions.max().item()
        goes_offroad = ((np.expand_dims(drivable, 0) == 0) & (ego[1:count + 1] == 1)).any()
        result[f"plan_boundary_{seconds}s"] = float(goes_offroad and not starts_offroad)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Lifted metric trajectory JSON")
    parser.add_argument("--split-json", required=True, help="Frozen integrated annotations with unique sample ids")
    parser.add_argument("--infos", required=True)
    parser.add_argument("--nusc-root", required=True, help="nuScenes root including maps/expansion")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = align_predictions(read_records(args.predictions), read_records(args.split_json))
    from ..data.infos import load_infos
    from .planning_utils import PlanningMetric
    import numpy as np

    infos = load_infos(args.infos)
    missing = predictions.keys() - infos.keys()
    if missing:
        raise ValueError(f"{len(missing)} evaluation ids are absent from infos")
    metric = PlanningMetric(args.nusc_root)
    sums = {}
    count = 0
    incomplete = 0
    for token, row in predictions.items():
        info = infos[token]
        if not np.asarray(info["gt_planning_mask"])[0].all():
            incomplete += 1
            continue
        # Fail on errors; never increment a denominator before scoring succeeds.
        values = score_sample(info, row["trajectory"], metric)
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    if not count:
        raise ValueError("No samples with complete ground-truth futures")
    result = {"protocol": "recovered nuScenes planning metric implementation", "samples": count,
              "input_samples": len(predictions), "excluded_incomplete_gt": incomplete,
              "rates_are_fractions": True, "metrics": {key: value / count for key, value in sums.items()}}
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
