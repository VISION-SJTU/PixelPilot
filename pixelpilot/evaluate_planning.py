# SPDX-License-Identifier: Apache-2.0
"""Evaluate metric trajectories against an explicit, id-aligned ground truth."""

import argparse
import json
import math

from .io import read_records, write_json


def by_id(records):
    result = {}
    for row in records:
        token = row.get("id")
        if not isinstance(token, str) or token in result:
            raise ValueError("Every trajectory requires a unique string id")
        result[token] = row
    return result


def trajectory(row):
    points = row.get("trajectory")
    if not isinstance(points, list) or len(points) != 6:
        raise ValueError(f"{row['id']}: expected six metric [x, y] waypoints")
    for p in points:
        if not isinstance(p, list) or len(p) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p
        ):
            raise ValueError(f"{row['id']}: invalid trajectory coordinates")
    return points


def planning_metrics(predictions, labels):
    pred, truth = by_id(predictions), by_id(labels)
    if pred.keys() != truth.keys():
        raise ValueError(f"Sample ids differ: {len(truth.keys() - pred.keys())} missing, {len(pred.keys() - truth.keys())} extra")
    errors = []
    for token, target in truth.items():
        output = pred[token]
        if target.get("frame") not in ("LIDAR_TOP", "EGO", "PLANNING") or output.get("frame") != target["frame"]:
            raise ValueError(f"{token}: coordinate frames missing or different")
        valid = target.get("valid")
        if not isinstance(valid, list) or len(valid) != 6 or any(type(v) is not bool for v in valid):
            raise ValueError(f"{token}: ground truth must include six boolean validity flags")
        a, b = trajectory(output), trajectory(target)
        errors.append([math.hypot(x[0] - y[0], x[1] - y[1]) if flag else None for x, y, flag in zip(a, b, valid)])
    result = {"samples": len(errors), "units": "meters", "sampling_interval_s": 0.5, "horizons": {}}
    for count in (2, 4, 6):
        # Equal sample weighting, requiring complete validity through the horizon.
        rows = [row[:count] for row in errors if all(value is not None for value in row[:count])]
        result["horizons"][f"{count // 2}s"] = {
            "valid_samples": len(rows),
            "prefix_mean_l2": sum(sum(row) / count for row in rows) / len(rows) if rows else None,
            "endpoint_l2": sum(row[-1] for row in rows) / len(rows) if rows else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = planning_metrics(read_records(args.predictions), read_records(args.ground_truth))
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
