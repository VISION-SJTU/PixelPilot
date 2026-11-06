# SPDX-License-Identifier: Apache-2.0
"""Deterministic text-output diagnostics; these are not the paper's full benchmark."""

import argparse
import json

from .formats import CLASSES, PATH_ACTIONS, SPEED_ACTIONS, parse_actions, parse_boxes
from .io import read_records, write_json


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}


def action_metrics(records):
    counts = {c: [0, 0, 0] for c in PATH_ACTIONS + SPEED_ACTIONS}
    correct = {"path_accuracy": 0, "speed_accuracy": 0, "joint_accuracy": 0}
    malformed = 0
    for row in records:
        truth = parse_actions(row["label"])
        try:
            pred = parse_actions(row["predict"])
        except (ValueError, TypeError):
            pred = set()
            malformed += 1
        for c in counts:
            counts[c][0] += int(c in pred and c in truth)
            counts[c][1] += int(c in pred and c not in truth)
            counts[c][2] += int(c not in pred and c in truth)
        correct["path_accuracy"] += int(pred & set(PATH_ACTIONS) == truth & set(PATH_ACTIONS))
        correct["speed_accuracy"] += int(pred & set(SPEED_ACTIONS) == truth & set(SPEED_ACTIONS))
        correct["joint_accuracy"] += int(pred == truth)
    metrics = {c: prf(*v) for c, v in counts.items()}
    active = [v["f1"] for v in metrics.values() if v["support"]]
    return {"samples": len(records), "malformed_predictions": malformed, "per_class": metrics,
            "macro_f1_supported_classes": sum(active) / len(active) if active else 0.0,
            "micro": prf(*(sum(v[i] for v in counts.values()) for i in range(3))),
            **{key: value / len(records) for key, value in correct.items()}}


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def box_metrics(records, threshold):
    from scipy.optimize import linear_sum_assignment

    counts = {c: [0, 0, 0] for c in CLASSES}
    malformed = 0
    for row in records:
        truth = parse_boxes(row["label"])
        try:
            pred = parse_boxes(row["predict"])
        except (ValueError, SyntaxError, TypeError):
            pred = []
            malformed += 1
        for c in CLASSES:
            p = [b["bbox_2d"] for b in pred if b["label"] == c]
            g = [b["bbox_2d"] for b in truth if b["label"] == c]
            tp = 0
            if p and g:
                overlaps = [[iou(a, b) for b in g] for a in p]
                # Maximize valid match count first, then IoU for deterministic tie resolution.
                cost = [[-(int(v >= threshold) * (min(len(p), len(g)) + 1) + v) for v in row] for row in overlaps]
                pi, gi = linear_sum_assignment(cost)
                tp = sum(overlaps[a][b] >= threshold for a, b in zip(pi, gi))
            counts[c][0] += int(tp)
            counts[c][1] += len(p) - int(tp)
            counts[c][2] += len(g) - int(tp)
    return {"samples": len(records), "malformed_predictions": malformed, "iou_threshold": threshold,
            "metric": "fixed-threshold class-aware Hungarian precision/recall/F1 (not mAP)",
            "per_class": {c: prf(*v) for c, v in counts.items()},
            "micro": prf(*(sum(v[i] for v in counts.values()) for i in range(3)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSON/JSONL with predict and label fields")
    parser.add_argument("--task", choices=["action", "boxes"], required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")
    records = read_records(args.predictions)
    if any(not isinstance(r.get("predict"), str) or not isinstance(r.get("label"), str) for r in records):
        raise ValueError("Every prediction requires text predict and label fields")
    ids = [r["id"] for r in records if "id" in r]
    if ids and (len(ids) != len(records) or len(set(ids)) != len(ids)):
        raise ValueError("Prediction ids must be unique and present on every row when used")
    metrics = action_metrics(records) if args.task == "action" else box_metrics(records, args.iou_threshold)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
