# SPDX-License-Identifier: Apache-2.0
"""Select candidates that correct action prediction and reduce metric trajectory L2."""

import argparse
import copy
import json
import math

from .evaluate_planning import by_id, trajectory
from .formats import extract_tag, parse_actions
from .io import read_records, write_json
from .reasoning import read_candidates


def select(baseline_actions, conditioned_actions, baseline_trajectories, conditioned_trajectories, ground_truth, candidates):
    datasets = [by_id(rows) for rows in (baseline_actions, conditioned_actions, baseline_trajectories, conditioned_trajectories, ground_truth)]
    ids = datasets[0].keys()
    if any(rows.keys() != ids for rows in datasets[1:]) or set(candidates) != set(ids):
        raise ValueError("Candidate, action, trajectory and GT files must contain the same unique ids")
    selected = []
    incomplete = 0
    for token in ids:
        base_action, cond_action, base_traj, cond_traj, truth = [rows[token] for rows in datasets]
        if parse_actions(base_action["label"]) != parse_actions(cond_action["label"]):
            raise ValueError(f"{token}: action ground truth differs between runs")
        expected = parse_actions(base_action["label"])
        try:
            before = parse_actions(base_action["predict"])
        except ValueError:
            before = set()
        try:
            after = parse_actions(cond_action["predict"])
        except ValueError:
            after = set()
        if truth.get("frame") not in ("PLANNING", "LIDAR_TOP", "EGO") or any(row.get("frame") != truth["frame"] for row in (base_traj, cond_traj)):
            raise ValueError(f"{token}: coordinate frames differ")
        valid = truth.get("valid")
        if not isinstance(valid, list) or len(valid) != 6 or any(type(v) is not bool for v in valid):
            raise ValueError(f"{token}: invalid ground-truth mask")
        gt, a, b = trajectory(truth), trajectory(base_traj), trajectory(cond_traj)
        if not all(valid):
            incomplete += 1
            continue
        def l2(points):
            return sum(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(points, gt)) / 6
        if before != expected and after == expected and l2(b) < l2(a):
            selected.append(token)
    return selected, incomplete


def attach_reasoning(row, reason):
    result = copy.deepcopy(row)
    content = row["messages"][1]["content"]
    if "<think>" in content:
        raise ValueError("Refusing to replace existing target reasoning")
    result["messages"][1]["content"] = (
        f"<perception>{extract_tag(content, 'perception')}</perception><think>{reason}</think>"
        f"<action>{extract_tag(content, 'action')}</action><answer>{extract_tag(content, 'answer')}</answer>"
    )
    result["messages"][0]["content"] += "\nInclude visual reasoning grounded in bounding boxes in <think> </think> between perception and action.\n"
    result["task"] = "reasoning_integrated"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-actions", "conditioned-actions", "baseline-trajectories", "conditioned-trajectories", "ground-truth", "candidates", "annotations", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    candidates = read_candidates(args.candidates)
    selected, incomplete = select(
        read_records(args.baseline_actions), read_records(args.conditioned_actions), read_records(args.baseline_trajectories),
        read_records(args.conditioned_trajectories), read_records(args.ground_truth), candidates,
    )
    annotations = by_id(read_records(args.annotations))
    if set(candidates) - annotations.keys():
        raise ValueError("Candidates absent from original annotations")
    rows = [attach_reasoning(annotations[token], candidates[token]["reasoning"]) for token in selected]
    # An empty selection is meaningful; retain it without inventing training data.
    write_json(args.output, rows)
    print(json.dumps({"candidates": len(candidates), "selected": len(rows), "excluded_incomplete_gt": incomplete}))


if __name__ == "__main__":
    main()
