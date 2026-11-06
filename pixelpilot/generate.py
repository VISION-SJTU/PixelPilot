# SPDX-License-Identifier: Apache-2.0
"""Generate the recovered perception/action/trajectory SFT mix from nuScenes infos."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from .formats import CLASSES, NUMBER
from .io import read_records, validate_annotations, write_json


def scale_box(box):
    return [int(float(v) * factor + 0.5) for v, factor in zip(box, (784 / 1600, 448 / 900) * 2)]


def info_labels(info, infos):
    """Preserve the selected source script's labels, thresholds and box filtering."""
    import numpy as np

    required = {"gt_planning", "can_bus", "gt_planning_command", "next", "bboxes2d", "labels2d"}
    missing = required - info.keys()
    if missing:
        raise ValueError(f"{info['token']}: missing info fields {sorted(missing)}")
    trajectory = np.asarray(info["gt_planning"])[0, :, :2].round(2)
    if trajectory.shape != (6, 2) or not np.isfinite(trajectory).all():
        raise ValueError(f"{info['token']}: expected six finite gt_planning waypoints")
    velocity = round(float(info["can_bus"][-3]), 2)
    if abs(trajectory[-1, 0] - trajectory[0, 0]) < 0.1 and abs(trajectory[-1, 1] - trajectory[0, 1]) < 0.1:
        speed = "STOP"
    elif not info["next"]:
        speed = "KEEP"
    else:
        successor = infos.get(info["next"])
        next_velocity = round(float(successor["can_bus"][-3]), 2) if successor is not None else 0.0
        delta = next_velocity - velocity
        speed = "ACCELERATE" if delta > 0.1 else "DECELERATE" if delta < -0.1 else "KEEP"
    command = int(info["gt_planning_command"])
    if command not in (0, 1, 2):
        raise ValueError(f"Unknown command: {command}")
    path = {0: "LEFT", 1: "RIGHT", 2: "FORWARD"}[command]
    boxes, labels = info["bboxes2d"][0], info["labels2d"][0]
    if len(boxes) != len(labels):
        raise ValueError(f"{info['token']}: box and class counts differ")
    perception = []
    for box, label in zip(boxes, labels):
        box = scale_box(box)
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            continue
        label = int(label)
        if not 0 <= label < len(CLASSES):
            raise ValueError(f"Unknown class index: {label}")
        perception.append({"bbox_2d": box, "label": CLASSES[label]})
    return {
        "gt_traj": trajectory, "velocity": velocity, "acc_x": round(float(info["can_bus"][4]), 2),
        "acc_y": round(float(info["can_bus"][5]), 2), "meta_action": f"{path}, {speed}",
        "percep": perception[:30],
    }


def coordinate_text(points, metric, history=False):
    """Preserve historical near-ego tokens and rear-history coordinate convention."""
    import numpy as np

    result = []
    for i, point in enumerate(np.rint(points).astype(int).tolist()):
        longitudinal = float(metric[i, 0])
        near = longitudinal >= -1.0 if history else longitudinal <= 1.0
        if near:
            value = float(np.round(longitudinal, 1))
            value = 0.0 if value == 0 else value
            text = f"<traj{value:.1f}>"
        else:
            text = str(point)
        if history:
            text = f"(t-{(len(points) - i - 1) * 0.5}s) {text}"
        result.append(text)
    return ", ".join(result)


def frame_records(raw, labels, projector, rng, image_directory, include_mix=True):
    import numpy as np

    token = raw["id"]
    history = np.array(re.findall(rf"\[({NUMBER}),\s*({NUMBER})\]", raw["messages"][0]["content"]), dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or not len(history):
        raise ValueError(f"{token}: no metric history found in the raw prompt")
    future = labels["gt_traj"].copy()
    # Source preprocessing zeros small values in place, so use independent arrays.
    history_3d = projector.preprocess_trajectory(history.copy())
    future_3d = projector.preprocess_trajectory(future.copy())
    future_pixels, _ = projector.project_trajectory_to_camera(future_3d, token, "CAM_FRONT")
    history_pixels, _ = projector.project_trajectory_to_camera(history_3d, token, "CAM_BACK")
    future_pixels = future_pixels * np.array([784 / 1600, 448 / 900])
    history_pixels = np.array([784.0, 896.0]) - history_pixels * np.array([784 / 1600, 448 / 900])
    hist_text = coordinate_text(history_pixels, history, history=True)
    answer = f"<answer>{coordinate_text(future_pixels, future)}</answer>"
    perception = f"<perception>{labels['percep']}</perception> "
    action = f"<action>{labels['meta_action']}</action> "
    base = (
        "You are an autonomous driving agent. You have access to the vehicle's front view <image>, "
        "and your task is to drive safely. "
        f"Provided are the previous ego vehicle status recorded over the last {(len(history) - 1) * 0.5} seconds "
        "(at 0.5-second intervals). This includes the pixel coordinates of the ego vehicle in the corresponding image. "
        f"The data is:{hist_text}. Speed: {labels['velocity']} m/s. "
        f"Forward Acceleration: {labels['acc_x']} m/s^2. Leftward acceleration: {labels['acc_y']} m/s^2. "
    )
    tasks = [
        ("perception", f"Output the bbox coordinates for {list(CLASSES)} using JSON format in <perception> </perception> tags.", perception),
        ("action", "Output the planning meta action combining a PATH (FORWARD, LEFT, RIGHT) and a SPEED (KEEP, ACCELERATE, DECELERATE, STOP) in <action> </action> tags.", action),
        ("trajectory", "Output the 3-second pixel coordinates of the vehicle in <answer> </answer> tags.", answer),
        ("integrated", f"Output: (1) The bbox coordinates for {list(CLASSES)} using JSON format in <perception> </perception> tags, "
         "(2) The planning meta action combining a PATH (FORWARD, LEFT, RIGHT) and a SPEED (KEEP, ACCELERATE, DECELERATE, STOP) "
         "in <action> </action> tags, and (3) The 3-second pixel coordinates of the vehicle in <answer> </answer> tags. ",
         perception + action + answer),
    ]
    rows = []
    for name, prompt, target in tasks:
        if name != "integrated" and not include_mix:
            continue
        if name in ("perception", "action") and rng.choice([True, False]):
            continue
        if name == "perception" and not labels["percep"]:
            continue
        rows.append({"id": token, "task": name, "images": [f"{image_directory}/{token}.jpg"], "messages": [
            {"role": "user", "content": base + prompt + "\n"}, {"role": "assistant", "content": target},
        ]})
    return rows


def generate_split(raw_records, infos, projector, seed, image_directory, limit=None):
    validate_annotations(raw_records)
    rng = random.Random(seed)
    selected = raw_records[:limit] if limit else raw_records
    mixed, integrated, metric = [], [], []
    for raw in selected:
        token = raw["id"]
        if token not in infos:
            raise ValueError(f"{token}: raw JSON token absent from supplied infos")
        info = infos[token]
        labels = info_labels(info, infos)
        rows = frame_records(raw, labels, projector, rng, image_directory)
        mixed.extend(rows)
        integrated.extend(row for row in rows if row["task"] == "integrated")
        mask = info.get("gt_planning_mask")
        if mask is None:
            raise ValueError(f"{token}: missing gt_planning_mask; cannot export metric GT")
        import numpy as np
        valid = np.asarray(mask)[0]
        if valid.shape[0] != 6:
            raise ValueError(f"{token}: invalid gt_planning_mask shape")
        valid = np.asarray(valid).reshape(6, -1).all(axis=1).tolist()
        metric.append({"id": token, "frame": "PLANNING", "trajectory": labels["gt_traj"].tolist(), "valid": valid})
    rng.shuffle(mixed)
    return mixed, integrated, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, help="Original single-turn metric nuScenes JSON")
    parser.add_argument("--train-infos", required=True)
    parser.add_argument("--val-json")
    parser.add_argument("--val-infos")
    parser.add_argument("--nusc-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if bool(args.val_json) != bool(args.val_infos):
        parser.error("Provide --val-json and --val-infos together")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    from nuscenes.nuscenes import NuScenes
    from .data.infos import load_infos
    from .data.projection import TrajectoryProjector

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    projector = TrajectoryProjector(nusc)
    sources = [("train", args.train_json, args.train_infos)]
    if args.val_json:
        sources.append(("val", args.val_json, args.val_infos))
    summary = {"recipe": "0.5 perception / 0.5 action / 1 trajectory / 1 integrated", "seed": args.seed, "splits": {}}
    seen = set()
    for index, (split, raw_path, info_path) in enumerate(sources):
        raw = read_records(raw_path)
        ids = validate_annotations(raw)
        if seen & ids:
            raise ValueError("Train and validation frames overlap")
        seen.update(ids)
        infos = load_infos(info_path)
        mix, integrated, metric = generate_split(raw, infos, projector, args.seed + index, f"nusc_fb_{split}", args.limit)
        write_json(output / f"{split}_sft.json", mix)
        write_json(output / f"{split}_integrated.json", integrated)
        write_json(output / f"{split}_metric_gt.json", metric)
        summary["splits"][split] = {"frames": len(integrated), "tasks": dict(Counter(row["task"] for row in mix))}
        del infos
    write_json(output / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
