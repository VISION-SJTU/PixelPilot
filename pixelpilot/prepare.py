# SPDX-License-Identifier: Apache-2.0
"""Validate PixelPilot annotations and export local GRPO/SFT inputs."""

import argparse
import json
from pathlib import Path
import re

from .formats import extract_tag, parse_actions, parse_boxes, parse_waypoints
from .io import image_path, read_records, validate_annotations, write_json


def validate_targets(records, multitask=False):
    for row in records:
        try:
            content = row["messages"][1]["content"]
            validators = {"action": parse_actions, "perception": parse_boxes, "answer": parse_waypoints}
            present = [tag for tag in validators if f"<{tag}>" in content]
            if not present or (not multitask and len(present) != 3):
                raise ValueError("Expected a supported SFT task or a complete integrated GRPO target")
            for tag in present:
                validators[tag](content)
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError(f"{row['id']}: invalid target: {error}") from error


def validate_legacy_grpo_targets(records):
    """Reject representations the retained historical reward cannot parse correctly."""
    for row in records:
        text = extract_tag(row["messages"][1]["content"], "answer")
        point = r"(?:\[\d+, \d+\]|<traj-?\d+\.\d+>)"
        if not re.fullmatch(rf"{point}(?:,\s*{point}){{5}}", text):
            raise ValueError(f"{row['id']}: use '[u, v]' integer pairs with comma-space, or decimal '<trajY>' tokens for GRPO")


def grpo_rows(records, image_root):
    from PIL import Image

    for row in records:
        path = image_path(image_root, row["images"][0])
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        content = row["messages"][1]["content"]
        yield {
            "id": row["id"],
            "problem": row["messages"][0]["content"],
            # Preserve the original trainer's trajectory-only ground-truth contract.
            "solution": extract_tag(content, "answer"),
            "meta_action": extract_tag(content, "action"),
            "perception": extract_tag(content, "perception"),
            "image": str(path),
            "img_height": height,
            "img_width": width,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--extra-train-json", action="append", default=[], help="Append selected reasoning SFT records; may be repeated")
    parser.add_argument("--val-json")
    parser.add_argument("--image-root", help="Directory containing annotation-relative image paths")
    parser.add_argument("--output-dir")
    parser.add_argument("--format", choices=["grpo", "sft"], default="grpo")
    parser.add_argument("--check-only", action="store_true", help="Check text/schema; also images if --image-root is set")
    args = parser.parse_args()
    splits = {"train": read_records(args.train_json)}
    if args.extra_train_json and args.format != "sft":
        parser.error("--extra-train-json is supported for multi-task SFT only")
    for extra in args.extra_train_json:
        splits["train"].extend(read_records(extra))
    if args.val_json:
        splits["val"] = read_records(args.val_json)
    ids = {}
    for split, records in splits.items():
        ids[split] = validate_annotations(records, args.image_root, allow_repeated_ids=args.format == "sft")
        validate_targets(records, multitask=args.format == "sft")
        if args.format == "grpo":
            validate_legacy_grpo_targets(records)
    if "val" in ids and ids["train"] & ids["val"]:
        raise ValueError("Train and validation sample ids overlap")
    counts = {split: len(rows) for split, rows in splits.items()}
    if args.check_only:
        print(json.dumps({"valid": True, "samples": counts, "images_checked": bool(args.image_root)}))
        return
    if not args.output_dir or not args.image_root:
        parser.error("Export requires --output-dir and --image-root")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.format == "grpo":
        from datasets import Dataset, DatasetDict

        # Validate every image before creating the output directory.
        rows = {name: list(grpo_rows(items, args.image_root)) for name, items in splits.items()}
        DatasetDict({name: Dataset.from_list(items) for name, items in rows.items()}).save_to_disk(str(output))
    else:
        info = {}
        for split, records in splits.items():
            converted = []
            for row in records:
                converted.append({
                    "id": row["id"],
                    "messages": row["messages"],
                    "images": [str(image_path(args.image_root, row["images"][0]))],
                })
            write_json(output / f"{split}.json", converted)
            info[f"pixelpilot_{split}"] = {
                "file_name": f"{split}.json", "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
            }
        write_json(output / "dataset_info.json", info)
    write_json(output / "preparation_summary.json", {
        "samples": counts, "unique_frames": {split: len(tokens) for split, tokens in ids.items()}, "format": args.format,
    })
    print(json.dumps({"output": str(output), "samples": counts}))


if __name__ == "__main__":
    main()
