# SPDX-License-Identifier: Apache-2.0
"""JSON helpers with explicit errors and non-destructive output handling."""

import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array or JSONL file: {path}")
    if any(not isinstance(item, dict) for item in records):
        raise ValueError("Each record must be an object")
    return records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def image_path(root, relative):
    """Resolve only paths contained in the explicitly selected image directory."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Image paths in annotations must be relative to --image-root")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Image path escapes --image-root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_annotations(records, image_root=None, allow_repeated_ids=False):
    """Validate single-turn records; multi-turn files need a deliberate conversion."""
    seen = set()
    seen_records = set()
    for row in records:
        token = row.get("id")
        if not isinstance(token, str) or not token or (token in seen and not allow_repeated_ids):
            raise ValueError(f"Missing or duplicate sample id: {token!r}")
        seen.add(token)
        images, messages = row.get("images"), row.get("messages")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"{token}: expected one composite image")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{token}: expected a single user/assistant turn")
        if [m.get("role") for m in messages] != ["user", "assistant"]:
            raise ValueError(f"{token}: invalid message roles")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError(f"{token}: message content must be text")
        if messages[0]["content"].count("<image>") != 1:
            raise ValueError(f"{token}: prompt must contain exactly one <image> marker")
        signature = (token, tuple(images), messages[0]["content"], messages[1]["content"])
        if signature in seen_records:
            raise ValueError(f"{token}: exact duplicate training record")
        seen_records.add(signature)
        if image_root is not None:
            image_path(image_root, images[0])
    return seen
