# SPDX-License-Identifier: Apache-2.0
"""Parse the text formats found in the supplied research annotations."""

import ast
import math
import re

PATH_ACTIONS = ("FORWARD", "LEFT", "RIGHT")
SPEED_ACTIONS = ("ACCELERATE", "DECELERATE", "STOP", "KEEP")
CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
WAYPOINT = re.compile(rf"\[\s*({NUMBER})\s*,\s*({NUMBER})\s*\]|<traj({NUMBER})>")


def extract_tag(text, tag):
    """Require exactly one matching pair of tags."""
    matches = re.findall(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected one <{tag}>...</{tag}> field, found {len(matches)}")
    return matches[0].strip()


def parse_actions(text):
    words = [word.strip() for word in extract_tag(text, "action").split(",")]
    actions = set(words)
    if len(words) != 2 or len(actions & set(PATH_ACTIONS)) != 1 or len(actions & set(SPEED_ACTIONS)) != 1:
        raise ValueError("Expected one path action and one speed action")
    return actions


def parse_boxes(text):
    """Accept JSON and the Python-literal dictionaries present in legacy labels."""
    boxes = ast.literal_eval(extract_tag(text, "perception"))
    if not isinstance(boxes, list):
        raise ValueError("Perception must be a list")
    result = []
    for box in boxes:
        if not isinstance(box, dict) or box.get("label") not in CLASSES:
            raise ValueError("Unknown or missing box label")
        coords = box.get("bbox_2d")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("bbox_2d must have four coordinates")
        if any(isinstance(c, bool) or not isinstance(c, (float, int)) or not math.isfinite(c) for c in coords):
            raise ValueError("Box coordinates must be finite numbers")
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise ValueError("Box coordinates must have positive area")
        result.append({"label": box["label"], "bbox_2d": list(coords)})
    return result


def parse_waypoints(text, tagged=True):
    """Return six pixel pairs or legacy <trajY> tokens, preserving their order."""
    body = extract_tag(text, "answer") if tagged else text.strip()
    matches = list(WAYPOINT.finditer(body))
    if len(matches) != 6 or re.sub(r"[\s,]", "", WAYPOINT.sub("", body)):
        raise ValueError("Expected exactly six [x, y] waypoints or <trajY> tokens")
    result = []
    for match in matches:
        if match.group(3) is not None:
            value = float(match.group(3))
            if not math.isfinite(value):
                raise ValueError("Non-finite waypoint")
            result.append({"traj": value})
        else:
            point = [float(match.group(1)), float(match.group(2))]
            if not all(math.isfinite(x) for x in point):
                raise ValueError("Non-finite waypoint")
            result.append(point)
    return result
