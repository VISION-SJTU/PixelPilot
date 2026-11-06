# SPDX-License-Identifier: Apache-2.0
"""Load the supplied OmniDrive-style info schema without an mmcv dependency."""

import pickle
from pathlib import Path


def load_infos(path):
    """Read a locally prepared, trusted pickle and reject ambiguous sample ids."""
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("infos"), list):
        raise ValueError("Expected a pickle containing an 'infos' list")
    result = {}
    for info in data["infos"]:
        token = info.get("token")
        if not isinstance(token, str) or not token or token in result:
            raise ValueError(f"Missing or duplicate nuScenes token: {token!r}")
        result[token] = info
    return result
