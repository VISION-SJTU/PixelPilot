# SPDX-License-Identifier: Apache-2.0
"""Create the front/rotated-rear composite used by the supplied research scripts."""

import argparse
from pathlib import Path

from .io import read_records, validate_annotations


def compose(front, rear, width=784, height=448):
    from PIL import Image

    front = front.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    rear = rear.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    rear = rear.transpose(Image.Transpose.ROTATE_180)
    result = Image.new("RGB", (width, height * 2))
    result.paste(front, (0, 0))
    result.paste(rear, (0, height))
    return result


def compose_native(front, rear):
    """Preserve the recovered native nuScenes collage, resized later by the VLM."""
    from PIL import Image

    if front.size != (1600, 900) or rear.size != (1600, 900):
        raise ValueError("Recovered native collage expects two 1600x900 nuScenes camera frames")
    result = Image.new("RGB", (1600, 1800))
    result.paste(front.convert("RGB"), (0, 0))
    result.paste(rear.convert("RGB").transpose(Image.Transpose.ROTATE_180), (0, 900))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="JSON/JSONL with nuScenes sample ids")
    parser.add_argument("--nusc-root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--native-resolution", action="store_true", help="Match recovered 1600x1800 image storage; VLM later resizes to label coordinates")
    args = parser.parse_args()
    from nuscenes.nuscenes import NuScenes
    from PIL import Image

    records = read_records(args.annotations)
    validate_annotations(records, allow_repeated_ids=True)
    records = list({row["id"]: row for row in records}.values())
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    # Check all tokens before writing a partial image directory.
    samples = [nusc.get("sample", row["id"]) for row in records]
    output.mkdir(parents=True)
    for row, sample in zip(records, samples):
        token = row["id"]
        if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError(f"Invalid nuScenes sample token: {token!r}")
        paths = [nusc.get_sample_data_path(sample["data"][camera]) for camera in ("CAM_FRONT", "CAM_BACK")]
        with Image.open(paths[0]) as front, Image.open(paths[1]) as rear:
            result = compose_native(front, rear) if args.native_resolution else compose(front, rear)
            result.save(output / f"{token}.jpg", quality=95)
    print(f"Created {len(records)} composites at {output}")


if __name__ == "__main__":
    main()
