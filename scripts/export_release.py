#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export only allowlisted source files, with per-file SHA-256 provenance."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "README_zh.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    "CITATION.cff", "pyproject.toml", "MANIFEST.in", ".gitignore",
)
SOURCE_DIRS = (".github", "configs", "docs", "licenses", "pixelpilot", "requirements", "scripts", "tests", "examples", "verl", "LLaMAFactory")
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", ".ruff_cache", ".pytest_cache", "build", "dist", "venv"}
EXTENSIONS = {".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".cff", ".toml", ".in"}


def release_files(root=ROOT):
    paths = [root / name for name in ROOT_FILES]
    for name in SOURCE_DIRS:
        paths.extend(path for path in (root / name).rglob("*") if path.is_file())
    selected = []
    for path in paths:
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS or part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
            continue
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise ValueError(f"Refusing symlink/outside file: {relative}")
        if relative.as_posix() not in ROOT_FILES and path.suffix not in EXTENSIONS and path.name != "LICENSE":
            raise ValueError(f"Unexpected release file type: {relative}")
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError(f"Missing or oversized release file: {relative}")
        selected.append(path)
    return sorted(set(selected))


def export(output, root=ROOT):
    files = release_files(root)
    manifest = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                contents = [(row["path"], (root / row["path"]).read_bytes()) for row in manifest]
                contents.append(("RELEASE_MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode()))
                for name, content in contents:
                    item = tarfile.TarInfo("PixelPilot/" + name)
                    item.size = len(content)
                    item.mode = 0o755 if name.endswith(".sh") else 0o644
                    item.mtime = 0
                    archive.addfile(item, io.BytesIO(content))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Exported {export(args.output)} source files to {args.output}")


if __name__ == "__main__":
    main()
