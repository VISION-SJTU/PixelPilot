# SPDX-License-Identifier: Apache-2.0
"""Validate and launch the recovered 7B SFT recipe with the bundled framework."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .io import read_records, validate_annotations
from .prepare import validate_targets


ROOT = Path(__file__).resolve().parents[1]


def effective_config(config_path, model=None, dataset_dir=None, output_dir=None):
    import yaml

    with Path(config_path).open() as stream:
        config = yaml.safe_load(stream)
    if model:
        config["model_name_or_path"] = model
    for key, value in (("dataset_dir", dataset_dir), ("output_dir", output_dir)):
        if value:
            config[key] = value
        config[key] = str(Path(config[key]).resolve())
    config["deepspeed"] = str(Path(config["deepspeed"]).resolve())
    if config.get("stage") != "sft" or config.get("finetuning_type") != "full":
        raise ValueError("The recovered release recipe expects full SFT")
    if not Path(config["deepspeed"]).is_file():
        raise FileNotFoundError(config["deepspeed"])
    return config


def check_data(config):
    directory = Path(config["dataset_dir"])
    registration = json.loads((directory / "dataset_info.json").read_text())
    ids = set()
    total = 0
    for name in config["dataset"].split(","):
        name = name.strip()
        if name not in registration:
            raise ValueError(f"Unregistered SFT dataset: {name}")
        entry = registration[name]
        if entry.get("formatting") != "sharegpt":
            raise ValueError(f"{name}: expected ShareGPT registration")
        path = (directory / entry["file_name"]).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("Dataset filename escapes dataset directory")
        rows = read_records(path)
        # The release converter resolves image paths; validate them explicitly here.
        for row in rows:
            for image in row.get("images", []):
                if not Path(image).is_absolute() or not Path(image).is_file():
                    raise FileNotFoundError(f"Re-run SFT preparation on this machine; missing image: {image}")
        ids.update(validate_annotations(rows, allow_repeated_ids=True))
        validate_targets(rows, multitask=True)
        total += len(rows)
    return {"training_records": total, "unique_frames": len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_7b.yaml")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--dataset-dir", default=os.environ.get("SFT_DATA"))
    parser.add_argument("--output-dir", default=os.environ.get("SFT_OUTPUT"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NPROC_PER_NODE", "8")))
    parser.add_argument("--dry-run", action="store_true", help="Show the resolved configuration without training or network access")
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    config = effective_config(args.config, args.model, args.dataset_dir, args.output_dir)
    vendor = ROOT / "LLaMAFactory/src"
    if not (vendor / "llamafactory/train/sft/workflow.py").is_file():
        raise FileNotFoundError("Bundled SFT source missing; run from the complete source release")
    command = [sys.executable, "-m", "llamafactory.cli", "train", "<resolved-config.yaml>"]
    if args.dry_run:
        print(json.dumps({"framework": "bundled LLaMA-Factory 0.9.4.dev0", "num_gpus": args.num_gpus,
                          "command": command, "config": config}, indent=2))
        return
    summary = check_data(config)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing SFT output: {output}")
    missing = [name for name in ("torch", "transformers", "accelerate", "deepspeed", "trl") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Install requirements/sft.txt in the SFT environment; missing {missing}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(f"Need {args.num_gpus} visible CUDA GPUs; found {torch.cuda.device_count()}")
    import yaml
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vendor) + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_TORCHRUN"] = "1"
    env["USE_V1"] = "0"
    env["NPROC_PER_NODE"] = str(args.num_gpus)
    env.setdefault("NNODES", "1")
    if env["NNODES"] != "1":
        raise ValueError("This portable launcher supports one node; use the framework launcher for multiple nodes")
    print(json.dumps(summary))
    # The framework saves training arguments in the checkpoint; only the transient launch file is removed.
    with tempfile.TemporaryDirectory(prefix="pixelpilot-sft-") as folder:
        path = Path(folder) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command[-1] = str(path)
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
