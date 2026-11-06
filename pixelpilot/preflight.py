# SPDX-License-Identifier: Apache-2.0
"""Check configuration, local artifacts and CUDA before a GRPO job starts."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_7b.yaml")
    parser.add_argument("--model")
    parser.add_argument("--train-data")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    replacements = {}
    if args.model:
        replacements["worker"] = {"actor": {"model": {"model_path": args.model}}}
    if args.train_data:
        replacements["data"] = {"train_files": args.train_data}
    config = OmegaConf.merge(config, replacements, OmegaConf.from_dotlist(args.overrides))
    errors = []
    model = Path(config.worker.actor.model.model_path)
    dataset = Path(config.data.train_files)
    if not (model / "config.json").is_file():
        errors.append(f"Missing local SFT/merged model config: {model / 'config.json'}")
    if not (dataset / "dataset_dict.json").is_file():
        errors.append(f"Missing prepared DatasetDict: {dataset}")
    nusc_root = Path(os.environ.get("NUSCENES_ROOT", "data/nuscenes"))
    version = os.environ.get("NUSCENES_VERSION", "v1.0-trainval")
    for name in ("sample", "sample_data", "calibrated_sensor", "ego_pose"):
        if not (nusc_root / version / f"{name}.json").is_file():
            errors.append(f"Missing nuScenes calibration table: {nusc_root / version / (name + '.json')}")
    if config.worker.reward.compute_score != "nuscenes":
        errors.append("Only the supplied historical nuscenes reward is available")
    if config.trainer.val_before_train or config.trainer.val_only or config.trainer.test_freq > 0:
        errors.append("This snapshot has no validation dataloader: disable in-training validation")
    if config.data.val_files:
        errors.append("This snapshot ignores data.val_files; evaluate separately and set it to null")
    if args.training:
        pinned = {"torch": "2.6.0", "transformers": "4.51.3", "vllm": "0.8.5.post1", "flash-attn": "2.7.4.post1"}
        for package, expected in pinned.items():
            try:
                actual = importlib.metadata.version(package)
                if actual.split("+")[0] != expected:
                    errors.append(f"{package}: expected {expected}, found {actual}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing package: {package}=={expected}")
        for module in ("ray", "tensordict", "datasets", "nuscenes", "qwen_vl_utils", "xformers"):
            if importlib.util.find_spec(module) is None:
                errors.append(f"Missing module: {module}")
        if not errors:
            import torch
            from datasets import load_from_disk

            workers = config.trainer.n_gpus_per_node
            if config.trainer.nnodes != 1:
                errors.append("The supplied launch script supports one node only")
            if not torch.cuda.is_available() or torch.cuda.device_count() < workers:
                errors.append(f"Need {workers} visible CUDA GPUs; found {torch.cuda.device_count()}")
            if workers % config.worker.rollout.tensor_parallel_size:
                errors.append("GPU count must be divisible by rollout tensor parallel size")
            loaded = load_from_disk(str(dataset))
            if "train" not in loaded:
                errors.append("Prepared dataset has no train split")
            else:
                train = loaded["train"]
                required = {"id", "problem", "solution", "image", "img_height", "img_width"}
                if not required.issubset(train.column_names):
                    errors.append(f"Dataset is missing columns: {sorted(required - set(train.column_names))}")
                else:
                    if len(train) < config.data.rollout_batch_size:
                        errors.append("Train set is smaller than rollout_batch_size (drop_last is enabled)")
                    tokens = train["id"]
                    if len(set(tokens)) != len(tokens):
                        errors.append("Duplicate training sample ids")
                    with (nusc_root / version / "sample.json").open() as stream:
                        known = {row["token"] for row in json.load(stream)}
                    unknown = set(tokens) - known
                    if unknown:
                        errors.append(f"{len(unknown)} training ids are absent from nuScenes {version}")
                    missing = sum(not Path(path).is_file() for path in train["image"])
                    if missing:
                        errors.append(f"{missing} composite images are missing; re-run preparation on this machine")
    print(json.dumps({"ok": not errors, "model": str(model), "dataset": str(dataset), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
