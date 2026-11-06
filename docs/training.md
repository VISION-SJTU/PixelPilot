# Training

Run commands from the repository root. Follow [installation](installation.md)
to keep SFT and GRPO in separate environments, and generate the annotations and
images described in [data preparation](data.md).

## Multi-task SFT

Prepare the mixed training tasks and the independent integrated validation set:

```bash
python -m pixelpilot.prepare --format sft \
  --train-json data/annotations/train_sft.json \
  --val-json data/annotations/val_integrated.json \
  --image-root data --output-dir data/nuscenes_sft

bash scripts/train_sft.sh --dry-run
bash scripts/train_sft.sh --num-gpus 8
```

Preparation writes ShareGPT records with absolute image paths and a
`dataset_info.json` registration. Re-run it after moving the data to another
machine. Different SFT tasks may share a frame id; exact duplicate records and
train/validation overlap are rejected. GRPO instead requires unique frames.

The launcher validates registration, images and targets before invoking the
bundled [LLaMA-Factory](../LLaMAFactory). A launch preview resolves configuration
without loading a model or requiring the dataset.

| Parameter | Default |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Fine-tuning | Full SFT, ZeRO-3, BF16 |
| GPUs | 8 on one node |
| Epochs / learning rate | 2 / `5e-6` |
| Batch size per device / gradient accumulation | 2 / 8 |
| Image pixel limit / token cutoff | 720000 / 8192 |
| Internal validation | `val_size=0.01`, `eval_strategy=no` |

These defaults come from a supplied experiment. They are not a certified match
to the paper's final checkpoint. The external validation set remains separate;
the inherited internal split is not asserted to be scene-disjoint.

Override locations explicitly when needed:

```bash
bash scripts/train_sft.sh --num-gpus 8 \
  --model pretrained_models/Qwen2.5-VL-7B-Instruct \
  --dataset-dir data/nuscenes_sft \
  --output-dir checkpoints/pixelpilot-7b-sft
```

Use `--config` for a custom recipe. The portable default is
[`configs/sft_7b.yaml`](../configs/sft_7b.yaml); the three original variants remain
under [`configs/historical/`](../configs/historical) for reference. The launcher
refuses a non-empty output directory.

## Selected visual reasoning

The [reasoning workflow](reasoning.md) covers teacher candidates, paired student
inference, action correction and metric-L2 improvement selection. For a non-empty
selected set, append it to the multi-task corpus and continue SFT:

```bash
python -m pixelpilot.prepare --format sft \
  --train-json data/annotations/train_sft.json \
  --extra-train-json data/selected_reasoning_sft.json \
  --val-json data/annotations/val_integrated.json \
  --image-root data --output-dir data/nuscenes_sft_with_reasoning

bash scripts/train_sft.sh --model checkpoints/pixelpilot-7b-sft \
  --dataset-dir data/nuscenes_sft_with_reasoning \
  --output-dir checkpoints/pixelpilot-7b-sft-with-reasoning
```

Mine cases only from training data. The connecting conditioning and selection
code is a release implementation; its historical counterpart and final paper
mixture have not been recovered. Teacher API configuration is documented in
[reasoning data](reasoning.md).

## GRPO

Switch to the GRPO environment and convert unique integrated tasks:

```bash
python -m pixelpilot.prepare \
  --train-json data/annotations/train_integrated.json \
  --image-root data --output-dir data/nuscenes_grpo

export MODEL_PATH="$PWD/checkpoints/pixelpilot-7b-sft"
export NUSCENES_ROOT="$PWD/data/nuscenes"
bash scripts/train_grpo.sh --dry-run
bash scripts/train_grpo.sh
```

Set `MODEL_PATH` to the selected-reasoning checkpoint if using that stage. The
launcher runs the bundled `verl.trainer.main` with
[`configs/grpo_7b.yaml`](../configs/grpo_7b.yaml). Defaults are four GPUs, eight
completions per prompt, global batch size 16 and actor learning rate `1e-6`.
Console-only logging is enabled.

Use `TRAIN_DATA`, `CONFIG_PATH` and `CUDA_VISIBLE_DEVICES` for locations and GPU
visibility; additional arguments are passed as configuration overrides:

```bash
TRAIN_DATA=data/nuscenes_grpo bash scripts/train_grpo.sh \
  trainer.experiment_name=nuscenes_7b_grpo \
  trainer.save_checkpoint_path=checkpoints/nuscenes_7b_grpo
```

The retained historical reward differs from the paper's full reward, including
inactive perception/action contributions. Read [reproduction status](reproduction.md)
before interpreting results. Local CPU checks do not validate distributed
training or target-server memory requirements.

## Checkpoints

Full-SFT checkpoints are directly usable for inference. GRPO FSDP actor shards
must first be merged into a Hugging Face checkpoint; see [inference](inference.md).
Model weights are not bundled, and no verified release download is supplied.
