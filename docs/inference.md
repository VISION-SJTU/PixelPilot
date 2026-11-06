# Inference

Run commands from the repository root in the [GRPO / inference environment](installation.md).
Provide a local Qwen2.5-VL full-SFT or merged GRPO checkpoint, the integrated
annotations and the image collages from [data preparation](data.md).

## Checkpoint preparation

A full-SFT checkpoint can be used directly. For a GRPO FSDP actor checkpoint,
merge its shards first:

```bash
python scripts/merge_checkpoint.py --local_dir checkpoints/nuscenes_7b_grpo/PATH_TO_ACTOR
```

Point inference at the actor directory's generated `huggingface/` subdirectory.
The checkpoint must contain `config.json`, model weights and processor/tokenizer
files. No model weights are bundled with the source release.

## Generate responses

```bash
python -m pixelpilot.infer --model checkpoints/pixelpilot-7b-sft \
  --annotations data/annotations/val_integrated.json --image-root data \
  --output outputs/predictions.jsonl
```

The command uses a local checkpoint, BF16, FlashAttention 2 and a CUDA GPU.
Defaults are 720000 maximum image pixels, 4096 new tokens, temperature 1.0,
top-p 0.5, top-k 20 and seed 42. Each JSONL row retains its frame id, prediction,
target and decoding configuration. Existing output files are not overwritten.

Use `--limit 10` for a small run or `--greedy` for deterministic decoding in
paired diagnostics. These options change the evaluation setup and should be
reported with results. Reasoning-conditioned inference is described in
[reasoning data](reasoning.md).

## Text diagnostics

```bash
python -m pixelpilot.evaluate --predictions outputs/predictions.jsonl --task action
python -m pixelpilot.evaluate --predictions outputs/predictions.jsonl --task boxes
```

Predictions are evaluated as written. Box diagnostics report fixed-IoU
precision/recall/F1; they do not compute mAP. A small CPU-only example is included:

```bash
python -m pixelpilot.evaluate --predictions examples/predictions.jsonl --task action
```

## Lift waypoints and evaluate planning

```bash
python -m pixelpilot.lift --predictions outputs/predictions.jsonl \
  --nusc-root data/nuscenes --ground-z -1.0 --frame PLANNING \
  --output outputs/trajectories.json

python -m pixelpilot.evaluate_planning --predictions outputs/trajectories.json \
  --ground-truth data/annotations/val_metric_gt.json

python -m pixelpilot.benchmark.evaluate --predictions outputs/trajectories.json \
  --split-json data/annotations/val_integrated.json --infos data/infos/val.pkl \
  --nusc-root data/nuscenes --output outputs/nuscenes_metrics.json
```

Install `requirements/evaluation.txt` for the benchmark evaluator and provide
nuScenes maps. `PLANNING` is the recovered `(forward, left)` convention;
`--ground-z` is measured in current LIDAR_TOP coordinates. Metric ground truth
comes independently from info-file planning targets.

See [evaluation](evaluation.md) for frame alignment, validity masks, denominators
and metric definitions. Local validation has not exercised 7B inference or
real-map collision/boundary evaluation; see [validation](validation.md).
