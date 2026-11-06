# PixelPilot

**Scalable Vision-Language-Action Models for End-to-End Autonomous Driving (ECCV 2026)**

[Paper](https://arxiv.org/abs/2607.04637) · [Project Page](https://pixelpilotvla.github.io/) · [中文](README_zh.md)

PixelPilot decouples image-space planning from geometric lifting. The model
predicts perception, driving actions and future waypoints in the image plane;
the waypoints are lifted to metric trajectories using camera calibration at
inference time. Training combines multi-task SFT, selected visual reasoning and GRPO.

This repository provides the **nuScenes / Qwen2.5-VL-7B** implementation, with
supervised training built on **LLaMA-Factory** and reinforcement learning built
on the supplied **Seg-Zero / EasyR1 / veRL** code. The GRPO recipe retains a
historical reward variant; checkpoint availability and differences from the
paper are documented in [reproduction status](docs/reproduction.md).

## 🛠️ Installation

Use Linux and Python 3.10 for GPU training. Create separate environments for
SFT and GRPO, and install a compatible PyTorch/CUDA stack in each environment.
Run all commands from the repository root.

```bash
# Data preparation environment
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .

# SFT environment
python -m pip install -r requirements/sft.txt
python -m pip install --no-deps -e ./LLaMAFactory
python -m pip install --no-deps -e .

# GRPO / inference environment
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install --no-deps -e .
```

Download [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
as the base model. See [installation](docs/installation.md) for the environment
setup, pinned dependencies and optional evaluation tools.

## 📂 Data Preparation

Prepare the [nuScenes](https://www.nuscenes.org/nuscenes) camera data, calibration
tables and maps, together with **OmniDrive-style planning info pickles** and the
**original metric-space conversation JSON**:

```text
data/
├── nuscenes/                  # Camera data, metadata and maps
├── infos/{train,val}.pkl      # Planning infos
└── raw/nuscenes_{train,val}.json
```

```bash
python -m pixelpilot.generate \
  --train-json data/raw/nuscenes_train.json --train-infos data/infos/train.pkl \
  --val-json data/raw/nuscenes_val.json --val-infos data/infos/val.pkl \
  --nusc-root data/nuscenes --output-dir data/annotations --seed 42

python -m pixelpilot.collage --annotations data/annotations/train_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_train --native-resolution
python -m pixelpilot.collage --annotations data/annotations/val_integrated.json \
  --nusc-root data/nuscenes --output-dir data/nusc_fb_val --native-resolution
```

This generates multi-task supervision, integrated tasks, independent metric
ground truth and front/rear image collages. The upstream raw-nuScenes-to-info
converter is not included. See [data preparation](docs/data.md) for input fields,
coordinate conventions and generated files.

## ⚙️ Training

### Stage 1: Multi-task SFT

Convert the annotations and train the 7B model with the bundled LLaMA-Factory:

```bash
python -m pixelpilot.prepare --format sft \
  --train-json data/annotations/train_sft.json \
  --val-json data/annotations/val_integrated.json \
  --image-root data --output-dir data/nuscenes_sft

bash scripts/train_sft.sh --num-gpus 8
```

The default [SFT configuration](configs/sft_7b.yaml) uses full fine-tuning,
ZeRO-3, two epochs and a learning rate of `5e-6`. Override model, data and output
paths with `--model`, `--dataset-dir` and `--output-dir`.

### Stage 2: Visual Reasoning

Generate teacher explanations for training failure cases, run paired student
inference, and select explanations that correct the action and improve metric
trajectory L2. Append the selected records with
`pixelpilot.prepare --format sft --extra-train-json ...`, then continue SFT.
The complete workflow is in [reasoning data](docs/reasoning.md).

### Stage 3: GRPO

Switch to the GRPO environment and prepare the integrated training tasks:

```bash
python -m pixelpilot.prepare \
  --train-json data/annotations/train_integrated.json \
  --image-root data --output-dir data/nuscenes_grpo

export MODEL_PATH="$PWD/checkpoints/pixelpilot-7b-sft"
export NUSCENES_ROOT="$PWD/data/nuscenes"
bash scripts/train_grpo.sh
```

The [GRPO configuration](configs/grpo_7b.yaml) uses four GPUs and eight
completions per prompt. See [training](docs/training.md) for configuration details,
launch previews and checkpoint handling.

## 🔑 Inference and Evaluation

Run inference with a local full-SFT or merged GRPO checkpoint, then lift the
predicted image-space waypoints:

```bash
python -m pixelpilot.infer --model checkpoints/pixelpilot-7b-sft \
  --annotations data/annotations/val_integrated.json --image-root data \
  --output outputs/predictions.jsonl

python -m pixelpilot.lift --predictions outputs/predictions.jsonl \
  --nusc-root data/nuscenes --ground-z -1.0 --frame PLANNING \
  --output outputs/trajectories.json
```

See [inference](docs/inference.md) for checkpoint merging and text diagnostics,
and [evaluation](docs/evaluation.md) for L2, collision and boundary metrics.
Local validation covers CPU checks; GPU training and real-map benchmark runs
remain unverified. The observed checks are recorded in [validation](docs/validation.md).

## 🗂️ Repository

```text
PixelPilot/
├── LLaMAFactory/       # Bundled SFT framework (0.9.4.dev0)
├── pixelpilot/         # Data, reasoning, inference, lifting and evaluation
├── verl/               # GRPO framework and reward implementation
├── configs/            # SFT / GRPO recipes and historical configurations
├── scripts/            # Training launchers, checkpoint merge and source export
├── requirements/       # Dependencies for each environment
├── examples/           # Small annotation and prediction examples
├── tests/              # CPU regression tests
└── docs/               # Usage, reproduction status and validation
```

Browse the [documentation index](docs/README.md) for detailed guides and
[development](docs/development.md) for local checks and source packaging.

## 📝 Citation

```bibtex
@article{tang2026pixelpilot,
  title={PixelPilot: Scalable Vision-Language-Action Models for End-to-End Autonomous Driving},
  author={Tang, Pin and Wang, Guoqing and Ren, Xiangxuan and Wang, Zhongdao and
          Zhao, Guodongfang and Feng, Bailan and Ma, Chao},
  journal={arXiv preprint arXiv:2607.04637},
  year={2026}
}
```

## 🙏 Acknowledgements

This implementation builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
[Seg-Zero](https://github.com/dvlab-research/Seg-Zero),
[EasyR1](https://github.com/hiyouga/EasyR1),
[veRL](https://github.com/volcengine/verl),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
[OmniDrive](https://github.com/NVlabs/OmniDrive) and
[nuScenes](https://www.nuscenes.org/nuscenes).

The code is released under [Apache-2.0](LICENSE). Third-party notices are retained
in [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Dataset and pretrained-model terms apply separately.
