# Installation

## Environments

Use Linux and Python 3.10 for the historical GPU stack. Keep SFT and GRPO in
separate environments: the supplied LLaMA-Factory requires safetensors <=0.5.3,
whereas the supplied GRPO run used 0.6.2. Do not merge their requirements files.
The root package installs `pixelpilot` and the vendored `verl` module. A second
veRL/EasyR1 package in the same environment can shadow that module.

Data preparation:

```bash
python -m venv .venv-data
source .venv-data/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .
```

The converter reads trusted local info pickles containing numpy and Shapely
objects. Dataset downloads and preprocessing outputs are stored under `data/`.
The upstream raw-nuScenes-to-OmniDrive-info converter is not included.

## SFT

Create a separate environment and install a matching CUDA-enabled PyTorch stack:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements/sft.txt
python -m pip install --no-deps -e LLaMAFactory
python -m pip install --no-deps -e .
bash scripts/train_sft.sh --dry-run
```

The actual supplied SFT implementation is bundled. Its internal version is
`0.9.4.dev0`; no upstream commit identifier was supplied. The launcher prepends
this exact source directory to Python's import path, chooses the legacy SFT
pipeline, checks data registration and images, and invokes distributed training.
It defaults to eight processes and supports `--num-gpus`; target-server memory
requirements have not been validated locally.

The default configuration is derived from the actual two-epoch `p_a_t_pat`
configuration. The one-epoch mixed and reasoning-mixture variants are retained
under `configs/historical/` as provenance, with their original dataset names.
Use the portable main configuration and generated `pixelpilot_train` registration.
The main recipe preserves `val_size=0.01` and `eval_strategy=no` from the source;
external validation uses the separately frozen validation split.

## GRPO and GPU inference

In a separate GRPO environment:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install --no-deps -e .
```

The supplied stack uses Transformers 4.51.3, vLLM 0.8.5.post1 and FlashAttention
2.7.4.post1. Its direct dependencies are pinned from local experiment metadata;
it is not a complete transitive lockfile or a newly verified CUDA installation.
Choose the matching driver/toolkit on the target server. GPU inference currently
uses the GRPO-style environment because it explicitly loads FlashAttention.

## Optional reasoning and planning evaluation

Teacher reasoning tools: install `requirements/reasoning.txt` in the data
workspace. The default teacher endpoint follows the supplied script, and the
model id must be explicitly chosen. `--dry-run` performs no external request.
Actual generation requires `DASHSCOPE_API_KEY`; this preparation task did not
call the teacher API or upload any dataset images.

Planning metrics: install `requirements/evaluation.txt` in a compatible evaluation
environment and provide nuScenes `maps/expansion`. This includes recovered
`PlanningMetric` and map code. Real-map evaluation remains unverified.

## Local CPU tests

```bash
python -m pip install numpy==1.26.4 scipy==1.13.1 Pillow==11.3.0 \
  PyYAML==6.0.2 omegaconf==2.3.0 pyquaternion==0.9.9
python -m unittest discover -s tests -v
```

The local macOS scipy 1.15.3 wheel could not load a native component, so local
CPU validation uses 1.13.1. Linux training retains the historical 1.15.3 pin.
Detailed observed evidence is in [validation](validation.md).
