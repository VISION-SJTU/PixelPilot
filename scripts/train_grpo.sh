#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to a local PixelPilot 7B SFT checkpoint}"
export NUSCENES_ROOT="${NUSCENES_ROOT:-$TASK_ROOT/data/nuscenes}"
export NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/grpo_7b.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/nuscenes_grpo}"

if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf '%q ' "$PYTHON_BIN" -m verl.trainer.main "config=$CONFIG_PATH" \
        "data.train_files=$TRAIN_DATA" "worker.actor.model.model_path=$MODEL_PATH" "$@"
    printf '\n'
    exit 0
fi
"$PYTHON_BIN" -m pixelpilot.preflight --config "$CONFIG_PATH" \
    --model "$MODEL_PATH" --train-data "$TRAIN_DATA" --training --overrides "$@"
exec "$PYTHON_BIN" -m verl.trainer.main "config=$CONFIG_PATH" \
    "data.train_files=$TRAIN_DATA" "worker.actor.model.model_path=$MODEL_PATH" "$@"
