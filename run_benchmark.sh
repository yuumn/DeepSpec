#!/usr/bin/env bash
set -euo pipefail
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec/.venv/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_CONFIG="${TARGET_CONFIG:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B/config.json}"

cd "${SCRIPT_DIR}"

OUTPUT_ARGS=()
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  OUTPUT_ARGS=(--output "${OUTPUT_PATH}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
python -u benchmark.py \
  --config "${TARGET_CONFIG}" \
  --models dflash myspec \
  --context-lengths 128 256 512 1024 2048 4096 8192 16384 \
  --warmup 5 \
  --repeats 30 \
  "${OUTPUT_ARGS[@]}" \
  "$@"
