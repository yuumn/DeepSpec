#!/usr/bin/env bash
set -euo pipefail

DEEPSPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC_DIR="$(dirname "${DEEPSPEC_DIR}")"
LOWER_MODEL_NAME=${LOWER_MODEL_NAME:-qwen3_8b}
SPEC_MODE=${SPEC_MODE:-myspec}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH=${DEEPSPEC_DIR}:${PYTHONPATH:-}

TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"${SPEC_DIR}/train_datasets/${LOWER_MODEL_NAME}/perfectblend_train_regen.jsonl"}
TOKEN_CACHE_PATH=${TOKEN_CACHE_PATH:-"${DEEPSPEC_DIR}/.cache/${LOWER_MODEL_NAME}_token_cache"}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-14}

cd "${DEEPSPEC_DIR}"
python scripts/data/prepare_token-ids_loss-mask.py \
    --config "config/${SPEC_MODE}/${SPEC_MODE}_${LOWER_MODEL_NAME}.py" \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --output-dir "${TOKEN_CACHE_PATH}" \
    --local-batch-size "${LOCAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --min-loss-tokens "${MIN_LOSS_TOKENS}"
