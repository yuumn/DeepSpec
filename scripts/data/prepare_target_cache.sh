#!/usr/bin/env bash
set -euo pipefail

DEEPSPEC_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONPATH=${DEEPSPEC_DIR}:${PYTHONPATH:-}

python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
    --output-dir ${DEEPSPEC_DIR}/.cache/qwen3_4b_target_cache \
    --local-batch-size 16
