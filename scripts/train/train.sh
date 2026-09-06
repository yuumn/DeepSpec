#!/usr/bin/env bash

DEEPSPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
spec_mode=dspark
LOWER_MODEL_NAME=qwen3_4b
OUTPUT_DIR=${DEEPSPEC_DIR}/train_log_checkpoints/train_${spec_mode}_${LOWER_MODEL_NAME}_${TIMESTAMP}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export BASE_TB_DIR=${BASE_TB_DIR:-${OUTPUT_DIR}/tensorboard}
export BASE_CKPT_DIR=${BASE_CKPT_DIR:-${OUTPUT_DIR}/checkpoints}

target_cache_dir=${target_cache_dir:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec/.cache/qwen3_4b_target_cache}

# MODEL_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models
TRAIN_LOG_CHECKPOINTS=${DEEPSPEC_DIR}/train_log_checkpoints
REUSE_CKPT_DIR=${$REUSE_CKPT_DIR:-}
REUSE_CKPT_DIR=${TRAIN_LOG_CHECKPOINTS}/${REUSE_CKPT_DIR}

if [ -n "${REUSE_CKPT_DIR:-}" ] && [ -d "$REUSE_CKPT_DIR" ]; then
    export BASE_TB_DIR=$REUSE_CKPT_DIR/tensorboard
    export BASE_CKPT_DIR=$REUSE_CKPT_DIR/checkpoints
    OUTPUT_DIR=$REUSE_CKPT_DIR
    first_checkpoint_dir=$(find "$REUSE_CKPT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -print -quit)
    if [ -z "$first_checkpoint_dir" ]; then
        echo "错误：未在 $checkpoint_root 找到 checkpoint 子目录" >&2
        exit 1
    fi
    CMD_SUFFIX="--opts logging.resume_checkpoint_dir=${first_checkpoint_dir}"
else
    mkdir -p ${OUTPUT_DIR}
    cp -r ${DEEPSPEC_DIR}/config ${OUTPUT_DIR}/
    cp -r ${DEEPSPEC_DIR}/deepspec ${OUTPUT_DIR}/
fi
# export PROFILE_STEPS=3
export TIMESTAMP=${TIMESTAMP}

# nsys profile \
#     -o /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec/scripts/train/profile/out_${TIMESTAMP} \
    python train.py \
    --config config/${spec_mode}/${spec_mode}_qwen3_4b.py \
    --opts "data.target_cache_path=${target_cache_dir}" \
    --opts "data.num_workers=16" \
    --opts "train.local_batch_size=4" \
    --opts "logging.checkpointing_steps=100" \
    --opts "logging.logging_steps=10" \
    --opts "train.sharding_strategy=no_shard" ${CMD_SUFFIX:-} \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log

    # --opts "logging.tensorboard_dir=${REUSE_CKPT_DIR}/tensorboard/${spec_mode}_block7_qwen3_4b" \
    # no_shard shard_grad_op full_shard hybrid_shard hybrid_shard_zero2/_hybrid_shard_zero2
    # --opts "model.target_model_name_or_path=${MODEL_DIR}/Qwen/Qwen3-4B" \
