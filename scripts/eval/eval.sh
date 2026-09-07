
set -o pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29900
export RANK=0
export WORLD_SIZE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models

target_name_or_path=${MODEL_DIR}/Qwen/Qwen3-4B

checkpoint_dir=${1:-}
STRIDE=${2:-2616}
START=${3:-1}
END=${4:-1}

checkpoint_dir="${DEEPSPEC_DIR}/train_log_checkpoints/${checkpoint_dir}"

if [ -d "$checkpoint_dir" ]; then
    first_checkpoint_dir=$(find "$checkpoint_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d -print -quit)
    if [ -z "$first_checkpoint_dir" ]; then
        echo "错误：未在 $checkpoint_root 找到 checkpoint 子目录" >&2
        exit 1
    fi
    # checkpoint_subdir=${checkpoint_dir}/checkpoints/${first_checkpoint_dir}
    checkpoint_subdir=${first_checkpoint_dir}
    echo "checkpoint_dir: $checkpoint_dir"
else
    echo "错误：checkpoint 目录不存在：$draft_name_or_path" >&2
    exit 1
fi
output_dir=${checkpoint_dir}/eval
mkdir -p ${output_dir}

for epoch in $(seq $START $END); do
    STEP=$((epoch * STRIDE))
    draft_name_or_path=${checkpoint_subdir}/step_${STEP}

    suffix="STEP_$STEP"
    if [ $STRIDE -eq 2616 ]; then
        suffix="epoch_${epoch}"
    fi
    echo "eval $suffix"
    DEEPSPEC_INFERENCE_EVAL=1 python eval.py \
        --target_name_or_path ${target_name_or_path} \
        --draft_name_or_path ${draft_name_or_path} \
        2>&1 | tee -a ${output_dir}/myspec_${suffix}.log
done
