
set -o pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29900
export RANK=0
export WORLD_SIZE=1
DEEPSEED_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec
MODEL_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models

target_name_or_path=${MODEL_DIR}/Qwen/Qwen3-4B


STRIDE=2616
checkpoint_dir=${DEEPSEED_DIR}/train_log_checkpoints/train_myspec_qwen3_4b_draft-cot_20260825_215515
output_dir=${checkpoint_dir}/eval
echo "checkpoint_dir: $checkpoint_dir"
mkdir -p ${output_dir}

for epoch in $(seq 1); do
    STEP=$((epoch * STRIDE))

    draft_name_or_path=${checkpoint_dir}/checkpoints/myspec_latent_cot_l2_d3_block7_qwen3_4b/step_${STEP}

    suffix="STEP_$STEP"
    if [ $STRIDE -eq 2616 ]; then
        suffix="epoch_${epoch}"
    fi
    echo "eval $suffix"
    python eval.py \
        --target_name_or_path ${target_name_or_path} \
        --draft_name_or_path ${draft_name_or_path} \
        2>&1 | tee -a ${output_dir}/myspec_${suffix}.log
done
