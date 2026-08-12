# Local launch mirrors the repo's node launcher, not standard torchrun
# semantics. eval.py spawns one worker per visible GPU by itself.
# Here RANK/WORLD_SIZE mean node_rank/node_count, so WORLD_SIZE=1 is a
# single-node local run; total GPU workers come from CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29900
export RANK=0
export WORLD_SIZE=1
DEEPSEED_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec
MODEL_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models
# Match this to the target model used by the draft checkpoint.
# target_name_or_path=Qwen/Qwen3-4B
target_name_or_path=${MODEL_DIR}/Qwen/Qwen3-4B

# Training writes checkpoints under ~/checkpoints/<project_name>/<exp_name>/step_*.
# Use step_latest for the most recent checkpoint, or replace it with step_<N>.

# draft_name_or_path=${HOME}/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
# draft_name_or_path=${MODEL_DIR}/deepseek-ai/eagle3_qwen3_4b_ttt7
checkpoint_dir=${DEEPSEED_DIR}/train_log_checkpoints/train_dspark_qwen3_4b_20260717_013723
for epoch in $(seq 5 10); do
    STEP=$((epoch * 2616))

# for STEP in 10464; do
    # CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint-${STEP}"
    # ... existing code ...
    draft_name_or_path=${checkpoint_dir}/checkpoints/dspark_block7_qwen3_4b/step_${STEP}

    output_dir=${checkpoint_dir}/eval
    mkdir -p ${output_dir}
    # touch ${output_dir}/dspark_epoch_${epoch}.log

    python eval.py \
        --target_name_or_path ${target_name_or_path} \
        --draft_name_or_path ${draft_name_or_path} \
        2>&1 | tee -a ${output_dir}/dspark_epoch_${epoch}.log
done
