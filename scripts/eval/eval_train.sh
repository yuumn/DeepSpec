# Local launch mirrors the repo's node launcher, not standard torchrun
# semantics. eval.py spawns one worker per visible GPU by itself.
# Here RANK/WORLD_SIZE mean node_rank/node_count, so WORLD_SIZE=1 is a
# single-node local run; total GPU workers come from CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
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
draft_name_or_path=${DEEPSEED_DIR}/train_log_checkpoints/train_qwen3_4b_20260705_174128/checkpoints/dspark_block7_qwen3_4b/step_2616

output_dir=${DEEPSEED_DIR}/logs/eval_train_qwen3_4b
mkdir -p ${output_dir}


python eval.py \
    --target_name_or_path ${target_name_or_path} \
    --draft_name_or_path ${draft_name_or_path} \
    2>&1 | tee ${output_dir}/dspark_1epoch.log
