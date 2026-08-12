

tensorboard_path=${1:-}
PORT=${2:-6006}


echo "ssh -L $PORT:localhost:$PORT h20"
echo "http://localhost:$PORT"

tensorboard --logdir \
    $tensorboard_path \
    --port $PORT \
    --bind_all

    # /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/DeepSpec/.tensorboard/deepspec/dspark_block7_qwen3_4b \

