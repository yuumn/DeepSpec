#!/usr/bin/env bash
set -euo pipefail

# One host / eight GPUs: physical GPU 0 serves target hidden states, GPUs 1-7 train.
# Requires the CUDA-matched mooncake-transfer-engine package and mooncake_master.
DEEPSPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DEEPSPEC_DIR"

TARGET_GPU="${TARGET_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3,4,5,6,7}"
CONFIG_PATH="${CONFIG_PATH:-config/myspec/myspec_qwen3_4b.py}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?set TRAIN_DATA_PATH to a raw training JSONL file}"
TARGET_SERVER_PORT="${TARGET_SERVER_PORT:-31000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-504}"
START_TIMEOUT_S="${START_TIMEOUT_S:-900}"

IFS=',' read -r -a train_gpu_array <<< "$TRAIN_GPUS"
if [[ "${#train_gpu_array[@]}" -ne 7 ]]; then
    echo "TRAIN_GPUS must contain exactly seven GPU ids; got: $TRAIN_GPUS" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    echo "training JSONL does not exist: $TRAIN_DATA_PATH" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % (7 * LOCAL_BATCH_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by 7 * LOCAL_BATCH_SIZE" >&2
    exit 1
fi
command -v mooncake_master >/dev/null || {
    echo "mooncake_master is not on PATH" >&2
    exit 1
}
if ! mooncake_master --help 2>&1 \
    | grep -F enable_http_metadata_server >/dev/null; then
    echo "mooncake_master is too old; install Mooncake 0.3.13 or newer" >&2
    exit 1
fi
command -v curl >/dev/null || {
    echo "curl is required by this launcher" >&2
    exit 1
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${DEEPSPEC_DIR}/train_log_checkpoints/train_myspec_qwen3_4b_realtime_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

export MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-35551}"
export MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-35880}"
export MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-35903}"
export MOONCAKE_DEFAULT_KV_LEASE_TTL="${MOONCAKE_DEFAULT_KV_LEASE_TTL:-500}"
export MOONCAKE_MASTER_SERVER_ADDR="127.0.0.1:${MOONCAKE_RPC_PORT}"
export MOONCAKE_METADATA_SERVER="http://127.0.0.1:${MOONCAKE_HTTP_PORT}/metadata"
export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}"
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_RDMA_DEVICES="${MOONCAKE_RDMA_DEVICES:-}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}"
export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((1 << 30))}"
export DEEPSPEC_MOONCAKE_STORE_ID="${DEEPSPEC_MOONCAKE_STORE_ID:-deepspec-${TIMESTAMP}}"
export MC_TCP_BIND_ADDRESS="${MC_TCP_BIND_ADDRESS:-$MOONCAKE_LOCAL_HOSTNAME}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export BASE_TB_DIR="${BASE_TB_DIR:-${OUTPUT_DIR}/tensorboard}"
export BASE_CKPT_DIR="${BASE_CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"
export TIMESTAMP

master_pid=""
target_pid=""
cleanup() {
    if [[ -n "$target_pid" ]]; then
        kill "$target_pid" 2>/dev/null || true
        wait "$target_pid" 2>/dev/null || true
    fi
    if [[ -n "$master_pid" ]]; then
        kill "$master_pid" 2>/dev/null || true
        wait "$master_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=127.0.0.1 \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port="$MOONCAKE_METRICS_PORT" \
    --default_kv_lease_ttl="$MOONCAKE_DEFAULT_KV_LEASE_TTL" \
    >"$OUTPUT_DIR/mooncake.log" 2>&1 &
master_pid="$!"

started="$(date +%s)"
until curl -sS --max-time 1 -o /dev/null \
    "${MOONCAKE_METADATA_SERVER}?key=deepspec-health-check" \
    && python -c \
        'import socket,sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), 1).close()' \
        127.0.0.1 "$MOONCAKE_RPC_PORT"; do
    kill -0 "$master_pid" 2>/dev/null || {
        echo "Mooncake exited; see $OUTPUT_DIR/mooncake.log" >&2
        exit 1
    }
    if (( $(date +%s) - started >= START_TIMEOUT_S )); then
        echo "timed out waiting for Mooncake" >&2
        exit 1
    fi
    sleep 1
done

CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m deepspec.data.target_realtime_prepare \
    --config "$CONFIG_PATH" \
    --host 127.0.0.1 \
    --port "$TARGET_SERVER_PORT" \
    >"$OUTPUT_DIR/target_server.log" 2>&1 &
target_pid="$!"

started="$(date +%s)"
until curl -fsS --max-time 1 -o /dev/null \
    "http://127.0.0.1:${TARGET_SERVER_PORT}/health"; do
    kill -0 "$target_pid" 2>/dev/null || {
        echo "target service exited; see $OUTPUT_DIR/target_server.log" >&2
        exit 1
    }
    if (( $(date +%s) - started >= START_TIMEOUT_S )); then
        echo "timed out waiting for target service" >&2
        exit 1
    fi
    sleep 2
done

echo "target hidden states: physical GPU $TARGET_GPU"
echo "DeepSpec training: physical GPUs $TRAIN_GPUS"
echo "gradient accumulation: $((GLOBAL_BATCH_SIZE / (7 * LOCAL_BATCH_SIZE)))"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" python train.py \
    --config "$CONFIG_PATH" \
    --opts "data.hidden_state_source=transformers_realtime" \
    --opts "data.train_data_paths=['${TRAIN_DATA_PATH}']" \
    --opts "data.target_server_url=http://127.0.0.1:${TARGET_SERVER_PORT}" \
    --opts "data.num_workers=0" \
    --opts "train.local_batch_size=${LOCAL_BATCH_SIZE}" \
    --opts "train.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    2>&1 | tee -a "$OUTPUT_DIR/train.log"
