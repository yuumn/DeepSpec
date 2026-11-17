#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepSpec
export http_proxy=http://127.0.0.1:18888
export https_proxy=http://127.0.0.1:18888
export no_proxy=localhost,127.0.0.1,pypi.tuna.tsinghua.edu.cn
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/DeepSpec
mkdir -p benchmarks/output
/root/miniconda3/bin/uv pip freeze --python .venv/bin/python > benchmarks/output/environment.txt
git rev-parse HEAD > benchmarks/output/source_commit.txt
nvidia-smi -q > benchmarks/output/nvidia_smi_before.txt
.venv/bin/python benchmarks/benchmark_draft_h800.py \
  --models benchmarks/models --output benchmarks/output/results.json \
  --rounds 5 --repeats 200 2>&1 | tee benchmarks/output/benchmark.log
nvidia-smi -q > benchmarks/output/nvidia_smi_after.txt
