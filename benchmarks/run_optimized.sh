#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepSpec
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/DeepSpec
mkdir -p benchmarks/output
.venv/bin/python benchmarks/validate_optimized.py \
  --models benchmarks/models --output benchmarks/output/validation_final.json \
  2>&1 | tee benchmarks/output/validation_final.log
.venv/bin/python benchmarks/benchmark_optimized.py \
  --models benchmarks/models --output benchmarks/output/optimized_final.json \
  --variant rope --include-reference --rounds 5 --repeats 200 \
  2>&1 | tee benchmarks/output/optimized_final.log

.venv/bin/python benchmarks/check_optimized_api.py
