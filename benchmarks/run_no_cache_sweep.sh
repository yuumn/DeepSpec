#!/usr/bin/env bash
set -euo pipefail
cd /root/DeepSpec
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/DeepSpec
mkdir -p benchmarks/output
.venv/bin/python benchmarks/validate_optimized.py \
  --models benchmarks/models --output benchmarks/output/no_cache_validation.json \
  --lengths 128 256 512 1024 2048 4096 8192 16384 --cache-modes no_cache \
  2>&1 | tee benchmarks/output/no_cache_validation.log
.venv/bin/python benchmarks/benchmark_optimized.py \
  --models benchmarks/models --output benchmarks/output/no_cache_sweep.json \
  --variant rope --include-reference --rounds 5 --repeats 200 \
  --lengths 128 256 512 1024 2048 4096 8192 16384 --cache-modes no_cache \
  2>&1 | tee benchmarks/output/no_cache_sweep.log
