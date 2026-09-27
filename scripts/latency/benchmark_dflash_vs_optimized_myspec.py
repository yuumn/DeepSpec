#!/usr/bin/env python3
"""Benchmark DFlash against grouped-GEMM optimized MySpec on one GPU."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

import benchmark_draft_latency as baseline_benchmark
import benchmark_optimized_myspec as myspec_benchmark
from optimized_myspec import optimize_myspec_for_inference


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument(
        "--dflash-config",
        type=Path,
        default=ROOT / "config/dflash/dflash_qwen3_4b.py",
    )
    parser.add_argument(
        "--myspec-config",
        type=Path,
        default=ROOT / "config/myspec/myspec_qwen3_4b.py",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dflash = baseline_benchmark.build_model("dflash", args)
    dflash_rows = {}
    for length in args.context_lengths:
        dflash_rows[length] = baseline_benchmark.benchmark_one(
            "dflash", dflash, length, args.warmup, args.repeats
        )
    del dflash
    gc.collect()
    torch.cuda.empty_cache()

    myspec = myspec_benchmark.build_model(args)
    optimize_myspec_for_inference(myspec)
    rows = []
    for length in args.context_lengths:
        inputs = myspec_benchmark.make_inputs(myspec, length)
        optimized = myspec_benchmark.measure(
            myspec, inputs, True, args.warmup, args.repeats
        )
        dflash_mean = dflash_rows[length]["mean_ms"]
        myspec_mean = optimized["mean_ms"]
        row = {
            "context_length": length,
            "dflash_mean_ms": dflash_mean,
            "optimized_myspec_mean_ms": myspec_mean,
            "myspec_vs_dflash_percent": 100.0
            * (myspec_mean - dflash_mean)
            / dflash_mean,
            "dflash_over_myspec_speed_ratio": dflash_mean / myspec_mean,
            "dflash_p50_ms": dflash_rows[length]["p50_ms"],
            "optimized_myspec_p50_ms": optimized["p50_ms"],
            "dflash_p90_ms": dflash_rows[length]["p90_ms"],
            "optimized_myspec_p90_ms": optimized["p90_ms"],
            "repeats": args.repeats,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
