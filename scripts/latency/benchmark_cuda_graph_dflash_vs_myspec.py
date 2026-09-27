#!/usr/bin/env python3
"""Compare per-context CUDA graphs for DFlash and packed-GEMM MySpec."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

import benchmark_draft_latency as dflash_benchmark
import benchmark_optimized_myspec as myspec_benchmark
from benchmark_cuda_graph_myspec import measure_graph
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
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def capture_callable(fn, warmup: int):
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        static_outputs = fn()
    torch.cuda.synchronize()
    return graph, static_outputs, (time.perf_counter() - started) * 1000.0


def benchmark_dflash(args: argparse.Namespace) -> dict[int, dict]:
    model = dflash_benchmark.build_model("dflash", args)
    rows = {}
    for length in args.context_lengths:
        inputs = dflash_benchmark.make_inputs(model, length)
        graph, outputs, capture_ms = capture_callable(
            lambda: dflash_benchmark.propose("dflash", model, inputs),
            args.capture_warmup,
        )
        metrics = measure_graph(graph, args.warmup, args.repeats)
        metrics["capture_ms"] = capture_ms
        rows[length] = metrics
        del graph, outputs, inputs
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def benchmark_myspec(args: argparse.Namespace) -> dict[int, dict]:
    model = myspec_benchmark.build_model(args)
    optimize_myspec_for_inference(model)
    rows = {}
    for length in args.context_lengths:
        inputs = myspec_benchmark.make_inputs(model, length)
        graph, outputs, capture_ms = capture_callable(
            lambda: myspec_benchmark.draft(model, inputs, optimized=True),
            args.capture_warmup,
        )
        metrics = measure_graph(graph, args.warmup, args.repeats)
        metrics["capture_ms"] = capture_ms
        rows[length] = metrics
        del graph, outputs, inputs
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dflash = benchmark_dflash(args)
    myspec = benchmark_myspec(args)
    rows = []
    for length in args.context_lengths:
        dflash_mean = dflash[length]["mean_ms"]
        myspec_mean = myspec[length]["mean_ms"]
        row = {
            "context_length": length,
            "dflash_graph_mean_ms": dflash_mean,
            "myspec_graph_mean_ms": myspec_mean,
            "myspec_vs_dflash_percent": 100.0
            * (myspec_mean - dflash_mean)
            / dflash_mean,
            "dflash_over_myspec_ratio": dflash_mean / myspec_mean,
            "dflash_graph_p50_ms": dflash[length]["p50_ms"],
            "myspec_graph_p50_ms": myspec[length]["p50_ms"],
            "dflash_graph_p90_ms": dflash[length]["p90_ms"],
            "myspec_graph_p90_ms": myspec[length]["p90_ms"],
            "dflash_capture_ms": dflash[length]["capture_ms"],
            "myspec_capture_ms": myspec[length]["capture_ms"],
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
