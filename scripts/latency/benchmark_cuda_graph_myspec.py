#!/usr/bin/env python3
"""Benchmark per-context CUDA graphs for packed-GEMM MySpec proposals."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from benchmark_optimized_myspec import build_model, draft, make_inputs, measure
from optimized_myspec import optimize_myspec_for_inference


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", type=Path, required=True)
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


def capture_proposal(model, inputs, warmup: int):
    """Capture one fixed-shape proposal and retain all referenced tensors."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            draft(model, inputs, optimized=True)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        static_outputs = draft(model, inputs, optimized=True)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - capture_started) * 1000.0
    return graph, static_outputs, capture_ms


def measure_graph(graph: torch.cuda.CUDAGraph, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    samples.sort()
    return {
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p50_ms": statistics.median(samples),
        "p90_ms": samples[max(0, int(0.9 * len(samples)) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.capture_warmup < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("invalid warmup/repeat counts")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build_model(args)
    optimize_myspec_for_inference(model)
    rows = []
    for length in args.context_lengths:
        # Each length owns static addresses and a separately captured graph.
        inputs = make_inputs(model, length)
        eager = measure(model, inputs, True, args.warmup, args.repeats)
        graph, static_outputs, capture_ms = capture_proposal(
            model, inputs, args.capture_warmup
        )
        graphed = measure_graph(graph, args.warmup, args.repeats)
        row = {
            "context_length": length,
            "eager_mean_ms": eager["mean_ms"],
            "graph_mean_ms": graphed["mean_ms"],
            "speedup": eager["mean_ms"] / graphed["mean_ms"],
            "reduction_percent": 100.0
            * (eager["mean_ms"] - graphed["mean_ms"])
            / eager["mean_ms"],
            "capture_ms": capture_ms,
            "eager_p50_ms": eager["p50_ms"],
            "graph_p50_ms": graphed["p50_ms"],
            "eager_p90_ms": eager["p90_ms"],
            "graph_p90_ms": graphed["p90_ms"],
            "repeats": args.repeats,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        # Keep outputs alive through all replays; release this length only after
        # its measurements complete.
        del static_outputs, graph, inputs
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
