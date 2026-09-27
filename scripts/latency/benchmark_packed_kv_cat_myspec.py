#!/usr/bin/env python3
"""Measure one packed K/V cat versus separate K and V cats in MySpec graphs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from benchmark_cuda_graph_dflash_vs_myspec import capture_callable
from benchmark_cuda_graph_myspec import measure_graph
from benchmark_optimized_myspec import build_model, draft, make_inputs
from optimized_myspec import optimize_myspec_for_inference


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument(
        "--myspec-config", type=Path, default=ROOT / "config/myspec/myspec_qwen3_4b.py"
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace, packed_kv_cat: bool, references=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(args)
    optimize_myspec_for_inference(model, packed_kv_cat=packed_kv_cat)
    metrics = {}
    outputs_by_length = {}
    for length in args.context_lengths:
        torch.manual_seed(args.seed + length)
        torch.cuda.manual_seed_all(args.seed + length)
        inputs = make_inputs(model, length)
        graph, outputs, _ = capture_callable(
            lambda: draft(model, inputs, optimized=True), args.capture_warmup
        )
        current = tuple(output.clone() for output in outputs)
        if references is not None:
            torch.testing.assert_close(
                current[0], references[length][0].to(current[0].device), rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                current[1], references[length][1].to(current[1].device), rtol=2e-2, atol=2e-4
            )
        else:
            outputs_by_length[length] = tuple(output.cpu() for output in current)
            torch.cuda.synchronize()
        metrics[length] = measure_graph(graph, args.warmup, args.repeats)
        del graph, outputs, current, inputs
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, outputs_by_length


def main() -> None:
    args = parse_args()
    baseline, references = run(args, packed_kv_cat=False)
    packed, _ = run(args, packed_kv_cat=True, references=references)
    rows = []
    for length in args.context_lengths:
        base_ms = baseline[length]["mean_ms"]
        packed_ms = packed[length]["mean_ms"]
        row = {
            "context_length": length,
            "separate_cat_mean_ms": base_ms,
            "packed_kv_cat_mean_ms": packed_ms,
            "speedup": base_ms / packed_ms,
            "reduction_percent": 100.0 * (base_ms - packed_ms) / base_ms,
            "separate_cat_p50_ms": baseline[length]["p50_ms"],
            "packed_kv_cat_p50_ms": packed[length]["p50_ms"],
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
