#!/usr/bin/env python3
"""Compare baseline and embedding/KV-fused MySpec first-proposal latency."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from transformers import DynamicCache, Qwen3Config

from deepspec.eval.myspec.draft_ops import (
    build_myspec_proposal,
    forward_myspec_draft_block,
)
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.modeling.myspec.qwen3.config import build_draft_config
from deepspec.utils import load_config
from optimized_myspec import (
    forward_optimized_myspec_draft_block,
    optimize_myspec_for_inference,
)


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> Qwen3MySpecModel:
    model_args = load_config(args.myspec_config).model
    target_config = Qwen3Config.from_json_file(str(args.target_config))
    config = build_draft_config(target_config, model_args)
    config._attn_implementation = "sdpa"
    return Qwen3MySpecModel(config).to(device="cuda", dtype=torch.bfloat16).eval()


def make_inputs(model: Qwen3MySpecModel, context_length: int) -> dict:
    device = next(model.parameters()).device
    draft_ids = torch.full(
        (1, model.block_size), model.mask_token_id, device=device, dtype=torch.long
    )
    draft_ids[:, 0] = 1
    latent_ids = torch.full(
        (1, model.num_latent_tokens),
        model.latent_token_id,
        device=device,
        dtype=torch.long,
    )
    latent_ids[:, 0] = draft_ids[:, 0]
    return {
        "draft_ids": draft_ids,
        "latent_ids": latent_ids,
        "target_hidden": torch.randn(
            1,
            context_length,
            len(model.target_layer_ids) * model.config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ),
        "positions": torch.arange(
            context_length + model.block_size, device=device
        ).unsqueeze(0),
        "start": context_length,
    }


@torch.inference_mode()
def draft(model: Qwen3MySpecModel, inputs: dict, optimized: bool) -> torch.Tensor:
    common = dict(
        position_ids=inputs["positions"],
        past_key_values_draft=DynamicCache(),
        target_hidden_states=inputs["target_hidden"],
        start=inputs["start"],
        block_size=model.block_size,
    )
    if optimized:
        hidden = forward_optimized_myspec_draft_block(
            model, anchor_ids=inputs["draft_ids"][:, :1], **common
        )
    else:
        hidden = forward_myspec_draft_block(
            model,
            draft_latent_input_ids=inputs["latent_ids"],
            draft_input_ids=inputs["draft_ids"],
            **common,
        )
    proposal = build_myspec_proposal(
        model,
        draft_input_ids=inputs["draft_ids"],
        block_hidden=hidden,
        block_size=model.block_size,
        temperature=1.0,
        confidence_threshold=0.0,
    )
    return hidden, proposal.draft_probs


def measure(model, inputs, optimized, warmup, repeats) -> dict[str, float]:
    for _ in range(warmup):
        draft(model, inputs, optimized)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        draft(model, inputs, optimized)
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_model(args)
    inputs_by_length = {
        length: make_inputs(model, length) for length in args.context_lengths
    }

    # Capture the reference before installing the fused forwards.
    references = {
        length: draft(model, inputs, optimized=False)
        for length, inputs in inputs_by_length.items()
    }
    baseline_rows = {
        length: measure(model, inputs, False, args.warmup, args.repeats)
        for length, inputs in inputs_by_length.items()
    }

    optimize_myspec_for_inference(model)
    for length, inputs in inputs_by_length.items():
        actual_hidden, actual_probs = draft(model, inputs, optimized=True)
        expected_hidden, expected_probs = references[length]
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual_probs, expected_probs, rtol=2e-2, atol=2e-4)

    rows = []
    for length, inputs in inputs_by_length.items():
        optimized = measure(model, inputs, True, args.warmup, args.repeats)
        baseline = baseline_rows[length]
        row = {
            "context_length": length,
            "baseline_mean_ms": baseline["mean_ms"],
            "optimized_mean_ms": optimized["mean_ms"],
            "speedup": baseline["mean_ms"] / optimized["mean_ms"],
            "reduction_percent": 100.0
            * (baseline["mean_ms"] - optimized["mean_ms"])
            / baseline["mean_ms"],
            **{f"baseline_{key}": value for key, value in baseline.items() if key != "mean_ms"},
            **{f"optimized_{key}": value for key, value in optimized.items() if key != "mean_ms"},
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
