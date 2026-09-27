#!/usr/bin/env python3
"""Benchmark first-proposal latency for DFlash and MySpec on one GPU.

The timed region matches each evaluator's ``_propose`` implementation: draft
backbone forward, LM head, and token sampling.  Every measurement starts with
an empty KV cache, so ``context_length`` is the context prefetched by that
single proposal rather than a steady-state cached decode length.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path

import torch
from transformers import DynamicCache, Qwen3Config

from deepspec.eval.dspark.draft_ops import (
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.eval.myspec.draft_ops import (
    build_myspec_proposal,
    forward_myspec_draft_block,
)
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config as build_dflash_config
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.modeling.myspec.qwen3.config import build_draft_config as build_myspec_config
from deepspec.utils import load_config


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-config",
        type=Path,
        required=True,
        help="Qwen3 target config.json (weights are not needed).",
    )
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


def build_model(kind: str, args: argparse.Namespace) -> torch.nn.Module:
    config_path = args.dflash_config if kind == "dflash" else args.myspec_config
    model_args = load_config(config_path).model
    target_config = Qwen3Config.from_json_file(str(args.target_config))
    if kind == "dflash":
        config = build_dflash_config(target_config, model_args)
        model = Qwen3DSparkModel(config)
    else:
        config = build_myspec_config(target_config, model_args)
        model = Qwen3MySpecModel(config)
    # Evaluation explicitly uses SDPA.  Keep the benchmark on the same path.
    model.config._attn_implementation = "sdpa"
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def make_inputs(model: torch.nn.Module, context_length: int) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    # Evaluator semantics: target prefill covers ``context_length`` tokens and
    # the first sampled token lives at index ``start == context_length``.
    start = context_length
    position_ids = torch.arange(
        context_length + int(model.block_size), device=device, dtype=torch.long
    ).unsqueeze(0)
    target_hidden_states = torch.randn(
        1,
        context_length,
        len(model.target_layer_ids) * int(model.config.hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    draft_input_ids = torch.full(
        (1, int(model.block_size)),
        int(model.mask_token_id),
        device=device,
        dtype=torch.long,
    )
    draft_input_ids[:, 0] = 1
    return {
        "start": start,
        "position_ids": position_ids,
        "target_hidden_states": target_hidden_states,
        "draft_input_ids": draft_input_ids,
    }


@torch.inference_mode()
def propose(kind: str, model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> None:
    cache = DynamicCache()
    common = dict(
        position_ids=inputs["position_ids"],
        past_key_values_draft=cache,
        target_hidden_states=inputs["target_hidden_states"],
        start=int(inputs["start"]),
        block_size=int(model.block_size),
    )
    if kind == "dflash":
        hidden = forward_dspark_draft_block(
            model, draft_input_ids=inputs["draft_input_ids"], **common
        )
        build_dspark_proposal(
            model,
            draft_input_ids=inputs["draft_input_ids"],
            block_hidden=hidden,
            block_size=int(model.block_size),
            temperature=1.0,
            confidence_threshold=0.0,
        )
    else:
        latent_ids = torch.full(
            (1, int(model.num_latent_tokens)),
            int(model.latent_token_id),
            device=inputs["draft_input_ids"].device,
            dtype=torch.long,
        )
        latent_ids[:, 0] = inputs["draft_input_ids"][:, 0]
        hidden = forward_myspec_draft_block(
            model,
            draft_latent_input_ids=latent_ids,
            draft_input_ids=inputs["draft_input_ids"],
            **common,
        )
        build_myspec_proposal(
            model,
            draft_input_ids=inputs["draft_input_ids"],
            block_hidden=hidden,
            block_size=int(model.block_size),
            temperature=1.0,
            confidence_threshold=0.0,
        )


def benchmark_one(
    kind: str,
    model: torch.nn.Module,
    context_length: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    inputs = make_inputs(model, context_length)
    for _ in range(warmup):
        propose(kind, model, inputs)
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        propose(kind, model, inputs)
        end_event.record()
        end_event.synchronize()
        timings.append(float(start_event.elapsed_time(end_event)))

    timings.sort()
    p50 = statistics.median(timings)
    p90 = timings[max(0, int(0.9 * len(timings)) - 1)]
    return {
        "model": kind,
        "context_length": context_length,
        "mean_ms": statistics.mean(timings),
        "stdev_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "p50_ms": p50,
        "p90_ms": p90,
        "min_ms": timings[0],
        "max_ms": timings[-1],
        "repeats": repeats,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []
    for kind in ("dflash", "myspec"):
        model = build_model(kind, args)
        for context_length in args.context_lengths:
            row = benchmark_one(
                kind, model, context_length, args.warmup, args.repeats
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
