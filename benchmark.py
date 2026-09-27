import argparse
import csv
import gc
import importlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import Qwen3Config

from deepspec.modeling.myspec.qwen3_opt.modeling import Qwen3MySpecModel

Qwen3DSparkModel = importlib.import_module(
    "deepspec.modeling.dspark.qwen3-opt.modeling"
).Qwen3DSparkModel

QWEN3_4B_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

DFLASH_CONDIF = {
    **QWEN3_4B_CONFIG,
    "architectures": ["Qwen3DSparkModel"],
    "num_target_layers": 36,
    "num_hidden_layers": 5,
    "layer_types": ["full_attention"] * 5,
    "block_size": 7,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "mask_token_id": 151669,
    "num_anchors": 512,
    "enable_confidence_head": False,
    "markov_rank": 0,
}

MYSPEC_CONFIG = {
    **QWEN3_4B_CONFIG,
    "architectures": ["Qwen3MySpecModel"],
    "num_target_layers": 36,
    "num_hidden_layers": 3,
    "layer_types": ["full_attention"] * 3,
    "block_size": 7,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "mask_token_id": 151669,
    "num_anchors": 512,
    "enable_confidence_head": False,
    "markov_rank": 0,
    "num_latent_layers": 2,
    "num_latent_tokens": 4,
    "latent_token_id": 151670,
}

DRAFT_CONFIG_FIELDS = {
    "architectures",
    "num_target_layers",
    "num_hidden_layers",
    "layer_types",
    "block_size",
    "target_layer_ids",
    "mask_token_id",
    "num_anchors",
    "enable_confidence_head",
    "markov_rank",
    "num_latent_layers",
    "num_latent_tokens",
    "latent_token_id",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark only DFlash/MySpec _forward_backbone()."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional Qwen3 target config.json; built-in Qwen3-4B values are the default.",
    )
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--models", nargs="+", choices=("dflash", "myspec"), default=["dflash", "myspec"]
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture and replay one fixed-shape CUDA Graph per model/context.",
    )
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Write one Chrome trace per model/context length when specified.",
    )
    return parser.parse_args()


def _parse_overrides(values):
    overrides = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--opts must use KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        try:
            overrides[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            overrides[key] = raw_value
    return overrides


def build_config(kind, args):
    draft_payload = DFLASH_CONDIF if kind == "dflash" else MYSPEC_CONFIG
    if args.config is not None:
        if not args.config.is_file():
            raise FileNotFoundError(f"target config does not exist: {args.config}")
        payload = Qwen3Config.from_json_file(str(args.config)).to_dict()
        num_target_layers = int(payload["num_hidden_layers"])
        payload.update(
            {key: value for key, value in draft_payload.items() if key in DRAFT_CONFIG_FIELDS}
        )
        payload["tie_word_embeddings"] = False
        payload["num_target_layers"] = num_target_layers
    else:
        payload = dict(draft_payload)
    payload.update(_parse_overrides(args.opts))
    config = Qwen3Config(**payload)
    config._attn_implementation = "sdpa"
    return config


def build_model(kind, args, dtype):
    config = build_config(kind, args)
    model_cls = Qwen3DSparkModel if kind == "dflash" else Qwen3MySpecModel
    return model_cls(config).to(device=args.device, dtype=dtype).eval()


@torch.inference_mode()
def make_inputs(model, context_length, seed):
    generator = torch.Generator(device=model.device).manual_seed(seed + context_length)
    draft_ids = torch.full(
        (1, model.block_size),
        model.mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    draft_ids[:, 0] = 1
    inputs = {
        "noise_embedding": model.embed_tokens(draft_ids),
        "position_ids": torch.arange(
            context_length + model.block_size,
            dtype=torch.long,
            device=model.device,
        ).unsqueeze(0),
        "target_hidden": torch.randn(
            1,
            context_length,
            len(model.target_layer_ids) * model.config.hidden_size,
            dtype=next(model.parameters()).dtype,
            device=model.device,
            generator=generator,
        ),
    }
    if hasattr(model, "num_latent_tokens"):
        latent_ids = torch.full(
            (1, model.num_latent_tokens),
            model.latent_token_id,
            dtype=torch.long,
            device=model.device,
        )
        latent_ids[:, 0] = draft_ids[:, 0]
        context_positions = torch.arange(
            context_length, dtype=torch.long, device=model.device
        ).unsqueeze(0)
        latent_positions = torch.arange(
            context_length,
            context_length + model.num_latent_tokens,
            dtype=torch.long,
            device=model.device,
        ).unsqueeze(0)
        mask_positions = torch.arange(
            context_length,
            context_length + model.block_size,
            dtype=torch.long,
            device=model.device,
        ).unsqueeze(0)
        inputs["latent_noise_embedding"] = model.embed_tokens(latent_ids)
        inputs["latent_position_ids"] = torch.cat(
            (context_positions, latent_positions), dim=1
        )
        inputs["position_ids"] = torch.cat(
            (context_positions, latent_positions, mask_positions), dim=1
        )
    return inputs


@torch.inference_mode()
def run_backbone(kind, model, inputs):
    common = dict(
        position_ids=inputs["position_ids"],
        target_hidden_states=inputs["target_hidden"],
        noise_embedding=inputs["noise_embedding"],
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
    if kind == "dflash":
        return model._forward_backbone(**common)
    return model._forward_backbone(
        latent_noise_embedding=inputs["latent_noise_embedding"],
        latent_position_ids=inputs["latent_position_ids"],
        latent_attention_mask=None,
        **common,
    )


def measure(kind, model, inputs, warmup, repeats):
    for _ in range(warmup):
        run_backbone(kind, model, inputs)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    samples = []
    for _ in range(repeats):
        if model.device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            run_backbone(kind, model, inputs)
            end.record()
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))
        else:
            import time

            started = time.perf_counter()
            run_backbone(kind, model, inputs)
            samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": samples[max(0, int(0.9 * len(samples)) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def capture_graph(kind, model, inputs, warmup):
    warmup_stream = torch.cuda.Stream(device=model.device)
    warmup_stream.wait_stream(torch.cuda.current_stream(model.device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup):
            run_backbone(kind, model, inputs)
    torch.cuda.current_stream(model.device).wait_stream(warmup_stream)
    torch.cuda.synchronize(model.device)

    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        static_output = run_backbone(kind, model, inputs)
    torch.cuda.synchronize(model.device)
    capture_ms = (time.perf_counter() - capture_started) * 1000.0
    return graph, static_output, capture_ms


def measure_graph(graph, device, warmup, repeats):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
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
        "p50_ms": statistics.median(samples),
        "p90_ms": samples[max(0, int(0.9 * len(samples)) - 1)],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def export_profile(kind, model, inputs, output_path, graph=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if model.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True) as prof:
        if graph is None:
            with record_function(f"{kind}_forward_backbone"):
                run_backbone(kind, model, inputs)
        else:
            with record_function(f"{kind}_forward_backbone_cuda_graph"):
                graph.replay()
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
    prof.export_chrome_trace(str(output_path))


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    if args.capture_warmup < 1:
        raise ValueError("capture-warmup must be >= 1")
    if any(length < 1 for length in args.context_lengths):
        raise ValueError("context lengths must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    for kind in args.models:
        model = build_model(kind, args, dtype)
        if args.cuda_graph and hasattr(model, "prepare_for_cuda_graph"):
            with torch.inference_mode():
                model.prepare_for_cuda_graph()
        for context_length in args.context_lengths:
            inputs = make_inputs(model, context_length, args.seed)
            graph = None
            static_output = None
            capture_ms = None
            if args.cuda_graph:
                if model.device.type != "cuda":
                    raise ValueError("--cuda-graph requires a CUDA device")
                graph, static_output, capture_ms = capture_graph(
                    kind, model, inputs, args.capture_warmup
                )
                metrics = measure_graph(
                    graph, model.device, args.warmup, args.repeats
                )
            else:
                metrics = measure(kind, model, inputs, args.warmup, args.repeats)
            row = {
                "model": kind,
                "scope": (
                    "forward_backbone_cuda_graph"
                    if args.cuda_graph
                    else "forward_backbone"
                ),
                "context_length": context_length,
                **metrics,
                "warmup": args.warmup,
                "repeats": args.repeats,
            }
            if capture_ms is not None:
                row["capture_ms"] = capture_ms
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if args.profile_dir is not None:
                export_profile(
                    kind,
                    model,
                    inputs,
                    args.profile_dir / f"{kind}_{context_length}.json",
                    graph=graph,
                )
            del graph, static_output, inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)



if __name__ == "__main__":
    main()
