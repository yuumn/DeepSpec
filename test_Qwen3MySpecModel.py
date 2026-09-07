"""Self-contained forward smoke test for :class:`Qwen3MySpecModel`.

This test intentionally does not download or instantiate all Qwen3-4B weights.
It reads the real Qwen3-4B model configuration and the MySpec settings from
``config/myspec/myspec_qwen3_4b.py``, then uses small synthetic inputs so that
edits and print statements inside ``forward`` can be exercised quickly.

Run directly to see all output (including prints added inside ``forward``)::

    python test_Qwen3MySpecModel.py

It can also be run with pytest::

    pytest -s test_Qwen3MySpecModel.py
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
from transformers import Qwen3Config

from deepspec.modeling.myspec.qwen3.config import build_draft_config
from deepspec.modeling.myspec.qwen3.modeling import Qwen3MySpecModel
from deepspec.utils import load_config


ROOT = Path(__file__).resolve().parent
MYSPEC_CONFIG_PATH = ROOT / "config" / "myspec" / "myspec_qwen3_4b.py"
TARGET_CONFIG_PATH = Path(
    "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/"
    "workspace/spec/models/Qwen/Qwen3-4B/config.json"
)

# These fields determine the Qwen3 backbone shapes and attention behavior. They
# must be inherited unchanged from the target configuration by the draft model.
INHERITED_TARGET_CONFIG_FIELDS = (
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_parameters",
    "rope_scaling",
    "sliding_window",
    "use_sliding_window",
    "vocab_size",
)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {device!r}, but CUDA is not available."
            )
        return resolved
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(
    *,
    device: torch.device,
    num_anchors: int,
    target_config_path: str | Path,
) -> tuple[Qwen3MySpecModel, object, Qwen3Config]:
    """Build the draft model from the real target and checked-in MySpec configs."""
    cfg = load_config(MYSPEC_CONFIG_PATH)
    model_args = cfg.model.copy()

    # The checked-in value (512) is appropriate for training, but would create
    # 512 * block_size draft positions in this smoke test. Keep the algorithmic
    # setting while shrinking only the amount of synthetic work.
    model_args.num_anchors = int(num_anchors)

    target_config_path = Path(target_config_path).expanduser().resolve()
    if not target_config_path.is_file():
        raise FileNotFoundError(
            f"Qwen3 target config does not exist: {target_config_path}"
        )
    target_config = Qwen3Config.from_json_file(str(target_config_path))
    draft_config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
    )

    for field_name in INHERITED_TARGET_CONFIG_FIELDS:
        assert getattr(draft_config, field_name) == getattr(
            target_config, field_name
        ), f"draft config did not inherit target field {field_name!r}"
    assert draft_config.num_target_layers == target_config.num_hidden_layers
    assert draft_config.num_hidden_layers == int(model_args.num_draft_layers)
    assert draft_config.num_attention_heads % draft_config.num_key_value_heads == 0

    # Latent CoT settings come from myspec_qwen3_4b.py rather than the target
    # model config, so check all three explicitly before model construction.
    assert draft_config.num_latent_layers == int(model_args.num_latent_layers)
    assert draft_config.num_latent_tokens == int(model_args.num_latent_tokens)
    assert draft_config.latent_token_id == int(model_args.latent_token_id)
    assert 0 <= draft_config.latent_token_id < draft_config.vocab_size

    model = Qwen3MySpecModel(draft_config).to(device=device)
    model.eval()
    assert len(model.latent_layers) == draft_config.num_latent_layers
    return model, model_args, target_config


def _print_tensor(name: str, value: torch.Tensor | None) -> None:
    if value is None:
        print(f"{name}: None", flush=True)
        return
    summary = (
        f"shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"device={value.device}"
    )
    if value.dtype == torch.bool:
        summary += f", true_count={int(value.sum())}"
    elif value.numel() > 0:
        flat = value.detach().reshape(-1)
        summary += f", sample={flat[: min(5, flat.numel())].cpu().tolist()}"
    print(f"{name}: {summary}", flush=True)


def run_forward_smoke_test(
    *,
    device_name: str | None = None,
    batch_size: int = 1,
    seq_len: int = 16,
    num_anchors: int = 2,
    seed: int = 42,
    target_config_path: str | Path = TARGET_CONFIG_PATH,
):
    """Construct synthetic inputs, call ``Qwen3MySpecModel.forward``, and check it."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if num_anchors < 1:
        raise ValueError("num_anchors must be >= 1")

    device = _resolve_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, model_args, target_config = _build_model(
        device=device,
        num_anchors=num_anchors,
        target_config_path=target_config_path,
    )
    config = model.config

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    # All non-padding positions participate, ensuring enough valid anchors.
    loss_mask = torch.ones(
        batch_size,
        seq_len,
        dtype=torch.float32,
        device=device,
    )
    target_hidden_states = torch.randn(
        batch_size,
        seq_len,
        len(config.target_layer_ids) * config.hidden_size,
        device=device,
    )
    target_last_hidden_states = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        device=device,
    )

    print(f"target config source: {Path(target_config_path).resolve()}", flush=True)
    print(f"MySpec config source: {MYSPEC_CONFIG_PATH}", flush=True)
    print(
        "Qwen3-4B target settings: "
        f"hidden_size={target_config.hidden_size}, "
        f"intermediate_size={target_config.intermediate_size}, "
        f"target_layers={target_config.num_hidden_layers}, "
        f"attention_heads={target_config.num_attention_heads}, "
        f"kv_heads={target_config.num_key_value_heads}, "
        f"head_dim={target_config.head_dim}, "
        f"vocab_size={target_config.vocab_size}",
        flush=True,
    )
    print(
        "MySpec settings: "
        f"target={model_args.target_model_name_or_path}, "
        f"block_size={config.block_size}, "
        f"draft_layers={config.num_hidden_layers}, "
        f"target_layer_ids={config.target_layer_ids}, "
        f"test_num_anchors={config.num_anchors}, "
        f"latent_layers={config.num_latent_layers}, "
        f"latent_tokens={config.num_latent_tokens}, "
        f"latent_token_id={config.latent_token_id}",
        flush=True,
    )
    print(
        f"draft model: device={device}, hidden_size={config.hidden_size}, "
        f"vocab_size={config.vocab_size}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    _print_tensor("input_ids", input_ids)
    _print_tensor("target_hidden_states", target_hidden_states)
    _print_tensor("loss_mask", loss_mask)
    _print_tensor("target_last_hidden_states", target_last_hidden_states)

    print("\ncalling Qwen3MySpecModel.forward ...", flush=True)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )
    print("forward completed\n", flush=True)

    for output_field in fields(outputs):
        _print_tensor(output_field.name, getattr(outputs, output_field.name))

    expected_prefix = (
        batch_size,
        num_anchors,
        int(config.block_size),
    )
    assert outputs.draft_logits.shape == (*expected_prefix, config.vocab_size)
    assert outputs.target_ids.shape == expected_prefix
    assert outputs.eval_mask.shape == expected_prefix
    assert outputs.block_keep_mask.shape == (batch_size, num_anchors)
    assert outputs.aligned_target_logits is not None
    assert outputs.aligned_target_logits.shape == (
        *expected_prefix,
        config.vocab_size,
    )
    assert torch.isfinite(outputs.draft_logits).all()
    assert torch.isfinite(outputs.aligned_target_logits).all()
    print(
        "\nPASS: forward outputs have the expected shapes and finite values.",
        flush=True,
    )
    return outputs


def test_qwen3_myspec_forward() -> None:
    """Pytest entry point; use ``pytest -s`` to retain debug prints."""
    run_forward_smoke_test()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=None,
        help="Execution device, for example cpu, cuda, or cuda:1 (default: auto).",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=TARGET_CONFIG_PATH,
        help=f"Path to the Qwen3 target config (default: {TARGET_CONFIG_PATH}).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-anchors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_forward_smoke_test(
        device_name=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_anchors=args.num_anchors,
        seed=args.seed,
        target_config_path=args.target_config,
    )
