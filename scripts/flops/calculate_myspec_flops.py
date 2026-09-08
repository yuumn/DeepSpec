#!/usr/bin/env python3
"""Calculate analytical training and inference FLOPs for MySpec.

The defaults match ``config/myspec/myspec_qwen3_4b.py`` and the modified
``config/dflash/dflash_qwen3_4b.py``:

* Qwen3-4B dimensions;
* 512 training anchors and 7 draft tokens per anchor;
* 2 latent layers, 2 latent tokens, and 3 mask/draft layers for MySpec;
* 5 draft layers for DFlash;
* frozen embedding and language-model head;
* cached target hidden states during draft-model training.

FLOP convention
---------------
One multiplication and one addition count as two FLOPs.  The reported totals
cover the dominant matrix multiplications and the QK^T / attention-value
products.  Elementwise operations (RMSNorm, RoPE, SiLU, softmax, L1/CE loss,
mask construction, and sampling) and optimizer updates are deliberately not
included because their FLOP definitions are implementation dependent and they
are small compared with the vocabulary and decoder matrix multiplications.

Notation
--------
d       hidden size
f       MLP intermediate size
dq      num_attention_heads * head_dim
dkv     num_key_value_heads * head_dim
V       vocabulary size
K       number of cached target feature layers
N       number of training anchors
B       number of draft tokens per anchor
L       number of latent tokens per anchor
Ml      number of latent layers
Mm      number of MySpec mask/draft layers
Md      number of DFlash draft layers
S       padded training sequence length
a       mean valid anchor position
C       committed context length during inference
R       uncached target-context tokens added to the draft KV cache this step
Q=N*B   training draft-token slots
Z=N*L   training latent-token slots

Training formulas
-----------------
The target-feature fusion linear receives cached, non-differentiable inputs.
It therefore has a forward GEMM and a weight-gradient GEMM, but no input
gradient GEMM:

    Fuse_train(S) = 4 * S * K * d^2

For a trainable decoder layer with q query tokens and k newly projected KV
inputs, the Q/O projections, K/V projections, and gated Qwen3 MLP cost:

    Linear_train(q, k)
      = 6 * [2*q*d*dq + 2*k*d*dkv + 3*q*d*f]

The factor 6 is 2 FLOPs/MAC times forward + input-gradient + weight-gradient.
For v logically visible keys per query, QK^T and attention-value products cost:

    Attn_train(q, v) = 12 * q * v * dq

The frozen draft LM head needs a forward GEMM and an input-gradient GEMM.  The
aligned teacher head only needs a forward GEMM.  Both are executed by the
current CE+L1 training path:

    Head_train(Q) = 4*Q*d*V + 2*Q*d*V = 6*Q*d*V

The complete MySpec training formula is:

    F_myspec_train = Fuse_train(S)
      + Ml * [Linear_train(Z, S+Z) + Attn_train(Z, a+L)]
      + Mm * [Linear_train(Q, S+Z+Q) + Attn_train(Q, a+L+B)]
      + Head_train(Q)

For comparison, the modified DFlash training formula is:

    F_dflash_train = Fuse_train(S)
      + Md * [Linear_train(Q, S+Q) + Attn_train(Q, a+B)]
      + Head_train(Q)

Inference formulas
------------------
With KV caching, only R new target-context tokens are fused/projected, while
attention still reads the full context C.  Forward-only primitives are:

    Fuse_infer(R) = 2 * R * K * d^2

    Linear_infer(q, k)
      = 2 * [2*q*d*dq + 2*k*d*dkv + 3*q*d*f]

    Attn_infer(q, v) = 4 * q * v * dq

    Head_infer(B) = 2 * B * d * V

The complete MySpec draft-proposal inference formula is:

    F_myspec_infer = Fuse_infer(R)
      + Ml * [Linear_infer(L, R+L) + Attn_infer(L, C+L)]
      + Mm * [Linear_infer(B, R+L+B) + Attn_infer(B, C+L+B)]
      + Head_infer(B)

The corresponding DFlash formula is:

    F_dflash_infer = Fuse_infer(R)
      + Md * [Linear_infer(B, R+B) + Attn_infer(B, C+B)]
      + Head_infer(B)

On the first draft call, use R=C because the draft KV cache is empty.  During
steady-state speculative decoding, R is the number of newly committed target
states since the previous draft call (normally 1 through B+1 in this code).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModelDimensions:
    """Transformer dimensions used by both the draft and target models."""

    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151_936
    num_target_layers: int = 36

    @property
    def query_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class DraftArchitecture:
    """MySpec/DFlash architectural and training-shape parameters."""

    num_feature_layers: int = 5
    num_anchors: int = 512
    block_size: int = 7
    num_latent_tokens: int = 2
    num_latent_layers: int = 2
    num_mask_layers: int = 3
    num_dflash_layers: int = 5

    @property
    def training_draft_tokens(self) -> int:
        return self.num_anchors * self.block_size

    @property
    def training_latent_tokens(self) -> int:
        return self.num_anchors * self.num_latent_tokens


def train_feature_fusion_flops(
    sequence_length: int,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> float:
    """Forward + weight gradient for cached target features; no input grad."""

    d = dims.hidden_size
    return 4.0 * sequence_length * arch.num_feature_layers * d * d


def train_decoder_linear_flops(
    query_tokens: int,
    projected_kv_tokens: int,
    dims: ModelDimensions,
) -> float:
    """Training FLOPs for Q/K/V/O and the three Qwen3 gated-MLP matrices."""

    d = dims.hidden_size
    f = dims.intermediate_size
    dq = dims.query_projection_size
    dkv = dims.kv_projection_size
    matrix_macs = (
        2 * query_tokens * d * dq
        + 2 * projected_kv_tokens * d * dkv
        + 3 * query_tokens * d * f
    )
    return 6.0 * matrix_macs


def train_attention_flops(
    query_tokens: int,
    visible_keys_per_query: float,
    dims: ModelDimensions,
) -> float:
    """Forward/backward FLOPs for QK^T and attention-value products."""

    return (
        12.0
        * query_tokens
        * visible_keys_per_query
        * dims.query_projection_size
    )


def train_head_flops(
    draft_tokens: int,
    dims: ModelDimensions,
) -> dict[str, float]:
    """Current frozen draft-head and aligned-teacher-head training costs."""

    d = dims.hidden_size
    vocab = dims.vocab_size
    return {
        "draft LM head (forward + input grad)": 4.0 * draft_tokens * d * vocab,
        "aligned target LM head (forward)": 2.0 * draft_tokens * d * vocab,
    }


def myspec_training_breakdown(
    sequence_length: int,
    mean_anchor_position: float,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> dict[str, float]:
    """Return complete dominant MySpec training FLOPs for one sample."""

    draft_tokens = arch.training_draft_tokens
    latent_tokens = arch.training_latent_tokens
    result = {
        "target-feature fusion": train_feature_fusion_flops(
            sequence_length, dims, arch
        ),
        "latent-layer linears": arch.num_latent_layers
        * train_decoder_linear_flops(
            latent_tokens,
            sequence_length + latent_tokens,
            dims,
        ),
        "latent attention": arch.num_latent_layers
        * train_attention_flops(
            latent_tokens,
            mean_anchor_position + arch.num_latent_tokens,
            dims,
        ),
        "mask-layer linears": arch.num_mask_layers
        * train_decoder_linear_flops(
            draft_tokens,
            sequence_length + latent_tokens + draft_tokens,
            dims,
        ),
        "mask attention": arch.num_mask_layers
        * train_attention_flops(
            draft_tokens,
            mean_anchor_position + arch.num_latent_tokens + arch.block_size,
            dims,
        ),
    }
    result.update(train_head_flops(draft_tokens, dims))
    return result


def dflash_training_breakdown(
    sequence_length: int,
    mean_anchor_position: float,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> dict[str, float]:
    """Return modified-DFlash training FLOPs using the same CE+L1 head path."""

    draft_tokens = arch.training_draft_tokens
    result = {
        "target-feature fusion": train_feature_fusion_flops(
            sequence_length, dims, arch
        ),
        "draft-layer linears": arch.num_dflash_layers
        * train_decoder_linear_flops(
            draft_tokens,
            sequence_length + draft_tokens,
            dims,
        ),
        "draft attention": arch.num_dflash_layers
        * train_attention_flops(
            draft_tokens,
            mean_anchor_position + arch.block_size,
            dims,
        ),
    }
    result.update(train_head_flops(draft_tokens, dims))
    return result


def infer_feature_fusion_flops(
    new_context_tokens: int,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> float:
    """Forward-only fusion for target states not already in the draft cache."""

    d = dims.hidden_size
    return 2.0 * new_context_tokens * arch.num_feature_layers * d * d


def infer_decoder_linear_flops(
    query_tokens: int,
    newly_projected_kv_tokens: int,
    dims: ModelDimensions,
) -> float:
    """Forward-only FLOPs for Q/K/V/O and the Qwen3 gated MLP."""

    d = dims.hidden_size
    f = dims.intermediate_size
    dq = dims.query_projection_size
    dkv = dims.kv_projection_size
    matrix_macs = (
        2 * query_tokens * d * dq
        + 2 * newly_projected_kv_tokens * d * dkv
        + 3 * query_tokens * d * f
    )
    return 2.0 * matrix_macs


def infer_attention_flops(
    query_tokens: int,
    visible_keys_per_query: float,
    dims: ModelDimensions,
) -> float:
    """Forward-only FLOPs for QK^T and attention-value products."""

    return (
        4.0
        * query_tokens
        * visible_keys_per_query
        * dims.query_projection_size
    )


def infer_head_flops(draft_tokens: int, dims: ModelDimensions) -> float:
    """Forward-only frozen LM-head projection for one draft proposal."""

    return 2.0 * draft_tokens * dims.hidden_size * dims.vocab_size


def myspec_inference_breakdown(
    context_length: int,
    new_context_tokens: int,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> dict[str, float]:
    """Return MySpec draft-model inference FLOPs for one proposal."""

    latent_tokens = arch.num_latent_tokens
    draft_tokens = arch.block_size
    return {
        "target-feature fusion": infer_feature_fusion_flops(
            new_context_tokens, dims, arch
        ),
        "latent-layer linears": arch.num_latent_layers
        * infer_decoder_linear_flops(
            latent_tokens,
            new_context_tokens + latent_tokens,
            dims,
        ),
        "latent attention": arch.num_latent_layers
        * infer_attention_flops(
            latent_tokens,
            context_length + latent_tokens,
            dims,
        ),
        "mask-layer linears": arch.num_mask_layers
        * infer_decoder_linear_flops(
            draft_tokens,
            new_context_tokens + latent_tokens + draft_tokens,
            dims,
        ),
        "mask attention": arch.num_mask_layers
        * infer_attention_flops(
            draft_tokens,
            context_length + latent_tokens + draft_tokens,
            dims,
        ),
        "draft LM head (forward)": infer_head_flops(draft_tokens, dims),
    }


def dflash_inference_breakdown(
    context_length: int,
    new_context_tokens: int,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> dict[str, float]:
    """Return DFlash draft-model inference FLOPs for one proposal."""

    draft_tokens = arch.block_size
    return {
        "target-feature fusion": infer_feature_fusion_flops(
            new_context_tokens, dims, arch
        ),
        "draft-layer linears": arch.num_dflash_layers
        * infer_decoder_linear_flops(
            draft_tokens,
            new_context_tokens + draft_tokens,
            dims,
        ),
        "draft attention": arch.num_dflash_layers
        * infer_attention_flops(
            draft_tokens,
            context_length + draft_tokens,
            dims,
        ),
        "draft LM head (forward)": infer_head_flops(draft_tokens, dims),
    }


def target_verification_flops(
    context_length: int,
    dims: ModelDimensions,
    arch: DraftArchitecture,
) -> float:
    """Qwen target-model forward FLOPs for verifying one full proposal."""

    verify_tokens = arch.block_size + 1
    per_layer_linears = infer_decoder_linear_flops(
        verify_tokens,
        verify_tokens,
        dims,
    )
    # Each new target token sees the complete past plus its causal prefix in
    # the current verification block.
    visible_pairs = (
        verify_tokens * context_length
        + verify_tokens * (verify_tokens + 1) / 2.0
    )
    per_layer_attention = (
        4.0 * visible_pairs * dims.query_projection_size
    )
    lm_head = 2.0 * verify_tokens * dims.hidden_size * dims.vocab_size
    return dims.num_target_layers * (
        per_layer_linears + per_layer_attention
    ) + lm_head


def total(breakdown: Mapping[str, float]) -> float:
    return sum(breakdown.values())


def reduction_percent(baseline: float, candidate: float) -> float:
    return 100.0 * (1.0 - candidate / baseline)


def human_flops(value: float) -> str:
    for scale, suffix in (
        (1e15, "PFLOPs"),
        (1e12, "TFLOPs"),
        (1e9, "GFLOPs"),
        (1e6, "MFLOPs"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:,.6f} {suffix}"
    return f"{value:,.0f} FLOPs"


def print_breakdown(
    title: str,
    breakdown: Mapping[str, float],
    token_divisor: int,
    token_label: str,
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for name, value in breakdown.items():
        print(
            f"{name:<42} {human_flops(value):>20}"
            f"  ({human_flops(value / token_divisor)}/{token_label})"
        )
    value = total(breakdown)
    print(f"{'TOTAL':<42} {human_flops(value):>20}")
    print(f"{'TOTAL per ' + token_label:<42} {human_flops(value / token_divisor):>20}")


def print_formulas() -> None:
    print("FLOP formulas")
    print("==============")
    print("Fuse_train(S)       = 4*S*K*d^2")
    print("Linear_train(q,k)   = 6*(2*q*d*dq + 2*k*d*dkv + 3*q*d*f)")
    print("Attn_train(q,v)     = 12*q*v*dq")
    print("Head_train(Q)       = 6*Q*d*V")
    print("F_myspec_train      = Fuse_train(S)")
    print("  + Ml*[Linear_train(Z,S+Z) + Attn_train(Z,a+L)]")
    print("  + Mm*[Linear_train(Q,S+Z+Q) + Attn_train(Q,a+L+B)]")
    print("  + Head_train(Q)")
    print()
    print("Fuse_infer(R)       = 2*R*K*d^2")
    print("Linear_infer(q,k)   = 2*(2*q*d*dq + 2*k*d*dkv + 3*q*d*f)")
    print("Attn_infer(q,v)     = 4*q*v*dq")
    print("Head_infer(B)       = 2*B*d*V")
    print("F_myspec_infer      = Fuse_infer(R)")
    print("  + Ml*[Linear_infer(L,R+L) + Attn_infer(L,C+L)]")
    print("  + Mm*[Linear_infer(B,R+L+B) + Attn_infer(B,C+L+B)]")
    print("  + Head_infer(B)")


def add_positive_int_argument(
    parser: argparse.ArgumentParser,
    *flags: str,
    default: int,
    help_text: str,
    dest: str | None = None,
) -> None:
    def positive_int(raw: str) -> int:
        value = int(raw)
        if value <= 0:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return value

    parser.add_argument(
        *flags,
        dest=dest,
        type=positive_int,
        default=default,
        help=help_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate MySpec training/inference FLOPs and compare against "
            "the modified DFlash baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_positive_int_argument(
        parser,
        "--num-latent-tokens",
        "--num_latent_tokens",
        "--num_latent_token",
        dest="num_latent_tokens",
        default=2,
        help_text="latent tokens per anchor/proposal",
    )
    add_positive_int_argument(
        parser,
        "--training-sequence-length",
        default=4096,
        help_text="padded training sequence length",
    )
    parser.add_argument(
        "--mean-anchor-position",
        type=float,
        default=None,
        help=(
            "mean valid training-anchor index; omitted means the expectation "
            "for uniform candidates, (S-2)/2"
        ),
    )
    add_positive_int_argument(
        parser,
        "--context-length",
        default=4096,
        help_text="committed inference context length C",
    )
    add_positive_int_argument(
        parser,
        "--new-context-tokens",
        default=1,
        help_text="uncached context tokens R in a steady-state draft step",
    )
    add_positive_int_argument(
        parser,
        "--num-anchors",
        default=512,
        help_text="training anchors N",
    )
    add_positive_int_argument(
        parser,
        "--block-size",
        default=7,
        help_text="draft tokens B per anchor/proposal",
    )
    add_positive_int_argument(
        parser,
        "--num-feature-layers",
        default=5,
        help_text="cached target feature layers K",
    )
    add_positive_int_argument(
        parser,
        "--num-latent-layers",
        default=2,
        help_text="MySpec latent layers Ml",
    )
    add_positive_int_argument(
        parser,
        "--num-mask-layers",
        default=3,
        help_text="MySpec mask/draft layers Mm",
    )
    add_positive_int_argument(
        parser,
        "--num-dflash-layers",
        default=5,
        help_text="DFlash draft layers Md",
    )
    add_positive_int_argument(
        parser,
        "--hidden-size",
        default=2560,
        help_text="model hidden size d",
    )
    add_positive_int_argument(
        parser,
        "--intermediate-size",
        default=9728,
        help_text="MLP intermediate size f",
    )
    add_positive_int_argument(
        parser,
        "--num-attention-heads",
        default=32,
        help_text="query attention heads",
    )
    add_positive_int_argument(
        parser,
        "--num-key-value-heads",
        default=8,
        help_text="key/value attention heads",
    )
    add_positive_int_argument(
        parser,
        "--head-dim",
        default=128,
        help_text="attention head dimension",
    )
    add_positive_int_argument(
        parser,
        "--vocab-size",
        default=151_936,
        help_text="vocabulary size V",
    )
    add_positive_int_argument(
        parser,
        "--num-target-layers",
        default=36,
        help_text="target-model layers for end-to-end verification",
    )
    parser.add_argument(
        "--hide-formulas",
        action="store_true",
        help="do not print the symbolic formulas",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    max_anchor = args.training_sequence_length - 2
    if max_anchor < 0:
        raise ValueError("training-sequence-length must be at least 2")
    if args.mean_anchor_position is not None and not (
        0.0 <= args.mean_anchor_position <= max_anchor
    ):
        raise ValueError(
            "mean-anchor-position must lie in [0, training_sequence_length-2]"
        )
    if args.new_context_tokens > args.context_length:
        raise ValueError("new-context-tokens cannot exceed context-length")


def main() -> None:
    args = parse_args()
    validate_args(args)

    dims = ModelDimensions(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        num_target_layers=args.num_target_layers,
    )
    arch = DraftArchitecture(
        num_feature_layers=args.num_feature_layers,
        num_anchors=args.num_anchors,
        block_size=args.block_size,
        num_latent_tokens=args.num_latent_tokens,
        num_latent_layers=args.num_latent_layers,
        num_mask_layers=args.num_mask_layers,
        num_dflash_layers=args.num_dflash_layers,
    )
    mean_anchor = (
        args.mean_anchor_position
        if args.mean_anchor_position is not None
        else (args.training_sequence_length - 2) / 2.0
    )

    if not args.hide_formulas:
        print_formulas()

    print("\nResolved parameters")
    print("===================")
    print(
        f"d={dims.hidden_size}, f={dims.intermediate_size}, "
        f"dq={dims.query_projection_size}, dkv={dims.kv_projection_size}, "
        f"V={dims.vocab_size}"
    )
    print(
        f"N={arch.num_anchors}, B={arch.block_size}, "
        f"L={arch.num_latent_tokens}, Ml={arch.num_latent_layers}, "
        f"Mm={arch.num_mask_layers}, Md={arch.num_dflash_layers}"
    )
    print(
        f"S={args.training_sequence_length}, a={mean_anchor:g}, "
        f"C={args.context_length}, R={args.new_context_tokens}"
    )
    print(
        f"Q={arch.training_draft_tokens} training draft slots, "
        f"Z={arch.training_latent_tokens} training latent slots"
    )

    myspec_train = myspec_training_breakdown(
        args.training_sequence_length,
        mean_anchor,
        dims,
        arch,
    )
    dflash_train = dflash_training_breakdown(
        args.training_sequence_length,
        mean_anchor,
        dims,
        arch,
    )
    train_token_count = arch.training_draft_tokens
    print_breakdown(
        "MySpec training FLOPs per sample",
        myspec_train,
        train_token_count,
        "draft token",
    )
    dflash_train_total = total(dflash_train)
    myspec_train_total = total(myspec_train)
    print("\nTraining comparison with modified DFlash")
    print("----------------------------------------")
    print(f"DFlash total                 {human_flops(dflash_train_total)}")
    print(f"MySpec total                 {human_flops(myspec_train_total)}")
    print(
        "DFlash per draft token       "
        f"{human_flops(dflash_train_total / train_token_count)}"
    )
    print(
        "MySpec per draft token       "
        f"{human_flops(myspec_train_total / train_token_count)}"
    )
    print(
        "MySpec reduction             "
        f"{reduction_percent(dflash_train_total, myspec_train_total):.4f}%"
    )
    train_head = sum(
        value for name, value in myspec_train.items() if "LM head" in name
    )
    print(
        "Backbone-only reduction      "
        f"{reduction_percent(dflash_train_total - train_head, myspec_train_total - train_head):.4f}%"
    )

    myspec_infer = myspec_inference_breakdown(
        args.context_length,
        args.new_context_tokens,
        dims,
        arch,
    )
    dflash_infer = dflash_inference_breakdown(
        args.context_length,
        args.new_context_tokens,
        dims,
        arch,
    )
    print_breakdown(
        "MySpec steady-state draft inference FLOPs per proposal",
        myspec_infer,
        arch.block_size,
        "draft token",
    )
    dflash_infer_total = total(dflash_infer)
    myspec_infer_total = total(myspec_infer)
    print("\nSteady-state inference comparison with DFlash")
    print("---------------------------------------------")
    print(f"DFlash per proposal           {human_flops(dflash_infer_total)}")
    print(f"MySpec per proposal           {human_flops(myspec_infer_total)}")
    print(
        "DFlash per draft token       "
        f"{human_flops(dflash_infer_total / arch.block_size)}"
    )
    print(
        "MySpec per draft token       "
        f"{human_flops(myspec_infer_total / arch.block_size)}"
    )
    print(
        "MySpec reduction             "
        f"{reduction_percent(dflash_infer_total, myspec_infer_total):.4f}%"
    )
    infer_head = infer_head_flops(arch.block_size, dims)
    print(
        "Backbone-only reduction      "
        f"{reduction_percent(dflash_infer_total - infer_head, myspec_infer_total - infer_head):.4f}%"
    )

    myspec_first = total(
        myspec_inference_breakdown(
            args.context_length,
            args.context_length,
            dims,
            arch,
        )
    )
    dflash_first = total(
        dflash_inference_breakdown(
            args.context_length,
            args.context_length,
            dims,
            arch,
        )
    )
    print("\nFirst draft-cache fill (R=C)")
    print("----------------------------")
    print(f"DFlash per proposal           {human_flops(dflash_first)}")
    print(f"MySpec per proposal           {human_flops(myspec_first)}")
    print(
        "MySpec reduction             "
        f"{reduction_percent(dflash_first, myspec_first):.4f}%"
    )

    target_verify = target_verification_flops(
        args.context_length,
        dims,
        arch,
    )
    dflash_e2e = dflash_infer_total + target_verify
    myspec_e2e = myspec_infer_total + target_verify
    print("\nEnd-to-end speculative iteration")
    print("--------------------------------")
    print(f"Shared target verification    {human_flops(target_verify)}")
    print(f"DFlash + target               {human_flops(dflash_e2e)}")
    print(f"MySpec + target               {human_flops(myspec_e2e)}")
    print(
        "MySpec reduction             "
        f"{reduction_percent(dflash_e2e, myspec_e2e):.4f}%"
    )
    print(
        "Note: this per-iteration result does not include fewer verification "
        "rounds from a higher acceptance rate."
    )


if __name__ == "__main__":
    main()
