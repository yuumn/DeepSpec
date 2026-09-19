#!/usr/bin/env python3
"""Shared analytical FLOP model for the Qwen3 MySpec/DSpark variants.

The implementation mirrors the matrix multiplications in the current model
and evaluation code.  A multiplication and an addition are counted as two
FLOPs.  Elementwise operations, reductions, sampling, cache copies, and the
optimizer update are intentionally excluded; see ``FLOPS_COMPARISON.md`` for
the complete scope and assumptions.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModelDimensions:
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

    @classmethod
    def from_json(cls, path: Path) -> "ModelDimensions":
        payload = json.loads(path.read_text(encoding="utf-8"))
        hidden_size = int(payload["hidden_size"])
        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(payload["intermediate_size"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(
                payload.get(
                    "head_dim",
                    hidden_size // int(payload["num_attention_heads"]),
                )
            ),
            vocab_size=int(payload["vocab_size"]),
            num_target_layers=int(payload["num_hidden_layers"]),
        )


@dataclass(frozen=True)
class DraftArchitecture:
    name: str
    kind: str
    num_feature_layers: int
    num_anchors: int
    block_size: int
    num_draft_layers: int
    num_latent_layers: int = 0
    num_latent_tokens: int = 0
    markov_rank: int = 0
    confidence_enabled: bool = False
    confidence_with_markov: bool = False

    @property
    def training_draft_tokens(self) -> int:
        return self.num_anchors * self.block_size

    @property
    def training_latent_tokens(self) -> int:
        return self.num_anchors * self.num_latent_tokens

    @property
    def has_markov(self) -> bool:
        return self.markov_rank > 0


def _literal(node: ast.AST):
    """Evaluate the literal-only subset used by the experiment configs."""

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id != "dict" or node.args:
            raise ValueError("only dict(...) calls with keyword arguments are supported")
        return {keyword.arg: _literal(keyword.value) for keyword in node.keywords}
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal(key): _literal(value)
            for key, value in zip(node.keys, node.values)
        }
    return ast.literal_eval(node)


def load_config_assignment(path: Path, assignment: str) -> dict[str, object]:
    """Load a top-level literal ``dict`` assignment without importing config."""

    source = path.read_text(encoding="utf-8")
    match = re.search(rf"(?m)^{re.escape(assignment)}\s*=", source)
    if match is None:
        raise ValueError(f"could not find assignment {assignment!r} in {path}")

    # Parse only the requested expression.  Some historical experiment configs
    # contain unrelated syntax errors later in finalize_cfg; importing or
    # parsing the whole module would make their otherwise valid model dict
    # unusable for analysis.
    expression_start = match.end()
    expression = _extract_assignment_expression(source, expression_start)
    parsed = ast.parse(expression, filename=str(path), mode="eval")
    loaded = _literal(parsed.body)
    if not isinstance(loaded, dict):
        raise ValueError(f"{assignment!r} in {path} is not a dictionary")
    return loaded


def _extract_assignment_expression(source: str, start: int) -> str:
    index = start
    while index < len(source) and source[index].isspace():
        index += 1
    expression_start = index
    stack: list[str] = []
    quote: str | None = None
    triple_quoted = False
    escaped = False
    in_comment = False
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = set(opening.values())

    while index < len(source):
        char = source[index]
        if in_comment:
            if char == "\n":
                in_comment = False
                if not stack:
                    return source[expression_start:index].strip()
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if triple_quoted:
                if source.startswith(quote * 3, index):
                    index += 3
                    quote = None
                    triple_quoted = False
                    continue
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "#":
            in_comment = True
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            triple_quoted = source.startswith(char * 3, index)
            index += 3 if triple_quoted else 1
            continue
        if char in opening:
            stack.append(opening[char])
        elif char in closing:
            if not stack or char != stack.pop():
                raise ValueError("unbalanced delimiters in config assignment")
            if not stack:
                return source[expression_start : index + 1].strip()
        elif char == "\n" and not stack:
            return source[expression_start:index].strip()
        index += 1
    if quote is not None or stack:
        raise ValueError("unterminated config assignment expression")
    return source[expression_start:].strip()


def architecture_from_config(
    path: Path,
    *,
    name: str,
    kind: str,
) -> DraftArchitecture:
    model = load_config_assignment(path, "model")
    target_layer_ids = model["target_layer_ids"]
    if not isinstance(target_layer_ids, list):
        raise ValueError(f"target_layer_ids in {path} must be a list")
    confidence_enabled = float(model.get("confidence_head_alpha", 0.0)) > 0.0
    architecture = DraftArchitecture(
        name=name,
        kind=kind,
        num_feature_layers=len(target_layer_ids),
        num_anchors=int(model["num_anchors"]),
        block_size=int(model["block_size"]),
        num_draft_layers=int(model["num_draft_layers"]),
        num_latent_layers=int(model.get("num_latent_layers", 0)),
        num_latent_tokens=int(model.get("num_latent_tokens", 0)),
        markov_rank=int(model.get("markov_rank", 0)),
        confidence_enabled=confidence_enabled,
        confidence_with_markov=bool(
            model.get("confidence_head_with_markov", False)
        ),
    )
    validate_architecture(architecture)
    return architecture


def validate_architecture(architecture: DraftArchitecture) -> None:
    if architecture.kind not in {"myspec", "dspark"}:
        raise ValueError(f"unsupported architecture kind: {architecture.kind}")
    positive = {
        "num_feature_layers": architecture.num_feature_layers,
        "num_anchors": architecture.num_anchors,
        "block_size": architecture.block_size,
        "num_draft_layers": architecture.num_draft_layers,
    }
    for field, value in positive.items():
        if value <= 0:
            raise ValueError(f"{architecture.name}: {field} must be positive")
    if architecture.kind == "myspec":
        if architecture.num_latent_layers <= 0 or architecture.num_latent_tokens <= 0:
            raise ValueError(
                f"{architecture.name}: MySpec requires positive latent layers/tokens"
            )
    elif architecture.num_latent_layers or architecture.num_latent_tokens:
        raise ValueError(f"{architecture.name}: DSpark/DFlash cannot have latent layers")
    if architecture.markov_rank < 0:
        raise ValueError(f"{architecture.name}: markov_rank must be non-negative")
    if architecture.confidence_with_markov and not architecture.has_markov:
        raise ValueError(
            f"{architecture.name}: confidence_with_markov requires a Markov head"
        )


def validate_comparison(
    candidate: DraftArchitecture,
    baseline: DraftArchitecture,
) -> None:
    fields = ("num_feature_layers", "num_anchors", "block_size")
    mismatches = [
        field
        for field in fields
        if getattr(candidate, field) != getattr(baseline, field)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: {getattr(candidate, field)} != {getattr(baseline, field)}"
            for field in mismatches
        )
        raise ValueError(f"comparison configs are incompatible ({details})")


def _linear_forward(tokens: int, input_size: int, output_size: int) -> float:
    return 2.0 * tokens * input_size * output_size


def _linear_training(
    tokens: int,
    input_size: int,
    output_size: int,
    *,
    input_gradient: bool,
) -> float:
    # Forward + weight gradient, plus input gradient when the source requires it.
    multiplier = 6.0 if input_gradient else 4.0
    return multiplier * tokens * input_size * output_size


def feature_fusion_training_flops(
    sequence_length: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
) -> float:
    # Cached target features are detached: forward + weight gradient, no dInput.
    return _linear_training(
        sequence_length,
        architecture.num_feature_layers * dims.hidden_size,
        dims.hidden_size,
        input_gradient=False,
    )


def feature_fusion_inference_flops(
    new_context_tokens: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
) -> float:
    return _linear_forward(
        new_context_tokens,
        architecture.num_feature_layers * dims.hidden_size,
        dims.hidden_size,
    )


def decoder_layer_training_linear_flops(
    *,
    query_tokens: int,
    external_kv_tokens: Sequence[int],
    dims: ModelDimensions,
    first_layer_in_stack: bool,
) -> float:
    """Exact GEMMs for one trainable Qwen draft decoder layer.

    ``query_tokens`` supply Q and their own K/V.  Every entry in
    ``external_kv_tokens`` is projected by a separate K/V pair (target context,
    and for MySpec mask layers, latent states).  The initial noise embedding is
    frozen, so the first layer of each stack omits only its Q/K/V input-gradient
    GEMMs; all weight gradients and the complete attention/MLP backward remain.
    """

    d = dims.hidden_size
    f = dims.intermediate_size
    dq = dims.query_projection_size
    dkv = dims.kv_projection_size
    query_input_gradient = not first_layer_in_stack
    total = 0.0
    total += _linear_training(
        query_tokens, d, dq, input_gradient=query_input_gradient
    )
    total += _linear_training(query_tokens, dq, d, input_gradient=True)
    total += 2.0 * _linear_training(
        query_tokens, d, dkv, input_gradient=query_input_gradient
    )
    for token_count in external_kv_tokens:
        total += 2.0 * _linear_training(
            token_count, d, dkv, input_gradient=True
        )
    # Qwen3 gated MLP: gate_proj, up_proj, down_proj.
    total += 3.0 * _linear_training(
        query_tokens, d, f, input_gradient=True
    )
    return total


def decoder_layer_inference_linear_flops(
    *,
    query_tokens: int,
    external_kv_tokens: Sequence[int],
    dims: ModelDimensions,
) -> float:
    d = dims.hidden_size
    f = dims.intermediate_size
    dq = dims.query_projection_size
    dkv = dims.kv_projection_size
    total = 0.0
    total += _linear_forward(query_tokens, d, dq)
    total += _linear_forward(query_tokens, dq, d)
    total += 2.0 * _linear_forward(query_tokens, d, dkv)
    for token_count in external_kv_tokens:
        total += 2.0 * _linear_forward(token_count, d, dkv)
    total += 3.0 * _linear_forward(query_tokens, d, f)
    return total


def attention_training_flops(visible_query_key_pairs: float, dims: ModelDimensions) -> float:
    # QK^T + AV, each with forward and two backward GEMMs.
    return 12.0 * visible_query_key_pairs * dims.query_projection_size


def attention_inference_flops(visible_query_key_pairs: float, dims: ModelDimensions) -> float:
    # Forward QK^T + AV.
    return 4.0 * visible_query_key_pairs * dims.query_projection_size


def output_heads_training_breakdown(
    tokens: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
) -> dict[str, float]:
    d = dims.hidden_size
    vocab = dims.vocab_size
    result = {
        # Frozen head: forward + gradient into the draft hidden states.
        "draft LM head": 4.0 * tokens * d * vocab,
        # Cached aligned target state and frozen head: forward only.
        "aligned target LM head": 2.0 * tokens * d * vocab,
    }
    if architecture.has_markov:
        # Trainable rank->vocab projection: forward + dWeight + dInput.
        result["Markov head"] = (
            6.0 * tokens * architecture.markov_rank * vocab
        )
    if architecture.confidence_enabled:
        input_size = d
        if architecture.confidence_with_markov:
            input_size += architecture.markov_rank
        result["confidence head"] = 6.0 * tokens * input_size
    return result


def output_heads_inference_breakdown(
    tokens: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
) -> dict[str, float]:
    d = dims.hidden_size
    vocab = dims.vocab_size
    result = {"draft LM head": 2.0 * tokens * d * vocab}
    if architecture.has_markov:
        result["Markov head"] = (
            2.0 * tokens * architecture.markov_rank * vocab
        )
    if architecture.confidence_enabled:
        input_size = d
        if architecture.confidence_with_markov:
            input_size += architecture.markov_rank
        result["confidence head"] = 2.0 * tokens * input_size
    return result


def _active_anchor_expectation(
    sequence_length: int,
    architecture: DraftArchitecture,
    mean_anchor_position: float | None,
) -> tuple[int, float]:
    if sequence_length < 2:
        raise ValueError("sequence length must be at least 2")
    active_anchors = min(architecture.num_anchors, sequence_length - 1)
    mean_anchor = (
        (sequence_length - 2) / 2.0
        if mean_anchor_position is None
        else float(mean_anchor_position)
    )
    if not 0.0 <= mean_anchor <= sequence_length - 2:
        raise ValueError(
            f"mean anchor position must lie in [0, {sequence_length - 2}]"
        )
    return active_anchors, mean_anchor


def training_breakdown(
    sequence_length: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
    *,
    mean_anchor_position: float | None = None,
) -> dict[str, float]:
    validate_architecture(architecture)
    active_anchors, mean_anchor = _active_anchor_expectation(
        sequence_length,
        architecture,
        mean_anchor_position,
    )
    draft_tokens = architecture.training_draft_tokens
    result: dict[str, float] = {
        "target-feature fusion": feature_fusion_training_flops(
            sequence_length, dims, architecture
        )
    }
    if architecture.kind == "myspec":
        latent_tokens = architecture.training_latent_tokens
        latent_linears = 0.0
        for layer_index in range(architecture.num_latent_layers):
            latent_linears += decoder_layer_training_linear_flops(
                query_tokens=latent_tokens,
                external_kv_tokens=(sequence_length,),
                dims=dims,
                first_layer_in_stack=layer_index == 0,
            )
        latent_pairs = (
            active_anchors
            * architecture.num_latent_tokens
            * (mean_anchor + architecture.num_latent_tokens)
        )
        result["latent-layer linears"] = latent_linears
        result["latent attention"] = (
            architecture.num_latent_layers
            * attention_training_flops(latent_pairs, dims)
        )

        mask_linears = 0.0
        for layer_index in range(architecture.num_draft_layers):
            mask_linears += decoder_layer_training_linear_flops(
                query_tokens=draft_tokens,
                external_kv_tokens=(sequence_length, latent_tokens),
                dims=dims,
                first_layer_in_stack=layer_index == 0,
            )
        mask_pairs = (
            active_anchors
            * architecture.block_size
            * (
                mean_anchor
                + architecture.num_latent_tokens
                + architecture.block_size
            )
        )
        result["mask-layer linears"] = mask_linears
        result["mask attention"] = (
            architecture.num_draft_layers
            * attention_training_flops(mask_pairs, dims)
        )
    else:
        draft_linears = 0.0
        for layer_index in range(architecture.num_draft_layers):
            draft_linears += decoder_layer_training_linear_flops(
                query_tokens=draft_tokens,
                external_kv_tokens=(sequence_length,),
                dims=dims,
                first_layer_in_stack=layer_index == 0,
            )
        draft_pairs = (
            active_anchors
            * architecture.block_size
            * (mean_anchor + architecture.block_size)
        )
        result["draft-layer linears"] = draft_linears
        result["draft attention"] = (
            architecture.num_draft_layers
            * attention_training_flops(draft_pairs, dims)
        )
    result.update(output_heads_training_breakdown(draft_tokens, dims, architecture))
    return result


def inference_breakdown(
    context_length: int,
    new_context_tokens: int,
    dims: ModelDimensions,
    architecture: DraftArchitecture,
) -> dict[str, float]:
    validate_architecture(architecture)
    if context_length <= 0:
        raise ValueError("context length must be positive")
    if not 0 < new_context_tokens <= context_length:
        raise ValueError("new context tokens must lie in [1, context length]")
    block_size = architecture.block_size
    result: dict[str, float] = {
        "target-feature fusion": feature_fusion_inference_flops(
            new_context_tokens, dims, architecture
        )
    }
    if architecture.kind == "myspec":
        latent_tokens = architecture.num_latent_tokens
        latent_linears = architecture.num_latent_layers * (
            decoder_layer_inference_linear_flops(
                query_tokens=latent_tokens,
                external_kv_tokens=(new_context_tokens,),
                dims=dims,
            )
        )
        latent_pairs = latent_tokens * (context_length + latent_tokens)
        result["latent-layer linears"] = latent_linears
        result["latent attention"] = (
            architecture.num_latent_layers
            * attention_inference_flops(latent_pairs, dims)
        )
        mask_linears = architecture.num_draft_layers * (
            decoder_layer_inference_linear_flops(
                query_tokens=block_size,
                external_kv_tokens=(new_context_tokens, latent_tokens),
                dims=dims,
            )
        )
        mask_pairs = block_size * (
            context_length + latent_tokens + block_size
        )
        result["mask-layer linears"] = mask_linears
        result["mask attention"] = (
            architecture.num_draft_layers
            * attention_inference_flops(mask_pairs, dims)
        )
    else:
        result["draft-layer linears"] = architecture.num_draft_layers * (
            decoder_layer_inference_linear_flops(
                query_tokens=block_size,
                external_kv_tokens=(new_context_tokens,),
                dims=dims,
            )
        )
        draft_pairs = block_size * (context_length + block_size)
        result["draft attention"] = (
            architecture.num_draft_layers
            * attention_inference_flops(draft_pairs, dims)
        )
    result.update(output_heads_inference_breakdown(block_size, dims, architecture))
    return result


def target_verification_flops(
    context_length: int,
    verify_tokens: int,
    dims: ModelDimensions,
) -> float:
    if context_length <= 0 or verify_tokens <= 0:
        raise ValueError("context length and verify tokens must be positive")
    per_layer_linears = decoder_layer_inference_linear_flops(
        query_tokens=verify_tokens,
        external_kv_tokens=(),
        dims=dims,
    )
    visible_pairs = (
        verify_tokens * context_length
        + verify_tokens * (verify_tokens + 1) / 2.0
    )
    per_layer_attention = attention_inference_flops(visible_pairs, dims)
    lm_head = _linear_forward(
        verify_tokens, dims.hidden_size, dims.vocab_size
    )
    return dims.num_target_layers * (
        per_layer_linears + per_layer_attention
    ) + lm_head


def total(breakdown: Mapping[str, float]) -> float:
    return float(sum(breakdown.values()))


def reduction_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline FLOPs must be positive")
    return 100.0 * (1.0 - candidate / baseline)


def human_flops(value: float) -> str:
    for scale, suffix in (
        (1e18, "EFLOPs"),
        (1e15, "PFLOPs"),
        (1e12, "TFLOPs"),
        (1e9, "GFLOPs"),
        (1e6, "MFLOPs"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:,.3f} {suffix}"
    return f"{value:,.0f} FLOPs"


def comparison_rows(
    *,
    contexts: Sequence[int],
    dims: ModelDimensions,
    candidate: DraftArchitecture,
    baseline: DraftArchitecture,
    new_context_tokens: int,
) -> list[dict[str, float | int]]:
    validate_comparison(candidate, baseline)
    rows: list[dict[str, float | int]] = []
    for context in contexts:
        if context < new_context_tokens:
            raise ValueError(
                f"context {context} is shorter than new_context_tokens={new_context_tokens}"
            )
        candidate_train = total(training_breakdown(context, dims, candidate))
        baseline_train = total(training_breakdown(context, dims, baseline))
        candidate_infer = total(
            inference_breakdown(context, new_context_tokens, dims, candidate)
        )
        baseline_infer = total(
            inference_breakdown(context, new_context_tokens, dims, baseline)
        )
        target_verify = target_verification_flops(
            context,
            candidate.block_size + 1,
            dims,
        )
        candidate_first = total(
            inference_breakdown(context, context, dims, candidate)
        )
        baseline_first = total(
            inference_breakdown(context, context, dims, baseline)
        )
        rows.append(
            {
                "context": context,
                "candidate_train": candidate_train,
                "baseline_train": baseline_train,
                "train_reduction": reduction_percent(
                    baseline_train, candidate_train
                ),
                "candidate_infer": candidate_infer,
                "baseline_infer": baseline_infer,
                "infer_reduction": reduction_percent(
                    baseline_infer, candidate_infer
                ),
                "candidate_first": candidate_first,
                "baseline_first": baseline_first,
                "first_reduction": reduction_percent(
                    baseline_first, candidate_first
                ),
                "target_verify": target_verify,
                "candidate_e2e": candidate_infer + target_verify,
                "baseline_e2e": baseline_infer + target_verify,
                "e2e_reduction": reduction_percent(
                    baseline_infer + target_verify,
                    candidate_infer + target_verify,
                ),
            }
        )
    return rows


def render_markdown_report(
    *,
    title: str,
    contexts: Sequence[int],
    dims: ModelDimensions,
    candidate: DraftArchitecture,
    baseline: DraftArchitecture,
    candidate_config: Path,
    baseline_config: Path,
    new_context_tokens: int,
) -> str:
    rows = comparison_rows(
        contexts=contexts,
        dims=dims,
        candidate=candidate,
        baseline=baseline,
        new_context_tokens=new_context_tokens,
    )
    lines = [
        f"# {title}",
        "",
        "## Configuration",
        "",
        f"- Candidate: `{candidate.name}` from `{candidate_config}`",
        f"- Baseline: `{baseline.name}` from `{baseline_config}`",
        f"- Qwen3 dimensions: d={dims.hidden_size}, f={dims.intermediate_size}, "
        f"Q heads={dims.num_attention_heads}, KV heads={dims.num_key_value_heads}, "
        f"head_dim={dims.head_dim}, V={dims.vocab_size}",
        f"- Shared training shape: anchors={candidate.num_anchors}, "
        f"block={candidate.block_size}, feature layers={candidate.num_feature_layers}",
        f"- Steady-state inference uses R={new_context_tokens} newly committed "
        "target states per proposal.",
        "",
        "## Training FLOPs per sample",
        "",
        f"| Context | {baseline.name} | {candidate.name} | Reduction |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['context']:,} | {human_flops(row['baseline_train'])} | "
            f"{human_flops(row['candidate_train'])} | "
            f"{row['train_reduction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Draft proposal inference FLOPs",
            "",
            f"| Context | {baseline.name} | {candidate.name} | Reduction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['context']:,} | {human_flops(row['baseline_infer'])} | "
            f"{human_flops(row['candidate_infer'])} | "
            f"{row['infer_reduction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## End-to-end speculative iteration",
            "",
            "This adds the shared Qwen3 target verification of B+1 tokens. It "
            "does not model a change in acceptance rate or confidence truncation.",
            "",
            f"| Context | {baseline.name} + target | {candidate.name} + target | Reduction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['context']:,} | {human_flops(row['baseline_e2e'])} | "
            f"{human_flops(row['candidate_e2e'])} | "
            f"{row['e2e_reduction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## First draft-cache fill",
            "",
            "The first proposal has R=C because the draft KV cache is empty.",
            "",
            f"| Context | {baseline.name} | {candidate.name} | Reduction |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['context']:,} | {human_flops(row['baseline_first'])} | "
            f"{human_flops(row['candidate_first'])} | "
            f"{row['first_reduction']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- One multiply plus one add is two FLOPs.",
            "- Included: feature fusion, Q/K/V/O projections, gated MLP GEMMs, "
            "attention QK/AV GEMMs, draft and aligned-target vocabulary heads, "
            "and enabled Markov/confidence heads.",
            "- Training counts the executed forward/backward GEMMs. The frozen "
            "draft LM head has an input gradient but no weight gradient; the "
            "aligned target head is forward-only; the first layer of each draft "
            "stack omits the frozen embedding input-gradient GEMMs.",
            "- Excluded: RMSNorm, RoPE, SiLU, softmax/CE/L1/BCE, mask creation, "
            "sampling, cache copies, communication, and optimizer updates.",
            "- Training attention assumes all loss-mask positions are valid and "
            "anchors are sampled uniformly. When S < anchors+1, only S-1 anchors "
            "have logical attention pairs; padded blocks still pay their linear/head GEMMs.",
            "- Results are analytical algorithmic FLOPs, not hardware FLOP/s and "
            "not wall-clock latency.",
            "",
        ]
    )
    return "\n".join(lines)


def override_architecture(
    architecture: DraftArchitecture,
    **overrides: int | bool | str,
) -> DraftArchitecture:
    """Small public wrapper used by CLIs and tests."""

    updated = replace(architecture, **overrides)
    validate_architecture(updated)
    return updated
