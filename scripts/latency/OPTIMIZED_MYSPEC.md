# MySpec packed-GEMM inference prototype

## Implemented optimizations

[`optimized_myspec.py`](./optimized_myspec.py) is an inference-only prototype
that leaves the training and evaluation implementation unchanged.

1. **One embedding lookup**
   - Builds one ID sequence containing anchor + latent tail + mask tail.
   - Calls `embed_tokens` once.
   - Reuses the anchor embedding when assembling both the latent and mask
     inputs. The original path calls `embed_tokens` twice and reads the anchor
     twice.
2. **Projection fusion on a shared input**
   - Latent attention: `Q + noise K + noise V` is one GEMM; context `K + V` is
     one GEMM.
   - Mask attention: `Q + mask K + mask V` is one GEMM; context `K + V` is one
     GEMM; latent `K + V` is one GEMM.
   - Outputs are split into the original tensors before RoPE, cache update and
     SDPA, so attention semantics are unchanged.

The benchmark checks both final hidden states and draft probabilities against
the baseline using BF16 tolerances before collecting latency.

## Why K/V cannot all become one projection

K and V operating on the **same input with the same layer** are packed along
the projection output dimension. Context, latent and mask projections remain
separate because they use different inputs and weights.

A cross-layer `torch._grouped_mm` version was also measured, but did not provide
a stable advantage: it reduced launch count while increasing simultaneous K/V
output residency and leaving downstream concat/cache copies unchanged. The
active implementation therefore uses the simpler per-layer packed-GEMM path.

## H20 results

NVIDIA H20-3e, BF16, SDPA, batch size 1, block size 7. Each point uses 5 warmup
runs followed by 30 measured runs.

| Context | Baseline mean (ms) | Optimized mean (ms) | Reduction | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 4.610 | 4.382 | 4.95% | 1.052x |
| 256 | 4.585 | 4.492 | 2.02% | 1.021x |
| 512 | 4.573 | 4.405 | 3.68% | 1.038x |
| 1024 | 4.563 | 4.388 | 3.84% | 1.040x |
| 2048 | 5.025 | 4.900 | 2.50% | 1.026x |
| 4096 | 7.744 | 7.702 | 0.55% | 1.006x |

Complete packed-GEMM statistics are in
[`optimized_myspec_h20.csv`](./optimized_myspec_h20.csv). Historical
cross-layer grouped-GEMM measurements remain in
[`grouped_gemm_myspec_h20.csv`](./grouped_gemm_myspec_h20.csv).

The benefit decreases with context length because attention and context GEMM
work dominates the fixed kernel-launch savings.

## Usage

```python
from optimized_myspec import (
    forward_optimized_myspec_draft_block,
    optimize_myspec_for_inference,
)

model.eval()
optimize_myspec_for_inference(model)  # pack weights once, outside timing
hidden = forward_optimized_myspec_draft_block(
    model,
    anchor_ids=anchor_ids,
    position_ids=position_ids,
    past_key_values_draft=cache,
    target_hidden_states=target_hidden_states,
    start=start,
    block_size=model.block_size,
)
```

This prototype keeps the original projection parameters and adds packed weight
copies for easy A/B testing. It no longer depends on private grouped-GEMM APIs.
A production checkpoint should store only packed inference weights to avoid
this temporary extra GPU-memory cost.
