# DFlash vs. MySpec CUDA Graph latency

Both implementations are captured and replayed as a separate CUDA Graph for
each context length. The table reports GPU-event latency over 30 graph replays
after 3 capture warmups and 5 replay warmups on one H20 GPU.

This is a strict single-proposal comparison. MySpec performs the complete
target feature projection during every replay; no projection result or other
cross-round computation is cached.

| Context | DFlash graph (ms) | MySpec graph (ms) | MySpec vs. DFlash |
|---:|---:|---:|---:|
| 128 | 1.692 | 1.679 | -0.76% |
| 256 | 1.855 | 1.860 | +0.28% |
| 512 | 2.229 | 2.229 | -0.00% |
| 1024 | 3.024 | 2.993 | -1.01% |
| 2048 | 4.539 | 4.527 | -0.26% |
| 4096 | 7.501 | 7.563 | +0.83% |
| 8192 | 13.237 | 13.628 | +2.96% |
| 16384 | 24.718 | 25.582 | +3.49% |

Negative percentages mean MySpec is faster. Through context length 4096, the
two implementations are effectively tied, with differences no larger than
1.01%. At 8192 and 16384, DFlash is respectively 2.96% and 3.49% faster,
showing a small but consistent advantage once long-context GPU work dominates.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts/latency:. \
python -u scripts/latency/benchmark_cuda_graph_dflash_vs_myspec.py \
  --target-config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B/config.json \
  --capture-warmup 3 --warmup 5 --repeats 30 \
  --output scripts/latency/cuda_graph_dflash_vs_myspec_single_round_h20.csv
```

The exact target-config path is machine-specific. The raw measurements,
including P50/P90 and graph-capture time, are in
`cuda_graph_dflash_vs_myspec_single_round_h20.csv`.
The 8192 and 16384 measurements are in
`cuda_graph_dflash_vs_myspec_long_context_h20.csv`.

## Scope

- Graph replay latency is measured; one-time graph capture is excluded.
- Each context length has a fixed input shape and its own graph.
- Inputs remain at stable addresses during replay. A serving integration must
  copy new values into persistent static input buffers before replay.
- Host-side scheduling and input-copy overhead are not included.
