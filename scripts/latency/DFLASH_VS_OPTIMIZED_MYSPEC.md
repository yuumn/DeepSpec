# DFlash vs packed-GEMM MySpec

Both models were measured in the same idle-GPU run on an NVIDIA H20-3e using
BF16, SDPA, batch size 1 and block size 7. Every point starts from an empty
draft KV cache, warms up 5 times, and reports the mean of 30 CUDA Event
measurements.

| Context | DFlash mean (ms) | Optimized MySpec mean (ms) | MySpec difference | DFlash / MySpec |
|---:|---:|---:|---:|---:|
| 128 | 4.238 | 4.477 | +5.65% | 0.947x |
| 256 | 4.215 | 4.504 | +6.86% | 0.936x |
| 512 | 4.194 | 4.510 | +7.54% | 0.930x |
| 1024 | 4.192 | 4.578 | +9.22% | 0.916x |
| 2048 | 4.783 | 4.837 | +1.14% | 0.989x |
| 4096 | 7.525 | 7.645 | +1.60% | 0.984x |

Packed-GEMM MySpec is still slower than DFlash at every tested context length,
with a mean gap of 5.65%--9.22% through 1024 tokens and 1.14%--1.60% at
2048--4096. P50 and P90 show the same ordering. Short-context MySpec samples
have more variance in this run, but their medians remain slower than DFlash.

Full statistics are in
[`dflash_vs_packed_myspec_h20.csv`](./dflash_vs_packed_myspec_h20.csv).
The comparison can be reproduced with
[`benchmark_dflash_vs_optimized_myspec.py`](./benchmark_dflash_vs_optimized_myspec.py).
