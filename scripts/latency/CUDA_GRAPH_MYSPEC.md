# MySpec per-context CUDA Graph benchmark

The packed-GEMM MySpec proposal was captured independently for context lengths
128, 256, 512, 1024, 2048, and 4096. Each graph has fixed tensor shapes and
addresses. Capture occurs after three side-stream warmups and is excluded from
proposal latency. Reported latency uses five replay warmups followed by 30 CUDA
Event measurements on an otherwise idle NVIDIA H20-3e.

This is a strict single-proposal measurement: every invocation performs the
complete target feature projection and no result is cached across rounds.

| Context | Eager mean (ms) | Graph mean (ms) | Reduction | Speedup | Capture (ms) |
|---:|---:|---:|---:|---:|---:|
| 128 | 4.384 | 1.677 | 61.75% | 2.614x | 8.97 |
| 256 | 4.365 | 1.870 | 57.17% | 2.335x | 7.27 |
| 512 | 4.415 | 2.242 | 49.23% | 1.970x | 6.13 |
| 1024 | 4.375 | 3.007 | 31.26% | 1.455x | 11.16 |
| 2048 | 5.075 | 4.536 | 10.63% | 1.119x | 7.96 |
| 4096 | 8.104 | 7.565 | 6.64% | 1.071x | 14.03 |

CUDA Graph is most effective for short contexts, where Python dispatch and
kernel launch gaps dominate. At long contexts the large feature/context GEMMs
dominate, so replay still helps but by only 6%--11%.

## Integration constraints

- A graph is valid only for the captured shapes and memory addresses. Runtime
  integration needs a graph cache keyed by context-length bucket.
- Dynamic request data must be copied into each graph's static input buffers,
  or the target model must write directly to those buffers. This benchmark
  measures replay and does not include an external-to-static input copy.
- Exact lengths were captured here. Production traffic with arbitrary lengths
  needs padding/bucketing or additional graphs.
- Graph-private activation pools consume memory. Keeping all length graphs live
  trades GPU memory for eliminating recapture.
- Model weights must retain their addresses for the lifetime of every graph.

Raw results are in
[`cuda_graph_myspec_single_round_h20.csv`](./cuda_graph_myspec_single_round_h20.csv),
and the reproducible benchmark is
[`benchmark_cuda_graph_myspec.py`](./benchmark_cuda_graph_myspec.py).
