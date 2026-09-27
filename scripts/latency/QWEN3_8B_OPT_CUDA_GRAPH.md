# Qwen3-8B DFlash Opt vs. MySpec Opt

## 测试条件

- GPU：NVIDIA H20-3e；测试前后利用率 0%，显存占用 1 MiB
- Qwen3-8B：hidden size 4096，intermediate size 12288，32 attention heads
- BF16、SDPA、batch size 1、block size 7
- DFlash Opt 和 MySpec Opt 都使用固定形状 CUDA Graph
- 每个点：5 次 capture warmup、20 次 replay warmup、100 次 GPU Event 计时
- 统计范围：`_forward_backbone` graph replay，不包含 graph capture

## 空闲 GPU 新测试结果

| Context | DFlash 正序 (ms) | MySpec 正序 (ms) | 正序降幅 | DFlash 反序 (ms) | MySpec 反序 (ms) | 反序降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.251080 | 1.245130 | 0.476% | 1.314806 | 1.249507 | 4.966% |
| 256 | 1.610992 | 1.543326 | 4.200% | 1.610508 | 1.545687 | 4.025% |
| 512 | 2.149896 | 2.084351 | 3.049% | 2.148602 | 2.090894 | 2.686% |
| 1024 | 3.199259 | 3.144335 | 1.717% | 3.199010 | 3.149037 | 1.562% |
| 2048 | 5.467649 | 5.361582 | 1.940% | 5.463820 | 5.368812 | 1.739% |
| 4096 | 9.784226 | 9.432538 | 3.594% | 9.780388 | 9.445577 | 3.423% |
| 8192 | 17.824345 | 17.618940 | 1.152% | 17.839182 | 17.621368 | 1.221% |
| 16384 | 34.216841 | 33.724154 | 1.440% | 34.208695 | 33.739523 | 1.371% |

MySpec Opt 在正序和反序的全部 16 个配对点上均快于 DFlash Opt。8 个
context length 的非加权算术平均中，正序延迟降低 1.788%，反序降低
1.793%。128 token 的 DFlash 对运行顺序较敏感，因此该点以正序的绝对延迟
为主；其余长度的正反序结果接近。

原始数据：

- `benchmark_qwen3_8b_dflash_opt_vs_myspec_opt_cuda_graph_h20_idle.csv`
- `benchmark_qwen3_8b_myspec_opt_vs_dflash_opt_cuda_graph_h20_idle_reverse.csv`
