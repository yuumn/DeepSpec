# DFlash Opt vs. MySpec Opt（CUDA Graph）

## 结论

在 NVIDIA H20-3e、BF16、SDPA、batch size 1、block size 7 条件下，
MySpec Opt 在测试的 8 个 context length 上均快于 DFlash Opt。正序和反序
各自独立构建模型并捕获 CUDA Graph，两轮共 16 个配对点，结果方向全部一致。

每个点执行 5 次 capture warmup、20 次 replay warmup 和 100 次 GPU Event
计时。表内延迟只包含 `_forward_backbone` 的 graph replay，不包含一次性的
graph capture 时间。

| Context | DFlash Opt 正序 (ms) | MySpec Opt 正序 (ms) | MySpec 正序降幅 | MySpec 反序降幅 |
|---:|---:|---:|---:|---:|
| 128 | 0.848676 | 0.839315 | 1.103% | 8.016% |
| 256 | 1.078347 | 1.004756 | 6.824% | 6.650% |
| 512 | 1.390407 | 1.326409 | 4.603% | 4.563% |
| 1024 | 1.976875 | 1.922704 | 2.740% | 2.782% |
| 2048 | 3.238392 | 3.164695 | 2.276% | 2.488% |
| 4096 | 5.537192 | 5.243933 | 5.296% | 5.416% |
| 8192 | 9.729712 | 9.489257 | 2.471% | 2.351% |
| 16384 | 18.418582 | 17.927992 | 2.664% | 2.559% |

8 个长度的非加权算术平均中，正序 MySpec 延迟低 3.077%，反序低 3.178%。
128 的 DFlash 反序结果存在明显的运行顺序敏感性，因此逐长度比较以正序表中的
绝对延迟为主，反序只用于确认结论没有反转。

原始统计（含 min、P50、P90、max 和 capture 时间）位于：

- `benchmark_dflash_opt_vs_myspec_opt_cuda_graph_h20.csv`
- `benchmark_myspec_opt_vs_dflash_opt_cuda_graph_h20_reverse.csv`

## MySpec Opt 的关键优化

1. 为固定形状推理增加显式 `prepare_for_cuda_graph()`，并在训练、权重加载和
   device/dtype 转换时正确失效 graph 专用缓存和辅助 stream。
2. Triton RMSNorm 支持 Q 投影产生的最后一维连续、其余维带 stride 的 view，
   避免隐式 contiguous/copy；最后一个 latent layer 使用 residual-add +
   RMSNorm 融合内核。
3. 对共享 context 输入的多层 K/V 投影做跨层 batched matmul，并按 context
   长度选择合并投影或双 stream 投影。
4. 批量计算 mask layer 使用的 latent K/V；长 context 下把该投影放到辅助
   stream，并与首个 mask layer 的 norm 和 Q/K/V 投影重叠。
5. 短 context 把 K/V 片段合并为一次 packed cat；长 context 使用 Triton
   内核把 K 片段拼接、逐 head RMSNorm 和 BHSD 布局写出融合为一次操作，删除
   每层独立的 K cat 和中间张量写回。
6. latent 和 mask RoPE 共用一次生成结果；latent layer 复用其前缀。

## 等价性验证

- 长序列 fused K 内核对原 `cat -> view -> RMSNorm -> transpose`：BF16
  `max_abs = 0`。
- 完整小模型与原始 `myspec/qwen3` 使用同一 state dict，在 context
  32/4096/8192 上的最大绝对误差分别为 0、0.001953125、0.015625；graph
  优化路径与 optimized eager 路径误差相同，均通过 `atol=rtol=0.02`。

## 复现

```bash
ssh mllm
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code

OUTPUT_PATH=benchmark_dflash_opt_vs_myspec_opt_cuda_graph_h20.csv \
./run_benchmark.sh --cuda-graph --capture-warmup 5 \
  --context-lengths 128 256 512 1024 2048 4096 8192 16384 \
  --warmup 20 --repeats 100

OUTPUT_PATH=benchmark_myspec_opt_vs_dflash_opt_cuda_graph_h20_reverse.csv \
./run_benchmark.sh --cuda-graph --capture-warmup 5 \
  --models myspec dflash \
  --context-lengths 128 256 512 1024 2048 4096 8192 16384 \
  --warmup 20 --repeats 100
```

每个 context length 使用独立 CUDA Graph 和稳定地址的静态输入。真实服务接入
时，需要在 replay 前把新请求复制到对应静态 input buffer；本报告不包含该输入
复制和 host 调度开销。
