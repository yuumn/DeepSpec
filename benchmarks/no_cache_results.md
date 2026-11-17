# H800 无 KV cache：128–16384 上下文延迟

同一次运行对比原版 DSpark、原版 DSpark-latentspec 和 RoPE 融合优化版。所有调用明确设置 `past_key_values=None`、`use_cache=False`；不创建、写入或复用模型 KV cache，每次完整处理 N 个上下文 hidden states。

单 H800 PCIe 80GB，batch=1，BF16，SDPA，CUDA Graph，每次起草 7 个 token，greedy sampling；启用 Markov 和 confidence head，confidence threshold=0。

| 上下文长度 | DSpark（ms） | 原版 latent（ms） | 优化 latent（ms） | 优化版相对 DSpark 延迟下降 |
|---:|---:|---:|---:|---:|
| 128 | 2.4929 | 2.5890 | 2.4511 | 1.68% |
| 256 | 2.5317 | 2.6349 | 2.4883 | 1.71% |
| 512 | 2.6338 | 2.7396 | 2.5801 | 2.04% |
| 1024 | 2.9031 | 3.0094 | 2.8183 | 2.92% |
| 2048 | 3.3974 | 3.5185 | 3.2615 | 4.00% |
| 4096 | 4.3868 | 4.5106 | 4.1327 | 5.79% |
| 8192 | 6.8388 | 6.9613 | 6.2913 | 8.00% |
| 16384 | 11.6558 | 11.8077 | 10.4068 | 10.72% |

表中为每组 1,000 次端到端计时的均值。标准差、GPU Event 均值和全部原始样本保存在 CSV/JSON。

- 50 次 eager warmup，然后至少 200 次 graph replay 且持续至少 3 秒；正式计时 5 轮 × 200 次。
- 端到端范围：CPU 发起 graph.replay() 到 cuda.synchronize() 返回，包含上下文投影、完整起草 backbone、LM head、Markov 采样、confidence head 和 proposal 构建；输出留在 GPU。
- 不含目标模型 forward/verification、tokenizer、CPU/GPU 输入传输、模型加载、图捕获和首次 Triton 编译。
- 采用与指定配置相同的随机初始化权重和合成 GPU hidden states，不能解释为训练 checkpoint 的真实文本生成吞吐。
- 3 个输入种子 × 8 个长度的无缓存对照：优化前后 hidden states、base/corrected logits、confidence、tokens、probabilities 全部逐元素完全一致。
- 三种模型、8 个长度均通过 CUDA Graph 与自身 eager 输出的零容差一致性检查。
- PyTorch 2.9.1+cu128；CUDA runtime 12.8；Transformers 5.10.2；source commit `cf4c7da6e69f53d9b0ad4b8fa45174cb61178475` 加工作区 RoPE 优化。

复现：

```bash
ssh h800 'bash /root/DeepSpec/benchmarks/run_no_cache_sweep.sh'
```

原始数据：`output/no_cache_sweep.json`；数值校验：`output/no_cache_validation.json`；汇总：`output/no_cache_summary.csv`；无缓存执行路径：`no_cache_draft.py`。
