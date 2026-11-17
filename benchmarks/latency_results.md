# H800 单次起草 CUDA Graph 延迟

实测设备：NVIDIA H800 PCIe 80GB，单卡、batch size 1、BF16、SDPA。每次起草 7 个 token，greedy sampling，confidence threshold=0（计算置信度但不提前截断）。

**这是随机初始化权重和合成输入的结构级基准，不是训练 checkpoint 的真实文本推理测试。** 配置已与任务指定的两个 Python 文件逐项核对；使用本地同结构 checkpoint 的 config.json 构造模型，没有加载其权重。

## 完整上下文起草（主结果）

每次从空 draft KV cache 开始，完整处理 N 个 target hidden states。端到端计时为 CPU 发起 graph.replay() 到 cuda.synchronize() 返回；输出留在 GPU。

| 上下文长度 | DSpark（ms） | DSpark latent（ms） | latent / DSpark |
|---:|---:|---:|---:|
| 512 | 2.7018 ± 0.0553 | 2.8114 ± 0.0036 | 1.041× |
| 1024 | 3.0064 ± 0.0245 | 3.1188 ± 0.0039 | 1.037× |
| 2048 | 3.5818 ± 0.0068 | 3.6996 ± 0.0045 | 1.033× |
| 4096 | 4.7486 ± 0.0095 | 4.8758 ± 0.0044 | 1.027× |

## 已有 KV cache 的增量起草（补充）

预先缓存 N−7 个上下文 token，每次计时仅处理新增的 7 个 target hidden states，并关注完整 N-token 上下文；前缀缓存建立不计时。不能与主表混为同一测试。

| 上下文长度 | DSpark（ms） | DSpark latent（ms） | latent / DSpark |
|---:|---:|---:|---:|
| 512 | 2.5914 ± 0.0081 | 2.7003 ± 0.0036 | 1.042× |
| 1024 | 2.7060 ± 0.0041 | 2.8092 ± 0.0034 | 1.038× |
| 2048 | 2.9118 ± 0.0051 | 3.0157 ± 0.0029 | 1.036× |
| 4096 | 3.3274 ± 0.0056 | 3.4315 ± 0.0050 | 1.031× |

表内为 1,000 次测量的均值 ± 样本标准差。

## 测试口径与复现

- DSpark：5 层 draft；latent：2 层 latent + 3 层 draft、4 个 latent token；均启用 rank=256 的 vanilla Markov head 和 confidence head。
- 包含输入 token 构造、embedding、上下文投影、RoPE、注意力/MLP、KV cache 拼接、LM head、Markov 自回归采样、confidence head、proposal 拼接和 draft probabilities。调用原仓库 eval draft ops，没有修改模型计算代码。
- 不包含目标模型 forward、tokenizer、CPU/GPU 输入传输、target verification、模型加载、graph capture。输入为 GPU 上预生成的随机 target hidden states。
- 每个模型/长度/cache 模式单独 capture；50 次 eager 预热，再至少 200 次 graph replay 且持续至少 3 秒；5 轮 × 200 次逐次同步计时。
- 每个组合校验 graph 输出和 eager 输出逐元素完全一致，且输出有限。另以 CUDA Event 记录 GPU-only 均值，详见 results.json。
- 随机权重仅用于比较固定计算路径耗时，不能据此推断接受率或整体生成 TPS；未使用 torch.compile 或额外 fused/packed 优化。
- 代码：`rebuttal-1`，commit `d8009b87fb8b38b6ed2abbf6925d4accb8bca7d3`。
- Python 环境：uv 创建的 Python 3.12；PyTorch 2.9.1+cu128、CUDA runtime 12.8、Transformers 5.10.2。

远端执行：

```bash
ssh h800 'bash /root/DeepSpec/benchmarks/run_benchmark.sh'
```

本地文件：`benchmark_draft_h800.py`、`run_benchmark.sh`、`config_provenance.json`、`configs/`；原始数据与完整环境在 `output/`。
