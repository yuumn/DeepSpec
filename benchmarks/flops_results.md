# DSpark 与 DSpark+latentspec 无缓存单次起草 FLOPs

计算范围与上一轮延迟测试对应：batch=1、上下文 N、draft block=7、无 KV cache；上下文 target hidden states 已给定，不含目标模型生成 hidden states 和验证草稿的计算。

这里的 FLOPs 是每次起草的浮点运算量，不是每秒 FLOP/s。采用常用的矩阵乘/attention 主计算量口径：一个乘加算 2 FLOPs。包含 feature fusion、Q/K/V/O、MLP 三个矩阵乘、attention QKᵀ/AV、LM head、7 次 vanilla Markov head、confidence head。

**不计** RMSNorm、RoPE（含频率生成）、SiLU、softmax、bias/Markov logits 的逐元素相加、argmax、embedding lookup、数据拷贝等；表中不是逐条 GPU 指令的总数。

| 上下文长度 | DSpark（GFLOPs/次） | DSpark+latentspec（GFLOPs/次） | 计算量下降 |
|---:|---:|---:|---:|
| 128 | 28.232 | 27.134 | 3.888% |
| 256 | 43.405 | 42.294 | 2.558% |
| 512 | 73.750 | 72.615 | 1.539% |
| 1024 | 134.442 | 133.256 | 0.882% |
| 2048 | 255.825 | 254.539 | 0.503% |
| 4096 | 498.591 | 497.104 | 0.298% |
| 8192 | 984.124 | 982.234 | 0.192% |
| 16384 | 1955.189 | 1952.494 | 0.138% |

1 GFLOP = 10⁹ 次浮点运算。RoPE 融合版与原版 latent 的矩阵运算形状相同，因此在该口径下 FLOPs 完全一致。其时延优化来自算子融合和减少中间访存，并非减少上述矩阵运算量。

## 公式

维度：d=2560，f=9728，Dq=32×128=4096，Dkv=8×128=1024，V=151936，r=256；输入由 5 层 target features 拼接。DSpark 为 5 个 draft 层；latent 版本为 2 个 latent 层（4 个 token）+3 个 draft 层（7 个 token）。

对于 query token 数 q、额外 K/V token 数 m，一层的线性运算量：

```text
G(q,m) = 4*q*d*Dq + 4*(q+m)*d*Dkv + 6*q*d*f
F_fusion(N) = 2*N*(5*d)*d
F_heads = 2*7*d*V + 2*7*r*V + 2*7*(d+r)
F_DSpark(N) = F_fusion(N) + 5*G(7,N) + 4*Dq*5*7*(N+7) + F_heads
F_latent(N) = F_fusion(N) + 2*G(4,N) + 3*G(7,N+4)
              + 4*Dq*[2*4*(N+4) + 3*7*(N+4+7)] + F_heads
```

- DSpark：`F(N) = 118,538,240 × N + 13,058,759,168` FLOPs。
- DSpark+latentspec：`F(N) = 118,439,936 × N + 11,973,777,920` FLOPs。

上下文处理的 feature fusion 和 5 层 context K/V 投影在两种模型中相同，占长上下文的大部分运算，因此长度越大，latent 的 FLOPs 相对节省越小。

## 验证与复现

在 H800 上分别执行两种模型的 8 个长度。为每个 nn.Linear 注册输入/输出 shape hook，并记录实际 SDPA 的 Q/K/V shape；按观测形状计数后，与独立的逐层公式逐项断言相等。16 个组合全部通过。只需执行一次即可计数，不用通过重复计时估计 FLOPs。
计数使用 eager 前向以记录形状；CUDA Graph replay 的计算形状相同，预热/capture 不重复计入单次起草。权重数值不影响这些固定形状运算的 FLOPs。

```bash
ssh h800 'cd /root/DeepSpec && PYTHONPATH=. .venv/bin/python benchmarks/count_draft_flops.py --models benchmarks/models --output benchmarks/output/no_cache_flops.json'
```

逐算子 shape、分项计算量和验证结果：`output/no_cache_flops.json`；汇总：`output/no_cache_flops_summary.csv`；脚本：`count_draft_flops.py`。
