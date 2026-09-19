# Qwen3-4B MySpec / DFlash / DSpark FLOPs 对比

## 结论

按当前四份指定配置、乘加计 2 FLOPs、稳态每轮新增目标状态 `R=8` 计算：

- MySpec 对比 DFlash：上下文 512～4096 时，单样本训练总 FLOPs 下降 **9.083%～9.234%**，单次 draft proposal 推理 FLOPs 下降 **8.256%～9.411%**。
- MySpec-Markov-Conf 对比 DSpark：上下文 512～4096 时，单样本训练总 FLOPs 下降 **8.709%～8.897%**，单次 draft proposal 推理 FLOPs 下降 **7.941%～9.098%**。
- 两组对比去掉相同的输出头后，draft backbone 的训练 FLOPs 均下降 **14.859%～15.916%**。总量降幅较小，是因为 frozen LM head 和 aligned-target LM head 占比很高且双方相同。
- 若把共享的 Qwen3-4B target 对 8 个 token 的验证也计入一次 speculative iteration，端到端 FLOPs 降幅分别为 **1.410%～1.495%** 和 **1.400%～1.487%**。这里没有假设 acceptance rate 或 confidence 截断带来额外收益。

## 配置

使用的实验配置：

- DFlash：`spec/DeepSpec/config/dflash/dflash_qwen3_4b.py`
- MySpec：`config/myspec/myspec_qwen3_4b_eager.py`
- MySpec-Markov-Conf：`config/myspec/myspec_qwen3_4b.py`
- DSpark：`config/dspark/dspark_qwen3_4b.py`

共享 Qwen3-4B 参数为：`d=2560`、`f=9728`、Q heads=32、KV heads=8、`head_dim=128`、`V=151936`、target layers=36。四份配置均使用 5 个 target feature layers、512 个训练 anchors、block size 7。MySpec 使用 2 个 latent layers、4 个 latent tokens 和 3 个 mask/draft layers；DFlash/DSpark 使用 5 个 draft layers。MySpec-Markov-Conf 和 DSpark 都使用 rank-256 vanilla Markov head 与 confidence head。

## 现有脚本审计

原 `calculate_myspec_flops.py` 的主要矩阵拓扑、KV-cache 增量推理、frozen draft LM head 与 aligned-target LM head的处理是正确的，但不能直接用于本次指定配置：

1. 脚本默认 `num_latent_tokens=2`，指定的两份 MySpec 配置实际均为 4。
2. 缺少 MySpec-Markov-Conf 对比 DSpark，且未计算 rank-256 Markov projection 和 confidence projection。
3. 所有 decoder linear 统一按 `3 × forward` 估算；实际每个 latent/draft stack 的第一层输入来自 frozen embedding，不会计算 Q/K/V 的 embedding input-gradient GEMM。新脚本按当前 autograd 路径分别计算 forward、weight gradient 和需要的 input gradient。
4. 当 `S=512` 时最多只有 511 个有效 anchor；原脚本仍把 512 个 block 全部计入逻辑 attention。新脚本对 1 个 padded block 保留实际会执行的 linear/head GEMM，但不计被 mask 掉的 query-key pairs。
5. 指定 DFlash 配置的 `finalize_cfg` 中存在一处与 FLOPs 无关的既有非法 f-string。新脚本只静态解析 `model` 字典，不导入或执行配置，因此不受该问题影响。

## MySpec 对比 DFlash

### 单样本训练 FLOPs

| 上下文 S | DFlash | MySpec | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 19.481 TFLOPs | 17.711 TFLOPs | 9.083% |
| 1,024 | 19.854 TFLOPs | 18.046 TFLOPs | 9.107% |
| 2,048 | 20.601 TFLOPs | 18.715 TFLOPs | 9.152% |
| 4,096 | 22.093 TFLOPs | 20.053 TFLOPs | 9.234% |

### 稳态单次 draft proposal 推理 FLOPs（R=8）

| 上下文 C | DFlash | MySpec | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 13.752 GFLOPs | 12.616 GFLOPs | 8.256% |
| 1,024 | 14.045 GFLOPs | 12.859 GFLOPs | 8.442% |
| 2,048 | 14.632 GFLOPs | 13.346 GFLOPs | 8.791% |
| 4,096 | 15.807 GFLOPs | 14.319 GFLOPs | 9.411% |

### 单次 speculative iteration（draft + 共享 target 验证）

| 上下文 C | DFlash + target | MySpec + target | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 80.545 GFLOPs | 79.410 GFLOPs | 1.410% |
| 1,024 | 83.255 GFLOPs | 82.069 GFLOPs | 1.424% |
| 2,048 | 88.674 GFLOPs | 87.387 GFLOPs | 1.451% |
| 4,096 | 99.512 GFLOPs | 98.024 GFLOPs | 1.495% |

### 首次 draft cache fill（R=C）

| 上下文 C | DFlash | MySpec | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 73.206 GFLOPs | 72.070 GFLOPs | 1.551% |
| 1,024 | 133.897 GFLOPs | 132.712 GFLOPs | 0.885% |
| 2,048 | 255.280 GFLOPs | 253.994 GFLOPs | 0.504% |
| 4,096 | 498.047 GFLOPs | 496.559 GFLOPs | 0.299% |

## MySpec-Markov-Conf 对比 DSpark

### 单样本训练 FLOPs

| 上下文 S | DSpark | MySpec-Markov-Conf | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 20.317 TFLOPs | 18.548 TFLOPs | 8.709% |
| 1,024 | 20.691 TFLOPs | 18.883 TFLOPs | 8.739% |
| 2,048 | 21.437 TFLOPs | 19.552 TFLOPs | 8.795% |
| 4,096 | 22.930 TFLOPs | 20.890 TFLOPs | 8.897% |

### 稳态单次 draft proposal 推理 FLOPs（R=8）

| 上下文 C | DSpark | MySpec-Markov-Conf | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 14.296 GFLOPs | 13.161 GFLOPs | 7.941% |
| 1,024 | 14.590 GFLOPs | 13.404 GFLOPs | 8.127% |
| 2,048 | 15.177 GFLOPs | 13.891 GFLOPs | 8.475% |
| 4,096 | 16.351 GFLOPs | 14.864 GFLOPs | 9.098% |

### 单次 speculative iteration（draft + 共享 target 验证）

| 上下文 C | DSpark + target | MySpec-Markov-Conf + target | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 81.090 GFLOPs | 79.954 GFLOPs | 1.400% |
| 1,024 | 83.799 GFLOPs | 82.613 GFLOPs | 1.415% |
| 2,048 | 89.218 GFLOPs | 87.932 GFLOPs | 1.442% |
| 4,096 | 100.056 GFLOPs | 98.569 GFLOPs | 1.487% |

### 首次 draft cache fill（R=C）

| 上下文 C | DSpark | MySpec-Markov-Conf | 下降 |
| ---: | ---: | ---: | ---: |
| 512 | 73.750 GFLOPs | 72.615 GFLOPs | 1.539% |
| 1,024 | 134.442 GFLOPs | 133.256 GFLOPs | 0.882% |
| 2,048 | 255.825 GFLOPs | 254.539 GFLOPs | 0.503% |
| 4,096 | 498.591 GFLOPs | 497.104 GFLOPs | 0.298% |

## 计算口径

- 一次乘法加一次加法计 2 FLOPs。
- 训练只计算 draft model；target hidden states 和 target last hidden states 来自离线 cache，不计算 target model forward。
- 计入 feature fusion、Q/K/V/O projection、Qwen3 gated MLP 三个矩阵、attention 的 QK/AV、draft LM head、aligned-target LM head，以及启用的 Markov/confidence head。
- frozen draft LM head 计 forward + hidden-state input gradient；aligned-target LM head 的输入与权重都不求导，只计 forward；trainable linear 按实际是否需要 input gradient 计算。
- 稳态推理的 `R` 是上轮新提交且尚未进入 draft KV cache 的 target states 数量，实际范围通常为 1～8。主表采用完整接受时的 `R=8`；若采用 `R=1`，两组 draft 推理降幅范围分别是 8.783%～9.930% 和 8.428%～9.582%。
- attention FLOPs 按 mask 后逻辑可见的 query-key pairs 计算；这是算法 FLOPs，不等同于特定 FlexAttention kernel 的 block padding 或硬件利用率。
- 未计入 RMSNorm、RoPE、SiLU、softmax、CE/L1/BCE、mask 构造、采样、KV cache copy/crop、分布式通信和 optimizer update。它们的口径依实现而异，且相对主要 GEMM 较小。
- 首次 draft proposal 的 cache 为空，因此 `R=C`；target prefill 是双方共享成本，未重复加入表格。
- 端到端表固定验证完整 `B+1=8` 个 token，不推断不同模型的实际 acceptance rate 或 confidence threshold 收益。

## 复现

在仓库根目录执行：

```bash
python scripts/flops/calculate_myspec_flops.py
python scripts/flops/calculate_myspec_markov_conf_vs_dspark_flops.py
python scripts/flops/test_flops.py
```

两个计算脚本都支持 `--context-lengths`、`--new-context-tokens`、`--target-config` 和 `--output`；默认 Qwen3-4B 维度已与本地 Hugging Face `config.json` 核验一致。
