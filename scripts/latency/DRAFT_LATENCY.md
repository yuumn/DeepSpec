# DFlash vs MySpec 单次起草延迟

## 结论

在 NVIDIA H20-3e 上，当前指定配置的 MySpec 单次首次起草耗时均高于
DFlash。短上下文（128--1024）平均慢约 22%--25%；随着上下文增长，固定的
latent 路径开销被上下文处理成本摊薄，在 4096 上差距缩小至 3.38%。

| 上下文长度 | DFlash mean (ms) | MySpec mean (ms) | MySpec 相对 DFlash |
|---:|---:|---:|---:|
| 128 | 4.301 | 5.242 | +21.88% |
| 256 | 4.230 | 5.252 | +24.16% |
| 512 | 4.225 | 5.253 | +24.33% |
| 1024 | 4.216 | 5.254 | +24.61% |
| 2048 | 4.764 | 5.227 | +9.72% |
| 4096 | 7.498 | 7.752 | +3.38% |

完整的 mean、标准差、P50、P90、min、max 数据见
[`draft_latency_h20.csv`](./draft_latency_h20.csv)。

## 测量口径

- “单次起草”与 evaluator 的 `_propose` 路径一致，包括 draft backbone、
  LM head 和 token sampling，不包括 target model prefill 和 verification。
- 测量首次 proposal：每次从空 draft KV cache 开始，因此包含给定长度上下文
  的 draft prefill。这不是已有 KV cache 后的稳态 decode 延迟。
- batch size 为 1，block size 为 7，BF16，SDPA；temperature 为 1.0。
- 每个数据点独立预热 5 次，再用 CUDA Event 测量 30 次。
- 使用指定的 `dflash_qwen3_4b.py` 与 `myspec_qwen3_4b.py` 构造网络。未加载
  checkpoint 权重，因为权重数值不改变此固定执行路径的算子形状和耗时口径。
- 测试期间 GPU 无其他计算进程。

MySpec 配置为 2 层 latent + 3 层 draft，DFlash 为 5 层 draft。虽然总层数相同，
MySpec 还要处理 4 个 latent token，并包含 latent/mask 两段路径，因此短上下文下
固定开销更明显；长上下文时两者共同的上下文相关成本占比上升，差距随之缩小。

## 环境

- 主机：`set-hldy-llm-multimodal-worker07.mt`（SSH 别名 `mllm`）
- GPU：NVIDIA H20-3e
- Driver：550.127.08
- PyTorch：2.9.1+cu128
- Transformers：5.10.2
- CUDA runtime：12.8

## 复现

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec_latent_reasoning_diff-pos_0908_code
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-efficient-llm/yuanerhang/workspace/spec/DeepSpec/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/latency/benchmark_draft_latency.py \
  --target-config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/yuanerhang/workspace/spec/models/Qwen/Qwen3-4B/config.json \
  --warmup 5 \
  --repeats 30 \
  --output scripts/latency/draft_latency_h20.csv
```
