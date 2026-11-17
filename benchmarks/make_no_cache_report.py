import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'output/no_cache_sweep.json').read_text())
checks=json.loads((root/'output/no_cache_validation.json').read_text())
lengths=[2**i for i in range(7,15)]
assert len(report['results'])==24 and len(checks)==24
assert all(e['exact'] for c in checks for e in c['errors'].values())
summary=[]
for n in lengths:
 ds,old,opt=[next(r for r in report['results'] if r['model']==name and r['context_length']==n)
    for name in ['dspark','dspark_latentspec_original','dspark_latentspec']]
 for r in [ds,old,opt]:
  assert r['cache_mode']=='no_cache' and len(r['samples_ms'])==1000 and r['graph_matches_eager']
 summary.append(dict(context_length=n,dspark_ms=ds['mean_ms'],original_latent_ms=old['mean_ms'],optimized_latent_ms=opt['mean_ms'],
    reduction_vs_dspark_percent=100*(1-opt['mean_ms']/ds['mean_ms']),
    dspark_std_ms=ds['std_ms'],original_latent_std_ms=old['std_ms'],optimized_latent_std_ms=opt['std_ms'],
    dspark_gpu_ms=ds['gpu_mean_ms'],original_latent_gpu_ms=old['gpu_mean_ms'],optimized_latent_gpu_ms=opt['gpu_mean_ms']))
with (root/'output/no_cache_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
lines=['# H800 无 KV cache：128–16384 上下文延迟','','同一次运行对比原版 DSpark、原版 DSpark-latentspec 和 RoPE 融合优化版。所有调用明确设置 `past_key_values=None`、`use_cache=False`；不创建、写入或复用模型 KV cache，每次完整处理 N 个上下文 hidden states。','',
 '单 H800 PCIe 80GB，batch=1，BF16，SDPA，CUDA Graph，每次起草 7 个 token，greedy sampling；启用 Markov 和 confidence head，confidence threshold=0。','',
 '| 上下文长度 | DSpark（ms） | 原版 latent（ms） | 优化 latent（ms） | 优化版相对 DSpark 延迟下降 |',
 '|---:|---:|---:|---:|---:|']
for r in summary:
 lines.append(f"| {r['context_length']} | {r['dspark_ms']:.4f} | {r['original_latent_ms']:.4f} | {r['optimized_latent_ms']:.4f} | {r['reduction_vs_dspark_percent']:.2f}% |")
lines+=['','表中为每组 1,000 次端到端计时的均值。标准差、GPU Event 均值和全部原始样本保存在 CSV/JSON。','',
 '- 50 次 eager warmup，然后至少 200 次 graph replay 且持续至少 3 秒；正式计时 5 轮 × 200 次。',
 '- 端到端范围：CPU 发起 graph.replay() 到 cuda.synchronize() 返回，包含上下文投影、完整起草 backbone、LM head、Markov 采样、confidence head 和 proposal 构建；输出留在 GPU。',
 '- 不含目标模型 forward/verification、tokenizer、CPU/GPU 输入传输、模型加载、图捕获和首次 Triton 编译。',
 '- 采用与指定配置相同的随机初始化权重和合成 GPU hidden states，不能解释为训练 checkpoint 的真实文本生成吞吐。',
 '- 3 个输入种子 × 8 个长度的无缓存对照：优化前后 hidden states、base/corrected logits、confidence、tokens、probabilities 全部逐元素完全一致。',
 '- 三种模型、8 个长度均通过 CUDA Graph 与自身 eager 输出的零容差一致性检查。',
 f"- PyTorch {report['metadata']['torch']}；CUDA runtime {report['metadata']['cuda']}；Transformers {report['metadata']['transformers']}；source commit `{report['metadata']['source_commit']}` 加工作区 RoPE 优化。",'',
 '复现：','',"```bash\nssh h800 'bash /root/DeepSpec/benchmarks/run_no_cache_sweep.sh'\n```",'',
 '原始数据：`output/no_cache_sweep.json`；数值校验：`output/no_cache_validation.json`；汇总：`output/no_cache_summary.csv`；无缓存执行路径：`no_cache_draft.py`。','']
(root/'no_cache_results.md').write_text('\n'.join(lines))
print('\n'.join(lines[:16]))
