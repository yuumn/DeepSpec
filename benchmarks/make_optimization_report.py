import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'output/optimized_final.json').read_text())
checks=json.loads((root/'output/validation_final.json').read_text())
rows=report['results']
assert len(rows)==24
assert len(checks)==24 and all(e['exact'] for c in checks for e in c['errors'].values())
summary=[]
for mode in ['full_context','cached_7_new']:
 for n in [512,1024,2048,4096]:
  ds,old,opt=[next(r for r in rows if r['model']==name and r['cache_mode']==mode and r['context_length']==n)
              for name in ['dspark','dspark_latentspec_original','dspark_latentspec']]
  for r in [ds,old,opt]:
   assert len(r['samples_ms'])==1000 and r['graph_matches_eager']
  assert opt['mean_ms'] < ds['mean_ms'],(mode,n)
  summary.append(dict(cache_mode=mode,context_length=n,dspark_ms=ds['mean_ms'],original_latent_ms=old['mean_ms'],optimized_latent_ms=opt['mean_ms'],
    reduction_vs_dspark_percent=100*(1-opt['mean_ms']/ds['mean_ms']),reduction_vs_original_latent_percent=100*(1-opt['mean_ms']/old['mean_ms']),
    dspark_std_ms=ds['std_ms'],optimized_std_ms=opt['std_ms'],
    all_rounds_separated=max(opt['round_means_ms'])<min(ds['round_means_ms']),
    dspark_gpu_ms=ds['gpu_mean_ms'],optimized_gpu_ms=opt['gpu_mean_ms']))
with (root/'output/optimization_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
text=['# DSpark-latentspec H800 优化结果','','优化后在 512、1024、2048、4096 四种长度、两种缓存口径下，平均单次起草延迟均低于原版 DSpark。比较对象是本仓库未做 RoPE 融合的 DSpark，并不表示对同样优化后的 DSpark 也有优势。','',
      '实验沿用配置一致的随机权重和合成 GPU hidden states；不是训练 checkpoint 的真实文本生成吞吐测试。单 H800 PCIe 80GB，batch=1、BF16、SDPA、CUDA Graph、block_size=7、greedy sampling、confidence threshold=0。','']
for mode,title in [('full_context','完整上下文、空 draft KV cache'),('cached_7_new','已有 N−7 个上下文 token 的 KV cache，每轮新增 7 个')]:
 text += ['', f'## {title}','','单位为 ms；每项先预热，随后 5 轮 × 200 次测量并取平均。','',
          '| 上下文 | DSpark | 原版 latent | 优化 latent | 相比 DSpark 延迟下降 |',
          '|---:|---:|---:|---:|---:|']
 for r in summary:
  if r['cache_mode']==mode:text.append(f"| {r['context_length']} | {r['dspark_ms']:.4f} | {r['original_latent_ms']:.4f} | {r['optimized_latent_ms']:.4f} | {r['reduction_vs_dspark_percent']:.2f}% |")
text += ['', '## 实现与正确性','',
 '- 只优化 RoPE：每个 attention 的 Q、K 各使用一个 Triton kernel，融合 rotate-half、cos/sin 乘法及加法。',
 '- 保留每次 BF16 乘法的中间舍入，禁用 FMA，并保留原张量 stride；没有改模型权重、层数、latent token 数、注意力后端、Markov head 或 confidence head。',
 '- 没有提前缓存输入相关的 RoPE 结果或把被测计算移到计时区间之外。前缀 KV cache 的建立只在明确标记的增量口径中排除。',
 '- 采用每个模型实例独立的 forward 替换，原版模型不受影响；`restore_latentspec()` 可恢复。限制为 eval + no_grad/inference_mode。',
 '- 3 个输入随机种子 × 4 种长度 × 2 种缓存条件，共 24 组；hidden states、base/corrected logits、KV keys/values、confidence、tokens、draft probabilities 全部逐元素完全一致（零容差）。原版与优化版使用相同权重。',
 '- 最终每个性能组合还校验 CUDA Graph 输出与自身 eager 输出逐元素完全一致。',
 '- 仓库内 API 检查通过：重复启用无副作用、原模型不受影响、autograd 防护、capture 后修改输入再 replay 与原版完全一致、恢复原实现后输出完全一致。',
 '- 计时为 `graph.replay()` 到 `cuda.synchronize()` 返回，包含完整起草与 proposal 构建，输出留在 GPU；不含目标模型 forward/verification、CPU 输入传输、模型加载、capture。',
 '- 每组 50 次 eager warmup，随后至少 200 次 replay 且持续至少 3 秒；原始样本、标准差、GPU Event 计时和每轮均值均保留。',
 f"- 八组比较中，每个优化版轮均值均低于对应 DSpark 的所有轮均值：{all(r['all_rounds_separated'] for r in summary)}。", '',
 '## 使用与复现','',
 '远端 `/root/DeepSpec` 中已提供 `deepspec/modeling/myspec/qwen3/optimized_inference.py` 和 `fused_rope.py`：','',
 '```python','from deepspec.modeling.myspec.qwen3.optimized_inference import optimize_latentspec, restore_latentspec',
 'model.eval()','optimize_latentspec(model)',
 '# 使用 torch.inference_mode() 调用原有 forward_myspec_draft_block / build_myspec_proposal。',
 '# CUDA Graph 捕获前先 warmup，使 Triton 完成编译。','```','',
 '复现完整验证和三方对比：','',
 '```bash',"ssh h800 'bash /root/DeepSpec/benchmarks/run_optimized.sh'",'```','',
 f"原仓库 commit：`{report['metadata']['source_commit']}`，优化为工作区新增实现。PyTorch {report['metadata']['torch']}，CUDA {report['metadata']['cuda']}，Transformers {report['metadata']['transformers']}。",'',
 '本地对应实现：`optimized_myspec.py`、`fused_rope.py`；测试：`validate_optimized.py`、`benchmark_optimized.py`；原始结果：`output/optimized_final.json`、`output/validation_final.json`；汇总：`output/optimization_summary.csv`。未采用的探索版本留在 `experiments/`。','']
(root/'optimization_results.md').write_text('\n'.join(text))
print('\n'.join(text[:24]))
