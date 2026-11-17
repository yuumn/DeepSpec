"""Single-request full-context draft latency, using unmodified DeepSpec eval ops.
Run with PYTHONPATH pointing at the recorded DeepSpec source snapshot.
"""
import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache, Qwen3Config
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block, build_dspark_proposal
from deepspec.eval.myspec.draft_ops import forward_myspec_draft_block, build_myspec_proposal


def summary(xs):
    ys = sorted(xs)
    return dict(mean_ms=statistics.mean(xs), std_ms=statistics.stdev(xs),
                median_ms=statistics.median(xs), p95_ms=ys[int(.95*(len(ys)-1))],
                min_ms=min(xs), max_ms=max(xs))


@torch.inference_mode()
def benchmark(model, name, length, rounds, repeats, cache_mode):
    torch.manual_seed(42 + length)
    c = model.config
    context = torch.randn(1, length, len(c.target_layer_ids)*c.hidden_size,
                          device='cuda', dtype=torch.bfloat16)
    first_token = torch.tensor([1234], device='cuda')
    positions = torch.arange(length + c.block_size, device='cuda')[None]
    latent = name == 'dspark_latentspec'

    prefix_cache = None
    if cache_mode == 'cached_7_new':
        prefix_cache = DynamicCache()
        prefix_length = length - c.block_size
        prefix_ids = torch.full((1, c.block_size), c.mask_token_id, device='cuda', dtype=torch.long)
        prefix_ids[:, 0] = first_token
        prefix_kwargs = dict(draft_input_ids=prefix_ids, position_ids=positions,
                             past_key_values_draft=prefix_cache,
                             target_hidden_states=context[:, :prefix_length],
                             start=prefix_length, block_size=c.block_size)
        if latent:
            latent_ids = torch.full((1, c.num_latent_tokens), c.latent_token_id, device='cuda', dtype=torch.long)
            latent_ids[:, 0] = first_token
            forward_myspec_draft_block(model, draft_latent_input_ids=latent_ids, **prefix_kwargs)
        else:
            forward_dspark_draft_block(model, **prefix_kwargs)
        assert prefix_cache.get_seq_length() == prefix_length
        context = context[:, prefix_length:].contiguous()

    def draft():
        cache = DynamicCache()
        if prefix_cache is not None:
            for i, layer in enumerate(prefix_cache.layers):
                cache.update(layer.keys, layer.values, i)

        ids = torch.full((1, c.block_size), c.mask_token_id, device='cuda', dtype=torch.long)
        ids[:, 0] = first_token
        kwargs = dict(draft_input_ids=ids, position_ids=positions,
                      past_key_values_draft=cache, target_hidden_states=context,
                      start=length, block_size=c.block_size)
        if latent:
            latent_ids = torch.full((1, c.num_latent_tokens), c.latent_token_id,
                                    device='cuda', dtype=torch.long)
            latent_ids[:, 0] = first_token
            hidden = forward_myspec_draft_block(model, draft_latent_input_ids=latent_ids, **kwargs)
            builder = build_myspec_proposal
        else:
            hidden = forward_dspark_draft_block(model, **kwargs)
            builder = build_dspark_proposal
        proposal = builder(model, draft_input_ids=ids, block_hidden=hidden,
                           block_size=c.block_size, temperature=0., confidence_threshold=0.)
        return proposal.verify_input_ids, proposal.confidence_logits, proposal.draft_probs

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(50):
            expected = draft()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = draft()
    graph.replay()
    torch.cuda.synchronize()
    for ref, got in zip(expected, actual):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        assert torch.isfinite(got).all().item()
    # At least 200 replays AND three seconds of warmup, excluding all from timing.
    start = time.perf_counter()
    warmups = 0
    while warmups < 200 or time.perf_counter()-start < 3:
        graph.replay()
        torch.cuda.synchronize()
        warmups += 1
    samples, round_means, device_samples = [], [], []
    for _ in range(rounds):
        current = []
        for _ in range(repeats):
            t = time.perf_counter_ns()
            graph.replay()
            torch.cuda.synchronize()
            current.append((time.perf_counter_ns()-t)/1e6)
        samples.extend(current)
        round_means.append(statistics.mean(current))
    # Separate GPU-only timing; do not perturb wall-clock samples with CUDA events.
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            graph.replay()
        b.record()
        b.synchronize()
        device_samples.append(a.elapsed_time(b)/repeats)
    row = dict(model=name, cache_mode=cache_mode, context_length=length, **summary(samples),
               round_means_ms=round_means, gpu_mean_ms=statistics.mean(device_samples),
               gpu_round_means_ms=device_samples, warmup_replays=warmups,
               warmup_eager=50, samples_ms=samples, graph_matches_eager=True,
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}), flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--variant', choices=['baseline', 'packed'], default='packed')
    p.add_argument('--lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    report = dict(metadata=dict(host=platform.node(), torch=torch.__version__,
                  cuda=torch.version.cuda, transformers=transformers.__version__,
                  cuda_home=os.getenv('CUDA_HOME'), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), gpu=torch.cuda.get_device_name(),
                  gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,power.limit', '--format=csv'], text=True),
                  batch_size=1, dtype='bfloat16', attention='sdpa', block_size=7,
                  temperature=0, confidence_threshold=0, cache_modes=['full_context', 'cached_7_new'], weights='random initialized from checkpoint architecture config, seed 42',
                  input='seeded synthetic target hidden states, resident on GPU',
                  timing='perf_counter_ns: graph.replay plus cuda.synchronize; GPU outputs',
                  rounds=a.rounds, repeats=a.repeats, variant=a.variant), results=[])
    for name, cls in [('dspark', Qwen3DSparkModel), ('dspark_latentspec', Qwen3MySpecModel)]:
        checkpoint = a.models/name/'step_2616'
        config = Qwen3Config.from_pretrained(checkpoint)
        config._attn_implementation = 'sdpa'
        with torch.device('cuda'):
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            model = cls(config).eval()
            torch.set_default_dtype(old_dtype)
        if name == 'dspark_latentspec' and a.variant == 'packed':
            from optimized_myspec import optimize_latentspec
            optimize_latentspec(model)
        for length in a.lengths:
            torch.cuda.reset_peak_memory_stats()
            for cache_mode in ['full_context', 'cached_7_new']:
                report['results'].append(benchmark(model, name, length, a.rounds, a.repeats, cache_mode))
            a.output.write_text(json.dumps(report, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
