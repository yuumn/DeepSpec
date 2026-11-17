"""Check the repository API, graph input refresh, and reversible installation."""
import copy
import json
from pathlib import Path
import torch
from validate_optimized import make_model, forward
from deepspec.modeling.myspec.qwen3.optimized_inference import optimize_latentspec, restore_latentspec


def main():
    torch.set_num_threads(1)
    ref=make_model(Path('benchmarks/models/dspark_latentspec/step_2616'))
    opt=copy.deepcopy(ref)
    original=opt.layers[0].self_attn.forward.__func__
    ref_fn=ref.layers[0].self_attn.forward.__func__
    optimize_latentspec(opt)
    installed=opt.layers[0].self_attn.forward.__func__
    assert optimize_latentspec(opt) is opt
    assert opt.layers[0].self_attn.forward.__func__ is installed
    assert ref.layers[0].self_attn.forward.__func__ is ref_fn
    try:
        opt.layers[0].self_attn.forward()
    except RuntimeError as e:
        assert 'inference_mode' in str(e)
    else:
        raise AssertionError('autograd guard missing')
    with torch.inference_mode():
        context=torch.randn(1,512,12800,device='cuda',dtype=torch.bfloat16)
        anchor=torch.tensor([1234],device='cuda')
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):forward(opt,context,anchor,'full_context')
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            actual=forward(opt,context,anchor,'full_context')
        # Change both GPU-resident inputs after capture.
        context.normal_()
        anchor.fill_(5678)
        expected=forward(ref,context,anchor,'full_context')
        graph.replay()
        torch.cuda.synchronize()
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        restore_latentspec(opt)
        assert opt.layers[0].self_attn.forward.__func__ is original
        restored=forward(opt,context,anchor,'full_context')
        for key in expected:torch.testing.assert_close(restored[key],expected[key],rtol=0,atol=0)
    result=dict(repository_api=True,idempotent=True,reference_unmodified=True,
        autograd_guard=True,graph_reads_updated_inputs=True,restore_exact=True)
    Path('benchmarks/output/api_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__=='__main__':main()
