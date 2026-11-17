"""Compare original/packed inference on identical weights and diverse inputs."""
import argparse
import copy
import json
from pathlib import Path
import torch
from transformers import DynamicCache, Qwen3Config
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.eval.myspec.draft_ops import forward_myspec_draft_block, build_myspec_proposal
from optimized_myspec import optimize_latentspec


def make_model(path):
    torch.manual_seed(42)
    config = Qwen3Config.from_pretrained(path)
    config._attn_implementation = 'sdpa'
    with torch.device('cuda'):
        dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen3MySpecModel(config).eval()
        torch.set_default_dtype(dtype)
    return model


def forward(model, context, anchor, mode):
    c = model.config
    n = context.shape[1]
    positions = torch.arange(n+c.block_size, device='cuda')[None]
    ids = torch.full((1,c.block_size), c.mask_token_id, device='cuda',dtype=torch.long)
    lids = torch.full((1,c.num_latent_tokens), c.latent_token_id,device='cuda',dtype=torch.long)
    ids[:,0] = lids[:,0] = anchor
    cache = DynamicCache()
    def block(ctx, start):
        return forward_myspec_draft_block(model, draft_latent_input_ids=lids,
            draft_input_ids=ids, position_ids=positions, past_key_values_draft=cache,
            target_hidden_states=ctx, start=start, block_size=c.block_size)
    if mode == 'cached_7_new':
        block(context[:,:n-c.block_size], n-c.block_size)
        ctx = context[:,n-c.block_size:].contiguous()
    else:
        ctx = context
    hidden = block(ctx, n)
    logits = model.compute_logits(hidden)
    tokens, corrected = model.sample_draft_tokens(logits, first_prev_token_ids=ids[:,0],temperature=0.,hidden_states=hidden)
    proposal = build_myspec_proposal(model, draft_input_ids=ids, block_hidden=hidden,
        block_size=c.block_size,temperature=0.,confidence_threshold=0.)
    return dict(hidden=hidden,logits=logits,corrected_logits=corrected,
        tokens=proposal.verify_input_ids, confidence=proposal.confidence_logits,
        probs=proposal.draft_probs,
        cache_keys=torch.stack([x.keys for x in cache.layers]),
        cache_values=torch.stack([x.values for x in cache.layers]))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    model=make_model(a.models/'dspark_latentspec/step_2616')
    opt=optimize_latentspec(copy.deepcopy(model))
    results=[]
    for seed in [11,42,2026]:
      for n in [512,1024,2048,4096]:
       for mode in ['full_context','cached_7_new']:
        torch.manual_seed(seed+n)
        context=torch.randn(1,n,12800,device='cuda',dtype=torch.bfloat16)
        anchor=torch.randint(0,model.config.vocab_size,(1,),device='cuda')
        ref=forward(model,context,anchor,mode)
        got=forward(opt,context,anchor,mode)
        errors={}
        for key in ref:
            x,y=ref[key],got[key]
            assert torch.isfinite(y).all(),key
            if key in ['tokens','probs']:
                torch.testing.assert_close(y,x,rtol=0,atol=0)
            else:
                torch.testing.assert_close(y,x,rtol=.02,atol=.03)
            delta=(y.float()-x.float())
            errors[key]=dict(max_abs=delta.abs().max().item(),relative_l2=(delta.norm()/x.float().norm().clamp_min(1e-12)).item(),exact=torch.equal(x,y))
        results.append(dict(seed=seed,length=n,cache_mode=mode,errors=errors))
        print(json.dumps(results[-1]),flush=True)
        a.output.write_text(json.dumps(results,indent=2))
    print('PASS: 24 input/cache cases; identical greedy tokens/probabilities, bounded BF16 tensor error.',flush=True)

if __name__=='__main__':main()
