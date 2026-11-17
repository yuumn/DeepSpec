"""Count actual Linear/SDPA shapes and independently check analytical FLOPs.
Multiply-add=2 FLOPs. Excludes elementwise/reductions/RoPE frequency generation,
embedding/indexing/copies, and target-model forward/verification.
"""
import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel
from deepspec.eval.dspark.draft_ops import build_dspark_proposal
from deepspec.eval.myspec.draft_ops import build_myspec_proposal
from no_cache_draft import forward_without_cache


def analytical(c,n,latent):
    d,f=c.hidden_size,c.intermediate_size
    dq,dk=c.num_attention_heads*c.head_dim,c.num_key_value_heads*c.head_dim
    b,r,v=c.block_size,c.markov_rank,c.vocab_size
    # Per-query Q/O projections + own K/V projections + gated MLP.
    def linears(q,extra): return 4*q*d*dq+4*(q+extra)*d*dk+6*q*d*f
    out={'feature_fusion':2*n*len(c.target_layer_ids)*d*d}
    if latent:
        l=c.num_latent_tokens
        out['latent_linears']=c.num_latent_layers*linears(l,n)
        out['latent_attention']=c.num_latent_layers*4*dq*l*(n+l)
        out['draft_linears']=c.num_hidden_layers*linears(b,n+l)
        out['draft_attention']=c.num_hidden_layers*4*dq*b*(n+l+b)
    else:
        out['draft_linears']=c.num_hidden_layers*linears(b,n)
        out['draft_attention']=c.num_hidden_layers*4*dq*b*(n+b)
    out.update(lm_head=2*b*d*v,markov_head=2*b*r*v,
               confidence_head=2*b*(d+r))
    return out


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    report={'scope':'Executed nn.Linear and attention QK^T/AV, MAC=2; no KV cache; batch=1; block=7',
            'excluded':['RMSNorm','RoPE incl frequency matmul','SiLU','softmax','bias additions','Markov logit additions','argmax','embedding/indexing/copies','target model'],
            'results':[]}
    for name,cls in [('dspark',Qwen3DSparkModel),('dspark_latentspec',Qwen3MySpecModel)]:
        torch.manual_seed(42)
        c=Qwen3Config.from_pretrained(a.models/name/'step_2616');c._attn_implementation='sdpa'
        with torch.device('cuda'):
            old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
            model=cls(c).eval();torch.set_default_dtype(old)
        latent=name=='dspark_latentspec'
        for n in [2**i for i in range(7,15)]:
            totals=defaultdict(int);trace=[];hooks=[]
            def record_linear(path):
                def hook(mod,inputs,output):
                    x=inputs[0];flops=2*x.numel()*mod.out_features
                    if path=='fc':group='feature_fusion'
                    elif path=='lm_head':group='lm_head'
                    elif path.startswith('markov_head'):group='markov_head'
                    elif path.startswith('confidence_head'):group='confidence_head'
                    elif path.startswith('latent_layers'):group='latent_linears'
                    else:group='draft_linears'
                    totals[group]+=flops
                    trace.append(dict(op='linear',name=path,input=list(x.shape),output=list(output.shape),flops=flops))
                return hook
            for path,mod in model.named_modules():
                if isinstance(mod,torch.nn.Linear):hooks.append(mod.register_forward_hook(record_linear(path)))
            original_sdpa=F.scaled_dot_product_attention
            attn_index=0
            def counted_sdpa(q,k,v,*args,**kwargs):
                nonlocal attn_index
                flops=4*math.prod(q.shape[:-2])*q.shape[-2]*k.shape[-2]*q.shape[-1]
                group='latent_attention' if latent and attn_index<c.num_latent_layers else 'draft_attention'
                attn_index+=1;totals[group]+=flops
                trace.append(dict(op='sdpa_qk_av',query=list(q.shape),key=list(k.shape),value=list(v.shape),flops=flops))
                return original_sdpa(q,k,v,*args,**kwargs)
            context=torch.randn(1,n,len(c.target_layer_ids)*c.hidden_size,device='cuda',dtype=torch.bfloat16)
            ids=torch.full((1,c.block_size),c.mask_token_id,device='cuda',dtype=torch.long);ids[:,0]=1234
            lids=None
            if latent:
                lids=torch.full((1,c.num_latent_tokens),c.latent_token_id,device='cuda',dtype=torch.long);lids[:,0]=1234
            positions=torch.arange(n+c.block_size,device='cuda')[None]
            F.scaled_dot_product_attention=counted_sdpa
            try:
                h=forward_without_cache(model,context=context,ids=ids,latent_ids=lids,positions=positions,start=n)
                builder=build_myspec_proposal if latent else build_dspark_proposal
                proposal=builder(model,draft_input_ids=ids,block_hidden=h,block_size=c.block_size,temperature=0.,confidence_threshold=0.)
                torch.cuda.synchronize()
            finally:
                F.scaled_dot_product_attention=original_sdpa
                for hook in hooks:hook.remove()
            expected=analytical(c,n,latent)
            assert dict(totals)==expected,(name,n,dict(totals),expected)
            row=dict(model=name,context_length=n,total_flops=sum(totals.values()),gflops=sum(totals.values())/1e9,
                breakdown=dict(totals),analytical_matches_runtime=True,executed_ops=trace)
            report['results'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='executed_ops'}),flush=True)
            a.output.write_text(json.dumps(report,indent=2))
            del context,h,proposal
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
