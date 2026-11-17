import copy
import torch
from validate_optimized import make_model,forward
from optimized_myspec import optimize_latentspec
from pathlib import Path
@torch.inference_mode()
def main():
 a=make_model(Path('benchmarks/models/dspark_latentspec/step_2616'));b=optimize_latentspec(copy.deepcopy(a))
 outputs={}
 def hook(name,which):
  def record(mod,args,out):
   if isinstance(out,torch.Tensor):outputs.setdefault(name,{})[which]=out.detach().clone()
  return record
 for name,m in a.named_modules():
  if name.endswith(('norm','layernorm','mlp')):m.register_forward_hook(hook(name,'ref'))
 for name,m in b.named_modules():
  if name.endswith(('norm','layernorm','mlp')):m.register_forward_hook(hook(name,'opt'))
 torch.manual_seed(523);x=torch.randn(1,512,12800,device='cuda',dtype=torch.bfloat16)
 forward(a,x,torch.tensor([1234],device='cuda'),'full_context');forward(b,x,torch.tensor([1234],device='cuda'),'full_context')
 for name,v in outputs.items():
  d=(v['ref'].float()-v['opt'].float());print(name,'max',d.abs().max().item(),'rel', (d.norm()/v['ref'].float().norm()).item(),flush=True)
if __name__=='__main__':main()
