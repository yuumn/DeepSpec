"""RoPE fusion preserving eager BF16 operation boundaries (no fused FMA)."""
import torch
import triton
import triton.language as tl

@triton.jit
def _rope(X,C,S,Y,N:tl.constexpr,H:tl.constexpr,L:tl.constexpr,D:tl.constexpr,
          XS0:tl.constexpr,XS1:tl.constexpr,XS2:tl.constexpr,XS3:tl.constexpr,
          CS0:tl.constexpr,CS1:tl.constexpr,CS2:tl.constexpr,
          SS0:tl.constexpr,SS1:tl.constexpr,SS2:tl.constexpr,
          OFFSET:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=idx%D;l=idx//D%L;h=idx//(D*L)%H;b=idx//(D*L*H)
    x=tl.load(X+b*XS0+h*XS1+l*XS2+d*XS3,idx<N,other=0).to(tl.float32)
    rd=tl.where(d<D//2,d+D//2,d-D//2)
    r=tl.load(X+b*XS0+h*XS1+l*XS2+rd*XS3,idx<N,other=0).to(tl.float32)
    r=tl.where(d<D//2,-r,r)
    c=tl.load(C+b*CS0+(l+OFFSET)*CS1+d*CS2,idx<N,other=0).to(tl.float32)
    s=tl.load(S+b*SS0+(l+OFFSET)*SS1+d*SS2,idx<N,other=0).to(tl.float32)
    dtype=Y.dtype.element_ty
    left=(x*c).to(dtype).to(tl.float32)
    right=(r*s).to(dtype).to(tl.float32)
    tl.store(Y+b*XS0+h*XS1+l*XS2+d*XS3,(left+right).to(dtype),idx<N)

def _apply(x,cos,sin,offset):
    y=torch.empty_like(x)
    assert y.stride()==x.stride()
    _rope[(triton.cdiv(x.numel(),256),)](x,cos,sin,y,x.numel(),*x.shape[1:],
        *x.stride(),*cos.stride(),*sin.stride(),offset,256,enable_fp_fusion=False)
    return y

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    assert unsqueeze_dim==1
    return _apply(q,cos,sin,k.shape[2]-q.shape[2]),_apply(k,cos,sin,0)
