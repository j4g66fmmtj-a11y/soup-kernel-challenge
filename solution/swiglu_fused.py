"""Fused SwiGLU (silu(gate) * up), forward + backward. Close to the starter example in structure."""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd(G, U, H, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0)
    s = (g * tl.sigmoid(g)).to(H.dtype.element_ty)  # как в HF: act(gate) округляется до dtype
    tl.store(H + offs, s * u, mask=mask)


@triton.jit
def _swiglu_bwd(DH, G, U, DG, DU, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dh = tl.load(DH + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    tl.store(DU + offs, (dh * g * sig).to(DU.dtype.element_ty), mask=mask)
    tl.store(DG + offs, (dh * u * sig * (1.0 + g * (1.0 - sig))).to(DG.dtype.element_ty), mask=mask)


class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        g, u = gate.contiguous(), up.contiguous()
        h = torch.empty_like(g)
        n = g.numel()
        _swiglu_fwd[(triton.cdiv(n, 1024),)](g, u, h, n, BLOCK=1024)
        ctx.save_for_backward(g, u)
        return h

    @staticmethod
    def backward(ctx, dh):
        g, u = ctx.saved_tensors
        dh = dh.contiguous()
        dg, du = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        _swiglu_bwd[(triton.cdiv(n, 1024),)](dh, g, u, dg, du, n, BLOCK=1024)
        return dg, du


def mlp_forward(self, x):
    return self.down_proj(SwiGLUFn.apply(self.gate_proj(x), self.up_proj(x)))


def patch_qwen2_mlp():
    from transformers.models.qwen2 import modeling_qwen2 as m
    m.Qwen2MLP.forward = mlp_forward
