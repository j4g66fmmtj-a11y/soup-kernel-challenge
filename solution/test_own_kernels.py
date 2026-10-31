"""Проверка собственных кернелов на GPU: pytest solution/test_own_kernels.py -q"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")


def _ref_rms(x, w, eps=1e-6):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


@pytest.mark.parametrize("wdtype", [torch.float16, torch.float32])
def test_rmsnorm(wdtype):
    from solution.kernels import RMSNormFn
    torch.manual_seed(0)
    x = torch.randn(4, 512, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(1536, device="cuda", dtype=wdtype)
    x2 = x.detach().clone().requires_grad_()
    r = _ref_rms(x, w)
    o = RMSNormFn.apply(x2, w, 1e-6)
    assert o.dtype == r.dtype
    dy = torch.randn_like(r)
    r.backward(dy)
    o.backward(dy)
    torch.testing.assert_close(o, r, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x2.grad, x.grad, rtol=2e-2, atol=2e-2)


def test_rope():
    from solution.kernels import apply_rotary_pos_emb
    import solution.kernels as K
    from transformers.models.qwen2 import modeling_qwen2 as m
    K._orig_rope = m.apply_rotary_pos_emb
    torch.manual_seed(0)
    B, S, D = 2, 128, 128
    q = torch.randn(B, 12, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, 2, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
    ang = torch.randn(B, S, D // 2, device="cuda")
    cos = torch.cat([ang.cos()] * 2, -1).half()
    sin = torch.cat([ang.sin()] * 2, -1).half()
    q2, k2 = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    rq, rk = K._orig_rope(q, k, cos, sin)
    oq, ok = apply_rotary_pos_emb(q2, k2, cos, sin)
    dq, dk = torch.randn_like(rq), torch.randn_like(rk)
    (rq * dq).sum().backward(retain_graph=False)
    (oq * dq).sum().backward()
    torch.testing.assert_close(oq, rq, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ok, rk, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q2.grad, q.grad, rtol=2e-2, atol=2e-2)


def test_log_softmax():
    from solution.logsoftmax import LogSoftmaxFn
    torch.manual_seed(0)
    x = torch.randn(64, 151936, device="cuda", dtype=torch.float32, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    r = torch.log_softmax(x, -1)
    o = LogSoftmaxFn.apply(x2)
    r.backward(dy)
    o.backward(dy)
    torch.testing.assert_close(o, r, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(x2.grad, x.grad, rtol=1e-3, atol=1e-4)
