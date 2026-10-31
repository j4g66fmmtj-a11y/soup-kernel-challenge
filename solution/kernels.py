"""Собственные Triton-кернелы команды: fused RMSNorm и fused RoPE (forward + backward)."""
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ RMSNorm
@triton.jit
def _rms_fwd(X, W, Y, RSTD, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    xn = (x * rstd).to(X.dtype.element_ty).to(tl.float32)  # как в HF: округление до dtype входа
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (xn * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_bwd(DY, X, W, RSTD, DX, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    xhat = x * rstd
    wdy = dy * w
    c = tl.sum(xhat * wdy, axis=0) / N
    tl.store(DX + row * stride + cols, ((wdy - xhat * c) * rstd).to(DX.dtype.element_ty), mask=mask)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty((M, N), dtype=torch.promote_types(x.dtype, weight.dtype), device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK = triton.next_power_of_2(N)
        _rms_fwd[(M,)](x2, weight, y, rstd, N, N, eps, BLOCK=BLOCK,
                       num_warps=8 if BLOCK >= 2048 else 4)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy2 = dy.reshape(M, N).contiguous()
        dx = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _rms_bwd[(M,)](dy2, x2, weight, rstd, dx, N, N, BLOCK=BLOCK,
                       num_warps=8 if BLOCK >= 2048 else 4)
        dw = None
        if ctx.needs_input_grad[1]:
            dw = (dy2.float() * (x2.float() * rstd[:, None])).sum(0).to(weight.dtype)
        return dx.reshape(ctx.shape), dw, None


def rmsnorm_forward(self, hidden_states):
    if not hidden_states.is_cuda:
        return _orig_rms(self, hidden_states)
    return RMSNormFn.apply(hidden_states, self.weight, self.variance_epsilon)


_orig_rms = None


# ------------------------------------------------------------------ RoPE
# q_e = q*cos + rotate_half(q)*sin, rotate_half(x) = cat(-x2, x1). Обратное: поворот на -угол.
@triton.jit
def _rope(X, COS, SIN, Y, H, S, D, CB, sxb, sxh, sxs, SIGN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)            # индекс (b, h, s)
    s = row % S
    bh = row // S
    h = bh % H
    b = bh // H
    half = D // 2
    cols = tl.arange(0, BLOCK)
    mask = cols < half
    base = b * sxb + h * sxh + s * sxs
    x1 = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X + base + half + cols, mask=mask, other=0.0).to(tl.float32)
    cb = ((b % CB) * S + s) * D
    c1 = tl.load(COS + cb + cols, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(COS + cb + half + cols, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(SIN + cb + cols, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(SIN + cb + half + cols, mask=mask, other=0.0).to(tl.float32)
    ob = row * D
    if SIGN == 1:   # forward
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    else:           # backward
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    tl.store(Y + ob + cols, y1.to(Y.dtype.element_ty), mask=mask)
    tl.store(Y + ob + half + cols, y2.to(Y.dtype.element_ty), mask=mask)


def _launch_rope(x, cos, sin, sign):
    B, H, S, D = x.shape
    y = torch.empty((B, H, S, D), dtype=x.dtype, device=x.device)
    BLOCK = triton.next_power_of_2(D // 2)
    _rope[(B * H * S,)](x, cos, sin, y, H, S, D, cos.shape[0], x.stride(0), x.stride(1), x.stride(2),
                        SIGN=sign, BLOCK=BLOCK)
    return y


class RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _launch_rope(x, cos, sin, 1)

    @staticmethod
    def backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _launch_rope(dy.contiguous(), cos, sin, -1), None, None


def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
    # быстрый путь: q,k (B,H,S,D), cos/sin (B или 1, S, D), unsqueeze_dim=1; иначе оригинал
    unsq = kwargs.get("unsqueeze_dim", args[-1] if args else 1)
    ok = (unsq == 1 and q.is_cuda and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3
          and cos.shape == sin.shape and cos.shape[1] == q.shape[2] and cos.shape[2] == q.shape[3]
          and cos.shape[0] in (1, q.shape[0]) and q.shape[-1] % 2 == 0)
    if not ok:
        return _orig_rope(q, k, cos, sin, *args, **kwargs)
    cos = cos.contiguous()
    sin = sin.contiguous()
    return RopeFn.apply(q, cos, sin), RopeFn.apply(k, cos, sin)


_orig_rope = None
