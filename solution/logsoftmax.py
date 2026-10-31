"""Собственный Triton-кернел log_softmax по большому словарю (forward + backward).

Подменяет F.log_softmax / torch.log_softmax / Tensor.log_softmax и F.cross_entropy
только для матриц fp32 на GPU с последним измерением >= 32768 (словарь Qwen: 151936).
Всё остальное идёт через оригинальные функции PyTorch.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MIN_VOCAB = 32768
_NEG = -1.0e30


@triton.jit
def _lsm_fwd(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    cols = tl.arange(0, BLOCK)
    m = tl.full([BLOCK], -1.0e30, dtype=tl.float32)
    s = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        idx = off + cols
        x = tl.load(X + base + idx, mask=idx < N, other=-1.0e30)
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
    mx = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - mx), axis=0)
    lse = mx + tl.log(total)
    for off in range(0, N, BLOCK):
        idx = off + cols
        x = tl.load(X + base + idx, mask=idx < N, other=0.0)
        tl.store(Y + base + idx, x - lse, mask=idx < N)


@triton.jit
def _lsm_bwd(DY, Y, DX, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        idx = off + cols
        acc += tl.load(DY + base + idx, mask=idx < N, other=0.0)
    gsum = tl.sum(acc, axis=0)
    for off in range(0, N, BLOCK):
        idx = off + cols
        mask = idx < N
        dy = tl.load(DY + base + idx, mask=mask, other=0.0)
        y = tl.load(Y + base + idx, mask=mask, other=0.0)
        tl.store(DX + base + idx, dy - tl.exp(y) * gsum, mask=mask)


class LogSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        _lsm_fwd[(M,)](x2, y, N, BLOCK=4096, num_warps=8)
        ctx.save_for_backward(y)
        ctx.shape = x.shape
        return y.reshape(x.shape)

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        M, N = y.shape
        dy2 = dy.reshape(M, N).contiguous()
        dx = torch.empty_like(y)
        _lsm_bwd[(M,)](dy2, y, dx, N, BLOCK=4096, num_warps=8)
        return dx.reshape(ctx.shape)


def _eligible(x, dim, kwargs, args):
    if args or kwargs.get("dtype") is not None or not set(kwargs) <= {"dtype", "_stacklevel"}:
        return False
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32):
        return False
    if x.dim() < 2 or x.shape[-1] < MIN_VOCAB:
        return False
    return dim in (-1, x.dim() - 1)


_orig = {}


def _make_lsm(orig):
    def fn(input, dim=None, *args, **kwargs):
        if _eligible(input, dim, kwargs, args):
            return LogSoftmaxFn.apply(input)
        return orig(input, dim, *args, **kwargs)
    return fn


def _tensor_lsm(self, dim=None, *args, **kwargs):
    if _eligible(self, dim, kwargs, args):
        return LogSoftmaxFn.apply(self)
    return _orig["tensor"](self, dim, *args, **kwargs)


def _cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                   reduce=None, reduction="mean", label_smoothing=0.0):
    if (weight is None and size_average is None and reduce is None and label_smoothing == 0.0
            and input.dim() == 2 and target.dim() == 1 and not target.is_floating_point()
            and _eligible(input, -1, {}, ())):
        logp = LogSoftmaxFn.apply(input)
        return F.nll_loss(logp, target, ignore_index=ignore_index, reduction=reduction)
    return _orig["ce"](input, target, weight=weight, size_average=size_average,
                       ignore_index=ignore_index, reduce=reduce, reduction=reduction,
                       label_smoothing=label_smoothing)


def patch_log_softmax():
    if _orig:
        return
    _orig["torch"], _orig["F"] = torch.log_softmax, F.log_softmax
    _orig["tensor"], _orig["ce"] = torch.Tensor.log_softmax, F.cross_entropy
    torch.log_softmax = _make_lsm(_orig["torch"])
    F.log_softmax = _make_lsm(_orig["F"])
    torch.Tensor.log_softmax = _tensor_lsm
    F.cross_entropy = _cross_entropy
