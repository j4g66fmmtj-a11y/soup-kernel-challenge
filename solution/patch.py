"""apply(): подменяет слои Qwen2 на наши Triton-кернелы до построения модели Soup."""

USE_SWIGLU = True  # поставьте False, если SwiGLU замедляет обучение


def apply(*args, **kwargs):
    from transformers.models.qwen2 import modeling_qwen2 as m

    from . import kernels, logsoftmax

    if kernels._orig_rms is None:  # защита от повторного вызова
        kernels._orig_rms = m.Qwen2RMSNorm.forward
        kernels._orig_rope = m.apply_rotary_pos_emb
    m.Qwen2RMSNorm.forward = kernels.rmsnorm_forward
    m.apply_rotary_pos_emb = kernels.apply_rotary_pos_emb
    logsoftmax.patch_log_softmax()
    if USE_SWIGLU:
        from .swiglu_fused import patch_qwen2_mlp
        patch_qwen2_mlp()
    return True
