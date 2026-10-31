# Soup Kernel Challenge: fused Triton kernels for Qwen2.5-1.5B QLoRA fine-tuning

Team: HeriotAI (Decentrathon 6.0). Public Kaggle notebook: https://www.kaggle.com/code/baqytse/notebook10cb55eaa6

## What we did
We replace small PyTorch operations of the Qwen2 decoder and of the loss with our own Triton
kernels (forward and backward, wrapped in `torch.autograd.Function`). The task, data, batches,
LoRA settings, optimizer and Soup's settings are not touched.

| Kernel | File | Replaces |
|---|---|---|
| `_rms_fwd`, `_rms_bwd` | `solution/kernels.py` | `Qwen2RMSNorm`: fp32 cast, square, mean, rsqrt, multiply, cast back, weight multiply, in one launch per row |
| `_rope` (forward and inverse rotation) | `solution/kernels.py` | `apply_rotary_pos_emb`: the rotate-half, two multiplies and an add, for q and k |
| `_lsm_fwd`, `_lsm_bwd` | `solution/logsoftmax.py` | `log_softmax` / `cross_entropy` over the 152k-token vocabulary (fp32, large vocabularies only; everything else goes to the original PyTorch functions) |
| `_swiglu_fwd`, `_swiglu_bwd` | `solution/swiglu_fused.py` | `act_fn(gate) * up` in `Qwen2MLP`; it follows the structure of the example in the starter kit, so we do not count it as an original kernel |

Design notes:
- RMSNorm: one program per row, the mean of squares in fp32, the normalised value is rounded to the
  input dtype before the weight multiply, like HF. The backward pass saves only `rstd`. The weight
  gradient is computed only when the weight requires it (frozen under LoRA).
- RoPE: one program per (batch, head, position); the backward pass is the same kernel with the
  rotation sign flipped. Unsupported shapes fall back to the original function.
- log_softmax: online (running max and sum) pass over the vocabulary, then a second pass writes
  `x - logsumexp`; the backward pass is `dy - exp(y) * sum(dy)`.
- `solution.apply()` (in `solution/patch.py`) patches all of them before Soup builds the model.
- No ready-made kernel libraries are used; the formulas were derived by hand and are checked in
  `solution/test_own_kernels.py` against plain PyTorch.

## How to run
1. Open the public Kaggle notebook above (or import `templates/kaggle_notebook.ipynb`), set
   Accelerator to GPU T4 x2 and Internet to On (phone-verified account).
2. Set `REPO_URL` to this repository and `COMMIT` to the commit in `SUBMISSION.md`, then Run All
   (about 10 minutes). It trains the base and our version, scores both adapters and runs the checks.
3. Tests of the kernels on a GPU: `python -m pytest solution/test_own_kernels.py -q`.
4. Locally: `python harness/run_task.py` (see the harness README section for options).
5. To switch the SwiGLU kernel off, set `USE_SWIGLU = False` in `solution/patch.py`.

## Results
Exact numbers of the saved notebook version (time, peak memory, eval loss, own-kernel share) are in
`SUBMISSION.md` and in `runs/result.json` / `runs/checks.json` of that version.

## Limits and next steps
- Our kernels cover the small element-wise operations and the log-softmax; the largest remaining
  costs are the 4-bit matrix multiplications. The log-softmax kernel still materialises the full
  logits, so it is not the memory fix a chunked loss would be.
- Next: a fused, chunked cross-entropy that never materialises the full logits (the main memory
  lever), keeping the entropy/accuracy metrics that TRL and the rewind log read from the logits;
  fusing the NF4 dequantisation into the LoRA path; removing pad tokens inside a batch.
- Ideas about kernel structure come from public descriptions of fused RMSNorm, RoPE and softmax
  kernels (for example the Triton tutorials); the code is our own.
