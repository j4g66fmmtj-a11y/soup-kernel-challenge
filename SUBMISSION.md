# Submission: HeriotAI

## Links
- Repository, final commit: https://github.com/j4g66fmmtj-a11y/soup-kernel-challenge/commit/<0672544>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/baqytse/notebook10cb55eaa6?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team
| Madina | Coder | j466fmmtj-a11y | madinaakhmetulla | chestnutaries |
| Baqyt| Coder | baqyt1337 | baqytse | qoyuqara |
| Eskendir | Coder| --- | Roboty01 | Roboty01 |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_rms_fwd`, `_rms_bwd` | `solution/kernels.py` | Qwen2RMSNorm | see `runs/checks.json`: <x> % for all own kernels together |
| `_rope` | `solution/kernels.py` | apply_rotary_pos_emb | (included above) |
| `_lsm_fwd`, `_lsm_bwd` | `solution/logsoftmax.py` | log_softmax / cross_entropy over the large vocabulary | (included above) |
| `_swiglu_fwd`, `_swiglu_bwd` | `solution/swiglu_fused.py` | silu(gate)*up in Qwen2MLP (close to the starter example, not counted as ours) | n/a |

Ideas taken from other kernels, with links (we wrote the code ourselves): Triton fused layer-norm
and softmax tutorials (https://triton-lang.org/main/getting-started/tutorials/).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
=== result ===
base      time     84.8 s   peak  14345 MiB   eval loss 2.04967
solution  time     75.8 s   peak  12381 MiB   eval loss 2.04966
speedup 1.119x   memory base/solution 1.159x   eval loss vs base -0.0 %
written: /kaggle/working/repo/runs/result.json

PASS  1. every base and solution run finished                                            1 base and 1 solution runs of 1 each
PASS  2. no ready-made kernel libraries                                                  loaded: none; imported in solution/: none; packages changed during the run: none
PASS  3. own @triton.jit kernels in solution/ run in training                            _lsm_bwd (solution/logsoftmax.py), _lsm_fwd (solution/logsoftmax.py), _rms_bwd (solution/kernels.py), _rms_fwd (solution/kernels.py), _rope (solution/kernels.py); the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
PASS  4. own kernels together take >= 5% of the training time                            8.6% of the training time (lowest of 1 runs, steps 3+)
PASS  5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind)  identical within tolerance
PASS  6. loss curve follows the base                                                     solution_1: first step 0.02%, mean of all steps 0.01% (max 1% and 3%)
PASS  7. quality: eval loss at most 0.5% above the base                                  -0.000%
PASS  8. not worse than the base on time or memory (10% margin)                          time x0.894, memory x0.863 of the base
PASS  9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree)    clean
PASS  10. Soup regression tests with your patches                                        79 passed, 14 warnings in 2.26s

ALL CHECKS PASSED
speedup 1.119x (counts as 1.119), memory base/solution 1.159x (counts as 1.159), eval loss vs base -0.000 %, all checks passed on this result: True
Speed and memory points depend on the best admitted team: 35 * ln(s) / ln(best s) and 30 * ln(m) / ln(best m).
## Presentation covers
Which operations we sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
The kernels cover the element-wise operations and the log-softmax; the 4-bit matmuls are untouched
and the full-vocabulary logits are still materialised. Next: chunked fused cross-entropy, NF4
dequantisation fused into LoRA.
