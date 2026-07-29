---
question: Was the -0.150 `object_recognition` collapse our wrong judge?
verdict: NO. The official judge moves the headline by -0.0014 and moves `object_recognition` ID UP by +0.0039. The instrument was wrong; it is not the gap.
status: MEASURED
date: 2026-07-28
measured_in: experiments/20-judge-swap/RESULTS.csv + runs/20_judge_swap_v1/summary.json
question_derived: false
---
# Finding: the wrong judge was real and it cost us nothing

[[wrong-judge-model]] established that every judged number in this repo came from
`Qwen/Qwen3-4B` while the SDK defaults to `Qwen/Qwen3.5-4B`. This re-scores rung 06 ep3's
committed predictions under **both judges in the same process on the same GPU** — no VLM
inference, no training, ~7 GPU-minutes total.

| | substitute `Qwen3-4B` | official `Qwen3.5-4B` | Δ |
|---|---|---|---|
| `bucket_mean` (headline) | 0.5724 | 0.5710 | **−0.0014** |
| `object_recognition` ID | 0.6520 | **0.6559** | **+0.0039** |
| `object_recognition` OOD | 0.6485 | 0.6391 | −0.0094 |
| `aggregation` ID / OOD | 0.4346 / 0.5547 | identical | **0.0000** |

`aggregation` is byte-identical because **no `aggregation` question is judged** — the judged
formats live entirely inside `object_recognition`.

⇒ **The −0.150 platform collapse is not our measuring stick.** It survives the correct judge
intact, so it is real behaviour on unseen centres. The remaining branch is the one
pre-registered in `HANDOFF_RUNG19.md`: a newer-**generation** backbone, not a bigger one.

## Agreement is high but not perfect, and the disagreements are one-shaped

99.2 % over all 6,252 questions; 93.5 % over the 759 judged ones. All 49 disagreements are
`open_ended` — **`multiple_choice` agrees on 202 of 202**. 40 of the 49 are OOD, and the
substitute was the lenient one in 32 of them.

The rows are committed at `experiments/20-judge-swap/DISAGREEMENTS.csv`. Nearly all are the
"relative central positions" template, and the failure inside them is not the judge:

```
REF: 1. Sponge: bottom/left      ANS: 1. Clip: bottom/right
REF: 1. Needle: top/left         ANS: 1. Clip: top/left
REF: 1. Sponge: bottom/left      ANS: 1. Silicone loop: bottom/left
```

The model emits the right structure and **the wrong class, collapsing toward `Clip`** — the
majority class ([[class-imbalance-not-counting]]). The substitute judge was calling those
CORRECT. So the corrected instrument is also a slightly *sharper* one on exactly the
capability that is failing.

⚠️ The official judge is not uniformly stricter: in 10 OOD rows it accepted answers the
substitute rejected (e.g. reference `Mesentery.` vs answer `Small intestine.` → CORRECT).
Both judges are noisy on `open_ended` at roughly the same rate; the noise nets out.

## What this does NOT settle

The rubric mismatch from [[wrong-judge-model]] stands and is **still unmeasured**:
`judges.py:45-63` says *"If the candidate contains the correct information but also includes
extraneous text, still judge it CORRECT"*, while our SFT trains terse template-locked strings.
This experiment shows the two *judges* agree — it says nothing about whether a *less terse
output policy* would score better under either. That is a separate, cheap experiment.

## Operational consequence: the official judge cannot run in the VLM's environment

`Qwen/Qwen3.5-4B` declares `model_type: qwen3_5`, which `transformers` 4.57 does not register
(`KeyError: 'qwen3_5'` from `AutoConfig`). 4.57 is the hard floor for Qwen3-VL and ms-swift
caps below 5.13, so **the judge must run in its own venv** — `/workspace/envs/judge35`,
transformers 5.14.1, torch inherited from the image. This is almost certainly where the false
"Qwen3.5-4B does not exist" comment came from: someone saw the load fail and concluded the
model was not real.

⚠️ Trap recorded: installing `accelerate` into that venv pulled `torch 2.13.0+cu130`, which
shadowed the image's 2.8.0+cu128 and desynced it from `torchvision 0.23+cu128`. The symptom is
misleading — `ModuleNotFoundError: Could not import module 'Qwen3_5ForCausalLM'`, whose real
cause is `RuntimeError: operator torchvision::nms does not exist`. Fix: uninstall torch and
torchvision from the venv so it falls back to the image's matched pair.

## Adoption

Switching the project's judge is its own call and is NOT made here. What is settled: the two
judges are interchangeable at the headline (|Δ| ≈ 0.001), so **no past number in the repo needs
restating**, and the ladder stays comparable either way.
