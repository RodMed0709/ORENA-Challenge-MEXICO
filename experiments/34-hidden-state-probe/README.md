# Rung 34 — the count IS in the hidden states. The head is what loses it.

**🟢 THESIS CONFIRMED.** A linear probe on the hidden state at the last prompt position, fitted on
`lapchole` (ID) videos only and read on the `heico` (OOD) half it never saw, beats the model's own
token output on the same 2,094 questions:

| | `number` accuracy on OOD |
|---|---|
| the model's token argmax (rung 33) | 0.4680 |
| **linear probe, layer 24, ID-fit → OOD-read** | **0.5264** |
| | **+0.0584** |

## The depth profile is the finding, not the single number

| layers | OOD transfer | reading |
|---|---|---|
| 0 – 16 | 0.365 – 0.437 | **below** the token head — the count is not yet linearly available |
| **18** | 0.466 | crosses it |
| **20 – 24** | 0.502 → 0.513 → **0.526** | the count becomes decodable, and peaks |
| 26 – 36 | 0.500 – 0.516 | it is still there when the model speaks, and the head does worse |

The count is **not** present early, **becomes** linearly decodable around **layers 18–24**, and is
**still present at layer 36** — the last state before the token head. The model holds the answer
and emits a worse one.

📌 That band is independently named: *Counting Circuits* (arXiv 2603.18523) localises
**cross-modal routing heads at layers 18–24** on **Qwen3-VL-8B, our exact backbone**, and calls
them the visual→numeric handover. We found the same band from the other direction, on our own
fine-tuned checkpoint, without looking for it.

## Why this verdict is admissible and the first one was not

The first fit returned `acc_ID_insample = 1.0` on **every** layer while transferring at 0.25–0.43.
4,096 features against ~768 ID rows separates any labelling perfectly — that probe memorised the
training half, and falsifying the campaign's central thesis on it would have been indefensible.

Corrected: PCA to 128 components **fitted on the ID half only** (never on the OOD half it is read
on), then a sweep over five decades of `C`. The honest fit now shows `acc_ID_insample ≈ 0.48`
against `acc_OOD_transfer ≈ 0.53` — in-sample and out-of-sample agree, so it is not memorising.
(The OOD half scores higher because `heico`'s count distribution is easier, not because the probe
prefers it.)

🔑 **And the transfer direction matters.** ID-fit → OOD-read is exactly where the value LUT died
([[count-calibration-dead]], **−0.016**). The probe gets **+0.058** on the same direction.

## What it settles

1. 🟢 **[[counting-is-a-mapping-failure]] is now measured, not inferred.** The sentence "the model
   SEES the objects and cannot EMIT the number" rested on a rank correlation between its
   *answers* and gold, which is compatible with the model tracking scene complexity and knowing
   nothing. It is now demonstrated on its *representations*.
2. 🟢 **The loss is localised to the mapping, downstream of layer ~24.** Perception is not the
   bottleneck at that depth, and any lever aimed at seeing better is aimed at the wrong stage.
3. 🟢 **We hold a second answer channel** that does not pass through the broken head — worth
   **+0.058 on `number`**, i.e. ≈ **+0.023 on the headline**. ⚠️ Above the readable floor, below
   the S1 ship bar of 0.03, so on its own it is not a submission; combined with the training-side
   family it is a different conversation.
4. 🔴 **It reframes rung 33's failure.** Gate A failed because the gold is often not even the
   runner-up *in token space* (0.456) — and that is now explained rather than merely observed:
   the information does not survive the projection into tokens. A decoding fix was working on the
   wrong side of the break.

## Caveats, stated

* One ID/OOD split, not a k-fold over videos. The direction is the hard one and the margin is
  +0.058, but the confidence interval is not computed here.
* Shipping this means `predict()` reads a hidden state and consults a fitted probe. We control the
  container and release everything, so it is permitted — but it is a new failure surface in the
  serving path, not a free win.
* The probe is fitted on gold. It is a *measurement* of what the model knows, and turning it into
  a shipped component requires fitting it on TRAIN only and re-reading it on the eval.

Artifacts: `_tools/hidden_probe.py`, `RESULTS_probe_full.csv`, `RESULTS_verdict_full.json`.
Env: the only package this rung adds is `scikit-learn 1.9.0`.

Ties: [[counting-is-a-mapping-failure]] (confirmed at last) · [[counting-has-two-failure-modes]] ·
[[count-calibration-dead]] (whose transfer direction this reverses) · rung 33 (whose gate this
explains).
