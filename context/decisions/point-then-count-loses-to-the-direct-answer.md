---
question: [[counting-is-a-mapping-failure]] verdicted "YES as POINT supervision, counting the emitted coordinates outside the model" and cited 94.96 % vs a fine-tune's 46.19 %. It was never executed. Does counting the shipped checkpoint's emitted points beat the answer that checkpoint gives directly?
verdict: NO — counting the points LOSES by −0.083. On 518 `number` rows, points scored 0.3108 (p1) / 0.3185 (p2) against the model's own direct answer at 0.3938, with 41–43 % of cases collapsing to a single point and a systematic under-emission (2.8 points vs 3.67 gold). Zero training, closed in 38 GPU-min. The 94.96 % never transferred to our checkpoint. 🟢 The gate's declared risk was BACKWARDS and that is the keeper: SFT did not erode pointing, it repaired it
status: MEASURED
date: 2026-09-07
measured_in: rung 65 — 518 rows × 2 prompts on the shipped `merged_r42`, 37.9 min, plus a 40-row control on the un-fine-tuned backbone
---

# Decision: point-then-count loses to the answer the model already gives

- **Applies when:** costing any inference-time re-formulation that asks the checkpoint to answer
  in a different representation and derives the scored answer from it.

## The result

Arm B, the shipped `merged_r42`, 518 `number` rows, emission 518/518 on both prompts:

| | count the emitted points | the model's own direct answer |
|---|---|---|
| p1 | 0.3108 | **0.3938** |
| p2 | 0.3185 | **0.3938** |

    mean points emitted   2.82 (p1) / 2.80 (p2)      mean gold  3.672
    collapsed to 1 point  42.9 % (p1) / 41.3 % (p2)

The route is not merely no better — it is **−0.083 worse than doing nothing**, and the mechanism
is visible: it under-emits, and in four cases out of ten it emits a single point regardless.

## 🟢 The keeper: the gate's declared risk was inverted

G1 asked whether SFT had **erased** the pointing pathway inherited from the backbone's PixMo
pre-training — if the checkpoint emitted coordinates in under 50 % of cases, the collapse *was*
the result. Arm C ran the same prompts on the un-fine-tuned backbone as the control:

| | emission rate | collapsed to 1 point |
|---|---|---|
| backbone, no fine-tune | 0.45 / 0.55 | 88.9 % / 77.3 % |
| **shipped `r42`** | **1.00 / 1.00** | 42.9 % / 41.3 % |

⇒ **Our SFT did not cost us the pointing route. It roughly halved the collapse and took emission
to 1.000.** The pathway is in better shape after fine-tuning than before it — the opposite of
what the note feared. What fails is not emission; it is that the emitted points do not
individuate the objects well enough to be counted.

🔻 This also retires the single piece of counter-evidence [[counting-is-a-mapping-failure]]
carried against itself — rung 16c's *"collapses to one point"* was measured on the **un-trained**
model, and this rung shows that measurement does not describe the shipped one.

## ⚠️ A methodology note worth more than the result

The 40-row smoke returned `acc_own_all = 0.0` for the same model that scores **0.3938** on the
518. The smoke was fit for the gate it guarded (emission rate, which it read correctly at 1.00)
and **useless as an estimate of the signal**. Do not let a gate's smoke double as a preview of
the effect — they size for different quantities.

## Why the 94.96 % did not transfer

It was measured on other models, other data, with **point supervision in training**. We ran
point *elicitation* at inference on a checkpoint trained on `number` targets. The decision note
said "as POINT supervision" and the honest reading is that its verdict was never tested here:
what is now closed is the **zero-training inference-time form**. Training on point targets
remains unmeasured — and out of scope, since it needs a training run and the model deadline is
2026-09-11 ([[the-final-bullet-is-rung-61]]).

## Links

- [[counting-is-a-mapping-failure]] — the verdict this executed, and partly retires
- [[the-gap-is-the-number-format]] — `number` is 71 % of r42's errors, which is why this was tried
- [[self-consistency-dead]] — the other inference-time counting lever that measured flat
