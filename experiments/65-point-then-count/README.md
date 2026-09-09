# Rung 65 — point-then-count, executed at last and closed

> **Status: CLOSED 2026-09-07.** Zero training, 38 GPU-min. Counting the checkpoint's emitted
> coordinates **loses to the answer it already gives**, by −0.083.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 16c-count-probes | pointing on the un-fine-tuned backbone | — | "collapses to one point" |
| 19-external-count | external counting data as `number` targets | A2 | shipped inside rung 42's corpus |
| **65 — point-then-count** | **elicit points, count them outside the model** | r42 ep4's own answer | 🔴 **DEAD** — −0.083 |

## The result — arm B, the shipped `merged_r42`, 518 `number` rows

| | count the emitted points | the model's own direct answer |
|---|---|---|
| p1 | 0.3108 | **0.3938** |
| p2 | 0.3185 | **0.3938** |

    emission           518/518 on both prompts
    mean points        2.82 (p1) / 2.80 (p2)     mean gold  3.672
    collapsed to 1     42.9 % / 41.3 %

Not merely no better — **worse than doing nothing**, and the mechanism is legible: it under-emits,
and four times in ten it emits a single point regardless of the scene.

## 🟢 The keeper: the gate's declared risk was backwards

G1 asked whether SFT had **erased** the pointing pathway from the backbone's PixMo pre-training.
Arm C ran the same prompts on the un-fine-tuned backbone as a control:

| | emission | collapsed to 1 point |
|---|---|---|
| backbone, no fine-tune | 0.45 / 0.55 | 88.9 % / 77.3 % |
| **shipped `r42`** | **1.00 / 1.00** | 42.9 % / 41.3 % |

⇒ **Our SFT did not cost us the pointing route — it repaired it.** Emission went to 1.000 and
the collapse roughly halved. What fails is not emission; it is that the points do not individuate
the objects well enough to be counted. This also retires rung 16c's *"collapses to one point"* as
a description of the shipped model: that was measured on the **un-trained** one.

## ⚠️ The smoke was fit for its gate and useless as a preview

The 40-row smoke returned `acc_own_all = 0.0` for the model that scores **0.3938** on the 518.
It read the emission rate correctly (1.00), which is what it guarded. Do not let a cost gate's
smoke double as an estimate of the effect — they size for different quantities.

## Files

    RESULTS_ptc_B_full.json    518 rows x 2 prompts on merged_r42 (the verdict)
    RESULTS_ptc_B_smoke.json   the 40-row gate — emission only, see the warning above
    RESULTS_ptc_C_smoke.json   the un-fine-tuned backbone control

Verdict and links: [[point-then-count-loses-to-the-direct-answer]].
