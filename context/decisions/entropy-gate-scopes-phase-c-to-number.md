---
question: Does GRPO have a gradient and a ceiling to work with on our own policy — and does the answer hold across an independent sample?
verdict: Phase C survives, scoped to `number` only, and the scoping REPLICATES across two independent seeds. `zero_advantage` is 0.280 / 0.270 for `number` against a 0.60 kill line, while `binary` (0.720 / 0.735) and `fo_class` (0.660 / 0.705) are dead in both. The decision rests on `zero_advantage`, which is stable to ±0.015; `headroom` is NOT stable and `fo_class` halved between samples (+0.090 → +0.045), so the conjunction is what saved that verdict, not its margin.
status: RE_SCOPED
date: 2026-08-06
basis: 2x 1h on 1x RTX 5090 — 600 train questions each, k=8, T=1.0, on A2 ep3 (`21_lr_2e4_v1/checkpoint-2703`), scored by the SDK verifier (9 evaluator passes per run). SEED 42 ran 2026-08-05, SEED 43 on 2026-08-06; both thresholds pre-registered before the first run
---

# The entropy gate replicates: phase C is alive, and only for `number`

## What was being asked

GRPO needs two things our checkpoint may not have. **A gradient**: if all k rollouts on a
question earn the same reward, the advantage is zero and the question contributes nothing.
**A ceiling**: if sampling cannot beat greedy, there is nothing to climb toward. Both come out
of one pass, and both are read **per `answer_format`, never averaged** — the formats are
different problems and one live format is worth scoping to.

| number | kills the phase if |
|---|---|
| `zero_advantage_frac` | **>= 0.60** |
| `pass@k − greedy` (headroom) | **< +0.05** |

Questions come from **train**, not val: this measures the policy where GRPO would actually run
and spends no validation signal.

## What it gave us — both samples

| format | n | `zero_adv` s42 | `zero_adv` s43 | headroom s42 | headroom s43 | verdict |
|---|---|---|---|---|---|---|
| `binary` | 200 | 0.720 | 0.735 | +0.050 | +0.040 | 🔴 DEAD (both) |
| `fo_class` | 200 | 0.660 | 0.705 | +0.090 | +0.045 | 🔴 DEAD (both) |
| **`number`** | 200 | **0.280** | **0.270** | **+0.240** | **+0.310** | 🟢 **ALIVE (both)** |

🟢 **All three verdicts replicate.** Phase C does not die — by the rung's own pre-registered rule
a single live format keeps it, scoped to that format. ⇒ **GRPO with a set-F1 reward is scoped to
`number`; rung 22 (loss-mass) stays parked as the pre-decided Plan B and is NOT triggered.**

🟢 **`number` does not scrape through.** 0.28 against a 0.60 kill line, and headroom of +0.24 to
+0.31 against a +0.05 minimum — 5x to 6x. Sampling finds the right answer 91–95% of the time
where greedy scores 60–71%: **24 to 31 points the model can already reach and does not.** And it
lands where it pays — `number` is 80.4% of `aggregation` ([[the-gap-is-the-number-format]]), one
of the four scored buckets.

🔴 **The two dead formats die of no gradient, not of no ceiling** (`mode_share` 0.94 and 0.90 —
the model repeats itself). More rollouts would not help; a different reward might.

## What the second seed actually bought — and it is not reassurance about everything

🟢 **`zero_advantage` is a number.** It moves ±0.015 at worst across two independent draws
(`number` ±0.010). The decision that scopes phase C rests on this quantity, and it is 0.33 away
from its threshold — no plausible sample crosses it.

🔴 **`headroom` is a range, and one verdict leaned on it.** It is a difference of two accuracies,
so it inherits both their sampling errors:

| | s42 | s43 | delta |
|---|---|---|---|
| `fo_class` headroom | +0.090 | +0.045 | **−0.045** |
| `number` greedy | 0.705 | 0.600 | **−0.105** |
| `number` `pass@8` | 0.945 | 0.910 | −0.035 |

⚠️ `fo_class` **passed** the headroom leg at seed 42 (+0.090 vs a +0.05 floor) and **fails** it at
seed 43. Its verdict never changed because it dies on `zero_advantage` in both — **the conjunction
saved it, not the margin.** A format whose life depended on headroom alone would have received two
different answers from two equally valid samples.

🔑 **The transferable finding is about the instrument.** Greedy accuracy on `number` moved **10.5
points** between two independent 200-question draws. At p ≈ 0.65 and n = 200 the standard error of
that difference is ~0.048, so 0.105 is ~2 se — noise, but large noise. ⇒ **a per-format accuracy
delta below ~0.10 on n=200 is not separable from which questions were drawn.** This is the same
class of limit the video jackknife found for `acc_OOD` (median shift 0.024 from dropping one
video, `RESULTS_jackknife_by_video.csv`), and it bears directly on `RULES §S1–S7`
([[significance-rule]]): a small delta measured this way has no sign to transfer.

🟢 **One loose end from seed 42 closes on its own.** `binary`'s headroom was recorded as a float
artifact — 0.985 − 0.935 = `0.04999999999999993`, killed by a strict `<` on a threshold it exactly
meets. At seed 43 it is 0.040, clearly below. The verdict never depended on that epsilon (it also
fails `zero_advantage` in both), but the comparison still wants a tolerance before another arm
dies of floating point.

## Cost, corrected for the record

**The full run is ~58 minutes, not 1h48.** Papermill's own cell timings for seed 42:
`7/11 [57:09]`, total `11/11 [57:47]` — 56.7 min of it the generation cell, and the nine
evaluator passes together take **~38 s**. Seed 43 reproduced this at 62 min. The 1h48 recorded on
2026-08-05 included the smoke and the artifact chain around it. ⚠️ Also measured while
diagnosing: `FrameProvider` (`src/frame/data.py:154`) reads **straight from the source video with
decord and never touches `/workspace/frames_cache`**, so during a run the GPU sits near 0% while
one CPU thread seeks inside multi-GB AVIs on the network volume. A stalled-looking run is the
normal profile, and the frame cache not growing is not a symptom of anything.

## Sources

- `experiments/10-self-consistency/` — `10b_entropy_gate.ipynb`
- `RESULTS_step5_gate.csv` / `RESULTS_step5_verdict.json` / `RESULTS_step5_per_question.csv` (seed 42)
- `RESULTS_step5_gate_seed43.csv` / `RESULTS_step5_verdict_seed43.json` / `RESULTS_step5_per_question_seed43.csv`
- Step 5 of the team's August plan; not a rung ([[august-plan-closes-the-ladder]])
