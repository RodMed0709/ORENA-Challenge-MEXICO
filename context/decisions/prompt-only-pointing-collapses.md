---
question: Does deriving the count as len(predicted_points) beat verbalising it, as Alghisi 2026 reports (+80.9 pts OOD)?
verdict: NOT PROMPT-ONLY — both pointing arms are WORSE than the bare integer in every cell, and the mechanism is visible: asked to point, the model emits exactly ONE point (a2 mean_pred = 1.0000 on all 681 questions). It does not enumerate. This bounds the PROMPT-ONLY path, which Alghisi themselves predicted would be weak; it does not bound the trained path
status: MEASURED
date: 2026-07-28
measured_in: experiments/16-count-probes/RESULTS_16c.csv (n=681, the full Clips template, 37 videos)
---

# Decision: prompt-only point-then-count collapses to a single point

- **Status:** MEASURED · 2026-07-28 · ~25 min of 5090, no training
- **Applies when:** costing any lever whose theory is *"make the model enumerate, then read the
  count off the enumeration"* — `len(points)`, numbered points, `#Coord`, point-then-count.

## The claim under test

Alghisi, Rizzoli, Mousavi, Riccardi 2026 (*Getting to the Point*, arXiv:2603.21746) on
Qwen2.5-VL-7B: on their OOD split (count-range extrapolation — structurally our shape), direct
count **23.41%**, point-then-count **verbalised** 14.04%, and **`#Coord` = `len(points)`
94.96%**. The lever is not pointing; it is *refusing to let the model state the number*.
**+80.9 points** — the largest single reported effect available to us.

Rung 15 replicated Gautam's structured count *format* and returned a null. It never tested this.

## What was measured

Three arms on rung 06 **ep3** (`checkpoint-2580`), single variable = the answer target, image and
question byte-identical across arms. Scored on the **full `Clips` template**: 681 val questions,
12 distinct true values, 37 videos — the single largest hole in the exam.

| arm | target | ID margin | OOD margin | ALL margin | mean_pred | bias | s/q |
|---|---|---|---|---|---|---|---|
| **a0** bare integer (control) | `"3"` | **+0.0877** | **−0.0025** | **+0.0367** | 2.85 | −0.661 | 0.37 |
| **a1** `point_2d` JSON, count = `len()` | native `[0,1000]` | +0.0316 | **−0.1162** | −0.0529 | 2.48 | −1.038 | 1.25 |
| **a2** numbered points, count = last index | Molmo style | −0.0456 | **−0.1389** | −0.0984 | **1.0000** | −2.515 | 0.50 |

Margin is over the template-aware trivial floor (0.1825 ID / 0.2828 OOD). Malformed: **0.0%** in
all three arms — the parsers worked; the arms simply lost.

## What it gives us

### 1. Both pointing arms are worse than the control, everywhere

Not a wash — a loss, in every cell, on both distributions. And at **3.4× the latency** for a1.

### 2. The mechanism is visible, and it is degenerate

🔴 **a2's `mean_pred` is exactly 1.0000 across all 681 questions.** Asked to point at every clip
and number them, the model emits **one** point, every single time. a1 does slightly better (2.48)
but still under the gold mean of 3.52, and its bias worsens from −0.66 to −1.04.

**The model does not enumerate. It points once.** So `len(points)` is not reading a count off an
enumeration — it is reading a constant. That is why the derivation *amplifies* the undercount
instead of curing it: a2's bias is **−2.52** against the bare integer's −0.66.

This is the same shape as [[naming-equals-counting]] (asked to NAME or to COUNT, the same frames
return the same multiplicity) one level out: asked to POINT, the model returns one object.

### 3. The probe validated its own instrument

`a0` OOD margin is **−0.0025**, reproducing the epoch-matched control's headline
(`number_margin_OOD` = 0.0 for rung 06 ep3) on an independent code path. The instrument reads
the number we already know before it reads the ones we do not.

## ⚠️ What this does NOT close

🔴 **It bounds the PROMPT-ONLY path, not the trained one — and Alghisi predicted exactly this.**
Their paper states training-free point-then-count is weak without task-specific supervision; our
result is the measured version of their caveat, on our backbone, in our domain. Their 94.96% was
obtained *with LoRA fine-tuning on point supervision*.

So the honest reading is: **`len(points)` is dead as an inference-time trick, and remains open as
a training target** — gated on 16b, which asks whether the gold corresponds to visible instances
at all. If 16b says the gold is not frame-visible, the trained version dies too and the whole
pointing line closes. That gate has to run before anyone spends a training run here.

Two further caveats carried from the source, both of which survive this null:
- A **black-screen substitution costs <2%** in their setup — the count is read off the model's own
  text, not re-derived from pixels. A win would not automatically have been a perception win.
- Coordinate↔count self-consistency runs as low as **20.56%**, which is why the deterministic
  `len()` fallback is mandatory rather than optional.

## Sources

- `experiments/16-count-probes/RESULTS_16c.csv` + `runs/16c_len_points_v1/`
- Alghisi et al. 2026, arXiv:2603.21746 (the claim, and the caveat this result confirms)
- Deitke et al. 2024, arXiv:2409.17146 (Molmo numbered-points serialisation = arm a2)
- ⚠️ Convention trap: Qwen2-VL `[0,1000]` → Qwen2.5-VL absolute pixels → Qwen3-VL back to
  `[0,1000]` with `point_2d` in JSON. Arm a1 used our backbone's **native** convention on
  purpose; Gautam and Alghisi both ran Qwen2.5-VL, whose serialisation is the wrong one for us.
  So this null is **not** explainable as a convention mismatch.
