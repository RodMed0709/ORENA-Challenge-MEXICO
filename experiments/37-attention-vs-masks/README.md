# Rung 37 — step 7: attention vs SAM masks

> 🟢 **STATUS 2026-08-11: the export has RUN. Nothing is adjudicated and no attention is read yet.**
>
> 🔒 **The number 37 is TAKEN — do not reassign it.** Renumbered from 36 on 2026-08-11 after a
> collision with `experiments/36-clip-sponge-probes/` (see `PLAN.md`). If you need a slot, take 38.
>
> **What exists now:** `_tools/export_masks_37.py` and `runs/37_masks_v1/` (on the pod, gitignored) —
> **45 frames**: the N=40 `G-BOUNDARY` sample, **5 per class across all eight** foreign-object
> classes, plus the **5 C2 pairs below the 0.90 floor**. The sample manifest is committed **outside
> `runs/`** as [`SAMPLE_37_masks_v1.json`](SAMPLE_37_masks_v1.json), because a number that moves a
> decision and lives only in a gitignored dir is the `RESULTS_controls.json` failure, twice over.
>
> ⏸️ **`G-BOUNDARY`'s formulation is still under review by legokna** and the export does not
> presume its outcome — it is the same 45 frames whichever way the gate is worded. **Do not
> adjudicate against the thresholds as currently written** until that review lands; a dry run over
> the 8 already-adjudicated C1 frames scores **B1 = 0.625** and fails both clauses, and all three
> `covered` failures are metallic clips, and on two of them the adjudicator's own note hedges on
> whether the object is visible at all (*"no sé si es mi sesgo"*, *"si es que los hay"*).
> 🔻 **CORRECTED 2026-08-11:** the first writing of this cited **r = −0.17** for the untrained eye.
> That number is **RETIRED** ([[gold-is-signal-model-underuses-it]]) — it was range-restricted to
> gold 3–6, and a blind full-range pass (n=106, 109 frames, 30 videos) scores **r = +0.7230** with
> only **2.8 % "cannot tell"**. The eye reads these frames *well*. What survives is narrower and
> first-hand: **metallic clips specifically**, on these frames, by this adjudicator.

## What it is

The rung asks **where the model looks, relative to where the foreign objects actually are, and
whether our fine-tuning moves it** — the same instrument rung 31 ran with luminance, with SAM masks
in place of the black letterbox.

Full design, arms, metrics, gate and pre-registered `NO VERDICT`: **[`PLAN.md`](PLAN.md)**.

## The ladder into it

| | |
|---|---|
| **step 6** ([[sam2-temporal-probe-closed]]) | CLOSED. The label moves 0.384, the model's error is 2.6× that ⇒ annotation noise is not the dominant cause. |
| **C1 as a clip-count control** | DEAD — it passed its threshold and did not license its interpretation. A human eye pass found SAM masks none of the visible clips in the `gold == 1` frames. |
| **C2** | Passes as pre-registered; margin thin (CI [0.8557, 0.9670] spans the 0.90 floor). |
| **rung 31** (`31-attention-probe/`) | The instrument, already validated on a different target: letterbox attention `base` 1.75× chance → `a2` at chance. |
| **rung 37** | ← this one (pre-registered as rung 36). |

## Inputs it will reuse

* `experiments/31-attention-probe/_tools/attention_probe.py` — unchanged, one variable: the checkpoint
* `experiments/29-sam2-temporal/_tools/export_masks.py` — the machinery `_tools/export_masks_37.py`
  reuses unchanged (`_corpus`, `_read`, `seed_instances`, `propagate_pair`, `grid=16`, `seed=42`),
  so only the *sample* differs from the run that produced `RESULTS_controls.json`
* `docs/viewers/sam2_masks_viewer.html` — the adjudication surface, with per-frame verdicts and
  Markdown export built in
* `local/fuentes/analisis-mascaras-sam2.md` — the eye pass that killed C1 and supplied this rung's
  enabling fact (masks do not merge tissue with foreign objects)
