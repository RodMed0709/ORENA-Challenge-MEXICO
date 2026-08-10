# Rung 36 — step 7: attention vs SAM masks

> 🟡 **STATUS 2026-08-09: PRE-REGISTERED, NOT BUILT, NOT RUN.**
>
> 🔒 **The number 36 is CLAIMED — do not reassign it.** The `24` collision cost a directory rename
> and a week of prose that still says the wrong rung. If you need a slot, take 37.
>
> 🛑 **Do NOT build this yet.** There is no engine, no `_tools/`, no notebook, and that is
> deliberate. `PLAN.md` is a pre-registration written before any GPU was spent so its thresholds
> could not be chosen after seeing a number (`RULES` S3, S7). Building ahead of the gate is how a
> design turns into a sunk cost that argues for itself.
>
> ⏸️ **One open item before it is buildable:** `G-BOUNDARY`'s formulation is under review by
> legokna, scheduled for the next session. Nothing else blocks it.

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
| **rung 36** | ← this one. |

## Inputs it will reuse

* `experiments/31-attention-probe/_tools/attention_probe.py` — unchanged, one variable: the checkpoint
* `experiments/29-sam2-temporal/_tools/export_masks.py` — the mask export
* `docs/viewers/sam2_masks_viewer.html` — the adjudication surface, with per-frame verdicts and
  Markdown export built in
* `local/fuentes/analisis-mascaras-sam2.md` — the eye pass that killed C1 and supplied this rung's
  enabling fact (masks do not merge tissue with foreign objects)
