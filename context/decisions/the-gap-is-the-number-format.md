---
question: Which answer_format composes the `aggregation` gap?
verdict: 80.4% `number`; `fo_class` does not appear in the bucket at all
status: MEASURED
date: 2026-07-19
measured_in: experiments/06-vit-lora/runs/06_vit_lora_v1/stratified.json (by_bucket_format)
amends: [class-imbalance-not-counting]
---
# Finding: the `aggregation` gap is the `number` FORMAT — 80.4% of it — and `fo_class` is not in it

- **Status:** MEASURED · 2026-07-19 · **zero GPU, zero new inference.** The number had been
  committed for days; nobody had read it.
- **Applies when:** choosing which lever to spend on. This note **re-scopes
  [[class-imbalance-not-counting]]** — see "What this amends".
- **Origin:** legokna asked which of two documented failure modes — per-class perception or
  counting — owns the 12.5-pt `aggregation` deficit. Neither note could answer it, because
  both were measured on the **format** axis and the gap lives on the **bucket** axis.

## Question

FRAME labels every question on **two independent axes**: the **bucket** (`capability_group` —
the axis we are scored on) and the **format** (`answer_format`). [[aggregation-is-the-gap]]
locates the deficit in the bucket `aggregation`. Every failure mode we had characterised —
per-class recall, the counting saturation, the image-ablation result — was measured **per
format**. The two had never been crossed.

**Which formats compose `aggregation`, and where inside it does the accuracy actually sit?**

## What we sought

`by_bucket_format` from `frame.metrics.stratified_report` — the group × {ID,OOD} × format
cross. It is already produced for every scored run and committed in each run's
`stratified.json`. No new measurement was needed.

## What it gave us

```
aggregation × ID   (n=955 · reproduces the canonical 0.4188 exactly)
  number        n=768   80.4%   acc 0.3268   floor 0.2409   margin +0.086
  binary        n=176   18.4%   acc 0.8352   floor 0.6591   margin +0.176
  open_ended    n= 11    1.2%   acc 0.1818   floor 0.9091   margin −0.727

aggregation × OOD  (n=1875)
  number        n=1326          acc 0.4819   floor 0.4691   margin +0.013   ← at the floor
  binary        n= 548          acc 0.7719   floor 0.7263   margin +0.046
```

🔴 **`fo_class` does not appear in `aggregation` at all.** It sits entirely inside
`object_recognition` (920 ID + 1755 OOD).

## Verdict

**1. The gap is `number`, by arithmetic.** `aggregation × ID` is 80.4% `number`. Moving 0.4188
→ 0.5438 requires ~119 more correct answers out of 955. Answering **100%** of `binary`
correctly only reaches 0.4492. **No path to the target avoids lifting `number` from 0.327 to
~0.482.**

**2. `number` is the format where the model barely uses the image.** Rung 05's ablation:
`number` clears the trivial floor by **+8.0 pts**, and with a **black image** returns
`0.3519579751671442` — the mode floor, identical to 16 digits. On OOD its margin is **+0.013**,
essentially at the floor. `fo_class`, on the same frozen ViT, clears its floor by **+32** and
drops **below** it when the image is shuffled — that is real perception.

**3. So the bucket worth 50% of the exam is 80% composed of the format that extracts almost no
visual information.** This is the same conclusion the counting deep-research reached from the
literature side ("when counting works in a VLM, it works by moving it out of the pixels and
into the text") — arrived at independently, from our own data.

## What this amends

🔴 **[[class-imbalance-not-counting]] measures a real failure in the wrong bucket.** Its
findings stand untouched as measurements — `sponge` 114 omissions (31.8%), `gallstone` recall
0.000, `clip` over-prediction, the ViT LoRA's +17.8 pts on `needle`. But they were all measured
on *"List all foreign objects that are visible"*, which is `fo_class`, which is **entirely
inside `object_recognition`** — the bucket where we **lead by +14.9**.

**Consequence: the per-class work defends our advantage; it does not close the gap.** Its
"Next" item 2 (the `sponge` error slice) remains worth doing, but as advantage-defence, not as
the route to 59%. It should not be funded out of the gap budget.

The same re-scoping applies to every data-supervision lever aimed at class balance
([[unused-metadata]]'s +77% aggregation pool is the exception — that one is labelled by
capability, not format, and needs its own check).

## Next

1. **`P(predicted | true)` per TEMPLATE** for `number` (~30 min, zero GPU, from rung 06's
   `predictions.json`). Rung 05b already established the aggregate behaviour — **the model
   counts but saturates at ~2** (Spearman 0.625, acc 0.803 on the mode vs 0.232 off it). What
   is missing is the per-template breakdown, required because `acc_number` is not interpretable
   across 8 templates (4 degenerate — data card §3). Decides whether the error has correctable
   structure or is noise.
2. Only then choose between the two surviving levers in this family: post-hoc count
   calibration (cheap, but its evidence fell in audit) and synthetic-counting SFT (the only
   idea with measured evidence on our backbone — and whose own stated risk is precisely
   finding 2 above).

## Sources

- `experiments/06-vit-lora/runs/06_vit_lora_v1/stratified.json` → `by_bucket_format`
  (produced by `src/frame/metrics.py:384`, committed since `da56eba`).
- Rung 05 ablation: `experiments/05-bottleneck-audit/RESULTS.csv`.
- Rung 05b counting probe: `experiments/05-bottleneck-audit/RESULTS_number_probe.csv`.
- Related: [[aggregation-is-the-gap]] · [[class-imbalance-not-counting]] · [[unused-metadata]] ·
  `experiments/08-data-card/` §3.

---

## Note on how this was found

Nothing here was measured today. The cross was computed by `stratified_report`, committed, and
read by nobody; rung 05's ablation and the per-class note had sat side by side for three days.
The session that produced this note rediscovered **four** pieces of already-existing work
(the per-class verdict, the rung 05 ablation, this cross, and a complete
`number-probe` spec **with its run already closed**).

This is the failure rung 05's own log named: *"the fix is NOT to document better — rung 00
documented it well and it was lost anyway."* Recorded here because the next person to ask
"has this been measured?" has no way to find out except by asking someone who remembers.
