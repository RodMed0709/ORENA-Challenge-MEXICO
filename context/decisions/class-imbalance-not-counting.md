---
question: Is the foreign-object failure a counting limit, or does the model not SEE certain classes?
verdict: per-CLASS — but inside `object_recognition`, NOT in the gap
status: RE_SCOPED
date: 2026-07-19
measured_in: experiments/06-vit-lora/_tools/per_class_recall.py
withdrawn: drop `silicone loop` from training — it is in the organizers' predefined list
amended_by: [the-gap-is-the-number-format]
---
# Finding: the failure is PER-CLASS, not per-count — and the ViT LoRA worked where it mattered

- **Status:** MEASURED · ⚠️ **its first recommendation is RETRACTED same-day** — see the
  retraction at the bottom and [[open-class-vocabulary]] **before acting on the data lever**.
- 🔴 **RE-SCOPED 2026-07-19 by [[the-gap-is-the-number-format]]: everything below is measured on
  `fo_class`, which sits ENTIRELY inside `object_recognition` — the bucket we LEAD by +14.9.**
  The measurements stand; their strategic reading does not. **This work defends our advantage;
  it does not close the `aggregation` gap** (which is 80.4% `number` format). Do not fund it
  out of the gap budget. See "Scope" below.
- **Status (original):** MEASURED (zero GPU, from artifacts we already had) · 2026-07-19
- **Applies when:** deciding whether to spend on data (rebalancing / synthetic QA), on
  perception (resolution, augmentation), or on capacity (bigger model).
- **Origin:** legokna pushed back on a proposal to measure counting on the `Clips`
  template — *"why only Clips, isn't that tunnel vision?"* The objection was right and
  reframed the question from *how many* to *which classes*.

## Question
`aggregation` is 50% of the exam and sits at margin +0.022 OOD, barely above a trivial
constant. The standing diagnosis was "multiplicity — the model saturates at ~2 objects".
Is that a counting limit, or does the model simply not SEE certain classes?

## What we sought
Per-class recall / precision on *"List all foreign objects that are visible"* (n=976),
in both arms, cross-referenced against each class's frequency in the training split.
Reproduce: `experiments/06-vit-lora/_tools/per_class_recall.py`.

## What it gave us

**1. Recall is wildly uneven, and tracks training frequency at the tails.**

| class | train ex. | recall (rung 06) |
|---|---|---|
| specimen bag | 345 | 0.831 |
| clip | 953 | 0.811 |
| external drain | 392 | 0.762 |
| specimen | 414 | 0.687 |
| sponge | 558 | 0.633 |
| **needle** | 154 | **0.511** |
| **gallstone** | **14** | **0.000** |

`gallstone` has **never once been named** by either arm. With 14 training examples that
is a data-coverage fact, not a perception failure — 12 val questions lost by construction.

**2. The ViT LoRA worked, and `bucket_mean` hid it.** Recall gains, rung 02 → rung 06:
`needle` **+17.8 pts** (0.333→0.511) · `external drain` +7.3 · `sponge` +5.7 ·
`clip` +1.9 · `specimen bag` 0.0. **The gain is largest exactly where perception was
worst** — which is what the perception-ceiling hypothesis predicts. The headline moved
+1.8 because it averages over buckets dominated by classes that already worked.

**3. Multiplicity, as recall:** k=1 → 79.2% · k=2 → 64.1% · k=3 → 57.7%.

**4. A phantom class in training.** `silicone loop` has **435 training examples (15% of
all listing answers) and ZERO occurrences in val**. Both arms emit it ~27 times: every
one is a guaranteed false positive.

**5. The model over-predicts the dominant class.** `clip` precision 0.619 — 212 of the
327 false positives are `clip`. Frequency-prior bias, not a perception limit.

## Verdict — and the ceiling on the data lever
🔴 **Rebalancing data is worth less than it looks.** Of 359 omissions:

- **68.2%** are on classes with **>400 training examples** (sponge 114, clip 80, specimen 51)
- only **15.6%** are on the rare classes (needle 44 + gallstone 12)

**`sponge` is the single largest failure (114 misses, 31.8%) and it has 558 training
examples.** More sponge data will not fix sponges. Sponges are deformable, blood-soaked
and blend into tissue — this is the case THE_MAP's "visual-degradation slice" names.

So the data lever splits into two very different bets:
- ~~**Cheap and bounded:** drop `silicone loop` (guaranteed FPs)~~ 🔴 **RETRACTED — see
  bottom.** Minting `gallstone`/`needle` and reducing `clip`'s dominance remain arguable,
  but address only ~16% of omissions plus a slice of the false positives.
- **The actual prize is `sponge`+`clip` perception**, which is NOT a data-volume problem.

⚠️ **This qualifies THE_MAP's "`number` is NOT a data problem".** That claim was measured
on the *distribution of numeric answers* (31% of train, well spread). It was never
measured on **class balance**, where the imbalance is 953:14 plus a phantom class.

## Next (cheapest first)
1. 🔴 ~~Drop `silicone loop` from training.~~ **RETRACTED — it is in the organizers'
   predefined class list; "never correct in val" is a fact about OUR proxy, not the
   hidden test. See [[open-class-vocabulary]].**
2. Per-class error slice on `sponge` — is it occlusion, blood, or scale? Decides whether
   augmentation, resolution or ROI is the right lever, and all three are currently
   un-evidenced guesses.
3. Only then consider minting QA for `gallstone`/`needle` — bounded upside, known ceiling.

## Sources
- `experiments/06-vit-lora/_tools/per_class_recall.py` (zero GPU).
- Train/val class counts via `frame.ledger.gold_from_frame_parquets(split=...)`.
- Related: [[vit-lora-partial]], [[checkpoint-selection-vs-number]], `experiments/08-data-card/`.


---

## 🔴 Retraction (same day) — "drop `silicone loop`" was fitting the proxy

The recommendation rested on `silicone loop` never being correct **in val**. But val is
our Sigmoid/chole proxy, not the judges' hidden set, and `silicone loop` **is in the
organizers' own predefined foreign-object list** (verified: 9 questions across the corpus
enumerate it). Removing it from training would blind the model to a class the challenge
explicitly names, to buy points on a local metric.

That is the same error class this repo guards against everywhere else — optimising against
the measuring instrument. It is recorded rather than deleted because the reasoning looked
sound and cheap, which is exactly when this error gets made.

**What survives untouched:** the per-class recall measurements, the finding that the ViT
LoRA lifted `needle` +17.8 pts, and the ceiling calculation (68% of omissions sit on
classes with >400 training examples, so data volume is not the lever). See
[[open-class-vocabulary]] for what the class set actually is.

---

## 🔴 Scope (added 2026-07-19) — this measures `object_recognition`, not the gap

Everything above was measured on *"List all foreign objects that are visible"* (n=976), which
is **`answer_format == fo_class`**. The bucket × format cross
([[the-gap-is-the-number-format]]) shows `fo_class` appears **exclusively** in
`object_recognition` — **zero occurrences in `aggregation`**.

So this note answers *"why do we miss foreign objects when listing them?"*, in the bucket where
we already **lead the external reference by +14.9 pts**. It does **not** answer *"why is
`aggregation` 12.5 pts behind?"* — that bucket is **80.4% `number` format**, and `number`
clears its trivial floor by only +8.0 pts (+0.013 margin on OOD).

**How to read this note now:**
- ✅ Still valid as the map of our `object_recognition` weaknesses, and as evidence the ViT LoRA
  did real perceptual work that `bucket_mean` averaged away.
- ✅ Its "Next" item 2 (the `sponge` error slice) is still worth doing — as advantage-defence.
- 🔴 **Invalid** as a rationale for spending the gap budget. `sponge` is 31.8% of omissions in a
  bucket that is not the problem.

The framing error is worth naming: this note was written to correct a **tunnel-vision** objection
(*"why only the `Clips` template?"*) and it did widen the question correctly — from *how many* to
*which classes*. But it widened along the **format** axis while the score is computed along the
**bucket** axis, so the wider view still missed the target. Widening is not the same as aiming.
