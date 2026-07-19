# Finding: the failure is PER-CLASS, not per-count — and the ViT LoRA worked where it mattered

- **Status:** MEASURED (zero GPU, from artifacts we already had) · 2026-07-19
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
- **Cheap and bounded:** drop `silicone loop` (guaranteed FPs), mint `gallstone`/`needle`,
  reduce `clip`'s dominance to cut the 212 clip FPs. Real, but it addresses ~16% of
  omissions plus a slice of the false positives.
- **The actual prize is `sponge`+`clip` perception**, which is NOT a data-volume problem.

⚠️ **This qualifies THE_MAP's "`number` is NOT a data problem".** That claim was measured
on the *distribution of numeric answers* (31% of train, well spread). It was never
measured on **class balance**, where the imbalance is 953:14 plus a phantom class.

## Next (cheapest first)
1. Drop `silicone loop` from training; re-price. Zero risk, removes guaranteed FPs.
2. Per-class error slice on `sponge` — is it occlusion, blood, or scale? Decides whether
   augmentation, resolution or ROI is the right lever, and all three are currently
   un-evidenced guesses.
3. Only then consider minting QA for `gallstone`/`needle` — bounded upside, known ceiling.

## Sources
- `experiments/06-vit-lora/_tools/per_class_recall.py` (zero GPU).
- Train/val class counts via `frame.ledger.gold_from_frame_parquets(split=...)`.
- Related: [[vit-lora-partial]], [[checkpoint-selection-vs-number]], `experiments/08-data-card/`.
