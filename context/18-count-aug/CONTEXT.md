# context/18-count-aug/CONTEXT.md — the curated read

> The artifact dir is `experiments/18-count-aug/`. This file is the *curated* half of the
> two-part store: why the rung exists, what was measured while building it, and what to read
> when it lands. Numbers live in `RESULTS.csv` / `RESULTS_rank_ep*.csv`; verdicts live in
> `context/decisions/`.

## Status

**RUN and SCORED.** `bucket_mean` 0.5255 / 0.5488 / **0.5721** across three epochs; proxy
0.4877 / 0.5073 / **0.5421**. Read against rung 06 ep3's 0.5724 that is a **null** — but see
the update below, because this rung's real value turned out to be something else entirely.

## 🟢 Update 2026-07-30 — this rung became the control that broke the plateau

Rung 21 used **this rung's `train.jsonl`, byte-identical and in place** (sha256 asserted), and
changed nothing but the learning rate. On that unchanged data the proxy went **0.5421 →
0.6104** and `bucket_mean` **0.5721 → 0.6496** ([[recipe-axis-is-the-learning-rate]]).

⇒ 🔴 **Two readings here have to change.**

1. **This rung's null was a recipe null, not a data null.** The augmentation was evaluated at
   lr 2e-5 for 3 epochs — the configuration rung 21 measured as roughly 1/5 to 1/10 of anyone
   else's optimisation distance. **Whether L1 and L2 help at 2e-4 has never been measured.**
   The data is not refuted; it was read through an under-trained model.
2. **"Training erases OOD counting" is RETIRED.** This rung's ep3 `number_margin_OOD` of
   −0.0098, and rung 06 ep3's of exactly 0.000000, read as evidence that more training destroys
   the counter, and that reading shaped several rungs. At lr 1e-4 the *same three-epoch
   schedule on this exact dataset* gives **+0.042 (ep2) / +0.026 (ep3)**. The counter was never
   being erased; it was never being trained ([[undertrained-was-real]]).

⚠️ **What survives untouched:** every gold-only measurement in the build section below. Those
are properties of the data, not of the run — the minted-zero dosing, the 330 co-occurrence
repairs, the byte-identity of the flags-off export. They are why the rung was a usable control
in the first place.

## Why this rung exists

Rung 06 ep3 (`checkpoint-2580`, `bucket_mean` **0.5724**) leaves `number_margin_OOD` at
**exactly 0.000000** — accuracy 0.469080 against a template-aware floor of 0.469080. Rung 16
spent ~5 GPU-hours on four probes to decide which `number` lever was worth a 7.5 h run, and
licensed exactly two, both of them DATA:

- **L1** — minted zero-count rows, `number` only. 16a measured the model emitting `0` on
  0.008 (ID) / 0.017 (OOD) of `number` questions while emitting `no` on 0.82 of `binary`
  ones. That maps one-to-one onto supervision holding 1,008 `no` golds and **zero** numeric
  zeros. The base model is not a counter-example — it is a zero attractor negating 83% of
  PRESENT cases — so the capability never existed in this domain and regularisation has
  nothing to restore ([[zero-is-format-localized]]).
- **L2** — question-surface variation and format-tail dropout. The format is glued to the
  literal string *"Please provide a number."*.

## 🔴 The premise that broke while this rung was being built

The PLAN was framed on [[gold-is-signal-model-underuses-it]]'s *"a blind human orders counts
better (0.72) than our model (0.43)"*. Writing this rung's own metric meant computing that
baseline, and it did not reproduce: **the 0.43 is a gold 3–6 range-restricted number and the
0.72 is full-range**. On the human's own frames the model scores **0.8303** vs the human's
**0.7230**; on the full `Clips` template **0.5866**. Full account in
[[model-out-ranks-the-blind-human]]; RULES §13b/§13c added in the same commit.

Consequences here, in order of how much they change the rung:

1. **The comparator moves from 0.43 to 0.5866.** Against 0.43 this run would have "won"
   before it started.
2. **The "large discrimination headroom" argument is withdrawn**, and with it *"the model
   emits the marginal distribution"* — a model emitting the marginal cannot score r = 0.59.
3. **The levers are untouched.** L1 rests on emission rates and L2 on the format glue;
   neither ever referenced the human comparison.
4. 🟢 **A sharper statement replaces it:** the model orders OOD counts at **r = 0.4997** and
   adds **exactly zero** exact-match margin there. Ordering skill is real and does not
   convert into points ⇒ a rise in `r` here is a rise in `r`, not a projected score gain.

## What the build itself measured (gold-only, zero GPU)

These are results, not wiring, and they are final — the levers and their gates run at FULL
scale even in SMOKE (a dose gate evaluated on 64 rows tests nothing).

| | value |
|---|---|
| flags-off export vs rung 06's `train.jsonl` | **byte-identical**, sha256 `0254404343c4…` |
| corpus plurals reproduced by rule | **8/8** (Clips 1390, Sponges 192, External drains 27, Specimens 20, Silicone loops 17, Needles 10, Specimen bags 7, Gallstones 4) |
| real per-class count rows, TRAIN split | **1,667** (the PLAN's 2,495 is corpus-wide: 1,667 train + 828 val) |
| minted zeros | **667** at dose 2.5:1 — target 667, 0.0% off |
| minted class mix | Clip 112 · Specimen Bag 90 · Gallstone 89 · Silicone Loop 84 · External Drain 80 · Needle 73 · Specimen 54 · **Mesh 41** · **Absorbable Hemostatic Agent 26** · Sponge 18 |
| never-seen-class share | 67/667 = **10.0%**, exactly the cap |
| frames touched | 643, cap 2 per frame |
| L2 realized | stem **0.3989** (requested 0.40), tail-dropout **0.2605** (requested 0.25) over **4,929** `number` rows (4,262 real + 667 minted) |
| final dataset | 14,415 rows (13,748 real + 667 minted), +4.9% |

### 🔴 The PLAN's speed lever was measured and it was backwards

The PLAN proposed `per_device 4 × grad_accum 4` for rung 06's `1 × 16` — same effective batch,
on the theory that *"batch-1 wastes the card"*. On the 32 GB RTX 5090, paired inside one
notebook run against this rung's real `train.jsonl` (`RESULTS_vram.csv`):

| shape | peak GPU | s/it (eff. 16) | 3 epochs |
|---|---|---|---|
| **1 × 16** | **22,210 MiB** | **11.56** | **8.68 h** |
| 2 × 8 | 26,370 MiB | 13.16 | 9.88 h |
| 4 × 4 | **OOM** (32,076) | — | — |
| 6 × 2 | **OOM** (31,002) | — | — |

`4 × 4` does not fit at all, and batch-1 is **12% faster and 4.2 GB lighter** than `2 × 8`.
FRAME sequences are dominated by a variable count of vision tokens, so a micro-batch of 2 pads
to the longer sample and the padding waste exceeds the parallelism gain; a micro-batch of 1
pads nothing. ⚠️ Card-specific — it does not generalise to the 80 GB boxes.

⇒ The run goes at rung 06's own shape, which makes it strictly better on all three axes:
faster, 10.4 GB of headroom instead of 5.7 over a nine-hour run, and **`diff_vs_rung06()`
returns `{}`** — zero flags differ, so the A/B is purely the two data levers. The PLAN's
"utilisation, not a recipe change" clause is retracted; the design it was defending got
cleaner by losing it.

### 🟢 The finding worth keeping: the independent channel refutes far better than it confirms

The 1,230 co-occurrence binaries are the only channel that does not read `fo_class`. Folding
them in:

| | count |
|---|---|
| co-occurrence rows read | 597 |
| "all objects the same class" rows read | 367 |
| **presences the `fo_class` inventory had MISSED** | **330**, over 260 frames |
| minted zeros positively **confirmed** by the channel | **9** |
| gold self-contradictions found | **0** |

**330 repairs is the 8.4% under-naming caught in the act**, and each one is a minted zero
that would have been wrong in the exact direction the model is already biased toward. The
channel is thin as a *certifier* — only 9 of 667 minted zeros are positively confirmed,
because the binaries only ever ask about specific class pairs — so 658 rows still rest on the
closure assumption and the residual noise is real. Both numbers are carried in
`runs/<run>/minted.csv` per row (`evidence` = `binary` | `closure`).

⚠️ **0 gold self-contradictions** is worth noting on its own: no frame answers "all visible
objects are the same class" = yes while its `fo_class` gold names two. The closure inventory
and the binaries disagree only by omission, never by conflict.

## What to read when it lands, in this order *(written before the run; kept as the record of what was pre-registered)*

1. **Spearman r** on the `Clips` template vs **0.5866** — `frame.metrics.count_rank_report`,
   quoting the template and n every time (RULES §13b).
2. **`margin_OOD`** — pre-registered as a NO-FALL condition. r bought by giving up OOD margin
   is not a win.
3. `bucket_mean` vs **0.5724**, per epoch, epoch-matched (RULES §6b).
4. **L1's own readout** — probe 16a on the held-out slice, ABSENT **and** PRESENT arms. Never
   ABSENT alone: reading it alone is what would have sent rung 16 to repair a capability that
   never existed.
5. **L2's own readout** — probe 16d's illegal rate on `number`, with `binary` / `fo_class` as
   the within-run control this rung deliberately never touched.

⚠️ An L1×L2 interaction is **unattributable** from one run. That was accepted up front; what
it buys is that a null is diagnosable rather than mute.

## Links

[[zero-is-format-localized]] · [[model-out-ranks-the-blind-human]] ·
[[gold-is-signal-model-underuses-it]] · [[epoch-matched-control]] ·
[[the-gap-is-the-number-format]] · [[prompt-only-pointing-collapses]] ·
[[count-calibration-dead]] · [[open-class-vocabulary]] · [[the-negatives-dosing]] ·
[[recipe-axis-is-the-learning-rate]] · [[undertrained-was-real]]
