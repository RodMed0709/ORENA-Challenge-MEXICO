# Rung 49 — the `Clip` attractor: how big is it, and does external data move it?

> **Status: 49c RUNNING** (2026-08-22, UNAM `tmux rod-rung49c`, GPU 0).
> Pre-registered before the GPU. Heavy artifacts in `/mnt/storage/uaq_user/rung49/`.

## Why this rung exists

Rung 48 measured the defect and named the mechanism: on frames where a clip **provably does not
exist yet**, all four anchors answer the bare string `Clip` in **85–95 %** of them, and the rate
**climbs with proximity to the clipping event** (0.567 → 1.000). The models read the surgical
**phase** and infer an object that has not been placed
([[clip-is-inferred-from-phase]]).

That defect is not a curiosity: `fo_class` is **42.8 %** of the eval and the challenge scores it
by **exact set equality**, so every spurious `Clip` is a zero — and it is a zero that lands on
whatever else was in the frame. This is the `Sponge` problem stated from the other side.

Rung 19b then added **5,718 Strasbourg positives** to the corpus and came out a **wash** on the
headline (`ALL_ID` −0.0089, `ALL_OOD` +0.0075, neither excluding zero). **Nobody has asked
whether it moved the attractor.** An arm can be flat on `bucket_mean` and still have changed the
failure this rung is about — or be flat because it changed nothing at all. Those are different
worlds and they choose different next moves.

## The ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 48 | *nothing trained* — the centre probe, v2 negatives | — | scored; `bag_f1` is the ruler, the `Clip` cell is the result |
| 19b | `--dataset`: + 5,718 Strasbourg **positives** | 47 ep4 | trained; **wash** on the headline |
| **49c (this)** | *nothing trained* — 19b read through rung 48's probe | 42 / 47 anchors | **running** |

## 49c — the pre-registered question and its readings

**Does adding external recognition POSITIVES move the phase attractor?**
Scored on rung 48's v2 probe, 4,890 items over the 15 held-out CholecT50 videos, through
`cholect50.score_per_class` and `clip_fp_anatomy.anatomy` — the same two functions that produced
the anchors, so the numbers are comparable by construction.

| outcome | reading | what it licenses next |
|---|---|---|
| **`fp_rate` drops materially** (≥0.05 vs r47) | positives DO touch the attractor; the headline wash is a different limit | more/better positives, and a headline that is not the right instrument for this defect |
| **`fp_rate` flat** | positives are inert against a phase prior | **negative supervision is the only remaining data route** — and it must be paid for |
| **`fp_rate` rises** | the extra Strasbourg clip frames *strengthened* the prior | external clip data is actively harmful here; stop the branch |

⚠️ **This is anatomy, not a ruler.** Rung 48 established that clip aggressiveness does **not**
track the platform — A2 is the least aggressive anchor and rung 42 beats it by 0.052. A drop in
`fp_rate` is **not** a promotion signal and must never be quoted as one. `bag_f1` remains the
only cell with ordinal validity, and it is reported beside it for exactly that reason.

⚠️ For the anchors the 15 videos are **unseen centre**; for 19b they are **unseen scene**, since
19b met Strasbourg in training. The paired comparison stays valid; what 19b's own number *means*
does not. Reported as such.

## What is NOT in this rung

- **No training.** 49c merges checkpoints that already exist and answers an existing item set.
- **The negative-supervision arm.** It is blocked: the CholecT50 **source** (videos + triplet
  labels) is **no longer on the box** — only the 5,718 extracted positive frames and the 4,890
  probe frames survive. Building negatives for the 35 training videos needs the labels back.
  49c is what decides whether that re-acquisition is worth paying for.

---

# 49a — MEASURED 2026-08-22. The `+0.0702` was a ceiling, and the defect is not a PAIR.

Zero GPU. Six arms (rung 19b ep3–5, rung 47 ep3–5), the full 6,252-question eval, every
number through `frame.metrics.stratified_report`. `RESULTS_fo_class_anatomy.csv`,
`RESULTS_fo_class_headroom.json`, `RESULTS_fo_class_confusion.csv`.

## 🔴 The headroom, priced five ways — and only one of them was ever quoted

Rung 19b ep4, `bucket_mean` **0.6479**. Each row flips `correct` to True on exactly its own
error class and re-scores:

| bound | rows | Δ headline | what it is |
|---|---:|---:|---|
| **loose** — the error *involves* Clip or Sponge | 611 | **+0.0878** | **the number the campaign quotes.** It is 90.7 % of every `fo_class` error, so it is ≈ *"fix all of `fo_class`"* — a CEILING, not a lever |
| **`clip_fp_any`** — model named Clip, gold has none | 296 | **+0.0387** | **the Clip attractor's total mass. This is the honest size of the front.** |
| mid — only Clip/Sponge membership is wrong | 285 | +0.0397 | the Clip-vs-Sponge decision alone |
| strict — singleton `{Clip}`↔`{Sponge}` swaps | 110 | +0.0141 | the pair confusion, as a pure substitution |
| declip — a spurious Clip *added* to an otherwise-right answer | 77 | +0.0105 | the one-token repair |
| addclip — a real Clip *missed* | 50 | +0.0078 | the mirror |

⇒ 🔑 **The front is real but roughly HALF its advertised size.** +0.0387 still clears the
0.03 ship bar and is comparable to the +0.0426 that separates us from rank 1 — it is worth
funding. It is **not** the +0.0702 that has been used to rank it above every other branch.

⚠️ Every bound is an ORACLE: it assumes a fix costs nothing elsewhere. A real intervention
that suppresses `Clip` will also delete true positives — `Clip` is 720 of the singleton golds
at 0.846 accuracy, the largest and one of the best-served classes. Read these as ceilings on
their own error class, never as expected gains.

## 🔴 It is a SINK, not a pair — and `Sponge` is not the worst victim

Singleton golds, rung 19b ep4, from `RESULTS_fo_class_confusion.csv`:

| gold | n | acc | → bare `Clip` |
|---|---:|---:|---:|
| Clip | 720 | 0.846 | — |
| **Needle** | 199 | 0.633 | **49 · 24.6 %** |
| Sponge | 481 | 0.780 | 64 · 13.3 % (+29 more as `Clip,Sponge`) |
| Specimen | 160 | 0.762 | 20 · 12.5 % |
| Gallstone | 5 | **0.000** | 4 · 80 % ⚠️ n=5 |
| Specimen Bag | 152 | 0.737 | — |
| External Drain | 348 | 0.891 | 8 · 2.3 % |

**`Clip` absorbs a share of every other class.** Framing the defect as *Clip↔Sponge* names the
second-worst victim and misses the first: **`Needle` loses a quarter of its rows to `Clip`.**
Any lever scoped to the Clip/Sponge pair is scoped to less than the defect.

📌 `pred_illegal = 0` on all six arms and both halves. **The format is not the problem** — this
is a recognition defect, not a parsing one, and no answer-canon work can touch it.

📌 The failure is **asymmetric across the split**: on ID the model *misses* classes (83 missed
vs 45 added), on OOD it *adds* them (40 missed vs 78 added). Out of its training distribution
it does not go quiet — it guesses, and it guesses `Clip`.

## 🟡 The phase ramp does NOT reproduce here, and the test is underpowered

Rung 48's ramp on CholecT50 runs **0.567 → 1.000** with an overall FP rate of 0.85–0.95. The
same construction on our own eval — per video, rows before the first Clip-positive timestamp
whose gold has no Clip — gives an FP rate of **0.14–0.25** and **no trend**
(`RESULTS_clip_fp_ramp_ours.csv`; deciles bounce 0.05–0.38).

🔴 **This does not refute rung 48 and must not be quoted as if it did.** The window holds only
**190 rows over 16 videos**, ~11–30 per decile, which is far too thin to see a ramp that
exists. What it does establish is that **the attractor is several times weaker on the data we
are scored on than on CholecT50**, so a CholecT50-sized effect cannot be assumed to transfer.

⇒ The negative-supervision arm — 12,281 clip negatives, 23,331 bag negatives — was sized
against the CholecT50 rate. On our own axis the target is 0.14–0.25, not 0.85. **That arm
needs re-costing before it is worth a 60 GB re-acquisition and ~36 h of GPU.**
