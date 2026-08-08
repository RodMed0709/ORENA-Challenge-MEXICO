---
question: `ERROR_ANATOMY` reports the gold clip count moving ±0.86 between frames "less than a second apart", against a model error of 1.01 — the number used to argue the model sits at its label's noise floor. Does it reproduce?
verdict: No. The population is identical (1,946 consecutive pairs, reproduced exactly) but July's `gap` is the true gap in SECONDS divided by the video's fps, so its sub-second buckets describe pairs that are in fact 12–25 s apart. There is no sub-second population and there never was. At the corpus's true minimum separation of 1 s (n=461) the gold moves 0.384 and changes in 29.9% of pairs, against the published 0.86 / 56.6%. Against the model's 1.01 that is 2.6×, not 1.2× — the target is markedly MORE stable than the model, so "the model is at its label's noise floor" does not hold. The `8 → 14 → 6 in 690 ms` extreme was ~17 real seconds, which is ordinary scene change.
status: RETRACTED
date: 2026-08-08
basis: Zero GPU, fully local. Re-derived from the 4 corpus parquets (20,000 FRAME rows) in `experiments/29-sam2-temporal/_tools/label_gap_audit.py`, which reproduces both the published table under the unit bug and the true one beside it. The original July computation was never committed — `d71c305` and `d6ffdd0` touch markdown only
---

# The ±0.86 does not reproduce: it was a unit error, and the label is half as noisy

## What the number was doing

`context/ERROR_ANATOMY.md` reported the gold clip count moving **±0.86** between annotated frames
*"a fraction of a second apart"*, changing in **56.6%** of pairs, with observed extremes of
`8 → 14 → 6` in **690 ms** and `7 → 7 → 1` in **270 ms**. Nobody places six clips in 690 ms, so the
reading was: **the annotator counts the same image differently**.

Set beside the model's own mean absolute counting error of **1.01**, that made the model sit at
**~1.2×** the movement of its own target — i.e. at the label's noise floor, where more training
cannot help. That argument closed levers ([[count-calibration-dead]],
[[synthetic-counting-reconciled]]) and framed step 6 ([[covt-reduced-sam-route]]: *"steady track +
jumping gold = the gold is wrong"*).

## What it actually measured

The 2026-08-06 census had already found the premise impossible: **every `timestamp_start` in the
corpus is `HH:MM:SS` with no fractional part** — 12 parquets, 3 tracks, 40,000 rows, zero
fractional values — so no two annotated frames can be less than **1 second** apart, and
`frame_index` cannot rescue it (`data.py:117` derives it from the same field).

Re-deriving from the corpus: **the population is identical.** The selection reproduces
`ERROR_ANATOMY`'s **1,946** consecutive pairs exactly. Only the time axis differs.

🔴 **July's `gap` is the true gap in seconds divided by the video's fps** (25 `heico`, 30
`lapchole` — `config.py:26`). A pair genuinely **25 s** apart was filed as *"1 second apart"*; one
**12 s** apart became *"0.5 s"*.

| bucket | published (July) | reproduced under `gap_s / fps` | true gap |
|---|---|---|---|
| ≤0.5 s | n=**1257** μ=0.77 ident=48.5% jump≥3=**6.0%** max=**8** | n=1194 μ=0.75 ident=49.5% jump≥3=**5.9%** max=**8** | **n=0** |
| 0.5–1 s | n=**242** μ=**1.33** ident=16.9% jump≥3=**11.2%** max=**7** | n=**241** μ=**1.33** ident=18.7% jump≥3=**11.2%** max=**7** | — |
| 1–3 s | n=236 μ=1.54 ident=12.3% max=7 | n=270 μ=1.46 ident=13.3% max=7 | n=263 μ=0.66 |
| **≤1 s** | **0.86 / 56.6%** | **0.85 / 55.7%** | **0.384 / 29.9%** |

The `0.5–1 s` row matches on **four independent statistics**. ⚠️ **That is a fingerprint, not a
proof** — the original computation does not exist to inspect. What is established is that the
published table is **not reproducible from this corpus**, and that one specific unit error
reproduces it almost cell for cell.

## The corrected numbers

At the corpus's true minimum separation, **1 s**, n=461:

| | published | **true** |
|---|---|---|
| mean \|Δ\| | 0.86 | **0.384** |
| changes in | 56.6% | **29.9%** |
| jump ≥3 | 6–11% | **2.4%** |
| max jump | 8 | **4** |

And the separation between annotated frames is nothing like *"contiguous"*: **min 1 s, p25 2 s,
median 8 s, p75 31 s, p90 128 s**. Only **23.7%** of pairs are at the 1 s minimum; **15.4%** are
more than a minute apart.

🔑 **The argument inverts.** `1.01 / 0.86 = 1.2×` said the model was at its label's noise floor.
`1.01 / 0.384 = **2.6×**` says the target is markedly more stable than the model — **there is
room**. It is consistent with step 5's independently measured 24-point headroom on `number`
(pass@8 0.945 vs greedy 0.705), which a model at its label's noise floor could not have.

📌 **And the extremes stop being evidence.** `8 → 14 → 6 in 690 ms` was **~17 real seconds**;
`7 → 7 → 1 in 270 ms` was ~7 s. In 17 seconds of laparoscopic surgery clips are placed, the
laparoscope moves and things occlude. Those are compatible with ordinary scene change, which is
the opposite of what they were cited for.

## What this does and does not do

🔴 **It does NOT revive the levers that were closed with it.** It removes one argument they
rested on. [[count-calibration-dead]] has its own mechanism (three true values share one modal
prediction, so a LUT cannot separate them) and that is untouched. Each note must be re-read on its
own merits — this is a pointer, not a reopening.

🟢 **The caveat `ERROR_ANATOMY` already carried still applies, to 0.384.** A 1 s gap still mixes
real scene change with annotation error, so **0.384 remains an upper bound on annotation noise**.
Separating the two is exactly what step 6 was built to do — and it now has a much smaller quantity
to decompose, with `jump ≥3` at **2.4%** rather than 6–11%.

⚠️ **Three quoted numbers are affected and are NOT independently rescued:** `ERROR_ANATOMY`'s
*"≤0.5 s, n=1257"*, the **31.1%** of pairs at ≤1 s in [[covt-reduced-sam-route]], and the ±0.86
described there as *"less than a second apart"*. All three come from the same gap column.

## Why the computation is committed

The July figure reached **16 files** and closed levers while **its computation was never
committed** — `d71c305` and `d6ffdd0` touch markdown only, and nothing in `src/` or any `_tools/`
computes a gap between annotated frames. It stood for 17 days because there was nothing to check.

`experiments/29-sam2-temporal/_tools/label_gap_audit.py` runs both derivations. **A number without
a runnable derivation is not a record** — the same rule [[trailing-period-costs-nothing-scored]]
was written under, on the same day and because of this.
