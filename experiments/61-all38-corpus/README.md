# Rung 61 — the last 8 held-out videos go into training

> **Status: TRAINED, CLOSED UNSCORED, SHIPPED as submission 08 (2026-09-08).**
> One variable against the arm that shipped as submissions 03 and 06: `--dataset`.
> The price — losing every local eval we own — was pre-registered before the run.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 of 38 test videos into training | 21 A2 | shipped as submission 03 — **0.5809** |
| 47-epochs-vs-corpus | attributed rung 42's +0.0402 between epochs and corpus | 42 | corpus share ≈ +0.0276 |
| 58-self-veto-and-pair | a second checkpoint; shorter class list wins | r42 ep4 | shipped as submission 06 — **0.58128** |
| 60 (UNAM) | external CholecT50 rows on top of the promoted ones | 19b | 🔴 **null**, +0.0016 CI [−0.0233, +0.0156] |
| **61 — all 38** | **the remaining 8 videos into training** | r42 | 🟡 **trained, unscoreable locally** — shipped as submission 08 |

## The corpus

    rung 42 corpus  19,384 rows = rung 18's 14,415 + 4,969 from 30 promoted videos
    rung 61 corpus  20,667 rows = the same 19,384 + 1,283 from the 8 that were held
    videos          122 / 130  ->  130 / 130   (+6.6 %, the same fraction as the rows)

`RESULTS_build61.json` carries four build gates, all passing. Gate 1 is the load-bearing one:
the builder **reproduces the shipped 4,969 rows byte for byte**, so the 1,283 are an addition,
not a re-generation. `RESULTS_diff_vs_r42.json` is the single-variable proof — one key,
`--dataset`. `RESULTS_gates.json` records the 38 video ids actually present in the corpus.

## Training

5 epochs, 6,460 steps, **1d 6h 17m** on one RTX 6000 Ada. Final `train_loss` 0.204,
`token_acc` 0.990, 18.68 GiB peak. Checkpoints every epoch (1292 / 2584 / 3876 / 5168 / 6460),
so ep2 = 2584 and ep4 = 5168 — the pair that ships.

## 🔴 Why there is no number here

All 38 videos are in training, so `bucket_mean` on the 1,283 is a memorisation readout. The
docstring of `_tools/train_rung61.py` states this before the run, along with the expected
outcome and its mechanism:

> *"rung 42's +0.0402 came from promoting 30 videos, of which 3,200 rows were Sigmoid — a
> procedure with ZERO training videos. The remaining 8 add no new procedure and no new centre.
> This is the same lever at 26 % of the stroke, on a model that has already had it. A null is
> the likely outcome and gets published as one."*

Scoring it on the only never-trained data left (rung 60's external rows) was costed at ~5.5 GPU-h
and **declined**: four platform anchors, two of them a tie, spanning 0.53–0.58 against a quantity
of ±0.01. The instrument is coarser than the effect by a factor of five. Full reasoning in
[[promoting-the-last-eight-videos-is-unscoreable]].

## 🟢 What it IS legitimate for

The 38 are the organizers' released train/test partition **with ground truth**, not the hidden
set the platform scores on. The empirical proof costs nothing: rung 42 trained on 30 of the 38
and scored 0.5809, against 19b's 0.5524 — a 0.03 gap, not the collapse leakage would produce.

## Files

    RESULTS_build61.json       corpus digest + 4 build gates
    RESULTS_diff_vs_r42.json   the single-variable proof (one key)
    RESULTS_gates.json         run-time gates, the 38 video ids, steps/epoch
    _tools/train_rung61.py     the engine, with the pre-registration in its docstring
