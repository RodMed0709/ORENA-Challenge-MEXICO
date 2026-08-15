---
question: Does promoting our own public TEST videos into training buy anything, at what epoch, and what does it cost us as an instrument?
verdict: YES on the ID half and it ships — rung 42 epoch 4 beats A2 ep3 by +0.0402 held-out `bucket_mean`, with paired video-clustered CIs excluding zero on all three ID cells and no cell showing harm. Epoch 5 FALLS on both halves while train `token_acc` climbs to 0.987, so the peak ships and the last epoch does not. The price is that this rung can no longer measure procedure-OOD locally: 8 of the 10 `heico` test videos are now IN its training set, so `RULES §3`'s qID→OOD reading is FALSE for this checkpoint alone.
status: SETTLED
date: 2026-08-15
basis: 5 checkpoints merged and scored on the 8 held-out videos (1,283 questions) on one A100, ~1.2 h; control recomputed from A2's ARCHIVED answers restricted to the same qIDs, zero GPU. `experiments/42-merged-corpus/RESULTS.csv`, `RESULTS_paired_ci.csv`, `RESULTS_class_f1.csv`, `RESULTS_heldout_eval.json`, notebook `01_eval_heldout.ipynb`.
---

# The corpus merge buys the ID half, and spends the OOD instrument to do it

## What rung 42 is

The A2 recipe (lr 2e-4, rank 8, α 32, `freeze_vit false`) with `target_modules` extended to
name the connector explicitly — `visual.merger.linear_fc{1,2}` plus the three
`deepstack_merger_list` blocks, because [[the-merger-is-unreachable-by-default]] measured
that `all-linear` is literal in neither ms-swift nor Unsloth. Five epochs on **19,384 rows**:
rung 18's 14,415 plus 4,969 generated from **30 of the 38 public test videos**, promoted
into training.

Legal, and deliberately so: the hidden leaderboard set (20 videos) and the final test (200
videos) are untouched by the merge. [[three-datasets-and-splits]] priced this trade in
advance — *"+45 % data but it destroys the only local instrument"* — and this note is the
measurement of both halves of that sentence.

## 🔴 The label `OOD` does not mean what it means anywhere else in this repo

`RESULTS_split_42.json` held back 8 videos and promoted 30. Of the 10 `heico` test videos,
**8 went into training**. Sigmoid Resection — the challenge's out-of-distribution procedure —
is inside this model's training set.

| held out | n videos | n questions |
|---|---|---|
| `heico` (`0023`, `0027`) | 2 | 800 |
| `lapchole` | 6 | 483 |
| **total** | **8** | **1,283** |

⇒ every `*_OOD` cell for this checkpoint is **unseen VIDEO of a SEEN procedure**, which is
the same kind of held-out `lapchole` has always been. `RULES §3` is a faithful reconstruction
of the organizers' design for every *other* rung and **does not hold for this one**.

`frame.metrics` still prints `OOD`, because forking the canonical scorer for one rung is
exactly what `RULES §1` forbids. The correction travels beside the number instead: in the
notebook's opening cell, in `RESULTS_heldout_eval.json.ood_caveat`, and in the
`ood_procedure_in_train=True` column of `RESULTS.csv`. **Anyone quoting an OOD figure for
rung 42 without that sentence is quoting a different quantity than they think.**

Second limit, equally binding: effective n is **8 videos** (`RULES §13`), two on the heico
side. No heico CI here is readable. A jackknife moves `acc_OOD` by a median 0.024 on *38*
videos; on 2 it is not an instrument.

## The curve, and the memorisation it exposes

Selection axis declared **before** the numbers: `bucket_mean` on the held-out set, ID and
heico weighted equally, which is what the final Copeland ranking does (`RULES §4c`).
`RULES §6`'s `acc_OOD` axis does not exist for this rung.

| epoch | ckpt | `bucket_mean` | `acc_ID` | `acc_heico` | macro-F1 | illegal `fo_class` |
|---|---|---|---|---|---|---|
| 1 | 1212 | 0.5649 | 0.6004 | 0.5238 | 0.8297 | 0 |
| 2 | 2424 | 0.6242 | 0.6501 | 0.5938 | 0.8337 | 0 |
| 3 | 3636 | 0.6262 | 0.6812 | 0.5650 | 0.8846 | 0 |
| **4** | **4848** | **0.6744** | **0.7350** | **0.6075** | **0.9117** | **0** |
| 5 | 6060 | 0.6592 | 0.7184 | 0.5938 | 0.9069 | 1 |
| *A2 ep3 (control)* | *2703* | *0.6342* | *0.6646* | *0.5988* | *0.8670* | *0* |

🔑 **Epoch 5 falls on BOTH halves while training `token_acc` climbs to 0.987.** That is the
memorisation signature, and it was pre-registered as such before the sweep ran — which is the
only reason the number is readable as memorisation rather than as noise found after the fact.
It is also the whole justification for scoring five checkpoints instead of the last one.
**Epoch 4 ships.**

📌 This is `RULES §6` and `§6b` earning their keep in a case where they could easily have been
skipped: the corpus merge was expected to help, epoch 5 was the obvious default, and the
default was wrong by −0.0152.

## What is actually significant

Δ vs A2 ep3 = **+0.0402**, above the ≈0.03 bar `RULES §S1` acts on. Paired, video-clustered:

| cell | Δ | 95 % CI | excludes 0 | n videos |
|---|---|---|---|---|
| `aggregation_ID` | +0.0922 | [+0.0057, +0.1852] | ✅ | 6 |
| `object_recognition_ID` | +0.0461 | [+0.0009, +0.0949] | ✅ | 6 |
| `ALL_ID` | +0.0705 | [+0.0206, +0.1246] | ✅ | 6 |
| `aggregation_OOD` | +0.0049 | [−0.0427, +0.0686] | ❌ | 2 |
| `object_recognition_OOD` | +0.0207 | [−0.0209, +0.0649] | ❌ | 2 |
| `ALL_OOD` | +0.0088 | [−0.0200, +0.0412] | ❌ | 2 |

Read against `RULES §S8`: the declared cell wins, **no cell shows significant harm** (no
veto), but clause (b) — *holds on ID and OOD jointly* — is **NOT met**. All three OOD point
estimates are positive and all three CIs include zero, on two videos. 🔑 **The measured gain
is the ID half and nothing else should be claimed.** Deflated by `§S2` (÷1 `agg_ID`,
÷1.5 `obj_ID`) that is roughly +0.092 and +0.031 on the cells that pay.

⚠️ And `aggregation` is [[aggregation-is-the-gap]] — the bucket a 4B model beat us on. A
+0.0922 there is aimed at the right target, which is a reason to believe it *and* a reason to
check it on return rather than assume it.

## 🟢 The surprise: no tail collapse

[[class-balanced-f1-is-mandatory]] exists because SFT reliably crushes class-balanced F1
while exact-match rises — measured on our exact backbone (F1cls 20.7 → 15.3). Here macro-F1
climbs **with** exact-set accuracy, 0.8670 → 0.9117, on every cell. Whatever the extra 4,969
rows did, they did not do it by collapsing onto the head class.

⚠️ Read `per_class` beside the scalar before leaning on this: rung 21's +0.174 macro-F1 was
82 % one `Needle` question flipping.

📌 Epoch 5 emitted **1 illegal `fo_class` token** and epoch 4 emitted 0. Small, but it is the
first direct evidence that `RULES §8b`'s failure mode is reachable by a real checkpoint of
ours — and `FOClass.verify` **raises** on it rather than scoring it 0, taking the whole
answer down. Submission 03's container now reads `FOType.names()` at runtime and drops
unrecognised tokens.

## Cost and method notes worth keeping

- **~1.2 h on one A100 ($1.39/h)** for all five checkpoints: merge → eval on 1,283 →
  reclaim the 17 GB, one at a time. Peak disk +17 GB against ~73 GB of headroom.
- **The control was free.** A2 never trained on any test video, so its archived ep1/ep2/ep3
  answers are valid on these 8 videos; restricting them to the same 1,283 qIDs gives an
  epoch-matched control (`RULES §6b`) with **zero GPU and zero GPU-swap drift** — nothing is
  re-generated, so [[archived-results-not-bit-reproducible]] does not bite on the control.
  The arm was generated on this pod's GPU, so it bites on the comparison at ~0.5 %; that is
  inside these CIs, not outside them.
- Three chain failures cost ~$0.25 total and each was a gate firing before the GPU was spent:
  a missing `infer` kernelspec (registered per POD, not on the volume), `HF_HOME` pointing at
  `/workspace/hf_cache` when the judge is cached under `/workspace/.cache/huggingface`
  (rung 39's notebook comment says the wrong one), and `n_boot=0` — a value
  `stratified_report` accepts and then crashes on inside `_hier_bootstrap`.
- 🔴 **Open, and it is a real gap:** `frame.metrics.stratified_report(n_boot=0)` raises
  `IndexError` from `np.percentile` on an empty array. The argument is accepted, so the bug
  is reachable from any caller. Guard it in the module (`RULES §1` — extend, never
  reimplement beside it) rather than in each notebook.
