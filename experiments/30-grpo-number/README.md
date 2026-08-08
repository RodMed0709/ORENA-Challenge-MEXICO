# Rung 30 — GRPO on `number`: can the model be made to EMIT what it already reaches?

## The ladder

| rung | variable | verdict |
|---|---|---|
| 21 arm A2 | lr 2e-5 → 1e-4 → **2e-4** | 🟢 **the campaign's best.** `bucket_mean` 0.6496 → platform **0.5288**, rank 11, both official baselines beaten |
| 27 (ex-24) | `vit_lr` 2e-4 → 2e-5 | 🔴 significant negative; single-variable as of [[multimodal-optimizer-is-an-identity]]. `B_high` never ran |
| 24 | label-aware horizontal flip `p=0.25` | 🔴 not a win, but it narrows a real class→position shortcut |
| 28 | VCD gate | 🔴 **dead** — p(`Clip`) *falls* 0.7794 → 0.6198 under degradation. The model is looking |
| 10b | entropy gate, per format | 🟢 `number` ALIVE (`zero_advantage` 0.280/0.270 over two seeds); `binary` + `fo_class` **dead, no gradient** |
| 22 | loss-mass | 🔴 **NO-GO** — the committed hook is an arithmetic identity at `per_device=1` |
| **30** | **GRPO with a 0/1 exact-match reward, `number` only** | **⏳ pre-registered, this file** |

## The question

`number` greedy scores **0.705** where sampling at k=8 finds the right answer **94.5%** of the
time. **24 points the model already reaches and does not emit.** GRPO cannot teach it to see —
[[vcd-has-nothing-to-subtract]] showed it already looks — it can only re-weight which of its own
samples greedy lands on. That is exactly, and only, the defect that was measured.

## Why this is worth a GPU week — the argument the August thread missed

The thread closed GRPO as *"diluted, it only touches `number`"*. The arithmetic says otherwise:

```
number = 80.4% of aggregation ;  aggregation = 2 of 4 populated buckets = 50% of the score
                          =>  number carries ~40.2% of the headline
```

| bar | Δ needed on `number` | share of the 24pt headroom |
|---|---|---|
| **S8** significance on `aggregation_ID` | **+4.5pp** | 19% |
| **S1** ships (≥0.03 on the headline) | **+7.5pp** | 31% |

| if GRPO captures | Δ number | headline | where that lands |
|---|---|---|---|
| 25% | +6.0pp | 0.5529 | ~rank 4 |
| 33% | +7.9pp | 0.5606 | ~rank 2 |
| 50% | +12.0pp | **0.5770** | **rank 1** (today's rank 1 is 0.5653) |

`number` is not a slice. It is the largest single lever the campaign has identified.

## Pre-registration — declared BEFORE the full run, per `RULES §S3`/`S8`

**Primary cell: `aggregation_ID`.** `S3` forbids local `bucket_mean` (it overstates the judge by
+0.121 and inverts the bucket ordering). `RULES §12` forbids quoting `acc_number` as a headline
(8 templates, 4 degenerate). `aggregation_ID` is the best-calibrated cell we own (judge ratio
**0.98×**) and is 80.4% `number` by construction.

**Win requires all three (`S8`):**
1. `aggregation_ID` paired video-clustered CI **excludes zero in the arm's favour**;
2. it holds on **ID and OOD jointly** — `aggregation_OOD` may not move against us;
3. **no cell anywhere shows significant harm.**

🔴 **Declared veto cells, named now so they cannot be discovered later:**
`object_recognition_{ID,OOD}` first. `fo_class` is **71%** of `object_recognition`, and a
`number`-only objective is free to walk the policy away from it. **This is the modal failure of
this rung**, and `beta` (KL to A2) exists to bound it.

**Minimum detectable effect: ≈0.036** on `aggregation_ID` (rung 21's ID-cell CI half-width 0.023,
√n-rescaled to n=955; video clustering makes that optimistic). **Predicted effect: unknown** —
which is the honest statement, and the reason the arm runs at all.

## 🔴 The three things most likely to make this fail, recorded before the result

1. **The 24pt headroom was measured on TRAIN data the model saw for 3 epochs.** Greedy 0.705 on
   train vs the judge's `aggregation` at ~0.525. The gap may be memorisation slack, not reachable
   capability. ⇒ **`grpo_holdout.jsonl` (493 rows, 10%, seed 42) is held out of training** for
   exactly this: if the arm moves train and not holdout, the gain is sharpening, not learning.
2. **Majority-class floor is 0.2505.** Always emitting `"1"` scores 25%. A reward that collapses
   the policy onto the mode is a *reward-satisfying* degenerate solution. Watch the emitted-value
   histogram, not only the mean reward.
3. **`aggregation_OOD` carries 25% of the score on 188 questions.** One question there is worth
   4× one in `object_recognition_ID`. It is also where the local instrument is blind.

## Wiring, all verified against the installed build

`ms_swift==4.4.1`, source pinned verbatim in `RESULTS_msswift_source.txt`.

| thing | fact |
|---|---|
| GRPO present | 17 files; `rlhf_trainers/grpo_trainer.py`, `rl_core/grpo_algorithm.py` |
| multimodal | 🟢 supported — `grpo_trainer.py:105` `is_multimodal`, handled at `:743/:1690/:1977/:2002` |
| reward contract | `ORM` subclass registered into `swift.rewards.orms`; called as `reward_func(completions, **kwargs)` |
| gold delivery | `rl_core/data.py:128-148` flattens `extra`, so the JSONL's `solution` column arrives batched |
| `--train_type` | 🔻 **does not exist on this build** — renamed `tuner_type` (`base_args.py:93`) |
| `model_type` | 🔻 must be pinned; the volume weights match 3 registered types |
| extra dep | `msgspec==0.21.1`, the only package added; env diff is one line |

**The reward IS the scorer** — 0/1 under `focus.data.formats.Number`, plus the shipped
container's trailing-period repair. Not set-F1: that was designed for `fo_class` multi-label
sets, `fo_class` is dead, and a `number` answer is a scalar.

## Data

`_tools/build_number_subset.py`, from the exact corpus A2 trained on
(`18_count_aug_v1/train.jsonl`, 14,415 rows, sha256 `180e28f0…`, both asserted).

| | rows | sha256 |
|---|---|---|
| `number` selected | 4,929 | — |
| `grpo_train.jsonl` | 4,436 | `cb48cdfd8543…` |
| `grpo_holdout.jsonl` | 493 | `5d113c571c4e…` |

Format recovered two independent ways — SDK gold predicate **and** counting intent in the
question — with **gold-only disagreement = 0** enforced as a hard gate.

📌 **Side finding, not fixed here:** 32 counting questions carry golds malformed for `Number`
(`'2.'`, `'Two.'`, `'Intestine: 1.'`), *all* under the phrasing *"Please provide a single
integer"* which appears nowhere else in the corpus. A candidate source of the trailing-period
habit probe 16a measured at 86.7%. It is a data defect and belongs to its own rung.
