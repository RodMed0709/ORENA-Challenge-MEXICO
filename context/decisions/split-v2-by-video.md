---
question: Our eval set shrank to 8 videos and its confidence intervals stopped resolving anything. What partition replaces it, and what does that cost?
verdict: FOUR slices by video, stratified by procedure — train 72 / val_id 24 / test_id 24 / val_ood 10 (the whole unseen procedure). Split by VIDEO COUNT, not question mass, because frame.metrics clusters on video and clusters are what set the CI width. val_id and test_id go from 8 clusters to 24. The cost is that rung 42 trained on 122 of the 130 videos, so its score on any new slice is contaminated — the next training run (19b) becomes the clean baseline, and 0.6744 stops being the number to beat
status: SETTLED
date: 2026-08-19
measured_in: experiments/splits/frame_split_v2.csv (sha256 ae74ce4f…) · experiments/01-ood-split/02_build_split_v2.ipynb · epoch CI recomputed with frame.metrics.paired_delta_ci over experiments/47-epochs-vs-corpus inspect.csv (ep3/ep4/ep5)
---

# Decision: the split is rebuilt by video, and the old eval set is why

## What went wrong, exactly

`frame_ood_v1.csv` held out **38** videos: 28 lapchole (`val_id`) + 10 Sigmoid (`val_ood`).
Rung 42 promoted **30 of those 38** into training ([[merged-corpus-buys-the-id-half]]). **Eight
remained** — 6 lapchole (483 questions) and 2 Sigmoid (800). Every rung since has been measured
on those 8, and nobody re-derived the partition.

🔑 **On 2026-08-19 that came due.** The rung-47 epoch effect, run through
`frame.metrics.paired_delta_ci` (which bootstraps clusters on **video**), returned **no cell
excluding zero**:

| cell | n | videos | delta | CI 95 % |
|---|---:|---:|---:|---|
| `object_recognition_ID` | 250 | 6 | +0.0513 | [−0.0022, +0.1140] |
| `ALL_ID` | 483 | 6 | +0.0290 | [−0.0145, +0.0705] |
| `aggregation_ID` | 233 | 6 | +0.0103 | [−0.0993, +0.1080] |

The point estimates reproduce the ad-hoc bootstrap that `NOW.md` quoted (+0.0480 / +0.0290); the
**intervals do not**. The earlier bootstrap resampled QUESTIONS — 1,283 of them — as if they were
independent. They are not: they come from 8 videos. Same number, wrong denominator.

⇒ **The asymmetry that the rung-47 verdict rested on does not exist.** Epochs fail the same test
corpus failed. Neither lever was ever established, and the binding constraint is the eval set,
not the world. Amends [[rung42-gain-was-epochs-not-corpus]].

## The partition

| slice | videos | questions | for |
|---|---:|---:|---|
| `train` | 72 | 9,583 | training |
| `val_id` | 24 | 3,209 | epoch / checkpoint selection |
| `test_id` | 24 | 3,208 | reported once, at the end |
| `val_ood` | 10 | 4,000 | unseen PROCEDURE, read-only |

Stratified 6/2/2 in Proctocolectomy, 6/2/2 in Rectal Resection, 60/20/20 in Cholecystectomy;
the 10 Sigmoid videos leave whole. **val_id and test_id go from 8 clusters to 24.**

**Split by video COUNT, not question mass.** heico videos carry 400 questions each, lapchole ~80
— a 5:1 ratio, so "60/20/20 of questions" would put 3 heico videos against 50 lapchole ones in
one slice. Mass is used only to break ties, and lands at 3,209 / 3,208 — one question apart.

**`test_id` is the new slice, and it is what fixes selection-on-test.** Until now the epoch was
chosen by `argmax(bucket_mean)` over the same set we then reported, which inflates every number
including the ep4 of rung 42 that we submitted.

## 🔴 `val_ood` keeps its name for compatibility, not for accuracy

`metrics._distribution` (`metrics.py:165`), `delta.py:72` and `ledger.py:70` all hardwire the
literal string `"val_ood"`. Renaming the slice would make all three label **every video "ID"** —
silently, with no error. So the key stays.

**It means UNSEEN PROCEDURE.** The platform's OOD axis is **centre** (">5 centres not
represented"), and the organizers' `ood` column is `False` on all 20,000 public questions. The
two disagree, measured: the same rung-42 checkpoint reads `aggregation_OOD` **0.4086 locally and
0.6064 on the platform**. The manifest carries `ood_axis=procedure` so the file says it itself.
**Never report `val_ood` as a proxy for the platform's OOD half** — [[local-eval-is-ordinal-not-cardinal]].

## What it costs, and the cheap way to pay it

Rung 42 trained on **122 of the 130 videos**. Its score on any new slice is contaminated, and no
manifest can fix that — the videos are in the weights, not in the CSV. Only 8 videos are clean
for it, which is the corner we are leaving.

⇒ **The next training run is the baseline.** 19b is already on the board and has not run; trained
on v2 `train`, it *is* the clean reference. No dedicated retrain is needed — only the decision
that everything from here uses v2. **`0.6744` becomes historical and is not compared against v2
numbers.**

🟢 **v1 is not deleted.** Every earlier result was measured under it and stays valid there;
`frame_ood_v1.csv` remains committed. Reverting is free — the manifest is a versioned CSV, which
is the whole reason it is one.

## Sources

- `experiments/01-ood-split/02_build_split_v2.ipynb` — rebuilds it from v1 alone, **no challenge
  data and no GPU**; six guards, all passing.
- `src/frame/split.py::build_split_v2`. `_SPLITS` widened with `test_id`; `assert_no_leak` now
  compares train against *everything that is not train* instead of a fixed list, so a fourth
  slice cannot slip past it.
