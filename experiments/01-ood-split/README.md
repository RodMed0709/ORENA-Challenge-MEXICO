# Experiment 01 — train / validation split + leak-guard

> **What this does, in one line:** it ONLY builds the frozen train/val split manifest
> (`experiments/splits/frame_ood_v1.csv`) that every later experiment reuses. It trains
> nothing, scores nothing, produces no model — just the split + its leak checks. That's it.

> The measurement harness every fine-tuning claim depends on. Carves the FRAME data
> (heico + lapchole) into **train + validation only** — `train`, `val_id`, `val_ood` —
> by holding out a WHOLE `procedure_type` as the OOD validation slice, split by
> `(dataset, video_id)` — never by question/frame.
>
> **No local test set.** The real test is the leaderboard submission, so a reserved
> local test would only waste data we could train on. Validation is small and used
> only to select checkpoints (by `val_ood`, so we don't overfit to cholecystectomy).
> Holding out ONE procedure_type (not a whole dataset) keeps the other surgery types
> in train — the model still learns cross-procedure, which matters because OOD is half
> the leaderboard score. For the FINAL submission model, fold validation back into
> train and retrain on everything to maximize data.
> Not a scored rung; it produces the frozen split every later experiment reuses.

## Ladder

| Notebook | Rung | Output | Verdict |
|---|---|---|---|
| `01_build_ood_split.ipynb` | 01 | `experiments/splits/frame_ood_v1.csv` | infra — run on pod |

*Infra rung — no `pre_evaluation_score`. Success = a leak-free manifest that populates the 10 scored buckets.*

## What this experiment is

- **One variable:** n/a (builds the split; downstream rungs vary against it).
- **Why:** the zero-shot baseline (`00-baseline`) used both batches but had **no split** and marked every row `ood=False`, so it measured only 3 of the 10 scored buckets (all ID). Fine-tuning without a leak-guarded OOD split produces an untrustworthy number — OOD is ~half the score.
- **Engine:** the split logic is a library, `src/frame/split.py` (one-`src` rule). This notebook only supplies config and calls it.
- **Key guards:** split by `(dataset, video_id)`; whole held-out `procedure_type` = OOD; `assert_no_leak` (train ∩ eval = ∅); coverage report over `capability_group × {ID,OOD}`.
- **TODO (cross-corpus leak, rung 03 of THE_MAP):** our lapchole is Cholec80-lineage and the organizers' hidden lapchole test likely is too → add a pHash + videoID-intersect scrub vs the challenge test set before training. Internal video-disjoint splitting cannot catch that. Tracked in `context/01-ood-split/CONTEXT.md`.

## Layout

```
experiments/01-ood-split/
├── README.md                  # this file (opens with the ladder)
├── 01_build_ood_split.ipynb   # builds the split, writes the shared manifest
├── _tools/
│   └── test_split.py          # CPU/offline unit tests for the leak-guard + determinism
└── RESULTS.csv                # one row per manifest version

experiments/splits/            # shared, committed manifests (source of truth)
└── frame_ood_v<N>.csv         # written by the notebook, reloaded by every experiment
```

Context: `context/01-ood-split/CONTEXT.md`. Split library: `src/frame/split.py`.
