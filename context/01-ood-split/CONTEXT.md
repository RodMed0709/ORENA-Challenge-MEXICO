# CONTEXT — 01 OOD-safe split

## Objective
Build the frozen, leak-guarded **train + validation** partition of the FRAME data
(heico + lapchole) — `train`, `val_id`, `val_ood` — that every fine-tuning experiment
trains and selects checkpoints against. Without it, no LoRA number is trustworthy
(OOD ≈ half the score, currently untested).

**No local test set** (decided with Leo 2026-07-13): the real test is the leaderboard,
so a reserved local test would only waste trainable data. Validation is small and used
only for selection. Hold out ONE procedure_type (not a whole dataset) so the other
surgery types stay in train — the model must learn cross-procedure since OOD is half the
score. For the FINAL submission model, fold validation back into train and retrain on all.

## Setup-config
- Data: both batches — `heico` (colorectal, 3 procedure_types) + `lapchole`
  (cholecystectomy), from `data_root/<ds>/data/frame/test.parquet`.
- Split key: `(dataset, video_id)` — never a question/frame (one video → many items).
- The ONE lever: which whole `procedure_type` is held out as `val_ood` (HeiCo Stage-3
  domain gap). Default scope `ood_dataset="heico"`, `val_frac=0.15`, `seed=42`.
- Library: `src/frame/split.py`. Notebook: `experiments/01-ood-split/01_build_ood_split.ipynb`.
  Manifest: `experiments/splits/frame_ood_v1.csv` (committed, shared).

## Decisions
- Manifest is the single source of truth; every experiment reloads it, never re-derives
  (prevents silent drift between machines/teammates).
- Split lives in the ONE `src/frame` package (one-`src` rule), not per-experiment.
- OOD marking at EVAL time (setting `reference.ood` from the manifest) is deferred to the
  eval experiment (THE_MAP rung 05), where it's needed — the split notebook only produces
  the partition + a coverage report.

## Results (2026-07-13, real data via HF `orena-dkfz/{heico,lapchole}-focus-vqa`)
Used the **organizers' own train/test partition** (`build_official_split`) — faithful to
their OOD design, zero video overlap. Manifest `experiments/splits/frame_ood_v1.csv`
(sha256 `6fd34c2c…`):

| split | videos | questions | % | role |
|---|---|---|---|---|
| train | 92 | 13748 | 69% | procto+rectal (heico) + chole train |
| val_id | 28 | 2252 | 11% | chole test (ID) |
| val_ood | 10 | 4000 | 20% | **Sigmoid** (heico test, unseen procedure = OOD) |

**FRAME track scope (verified via the `track` column — all rows = `frame`):**
- Capability GROUPS present: **object_recognition + aggregation** only (temporal_grounding
  n=3 = noise). **event_understanding + complex_reasoning are ABSENT** — they live in the
  PROCEDURE/SEGMENT tracks, NOT FRAME. So FRAME is ~4 scored buckets (2 groups × ID/OOD),
  not 10.
- Answer formats present: **fo_class (biggest) + number** (the baseline's weak spots
  0.182/0.127), then binary / open_ended / multiple_choice. **No `time`, no `percentage`**
  in FRAME (those are other tracks).
- Implication: all effort → fo_class + number; no reasoning-QA generation, no time/percentage
  handling needed.

## Next
1. Run on pod; confirm zero train∩eval overlap + which buckets the OOD side populates.
2. **Cross-corpus leak scrub (THE_MAP rung 03):** add pHash + videoID-intersect of our
   Cholec80-lineage lapchole train vs the challenge hidden test — internal video-disjoint
   splitting cannot catch same-lineage leakage. Highest-priority follow-up before training.
3. Feed the manifest into the LoRA experiment; select checkpoints by per-bucket OOD margin.
