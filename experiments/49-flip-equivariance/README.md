# Experiment 49 — flip-equivariance probe on the shipped checkpoint (`49-flip-equivariance`)

| Notebook | Step | Metric | Verdict |
|---|---:|---:|---|
| `49_flip_audit_heldout8.ipynb` | 1 (audit) | transformable rows | **175 / 1,283 (13.6%)** — done |
| `02_flip_pair_probe.ipynb` | 2 (paired GPU probe) | accuracy delta + literal equivariance | **built, not launched** — needs a pod |

## Objective

Does the currently-shipped checkpoint (**rung 42 ep4**, `checkpoint-4848`, submission 03,
0.5809 on the platform) correctly track object LOCATION when the image is horizontally
flipped at inference time — i.e. does its answer to a quadrant/position question mirror the
way the pixels moved, or does it repeat what it would have said regardless? Not a training
rung: an inference-time equivariance test, no GPU spent yet.

This is a sharper test than rung 24's original position-prior probe, which compared accuracy
on naturally typical vs. atypical class→quadrant pairs in the existing eval set. Here the
image itself is flipped, so a model that is truly reading pixels must flip its answer with it.

## Why a new experiment number, not a note inside `24-geometric-aug`

Rung 24 built the flip machinery (`_models/flip_audit.py`, `_models/horizontal_flip.py`) and
tested it as a *training-time* augmentation on rung 21 arm A. This experiment reuses that
machinery **unchanged** but applies it to a different checkpoint lineage (rung 42) with a
different, and critically leak-free, eval set — see below. Different checkpoint under test =
new experiment, per repo convention; the module stays owned by rung 24 and is imported, not
copied.

## Step 1 — the leak-free eval set, and the transformable-row audit

Rung 42's training corpus (`experiments/42-merged-corpus/`) promotes **30 of the public test
set's 38 videos into training**. Testing flip-equivariance on those 30 would partly measure
memorisation, not generalisation. The only leak-free eval set for this checkpoint is its own
declared held-out set: **8 videos / 1,283 questions**
(`experiments/42-merged-corpus/RESULTS_split_42.json`).

**Data provenance, verified before trusting any count:** the local FRAME parquets used here
match rung 08's data card exactly (6,252 test / 13,748 train / 38 test videos / 92 train
videos / 4,486 test frames), and the 8 held-out videos' question count reproduces submission
03's own declared number (1,283) exactly — independent confirmation this is the same
canonical split rung 42 was evaluated against, not just a same-shape coincidence.

**Result:** `flip_audit` (rung 24's classifier, unchanged) restricted to those 1,283 questions
finds **175 transformable rows (13.6%)** — close to rung 24's original 871/6,252 (13.9%) on
the full 38-video set, so restricting to the 8 held-out videos did not disproportionately
starve the transformable pool.

| dataset | rule | n |
|---|---|---:|
| heico | `fixed_quadrant_class` | 66 |
| heico | `object_center_quadrant` | 7 |
| heico | `all_object_positions` | 23 |
| lapchole | `fixed_quadrant_class` | 31 |
| lapchole | `object_center_quadrant` | 31 |
| lapchole | `all_object_positions` | 17 |

Full row-level audit: `RESULTS_flip_audit_heldout8.csv`. Disposition summary:
`RESULTS_flip_audit_heldout8_summary.csv`. Re-executed successfully end-to-end under
`papermill` in a dedicated `conda` env (`data_study`) — real captured outputs, not a
workaround; an earlier attempt in a throwaway sandboxed venv silently dropped kernel
stdout/file-writes, which `data_study` does not hit.

## Step 2 — the paired GPU probe (`02_flip_pair_probe.ipynb`) — built, not launched

For the 175 transformable rows: merges rung 42 ep4's adapter (`merge_adapter`, imported
unchanged from `experiments/48-centre-probe/_tools/probe_runner.py`), materializes an
original + horizontally-flipped JPEG per row (`_tools/flip_pair_runner.py`), then answers
BOTH conditions — untouched image/question, and flipped image + the label-correct
question/gold `flip_audit.py` already computed in step 1 — in **one loaded-model session**
(`answer_paired`), to avoid the GPU-swap drift [[archived-results-not-bit-reproducible]]
documents. Scored through the SAME canonical path every other rung is scored through:
`focus.evaluation.Evaluator` with the real LLM judge (not a bespoke comparator) — the
model call is the canonical `frame.engine.QwenFrameEngine`, only the loop and the image
source are this experiment's own, same philosophy as rung 48's `probe_runner.py`.

Two reads: (1) paired accuracy delta (flip − original), video-clustered CI via
`frame.metrics.paired_delta_ci`, same instrument rung 42's own eval notebook used; (2)
literal answer equivariance — for `object_center_quadrant`/`all_object_positions` (the two
rules whose *answer* carries a quadrant token), does
`flip_audit.swap_left_right_quadrants(original_prediction)` actually equal the flipped
prediction? `fixed_quadrant_class` has no quadrant token in its answer, so only its paired
accuracy delta is meaningful.

**Verified locally** (no GPU/SDK needed): notebook JSON validity, `_tools/flip_pair_runner.py`
syntax and its cross-experiment import of `merge_adapter`, the qID construction/uniqueness
and `__flip`-suffix round-trip against the real 175-row audit, the SMOKE row-selection, and
`flip_audit.swap_left_right_quadrants` behavior on realistic prediction text. **Not
verified**: anything requiring the SDK (`focus`), the merged checkpoint, or a GPU — this
machine has the parquets but not the source videos, so `FrameProvider` has nothing to
decode. Needs: a pod with `orena-data`, the rung-42-ep4 adapter (path in the notebook's
params cell, from `submissions/03-rung42-connector-ood/README.md`), and a SMOKE run first
per repo convention (build → smoke → independent review → full).
