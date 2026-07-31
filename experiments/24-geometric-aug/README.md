# Experiment 24 — label-aware horizontal-flip augmentation (`24-geometric-aug`)

| Notebook | Rung | Metric (`bucket_mean`) | Verdict |
|---|---:|---:|---|
| `24_flip_audit.ipynb` | 24a | transformable QA rows | done — 871/6,252 test rows transformable |
| `24_horizontal_flip.ipynb` | 24b (train) | — | trained (`24_flip_p25_v1`, 3 epochs) |
| `24b_epoch_eval.ipynb` | 24b (eval) | 0.6333 @ ep3 (armA 0.6305) | **NOT A WIN, any epoch** — `margin_OOD` falls all 3 epochs; targeted 871-row check non-significant every epoch |
| `24_position_prior_probe.ipynb` | 24 (probe) | atypical−typical acc gap | *(probe)* — mechanistic follow-up, not yet run |

## Objective

Test whether **training-only horizontal flips**, coupled with the matching
camera-relative QA transformation, improve FRAME robustness over the rung-21
`1e-4` recipe. Evaluation images and questions remain untouched.

This first notebook is an audit, not a training run. It reads the local FRAME
Parquets and writes an inspectable row-level classification under
`runs/24_flip_audit_v1/`.

## Audit result

| split | rows | transformable | excluded `1e` situs | unresolved `1d` |
|---|---:|---:|---:|---:|
| train | 13,748 | **2,170** | 313 | **0** |
| test | 6,252 | **871** | 111 | **0** |

The transformable rows are fully covered by three rules:

| rule | train | test | horizontal-flip QA change |
|---|---:|---:|---|
| `fixed_quadrant_class` | 1,092 | 461 | Swap the queried quadrant in the question; preserve the class answer. |
| `object_center_quadrant` | 640 | 202 | Preserve the question; swap the quadrant answer. |
| `all_object_positions` | 438 | 208 | Preserve the question; swap every quadrant in the position-list answer. |

## Safety boundary

The augmentation is equivariant only for camera/image-relative **templates**.
Most are `primary_capability == "1d"`, but the fixed-quadrant class template is
labelled `1a` in the supplied Parquets and is included by its explicit text
pattern. Patient-anatomy/situs questions (`"1e"`) are excluded: a horizontal
image reflection does not establish that a patient-left label should become
patient-right.

For each camera-relative template, the audit states whether a flip changes the
question, the answer, neither, or needs manual review. No training JSONL or
image is produced here.

## Planned A/B contract

The eventual training arm will start from rung 21's exact recipe and data. Its
only variable will be a deterministic, label-aware horizontal-flip policy.
With that policy disabled, the training export must be byte-identical to the
rung-21 control. This repository currently contains no horizontal-flip or
other geometric image augmentation result.

## Evaluation (24b) — result: NOT A WIN

`24_flip_p25_v1` (`p=0.25`, three epochs) trained and was scored canonically against **rung 21
arm A's same epoch** (never rung 18 — this experiment's baseline is arm A's recipe and data),
following the epoch-matched-control standard `21b_epoch_eval.ipynb` set. Full results:
`RESULTS_flip_p25.csv` + `RESULTS_transformable_flip_p25_ep*.csv`.

| epoch | Δ proxy vs armA | Δ margin_OOD vs armA | targeted 871-row delta (CI) |
|---|---:|---:|---|
| 1 | −0.0171 | −0.0170 | −0.0383 [−0.094, 0.017] |
| 2 | +0.0153 | −0.0100 | +0.0116 [−0.037, 0.057] |
| 3 | +0.0107 | −0.0052 | +0.0302 [−0.008, 0.071] |

**Pre-registered verdict (proxy rises AND margin_OOD does not fall): NOT A WIN at any epoch** —
`margin_OOD` falls every epoch. The **targeted check** (the actual hypothesis — paired accuracy
on the 871 `transformable` rows against arm A) is **non-significant at every epoch**, every rule.
A separate, unrelated regression also surfaced: `fo_class × ID` class-balanced macro-F1 drops
substantially at ep1 (−0.093) and ep3 (−0.129) vs arm A, while exact-set accuracy on the same
cell stays flat — a tail-class collapse hidden by the headline.

## Known defect (found, not yet fixed) — the `1e`/situs exclusion doesn't fire at export time

`transform_qa` (`_models/horizontal_flip.py:82-89`) hardcodes `primary_capability="3a"` when
calling `flip_audit.classify_row`, so the `capability == "1e"` branch that defines
`excluded_situs` can never trigger during the actual training export — only in the standalone
audit (`24_flip_audit.ipynb`), which uses the real capability. Quantified against the real
`train.jsonl` (question-text join, sha256-verified): of 305 real `1e` rows, **136 are
"abdominal quadrant location" questions** (plausibly pixel-grounded, i.e. should have been
transformed like a spatial row) and **30 of those got flipped with their pre-flip quadrant
label left unchanged**. The other 169 `1e` rows (organ contact/identity/presence) are almost
certainly genuinely flip-invariant despite going through the same buggy path. Net: **~30/14,415
rows (≈0.2%) plausibly mislabeled** — real, but too small to be the primary explanation for the
eval result above. Must be fixed (text-pattern situs detection, not a capability field the
JSONL doesn't carry) before `p=0.50` scales the same bug proportionally.

## Position-prior probe (24, mechanistic follow-up)

Motivated by ["Your other Left!"](https://arxiv.org/abs/2508.00549) (MICCAI 2025): VLMs answer
medical position questions from a memorised class/anatomy prior rather than reading the image.
`24_position_prior_probe.ipynb` tests whether FRAME's model does the same — is quadrant accuracy
worse specifically when an object sits in an ATYPICAL quadrant for its class (e.g. a Needle in
bottom/right, which happens in only ~6% of train examples) — on both rung 21 arm A and the flip
arm, at the same epoch. Zero GPU; reads existing `results.csv` only. Not yet run.

## Registered probability sequence

`24b` starts with **`p=0.25`** against rung 21 arm A (`lr=1e-4`). `24c` may test
**`p=0.50`** only after 24b is evaluated; it is a second, separately reported
single-variable rung, not a hyperparameter sweep folded into one result.

The policy's draw is deterministic from the SHA-256 of rung 21's frozen control
JSONL plus its line number. This is intentional: that JSONL includes rung 18's
minted and paraphrased supervision, which must remain in the geometry A/B
rather than being re-exported from raw Parquets.
