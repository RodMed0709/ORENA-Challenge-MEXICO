# Experiment 24 — label-aware horizontal-flip augmentation (`24-geometric-aug`)

| Notebook | Rung | Metric (`bucket_mean`) | Verdict |
|---|---:|---:|---|
| `24_flip_audit.ipynb` | 24a | transformable QA rows | audit pending |

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

## Registered probability sequence

`24b` starts with **`p=0.25`** against rung 21 arm A (`lr=1e-4`). `24c` may test
**`p=0.50`** only after 24b is evaluated; it is a second, separately reported
single-variable rung, not a hyperparameter sweep folded into one result.

The policy's draw is deterministic from the SHA-256 of rung 21's frozen control
JSONL plus its line number. This is intentional: that JSONL includes rung 18's
minted and paraphrased supervision, which must remain in the geometry A/B
rather than being re-exported from raw Parquets.
