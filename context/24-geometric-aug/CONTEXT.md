# Experiment 24 — geometric augmentation

## Objective

Determine which FRAME QA examples admit a semantically correct horizontal
image reflection, then test a training-only label-aware flip policy against the
rung-21 `1e-4` recipe.

## Setup-config

The audit reads the local public FRAME Parquets supplied by the user. It writes
only generated audit CSVs to `experiments/24-geometric-aug/runs/`; it does not
write frames, training JSONL, weights, or evaluation predictions.

## Decisions

Camera-relative templates are transformed only with explicit, template-specific
rules. Most carry `1d`, but fixed-quadrant class selection is labelled `1a` in
the supplied Parquets and is matched by text. `1e` situs/anatomy questions are
excluded, because a pixel reflection does not prove an anatomical-left/right
answer mapping.

## Results

The refreshed audit covers all 20,000 supplied rows (`heico` + `lapchole`). It
marks **2,170 / 13,748** training rows and **871 / 6,252** test rows as
automatically transformable by one of three camera-relative rules:
fixed-quadrant class selection (question change), object-centre quadrant
(answer change), and all-object position lists (answer change). Every `1d` row
is classified; none requires manual review. The 313 train and 111 test `1e`
situs rows remain explicitly excluded.

## Next

The audit has no `manual_review` rows. The flip-export engine
(`_models/horizontal_flip.py`) landed with a deterministic flip rate, image
materialisation, and a flag-off byte-identity gate against the rung-21
control, keyed on the frozen rung-21 control JSONL's SHA-256 and row number
(preserving rung 18's minted/paraphrased rows exactly).

`24_flip_p25_v1` (`p=0.25`, three epochs) has been trained. `24b_epoch_eval.ipynb`
is built (not yet run): scores each epoch against **rung 21 arm A's same
epoch** (this experiment's baseline, not rung 18), plus a targeted paired
check on the 871 `transformable` test rows against arm A — the direct read on
whether the flip policy moved the population it targets, separate from the
generic leaderboard proxy. Needs a live GPU pod to execute (merge + infer);
`p=0.50` (24c) stays gated on this evaluation per the README's registered
sequence.
