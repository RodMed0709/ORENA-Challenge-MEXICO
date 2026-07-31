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

## Results (final) — CLOSED, NOT A WIN, but a real mechanistic effect confirmed

`24_flip_p25_v1` (`p=0.25`, three epochs) trained and was scored against rung 21 arm A's same
epoch. **Pre-registered verdict: NOT A WIN at any epoch** — `margin_OOD` falls every epoch, and
the targeted 871-row check is non-significant throughout. A separate `fo_class × ID`
class-balanced-F1 regression also surfaced, unexplained.

A real export bug was found (the `1e`/situs exclusion never fired at export time — capability
isn't in the training JSONL) and fixed: quantified at 32/14,415 rows (≈0.22%) plausibly
mislabeled, too small to explain the headline.

**The actual finding**, motivated by ["Your other Left!"](https://arxiv.org/abs/2508.00549)
(MICCAI 2025): built `24_position_prior_probe.ipynb` to test whether FRAME's model shows the
same class→position shortcut that paper found in medical VLMs generally. **It does, and
significantly**, on the unaffected control (rung 21 arm A). **The flip augmentation
significantly narrows that shortcut** (paired interaction test, CI excludes zero) — the
augmentation measurably does what it was designed to do, even though that didn't convert into a
net accuracy win at this dose.

Full writeup: `context/decisions/flip-narrows-shortcut-not-a-win.md`.

## Next

Experiment closed. `p=0.50` (24c) is **not** pursued as a direct scale-up — it would scale the
validated benefit and the unexplained `fo_class` regression together with no new information.
If revisited, the better-motivated next step is a differently-scoped rung (flip only genuinely
spatial rows, not the whole dataset) — a new decision against the campaign's other levers, not
one this rung's own evidence forces.
