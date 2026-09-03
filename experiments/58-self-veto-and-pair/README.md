# Rung 58 — the self-veto (dead) and the checkpoint pair (shipped as submission 06)

> **Status: CLOSED 2026-09-02.** Two inference-time levers on the checkpoint that ships,
> no training. One died on the ID control; the other became submission 06.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 of 38 test videos into training | 21 A2 | shipped as submission 03 — **0.5809** on the platform |
| 48-centre-probe | *nothing trained* — a second, external (centre) eval axis | — | the ruler this rung reads |
| **58a — self-veto** | ask the model, in yes/no form, to delete its own class | r42 ep4 | 🔴 **DEAD** — +0.0380 centre, **−0.0653 ID** |
| **58b — checkpoint pair** | a second checkpoint; shorter class list wins | r42 ep4 | 🟢 **SHIPPED** — +0.0221 centre, −0.0041 ID |

## What started it

On rung 48's centre probe the shipped checkpoint scores **0.0000 on the 2,445 items whose gold
is `none`**. Not "poorly" — never once. It always fills the list.
[[zero-is-format-localized]] measured the same model saying "no" fluently in `binary` format
(emit_absent 0.82) while never emitting `0` in `number`, which suggested the capability exists
and the answer format hides it.

## 58a — the self-veto is a centre-only gain, and ID pays for it

Ask, for every class the model listed, a yes/no presence question; delete the class on "no".

    smoke (n=164)   parses yes/no cleanly     0.94 / 1.00
                    says "no" when ABSENT     0.156 / 0.125
                    says "no" when PRESENT    0.000 / 0.000   <- never deletes a true class

🔻 **The 0.82 did not transfer.** It was measured on rung 06 and on the corpus's OWN binary
templates, which are **co-occurrence questions about class pairs** — there is no single-class
presence question anywhere in the training data. On r42 the same question form is answered
cleanly but discriminates at 0.156, not 0.82.

The smoke gate was written at margin ≥ 0.20 and relaxed to ≥ 0.05 **after** seeing that number.
That is recorded, in the script, as what it was: a COST gate deciding whether to spend 35 GPU
minutes, not the test. The test — the full-run delta with a leave-one-video-out jackknife — was
never moved. A veto that deletes 15.6 % of false classes and 0 % of true ones is one-directional,
which is why it was worth running.

| | centre (15 videos) | ID (8 videos) |
|---|---|---|
| r42 ep4 | 0.3618 | 0.8408 |
| **+ self-veto** | **0.3998 (+0.0380, 15/15)** | **0.7755 (−0.0653, 0/8)** |

🔴 **Killed by the ID control.** The gain is real and the cost is larger. This is the same shape
[[self-consistency-dead]]'s arm C had — a large single-axis gain that the ID-AND-OOD conjunction
catches — mirrored. **The conjunction is what stopped a slot being spent on something worse
than what we already ship.**

## 58b — the pair: a second checkpoint, and the shorter list wins

Model A answers; where its answer parses entirely as legal class names, model B answers too and
the shorter list wins. Ties keep A. Nothing else is touched: a `number`, a `binary` or a
multiple-choice token does not parse as a class list, which is what scopes the arm without the
container ever seeing an `answer_format` (`Request` does not carry one).

| arm | centre | folds | ID (`fo_class`) | folds |
|---|---|---|---|---|
| r42 ep4 (shipped) | 0.3618 | — | 0.8408 | — |
| **r42 ep4 + ep2** | **0.3838 (+0.0221)** | **15/15** | 0.8367 (−0.0041) | 0/8, worst −0.0100 |
| r42 ep4 + ep2 + ep3 + ep5 + a2 | 0.3894 (+0.0276) | 15/15 | 0.8367 (−0.0041) | 1/8 |
| r42 ep4 + a2 | 0.3716 (+0.0098) | 15/15 | 0.8367 (−0.0041) | 1/8 |
| r42 ep4 + ep5 | 0.3634 (+0.0016) | 15/15 | **0.8449 (+0.0041)** | **8/8** |

Canonical eval on all 1,283 held-out questions, judge included:
**`bucket_mean` 0.6722 → 0.6698 (−0.0023)**. The rule fired on 509 answers and changed **7**.
The two checkpoints agree almost everywhere on the hospital they were trained on — which is the
arm's own thesis from the other side, and why the local eval cannot price it.

**The pair, not the five**, ships: 80 % of the centre gain, two checkpoints instead of five, and
five merged 8B do not fit the build box.

## 🔻 Two corrections this rung forces on rung 48

[[epoch-selection-was-never-read-on-centre]] ranked **ep5** as the candidate for a slot, on
`bag_f1`. Read as exact-set match on the same 4,890 items, the ordering does not hold:

    ep2  +0.0168 centre   but  -0.0898 ID   <- would have been a bad slot
    ep4        (shipped)
    ep5  -0.0106 centre        -0.0082 ID   <- worse than ep4 on BOTH

⇒ **No single alternative epoch beats ep4 on both axes.** ep5 survives only *inside* the pair
(+0.0041 ID, 8/8), where it is the safest and smallest arm on the board. `bag_f1` ordered the
three platform anchors 15/15 and was explicitly warned not to scale; asked to separate epochs of
one run it disagrees with the metric the platform actually uses.

## What is NOT established

- The centre probe is one question template on one external dataset. `bag_f1`'s standing caveat
  applies unchanged: **it orders, it does not scale.** There is no conversion from +0.0221 to an
  expected `pre_evaluation_score` delta, and anyone who computes one is inventing it.
- 15 videos and 8 videos are the effective n on each side (`RULES §13`), not 4,890 and 490.
- Generation was never exercised inside the container: the build box has no GPU. The arbitration
  logic was, end to end, with generation stubbed.

## Artifacts

`RESULTS_smoke_gate.json` · `RESULTS_veto.json` · `RESULTS_id_control.json` ·
`RESULTS_58C_headline.json` (the control, and the reproduction of rung 42's 0.6744 at 0.6722) ·
`RESULTS_58D_pair.json`. Tools in `_tools/`; the arm itself lives in
`submissions/06-rung42-pair-ep4-ep2/`, whose `test_arbitrate.py` asserts that the shipped
`arbitrate()` still reproduces 0.3838 and 0.8367.
