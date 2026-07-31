# Context — 20-judge-swap

**Status: CLOSED 2026-07-28. Verdict MEASURED, verdict note
`context/decisions/judge-swap-is-not-the-gap.md`.** Zero training, ~7 GPU-minutes.

## Why the rung existed

Half of the FRAME eval is not scored by string match. `open_ended`, `multiple_choice` and
`matching` route to an LLM judge (`focus.data.formats.JUDGE_FORMATS`), and every judged
question in our eval lives inside `object_recognition` — the one bucket that collapsed
**−0.150** between our local val and the platform ([[leaderboard-metric-vs-our-headline]]).

And we were judging with the wrong model. `src/frame/config.py` pinned `Qwen/Qwen3-4B` behind a
comment asserting the SDK default `Qwen/Qwen3.5-4B` did not exist. It does
([[wrong-judge-model]]). So the largest unexplained gap in the project was being measured with
an instrument we knew was not the organizers'.

## What was run

Rung 06 ep3's **committed predictions**, re-scored under **both judges in one process on one
GPU**. That last detail is the design: roughly 0.5 % of stored answers change on a GPU swap — a fact
that currently lives **only in the maintainer's private notes and nowhere in this repo**, so
treat it as unverified here — and comparing against the archived `results.csv` would have mixed
judge disagreement with machine drift. Both
arms therefore re-run; the substitute arm is a CONTROL that must reproduce 0.5724246 exactly,
and it does.

Single variable: **the judge model**. Predictions, references, latency track, seed and the
scoring path (`frame.metrics.stratified_report`) are identical across arms.

## What it gave us

| | substitute `Qwen3-4B` | official `Qwen3.5-4B` |
|---|---|---|
| `bucket_mean` | 0.5724246 | 0.5710362 (**−0.0013884**) |
| `margin_ID` / `margin_OOD` | 0.22247 / 0.14475 | 0.22469 / 0.13975 |
| `object_recognition` ID | 0.6520 | **0.6559 (+0.0039)** |
| `acc_fo_class` / `number` / `binary` / `multiple_choice` | — | **byte-identical** |
| `acc_open_ended` | 0.6499 | **0.6230** (the only cell that moves) |

Agreement 99.2 % over 6,252; 93.5 % over the 759 judged rows; `multiple_choice` **202/202
identical**; all 49 disagreements `open_ended`, 40 of them OOD, substitute lenient in 32.

⇒ **The −0.150 is not our measuring stick.** It survives the correct judge, so it is behaviour
on unseen centres, and the newer-generation-backbone branch became the live one (and then died
in rung 23).

## What is worth carrying forward

- The disagreeing rows are committed (`DISAGREEMENTS.csv`) and they are not really about the
  judge: the model emits the right structure with the **wrong class, collapsing toward `Clip`**
  ([[class-imbalance-not-counting]]), and the substitute was calling those CORRECT.
- **The judge cannot run in the VLM's environment.** `Qwen/Qwen3.5-4B` declares
  `model_type: qwen3_5`, unknown to `transformers` 4.57, which is our hard floor. It needs its
  own venv (`/workspace/envs/judge35`, transformers 5.x). That load failure is almost certainly
  the origin of the false "does not exist" comment.
- **The substitute is kept deliberately.** The two judges agree to |Δ| ≈ 0.001 at the headline,
  so the whole ladder stays comparable under `Qwen3-4B` and no past number needs restating.
  Adopting the official judge would be its own decision, and this rung does not make it.

## Still open

The **rubric clash** is unmeasured. `judges.py:45-63` says *"If the candidate contains the
correct information but also includes extraneous text, still judge it CORRECT"* while our SFT
trains terse template-locked strings. This rung shows the two *judges* agree; it says nothing
about whether a less terse **output policy** would score better under either. Bounded: the
judged formats are 759 of 6,252 rows at 0.693 accuracy, so winning all of them is worth at most
**+0.037** raw — and it would take a training change, since SFT has already locked the format.

## Bookkeeping debt paid 2026-07-29

This rung originally shipped a transposed `RESULTS.csv` (metrics as rows, arms as columns) and no
`stratified.json`, so `frame.ledger` read it as 16 blank rows and it could never reach Tier 2/3.
Both are fixed: canonical two-row `RESULTS.csv`, canonical `stratified.json` per arm recovered
from the per-question `results.csv` on the volume, agreement detail moved to
`RESULTS_judge_agreement.csv`. It still has **no notebook** (spec §1) — the work lives in
`_tools/rejudge.py`, which does call `frame.metrics`.
