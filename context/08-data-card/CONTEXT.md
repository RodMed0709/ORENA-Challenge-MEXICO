# Context — 08-data-card

> Curated notes. The card is `experiments/08-data-card/README.md`; this file holds what the next person
> needs and the card does not say.

## Why this exists, in one line

**A0 ("calibrate the objective") was declared done on 2026-07-16 and was not.** Every audit since —
rungs 05, 05b, 07 — found a *measurement* defect, never a model fact. They were A0 surfacing in
disguise, and they would have kept coming.

## The cost of looking was artificial, and that is the whole story

**The four QA parquets are 410 KB.** Twenty thousand questions. They had lived for two months behind a
32 GB GPU pod, so characterising the data meant booking a pod, so nobody did it properly.

Once they were local, one afternoon with `pandas` produced: the parser verification, 188 templates, the
degenerate-template finding, the Simpson's-paradox correction to `acc_number`, the two-estimator split,
the capability tree, and the unseen-phrasing axis. **No GPU. No pod. Seconds per query.**

> **Keep them local.** If recomputing this card ever requires the pod again, the card will rot — that is
> exactly how it rotted the first time.

## What this card changed about numbers we had been quoting

| we said | it is | why |
|---|---|---|
| `number` is 37% of the exam | **33.5%** | never counted |
| `fo_class` is 39.1% | **42.8%** | never counted |
| `number` beats the floor by **+8** | **+4.9** | Simpson: the global floor is dumber than a per-template one |
| `raw_acc = 0.5662` | flat mean; the SDK's estimator says **0.5395** | two estimators, we quoted one for both jobs |
| "the 0.708" | **two numbers**: 0.7079 (rung-02 `eval_best`) and 0.7087 (rung-05 `a0_real`, a re-run) | different runs, ~0.0008 apart |
| `heico` = Sigmoid | **three procedures**; only Sigmoid is held out | never looked at train |
| 393 templates | **188** | the raw string counts each embedded timestamp as its own template |

**None of these is a model finding. All are measurement.** That was the argument for doing A0 first, and
the data made it better than the argument did.

## Three traps, in order of how much they would cost

1. **Stamping `ood=True` alone** → `pre_eval` 0.6389. Looks fixed. **Still +9 inflated**, because the
   orphan survives. It is the obvious first move and it is worse than doing nothing, because doing
   nothing at least *looks* broken. See card §5.
2. **Hand-rolling the capability mapping.** `Capability.from_any("1a").group` exists. Rungs 05 and 07
   wrote a set literal instead and dropped `instance_matching`. And `Capability.leaves()` is not what it
   sounds like — it returns every leaf of the enum, not the group's. **Read the SDK; do not infer it.**
3. **Counting templates on the raw question string.** Every `open_ended` question embeds its timestamp,
   so each becomes a "template" of n=1 — and a template with one question is degenerate by
   construction. That inflated the degenerate count from 126 to 330 and the exam from 188 templates to
   393. **Normalise `\d{2}:\d{2}:\d{2}` → `<TS>` first.** This trap caught the author mid-build: the
   card's assert fired, and the assert was right.

## The pattern worth naming

Four times now the answer was already in our possession and we built it by hand instead:

- `load_responses` — blocked as "probably invented". It existed.
- The leaf→group mapping — hand-rolled, incomplete. `Capability.group` existed.
- `ood` all-False — cost rung 05 an afternoon. **The SDK logs
  `only 3/10 group×distribution buckets are populated` on every single run.**
- The two estimators — `evaluator.py:465` documents the hierarchical bootstrap in its docstring.

> **The harness was never failing. It was warning, and nobody read the log.** That is the honest answer
> to *"is the harness broken, or are we using it wrong?"* — neither.

## What the card deliberately did not do

**Document, do not repair** — mixing the two is how A0 closed false the first time. Everything found is
in card §12 as its own atomic. The two that matter most:

- **The hand-rolled mapping is committed in `main`** (rungs 05 and 07). It does not bite today because
  `1b` is absent from both splits. It would bite silently.
- **THE_MAP still says "unfreeze the aligner."** The evidence says ViT. RodMed reads `main`.

## Two findings that are ammunition, not defects

- 🌟 **`procedure_type` has four values and reaches the model in every `Request`.** Never used. Note the
  shape: in our OOD slice its value is always one training never saw.
- 🌟 **`generation`** (`automatic` 3266 / `anchor` 616 / `manual` 118) — how each question was
  manufactured. Automatic and manual questions are different populations with different label noise,
  pooled into one score. It is *why* `number` is eight templates.

## For whoever runs the ViT-LoRA probe next

**Do not judge it on `acc_number`.** Card §3: eight templates, floors from 0.2394 to 1.0000, four
degenerate. Only three have enough answer variety for "truth = 2" to mean anything.

And know the bar it has to clear: **on the SDK's own estimator, `number` is 0.3803 with CI
[0.328, 0.428] against a template-aware floor of 0.3840.** At the video level we currently cannot
distinguish the model from a constant. That is the thing the probe has to move.
