---
question: The corpus carries counting golds malformed for `Number` (`'2.'`, `'Two.'`, `'Intestine: 1.'`) under one phrasing that appears nowhere else. Does that data defect — and the trailing-period habit probe 16a attributes to it — cost anything where we are scored?
verdict: No, and it is now measured on both halves. The phrasing lives in 25 of 20,000 rows, ALL of them `open_ended` and ALL of them `ood == False`; its 7 scored instances route to the LLM judge, which the period does not affect. Zero of the 2,094 `number` questions in the scored split use it. And on A2 ep3's raw outputs, 2,094 of 2,094 `number` answers are clean digits — no periods, no word-numbers, no prefixes. The habit is real off-template and costs nothing on-template. ⇒ Do NOT touch the corpus (it would invalidate the sha256 anchoring rungs 18/21/27/30) and do NOT remove `normalize_answer` (the final test set is different data). Closed on merit, no rung opened.
status: MEASURED
date: 2026-08-08
basis: Zero GPU. Corpus half from the 4 local parquets (20,000 FRAME rows); coverage half from `21_lr_2e4_v1/ep3_full` raw `predictions.json` on the pod. Both halves are `experiments/16-count-probes/_tools/trailing_period_audit.py`, runnable
---

# The trailing period is real, and it costs nothing where we are scored

## What was being asked

Rung 30's subset builder (`_tools/build_number_subset.py:30-46`) reported **32 rows** of
`train.jsonl` with counting intent and a gold that is not a digit — `'2.'`, `'3.'`, `'0.'`,
`'Two.'`, `'Intestine: 1.'` — **all** under the phrasing *"Please provide a single integer"*,
which appears nowhere else in the corpus. Rodrigo recorded them, correctly did not touch the
corpus, and proposed a rung of their own.

The hypothesis was sound: probe 16a measured our checkpoint emitting `"1."` — with the period —
on **86.7% ID / 87.5% OOD** of `number` questions phrased outside the corpus templates, against a
base-model rate of **0.0000**. Our own SFT installed the habit, and the SDK's `Number.verify` is
`text.strip().isdigit()`, so the period is scored wrong.

**But "the habit is real" and "the habit costs points" are different claims**, and only the first
had ever been measured. This note measures the second.

## What was measured

### Half 1 — where the phrasing lives (corpus, fully local)

| | |
|---|---|
| rows carrying *"single integer"* | **25** of 20,000 |
| `answer_format` | **`open_ended` in all 25.** Zero in `number` |
| split | 18 train (7 `heico` + 11 `lapchole`) · **7 test** |
| ID/OOD | **`ood == False` in all 25** |
| golds | `'3.'`×8 `'4.'`×4 `'2.'`×3 `'6.'`×3 `'1.'`×2 `'5.'` `'0.'` `'Two.'` `'One.'` `'Intestine: 1.'` |
| `number` questions in the scored split using it | **0** of 2,094 |

🔑 **All ten distinct golds end in a period.** This is not "some rows are malformed" — the
template is annotated with a trailing period **without exception**, which is why it is a coherent
teaching signal rather than noise.

🟢 **The 7 scored instances are `open_ended`**, so they route to the LLM judge. The container
already states why that is harmless (`submissions/02-rung21-a2/inference.py`, `normalize_answer`
docstring): *"an answer that is exactly `"3."` or `"No."` means the same thing without the
period."* ⇒ **direct cost: zero.**

### Half 2 — whether the habit reaches the scored `number` questions

On A2 ep3 (`21_lr_2e4_v1/ep3_full`, the checkpoint that shipped as submission 02), over the
**raw** model strings:

```
number questions            2094
  clean digits              2094   (100.00%)
  repaired by container        0   (0.00%)
  NOT covered by the repair    0   (0.00%)
```

⚠️ **Checked before relying on it:** nothing normalises before `predictions.json` is written —
`src/frame/metrics.py:1392`, *"`run.run_baseline` persists the **raw model strings**"*. The
measurement is not circular.

🔑 **So `normalize_answer` is currently repairing nothing on the scored path.** It is insurance,
not a repair. That reconciles exactly with what the container already said and nobody had
measured: *"it is invisible to every number we report because **the scored eval only ever asks
the corpus templates**."* 16a's 86.7% is measured on phrasings **16a itself generates**; the
scored eval never asks them.

## What follows

1. 🔴 **Do not touch the corpus.** There is nothing to gain, and `train.jsonl`'s sha256
   (`180e28f0…8e8b`) is what makes rungs 18, 21, 27 and 30 paired-comparable.
2. 🟢 **Keep `normalize_answer`.** The final test set is different data from the validation
   leaderboard, and the repair is free. The change is that we now keep it **knowing it is a no-op
   today**, instead of believing it saves 86.7%.
3. 🔴 **No rung.** The causal claim — that these rows *installed* the habit — is neither proved
   nor refuted here, and it never had a cheap test: it needs a training run. With the cost
   measured at zero it has lost its reason to exist. Read this as *"the defect does not pay"*,
   **not** as *"the defect is not there"*.

## Limits, and one thing to reconcile

⚠️ **One checkpoint, one eval.** A2 ep3 on the 6,252-question local eval. A different checkpoint
could behave differently, and the organizers' final test set is not this set.

⚠️ **18 training rows is a thin cause for an 86.7% effect.** The correlation is striking — the
phrasing is unique in the corpus and every one of its golds carries the period — but the
mechanism between 18 rows of 14,415 and a habit that dominates off-template output is not
established. Nothing here needs it to be.

📌 **Count discrepancy, open:** rung 30 reports **32** such rows in `train.jsonl`; the corpus's
own train split carries **18** under this phrasing. `train.jsonl` comes from `18-count-aug`, so
duplication during augmentation is the likely explanation. Not a correction — the two numbers
should come from one computation and currently do not.

## Why the computation is committed and not just the number

`experiments/16-count-probes/_tools/trailing_period_audit.py` runs both halves. This is a direct
consequence of the same day's `±0.86` failure: a corpus-quality claim reached **14 files** and
closed levers while its computation was never committed, and when it was finally re-derived it
did not reproduce. **A number without a runnable derivation is not a record.**
