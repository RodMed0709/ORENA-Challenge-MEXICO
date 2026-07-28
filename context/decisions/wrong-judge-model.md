---
question: Is our LLM judge the one that scores us?
verdict: NO. `Qwen/Qwen3.5-4B` exists (HF, created 2026-02-27, 6.4M downloads) — the code comment claiming it does not is FALSE, and every judged number in this repo used a substitute
status: MEASURED
date: 2026-07-28
measured_in: huggingface.co/api/models + src/frame/config.py + vendor formats.py
question_derived: true
---
# Finding: the only instrument in the project that is knowingly wrong

`src/frame/config.py` sets `judge_model = "Qwen/Qwen3-4B"` with the comment
*"real HF id (SDK default 'Qwen3.5-4B' does not exist)"*.

**Verified 2026-07-28 against the HF API:**

| model | HTTP | created | downloads |
|---|---|---|---|
| `Qwen/Qwen3.5-4B` (the SDK default) | **200** | **2026-02-27** | 6,364,356 |
| `Qwen/Qwen3-4B` (our substitute) | 200 | 2025-04-27 | 4,584,845 |

The comment was presumably true when written and is **false now**. Nobody re-checked.

## Why it matters more than a config nit

`open_ended`, `multiple_choice` and `matching` are routed to the LLM judge
(`vendor/orena-focus/src/focus/data/formats.py:326`). That is **376 of the 1,296
`object_recognition` ID questions** — 29 % of half the pre-eval gate — and
`object_recognition` is exactly the bucket that collapsed −0.150 between our local val and the
platform ([[leaderboard-metric-vs-our-headline]]).

⇒ Part of that collapse may be **our measuring stick**, not our model. It has never been measured.

## The move it licenses (cheapest decisive action available)

**Re-score the committed rung-06 predictions under `Qwen/Qwen3.5-4B`** — ~1 GPU-hour, ~$1, no
training, predictions already on disk (`ep3_full/predictions.json` is local; `eval_best` is on the
volume). It decides which branch gets the remaining weeks:

- **Judges disagree materially** → the −0.150 is substantially instrument + a terse output policy,
  and the cheap fixes (judge-aware answers, shipping `normalize_answer`) are a path to the gate.
- **Judges agree ≥97 %** → the collapse is real capability on unseen centres, and a
  newer-generation backbone screen becomes mandatory rather than optional.

## Related, unpriced, and in the same place

The judge's rubric is **readable** (`vendor/orena-focus/src/focus/evaluation/judges.py:45-63`): it
rewards an answer that contains the right core term inside extra text and explicitly says *"Do NOT
consider formatting differences as errors."* Our SFT trains terse, template-locked strings — the
worst policy under that rubric. **Zero of 19 rungs has ever targeted the judged formats.**

⚠️ Do NOT treat the re-judge as a score. It is an instrument check; the headline stays whatever
`frame.metrics` produces under the judge we finally adopt, and switching judges must be its own
recorded decision, not a silent change.
