---
question: Is the shipped submission container a source of lost points independent of model quality?
verdict: PARTIALLY — one real defect fixed (a trailing period makes an otherwise-correct `number`/`binary` answer auto-incorrect; probe 16a measured 87% on off-template phrasings), and one theoretical risk measured to zero (0 adversarial-signal hits across 146,004 committed answers, so no output filter is warranted or wanted)
status: MEASURED
date: 2026-07-28
measured_in: experiments/06-vit-lora/_tools/submission/test_normalize_answer.py + a scan of 34 predictions.json files on the pod
---

# Decision: the container audit — one defect fixed, one risk measured away

- **Status:** MEASURED · 2026-07-28 · **zero GPU** (static audit + a scan of already-committed answers)
- **Applies when:** touching `experiments/06-vit-lora/_tools/submission/inference.py`, or costing
  any explanation for a leaderboard score below the offline score.

## Why the container was audited at all

Submission 01 shipped rung 06 **ep2** (`checkpoint-1720`). Probe 16a measured the **ep3**
checkpoint emitting `"1."` — with a trailing period — on **86.7% (ID) / 87.5% (OOD)** of `number`
questions phrased outside the corpus's own templates, against a base-model rate of **0.0000**.
Those are different artifacts, so the shipped one had never been audited, and the container is
the last thing standing between the model and the scorer.

## Finding 1 — 🔴 REAL, and fixed: a trailing period is auto-incorrect

Read from the SDK, not inferred (`vendor/orena-focus/src/focus/data/formats.py`):

| verifier | exact gate | `"1."` | `"Yes."` |
|---|---|---|---|
| `Number.verify` | `text.strip().isdigit()` | **rejected** | — |
| `Binary.verify` | `text.strip().lower() in ("yes","no")` | — | **rejected** |

Both are exact-match after `strip()`. And the container did **no** normalisation: `answer_one`
ended in `.strip()` and a 300-char slice (`inference.py:336`), so any illegal string reached the
verifier unchanged and scored wrong.

⚠️ **The container cannot be format-aware.** `Request` carries qID, videoID, start/end time,
procedure_type and question — **no `answer_format`**; that field lives on `Reference`, which a
participant never sees. So the repair had to be format-agnostic.

**Fix:** `normalize_answer()` fires only when the *entire* answer is digits-then-period or
yes/no-then-period. Everything else is returned byte-identical, and every repair is logged
(`log.info("qID=%s normalised %r -> %r")`) — if it fires often, the model has an upstream problem
that deserves a real fix, not a patch here.

**Verified** by `test_normalize_answer.py`, which checks against the SDK's **real** verifier
objects rather than a copy of them: **7 repairs accepted, 22 inputs byte-identical**. The
identity half is the safety gate — this runs in the shipped container, so a normalisation that
touched anything beyond its target would silently alter answers that were already correct.
Non-targets confirmed untouched include `"3.5"`, `"1..."`, `"1,000"`, `"Clip, Sponge."`, `"12:03"`.

## Finding 2 — 🟢 measured to zero, and deliberately NOT "fixed"

`AdversarialDetector.check()` is called at `evaluator.py:217`, inside a `ThreadPoolExecutor`
future, and **raises `RuntimeError` on a hit**. Its own docstring: *"Any flagged submission raises
RuntimeError"* — a **submission-level** flag, not one wrong answer. Neither our engine nor the
container screens output.

**Measured:** every committed answer we own — **34 `predictions.json` files, 146,004 answers** —
scanned against the 12 public injection signals. **Zero hits.** Also: max answer length **199
chars**, p99 **30**, and **zero** answers over the 300-char cap, so the container's truncation has
never fired either.

🔴 **No output filter was added, and that is the decision, not an omission.** Two reasons:
1. At 0/146,004 the risk is theoretical.
2. The module states the full detector is withheld *"to prevent participants from tuning their
   outputs to bypass detection."* Stripping flagged phrases to evade it is exactly the behaviour
   that clause exists to stop. If our model ever emits one, the correct response is to fix the
   prompt or the model — not to launder the output.

⚠️ **Residual risk, stated plainly:** the public list is a **subset**; the official detector is
private and stronger. Zero hits on the public signals does not prove zero on the private ones.

## Finding 3 — ⚠️ a divergence that will bite the next fix

`frame.engine.QwenFrameEngine.predict()` has an `answer_postprocess` hook; the container has no
equivalent and never had one. Today that is harmless — rung 06 was scored with
`answer_postprocess=None` — but it means **a fix applied through the hook does not reach the
container**, and nothing asserts the two paths agree. The container's docstring claims it
"Mirrors QwenFrameEngine.predict()"; that claim is now one function out of date in the other
direction, and should be kept honest whenever either side changes.

## ⚠️ CORRECTION (2026-07-28) — the trigger is OOD questions, not paraphrasing

Probe 16d asked **real corpus questions** under six surface variants across base, `ep2_shipped`
and ep3: **0.0000 illegal answers, every model, every variant, every format** (smoke, n=6/format;
full pending). 16a's 87% came from applying one *synthetic* template across all eight classes,
five of which the model has almost never been asked to count (gallstone 4 training examples,
specimen bag 11, silicone loop 17, needle 19). Those five score **1.0000** illegal; `Clip`
(2,071 examples) scores 0.654.

⇒ **The defect fires on out-of-distribution (class, question) pairs, not on rewording.** The fix
below is still correct and still free, but the probability that submission 01 actually walked
into it is **much lower** than this note first implied. The honest exposure: if the organizers
ask a count for a class we barely trained on, we may emit an auto-incorrect string.

## ⚠️ What is still unmeasured

**Whether ep2 — the actually-shipped checkpoint — exhibits the trailing-period defect.** 16a
probed ep3 only. Probe **16d** now runs three arms (`base`, `ep2_shipped`, `ft`) and prints the
comparison directly. Until it reports, the honest statement is: *the defect is proven on ep3,
the container could not have caught it, and it now would.*

And the whole class is measured on **our** phrasings, not the organizers'. The reason it matters
anyway: our scored eval only ever asks the corpus's own templates, so a formatting fragility
outside them is invisible to every number we report — **by construction**.

## Sources

- `experiments/06-vit-lora/_tools/submission/inference.py` (`normalize_answer`, `answer_one`)
- `experiments/06-vit-lora/_tools/submission/test_normalize_answer.py`
- `vendor/orena-focus/src/focus/data/formats.py` (`Number.verify`, `Binary.verify`)
- `vendor/orena-focus/src/focus/evaluation/adversarial.py` + `evaluator.py:217`
- [[zero-is-format-localized]] (probe 16a, the 87% measurement) · [[submission-01-rung06]]
