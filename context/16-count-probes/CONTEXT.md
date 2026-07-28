# context/16-count-probes/CONTEXT.md — the curated read

> The artifact dir is `experiments/16-count-probes/`. This file is the *curated* half of the
> two-part store: why the rung exists, what it measured, what survives, and what it retires.
> Numbers live in `RESULTS_16*.csv`; verdicts live in `context/decisions/`.

## Why this rung is four probes and zero training runs

Rung 06 **ep3** (`checkpoint-2580`, `bucket_mean` **0.5724**) is the current best and the
epoch-matched control that closed rungs 14 and 15. It produced the sharpest number in the
campaign:

> **`number_margin_OOD` = 0.0 — exactly the trivial floor.**

Our best checkpoint adds *nothing* over "always answer the mode" on OOD counting. Every
remaining `number` idea — minted zeros, pointing supervision, tiled counting — costs a 7.5 h
run, and the 2026-07-27 literature sweep (five parallel readers, ~60 papers verified) found each
of them gated on a question nobody had measured. These probes measure the gates first.
**~5 GPU-hours against three 7.5 h runs.**

| probe | question | status |
|---|---|---|
| **16a** `zero_probe` | Can the model emit `0` at all — and did our SFT take it away? | ✅ **DONE**, n=960 |
| **16b** `detector_vs_gold` | Is the gold count something a detector can see in the frame? | 🔄 queued |
| **16c** `len_points` | Does `len(predicted_points)` beat verbalising the count? | ✅ **DONE, NEGATIVE**, n=681 |
| **16d** `format_audit` | Where else does fine-tuning emit SDK-illegal answers? | 🔄 queued |

⚠️ **The model under test is ep3 (`checkpoint-2580`), not ep2.** Verified from the run log, not
assumed. Rung 06's older artifacts reference `checkpoint-1720` (ep2) — do not mix them.

## A fifth question was closed for ZERO GPU

The **tiling / upsampling family**. `max_pixels = 1280×720 = 921,600`; Qwen3-VL uses 28×28 px per
visual token after the 2×2 merge. Over the measured 15,213-frame cache
(`experiments/11-resolution/.../RESULTS_dims_crosstab.csv`) the cap engages on **exactly one**
resolution — lapchole 1280×720, 5,036 frames (33.1%), all ID — at a **2.2% linear** cost
(1288×728 → 1260×700). Every other resolution passes untouched.

⇒ **No frame reaches the ViT meaningfully downscaled, so SAHI's mechanism — recovering detail
destroyed by a forced downscale — has nothing to recover.** Recorded in `src/frame/config.py`
with the derivation flagged *confirm on-pod*. Independently corroborated by CountGD (tiling moved
FSC-147 test MAE **6.75 → 6.75**, exactly zero) and by ViCrop's authors naming counting as the
category crop cannot help.

## 16a — what it gives us (n=960, 240 frames, balanced ID/OOD × absent/present × number/binary)

Full verdict: [[zero-is-format-localized]]. Three things, in order of how much they change plans.

### 1. The base model is a zero ATTRACTOR — the "restore the capability" plan is dead

`base` emits `0`/`no` on **83.1% of PRESENT cases**, where the object *is* there. Its 0.9375 on
ABSENT is negation, not discrimination (hmean 0.18–0.37 against `ft`'s 0.85 on binary).

**HoloCount's 96.4% does not transfer to laparoscopic frames.** The capability never existed in
this domain, so regularisation has nothing to restore.

🔴 **The PRESENT control arm is the entire reason we know this.** Reading the ABSENT column alone
would have said *"base has it, our SFT broke it"* — and sent rung 16 to repair something that
never existed. Any future absence/abstention probe must carry that arm.

### 2. The deficit is FORMAT-LOCALIZED, and the supervision explains it exactly

| ft cell | emit `0`/`no` on ABSENT |
|---|---|
| ID `binary` | **0.8167** |
| OOD `binary` | **0.8250** |
| ID `number` | **0.0083** |
| OOD `number` | **0.0167** |

The model says "no" fluently (0.82, only 0.10–0.12 false-negatives) and **never says "0"**: 1 case
in 120 ID, 2 in 120 OOD. That is one-to-one with the data — `binary` has **1,008 `no` golds**;
`number` has **zero zeros in 2,495**.

⇒ **Minting is licensed for `number` ONLY.** The `binary`/`fo_class` half of the original plan is
unnecessary: the model already holds that concept. Half the rows, half the cost, half the label
noise. Dose unchanged (LRV curve): **3:1–2:1**, adversarially sampled toward `Clip` and the
never-seen classes, verified against the 1,008 co-occurrence binaries, scored on both axes.

### 3. 🔴 The finding we were not looking for: fine-tuning broke `number` FORMATTING

On off-template phrasings `ft` emits **`"1."`** — trailing period — on **86.7% (ID) / 87.5% (OOD)**
of `number` questions. `Number.verify` gates on `str.strip().isdigit()`, so those are
**auto-incorrect**. `base` scores **0.0000** on the same metric. **We created this.**

It is not a counting error, and it is **invisible to every number we have reported**, because the
scored eval only asks the corpus templates.

⚠️ Scope, honestly: measured on *our probe's* phrasing, not the organizers'. It does not touch
rung 06's 0.5724. But the hidden test is their generator, and [[open-class-vocabulary]] already
records a possible train/test regime mismatch (70% of our questions lack the class list the spec
says they carry).

**Cheapest mitigation, zero training:** `answer_postprocess` already exists in `frame.config`
(rung 15) and already runs between generation and SDK verification. Stripping a trailing `.` from
a `number` answer is one line, flag-off byte-identical. **Measure it before anything else.**

## 16c — what it gives us (n=681, the full Clips template, 37 videos)

Full verdict: [[prompt-only-pointing-collapses]]. **Faithful negative.** Both pointing arms lose
to the bare-integer control in every cell — a1 OOD margin **−0.1162**, a2 **−0.1389**, against the
control's **−0.0025** — at 3.4× the latency for a1.

🔴 **The mechanism is degenerate and visible: a2's `mean_pred` is exactly 1.0000 across all 681
questions.** Asked to point at every clip and number them, the model emits **one** point, every
time. So `len(points)` is not reading a count off an enumeration — it is reading a constant, which
is why it *amplifies* the undercount (bias −2.52 vs the control's −0.66) instead of curing it.
Same shape as [[naming-equals-counting]], one level out.

🟢 **The probe validated its own instrument before it read anything new:** `a0`'s OOD margin
(−0.0025) reproduces the epoch-matched control's `number_margin_OOD` = 0.0 on an independent code
path.

⚠️ **This bounds the prompt-only path, not the trained one** — Alghisi state training-free
point-then-count is weak without task-specific supervision, and their 94.96% came *with*
point-supervised LoRA. It stays open as a training target, **gated on 16b**: if the gold is not
frame-visible, the trained version dies too.

## 16d exists because finding #3 was luck

The `"1."` bug surfaced because a control arm happened to use an off-corpus phrasing. 16d turns
that into a sweep, and it is nearly free for one reason:

🟢 **`sdk_valid` needs no ground truth.** "Would the SDK's verifier accept this string?" is a
property of the OUTPUT, so any phrasing can be audited without annotating anything.

🔴 **Which is also why the class of defect was invisible.** The scored eval only ever asks the
corpus's own templates; a fragility outside them cannot appear in any number we report, **by
construction**. That is a structural blind spot in our measurement, not a one-off bug.

Its product is `regressions()` — the places where `ft` is more illegal than `base` (did *we*
cause it?) or than its own `v0` (is it phrasing-induced, i.e. invisible to our eval?).
A faithful negative here is a real result and must be recorded as one.

## Instrument defects found and fixed while building this rung

Recorded because each one produced a plausible, wrong number before it was caught.

1. **Wrong SDK attribute, silent.** `Reference` carries the format as `_format`, not
   `answer_format`. `getattr(ref, "answer_format", "")` returned `""`, every branch missed, and the
   probe set came out **empty rather than raising** — burning a 17 GB checkpoint load on 0
   questions. Fixed with an accessor that RAISES on an unknown schema, plus notebook gates that
   abort on an empty or unbalanced probe set *before* a model loads.
2. **papermill parameters, silent.** Derived values computed *inside* the cell tagged
   `parameters` are fixed from the pre-injection values and never recomputed, so `-p SMOKE False`
   changed `SMOKE` and nothing else: the first "full" run was **the smoke again** (n=64).
   Parameters cells now hold raw literals only, a derive cell sits below them, and a gate asserts
   the realized frame count matches the declared mode. **A parameter that fails to take effect is
   otherwise indistinguishable from a successful run.**
3. **Parse conflation.** Counting `"1."` as *malformed* hid the actual finding (the model said
   ONE, not zero) behind a parse failure. `read_answer` now returns a lenient `value` for the
   capability question and a strict `sdk_valid` for the scoring question, reported separately.

## Reading rules for this rung

- **Read `emit_rate`, not `acc`.** The ABSENT label is closure-derived; measured non-circularly
  against the co-occurrence binaries it has **0.00% false positives** but **8.4% under-naming**
  (n=333; a second cross-check agrees at 10.7%). `emit_rate` is a property of the output and is
  immune; `acc` is not.
- **Never read ABSENT without PRESENT.** See finding #1.
- **These are probes, not scored runs.** They do not produce a `stratified.json` and are
  deliberately NOT in the `results/` ledger — `frame.ledger` is for canonical eval runs and
  injecting a probe would corrupt the comparison it exists to protect. Their numbers live in
  `experiments/16-count-probes/RESULTS_16*.csv`.
- **16b's thresholds are pre-registered** in its notebook and must not be tuned after seeing `r`.

## Links

[[zero-is-format-localized]] · [[epoch-matched-control]] · [[the-gap-is-the-number-format]] ·
[[resolution-is-not-the-gap]] · [[open-class-vocabulary]] · [[count-calibration-dead]] ·
[[naming-equals-counting]] · [[self-consistency-dead]]
