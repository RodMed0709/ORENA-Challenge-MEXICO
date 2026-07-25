# context/15-count-target — the structured counting target

> **Status: PRE-REGISTERED, NOT RUN.** Everything below the line "Pre-registration" was written
> **before any number from this rung existed**. A threshold edited after seeing a number stops
> being a gate and becomes a story (wave spec non-negotiable #7). If a threshold here ever
> changes, it changes in a commit that says so, with the reason, and the run is re-labelled.

Artifacts: `experiments/15-count-target/` · Control: `06-vit-lora` (`bucket_mean` 0.5667,
`checkpoint-1720`, merged weights already on the volume — **not retrained**).

---

## Objective

Rungs 02 and 06 train every `answer_format == "number"` row on a **bare integer** (`"3"`). The
whole gradient for a counting question lands on one token, so the objective is satisfiable by
learning a prior over that token instead of a count. What we measure is exactly that shape
(`context/ERROR_ANATOMY.md`, `context/decisions/count-calibration-dead.md`):

- bias **−0.66**; golds 3 and 4 share modal prediction **2**; golds 5–8 share modal prediction
  **4**; accuracy **0.000** at gold ≥ 7;
- the `number` margin over the template-aware floor **decays with training**, OOD:
  rung 02 +0.032 → +0.023 → +0.004; rung 06 +0.015 → +0.013 → **+0.000**;
- and `number` owns the gap: `aggregation × ID` is **80.4 % `number`**
  (`context/decisions/the-gap-is-the-number-format.md`).

**Question.** Is the counting collapse a property of the task, or of the target format we train
on?

## The single variable

The assistant `content` of `answer_format == "number"` rows, **and nothing else**. `system`,
`user`, `images`, `max_pixels`, LoRA recipe, LR, schedule, seed, epochs and frame cache are rung
06's, unchanged. Three gates make that a property of the code:

| gate | what it proves | where |
|---|---|---|
| **G-ARGV** | rung 06's own `_swift_args` on both configs → **0 flags differ** | `count_target.diff_vs_rung06` |
| **G-OFF** | `count_target=False` → `train.jsonl` **sha256-identical** to rung 06's | `count_target.assert_flag_off_identical` |
| **G-ON** | flag ON → every non-`number` line copied *verbatim*; every `number` line changed | `count_target.assert_nonnumber_lines_identical` |
| **G-REGEX** | `is_count_question` agrees with `answer_format` on **every** item, both splits | `count_target.assert_count_regex_matches_formats` |
| **G-PARSE-1** | the parser round-trips **100 %** of the exported targets | `count_target.assert_targets_round_trip` |
| **G-PARSE-2** | malformed generation degrades to a `Number`-valid answer, never raises | `count_target.assert_parser_never_raises` |
| **G-SAFE** | `SAFE_ANSWER` is not the modal answer of any `number` template | `count_target.assert_safe_answer_not_modal` |

Gates RAISE and are never disabled (`context/RULES.md` §7).

## Why this format — every choice cites a paper (non-negotiable #9)

Target, one line, no prose, no code fence:

```
{"label": "Clips", "counts": 2}
```

- **Field names `label` / `counts` are v05 Gautam 2025 verbatim** (arXiv:2505.16647,
  `literature/vlm-techniques/FICHAS.md`). Quoted: count annotations *"stored as
  `{"counts": 3, "label": "polyp"}`"*; key trick *"force a structured, parseable output (JSON
  with explicit `counts` field)"*. Reported effect: **Count MAE 9.86 → 0.26** on a Qwen2.5-VL-7B
  LoRA (r=16, ViT frozen, 5 epochs) — the same model family, the same PEFT family, our scale of
  data.
- **Label-first order** is the ONE declared deviation from v05's key order. Justified by **v28
  Guo 2025** (arXiv:2510.04401): compositional counting fails because object-type binding fails,
  and its stated consequence is that *"a per-class count target ('clips: 2, sponges: 0') could be
  more learnable than a single scalar"* — the count must be bound to a named class, and v28's own
  example writes the class before the number.
- **Why a structured target at all, on THIS backbone: v14 Qwen3-VL Technical Report**
  (arXiv:2511.21631). Quoted: pretraining includes *"normalized grounding with box-based,
  point-based and counting supervision in a [0,1000] coordinate system"*, and Qwen3-VL *"extends
  the grounding capacity to support counting"*. Counting is a pretrained **grounding** capability
  with its own output convention; a bare integer may be fighting that circuit rather than using it.
- **`label` is the question's own noun phrase, verbatim** — the capture group of
  `COUNT_TEMPLATE_RE`. A singularisation rule or a class-name lookup table would be invented
  content; a substring of the question is not.

**What we deliberately do NOT copy, and why.** v05's second half (joint pointing) and v14's
`[0,1000]` point convention both need localization labels. There are none: *"The parquets carry
`question` / `answer` and metadata only: no bounding boxes, no masks, no coordinates"*
(`context/ERROR_ANATOMY.md`). **Point-then-count is not reproducible on this dataset and is not
attempted.** This rung tests only the part of v05 our labels support: the structured, parseable,
label-bound count field. That is a real weakening of the v05 result and it is stated up front,
not discovered afterwards.

**One design choice is OURS, not a paper's,** and is declared as such: the parser's `salvaged`
tier (if the generation contains exactly one integer anywhere, use it; two or more is ambiguous
and degrades instead). Chosen because it is the strictest recovery rule that still catches a
truncated 64-token greedy generation.

## The parser — a first-class gate

`focus.data.formats.Number.verify` requires `text.strip().isdigit()`, and
`Evaluator._evaluate_single` (`vendor/orena-focus/src/focus/evaluation/evaluator.py:337-343`)
marks anything `fmt.read` rejects as **incorrect**. A structured answer reaching the judge is a
guaranteed zero. `parse_count` therefore runs inside `predict`, before `Response.content` exists
(there is no later hook — `run.py:186` builds the Response straight from the generation).

- **Round-trip: 100 %, gated over the whole exported set, RAISES.** If the training signal and
  the inference path disagree, every `number` number this rung produces is meaningless.
- **Never raises.** Degrades to `SAFE_ANSWER = "0"`, which is `Number`-valid *and* is not the
  modal answer of any `number` template (committed data-card tables: every modal answer is 1, 2
  or 3). So a malformed generation **costs** us instead of quietly inheriting the template floor.
  That is deliberate: it keeps the malformed rate an honest cost visible in the score.
- **Malformed rate is measured and reported next to every score**, against a published
  reference: **Perek 2026 (v02) measures 4.76 %** malformation for vanilla CoT-SFT.

## Setup / config

| | |
|---|---|
| backbone | `/workspace/models/qwen3-vl-8b` (Qwen3-VL-8B-Instruct, Apache-2.0) |
| tuner | LoRA r=8, α=32, dropout 0.1, `all-linear`, `freeze_vit=false`, `freeze_aligner=true` |
| optim | lr 2e-5, cosine, warmup 0.03, 3 epochs, bs 1 × grad-accum 16, seed 42 |
| pixels | `MAX_PIXELS = 1280×720` (train tokens == serve tokens) |
| split | `experiments/splits/frame_ood_v1.csv` — train 13,748 / eval 6,252 (`test`) |
| eval | full 6,252, `frame.run.run_baseline`, judge `Qwen/Qwen3-4B`, greedy |
| scoring | **only** `frame.metrics.stratified_report` |

## Pre-registration

### Arms

| arm | what | cost |
|---|---|---|
| **A0 — control** | rung 06, `06_vit_lora_v1`, `checkpoint-1720`. **Committed numbers reused; NOT retrained and NOT re-evaluated.** | 0 |
| **A1 — count target** | identical run, `count_target=True`. 3 epochs, all 3 merged and evaluated. | 1 training + 3 evals |

The flag-OFF configuration is **a gate, not an arm**: it exists to prove byte-identity and is
never scored (scoring it would just reproduce A0 at the cost of a training run).

No contingent second arm is pre-registered. Per-class count vectors (v28's `{"clip": 2,
"sponge": 0}`) would need a cross-question join to synthesise labels we do not have per row; it
is out of scope for this rung and is not built speculatively.

### Decision rule (the wave spec's, verbatim, with its instruments named)

> **WIN** = the `number` margin is up versus rung 06 in **BOTH** ID and OOD, with a paired
> video-clustered CI on the delta excluding 0, **AND** no other format's margin is down by more
> than **0.02**.

- **Margin** = accuracy − template-aware trivial floor (`frame.metrics`), never raw accuracy.
- **The paired delta on `number` correctness IS the margin delta.** The floor depends only on the
  gold answers and the templates, which are identical across arms, so it cancels exactly in the
  difference. `metrics.paired_delta_ci` (video→question two-level bootstrap, B=4000, seed 42) is
  therefore the right instrument and no new estimator is introduced.
- **Video-clustered, always.** Effective n ≈ **38 videos**, not 6,252 questions.
- **Checkpoint selection = rung 06's rule, unchanged: max `acc_OOD`** (`context/RULES.md` §6).
  Changing the selection criterion would be a second variable. **Every epoch is merged, evaluated
  and reported regardless**, with its `number` margin in both distributions, so a reader can see
  what any other rule would have chosen. Nothing is selected last-by-default.
- **Every format is reported in both distributions either way** — a faithful negative is the
  deliverable, not a failure.
- 🔴 **`acc_number` is never quoted.** It averages 8 val templates whose floors run 0.24–1.00,
  four of them degenerate (`context/RULES.md` §12; data card rung 08 §3). The read is the
  canonical `by_format` margin **plus** the per-template table
  (`RESULTS_number_by_template.csv`).

### Pre-registered risk: leakage into other formats

Changing the output distribution for `number` can leak into the four formats we did not touch —
that is why the "no other format down by more than 0.02" clause is part of WIN and not a
footnote. `fo_class` is the one to watch: it shares the multiplicity defect with `number`
(1.24 classes emitted vs 1.25 gold, with `clip` as an attractor emitted 875× vs 615 golds).

### Stopping tiers

| tier | what must hold | if it fails |
|---|---|---|
| **T0 — offline** | G-ARGV empty; G-PARSE-1 100 %; G-PARSE-2 clean; regex exact on both committed template tables | **STOP, fix the code.** Costs nothing, so it is never skipped. |
| **T1 — SMOKE** | tiny export + 2 train steps + 40-question eval complete; G-ON fires; `read_g1` shows LoRA (few M trainable) and vision-tower targets | **STOP.** A chain that does not run end-to-end never gets a full run. |
| **T2 — G-OFF** | flag-OFF full export sha256 == rung 06's `train.jsonl` | **STOP.** Without it this is not a single-variable A/B. |
| **T3 — train** | no NaN/inf; epoch-1 eval_loss better than the first train loss (rung 06's run guard) | Guard aborts → report the numbers; **do not** touch LR / batch / `max_pixels` (each is a second variable). |
| **T4 — eval** | malformed rate reported for every epoch | Malformed rate **> 4.76 %** (Perek 2026's vanilla-CoT-SFT figure): the format is not being learned reliably; the `number` result is reported **as confounded by malformation**, and the honest next move is the target's *length*, not another threshold. |
| **T5 — verdict** | the decision rule above, read by a human | Any outcome is recorded in the ladder. **Faithful negatives are not re-rolled.** |

### What each outcome will be taken to mean — committed in advance

The result must be attributed to one of two ceilings, and we say now which evidence maps to which:

- **WIN** ⇒ the collapse was the **target format**, i.e. a *model/objective* ceiling that the
  bare-integer target imposed. Follow-ups then earn their compute: NTL-WAS masked to `number`
  (v12), and SCALe-style answer-weighted loss (v02).
- **`number` margin flat or down, with malformed rate ≤ 4.76 %** ⇒ the format was learned and did
  not help ⇒ evidence for the **annotation ceiling**, and the strongest such evidence we would
  have. The write-up must then say so plainly and **stop spending training compute on counting**;
  what settles it next is annotation, not another rung.
- **`number` margin flat or down with malformed rate > 4.76 %** ⇒ **uninterpretable on the
  ceiling question.** Report as a format-compliance failure, not as evidence about counting.
- **`number` up but another format down by more than 0.02** ⇒ NOT a win. Report the trade-off
  curve; a format-local gain that costs a whole `bucket_mean` cell is worth −0.004 of headline at
  best (a `fo_class`-only gain of +0.02 buys +0.004 of headline; the arithmetic is symmetric).

## Known limits — stated before the run, not after

1. 🔴 **`number` may be an ANNOTATION ceiling, not a model ceiling** (`context/ERROR_ANATOMY.md`).
   The label moves **±0.86** between frames ≤ 1 s apart and **changes in 56.6 % of pairs**;
   the model's own mean absolute counting error is **1.01** — i.e. within ~1.2× of how much the
   *target itself* moves between adjacent frames. A blind non-clinical human scored **r = −0.17**
   against the gold on 40 frames (no relationship at all) while the model scores **r = +0.43**.
   **The model already outperforms an untrained human at this task.** If this rung produces no
   `number` gain, that is the reading it is consistent with, and the write-up will say so
   (see "What each outcome will be taken to mean").
   ⚠️ The honest caveat, from the same note: frame-to-frame change mixes genuine scene change
   with annotation noise, so 0.86 is an **upper bound** on the noise, not a clean estimate.
2. **No localization labels anywhere in the dataset**, so v05's pointing arm and v14's `[0,1000]`
   convention — the two mechanisms both papers credit for the gain — cannot be reproduced. This
   rung tests the weaker, label-supported half. A negative therefore does **not** falsify v05.
3. **The task is really "count clips."** 83 % of per-class counting questions are about `Clip`,
   and 93 % of the counted mass on the total-instances template is clips. A `label` field whose
   value is "Clips" in most rows carries less binding information than v28's multi-class example.
4. **One target format, one run.** Length, key order and the choice of `counts` over an
   enumeration are not swept; the literature does not give a dose curve for any of them, and
   inventing one is exactly what wave R1 did wrong.
5. **The parser is a real intervention on the answer path.** It can only help a structured
   generation and can never hurt a bare integer (tier 1 returns it unchanged), but it is
   nonetheless part of the A1 arm and not of A0 — the comparison is
   *(bare target, no parser)* vs *(structured target, parser)*, as a package. Splitting them
   would need a third arm and is not pre-registered.
6. **The parser reaches the answer path through `BaselineConfig.answer_postprocess`**
   (`src/frame/config.py:46`), applied by `QwenFrameEngine.predict` at `src/frame/engine.py:129`
   — after generation, before `Response.content` is built and before the SDK's format
   verification. `count_answer_hook` yields the callable and the parse log; flag OFF yields
   `None`, which is the field's own default, so the control arm's answer path is byte-identical
   and the notebook asserts the wiring matches `cfg.parse_answers`. This replaced a monkeypatch
   of `frame.run.QwenFrameEngine`: the patch wrapped the same `predict` at the same point and was
   functionally correct, but it was correct by coincidence of module-global resolution order, in
   the one place where a post-processor that fails to wire is indistinguishable from a model that
   cannot count.

## Results

*(empty — the run has not happened. Filled from `RESULTS.csv`, `RESULTS_epochs.csv`,
`RESULTS_number_by_template.csv` and `runs/<run>/paired_delta.csv`, never re-derived by hand.)*

## Next

Decided by T5 against the rule above, not before.
