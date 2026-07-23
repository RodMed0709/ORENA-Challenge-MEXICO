# CONTEXT — 13 WiSE-FT weight interpolation (NO TRAINING)

## Objective

Answer one question and only one: **can a single scalar α, mixed into the weights of a
model we already have, buy back the counting that fine-tuning erased — without
retraining?**

`θ(α) = (1−α)·θ_base + α·θ_finetuned`, elementwise over every tensor of rung 06's merged
`checkpoint-1720` against `/workspace/models/qwen3-vl-8b`. The specific hypothesis, in
the ficha's words: **there is an α where the `number` margin recovers faster than the
`fo_class` margin decays**, which would lift `bucket_mean` for free
(`literature/vlm-techniques/FICHAS.md` v23, "Transfer to us").

Cost: CPU + disk for the interpolation, ~72 min of eval GPU for three arms. No training.

## Setup-config

- **The ONE variable:** α. Everything else is byte-identical to rung 06's `eval_best`
  protocol — same 6252 questions, same judge (`Qwen/Qwen3-4B`), same `max_pixels`
  (1280×720), same seed (42), same greedy decoding, same `frame.run.run_baseline`.
- **Endpoints are not re-run.** α=0.0 is `00-baseline` (`bucket_mean` **0.2557**) and
  α=1.0 is rung 06 (**0.5667**); both have committed `stratified.json` files and their
  numbers are read from there (WAVE_SPEC §Rung 13).
- **Engine:** `experiments/13-wise-ft/_models/interpolate.py` (interpolation + gates).
  **Report:** `experiments/13-wise-ft/report.py`. Eval is NOT in the engine: it reuses
  `frame.run.run_baseline`, the same division of labour rung 02/06 use.
- **Method + citations:** Wortsman et al., *Robust fine-tuning of zero-shot models*
  (WiSE-FT), CVPR 2022, arXiv:2109.01903 — ficha **v23**. The contingent follow-up is
  Wang et al., *LiNeS*, ICLR 2025, arXiv:2410.17146 — ficha **v13**.

### 🔴 The arithmetic identity nobody should discover later

rung 06's merged checkpoint is `base + ΔW`, where `ΔW` is the merged LoRA update (25.67 M
of 8792.80 M parameters, 0.29 %, rung 06 README). Therefore

    θ(α) = (1−α)·base + α·(base + ΔW) = base + α·ΔW

**α is exactly a post-hoc scale on the LoRA update.** This rung is arithmetically
identical to re-merging the adapter with `lora_alpha` multiplied by α — we do it in
weight space because that needs no `swift`, no GPU and no re-merge. Two consequences,
both recorded before any number: (1) a null here is also a null for "just scale the
adapter down"; (2) LiNeS is the *depth-dependent* version of this same scale, which is
why it composes and why it is the named follow-up.

---

## PRE-REGISTRATION (written 2026-07-23, before any α was built or evaluated)

### Arms

| arm | α | source | GPU |
|---|---|---|---|
| reference | 0.00 | `00-baseline`, committed | none — reused |
| **A1** | **0.50** | interpolated | 1 full eval (~24 min) |
| **A2** | **0.70** | interpolated | 1 full eval (~24 min) |
| **A3** | **0.85** | interpolated | 1 full eval (~24 min) |
| reference | 1.00 | `06-vit-lora`, committed | none — reused |

### Why this α grid, and what it is a subset of

- The method and the sweep come from ficha **v23** (WiSE-FT): "linearly interpolate the
  zero-shot and fine-tuned parameters with a single coefficient α… improves accuracy both
  in-distribution and under distribution shift **over a broad α range**". The ficha's
  own recommendation for us is `α ∈ {0.2 … 0.9}`; `FICHAS.md` §"What to steal — ranked"
  #1 widens it to `{0.1, 0.2, …, 0.9}` at a cost of **~9 eval passes**.
- **We run 3 of those 9.** This is a cost decision (3 × 24 min GPU and 3 × 17 GB of disk
  churn instead of 9), and it is OURS, not the paper's — declared here as non-negotiable
  #9 requires.
- **Why the upper half.** Measured on our own eval, not assumed: at α=0 the base model is
  **below the template-aware trivial floor on every format except `multiple_choice` and
  `open_ended`** — `number` margin **−0.143 ID / −0.303 OOD**, `fo_class` **−0.101 /
  −0.204** (`experiments/00-baseline/runs/00_zeroshot_qwen3vl/stratified.json`). There is
  no reservoir of skill at the α→0 end to walk toward, so the informative region is the
  one adjacent to the checkpoint that works.
- **What the grid cannot do:** three points spaced 0.15–0.20 apart cannot resolve an
  optimum narrower than that. A null result falsifies the hypothesis **at this resolution
  in this range**, not WiSE-FT on this model.
- **Escalation is a new pre-registration, not an edit.** If α=0.85 is the best arm and the
  curve is still rising at the top, a follow-up at α∈{0.9, 0.95} is a *new* rung with its
  own pre-registration. Adding a point after seeing the curve and folding it into this
  rung's verdict would be exactly the "threshold edited after seeing a number" that
  non-negotiable #7 forbids.

### 🔴 Gates — both RAISE, neither is ever disabled (`context/RULES.md` §7)

| Gate | What it proves | When |
|---|---|---|
| **A.1 parity** | base and fine-tuned expose identical keys, shapes and dtypes | before a byte is written (headers only, seconds) |
| **A.2 weight identity** | α=1.0 re-read from disk is **bit-identical** to the fine-tuned checkpoint (only tolerated difference: a `−0.0 → +0.0` flip) | before any sweep GPU |
| **B prediction identity** | α=1.0 reproduces rung 06's `predictions.json` **verbatim** on a frozen 200-question probe | before any sweep GPU |
| gold coverage | every scored qID has a gold answer (a partial gold understates the floor and **overstates the margin**) | per α, before any margin is read |
| `assert_ood_from_qid` / `assert_all_rows_grouped` / `assert_no_dup_qid` / `assert_floors_vs_eval_set` | canonical scoring is intact | per α |
| `assert_floor_cancels` | the template-aware floor is identical in both arms, so Δmargin **is** Δaccuracy | per α |

**Why A and B are load-bearing.** A wrong interpolation — one mis-paired shard, one
silently skipped tensor — still loads, still answers, and still produces a smooth,
plausible, entirely fictional trade-off curve. There is no downstream metric that can
see it. That is the failure this rung cannot survive, so the identity is a function that
raises, not a comment.

**The probe:** 200 qIDs, 100 per dataset, proportional over `answer_format` inside each,
drawn from 2 videos per dataset, frozen with a sha256 sidecar via
`frame.subsample.freeze`. **Gate B evaluates exactly those 200 qIDs**, via
`run_baseline(..., qid_filter=...)` (`src/frame/run.py:158`) — an identity gate has to
compare the same questions the control answered, not a superset that contains them, and
the notebook asserts `n_compared == probe_n` so the binding cannot silently loosen. The
draw is confined to a few videos only to keep the gate a ~3-minute run; a gate that costs
a full eval is a gate that gets skipped.
⚠️ It covers 4 videos: it is an **identity check and nothing else**. No CI, no margin and
no verdict is ever read off it.

### Decision rule (fixed here before any number)

An α **WINS** if, versus rung 06, **all three** hold:

1. **(a)** `bucket_mean` is not lower by more than **0.005**;
2. **(b)** the `number` **margin** is strictly higher in **BOTH** ID and OOD;
3. **(c)** the **paired, video-clustered** CI on the `number` margin delta **excludes 0**
   in at least one distribution.

Implemented in `report.decide` and unit-tested offline. Δmargin is bootstrapped as
Δaccuracy because the template-aware floor is a property of the gold answers and cancels
exactly in the paired difference — `assert_floor_cancels` checks that rather than
assuming it. The bootstrap is `frame.metrics.paired_delta_ci` (video→question, B=2000),
never a re-derivation.

**Every α is reported on every format and every bucket regardless of the verdict.** The
trade-off curve is the deliverable even if nothing wins. A NaN anywhere is
`INDETERMINATE`, never a pass.

### Stopping tiers (pre-registered)

- **T1 — pipeline defect.** Gate A or Gate B fires → **STOP the rung**. Nothing is
  reported about WiSE-FT; it is a finding about our interpolation code. Fix, re-gate,
  restart. Never widen the gate.
- **T2 — degenerate arm.** An α whose eval shows >1 % `Inference Error:` or >1 % empty
  answers is marked **DEGENERATE**: its numbers are still reported, but it may not
  support a WIN. Two degenerate arms out of three → stop the sweep and report that
  interpolation produces broken models in this range (that is itself a result).
- **T3 — no admissible arm.** No α satisfies (b) → the pre-registered WIN is unreachable.
  Record a **faithful negative**, and do **not** open the contingent LiNeS arm — WAVE_SPEC
  gates LiNeS on WiSE-FT showing a favourable trade.
- **T4 — favourable trade without a win.** Some α satisfies (b) and (c) but fails (a) →
  **not a win**, but the one condition that opens LiNeS. Opening it requires a NEW
  pre-registration in `context/14…`-style form, not an edit of this file.
- **Resource stop.** < 25 GB free on the volume → stop before building the next α.
  `assert_disk_headroom` raises rather than half-writing a checkpoint.

### Known limits, recorded before the result

1. 🔴 **The α→0 endpoint has no counting skill to recover, by margin.** The premise in
   the ficha ("give me back the counting the pretrained model had") does **not** survive
   contact with our own measurement: the zero-shot model's `number` margin is −0.143 ID
   and −0.303 OOD. For the hypothesis to hold, the curve must be genuinely
   **non-monotone** with an interior optimum — a strictly stronger claim than "recover
   what was erased". The evidence that motivates it is the *epoch* trajectory (rung 02
   `number` margin OOD +0.032 → +0.023 → +0.004; rung 06 +0.015 → +0.013 → +0.000), i.e.
   *less* training preserves more counting — which is suggestive of an interior optimum
   but is not the same measurement.
2. **Style, not only capability.** At α<1 the model drifts toward the base model's answer
   style, and the judge scores style (`fo_class` accuracy is 0.188 at α=0 vs 0.612 at
   α=1). Any decay we measure mixes capability loss with format drift, and this rung does
   **not** separate them. The health columns (`n_empty_answers`, `n_inference_errors`,
   answer length) bound the crude failure mode, not the subtle one.
3. **WiSE-FT is validated on CLIP-style classifiers**, not on a generative VLM with a
   merged LoRA (the ficha says so itself). The interpolation is well-defined; the result
   is unproven in our setting.
4. **Grid resolution** — see above: 3 of the ficha's 9 points.
5. **Effective n ≈ 38 videos** (28 ID / 10 OOD), not 6252 questions. Every CI here is
   clustered by video; the `number` OOD cell rests on ~10 videos.
6. **`number` may be an annotation ceiling, not a model ceiling** — the label moves ±0.86
   between frames ≤1 s apart (changing in 56.6 % of pairs) while the model's MAE is 1.01.
   A faithful negative here is consistent with either ceiling, and the write-up must say
   which.
7. **`bucket_mean` arithmetic — the win is worth less than it sounds.** `bucket_mean` is
   the mean of 4 equally weighted cells. Every `number` question sits inside
   `aggregation` (rung 06 `by_bucket_format`: 768 of 955 `aggregation` ID rows and 1326
   of 1875 OOD rows are `number`), so a `number`-only gain of +0.02 moves the headline by
   `0.02 × (768/955 + 1326/1875) / 4` = **+0.0076**. An α can therefore win the
   pre-registered rule and still be near-worthless on the leaderboard. That is accepted,
   not overlooked: the rule is built to detect the *mechanism*, and the mechanism is what
   LiNeS would build on.

---

## Results

*(empty — nothing has been run. Any number that appears here without a green Gate A and
Gate B is invalid by construction.)*

## Next

1. Run the SMOKE (`SMOKE = True`, one α, 40 questions) and get an independent review
   before the full sweep — CONSTITUTION §VIII.6.
2. Full sweep, then regenerate the root ledger (`frame.ledger.build_results_ledger`).
3. LiNeS (ficha v13) **only** under T4. Not built speculatively.
