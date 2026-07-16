# Context 05 — Bottleneck Audit

## Objective
Determine whether the rung-02 LoRA learned to *see*, or merely learned a *text shortcut* for answering
visual questions. Diagnosis only — it bifurcates the roadmap; it produces no candidate.

## Setup-config
- **Baseline**: rung 02 (LoRA), merged `checkpoint-1720` (epoch 2) — the one that scored `pre_eval 0.708`.
  Not rung 00: the question is whether **the LoRA created** the shortcut.
- **Split**: `frame_ood_v1` complete (2252 ID + 4000 OOD = 6252).
- **The ONE variable**: the image fed to the model (`a0_real`, `a1_black`, `a2_shuffled`).
- **Judge**: the project's local judge (`Qwen3-4B`). It is our ruler, not our grade — arms are compared
  against each other under the same ruler.

## Data insight (T0)
- **`aggregation` is NOT just `number`.**
  - **`train`**: 77.1% `number` + 22.3% `binary` (all `binary` = 8.9% of the split).
  - **`val`**: **74.0% `number` + 25.6% `binary`** (all `binary` = 11.6% — much heavier in `val`).
- **Real score weights (val-based)**: `fo_class` ~39.1% · `number` ~37.0% · `binary` ~12.8%.
  **`fo_class` and `number` are ~1:1** — this refuted the earlier claim that "a point in `number`
  is worth two in `fo_class`". THE_MAP was right; the correction was made on an unverified premise.

## Decisions
- `a2_shuffled` (images swapped across different videos) is the **primary control arm**, not `a1_black`:
  a black image is OOD for the ViT and degrades by *rarity*, not only by absence of information →
  it exaggerates the drop. Shuffling preserves visual statistics and breaks only the image↔question
  link. **The full run confirmed the design**: `a1_black` (0.2752) < `a2_shuffled` (0.3338) < `a0_real`
  (0.5503), exactly as predicted.
- Read tensors from disk as **lossless PNG** before the inference loop (JPEG would introduce a second
  variable; pre-loading avoids thrashing the decord cache).
- Trivial floor is measured **against the evaluated set** (`val` majority per `answer_format`), with the
  `train` prior reported alongside for contrast. Measuring the floor against `train` was a spec defect.
- `bucket_mean` covers exactly the 4 buckets = {`object_recognition`, `aggregation`} × {ID, OOD}.
  `temporal_grounding` (n=1) is excluded via `valid_buckets` — ID/OOD must never be pooled
  (2252 vs 4000 would weight OOD at 64% instead of 25%).

## Results

**Verdict: NO SHORTCUT. The model looks.** No format triggers the pre-registered `SHORTCUT` condition.
Full numbers and the per-format rule application: `experiments/05-bottleneck-audit/README.md` +
`RESULTS.csv`.

- **Control valid**: `a0_real` `acc_OOD = 0.59175` reproduces rung 02's recorded `0.5918`.
- **G2 PASS** (3 distinct pixel-tensor hashes) · **G2b**: 4281/6252 (68.4%) answers change under ablation.
- **`fo_class`** (39.1%): 0.5895 → 0.2303 when shuffled, **below** the 0.2692 trivial floor. Strongest
  evidence of genuine perception.
- 🔴 **`number`** (37.0%): passes `LOOKS` **for the wrong reason** — `A_real` 0.4317 is only **+8 pts**
  over the 0.3520 floor, and `a2_shuffled` retains **82.6%**. `a1_black` scores 0.3519579751671442,
  identical to 16 digits to the floor: with a black image the model collapses to answering the mode.
  **The honest reading is "barely extracts anything" — and it carries 37% of the score.**
- **Defect in the pre-registered rule**: for `number` the `SHORTCUT` (≥0.3885) and `LOOKS` (≤0.4020)
  bands **overlap**; a value in between fires both. Undefined when `A_real ≈ A_trivial`. It did not bite
  (0.3567 falls outside) — by luck. Recorded, not retro-patched. Same pattern as the earlier spec
  defects: **what gets measured must be defined in columns, not in prose.**

### 🔴 Incidental finding — `pre_eval 0.708` is inflated by ONE question
Found while independently recomputing the aggregates from raw predictions; not what this rung tested.

- **The whole `RESULTS.csv` reproduces exactly from the per-arm raw `results.csv`** (all 3 arms:
  `acc_overall`, `acc_ID`, `acc_OOD`, the 4 buckets, every format). The aggregation is verified,
  not trusted.
- **But `pre_evaluation SCORE = 0.7087132444965948` — the project's `0.708`** — is the SDK's
  unweighted mean over **3 populated buckets**: `aggregation` 0.5170 (n=2830), `object_recognition`
  0.6092 (n=3421), **`temporal_grounding` 1.0 (n=1)**. Formula reconstructed exactly (`MATCH=True`)
  on all 3 arms against `focus/evaluation/evaluator.py:402-462`.
  **Without the n=1 bucket: 0.5631. One question is worth +14.6 points.**
- **Cause 1 — `ood` is `False` on all 6252 rows** → the evaluator sees only ID buckets → 3 of 10 → the
  OOD dimension vanishes from its own score. **Anything trusting `results_df["ood"]` reads the entire
  split as in-distribution.** This rung derives ID/OOD from the `qID` prefix (`heico`=OOD,
  `lapchole`=ID) — that is why `bucket_mean` holds where `pre_eval` does not.
  ⚠️ **Expected, NOT a defect** (`CONSTITUTION.md` §I.5): the **public** data ships without those labels;
  the **organizers populate them in the private test split**. Our OOD is a *proxy we invented* (rung 01,
  hold out Sigmoid) that we never stamp onto the field, so the SDK's scorer cannot sort our questions.
  **The submission is unaffected** — on the leaderboard they sort with their own labels. Only our local
  instrument is.
- **Cause 2 — `frame_ood_v1` carries 1 `temporal_grounding` question**, a group FRAME should not have;
  under an unweighted bucket mean it equals a bucket of 3421. **This one is ours.**
- 🔴 **Nothing to fix on the `ood` side — it is avoided, not repaired.** Stamping our own labels would
  buy a `pre_eval` over our *invented proxy* — which is what `bucket_mean` already is, and `bucket_mean`
  additionally drops the orphan. **The answer is: do not use `pre_eval` locally.** *(An earlier version
  of this file sold it as "a one-line, no-GPU atomic, the cheapest item open". Wrong on both counts.)*
- **Consequence:** `pre_eval` is not a usable selection metric on our split, and `0.708` is not
  comparable to the official baselines. `bucket_mean` (0.5503) is the honest number.
  **The shortcut verdict is unaffected** — it was computed per format from raw per-question correctness.
- **Not fixed here** (changing the split + the metric is its own atomic).
- **Also observed:** `latency` is `0.0` and `timed_out` `False` for all 6252 rows — this path records no
  timing. The deferred p99 atomic cannot be served by these artifacts; the column exists but is empty.
- **Also observed:** `summary.csv` `group`/`answer_format` rows are **video-clustered** estimates (wide
  CIs), *not* flat per-question means — e.g. `object_recognition` reads 0.6237 there vs 0.6092 flat.
  **Do not mix the two.** `pre_eval` uses flat; our `bucket_mean` uses flat.

### ⚠️ Open discrepancy — do not smooth over
The T0-era note claimed the rung-02 LoRA scored **0.597 on `binary`** (+3.8 over the 55.9% val-majority
floor). The full run measures **`a0_real` `acc_fmt_binary` = 0.7693** (+21.0 over the same floor).
Overall `acc_OOD` matches rung 02 exactly, so the control is sound — the earlier 0.597 figure is
unexplained (different denominator or parse). **Flagged, not reconciled.**

## Provenance
- **Canonical source**: `05_bottleneck_audit.ipynb`, `sha256 = d8d0459185b148b3b31c3d233f85b3006be359cdd6574d7abd9ce462f2c5334c`
- **Actually executed**: `05_bottleneck_audit.py`, `sha256 = a8ef860e21c93d7132c87f595a0a0f590f3182c585ff4fc2a98ebf3a74bc1f8f`
  — an export of the notebook, run headless with `nohup` because the full pass is 18,756 inferences
  (~2–6 h) and cannot be held open in Jupyter.
- **The two differ in exactly one line** (183 code lines each): `SMOKE = True` → `SMOKE = False`.
  Verified by diff, ignoring blanks/comments. The tracked notebook reproduces the run by flipping
  that boolean.
- The `.py` is **not committed**: §VIII binds — notebooks generate runs, `.py` files are libraries,
  never launchers. **Loose end**: `papermill` is the canonical headless-notebook path and is not yet
  set up. Its own atomic.
- **Hardware**: RTX PRO 4500 Blackwell 32 GB · driver 580.126.09 · `torch 2.8.0+cu128`. No OOM.
- **Branch/commit of the sources**: `task/bottleneck-audit` @ `6a2c69e`.

## 05b — number-probe (done, zero GPU)

**Verdict by the pre-registered rule: COUNTS BADLY** (`acc_non_mode` 0.232 < 0.30, Spearman `r` 0.625 > 0;
the DOES-NOT-COUNT branch fires on neither of its two conditions). → **Test B runs as designed.**

- **Source:** rung 02's `eval_best/predictions.json`, validated by reproducing `acc_OOD = 0.5917` before
  use. All gates green; checksums reproduce rung 05. `n_ambiguous_negation = 0` — the parser's known
  weakness never fired.
- **The model perceives quantity** (predictions rise monotonically with truth) **but its scale saturates
  at ~2.** 81.5% of errors are under-counts. Accuracy by truth: 1→0.803, 2→**0.459**, 3→0.192, 7→0.000.
- 🔴 **It is not about counting — it is about multiplicity.** The identical compression appears in
  `fo_class`, which requires no counting: at truth=2 it names **1.76** classes (vs `number` saying 1.69);
  at truth=3 it names **1.98** (vs 2.01). **Two unrelated output formats saturate at the same place.**
  If the output format were the bottleneck, `fo_class` would be healthy.
- **Refined diagnosis:** the model sees the dominant object well and **fails to perceive the additional
  ones**. That is upstream of the format → **perception** → **strengthens Test B**, and cuts against the
  "format, not eyes" reading that `number`'s +8.0 alone suggested.
- **Caveat kept on record:** the information could be in the encoder while the LLM ignores it; Test B
  would not fix that. Weak counter-evidence: the language-only LoRA had 13.7k examples and never learned
  to read it.
- **Where the money is:** the mass is at truth 1–3 (73.6% of `number`); the 7–12 tail is ~5% and already
  at zero. **The largest single pocket of loss in the project is truth=2: 527 questions at 0.459.**

## Next
- **Test B — LoRA on the ViT** (not fine-tune of the ViT; the A/B of one variable is "the LoRA also
  reaches the vision path"). Verify the `ms-swift` flag semantics by **trainable-parameter count**:
  a few M = LoRA ✅ / hundreds of M = fine-tune ❌ → stop.
- **If OOM: stop. Do not lower `max_pixels`** (second variable) — and suspect fine-tune was configured
  instead of LoRA before blaming the GPU.
- 🔴 **Judge Test B on accuracy at truth=2 and truth=3, in BOTH `number` and `fo_class` — not on overall
  `number` accuracy.** The tail is ~5% and already zero; an overall number would hide the only movement
  that matters. **Pre-registered target: mean prediction at truth=2 rises from 1.69, accuracy from
  0.459, and `fo_class` classes-named from 1.76.**
- Bifurcation: loss unblocks + truth=2 moves → **Capacity** branch. Truth=2 does not move in either
  format → **the capacity branch dies** (32B, resolution) and the lever is data/labels or how the LLM
  reads the visual tokens. **The CoA branch is ruled out by pre-registered data.**
