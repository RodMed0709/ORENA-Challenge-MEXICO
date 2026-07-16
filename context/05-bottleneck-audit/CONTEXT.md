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

## Next
- **Test B — LoRA on the ViT** (not fine-tune of the ViT; the A/B of one variable is "the LoRA also
  reaches the vision path"). Verify the `ms-swift` flag semantics by **trainable-parameter count**:
  a few M = LoRA ✅ / hundreds of M = fine-tune ❌ → stop.
- **If OOM: stop. Do not lower `max_pixels`** (second variable) — and suspect fine-tune was configured
  instead of LoRA before blaming the GPU.
- Bifurcation: loss unblocks → **Capacity** branch. Loss does not move → the roadmap is re-planned;
  the problem is data/labels. **The CoA branch is ruled out by pre-registered data.**
