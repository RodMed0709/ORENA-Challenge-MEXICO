# Experiment 49 — flip-equivariance probe on the shipped checkpoint (`49-flip-equivariance`)

| Notebook | Step | Metric | Verdict |
|---|---:|---:|---|
| `49_flip_audit_heldout8.ipynb` | 1 (audit) | transformable rows | **175 / 1,283 (13.6%)** — done |
| `02_flip_pair_probe.ipynb` | 2 (paired GPU probe) | accuracy delta + literal equivariance | accuracy delta **unreadable** (−0.0076, below `RULES §S4`'s 0.01 floor); literal equivariance **87.2%** (n=78) — done |
| `03_train_flip_probe.ipynb` | 3 (memorised-data probe) | same two reads, on `train.parquet` | **built, not launched** — needs a pod |

## Objective

Does the currently-shipped checkpoint (**rung 42 ep4**, `checkpoint-4848`, submission 03,
0.5809 on the platform) correctly track object LOCATION when the image is horizontally
flipped at inference time — i.e. does its answer to a quadrant/position question mirror the
way the pixels moved, or does it repeat what it would have said regardless? Not a training
rung: an inference-time equivariance test, no GPU spent yet.

This is a sharper test than rung 24's original position-prior probe, which compared accuracy
on naturally typical vs. atypical class→quadrant pairs in the existing eval set. Here the
image itself is flipped, so a model that is truly reading pixels must flip its answer with it.

## Why a new experiment number, not a note inside `24-geometric-aug`

Rung 24 built the flip machinery (`_models/flip_audit.py`, `_models/horizontal_flip.py`) and
tested it as a *training-time* augmentation on rung 21 arm A. This experiment reuses that
machinery **unchanged** but applies it to a different checkpoint lineage (rung 42) with a
different, and critically leak-free, eval set — see below. Different checkpoint under test =
new experiment, per repo convention; the module stays owned by rung 24 and is imported, not
copied.

## Step 1 — the leak-free eval set, and the transformable-row audit

Rung 42's training corpus (`experiments/42-merged-corpus/`) promotes **30 of the public test
set's 38 videos into training**. Testing flip-equivariance on those 30 would partly measure
memorisation, not generalisation. The only leak-free eval set for this checkpoint is its own
declared held-out set: **8 videos / 1,283 questions**
(`experiments/42-merged-corpus/RESULTS_split_42.json`).

**Data provenance, verified before trusting any count:** the local FRAME parquets used here
match rung 08's data card exactly (6,252 test / 13,748 train / 38 test videos / 92 train
videos / 4,486 test frames), and the 8 held-out videos' question count reproduces submission
03's own declared number (1,283) exactly — independent confirmation this is the same
canonical split rung 42 was evaluated against, not just a same-shape coincidence.

**Result:** `flip_audit` (rung 24's classifier, unchanged) restricted to those 1,283 questions
finds **175 transformable rows (13.6%)** — close to rung 24's original 871/6,252 (13.9%) on
the full 38-video set, so restricting to the 8 held-out videos did not disproportionately
starve the transformable pool.

| dataset | rule | n |
|---|---|---:|
| heico | `fixed_quadrant_class` | 66 |
| heico | `object_center_quadrant` | 7 |
| heico | `all_object_positions` | 23 |
| lapchole | `fixed_quadrant_class` | 31 |
| lapchole | `object_center_quadrant` | 31 |
| lapchole | `all_object_positions` | 17 |

Full row-level audit: `RESULTS_flip_audit_heldout8.csv`. Disposition summary:
`RESULTS_flip_audit_heldout8_summary.csv`. Re-executed successfully end-to-end under
`papermill` in a dedicated `conda` env (`data_study`) — real captured outputs, not a
workaround; an earlier attempt in a throwaway sandboxed venv silently dropped kernel
stdout/file-writes, which `data_study` does not hit.

## Step 2 — the paired GPU probe (`02_flip_pair_probe.ipynb`) — RUN, on a pod, 2026-08-21

For the 175 transformable rows: merges rung 42 ep4's adapter (`merge_adapter`, imported
unchanged from `experiments/48-centre-probe/_tools/probe_runner.py`), materializes an
original + horizontally-flipped JPEG per row (`_tools/flip_pair_runner.py`), then answers
BOTH conditions — untouched image/question, and flipped image + the label-correct
question/gold `flip_audit.py` already computed in step 1 — in **one loaded-model session**
(`answer_paired`), to avoid the GPU-swap drift [[archived-results-not-bit-reproducible]]
documents. Scored through the SAME canonical path every other rung is scored through:
`focus.evaluation.Evaluator` with the real LLM judge (not a bespoke comparator) — the
model call is the canonical `frame.engine.QwenFrameEngine`, only the loop and the image
source are this experiment's own, same philosophy as rung 48's `probe_runner.py`.

**Verified locally before launch** (no GPU/SDK needed): notebook JSON validity,
`_tools/flip_pair_runner.py` syntax and its cross-experiment import of `merge_adapter`, the
qID construction/uniqueness and `__flip`-suffix round-trip against the real 175-row audit,
the SMOKE row-selection, and `flip_audit.swap_left_right_quadrants` behavior on realistic
prediction text. Ran build → smoke (6 rows) → full (175 rows) on a pod per repo convention.

### Read 1 — paired accuracy delta: UNREADABLE, not just non-significant

| | accuracy |
|---|---:|
| original frames | 0.9086 |
| flipped frames | 0.9029 |

`frame.metrics.paired_delta_ci` (video-clustered, `n_boot=4000`): **delta −0.0076**, 95% CI
**[−0.107, +0.087]**, n=175 over 8 videos, 5 rows favour the flipped condition, 6 favour the
original. The CI crosses zero, but more to the point: **|−0.0076| is below `RULES §S4`'s
0.01 floor** — this delta is unreadable on magnitude alone, before the CI is even
considered. With only 8 videos backing it, this instrument also has little power: a real
3–5 point accuracy cost from flipping would likely not have been separable from noise
either. **Not the informative read — see below.**

### Read 2 — literal answer equivariance: 87.2%, and the failure split is informative

For the 78 rows whose ANSWER carries a quadrant token (`object_center_quadrant`,
`all_object_positions` — `fixed_quadrant_class`'s answer is a bare class name, so this
check is a no-op there by construction): does
`flip_audit.swap_left_right_quadrants(original_prediction)` literally equal the flipped
prediction?

| rule | n | literal equivariance |
|---|---:|---:|
| `object_center_quadrant` | 38 | **94.7%** |
| `all_object_positions` | 40 | **80.0%** |
| **both** | **78** | **87.2%** |

🟢 **The model's answer moves with the pixels most of the time** — not what a purely
memorised class→quadrant prior would produce.

🔻 **Corrected the same day — the first version of this section said the "multi-object
enumeration task is where equivariance breaks down." Checked, and that overclaims on two
counts.** (1) **Fisher's exact test on the 2×2 (rule × pass/fail) gives p = 0.088** — not
significant at conventional thresholds, and this project's own significance culture
(`RULES §S1`) would not sign that off as an established gap. (2) **The "multi-object"
framing mischaracterises what `all_object_positions` mostly contains**: 33 of its 40 rows
(82.5%) have exactly ONE object in the gold answer — content-difficulty nearly identical to
`object_center_quadrant`, just a different question template and answer format
(`open_ended` free text vs. `multiple_choice`). Splitting the 8 failures by item count —
6 in single-item rows, 2 in the 7 multi-item rows (18.2% vs 28.6%) — n=7 is far too small to
trust that direction either way.

🔑 **Of the 10 failures, only 2 are the "pure prior" pattern** — the prediction literally
unchanged despite the flip (e.g. `heico__2244592`: both conditions answer `1. Clip:
top/left`, expected `top/right`). **The other 8 changed, mostly by drifting in object COUNT
or CLASS, not by getting the left/right axis wrong** — e.g. `heico__2244558` (`"1. Sponge:
bottom/left"` → `"1. Clip: bottom/left 2. Clip: bottom/left"`, a new class AND a duplicated
item) and `lapchole__4669577` (`"1. Clip: top/right"` → three items, all `bottom/*`). One
`object_center_quadrant` failure genuinely does swap the wrong axis (`heico__2595416`:
`bottom/right` → `top/right`, expected `bottom/left`). **Best-supported read:**
`all_object_positions`'s free-text format is noisier under this probe's strict
exact-string check than `object_center_quadrant`'s constrained choice — plausibly a
generation-format effect, not a spatial-reasoning gap — but this run cannot separate that
from "genuinely worse at multi-item cases" with only 7 multi-item rows. Needs more data or
a looser (per-item) match criterion to settle, not asserted here.

⚠️ **Caveats, stated plainly:** n=78 (n=40 for `all_object_positions`, of which only 7 are
genuinely multi-object) over 8 videos is a small instrument, same class of limit as
everything else scored on this held-out set (`RULES §13`). No CI was computed on the
87.2%/94.7%/80.0% point estimates — the Fisher's-exact p-value above is the only formal test
run against them. This is a diagnostic probe, not a leaderboard-comparable number.

🔴 **Does NOT contradict rung 24's shortcut finding** (atypical-quadrant accuracy 8.5pts
below typical, CI excluding zero, measured on the *old* rung-21-arm-A checkpoint,
[[flip-narrows-shortcut-not-a-win]]) — different construct (natural-distribution
typical/atypical gap vs. equivariance-under-flip) on a different checkpoint. Both can be
true at once. Reading them as "the shortcut is gone in rung 42" would need the SAME
construct measured on both checkpoints, which this rung does not do.

Full results: `RESULTS_flip_pair_scored.csv` (per-row, judge-scored),
`RESULTS_flip_pair_paired_ci.csv` (read 1), `RESULTS_flip_pair_equivariance.csv` (read 2,
row-level), `RESULTS_flip_pair_summary.json` (headline numbers).

## Step 3 — does equivariance hold on MEMORISED data? (`03_train_flip_probe.ipynb`) — built, not launched

Step 2 measured the held-out (never-seen-in-any-form) case. This asks the sharper version
of the same question on `train.parquet` — **92 videos, real training supervision** (rung
18's base corpus, before rung 42's promoted-video extras). A row that stays equivariant
under a flip it was never shown, despite being data the model was trained to answer, is
stronger evidence against "it's leaning on memorisation" than the held-out result alone.
Framed as a **ceiling check, not a mechanism claim** — a non-equivariant row here is
ambiguous by construction (reciting the memorised answer vs. simply being wrong the same
way it would be on any hard row); this notebook alone can't tell those apart.

**Scope, agreed before building:** not the full 2,170 transformable train rows — a
~200-row sample, **stratified by VIDEO only**, every one of the 92 videos represented at
least once. Checked first, not assumed: stratifying by `(video, rule)` instead floors at
**261 rows minimum** (92 videos × up to 3 rules, every non-empty cell keeps ≥1) — already
past target. Video-only stratification floors at 92 and lands at 202. This does **not**
force rule-mix proportionality the way `(video, rule)` would; `_tools/video_subsample.py`
reports the drift rather than assuming it away.

**Built on `src/frame/subsample.py`'s existing machinery** (`freeze`/`load`, its manifest +
sha256-sidecar pattern — same discipline every other subsample in this repo uses) with one
new piece, `_tools/video_subsample.py`'s `choose_by_video`, since `subsample.choose`'s own
`(video, answer_format)` key is the thing that floors too high here. Reuses
`_tools/flip_pair_runner.py` (`merge_adapter`, `materialize_flip_pairs`, `answer_paired`)
unchanged from step 2 — same checkpoint, same merged-model cache path on purpose (skips
re-merging a 17GB checkpoint step 2 already produced), same one-session-per-condition
discipline, same canonical `Evaluator` + real-judge scoring path.

**Verified locally before launch** — further than step 2's pre-launch check, since the SDK
(`focus`) turned out to be importable here with `torch`/`transformers` (CPU-only) added to
`data_study`: not just syntax, but the REAL `load_frame_items` → `flip_audit.audit_dataframe`
→ `choose_by_video` → `freeze_manifest` → reload-and-verify pipeline, end to end, against
the actual local parquets (via a symlinked directory matching the pod's nested layout).
**Caught and fixed a real bug this way** — `report_drift`'s first version merged two frames
that both carried an `answer_format` column, which pandas silently resolves to `_x`/`_y`
suffixes instead of raising, so the drift report would have crashed on the pod after the
expensive part (merge + inference + judge-scoring) was already paid for. Fixed to read the
column directly off `transformable` (which already carries it from `flip_audit`'s own
output), no merge needed. Confirmed with real data: **202 rows, 92 of 92 videos**, manifest
round-trips through its sha256, SMOKE-path (8 rows) resolves and cross-checks question text
against the audit. Still not verified: anything past the merge/frame-fetch/inference/judge
boundary — no GPU, no video files, no `torch` with CUDA on this machine.

Read alongside step 2's held-out numbers once scored: 87.2% literal equivariance (n=78),
accuracy delta unreadable — same two metrics, same checkpoint, different population.
