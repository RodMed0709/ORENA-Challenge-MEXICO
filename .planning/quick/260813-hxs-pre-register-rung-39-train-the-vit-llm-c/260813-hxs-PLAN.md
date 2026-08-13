---
phase: quick-260813-hxs
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - experiments/39-connector-lora/PLAN.md
  - experiments/39-connector-lora/README.md
  - context/39-connector-lora/CONTEXT.md
  - experiments/39-connector-lora/_tools/reachability_gate.py
  - experiments/39-connector-lora/_tools/chain.py
  - experiments/39-connector-lora/_models/connector_lora_train.py
  - experiments/39-connector-lora/_models/README.md
  - experiments/39-connector-lora/00_connector_gate.ipynb
  - experiments/39-connector-lora/01_connector_arm.ipynb
autonomous: false
requirements: [QUICK-260813-hxs]
user_setup: []

must_haves:
  truths:
    - "A reader opening experiments/39-connector-lora/PLAN.md finds the single variable, the blocking gate with both legs and their pass criteria, the named control with its exact scored numbers, the declared primary cell, every declared deviation, and what a faithful negative looks like — and finds NO rung-39 number anywhere in the file."
    - "The gate builds an argv in which --target_modules is followed by 8 separate merger names, never one joined string — provable on a laptop, no GPU."
    - "The arm's single-variable claim is proven by a mechanical diff against A2's own argv, using a diff helper that does not drop splatted values."
    - "Rendering the chain produces a bash script that waits for the GPU, runs the gate first and skips the arm if it fails, commits only paths OUTSIDE runs/, pushes to HEAD:main with retries, and reads the RunPod key from a file it deletes."
    - "No .sh file is committed anywhere in the repo, and no top-level tests/ or scripts/ folder is created."
  artifacts:
    - path: "experiments/39-connector-lora/PLAN.md"
      provides: "The pre-registration, written before any number exists"
      contains: "0.4986"
    - path: "experiments/39-connector-lora/README.md"
      provides: "Ladder table as the first content in the file"
      contains: "| Notebook |"
    - path: "context/39-connector-lora/CONTEXT.md"
      provides: "Objective / Setup-config / Decisions / Results / Next"
    - path: "experiments/39-connector-lora/_tools/reachability_gate.py"
      provides: "Two-leg reachability gate with the splat fix and the explicit merger tuple"
      contains: "deepstack_merger_list"
    - path: "experiments/39-connector-lora/_models/connector_lora_train.py"
      provides: "Arm engine reusing rung 21's argv builder, with a multi-value-safe diff"
      contains: "assert_single_variable"
    - path: "experiments/39-connector-lora/_tools/chain.py"
      provides: "Importable renderer that returns the chain bash text; never a committed .sh"
      contains: "HEAD:main"
    - path: "experiments/39-connector-lora/00_connector_gate.ipynb"
      provides: "Runs BOTH gate legs, raises on failure so papermill exits non-zero"
    - path: "experiments/39-connector-lora/01_connector_arm.ipynb"
      provides: "The arm: train -> merge -> eval -> canonical scoring, SMOKE-parameterised"
  key_links:
    - from: "experiments/39-connector-lora/_tools/reachability_gate.py"
      to: "experiments/27-vit-lr-decouple/_tools/gcov_probe.py"
      via: "sys.path insert + from gcov_probe import static_leg"
      pattern: "static_leg"
    - from: "experiments/39-connector-lora/_models/connector_lora_train.py"
      to: "experiments/21-recipe-sweep/_models/recipe_sweep_train.py"
      via: "import of swift_args_21 / RecipeSweepConfig — reused, never copied"
      pattern: "swift_args_21"
    - from: "experiments/39-connector-lora/01_connector_arm.ipynb"
      to: "src/frame/metrics.py"
      via: "stratified_report + class_f1_report + assert_class_f1_reported"
      pattern: "stratified_report"
    - from: "experiments/39-connector-lora/_tools/chain.py"
      to: "experiments/39-connector-lora/01_connector_arm.ipynb"
      via: "papermill command in the rendered script — NOT a python launcher"
      pattern: "papermill"
---

<objective>
Pre-register rung 39 — **train the ViT→LLM connector** — and build the self-closing serial
chain that will run it on the RunPod pod.

Purpose: `[[the-merger-is-unreachable-by-default]]` established that the merger has never
received a gradient in 30+ rungs, and that the cause is a regex rather than a decision. Rung 39
is the rung that acts on it. It exists to be *falsifiable*: the pre-registration must be on disk,
committed, before a single rung-39 number exists — because rung 38 shipped three undeclared
deviations and this plan is the correction.

Output: the pre-registration (`PLAN.md`), the two-leg reachability gate with the splat fix, the
arm engine whose single-variable claim is proven mechanically, the two notebooks, and the chain
renderer. **This plan does NOT run the gate and does NOT run the arm.** It ends with everything
committed and reviewable.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@CONSTITUTION.md
@EXPERIMENT_REPO_STRUCTURE_SPEC.md
@context/RULES.md
@context/decisions/the-merger-is-unreachable-by-default.md
@experiments/32-aligner-unfreeze/_tools/reachability_smoke.py
@experiments/32-aligner-unfreeze/RESULTS_reachability.csv
@experiments/21-recipe-sweep/_models/recipe_sweep_train.py
@experiments/21-recipe-sweep/RESULTS_A2_lr.csv
@experiments/06-vit-lora/_models/vit_lora_train.py
@experiments/27-vit-lr-decouple/_tools/gcov_probe.py
@experiments/30-grpo-number/_models/grpo_train.py
@experiments/21-recipe-sweep/21b_epoch_eval.ipynb
</context>

<interfaces>
<!-- Extracted from the codebase. Use these directly; do not go exploring. -->

**`experiments/27-vit-lr-decouple/_tools/gcov_probe.py:137`**
```python
def static_leg(adapter: Path | None = None) -> dict:
    # returns: adapter, prefixes, n_trainable, n_vit, n_aligner, n_llm, orphans, n_orphans
```
🔴 Hand it the adapter **FILE** (`.../checkpoint-N/adapter_model.safetensors`). A directory
raises `IsADirectoryError` *after* the training is paid for.

**`experiments/21-recipe-sweep/_models/recipe_sweep_train.py`**
```python
def swift_args_21(cfg: RecipeSweepConfig) -> list[str]   # :231  rung 06's argv + max_grad_norm (+ vit_lr)
def _as_map(args: list[str]) -> dict[str, str]           # :251  🔴 SEE BUG BELOW
def diff_vs_control(cfg, arm="A_lr") -> dict[str, tuple] # :263  both sides built by the same builder
ARMS: dict[str, set[str]]                                # :285  declared flags per arm
def assert_single_variable(cfg, arm) -> dict             # :302  RAISES (RULES §7)
def control_cfg(cfg, arm="A_lr") -> RecipeSweepConfig    # :180
def effective_batch(cfg) -> int                          # :136  must stay 16
def main(cfg, stage) -> Path                             # :485
RUNS["A2_lr"] = experiments/21-recipe-sweep/runs/21_lr_1e4_v1   # :172  ⚠️ note the dir name
```

**`experiments/06-vit-lora/_models/vit_lora_train.py`**
```python
def _swift_args(cfg: ViTLoRAConfig) -> list[str]   # :88 — emits, verbatim:
#   --freeze_vit false  --freeze_aligner true  --target_modules all-linear
#   --lora_rank 8 --lora_alpha 32 --lora_dropout 0.1 --lr_scheduler_type cosine
#   --warmup_ratio 0.03 --gradient_checkpointing true --attn_impl sdpa --seed 42
def read_g1(cfg) -> dict     # :280 — parses `model_parameter_info` and `lora_config: target_modules=`
def _train(cfg) -> Path      # :221 — carries the broken-run guard + the log tee
def list_checkpoints(cfg); def merge_checkpoint(cfg, ckpt)
```

**`experiments/30-grpo-number/_models/grpo_train.py`**
```python
def _assert_supervised(path: Path) -> int          # :177  pre-flight, before the GPU
def _assert_learned(cfg, min_nonzero_frac=0.5)     # :228  reads the run's own grad_norm log
```

**`src/frame/`** — `run.run_baseline(cfg)` (:158), `metrics.stratified_report`,
`metrics.class_f1_report`, `metrics.assert_class_f1_reported`, `metrics.assert_no_dup_qid`,
`metrics.assert_ood_from_qid`, `metrics.assert_all_rows_grouped`,
`ledger.gold_from_frame_parquets(DATA_ROOT)`.

---

### 🔴 Two hazards I MEASURED while planning. Both are load-bearing.

**H1 — `_as_map` silently drops 7 of the 8 merger names.** `recipe_sweep_train.py:251-261`
consumes exactly one value per flag. Run on this laptop:
```
_as_map(['--target_modules','m1','m2','m3','--learning_rate','0.0002'])
  ->  {'--target_modules': 'm1', '--learning_rate': '0.0002'}
```
So `diff_vs_control` would compare on `m1` alone and `assert_single_variable` would pass while
7 of the 8 targets went unchecked. The arm engine MUST use its own multi-value-safe mapper.
⚠️ Do **NOT** edit rung 21's `_as_map` in place — it is the diff that proves the single-variable
claim of every rung-21 arm already scored, and rewriting it re-dates all of them. Follow the
precedent rung 21 itself set (`swift_args_21`'s docstring, :231-238): correct it in a local copy.

**H2 — the arm's `--target_modules` must be `all-linear` PLUS the 8 names, not the 8 names alone.**
A2's coverage is 720 tensors = 504 LLM + 216 ViT + 0 aligner. Passing only the merger names would
drop the LLM and the ViT and produce a completely different rung wearing rung 39's name. This is
why the gate needs a coverage criterion (Task 1, question B).
</interfaces>

<pre_registration_facts>
These are settled and must not be re-derived, re-litigated, or contradicted anywhere in the
files this plan produces.

**The 8 target Linear layers** (four merger blocks, not one — DeepStack wires them into the
first 3 LLM layers, so this IS the ViT↔LLM connection):
```
model.visual.merger.linear_fc1
model.visual.merger.linear_fc2
model.visual.deepstack_merger_list.0.linear_fc1
model.visual.deepstack_merger_list.0.linear_fc2
model.visual.deepstack_merger_list.1.linear_fc1
model.visual.deepstack_merger_list.1.linear_fc2
model.visual.deepstack_merger_list.2.linear_fc1
model.visual.deepstack_merger_list.2.linear_fc2
```

**🔴 THE LANDMINE** — `ms-swift pipelines/train/tuner.py:93`:
`if isinstance(args.target_modules, str): return args.target_modules`.
A **string returns early and is silently ignored**. It works today only because
`--target_modules all-linear` parses into the LIST `['all-linear']`. Pass the merger names as one
joined string and training runs happily, exits rc=0, and adapts NOTHING — the eighth silent no-op
in this repo. Documented at `experiments/06-vit-lora/_models/vit_lora_train.py:23-26`.

**The control the arm is read against.** `experiments/21-recipe-sweep/RESULTS_A2_lr.csv`,
run `21_lr_2e4_v1`, arm `A2_lr`, **epoch 1**, `checkpoint-901`:

| cell | value |
|---|---|
| `proxy_leaderboard` | **0.4986389858444832** |
| `bucket_mean` | **0.5592175321379278** |
| `aggregation_ID` | **0.3884816753926701** |
| `object_recognition_ID` | **0.6087962962962963** |
| `margin_OOD` | **0.16425** |

🔴 Do **NOT** use 0.5282 or 0.5486 — those are rung 02's numbers, a different rung and a different
recipe (lr 2e-5, LLM-only); adopting them sets the bar 0.031 too low, and that exact mistake in
rung 38 is why this pre-registration is being written. A2 **epoch 3** = proxy 0.6104 /
bucket_mean 0.6496 — that is the 3-epoch number and is **NOT** the bar for a 1-epoch arm.

**The A2 recipe, pinned:** `lora_rank 8`, `lora_alpha 32`, `lora_dropout 0.1`,
`learning_rate 2e-4`, `max_grad_norm 1.0`, `lr_scheduler_type cosine`, `warmup_ratio 0.03`,
`torch_dtype bfloat16`, `attn_impl sdpa`, `freeze_vit false`, `per_device_train_batch_size 1`,
`gradient_accumulation_steps 16`, `seed 42`. Dataset = rung 18's `train.jsonl`, 14,415 rows,
sha256-pinned by rung 21. 14,415 / 16 = **901 steps/epoch**; our 8B ran ~11.56 s/it ⇒ **~2.9 h**.

**Gate pass criteria (BLOCKING, pre-registered):**
- Subject leg (explicit merger targets, `freeze_aligner=false`): PASS = `n_aligner > 0` AND
  `n_orphans == 0`.
- Control leg (same explicit targets, `freeze_aligner=true`): must return `n_aligner == 0`.
  Without this control a pass could be an artifact of the prefix classifier rather than of the
  change.
- Gate FAILS ⇒ the chain does **not** train. It commits the gate result and stops the pod. That
  is a publishable result — *"the connector is unreachable even when named explicitly"* — for
  about $0.50.

**Guards that are mandatory because of past silent failures:**
- `rc=0` is NOT evidence a run trained. AdamW's decoupled weight decay moves every tensor at zero
  gradient, so a checkpoint diff does not separate a real run from a no-op — only `sum|Δ|` does
  (measured 1.34 vs 252.2). The instrument is the **`grad_norm` log**.
- The free-disk check must **NOT** use `statvfs`/`df` — on RunPod both report the MooseFS cluster
  (~314 TB) while the volume carries an invisible ~640 GB quota, and a run already died mid-merge
  on `Disk quota exceeded`. Measure with `du` against the configured quota, and **degrade to a
  warning** rather than blocking.

**Pod:** `y6h32tbhwhgxxe`. Its GPU is BUSY until the rung-38 eval finishes — the chain must block
until `/workspace/tmp/leo_eval38_full.log` stops advancing / its driver process exits, before
touching the GPU.

**The `ktchain.sh` properties, each bought by a past failure — reproduce all of them:** artifacts
committed **outside `runs/`** (gitignored; two earlier pushes were lost to this); push to
**`HEAD:main`**, never `main` (the pod checkout is detached, and `push main` cost rung 30 its
push); 3 push retries with `fetch` + `rebase`; the API key read from a file that is deleted
afterwards, **never inlined**.
</pre_registration_facts>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 1: Resolve the two open pre-registration questions</name>

  <decision>Two design questions must be answered BEFORE `PLAN.md` is written, because the win
  condition and the gate's pass criteria both depend on them. Neither can be settled after a
  number exists without the pre-registration becoming a retrofit.</decision>

  <context>
**Question A — the 1-epoch arm is NOT schedule-matched to the control, and that is a second
variable.**

The brief pins "one epoch" and "A2 recipe VERBATIM". Those two are in tension, and rung 21's own
engine already documents why (`recipe_sweep_train.py:288-295`, arm `C_epochs`): *"Cosine anneals
over the PLANNED steps … the trajectories differ from step 1 and neither contains the other."*

The arithmetic, on our numbers:

| | A2 control (`--num_train_epochs 3`) | naive arm (`--num_train_epochs 1`) |
|---|---|---|
| planned steps the scheduler is built over | 2703 | 901 |
| warmup steps (`0.03 ×`) | **81** | **27** |
| LR at step 901 | mid-cosine | **exactly 0.0** |

So an arm with `--num_train_epochs 1` differs from `checkpoint-901` in the **warmup length** and
the **entire LR trajectory**, on top of `--target_modules`. Reading it against
`21_lr_2e4_v1/checkpoint-901` would be a two-variable comparison labelled as one — precisely the
rung-38 failure this pre-registration exists to correct.

Three ways out:

| option | argv | GPU cost | schedule-matched? |
|---|---|---|---|
| **A1 (recommended)** | `--num_train_epochs 3`, **terminate the process once `checkpoint-901` is complete** | ~2.9 h — *identical* to the naive 1-epoch run | ✅ exact |
| **A2-full** | `--num_train_epochs 3`, run all 3 epochs, eval ep1/ep2/ep3 | ~8.7 h train + 3× (merge+eval) | ✅ exact, and satisfies RULES §6b for free — three epoch-matched controls already scored |
| **A3 (not recommended)** | `--num_train_epochs 1` | ~2.9 h | ❌ warmup and LR trajectory both differ |

A1 costs the same wall-clock as A3 and removes the confound entirely; the only added machinery is
a supervisor that waits for `checkpoint-901/adapter_model.safetensors` + `adapter_config.json`
to be written and then `SIGTERM`s the trainer. A2-full is the scientifically strongest and buys
two extra epoch-matched reads for roughly 6 more GPU-hours.

**Question B — should the gate also assert that coverage is PRESERVED?**

The pre-registered criteria are `n_aligner > 0` and `n_orphans == 0`. They do not rule out a
"pass" in which ms-swift honours the explicit merger names by *replacing* `all-linear` rather than
extending it — an adapter that reaches the merger and drops the 504 LLM + 216 ViT tensors. That
would be a different rung entirely, and it would pass the gate as written.

Proposed **addition** (strengthens, never weakens): the subject leg must ALSO show
`n_llm == 504` and `n_vit == 216` — A2's own census, already on disk at
`experiments/32-aligner-unfreeze/RESULTS_reachability.csv`. And record the exact `n_aligner`:
8 Linear layers × (LoRA A + B) ⇒ the sharp expectation is **16**. A count that is neither 0 nor 16
is a FINDING to read before the arm launches, not something to accept silently.

⚠️ Note for both branches: A2's argv carries `--freeze_aligner true` (`vit_lora_train.py:99`), so
whether the arm needs to flip that flag is decided BY the gate's control leg — and both outcomes
are pre-declared in Task 2, so neither is a post-hoc choice.
  </context>

  <options>
    <option id="A1-B-yes">
      <name>A1 + coverage criterion (recommended)</name>
      <pros>Same GPU cost as the naive plan; the comparison against `checkpoint-901` is exact; the gate cannot be passed by an adapter that silently dropped the LLM and the ViT.</pros>
      <cons>Needs a small process supervisor in the arm notebook (wait for the checkpoint, then SIGTERM).</cons>
    </option>
    <option id="A2full-B-yes">
      <name>All 3 epochs + coverage criterion</name>
      <pros>Strongest read: three epoch-matched controls already scored, RULES §6b satisfied by construction, no supervisor needed.</pros>
      <cons>~6 extra GPU-hours plus two more merge+eval cycles; the pod bill roughly triples.</cons>
    </option>
    <option id="A3-as-briefed">
      <name>Literal 1 epoch, criteria exactly as briefed</name>
      <pros>Nothing to build beyond the brief.</pros>
      <cons>🔴 Reintroduces an undeclared second variable (warmup 27 vs 81, LR 0.0 vs mid-cosine) into the one rung written to stop undeclared variables. If chosen, the deviation MUST be declared in PLAN.md and the arm read as a curve, not as a delta.</cons>
    </option>
  </options>

  <resume-signal>Select: `A1-B-yes`, `A2full-B-yes`, or `A3-as-briefed` — and state the pod id and the pod-side repo root (`/workspace/repo_rodri` or `/workspace/repo_leo`) the chain should target.</resume-signal>
</task>

<task type="auto">
  <name>Task 2: Write the pre-registration and commit it BEFORE any code exists</name>
  <files>
experiments/39-connector-lora/PLAN.md
experiments/39-connector-lora/README.md
context/39-connector-lora/CONTEXT.md
  </files>
  <action>
Write the text layer first and commit it on its own. Ordering is the point: a pre-registration
committed after the code that produces the number is not a pre-registration.

**All three files in English. No `Co-Authored-By` trailer, ever.**

---

**(1) `experiments/39-connector-lora/PLAN.md` — the pre-registration.** Sections, in order:

- **The question.** *Does putting LoRA on the ViT→LLM connector improve the model?* State that
  across 30+ rungs the merger has never received a gradient, that the cause is structural (both
  ms-swift's `all-linear` and Unsloth's vision regex require an attention/MLP token in the module
  path, applied with `re.fullmatch`, and `model.visual.merger.linear_fc1` carries none), and link
  `context/decisions/the-merger-is-unreachable-by-default.md`. Cite the two measurements:
  ms-swift `720 = 504 LLM + 216 ViT + 0 aligner` on both legs
  (`experiments/32-aligner-unfreeze/RESULTS_reachability.csv`) and Unsloth
  `merger 0 / deepstack 0` (`experiments/38-gen36-ft-screen/RESULTS_smoke_unsloth.json`).
- **The single variable.** `--target_modules` gains the 8 merger names. Reproduce the 8 names
  verbatim from `<pre_registration_facts>`. State H2: the flag value is `all-linear` **plus** the
  8 names — coverage is extended, never replaced.
- **🔴 The landmine and why the gate exists.** Quote `tuner.py:93` and state the consequence: a
  joined string trains happily, exits rc=0 and adapts nothing. State that the names are therefore
  passed as separate argv values and that the gate asserts the produced adapter actually contains
  aligner tensors.
- **The BLOCKING gate.** Both legs, their pass criteria as resolved in Task 1, and the
  consequence of failure: the chain does not train, it commits the gate result and stops the pod,
  and *"the connector is unreachable even when named explicitly"* is itself a publishable result
  for ~$0.50.
- **🔑 The gate's control leg decides the arm's flag count — both branches declared here, in
  advance:**
  - control leg returns `n_aligner == 0` (the expected outcome) ⇒ `freeze_aligner=true` blocks
    explicit targets, so the arm argv is A2 **+ `--target_modules <all-linear + 8 names>`
    + `--freeze_aligner false`**. That is ONE scientific variable (reaching the merger)
    implemented by two flags, and rung 32's `RESULTS_reachability.csv` is the evidence that the
    second flag is **inert on its own** — it changed nothing when the targets were generic.
    `ARMS["E_connector"] = {"--target_modules", "--freeze_aligner"}`.
  - control leg returns `n_aligner > 0` ⇒ explicit targets survive `freeze_aligner=true`, the arm
    keeps A2's `--freeze_aligner true` untouched, and
    `ARMS["E_connector"] = {"--target_modules"}`.
- **The arm.** One epoch on `/workspace/models/qwen3-vl-8b`, A2 recipe verbatim (reproduce the
  pinned table), rung 18's sha256-pinned `train.jsonl`, 901 steps, ~2.9 h at 11.56 s/it. Record
  the schedule decision from Task 1 explicitly, with its arithmetic (warmup 81 vs 27; LR at step
  901).
- **The control, named and numbered.** `21_lr_2e4_v1` arm `A2_lr` **epoch 1**
  `checkpoint-901`, with the five cells reproduced exactly from `<pre_registration_facts>`.
  Include the 🔴 warning against 0.5282 / 0.5486 (rung 02, a different recipe — 0.031 too low)
  and against A2 epoch 3 (0.6104 / 0.6496 — a 3-epoch number).
- **The declared primary cell (RULES §S3).** **`object_recognition_ID`, control 0.6087962962962963**
  — the connector carries visual features into the LLM, so object recognition is the bucket most
  directly downstream of the intervention. State plainly that it may NOT be local `bucket_mean`
  (§S3: that number overstates the judge by +0.12 and inverts the bucket ordering). Report
  `aggregation_ID`, `bucket_mean`, `proxy_leaderboard` and `margin_OOD` beside it as exploratory
  (§S5).
- **The win condition (RULES §S8 + §S1).** A win requires (a) the paired CI on the pre-registered
  cell to exclude zero in the arm's favour, (b) that to hold on **ID and OOD jointly**, never the
  aggregate alone and never one side, and (c) **no cell anywhere** showing significant harm — any
  cell may veto, only the declared cell may grant. A positive point estimate whose CI includes
  zero is a **NULL**. Below |Δ| = 0.01 nothing is readable (§S4); acting on the result needs
  |Δ| ≳ 0.03 (§S1) and that is a team call, not automatic.
- **Declared deviations from A2** — a numbered list, and this list is closed:
  1. `--target_modules` gains `all-linear` + the 8 merger names. **THE variable.**
  2. `--freeze_aligner` — moves only under the branch the gate selects (see above).
  3. the schedule/stop decision from Task 1, stated with its arithmetic.
  4. the gate does **not** pass `--adapters`, unlike rung 32's smoke, which warm-started from
     A2's checkpoint. Reason: a resumed adapter carries its own `target_modules` in
     `adapter_config.json`, so the gate would be measuring rung 21's LoRA geometry instead of
     rung 39's. The gate must construct the LoRA fresh, exactly as the arm will.
  5. `lora_dropout 0.1` is emitted by the gate (rung 32's smoke omitted it) so the gate mirrors
     the arm's recipe.
  6. non-scientific: run directories, output paths, logging.
  State explicitly: *rung 38 shipped three undeclared deviations (framework, `lora_dropout` 0.0
  vs 0.1, LoRA coverage). Anything not on this list is a defect, not a detail.*
- **What counts as a faithful negative.** Two shapes, both published, neither re-rolled (§S7,
  spec §8): (i) the gate fails — the connector is unreachable even when named explicitly;
  (ii) the gate passes, the arm trains with measured gradient on the merger, and the delta is
  null or negative — the connector receives gradient and it does not pay. Record either in the
  ladder as NO-GO with the number.
- **Publication obligations.** RULES §9b — a run that scores `fo_class` may not publish without
  `frame.metrics.class_f1_report`, gated by `assert_class_f1_reported`, and the `per_class` table
  must be read beside the scalar. RULES §1 — score ONLY via `frame.metrics.stratified_report`,
  never re-derived inline.
- **Launch preconditions.** (a) this file committed; (b) an independent read-only review
  returning GO with file:line (CONSTITUTION §VIII.6); (c) the pod GPU free of the rung-38 eval;
  (d) the chain rendered onto the pod, not committed.

🔴 **`PLAN.md` must contain no rung-39 result of any kind.** Every number in it is either the
control's (already scored) or an established fact. A `grep` for a rung-39 metric must come back
empty, and the reviewer should check that.

---

**(2) `experiments/39-connector-lora/README.md`** — **opens with the ladder** (spec §4, the first
thing in the file), with the rungs pending:

| Notebook | Rung | Metric (primary) | Verdict |
|---|---|---|---|
| — (control, rung 21) | `A2_lr` ep1 | `object_recognition_ID` 0.6088 | baseline |
| `00_connector_gate.ipynb` | 39-gate | `n_aligner` | pending |
| `01_connector_arm.ipynb` | 39 | `object_recognition_ID` | pending |

Then: the one variable, the pointer to `PLAN.md` as the pre-registration, the pointer to
`context/39-connector-lora/CONTEXT.md`, and a short "how to run" that names **papermill** and the
chain — never a `.sh` and never a hand-run `.py`.

---

**(3) `context/39-connector-lora/CONTEXT.md`** — the curated context, mandated by spec §2 and
CONSTITUTION §VIII.3, five headings and nothing else: **Objective** (1-2 lines) /
**Setup-config** (split, base model, the pinned A2 recipe, the ONE variable) / **Decisions**
(the Task 1 resolution, the no-`--adapters` choice, where the chain lives) / **Results**
(`pending — pre-registered <date>`) / **Next** (run the gate).

---

**Commit this task alone**, before Task 3 writes any code:
`docs(39): pre-register the connector-LoRA rung before any number exists`
  </action>
  <verify>
    <automated>cd "$REPO" &amp;&amp; python -c "
import pathlib,re,sys
p=pathlib.Path('experiments/39-connector-lora/PLAN.md'); t=p.read_text(encoding='utf-8')
need=['0.4986389858444832','0.6087962962962963','0.5592175321379278','tuner.py:93',
      'deepstack_merger_list.2.linear_fc2','object_recognition_ID','faithful negative',
      'freeze_aligner','all-linear']
miss=[n for n in need if n not in t]
assert not miss, f'PLAN.md missing: {miss}'
assert '0.5282' not in t and '0.5486' not in t, 'rung-02 numbers must not appear as the bar'
assert t.count('linear_fc')&gt;=8, 'all 8 merger names must be reproduced verbatim'
r=pathlib.Path('experiments/39-connector-lora/README.md').read_text(encoding='utf-8')
head=[l for l in r.splitlines() if l.strip()][:8]
assert any(l.lstrip().startswith('|') for l in head), 'README must OPEN with the ladder'
c=pathlib.Path('context/39-connector-lora/CONTEXT.md').read_text(encoding='utf-8')
for h in ('Objective','Setup-config','Decisions','Results','Next'):
    assert h in c, f'CONTEXT.md missing heading {h}'
assert not pathlib.Path('tests').exists() and not pathlib.Path('scripts').exists()
print('OK task2')
"</automated>
  </verify>
  <done>The three text files exist, `PLAN.md` carries the control's exact scored numbers and no rung-39 number, `README.md` opens with the ladder, `CONTEXT.md` has all five headings, and the commit landed with no `Co-Authored-By` trailer.</done>
</task>

<task type="auto">
  <name>Task 3: Build the gate, the arm engine, the two notebooks, and the chain renderer</name>
  <files>
experiments/39-connector-lora/_tools/reachability_gate.py
experiments/39-connector-lora/_models/connector_lora_train.py
experiments/39-connector-lora/_models/README.md
experiments/39-connector-lora/_tools/chain.py
experiments/39-connector-lora/00_connector_gate.ipynb
experiments/39-connector-lora/01_connector_arm.ipynb
  </files>
  <action>
Build in this order — the gate first, because it is what runs first and it is what makes the arm
legitimate. Every `.py` here is an **importable library**; none of them is ever run by hand
(CONSTITUTION §VIII.1).

---

**(3a) `_tools/reachability_gate.py`** — adapted from
`experiments/32-aligner-unfreeze/_tools/reachability_smoke.py`. Copy that file and change exactly
these things; **everything else is hard-won and must not drift**:

*Keep verbatim, and keep the comments explaining why:* `swift: Path("/workspace/envs/infer/bin/swift")`
as an **ABSOLUTE** path (a papermill kernel does not inherit the activated venv);
`--tuner_type`, not `--train_type` (renamed on ms-swift 4.4.1); `static_leg` handed the adapter
**FILE**, not its directory; the `PATH` prepend of `cfg.swift.parent`; `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; `argv.txt` written before the subprocess runs.

*Change:*
1. `target_modules: tuple[str, ...]` defaulting to `("all-linear",) + MERGER_TARGETS`, where
   `MERGER_TARGETS` is the module-level tuple of the 8 names.
2. 🔴 **The splat fix.** `build_argv` emits `["--target_modules", *cfg.target_modules]` — never
   `cfg.target_modules` as one value, never `" ".join(...)`, never a comma-join. Put the
   `tuner.py:93` quote in the docstring right above it. Add an in-module assertion helper
   `assert_splat(argv)` that RAISES unless the argv contains exactly one `--target_modules` token
   followed by `len(cfg.target_modules)` consecutive values none of which starts with `--`.
   `build_argv` calls it before returning.
3. **Drop `--adapters`.** The gate constructs the LoRA fresh, exactly as the arm will
   (declared deviation #4). Delete the `A2_CKPT` constant and the `adapters` config field.
4. Add `lora_dropout: float = 0.1` and emit `--lora_dropout` (declared deviation #5).
5. `freeze_aligner` stays THE leg selector: `False` = subject, `True` = control.
6. **Coverage + `adapter_config.json` assertions**, per the Task 1 resolution. After
   `static_leg`, also read the produced `checkpoint-*/adapter_config.json` and assert its
   `target_modules` list contains all 8 merger names — a direct check independent of the prefix
   classifier. Record `n_llm`, `n_vit`, `n_aligner`, `n_orphans` and the expected
   `n_aligner == 16` (8 layers × LoRA A+B) as a `sharp_expectation_met` boolean; a count that is
   neither 0 nor 16 is a FINDING to be read, not silently accepted.
7. `run_both(cfg) -> dict` — runs the subject leg then the control leg, applies the
   pre-registered criteria to each, writes `experiments/39-connector-lora/RESULTS_reachability39.csv`
   (**at the experiment root, OUTSIDE `runs/`** — that is what the chain commits) with one row
   per leg mirroring rung 32's column shape, and **RAISES `AssertionError`** if either leg fails.
   Raising is the mechanism: papermill turns it into a non-zero exit the chain reads.

⚠️ Keep rung 32's honesty about cost: this is **not** zero-GPU. It runs 5 real optimiser steps of
ms-swift on 32 rows, per leg. Minutes, not seconds. Say so in the docstring.

---

**(3b) `_models/connector_lora_train.py`** — the arm engine. Reuse, never copy
(the rung 06 → rung 21 precedent):

```python
from recipe_sweep_train import (RecipeSweepConfig, swift_args_21, control_cfg,
                                effective_batch, main as _main_21)
```

- `@dataclass ConnectorConfig(RecipeSweepConfig)` adding `target_modules: tuple[str, ...]` and
  `freeze_aligner: bool`, defaulted from the Task 1 branch.
- `swift_args_39(cfg)`: take `list(swift_args_21(cfg))` and **replace** the single
  `all-linear` value after `--target_modules` with the splat, and set the `--freeze_aligner`
  value if the selected branch moves it. Do not rewrite rung 21's or rung 06's builders — the
  same reason rung 21 refused to edit rung 06's (`recipe_sweep_train.py:231-238`).
- 🔴 **`_as_map_multi(args) -> dict[str, tuple[str, ...]]`** — a local, multi-value-safe mapper.
  Rung 21's `_as_map` (`:251`) keeps only the first value after a flag; measured on this
  laptop it turns `['--target_modules','m1','m2','m3']` into `{'--target_modules': 'm1'}`,
  which would let 7 of the 8 targets go unchecked while `assert_single_variable` reported a clean
  single-variable diff (hazard H1). Cite that line number in the docstring.
- `diff_vs_a2(cfg)` — builds BOTH sides with `swift_args_39` off `control_cfg(..., "A2_lr")` and
  off `cfg`, compares with `_as_map_multi`, returns every differing flag with **full** value
  tuples.
- `ARMS = {"E_connector": {...}}` — the set fixed by the Task 1 branch.
- `assert_single_variable_39(cfg)` — RAISES on any unexpected flag, on any declared flag that did
  not actually change, on `alpha/rank` movement, and on `effective_batch(cfg) != 16`. Same shape
  as `recipe_sweep_train.assert_single_variable` (`:302`).
- **Guards (mandatory, §9 of the facts):**
  - `assert_supervised(path)` — pre-flight on the dataset, before the GPU. Reuse the idea from
    `experiments/30-grpo-number/_models/grpo_train.py:177`.
  - `assert_learned(cfg, min_nonzero_frac=0.5)` — post-run, reads the run's own `grad_norm` log
    (`grpo_train.py:228`). Docstring must say why a checkpoint diff is not an acceptable
    substitute: AdamW's decoupled weight decay moves every tensor at zero gradient, so only
    `sum|Δ|` separates them (1.34 vs 252.2 measured), and `rc=0` is not evidence.
  - `assert_merger_reached(cfg)` — post-run, two independent readings: `read_g1(cfg)`
    (`vit_lora_train.py:280`) must show `merger` inside the log's `target_modules` string, and
    `gcov_probe.static_leg` on the **real** produced checkpoint must show `n_aligner > 0`. The
    gate's smoke result does not license the arm's result.
  - `warn_disk(quota_gb=640, floor_gb=120)` — 🔴 uses `du`, never `statvfs`/`df` (both report the
    MooseFS cluster at ~314 TB while the volume carries an invisible quota; a run already died
    mid-merge on `Disk quota exceeded`). **Warns, never blocks.**
- `main(cfg, stage)` — `stage ∈ {"train", "merge"}`, delegating to rung 21's `main` where the
  behaviour is unchanged.
- If Task 1 selected **A1**, add `wait_for_checkpoint_then_stop(proc, ckpt_dir, step=901)`: poll
  for `checkpoint-901/adapter_model.safetensors` **and** `adapter_config.json`, require the file
  size to be stable across two polls, then `SIGTERM`. Docstring states why the process is killed
  rather than the schedule shortened: shortening it changes warmup (81 → 27) and the LR
  trajectory, which is the second variable this arm exists to avoid.

**`_models/README.md`** — the engine ↔ rung ↔ notebook ↔ run index (spec §6), one row per engine,
plus a `DO NOT rename` note if anything outside the folder ends up importing it.

---

**(3c) `_tools/chain.py`** — the chain renderer. **This is where the chain legitimately lives,
and the reason must be written into its docstring:**

> Committing a `.sh` at the repo root is forbidden (CLAUDE.md, CONSTITUTION §VIII.1, §IX.3), and
> `.py` files are libraries, never launchers. So the chain is **not a committed file**: it is
> *rendered text*. `render(cfg) -> str` is an importable, folder-private library function; a
> notebook cell calls `write(cfg, Path("/workspace/tmp/rung39_chain.sh"))` and prints the
> `nohup bash …` line for the human. The rendered script lives in `/workspace/tmp/` — pod
> scratch, outside the repo, deleted when done (CONSTITUTION §IX.1-2). What is committed is the
> renderer, which is reproducible, reviewable and diffable; the artifact it produces is not.

The rendered script, reproducing every `ktchain.sh` property:

```bash
#!/bin/bash
exec >> /workspace/tmp/rung39_chain.log 2>&1
echo "===== chain start $(date -u) ====="
source /workspace/envs/infer/bin/activate
cd <REPO_ROOT>                      # from cfg — /workspace/repo_rodri or /workspace/repo_leo
# --- 0. WAIT for the GPU: rung 38's eval owns it -------------------------------
#     poll leo_eval38_full.log's mtime AND nvidia-smi compute-apps; bounded max wait;
#     abort loudly rather than sharing the card.
# --- 1. GATE (blocking) ---------------------------------------------------------
papermill 00_connector_gate.ipynb runs/<tag>/gate.ipynb -k <kernel> --log-output ; GATE=$?
if [ $GATE -ne 0 ]; then echo "GATE FAILED -> no training"; <commit+push+stop>; exit 1; fi
# --- 2. ARM smoke, then ARM full (build -> smoke -> full, CONSTITUTION §VIII.6) --
papermill 01_connector_arm.ipynb runs/<tag>/arm_smoke.ipynb -p SMOKE True  -k <kernel> || { <commit+push+stop>; exit 1; }
papermill 01_connector_arm.ipynb runs/<tag>/arm_full.ipynb  -p SMOKE False -k <kernel> ; echo "rc=$?"
# --- 3. commit ONLY paths outside runs/ (runs/ is gitignored — two pushes died here)
git config user.name RodMed0709
git config user.email medrod2010@hotmail.com
for f in experiments/39-connector-lora/RESULTS_*.csv experiments/39-connector-lora/RESULTS_*.json; do
  [ -f "$f" ] && git add -f "$f"; done
git commit -q -m "results(39): <message>"
# --- 4. push to HEAD:main — the pod checkout is detached; 'push main' cost rung 30 its push
for i in 1 2 3; do git fetch -q origin && git rebase -q origin/main && git push -q origin HEAD:main \
  && { echo 'push OK'; break; }; echo "retry $i"; sleep 30; done
# --- 5. stop the pod; the key is read from a file and the file is deleted --------
K=$(cat /workspace/tmp/.rung39key)
curl -s -X POST -H "Authorization: Bearer $K" https://rest.runpod.io/v1/pods/<POD_ID>/stop
rm -f /workspace/tmp/.rung39key
echo "===== end $(date -u) ====="
```

🔑 **The one deliberate deviation from `ktchain.sh`, and it must be stated in the docstring:**
`ktchain.sh` ran `python -u /workspace/tmp/<driver>.py full`. That is a launcher, which
CONSTITUTION §VIII.1 forbids. The sanctioned headless path is **papermill executing the notebook
itself** (spec §5b) — a *command*, not a launcher file — and it writes an output notebook that IS
the provenance record, with `metadata.papermill.parameters`, per-cell status, and a non-zero exit
plus `exception: True` on failure. That non-zero exit is exactly the mechanism the gate uses to
stop the chain.

`cfg` fields: `repo_root`, `pod_id`, `kernel`, `run_tag`, `key_file`, `log`, `wait_log`,
`max_wait_min`, `commit_message`. **The API key never appears in the rendered text** — only the
path of the file it is read from.

---

**(3d) `00_connector_gate.ipynb`** — spec §5 cell order: markdown title (the rung, the ONE
variable, the named baseline, and a 🔴 *"this is NOT the rung"* banner) → marker-anchored
bootstrap → sibling-rungs markdown → **`parameters`-tagged cell holding plain assignments only**
(`MAX_STEPS = 5`, `ROWS = 32`, `REPO_ROOT`, `POD_ID`) → a cell that imports
`reachability_gate` and calls `run_both(cfg)` (both legs in ONE notebook, so the chain needs a
single papermill invocation and a single rc — a declared deviation from rung 32's one-leg-per-run
notebook) → a cell that prints the census table and re-asserts the pre-registered criteria →
markdown result. Then the **launch cell**: `import chain; chain.write(cfg, ...)`, printing the
`nohup bash /workspace/tmp/rung39_chain.sh &` line — the human runs the command, the notebook does
not shell out.

**`01_connector_arm.ipynb`** — same skeleton, modelled cell-for-cell on
`experiments/21-recipe-sweep/21b_epoch_eval.ipynb`, which already solves every trap on this path:
the `PATH` prepend for `merge_checkpoint`'s bare `swift`; `HF_HOME` set **before** any HF import;
the parameters cell holding **raw literals only** with all derived values **below** it (the
rung-16 papermill trap); the DATA_ROOT resolved by *looking* and failing loudly; the **judge gate**
hoisted to the front (`AutoTokenizer.from_pretrained(judge)`) so a missing judge cache fails in
seconds instead of 45 minutes in; the merge in a `try/finally` that reclaims ~17 GB.

Cells: bootstrap → parameters (`SMOKE`, `RUN`, `DATA_ROOT`, `CONTROL_RUN`, `KEEP_MERGED`) →
derived → `assert_single_variable_39` + `assert_supervised` + `warn_disk` (**all before the GPU**)
→ train (+ the A1 supervisor if selected) → `assert_learned` + `assert_merger_reached` → merge →
`run_baseline` → canonical scoring: `ledger.gold_from_frame_parquets`, `metrics.assert_no_dup_qid`,
`metrics.assert_ood_from_qid`, `metrics.assert_all_rows_grouped`, `metrics.stratified_report`,
`metrics.class_f1_report` + `metrics.assert_class_f1_reported` (RULES §9b) → write
`RESULTS.csv`, `RESULTS_class_f1_*.csv`, `RESULTS_paired_ci_*.csv`, `RESULTS_gcov_arm.json`
**at the experiment root, outside `runs/`** → markdown result naming the declared primary cell and
its control value.

🔴 Score **only** via `frame.metrics` (RULES §1). Never re-derive a bucket, a floor or an accuracy
inline. Assert `answer_postprocess is None`, `n_samples == 1`, `enhance is None`,
`aux_view is None` — the inference path must stay rung 06's exactly, or the delta stops being
attributable.

---

**Cleanup (CONSTITUTION §IX).** Any scratch used while building goes to the session scratchpad,
not the repo, and is deleted. No `smoke_*.py`, no `run_*.py`, no `.sh` anywhere in the repo.

Commit: `feat(39): reachability gate with the splat fix, arm engine, and the chain renderer`
  </action>
  <verify>
    <automated>cd "$REPO" &amp;&amp; python -c "
import json,pathlib,py_compile,sys
E='experiments/39-connector-lora'
sys.path[:0]=['src',f'{E}/_models',f'{E}/_tools','experiments/21-recipe-sweep/_models',
              'experiments/18-count-aug/_models','experiments/06-vit-lora/_models',
              'experiments/02-lora-sft/_models','experiments/27-vit-lr-decouple/_tools']
for f in pathlib.Path(E).rglob('*.py'): py_compile.compile(str(f), doraise=True)

import reachability_gate as G
MT=G.MERGER_TARGETS
assert len(MT)==8 and all(m.startswith('model.visual.') for m in MT), MT
cfg=G.Config()
assert 'all-linear' in cfg.target_modules and all(m in cfg.target_modules for m in MT)
argv=G.build_argv(cfg, pathlib.Path('x.jsonl'))
i=argv.index('--target_modules')
vals=[]; j=i+1
while j&lt;len(argv) and not argv[j].startswith('--'): vals.append(argv[j]); j+=1
assert len(vals)==len(cfg.target_modules)==9, f'SPLAT BROKEN: {vals}'
assert all(m in vals for m in MT), 'merger names missing from argv'
assert argv.count('--target_modules')==1
assert '--adapters' not in argv, 'gate must not warm-start from an adapter'
assert '--lora_dropout' in argv and '--tuner_type' in argv and '--train_type' not in argv
assert str(G.Config().swift).startswith('/workspace/'), 'swift must be an absolute path'

import connector_lora_train as C
m=C._as_map_multi(['--target_modules','m1','m2','m3','--learning_rate','0.0002'])
assert tuple(m['--target_modules'])==('m1','m2','m3'), f'multi-value mapper drops values: {m}'
assert set(C.ARMS)=={'E_connector'}
d=C.diff_vs_a2(C.ConnectorConfig())
assert set(d)&lt;=C.ARMS['E_connector'] and '--target_modules' in d, f'diff not single-variable: {d}'

import chain
txt=chain.render(chain.ChainConfig(repo_root='/workspace/repo_rodri', pod_id='y6h32tbhwhgxxe'))
for tok in ['HEAD:main','papermill','00_connector_gate.ipynb','01_connector_arm.ipynb',
            'rest.runpod.io','rm -f','git fetch','rebase','leo_eval38_full.log']:
    assert tok in txt, f'chain missing {tok}'
assert 'runs/' not in txt.split('git add')[1].split('git commit')[0], 'chain must not commit runs/'
assert 'Bearer \$K' in txt or 'Bearer \$' in txt, 'key must be read from a var, not inlined'
assert 'python -u' not in txt and 'driver.py' not in txt, 'no launcher — papermill only'

for nb in ('00_connector_gate.ipynb','01_connector_arm.ipynb'):
    d=json.load(open(f'{E}/{nb}',encoding='utf-8'))
    assert any('parameters' in c.get('metadata',{}).get('tags',[]) for c in d['cells']), nb
assert not list(pathlib.Path('.').glob('*.sh')) and not list(pathlib.Path(E).rglob('*.sh'))
assert not pathlib.Path('tests').exists() and not pathlib.Path('scripts').exists()
print('OK task3')
"</automated>
  </verify>
  <done>All six files exist and compile; the gate's argv carries `--target_modules` followed by 9 separate values including all 8 merger names and carries no `--adapters`; the arm engine's multi-value mapper preserves splatted values and its diff against A2 is exactly the declared flag set; `chain.render` produces a script that waits on the rung-38 log, gates on papermill's exit code, commits nothing under `runs/`, pushes to `HEAD:main` with retries, reads the key from a file it deletes, and contains no python launcher; both notebooks carry a `parameters`-tagged cell; no `.sh` and no top-level `tests/`/`scripts/` exist.</done>
</task>

</tasks>

<verification>
1. `python -c "import py_compile,pathlib; [py_compile.compile(str(f),doraise=True) for f in pathlib.Path('experiments/39-connector-lora').rglob('*.py')]"` — every `.py` compiles.
2. Both notebooks parse as JSON and each carries exactly one `parameters`-tagged cell holding
   plain assignments only.
3. `git log --format=%B -3` contains **no** `Co-Authored-By` trailer.
4. `git status --short` is clean; `git ls-files '*.sh'` is empty; `tests/` and `scripts/` do not
   exist at the repo root.
5. `PLAN.md` was committed in an **earlier** commit than the code — `git log --oneline --
   experiments/39-connector-lora/PLAN.md` and `... -- experiments/39-connector-lora/_tools/`
   must show the pre-registration first. This is the whole point of the ordering.
6. `grep -rn "0.5282\|0.5486" experiments/39-connector-lora/` returns nothing.
7. Nothing under `experiments/39-connector-lora/runs/` is tracked.
</verification>

<success_criteria>
- The pre-registration exists on disk and in git **before** any rung-39 code, and contains zero
  rung-39 numbers.
- The single variable is `--target_modules` gaining `all-linear` + the 8 merger names, with the
  `--freeze_aligner` branch pre-declared for both possible gate outcomes.
- The landmine is defused **and proven defused without a GPU**: `--target_modules` is followed by
  9 separate argv values, asserted mechanically.
- The arm's single-variable claim is proven by a diff that does not drop splatted values (H1).
- The chain is a rendered artifact in pod scratch, not a committed `.sh`, and reproduces every
  `ktchain.sh` property plus the papermill substitution for the forbidden python launcher.
- Guards are wired: `assert_supervised` pre-flight, `assert_learned` on `grad_norm`,
  `assert_merger_reached` on the real checkpoint, `warn_disk` via `du` (warn, never block).
- Repo discipline holds: English only, no Claude contributor trailer, no `.sh`, no top-level
  `tests/`/`scripts/`, artifacts outside `runs/`, `_models/` engines only, README opens with the
  ladder, `context/39-connector-lora/CONTEXT.md` present.
- **The gate has not been run and the arm has not been run.** The next move — an independent
  read-only review returning GO with file:line, then launching the chain — is out of scope here.
</success_criteria>

<output>
After completion, create
`.planning/quick/260813-hxs-pre-register-rung-39-train-the-vit-llm-c/260813-hxs-SUMMARY.md`
recording: the Task 1 decision as taken, the files created, the two hazards (H1 `_as_map`, H2
coverage) and how each was handled, and the exact launch preconditions still outstanding.
</output>
