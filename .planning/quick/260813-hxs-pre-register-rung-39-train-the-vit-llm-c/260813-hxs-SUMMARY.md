---
phase: quick-260813-hxs
plan: 01
subsystem: experiments/39-connector-lora
tags: [pre-registration, lora, connector, ms-swift, reachability-gate, papermill]
requires:
  - experiments/21-recipe-sweep/_models/recipe_sweep_train.py
  - experiments/06-vit-lora/_models/vit_lora_train.py
  - experiments/27-vit-lr-decouple/_tools/gcov_probe.py
  - src/frame/metrics.py
provides:
  - experiments/39-connector-lora/PLAN.md
  - experiments/39-connector-lora/_tools/reachability_gate.py
  - experiments/39-connector-lora/_models/connector_lora_train.py
  - experiments/39-connector-lora/_tools/chain.py
affects: []
tech-stack:
  added: []
  patterns: [papermill-headless, rendered-chain-not-committed, splat-argv, single-variable-diff]
key-files:
  created:
    - experiments/39-connector-lora/PLAN.md
    - experiments/39-connector-lora/README.md
    - context/39-connector-lora/CONTEXT.md
    - experiments/39-connector-lora/_tools/reachability_gate.py
    - experiments/39-connector-lora/_tools/chain.py
    - experiments/39-connector-lora/_models/connector_lora_train.py
    - experiments/39-connector-lora/_models/README.md
    - experiments/39-connector-lora/00_connector_gate.ipynb
    - experiments/39-connector-lora/01_connector_arm.ipynb
  modified: []
decisions:
  - "Schedule = A1: --num_train_epochs 3, trainer terminated at checkpoint-901"
  - "Targets = all-linear PLUS the 8 merger names, with a coverage criterion on the gate"
  - "The control is D_clip's baseline, not BASELINES['A2_lr'] (that entry is arm A at lr 1e-4)"
metrics:
  duration: ~85 min
  completed: 2026-08-13
---

# Quick task 260813-hxs: Pre-register rung 39 — train the ViT→LLM connector — Summary

Rung 39 is pre-registered on disk and in git **before any rung-39 number exists**, with a blocking
two-leg reachability gate whose splat fix is proven without a GPU, an arm engine whose
single-variable claim is proven by a diff that does not drop splatted values, and a serial chain that
is rendered into pod scratch rather than committed as a `.sh`.

**The gate has not been run and the arm has not been run.** Nothing was launched on the pod; the pod
is stopped and stays stopped. Nothing was pushed to origin.

## Task 1 — the two decisions, as taken

Both were resolved by the project owner and recorded in the pre-registration as decided, with their
reasoning, rather than being asked again.

**Decision 1 — the schedule. Option A1: `--num_train_epochs 3`, terminate once `checkpoint-901` is
complete.** A naive `--num_train_epochs 1` arm builds its cosine over 901 planned steps instead of
2703 — warmup `0.03 × 901 = 27` steps against the control's 81, and LR exactly 0.0 at step 901
against the control's mid-descent. That is a second, undeclared variable on top of `--target_modules`
in the one rung written to end undeclared variables. The mechanism is already documented in the repo
at `experiments/21-recipe-sweep/_models/recipe_sweep_train.py`, in the `ARMS` comment on `C_epochs`
(*"Cosine anneals over the PLANNED steps … the trajectories differ from step 1 and neither contains
the other"*) — cited in `PLAN.md` §6a. A1 costs the same ~2.9 h and removes the confound.
Implemented as `connector_lora_train.train_to_step`.

**Decision 2 — the targets. `all-linear` PLUS the 8 explicit merger names, as separate argv values.**
A2's coverage is 720 tensors = 504 LLM + 216 ViT + 0 aligner; passing only the 8 names would collapse
the adapter to the merger and drop both other legs. The single variable is that the adapter **gains**
the connector, everything else held. The gate therefore carries an explicit coverage criterion
alongside `n_aligner > 0` / `n_orphans == 0`: `n_llm == 504` **and** `n_vit == 216` (A2's own census),
so a pass cannot be satisfied by an adapter that reached the merger and lost everything else.

## Files created

| path | role |
|---|---|
| `experiments/39-connector-lora/PLAN.md` | the pre-registration — 13 sections, zero rung-39 numbers |
| `experiments/39-connector-lora/README.md` | opens with the ladder |
| `context/39-connector-lora/CONTEXT.md` | Objective / Setup-config / Decisions / Results / Next |
| `experiments/39-connector-lora/_tools/reachability_gate.py` | two-leg gate, splat fix, coverage criterion, `adapter_config.json` cross-check |
| `experiments/39-connector-lora/_models/connector_lora_train.py` | arm engine, multi-value-safe diff, four guards, the A1 supervisor |
| `experiments/39-connector-lora/_models/README.md` | engine ↔ rung ↔ notebook ↔ run index |
| `experiments/39-connector-lora/_tools/chain.py` | renders the chain into `/workspace/tmp` — never a committed `.sh` |
| `experiments/39-connector-lora/00_connector_gate.ipynb` | both gate legs, one exit code |
| `experiments/39-connector-lora/01_connector_arm.ipynb` | train → merge → eval → canonical scoring |

**Commits** (ordering is the point — the pre-registration lands first):

| commit | what |
|---|---|
| `21b0919` | `docs(39): pre-register the connector-LoRA rung before any number exists` |
| `c798849` | `feat(39): reachability gate with the splat fix, arm engine, and the chain renderer` |

`git log -- experiments/39-connector-lora/PLAN.md` → `21b0919`; `git log -- .../\_tools/` → `c798849`.
The pre-registration is the parent commit of the code that will produce the number.

## The two hazards, and how each was handled

### H1 — `_as_map` silently drops splatted values. VERIFIED MECHANICALLY ON THIS LAPTOP.

Run in this session, no GPU, against the committed code:

```
INPUT  ['--target_modules', 'm1', 'm2', 'm3', '--learning_rate', '0.0002']
OUTPUT {'--target_modules': 'm1', '--learning_rate': '0.0002'}
SOURCE experiments/21-recipe-sweep/_models/recipe_sweep_train.py:251
```

Seven of nine values are discarded. Under that mapper, `diff_vs_control` would compare on
`all-linear` alone and `assert_single_variable` would report a clean single-variable diff while
**all eight merger names — the variable itself — went unchecked**.

**Handled by writing a local corrected copy, never by editing rung 21.** `recipe_sweep_train.py` is
untouched (`git status` clean, and it is not in either commit): its `_as_map` is the diff behind every
already-scored rung-21 arm, and rewriting it would re-date all of them.
`connector_lora_train._as_map_multi` returns full value tuples, and its docstring cites `:251` and the
measurement above. This follows the precedent rung 21 itself set for rung 06's `_swift_args`
(`recipe_sweep_train.py:231-238`). Verified:

```
_as_map_multi(['--target_modules','m1','m2','m3','--learning_rate','0.0002'])
  ->  {'--target_modules': ('m1','m2','m3'), '--learning_rate': ('0.0002',)}
```

and the arm's diff against A2 comes back as exactly two flags with full tuples:

```
--target_modules  : ('all-linear',) -> ('all-linear', <8 merger names>)
--freeze_aligner  : ('true',)       -> ('false',)
```

### H2 — the targets must EXTEND `all-linear`, not replace it. Guarded in three places.

1. `connector_lora_train._assert_targets_extend` — RAISES if `all-linear` is absent or any of the 8
   names is missing. Tested: passing `MERGER_TARGETS` alone raises.
2. The gate's runtime coverage criterion — `n_llm == 504` **and** `n_vit == 216` on **both** legs, so
   an adapter that reached the merger and lost the other 720 tensors fails.
3. `PLAN.md` §2 and §4a state it in prose, so a reviewer can check the code against the claim.

⚠️ One subtlety found and fixed during the build: `_assert_targets_extend` initially ran inside
`swift_args_39`, which also builds the **control's** argv — and the control legitimately carries
`("all-linear",)`. That made `diff_vs_a2` raise. H2 belongs to the arm, so it moved into
`assert_single_variable_39`; the builder keeps only the shape assertion, which is valid on both sides.

### A third hole, found by testing the fix

`assert_splat` originally checked only the **count** of values after `--target_modules`. A config
holding `("a,b,c,…",)` declares one target and passes one value — the arithmetic agrees with itself
while the value is exactly the joined string `ms-swift tuner.py:93` discards in silence. `assert_splat`
now also rejects any value containing `,`, whitespace, `[`, `]`, `'` or `"`. Verified: a comma-join, a
space-join and a stringified Python list are all refused, on both the gate and the arm; the control's
single legitimate `all-linear` value still passes.

## Verification run

`OK task2` and `OK task3` — the plan's own automated checks, both green. Plus:

- every `.py` under `experiments/39-connector-lora` compiles (`py_compile`, `doraise=True`);
- both notebooks pass `nbformat.validate`, every code cell parses with `ast`, and each carries
  **exactly one** `parameters`-tagged cell holding raw literal assignments only
  (gate: `MAX_STEPS, ROWS, REPO_ROOT, POD_ID, KERNEL, RUN_TAG`; arm: `SMOKE, RUN, REPO_ROOT,
  DATA_ROOT, CONTROL_RUN, CONTROL_EPOCH_DIR, STOP_AT_STEP, KEEP_MERGED`);
- the gate's argv carries `--target_modules` followed by **9 separate values** including all 8 merger
  names, no `--adapters`, `--lora_dropout` present, `--tuner_type` (not `--train_type`), and
  `str(Config().swift)` starts with `/workspace/` on Windows too (`PurePosixPath`, so the laptop check
  reads the string the pod will run);
- `chain.render` produces a script that waits on `leo_eval38_full.log` + `nvidia-smi` with a bounded
  wait and aborts rather than sharing the card, gates on papermill's exit code, stages exactly one
  allow-list of `RESULTS_*` files with no `runs/` between `git add` and `git commit`, pushes
  `HEAD:main` with three `fetch`+`rebase` retries, reads the key from `/workspace/tmp/.rung39key` and
  `rm -f`s it, and contains no `python -u` launcher;
- `grep -rn "0.5282\|0.5486" experiments/39-connector-lora/` → 0 hits;
- nothing tracked under `experiments/39-connector-lora/runs/`; `git status --short` clean;
- no top-level `tests/` or `scripts/`; no `.sh` produced by this task.

## Deviations

**1. [Rule 2 — missing critical functionality] `assert_splat` gained a per-value separator check.**
Found while testing the guard I had just written: the count check alone passes `("a,b,c",)`.
Fixed in `_tools/reachability_gate.py` (`assert_splat`), documented in the docstring as a hole found
by testing. Commit `c798849`.

**2. [Rule 1 — bug] `_assert_targets_extend` was applied to the control's argv.**
It made `diff_vs_a2` raise on a legitimate control. Moved into `assert_single_variable_39`.
Commit `c798849`.

**3. [Rule 1 — bug] The control was being built from the wrong baseline key.**
`recipe_sweep_train.BASELINES` maps an arm to *its* baseline, so `BASELINES["A2_lr"]` is **arm A at
lr 1e-4** — the thing A2 was a single variable against — and is **not** A2's own recipe. A2's recipe is
the baseline of the arms that sit on top of it (`C_epochs`, `D_clip`), and `D_clip` is the one that
also pins `max_grad_norm 1.0`; `BASELINE_RUN["D_clip"]` is `runs/21_lr_2e4_v1`, A2's own run dir.
`connector_lora_train.A2_BASELINE_KEY = "D_clip"`, with a loud comment. Had this gone unnoticed the
arm would have been read as a two-variable comparison (targets **and** learning rate).
Commit `c798849`.

**4. [Rule 2 — missing critical functionality] The arm's result row uses `macro_f1_<dist>`.**
`metrics.assert_class_f1_reported` scans for keys starting with `macro_f1`. Rung 21's row names them
`fo_class_macro_f1_ID`, and rung 21 never calls the gate — so copying its shape would have made the
RULES §9b gate raise on a run that *had* computed the metric. Commit `c798849`.

**5. [Deliberate, declared] The gate's pod-side paths are `PurePosixPath`, not `Path`.**
Rung 32 used `Path("/workspace/...")`. The property that matters is absoluteness, and the laptop check
is the entire proof that the splat is intact — a `WindowsPath` stringifies to `\workspace\...` and
would have made that proof check the wrong string. Documented in `Config.swift`.

**6. [Deliberate, declared] The control leg is a BRANCH SELECTOR, not a pass/fail on `n_aligner`.**
The plan's §4 criteria said the control "must return `n_aligner == 0`", while §5 pre-declares
`n_aligner > 0` as a legitimate branch. Those cannot both be gates. Resolved in `PLAN.md` §4b/§5: the
control fails only on a broken trainer, a missing adapter, orphans, or lost coverage; its `n_aligner`
selects the branch. Under branch `TARGETS_ONLY` the two legs no longer form the differential that
rules out a prefix-classifier artifact, so `PLAN.md` §5 pre-declares the substitute — rung 32's own
`RESULTS_reachability.csv`, identical recipe with generic targets and `n_aligner == 0` — **before** the
gate runs, precisely so it cannot be invented afterwards.

## Recorded, not litigated: the rung-38 NO-GO

The finished rung-38 eval is recorded where the pre-registration motivates the rung (`PLAN.md` §1a)
and in `CONTEXT.md` §Decisions: the fine-tuned gen-3.6 27B epoch 1 loses to A2 ep1 on all five cells
(`proxy_leaderboard` −0.0344, `object_recognition_ID` −0.0509, `bucket_mean` −0.0290, `margin_OOD`
−0.0255, `aggregation_ID` −0.0178), with the control re-derived from A2 ep1's archived answers by the
same `frame.metrics` code and reproducing `RESULTS_A2_lr.csv` to four places. It is written as a
**floor, not a ceiling** — the A2 recipe was never ported to that backbone, and the arm carries three
deviations of which one was declared — so it narrows the backbone lane without closing it. The
consequence stated is only that **rung 39 is now the main live lane rather than a side bet**, with an
explicit line saying that is not a prediction, not a prior, and must not inflate any claim about what
rung 39 will find.

## Authentication gates

None. Nothing in this task required credentials; the RunPod key is read by the rendered chain from a
file on the pod that the chain deletes, and the chain was not run.

## Known stubs

None. Two files are *intentionally* empty of results and say so: `RESULTS_reachability39.csv` and
`RESULTS.csv` do not exist yet, and both `README.md` and `CONTEXT.md` §Results record
`pending — pre-registered 2026-08-13`. That is the pre-registration working as designed, not a stub:
this task's goal was the pre-registration and the code, and running the gate is explicitly out of
scope.

## Deferred

`deferred-items.md` in this directory: three committed `.sh` files under `submissions/02-rung21-a2/`
pre-date this task and are out of scope. Nothing this task produced is a `.sh`.

## Launch preconditions still outstanding

The chain must **not** be started until all four hold (`PLAN.md` §12):

1. ✅ **(a)** the pre-registration is committed, and committed **before** the code — `21b0919` is the
   parent of `c798849`.
2. ⬜ **(b)** an independent **read-only review returning GO with file:line** (CONSTITUTION §VIII.6).
   Not done. Suggested focus: `reachability_gate.assert_splat` and `_apply_criteria` against
   `PLAN.md` §4a; `connector_lora_train.A2_BASELINE_KEY` against
   `recipe_sweep_train.py:146-177`; `train_to_step`'s monkeypatch/restore and its acceptance of a
   non-zero exit; and a `grep` confirming `PLAN.md` carries no rung-39 number.
3. ⬜ **(c)** the pod GPU free of the rung-38 eval. The chain blocks on it and aborts after
   `max_wait_min` rather than sharing the card, but the pod is currently **stopped** and must be
   started first.
4. ⬜ **(d)** the chain rendered onto the pod. It is **not** rendered: `chain.write` runs from the last
   cell of `00_connector_gate.ipynb`, on the pod. The RunPod API key must be placed at
   `/workspace/tmp/.rung39key` by hand before launching; the chain deletes it.

Also outstanding, and deliberately not done here: the repo was **not pushed to origin**, and the two
commits live on the worktree branch `worktree-agent-ae598cfca8a3e0e3f`. They need to reach `main`
before the pod can pull them — the pod gets code through GitHub only (spec §0a), never by direct copy.

## Self-Check: PASSED

Files (all FOUND): `experiments/39-connector-lora/{PLAN.md, README.md, 00_connector_gate.ipynb,
01_connector_arm.ipynb, _tools/reachability_gate.py, _tools/chain.py, _models/connector_lora_train.py,
_models/README.md}`, `context/39-connector-lora/CONTEXT.md`.

Commits (both FOUND in `git log`): `21b0919`, `c798849`. Neither carries a `Co-Authored-By` or
"Generated with" trailer; both are authored `RodMed0709 <medrod2010@hotmail.com>`.
