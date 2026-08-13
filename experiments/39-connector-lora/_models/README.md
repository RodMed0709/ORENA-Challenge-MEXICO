# 39 `_models/` — engines only

| Engine | Used by rung | Notebook | Latest run |
|---|---|---|---|
| `connector_lora_train.py` | 39 (`E_connector`) | `01_connector_arm.ipynb` | — (not run) |

Engines only (spec §6). No `smoke_*.py`, no `run_*.py`, no `.sh`, no `resume_*.py` — the `SMOKE`
toggle lives in the notebook's `parameters` cell and the chain is *rendered* by
`_tools/chain.py`, never committed.

## What this engine reuses, and what it owns

**Reused, never copied** — the rung 06 → rung 21 precedent:

| from | what | why not copied |
|---|---|---|
| `experiments/21-recipe-sweep/_models/recipe_sweep_train.py` | `RecipeSweepConfig`, `swift_args_21`, `control_cfg`, `effective_batch`, `main` | `swift_args_21` **is** the recipe; a copy would silently re-date every rung-21 comparison |
| `experiments/06-vit-lora/_models/vit_lora_train.py` (transitively) | `_train`'s broken-run guard and log tee, `read_g1`, `merge_checkpoint`, `list_checkpoints` | same reason, one level down |
| `experiments/39-connector-lora/_tools/reachability_gate.py` | `MERGER_TARGETS`, `assert_splat`, the branch constants | the eight long dotted names must exist in **exactly one** place; two hand-kept copies is a typo waiting to become a silent no-op |

**Owned here, and nowhere else:**

- `swift_args_39` — the splat and the branch-selected `--freeze_aligner`, applied to a local copy of
  rung 21's argv. It does **not** edit `swift_args_21` or `_swift_args`.
- `_as_map_multi` — the multi-value-safe flag→values mapper. 🔴 Rung 21's `_as_map`
  (`recipe_sweep_train.py:251`) keeps only the FIRST value after a flag; measured on a laptop
  2026-08-13, `_as_map(['--target_modules','m1','m2','m3'])` returns `{'--target_modules': 'm1'}`, so
  8 of this arm's 9 targets would go unchecked while the single-variable gate reported a clean diff.
  **`_as_map` is deliberately NOT edited** — it is the proof behind every already-scored rung-21 arm.
- `assert_single_variable_39`, `assert_supervised`, `assert_learned`, `assert_merger_reached`,
  `warn_disk` — the gates. All RAISE except `warn_disk`, which warns and never blocks.
- `train_to_step` — the A1 supervisor (`PLAN.md` §6a): train under the control's 3-epoch cosine and
  terminate the process once `checkpoint-901` is complete, because shortening the schedule *is* the
  second variable.

## DO NOT rename

`connector_lora_train.py` is imported by `01_connector_arm.ipynb` (same folder). Nothing outside
`experiments/39-connector-lora/` imports it today — grep the repo before moving or renaming it.

⚠️ It imports **upward** into `experiments/21-recipe-sweep/_models` and (through it)
`06-vit-lora`, `02-lora-sft`, `18-count-aug`. Renaming anything there breaks this engine.
