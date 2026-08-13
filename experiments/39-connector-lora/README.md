# 39 — connector LoRA

| Notebook | Rung | Metric (primary) | Verdict |
|---|---|---|---|
| — (control, rung 21 `A2_lr` ep1, `checkpoint-901`) | `A2_lr` ep1 | `object_recognition_ID` 0.6088 | baseline |
| `00_connector_gate.ipynb` | 39-gate | `n_aligner` | pending |
| `01_connector_arm.ipynb` | 39 | `object_recognition_ID` | pending |

**Status: NOT RUN.** The gate has not run and the arm has not run. Everything in this folder is the
pre-registration and the code.

## The ONE variable

`--target_modules` gains the eight merger `Linear` layers — `model.visual.merger.linear_fc{1,2}` and
`model.visual.deepstack_merger_list.{0,1,2}.linear_fc{1,2}` — as **`all-linear` PLUS the eight
names**, nine separate argv values. Coverage is extended, never replaced. Everything else is A2,
verbatim, read against `21_lr_2e4_v1` arm `A2_lr` **epoch 1**, `checkpoint-901`.

Across 30+ rungs the merger has never received a gradient, and the cause is a regex rather than a
decision: `context/decisions/the-merger-is-unreachable-by-default.md`.

## The pre-registration

**[`PLAN.md`](PLAN.md)** — written and committed **before** any rung-39 number exists. It carries the
single variable, the blocking gate with both legs and their pass criteria, the named control with its
exact scored numbers, the declared primary cell, the closed list of declared deviations, and what a
faithful negative looks like.

Curated context: **`context/39-connector-lora/CONTEXT.md`**.

## How to run

Headless, via **papermill** (spec §5b) — never a `.sh`, never a hand-run `.py`.

```
papermill 00_connector_gate.ipynb runs/<tag>/gate.ipynb -k <kernel> --log-output
papermill 01_connector_arm.ipynb  runs/<tag>/arm_full.ipynb -p SMOKE False -k <kernel> --log-output
```

The gate is **blocking**: it RAISES on failure, papermill turns that into a non-zero exit, and the
chain stops before any training GPU is spent.

For the pod, the serial chain (wait for the GPU → gate → smoke → full → commit → push → stop the pod)
is **rendered**, not committed: `_tools/chain.py` is an importable renderer, and the last cell of
`00_connector_gate.ipynb` writes the script to `/workspace/tmp/` and prints the `nohup bash …` line
for the human. Committing a `.sh` is forbidden (CLAUDE.md, CONSTITUTION §VIII.1, §IX.3) and `.py`
files are libraries, never launchers — so what is committed is the renderer, which is reproducible,
reviewable and diffable; the artifact it produces is pod scratch and is deleted when done.

## Files

| path | role |
|---|---|
| `PLAN.md` | the pre-registration |
| `_tools/reachability_gate.py` | two-leg reachability gate: the splat fix, the coverage criterion, the `adapter_config.json` cross-check |
| `_tools/chain.py` | renders the serial chain into pod scratch |
| `_models/connector_lora_train.py` | the arm engine: `swift_args_39`, the multi-value-safe diff, the guards, the A1 supervisor |
| `_models/README.md` | engine ↔ rung ↔ notebook ↔ run index |
| `00_connector_gate.ipynb` | both gate legs in one notebook, one exit code |
| `01_connector_arm.ipynb` | train → merge → eval → canonical scoring |
| `RESULTS_reachability39.csv` | written by the gate (pending) |
| `RESULTS.csv` | written by the arm (pending) |
