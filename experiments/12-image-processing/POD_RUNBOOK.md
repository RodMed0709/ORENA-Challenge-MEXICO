# Rung 12c — pod runbook

The operator starts the run and leaves. This is what runs, what watches it, and when the
machine may be powered off. Everything here is a **file read** — no scrollback, no guessing.

## What runs

`_models/pipeline_12c.py` drives the whole session as one resumable state machine
(11 stages). The launcher is `12_composite_pod.ipynb`.

```
env_check → freeze_subsample → export_control → train_control → eval_control
          → gate_harness → build_maps → export_composite → train_composite
          → eval_composite → report
```

- **Control arm first, alone.** If a 25 % subsample cannot be learned from, that costs
  ~1.5 h to find out, not a full two-arm A/B.
- 🔴 **`gate_harness` is the money stop.** Between the arms it asks one pre-registered
  question: did the subsampled control beat the trivial template-aware floor on **both** ID
  and OOD? If not, the composite arm (~2 h) is not worth running and the pipeline aborts.
  A fired gate is a **finding** (`RULES` §7), not a failure.

## How to watch it — one file, two booleans

```bash
python -m frame.runstate /workspace/orena_12c     # one line, measured ETA, staleness
watch -n5 python -m frame.runstate /workspace/orena_12c
```

`STATE.json` is written atomically (temp + fsync + rename), so a reader never sees a partial
file. `events.jsonl` is the append-only history for the post-mortem. Progress and ETA are
**measured** from the trainer's own step counter over a trailing window — when the numbers
are not available the fields are `null`, never invented.

## When agy may power the pod off

The decision is two booleans in `STATE.json`, never an interpretation of the log:

| `finished` | `safe_to_shutdown` | marker | action |
|---|---|---|---|
| `true` | `true` | `DONE` or `ABORTED` | **shut down.** Success, or a pre-registered abort. |
| `true` | `false` | `FAILED` | 🔴 **do NOT shut down.** A crash — keep the GPU alive for the post-mortem. Alert the operator. |
| `false` | — | none | still running. If `is_stale` (> 5 min no heartbeat), alert. |

```python
from frame import runstate as rs
s = rs.read('/workspace/orena_12c')
if s['finished'] and s['safe_to_shutdown']:
    shutdown_pod()
elif s['finished']:
    alert('crashed — GPU kept alive')
elif rs.is_stale('/workspace/orena_12c'):
    alert('no heartbeat > 5 min')
```

The default `shutdown_on_failure=False` is deliberate: a crash is exactly when the next
question is *why*, and that is far cheaper to answer on a live GPU than a dead one. Flip it
in the config only if the budget matters more than the post-mortem.

## Resuming after an interruption

Re-run the launch cell (or `P.run(cfg)`). Stamped stages are skipped; only the current stage
repeats. A dropped SSH session, an OOM, or a power cut costs one stage, not the session.

## 🔴 What is NOT verified off the pod — do this first, on the pod

`swift` is not installed locally and `external_data` carries no video files, so every stage
touching training, CUDA or video decode is **unverified code** until smoked on the pod:
`export_control`, `train_control`, `eval_control`, `build_maps`, `export_composite`,
`train_composite`, `eval_composite`, `report`. They are written as explicit `_todo` holes
that raise `NotImplementedError`, not plausible-looking guesses.

**On the pod, before the real frac:**
1. `dry_run=True` — proves orchestration/gates/paths in seconds.
2. Wire each `_todo` body to its rung-02/06 module (`lora_sft_train._export`,
   `vit_lora_train._train`, `frame.run`) and smoke it with a tiny frac.
3. Only when every stage's smoke is green, run the real frac.

## Verified locally (green before any GPU)

- `frame.subsample` — 13,748 → 3,449, 92/92 videos, per-format within 2 % of target,
  deterministic, tamper-detected.
- `frame.runstate` — atomic/durable writes, `\r`-aware fragment parsing (tqdm bars are
  visible), measured ETA, heartbeat on unparseable output, resumed-run reporting, the
  four-way terminal semantics above.
- `composite_train.consistency_gate` — JSONL layout checked against `engine._messages`
  itself. `gate.aux_view_payload_gate` — flag-off byte-identity + null-arm + typo-raises.
- `pipeline_12c` dry run — full 11-stage machine, resume, abort, crash.

## Pre-registration

Arms, subsample fraction, evaluation set, decision rule and the three stopping tiers are
fixed in `context/12-image-processing/CONTEXT.md` §12c PRE-REGISTRATION. Do not edit a
threshold after seeing a number — that turns a gate into a story.
