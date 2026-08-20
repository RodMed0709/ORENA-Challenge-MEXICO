# Rung 19b — does EXTERNAL recognition data buy what rung 47 could not?

> **Status: TRAINING.** Launched 2026-08-19 23:17, UNAM `tmux leo-rung19b`, GPU 0.
> Five gates passed before any GPU spend. Heavy artifacts in `/mnt/storage/uaq_user/rung19b/`.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 test videos into training | 21 A2 | shipped — 0.5809 on the platform |
| 47-epochs-vs-corpus | `--num_train_epochs 3→5`, A2's own corpus | 21 A2 | done — does not ship |
| 48-centre-probe | *nothing trained* — a second, external eval axis | — | scored; `bag_f1` is the ruler |
| **19b (this)** | **`--dataset`: + 5,718 Strasbourg rows** | **47 ep4** | **training** |

## The arm

`19b_merged_ep5_v1` — 20,133 rows = A2's 14,415 **plus** 5,718 rows built from the 35 CholecT50
videos `experiments/splits/cholect50_split_v1.csv` freezes for training. Recipe, schedule and
seed are rung 47's, unchanged: lr 2e-4, r8/α32, `max_grad_norm` 1.0, 1×16, 5 epochs, seed 42.

**Read ep4 against rung 47's ep4.** Both anneal a cosine over five epochs, so the epoch is
comparable and the corpus is the only difference — the comparison rung 42 could never make,
because it moved the corpus *and* ran an epoch its control never ran
([[rung42-gain-was-epochs-not-corpus]]).

## Five gates, all before the GPU (`runs/*/gates.json`)

| | gate | measured |
|---|---|---|
| G1 | corpus is the declared bytes | 20,133 rows, sha256 `5f26927b3f65…` |
| G2 | every image resolves | 20,133 paths, all under `/mnt`, all present |
| **G3** | **no held-out video leaks** | **5,718 CholecT50 rows / 35 videos / 0 of the 15 `hold`** |
| G4 | provenance of the challenge half | all 14,415 rung-47 rows present; 5,718 new |
| G5 | single variable **vs rung 47** | `--dataset`, and nothing else |

🔴 **G3 is the one that protects the ruler.** The moment a `hold` video enters training,
Strasbourg is inside the distribution and rung 48 stops measuring **centre**, measuring
**unseen scene** instead.

🔴 **G5 could not use `rst.diff_vs_control`.** That helper builds the baseline with
`control_cfg`, which shares `--dataset` *by construction* — so on this arm it returns an empty
diff and certifies as single-variable a run whose only variable it is structurally unable to
see. The gate is written against rung 47's own argv instead.

## Wall clock — and the estimate that was wrong

6,295 steps (1,259/epoch × 5). The smoke measured 14.55 s/it; **the live run settles at
~16.1 s/it ⇒ ~28 h, with ep4 (step 5,036) at ~22.5 h.** Take the live number: a 4-step smoke
runs warm and short and reads optimistic.

⚠️ The hand-off note carried **~14 h**. That was an estimate, never a measurement: 20,133 rows
are 40 % more than the 14,415 rung 47 trained on, and rung 47 took 21 h at 17.11 s/it.
`--save_strategy epoch` means every epoch stands alone, so a kill after ep4 costs only ep5.

## How it is read

**Primary: `bucket_mean` on the untouched 38-video / 6,252-question eval**, paired against rung
47 ep4 through `frame.metrics` (clusters on VIDEO — the correction that closed rung 47's CI).
**Secondary and never averaged with it:** rung 48's `bag_f1` over the 15 `hold` videos.

⚠️ For rung 47 those 15 are **unseen centre**; for this arm they are **unseen scene**, because it
has now met Strasbourg. The paired delta stays valid; what each arm's own number *means* does
not. Say which one you are reporting.

## The `swift` shim

`code/bin/swift` on the box. pip baked `#!/data/uaq_user/envs/…` into the console script and
`/data` has not existed since the 18-Aug reboot, so exec fails with a bare `ENOENT` that reads
as *"swift is not installed"*. The shim carries a live shebang and goes ahead of it on `PATH`,
for this run only — patching the env or `vit_lora_train.py` would edit a shared artifact that
every rung on the ladder is compared against.
