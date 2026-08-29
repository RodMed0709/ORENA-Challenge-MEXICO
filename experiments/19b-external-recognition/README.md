# Rung 19b — does EXTERNAL recognition data buy what rung 47 could not?

> **Status: CLOSED — FAITHFUL NEGATIVE (wash).** Trained 2026-08-19 23:17 → 2026-08-21 03:32
> (28.25 h), chained eval finished 2026-08-21 06:42. External recognition data buys **nothing**
> readable: the pre-registered cell straddles zero in both halves.
> Five gates passed before any GPU spend. Heavy artifacts in `/mnt/storage/uaq_user/rung19b/`.

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 test videos into training | 21 A2 | shipped — 0.5809 on the platform |
| 47-epochs-vs-corpus | `--num_train_epochs 3→5`, A2's own corpus | 21 A2 | done — does not ship |
| 48-centre-probe | *nothing trained* — a second, external eval axis | — | scored; `bag_f1` is the ruler |
| **19b (this)** | **`--dataset`: + 5,718 Strasbourg rows** | **47 ep4** | **done — wash, does not ship** |

## Verdict — a wash, and it does not ship

**The pre-registered cell** (`19b ep4 − 47 ep4`, paired, clustered on video, `frame.metrics`):

| cell | n | videos | delta | CI | excludes 0 |
|---|---|---|---|---|---|
| `ALL_ID` | 2,252 | 28 | **−0.0089** | [−0.0347, +0.0171] | no |
| `ALL_OOD` | 4,000 | 10 | **+0.0075** | [−0.0113, +0.0258] | no |

Neither excludes zero, and the two point estimates point opposite ways. **5,718 Strasbourg rows
bought nothing readable.** The arm's best epoch (ep5, `bucket_mean` **0.6557**) never reaches
rung 42 ep4's **0.6744**, so there is nothing here to ship either.

**The epoch curve** (`RESULTS_epochs_full6252.csv`), 36–38 min per epoch:

| epoch | ckpt | bucket_mean | acc_ID | acc_OOD | object_recognition_ID | object_recognition_OOD |
|---|---|---|---|---|---|---|
| 1 | `checkpoint-1259` | 0.5465 | 0.5222 | 0.5923 | 0.6134 | 0.6744 |
| 2 | `checkpoint-2518` | 0.5952 | 0.5404 | 0.6700 | 0.6273 | 0.7421 |
| 3 | `checkpoint-3777` | 0.6321 | 0.6115 | 0.6755 | 0.7137 | 0.7478 |
| 4 | `checkpoint-5036` | 0.6479 | 0.6372 | 0.6825 | 0.7377 | 0.7689 |
| **5** | `checkpoint-6295` | **0.6557** | 0.6350 | **0.6987** | 0.7299 | 0.7840 |

⚠️ `best = ep5` is a POINT estimate. Within-arm, **`ep5 − ep4` straddles zero everywhere**
(`ALL_ID` +0.0022, `ALL_OOD` −0.0162) — the same shape rung 47 had, and the reason the ladder
reads the CI and not the `max`.

🔴 **The curve gets WORSE before it recovers, significantly.** `ep2 − ep1` is **−0.0777**
[−0.1175, −0.0385] on `ALL_OOD` and `ep3 − ep2` is **−0.0710** [−0.1007, −0.0408] on `ALL_ID`,
both excluding zero. A run read at one epoch would have reported a different sign for this arm.

📌 `temporal_grounding_ID` has **n = 1** in this eval set; its ±1.0 rows are one question
flipping, not a result. Read `ALL_*` and the two owned groups.

**Artifacts** — `RESULTS_epochs_full6252.csv` (the curve), `RESULTS_corpus_vs_r47_paired_ci.csv`
(the rung's question), `RESULTS_epoch_curve_paired_ci.csv` (within-arm). Written by the chain at
06:42 into `repo_leo` on the UNAM box, where they sat uncommitted for four days — the verdict
reached `context/NOW.md` on 08-22 before the artifacts behind it reached git.

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
