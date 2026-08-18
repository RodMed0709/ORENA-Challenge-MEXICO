# context/NOW.md — what is happening RIGHT NOW

## 🔴 2026-08-18 17:00 — rung 47 ep3 scored **0.6142**, the control is RED, and the control was MIS-DESIGNED

**The number.** `checkpoint-2703` on the 8 held-out videos / 1,283 questions:
`bucket_mean` **0.6142** against A2's archived **0.6342** ⇒ **−0.0200**, twice the ±0.01 band
that was declared before the run. The notebook printed RED and said "a STACK effect is present".

🔴 **That conclusion does not follow, and the fault is in how the control was specified.**
`--num_train_epochs` is not only the number of epochs: with `--lr_scheduler_type cosine` it sets
the length the cosine is annealed over. So **"epoch 3 of a 5-epoch run" is not the same model as
"epoch 3 of a 3-epoch run"**, and never could have been. MEASURED in this run's own log:

| at step 2703 | learning rate |
|---|---|
| rung 47 (cosine over 4,505 steps) | **7.1e-05** — 35 % of the 2e-4 peak, mid-anneal |
| A2 (cosine over 2,703 steps) | **≈ 0** — fully annealed, its final checkpoint |

A half-annealed checkpoint scoring below a fully-annealed one is the expected result, not a
finding. ⇒ the −0.0200 is **stack + schedule**, and nothing here separates them. The RED is real;
its stated cause is not established and must not be quoted as "the stack costs 0.02".

## 🟢 What survives — and it is cleaner than the design assumed

**Rung 42 also ran 5 epochs** (1,212 steps each, checkpoints 1212…6060). So `47_ep3 − 42_ep3` is
**schedule-matched**: same epoch, same cosine length, same peak LR. Its only differences are the
corpus and the stack. Measured, video-clustered, n = 8 videos:

| cell | delta | CI | excludes 0 |
|---|---|---|---|
| `aggregation_ID` | +0.0217 | [−0.061, 0.109] | — |
| `aggregation_OOD` | +0.0130 | [−0.026, 0.048] | — |
| `object_recognition_ID` | −0.0620 | [−0.145, 0.021] | — |
| `object_recognition_OOD` | −0.0159 | [−0.082, 0.052] | — |
| `ALL_ID` | −0.0207 | [−0.067, 0.023] | — |
| `ALL_OOD` | −0.0038 | [−0.051, 0.039] | — |

`d(bucket_mean) = −0.0120`. **No cell wins and no cell vetoes.** At matched epoch and matched
schedule, A2's corpus and rung 42's merged corpus are not distinguishable on this eval set.

**The difference-in-differences is now the primary read on firmer ground than when it was
proposed:** both arms anneal a cosine over 5 epochs, so the schedule cancels inside each arm's
`ep4 − ep3`. Rung 42's own internal rise is **+0.0482** (and it is concentrated entirely in
`aggregation` — both counting cells exclude zero, neither recognition cell does). If rung 47 rises
by about the same on A2's corpus, the +0.0402 was **epochs**.

## 🔻 This retroactively qualifies a number rung 42 published

Rung 42's "at the matched epoch its corpus **loses 0.0079**" is `42_ep3 (0.6262)` against
`A2_ep3 (0.6342)` — and that pair carries the **same schedule confound**: a 5-epoch mid-anneal
checkpoint against a 3-epoch fully-annealed one. It was never a clean corpus comparison. The
schedule-matched version of that question is the table above, which says **no difference**.

🟢 **The confound runs AGAINST rung 42 in the number that shipped**, so `+0.0402` is if anything
understated: `42_ep4` (step 4848/6060, LR still ~1.9e-05) beat a fully-annealed A2 by that margin
while itself not being fully annealed.

## 🟢 Class-balanced F1 ran for the first time, and it is healthy

Never executed before — the smoke skips it and rung 42's archive has no `predictions.json`.
Pooled macro-F1 **0.8450** (n = 490), **0 illegal `fo_class` tokens**, exact-set 0.7776;
ID 0.8528, OOD 0.8303. The tail holds: `Sponge` at n = 12 scores F1 0.783, so there is **no
collapse onto the head class** — the failure mode rung 45 found in the 27B is absent here.

The counting crosstab shows the known shape unchanged: correct at 1–4, and a systematic
**undercount** above that (gold 9 → predicted 3–6, gold 11–12 → predicted 6). `number` remains
the error mass.

## Where the run is

GPU 0 still training, into epoch 4 (`tmux leo-rung47`). **ep4 lands at step 3604**, and that is the
one the rung exists for. `RESULTS.csv` is deliberately unwritten until the sweep holds ep3 **and**
ep4 — running the notebook with `EPOCHS=[3, 4]` then costs **zero GPU on ep3**, whose answers are
archived at `runs/47_a2_ep5_v1/eval/47_a2_ep5_v1_ep3_full/`.

## 🟢 2026-08-18 12:25 — FOR THE TEAM: UNAM rebooted and killed rung 47; it is RESUMED and running

**1. 🔴 There is a job on UNAM GPU 0 again. Do not kill it.** `tmux leo-rung47`, relaunched
12:24:41 from `checkpoint-1802`, stepping from **1805/4505**. **ETA ~12 h.** The do-not-kill note
on the box is `/mnt/storage/uaq_user/rung47/OWNER.md`.

```
tail -3 /mnt/storage/uaq_user/rung47/runs/47_a2_ep5_v1/ckpt/v0-*/logging.jsonl
```

**2. What happened.** The box rebooted at **11:55:53** after 38 days of uptime (previous boot
Jul 11). No shutdown record, and nothing we ran can reboot a machine — the training was healthy 6
minutes before at step 2390 with its `train_speed` unchanged. It died at **step 2400/4505, epoch
2.66, after 11 h 17 m**. ep3 (step 2703) was ~1 h 25 m away and never got written.

**3. 🟢 Nothing was lost, and the resume cost ~2 h 48 m instead of 11 h 17 m.** `checkpoint-901`
and `checkpoint-1802` carry optimiser, scheduler and RNG state, so `--resume_from_checkpoint`
restored rather than restarted — verified by the first logged step being **1810**, not 0.

## 🔴 `/data` is gone as a mountpoint — use `/mnt/storage`

| | before the reboot | now |
|---|---|---|
| `/dev/sda1` (16.4 T, ext4) | `/data` | **`/mnt/storage`** |
| `~/storage` → `/data/uaq_user` | fine | **dangling symlink** |

There is **no `/etc/fstab` entry for `/data`** — that mount was set up by hand and did not
survive. Worth asking whoever administers the box to add one, or the next reboot repeats this.

⚠️ **Do NOT "fix" this by repointing the `~/storage` symlink.** The envs bake the old path into
every pip console script (`#!/data/uaq_user/envs/orena-train/bin/python`), plus `argv.json`, the
Jupyter kernelspec and every recorded run path. Repointing leaves those broken and produces a box
that looks fixed and is not. What restores the old path exactly is
`sudo mkdir -p /data && sudo mount --bind /mnt/storage /data`, and `sudo` prompts for a password.

🟢 **But we are NOT blocked on root, and an earlier version of this section wrongly said we
were.** MEASURED 2026-08-18: the interpreter relocates fine —
`/mnt/storage/uaq_user/envs/orena-train/bin/python` reports the right `sys.prefix`, `torch.cuda`
is True, and `transformers` / `ms-swift` import. Conda derives its prefix from the binary's real
path. **Only the console scripts are broken**, and they are bypassed by calling the module:

```python
python -c "import sys; from swift.cli.main import cli_main; sys.exit(cli_main())" sft ...
```

⚠️ **A broken shebang reports as `FileNotFoundError: 'swift'`**, which reads as "swift is not
installed". It is — the file is on PATH. `execve` returns ENOENT for a missing *interpreter* too
and Python blames the command. That cost one smoke run; do not re-diagnose it as PATH.

**The env was NOT modified** — no shebang rewritten — so a later bind mount leaves nothing to undo.

## 🔴 Absolute paths inside the CORPUS are what actually stopped the resume

`corpus/train.jsonl` carries the **absolute image path in every one of its 14,415 rows** (that is
what `_record` writes), all under `/data/uaq_user/frames_cache/...`. The first resume attempt loaded
the 8B, reached swift's `train_dataset[0]` sanity check and died with

```
ValueError: Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` ...
```

which names `max_length` and `truncation_strategy` and has **nothing to do with either**. If you
see that error after a move, look at the paths inside the data, not at the tokenizer.

⇒ `--dataset` now points at **`corpus/train_mntpaths.jsonl`**. **`train.jsonl` was NOT touched** —
its sha256 is the provenance anchor against the digest rung 21 declared. The twin is proven to be a
pure prefix rewrite: undoing it reproduces the original **byte for byte**, and
`corpus/PATH_REWRITE.json` records both digests plus that proof. Same pattern as the existing
`train_podpaths.jsonl`, in the other direction.

## How it was relaunched

`rung47/code/resume_a2_ep5.py` **derives the argv from the run's own `argv.json`** instead of
retyping 24 flags — that is how a "resume" silently becomes a different experiment. It adds only
what the resume forces: `--resume_from_checkpoint`, `--output_dir` set to the existing `v0-*`
directory, `--add_version false` (without it swift's default creates a new `v1-<stamp>` and splits
ep1/ep2 from ep3/ep4/ep5 across two trees), and `--load_args false` (with the default True swift
reads `args.json` from the checkpoint, which is full of `/data` paths). The exact argv is archived
at `runs/47_a2_ep5_v1/argv_resume.json`.

Preflight, all before the GPU: no `/data` path survives the rewrite, every path exists, the
checkpoint is complete and at step 1802, the corpus twin is a pure prefix rewrite, and
`diff_vs_A2.json` still says the only difference from A2 is `--num_train_epochs`.

⚠️ `swift export` has the same disease: it resolves the base model from the **adapter's**
`args.json`, so a merge after a move needs `--model` passed explicitly or it dies with
`ValueError: path: '/data/...' not found` before touching a weight. `eval_arm47` now derives it.

## 🟢 The scoring for this rung is BUILT and already on the box

`experiments/47-epochs-vs-corpus/` — `01_eval_epochs.ipynb` + `_tools/eval_arm47.py` (which imports
rung 45's `eval_arm45.py` rather than copying it) + README. Synced to `repo_leo`, checksums verified
both sides. Every read-only gate passed **before** the reboot: judge resolves offline, 6 252 test
items load, split 8 videos / 1 283 questions, frames-cache 1 283/1 283.

Rung 42's per-question archive for all five epochs is already on the box at
`runs/47_a2_ep5_v1/controls/` (1 283 rows, 800 heico, verified).

🔴 **A2's per-question archive does NOT exist**, settled by a full-bucket S3 scan — so the ep3
control is a SCALAR check against 0.6342, with no paired CI. Two findings worth keeping: the
`tmp/leo_chain/critico/` and `tmp/leo_backup_*/` prefixes DO hold `runs/` trees that no `repo_*/`
prefix can (`runs/` is gitignored), which is where to look before declaring a run's artifacts
gone; and `tmp/leo_backup_20260806/step5_seed42/full/greedy/results.csv` is a **decoy** — A2
greedy, but rung 10's stratified subsample, 600 rows over 91 videos, **zero** of the 8 held-out,
~0.83 accuracy. Overlap with the 1,283 is 0.

### Three things measured on the way, so nobody pays for them twice

* 🔴 **UNAM has no `boto3` and no S3 credentials.** Fetch control archives on the LAPTOP and rsync
  them over; the notebook only verifies. A fetch written as a notebook cell runs on UNAM, which is
  how this was found.
* 🔴 **`s3.download_file` 403s on bucket `gf78k60nlt`.** Our credentials are denied `HeadObject`,
  and `download_file` HEADs before it GETs. `get_object` and `ListObjects` are allowed — so an
  object can be *listed* and not *downloaded* by the obvious call.
* 🔴 **`ping` is useless for deciding whether UNAM is up.** ICMP is blocked, so 100 % packet loss
  is the normal state and says nothing. Test with `ssh`. (This cost one wrong "the box is down"
  call today.)

### Jupyter now works in `orena-train`, and it cost the training nothing

`jupyterlab 4.6.3 + ipykernel 7.3.0 + papermill 2.7.0`, kernel registered as `orena-train`.
Installed with the env's own `pip freeze` as a **constraints file**, so pip could only ADD: dry-run
showed 71 packages and **zero** overlap with the 153 installed, and the post-install diff against
the run's own `pip_freeze.txt` was **0 packages changed**. `torch 2.11.0+cu128 / transformers 5.12.1
/ torchvision 0.26.0+cu128 / ms-swift 4.4.1` all intact; the training kept its 16.93 s/it.

⚠️ Papermill needs the parameters cell **tagged** `parameters`, or it injects above the imports
instead of below them (the rung-16 trap). The rung-47 notebook is tagged.

---

## 📋 2026-08-18 — FOR THE TEAM: a training run is LIVE on UNAM, and "UNAM cannot train the 8B" was false

> 🔻 **SUPERSEDED 2026-08-18 12:25 by the section above.** The run it describes was killed by the
> 11:55:53 reboot and has been RESUMED from `checkpoint-1802`; its ETA, its step count and every
> `~/storage/...` path here are stale (use `/mnt/storage/uaq_user/...`). Point 3 (the env myth is
> retracted) and the clone-shebang trap both still stand.

**1. 🔴 There is a job on UNAM GPU 0 right now. Do not kill it.** `tmux leo-rung47`, started
00:33:56 box time, **ETA ~20 h** (swift's own `remaining_time`, not an estimate of ours).

```
tail -3 ~/storage/rung47/runs/47_a2_ep5_v1/ckpt/v0-*/logging.jsonl
```

⚠️ Its `full.log` has been frozen at 44,854 bytes since startup because stdout is block-buffered.
**That is not a hang** — progress is in `logging.jsonl`, and the GPU reads 19.5 GiB at 100 %.

**2. 🎯 What it is: rung 21's arm `C_epochs`, finally run.** One variable off baseline A2 —
`--num_train_epochs` 3 → 5 — on A2's own corpus. Rung 21 designed it, VRAM-probed it
(`RESULTS_vram_C_epochs.json`) and never trained it. It has been waiting 20 days.

**Why it matters:** rung 42's shipped +0.0402 may be **epochs, not data**. At the matched epoch
its merged corpus *loses* 0.0079 against A2; the whole gain appears at epoch 4, which the control
never ran. Nobody separated the two. This run separates them — ep4 vs rung 42's ep4, corpus as the
only difference.

| ckpt | ~at | what it is |
|---|---|---|
| **ep3** | ~12 h | 🔑 **the stack control** — must reproduce A2's **0.6342** |
| **ep4** | ~16 h | 🎯 **the answer** — against rung 42's **0.6744** |
| ep5 | ~20 h | the epoch that already fell in rung 42 |

`--save_strategy epoch`, so value accrues and the run can be cut at any epoch without loss.

**3. 🔴 "UNAM cannot train the 8B" was FALSE and is retracted.** `transformers>=4.57` is a **floor,
not a ceiling** — all four envs import `Qwen3VLForConditionalGeneration`. No fifth env was built
from scratch: `~/storage/envs/orena-train` is a **clone of `orena-gen36`** plus `qwen-vl-utils
0.0.14` and `torchvision 0.26.0+cu128`, both `--no-deps`. `orena-gen36` is verified untouched.

⚠️ **Clone trap, worth knowing:** `conda create --clone` leaves pip console scripts carrying the
**source env's shebang**, so `swift` ran on gen36's interpreter and died on a missing
`qwen_vl_utils`. Fixed by rewriting line 1 of every script in `bin/` that pointed at the old env.

⚠️ **The stack is NOT the ladder's:** 5.12.1 / torch 2.11 vs the 4.57 / 2.8 every 8B rung was
scored on. That is why ep3 is a control and not just another checkpoint.

**4. 🟢 The corpus was rebuilt on the box and PROVEN, not uploaded.** No `train*.jsonl` existed on
UNAM. Regenerated from `orena-data` + `frames_cache` with rung 18's own builder: **14,415 rows**
(13,748 real + 667 minted), every rung-18 gate passing, and its pod-path twin hashes **exactly**
to the `180e28f0…fbd8e8b` rung 21 declared. Zero challenge data transferred. Gates cleared before
the GPU was touched: dataset sha256, a diff against A2 that is **exactly `--num_train_epochs`**,
and G1 (LoRA reaches the ViT, 25.67 M trainable).

**5. 🔴 Distilling the 27B is DEAD, by measurement.** The vocabularies do not match — **248,320
(27B, `TokenizersBackend`) vs 151,936 (8B, `Qwen2Tokenizer`)**, different `tokenizer.json`. GKD's
token-level JSD (`swift/rlhf_trainers/gkd_trainer.py:105`) has no route across that, and neither
ms-swift nor TRL ships a vocabulary mapping. ms-swift *does* support GKD/OPD-RL/OPSD on our exact
pin — the blocker is the pair, not the framework.

**6. 🔻 Two doc corrections, both pushed.** `--train_type` **does not exist** (it is `--tuner_type`)
and `CLAUDE.md:33` had carried it for a month while four other files had it right. And the aligner
flag **does** reach the connector on transformers 5.x (368 modules vs 360; the 8 extra are the
merger + deepstack) while rung 32's zero stands on **4.57** ⇒ version-dependent, so the ladder still
needs the explicit `target_modules`. See [[aligner-flag-reads-reachable-but-measured-zero]].

---

## 📋 2026-08-17 — FOR THE TEAM: rung 45 is SCORED, and the 27B line is closed

Everything below is on `main`. Not in git and cannot be: the rung-45 checkpoints and the 19b
corpus, both under gitignored `experiments/*/runs/`.

**1. 🔴 The merged corpus HARMS the 27B.** R00 (rung 18's 14 415) vs R0 (rung 42's merged
19 384), one variable — `diff_vs_R00` lists exactly two fields, `arm` and `corpus`. On the clean
1 283:

| | R00 | R0 | Δ |
|---|---|---|---|
| `bucket_mean` | 0.4315 | 0.3170 | **−0.1145** |
| `object_recognition_ID` | 0.5680 | 0.3120 | −0.2560 |
| `macro_f1_ID` | 0.6876 | 0.3465 | −0.3411 |
| train loss ep1 | 0.0694 | **0.0589** | — |

Paired, video-clustered: **four cells veto**, including `object_recognition` — the cell PLAN §6
named in advance as this rung's modal failure. **Same rows gained the 8B +0.0402 and cost the 27B
−0.1145.** Full numbers in `experiments/45-gen36-data-and-reg/RESULTS*.csv`.

🔑 **Three things harden the negative:** R0's train loss was **lower**, so selecting on loss picks
the worse arm; the OOD cells structurally favour R0 (8 of the 10 heico test videos are in its
training) **and it loses them anyway**; and it emits **zero** illegal `fo_class` tokens against
R00's 31 while macro-F1 halves — it stopped guessing outside the vocabulary and **collapsed onto the
head class**, answering `Clip` to almost everything.

**2. 🔴 The 27B line stops as a shipped candidate — see [[27b-is-a-teacher-not-a-candidate]].**
On the clean 1 283 at the same corpus: 8B A2 **0.6342**, 27B bf16 `conn4e5` **~0.558**, 27B NF4 R00
**0.4315**. Quantization costs −0.127 here, but handing all of it back still leaves the 27B
**−0.076 behind the 8B**, and rung 42's 0.6744 sits 0.116 above the best 27B imaginable.

⚠️ **This does NOT establish that the 27B backbone loses, and the note says so explicitly.** Every
27B arm ever run inherited hyperparameters swept **on an 8B**; the backbones train on different
frameworks; the only 27B-native recipe rung swept two knobs; and `weight_decay` differs (0.0 in
`conn4e5`, 0.1 in R00). **The stop is on time and resources, not on the backbone being fairly
beaten.** What would reopen it is listed in the note.

⇒ **R1, `alpha` 32→16 and the bf16 re-run are all dead with it.** They were levers on a model that
is no longer a candidate. Both GPUs are free.

**3. 🟢 The 27B becomes the teacher.** It counts BETTER than the 8B on external data
(`experiments/19-external-count/`): MISAW **0.3913 vs 0.2048**, SurgSigma **0.4905 vs 0.4027**,
Spearman **0.6921 vs 0.4965**. It does not merely get more right — it **orders quantities better**,
and `number` is 71 % of our error. ⇒ **ship the 8B, distil the 27B into it.**

The one live bet now has three sources, all pointing at `number` and all landing on rung 42's
checkpoint: the 27B as teacher, **SurgΣ-DB**'s 309 123 human traces, and the **19b** corpus (8 079
exact-count rows). ⚠️ They are **three interventions, not one** — folding them together repeats the
two-variable mistake rung 45 just paid for.

⚠️ **SurgΣ-DB is the dataset and SurgVLM is a model that never released weights** — verify-fail in
the licence sweep alongside LLaVA-Surg and Surgical-LVLM. There is no SurgVLM to use.

**4. 🔴 Nothing is ready to launch, and the blocker is the environment.** Qwen3-VL needs
`transformers 4.57`; UNAM's four envs are at 5.12 / 5.14 / 5.5 / 5.15. **UNAM cannot train the 8B
at all** — that line has always run on RunPod. Also missing on the box: the 19b corpus (6.3 GB,
local only), rung 42's checkpoint, and the 8B base. Where the distillation line runs is the
decision that unblocks everything else.

**5. 🟢 The box now has one checkout per person.** `~/storage/repo_leo` and `~/storage/repo_rod`.
`~/storage/repo` was **retired without a symlink on purpose** — a symlink would keep the old path
working straight into somebody else's copy. `~/storage/ESTADO.md`, regenerated by
`python3 ~/storage/estado.py`, says which checkout is at which commit and **which checkout a
running GPU process belongs to**. It also flags when a `SYNCED_FROM.md` is lying.

⚠️ **Launch inside `tmux`, one session per rung.** A 40 GB process started with `setsid nohup` is
invisible and looks like an abandoned orphan — one was killed mid-run on 17 Aug.

**6. 🔻 PLAN §5 of rung 45 was FALSE and is corrected.** It claimed rung 40 "never archived"
`optim` / `weight_decay` / `max_grad_norm`. They are in rung 40's own `RESULTS_arm.json`; only
`RESULTS.csv` omits them, and that is what got checked. **Lesson for rung 46: check
`RESULTS_arm.json`, not `RESULTS.csv`.**

**7. 🟢 The eval harness for gen-3.6 now exists**, and three things in it are new and reusable:
`frame.data.CachedFrameProvider` (UNAM holds every frame and **zero** `.mp4`), a **batch** hook on
`run_baseline` for vLLM (`batch_infer`, default OFF byte-identical), and the FP8 path — **HF
materialises the FP8 build at 43.32 GiB of a 47.37 GiB card and OOMs; vLLM loads it at 33.46.**
The 33.46 figure in `CLAUDE.md` is vLLM's, not HF's.

---

## 📋 2026-08-16 — FOR THE TEAM: submission 03 landed, and the counting question is answered

Everything here is on `main`. Detail in `experiments/19-external-count/README.md`,
`docs/viewers/backbone_scaling_viewer.html` and the decision notes linked below.

**1. 🟢 Submission 03 scored 0.5809 — top 5.** The margin over the official fine-tuned baseline
goes from **+0.0099 to +0.0620**, so the co-authorship condition is no longer a coin-flip. Current
standings: Qwen3.6 Finetuned **0.6235**, Galen 1xHigh **0.5908**, LLaVA-Med **0.5835**, us
**0.5809**, official FT baseline 0.5189. **The podium is +0.0026 away.**

⚠️ **A 7B medical model from 2023 (LLaVA-Med) is on that podium.** Backbone generation does not
order this leaderboard — see [[backbone-generation-is-not-the-lever]].

**2. 🔴 `number` is 71 % of everything rung 42 gets wrong** — 313 of its 442 errors. That is the
single largest identified pool of loss in the campaign.

**3. 🔴 The counting failure is ENUMERATION, not object size (rung 19a, RUN and CLOSED).** Asked
with the challenge's own template about **large, metallic laparoscopic instruments in our own
procedure**, both our models collapse to **0.040 at three objects** — *worse* than the 0.200 they
score on our 4 mm Clips. The mean answer saturates at ≈1 whether one object is present or nine.
⇒ external counting data is on target (**19b is licensed**) and the detector/SAM/resolution branch
is **not** the primary lever. [[counting-is-enumeration-not-small-object-perception]]

**4. 🔴 Thinking at inference is dead.** 0.4188 accuracy against 0.6485 without it, at **9.888
s/question** versus 0.515 — 19× the cost for 20 points less. But **83 % of its failed traces
contain the gold answer** (against a 27.5 % chance control), which is rung 34's hidden-state
finding seen one layer up: the model has the answer and loses it on the way out.

**5. 🟡 The 27B enumerates markedly better than the 8B** — 0.670 vs 0.317 at two objects on
identical frames, Spearman 0.692 vs 0.497. The larger backbone is better at the thing that is
71 % of our errors and still scores lower overall. Recorded, not resolved.

**6. 🟢 UNAM is now the team's machine.** One working copy at `~/storage/repo/` (no `.git`,
stamped with the commit it came from), a single `~/storage/frames_cache/` with **15,213 frames
covering all 20,000 questions**, the adapters under `experiments/*/runs/`, and the four
environments frozen in `requirements/unam-*.lock`. Refresh it by rsync; all versioning stays on
our own machines. See `SYNCED_FROM.md` on the box.

**7. 🔻 Two repo documents were lying and are fixed.** `CLAUDE.md` pinned `transformers==4.57.*`
and forbade 5.x while all four working environments run 5.x
([[transformers-pin-is-per-tool-not-global]]); `vendor/VENDORED.md` warned the SDK copy was
incomplete a month after it was fixed. Five decision notes stranded on an unmerged branch since
July are now on `main` — two of them contradicted recommendations made this same day.

**In flight:** a 401-question thinking run of the enumeration probe on UNAM GPU 0
(`~/storage/tmp/run19a_think.log`, ~2.3 h). CholecT50 is being downloaded (60 GB) to give raw
frames — the paired raw-vs-desmoke control failed today because Voxel51's frame numbering is not
CholecT50's, and the mismatch produced a spectacular false result before it was caught.

---

## 📋 2026-08-15 morning — FOR RODRIGO: four things that change what `main` says

Everything below is merged and on `main`. Detail in `experiments/43-*/PLAN.md`,
`experiments/44-*/PLAN.md`, and the rung-40 section further down.

**1. ✅ gen-3.6 IS deployable — measured, not extrapolated (rung 44, CLOSED, GO).**
The 27B in FP8 fits **one** L40S-generation card and answers in **2.84 s**:

| | measured | what main said before |
|---|---|---|
| FP8 size | **33.46 GiB** | ~41 GB (extrapolated) |
| `size_ratio` | **0.6466** | 0.756 — *from a 4.25 GiB SMOKE model* |
| loads on 1 card | ✅ vLLM 0.27.1, FP8 native | untested |
| latency | 2.84 s (worst of 11) | — |

Run on UNAM's RTX 6000 Ada (**CC 8.9 = the L40S exactly**), no challenge data, no rented pod.
⚠️ NVFP4 is **ruled out** for our submission — it needs Blackwell and the L40S is Ada.

**2. 🔴 The 5.0 s PER-QUESTION latency ceiling is not real, and our local harness is stricter
than the challenge.** `decisions/latency-budget-is-pooled.md` settled this on **2026-07-19**:
POOLED, `120 s setup + B × 5 s`, ≈ **11 s effective per question** at `B = 20`, and the forfeit
is per **batch**. But `enforce_latency` defaults **True** and the SDK gate
(`evaluator.py:218`) marks **any** response over 5.0 s as **INCORRECT**. ⇒ **our local eval can
manufacture a loss that the real evaluation would not.** This matters for anything slow —
thinking, self-consistency, higher resolution.

**3. 🔴 Rung 40 arm B's first continuation was killed — the GPU was capped, not the recipe.**
Pod `5btpl229y7kuar` ran **120.7 s/it** against the ep1's 13.1, constant from step 26. Cause
measured: `clocks.sm` **600 MHz of 3090**, `SW Power Cap: Active`, 36 °C,
`utilization.memory` **4 %**. Not MooseFS, not the recipe, not Unsloth. It reached step 252/1802
in 8 h 30 (~$26) with **zero checkpoints** — the engine uses `save_strategy="epoch"`, so nothing
is written before step 901 and it never got there. Relaunched 17:26 UTC on
`98yh51k6j6pv2h` at a healthy **13.2 s/it**.
📌 **Accept a pod in its first 10 minutes**: ignore steps 1-3 (kernel compilation — step 1 read
112 s/it on the *healthy* pod), read deltas between consecutive `[step]` lines and never tqdm's
`s/it` (a cumulative mean), and under load check **`utilization.memory`** — 32 % healthy vs 4 %
capped.

**4. ⚠️ Rung 43 was guaranteed to report a false NO-GO, and it is fixed.** Nothing in the
pipeline split on `</think>`; `screen_engine` returns `out[: answer_char_cap]` at 300 chars, so
the judge would have scored the **reasoning** and never the answer — the same defect that scored
rung 23a's smoke **0/24**. Fixed with an extractor, `answer_char_cap` 300→4000 for that arm only,
`n_truncated`, and a gate. Measured on the real 27B: **512 tokens is adequate** (greedy 6/6
closed, median trace 177 tok) and **greedy beats the vendor's sampling settings** (half the trace,
35 % faster). Rung 43 remains **STAGED, NOT RUN**.

---

> The living current-state of the project. Updated as things change. Read this + `context/INDEX.md`
> to get oriented fast. (Supersedes the older `HANDOFF.md` baseline-run handoff, kept as history.)
> Last updated: **2026-08-15** (rung 40 closed and merged to `main`).

## ⏱🔴 URGENT, measured 2026-08-15 14:05 UTC — `40_B_connector_ep23_v1` CANNOT FINISH. It is running 6.5× slower than its ETA and its own killer will stop it at ~25 %.

Read from the job's own log, `/workspace/tmp/armB_ep3.log`, on the shared volume:

```
161/1802 [5:26:30<55:07:23, 120.93s/it]   eta 3378.8 min
```

| | planned | actual |
|---|---|---|
| rate | ~18.6 s/it | **120.93 s/it** (6.5× slower) |
| ETA | ~17:40 UTC (~9 h) | **~55 h** |
| progress after 5.4 h | — | **step 161 / 1802 = 9 %** |

It was resumed at **08:27 UTC**. Its on-pod watchdog is **14 h** and the independent
killer is **15 h** ⇒ it gets stopped around **23:27 UTC at roughly step 447 / 1802 ≈ 25 %**.
🔴 **It will never reach epoch 3.** At $1.89/h that is **~$28 of a ~$35 remaining budget
spent on a checkpoint nobody can use.**

**This is a decision for its owner, not something to fix from outside** — but it needs to
be made before ~23:27 UTC. The options are to raise both stop layers (the 15 h one lives
on the operator's machine, `killer_b200.py`), to accept a partial ep2 checkpoint, or to
stop it now and keep the budget. Nobody has been asked yet; it is recorded here because
the numbers are measured and the clock is real.

### 🔑 CAUSE FOUND 15:25 UTC — the GPU is power-capped at 600 MHz. It is the HOST, not us.

🔻 **Retracts this section's first guess** ("MooseFS contention is the likely cause"). Measured on
the pod, sustained over 36 s of sampling:

```
clocks.sm            600 MHz   of a 3090 MHz max   → 19 % of rated clock
SW Power Cap       : Active
temperature          37 °C                          → not thermal
utilization.memory   2–5 %                          → not I/O, not offload
```

**`utilization.memory` at 2–5 % kills BOTH earlier hypotheses at once**: MooseFS contention and
Unsloth's gradient offload would each saturate the memory/PCIe path, and neither does. The
`utilization.gpu = 100 %` that looked alarming only means a kernel is always resident — it is
resident and crawling at a fifth of the clock.

Everything else was checked and **matches ep1 exactly**: `per_device_train_batch_size` 1,
`gradient_accumulation_steps` 16 (so a "step" is the same unit in both — 14415/901 = 16),
`max_pixels` 921600, and **both logs print the same `offload gradients` line**. The one real
difference is the silicon: ep1 ran on a **RTX PRO 6000 Blackwell _Server_ Edition** at
**13.1 s/it** (901 steps in 3.28 h, from its `RESULTS.csv`); this is a **_Workstation_ Edition**
throttled to 600 MHz at **120.9 s/it**.

⇒ 🔑 **The run is not too slow to finish — it is on a bad host.** On a healthy GPU the 1802 steps
are **~6.5 h**, comfortably inside the 14 h watchdog and **~$12**. The fix is a different machine,
not a different recipe and not fewer pods.

📌 After 6.7 h there is **not one checkpoint on disk** (first lands at step 200, `save_steps=200`),
so ~$12.7 has bought nothing recoverable yet.

⚠️ Unexplained, recorded rather than smoothed over: `power.draw` reads **800–1148 W against a
600 W limit**. Possibly chassis-level reporting or a driver bug. The clock pin and the active
`SW Power Cap` are unambiguous regardless.

⚠️ **The slowdown predates the rung-43 pod** (rented 14:02, stopped 14:15): the 120.9 s/it is the
average over the preceding 5.4 h.

## 🟢 2026-08-15 — rung 40 is CLOSED and MERGED: **both arms win, the connector wins bigger.** The live lever is the CONNECTOR, not the recipe.

> ⬆️ **This supersedes the rung-38 section below on WHICH AXIS IS LIVE**, and closes the
> "live work that is NOT on `main`" section with it. Merged as `6c2f504` (`--no-ff`, the rung is
> one revertible unit). Numbers: `experiments/40-gen36-recipe-connector/RESULTS.csv`; full reading:
> `context/40-gen36-recipe-connector/CONTEXT.md`.

Two **pre-registered** arms on gen-3.6, single-variable each, read against the **rung 38 control**
(`38_qwen36_27b_v1` ep1, proxy 0.4643 — 🔴 **not A2**, which is a different backbone). Both
evaluated on all **6252** questions, **0 timeouts, 0 errors**, paired video-clustered CIs:

| arm | change | `proxy_leaderboard` | Δ vs control | paired CI (ALL) |
|---|---|---|---|---|
| **`conn4e5`** (`40_B_connector_v1`) | connector trains @ **4e-5** | **0.5260** | **+0.0617** | **+0.0540 [+0.0312,+0.0784]** |
| `alpha16` (`40_A_alpha_v1`) | `lora_alpha` 32 → 16 | 0.4972 | +0.0329 | +0.0251 [+0.0030,+0.0498] |
| control (rung 38 ep1) | — | 0.4643 | — | — |

🔑 **`conn4e5` beats `alpha16` head to head: +0.0289 `[+0.0104,+0.0476]`, excludes zero.** It is the
**only arm that moves OOD** (+0.0312 `[+0.0008,+0.0598]`, where `alpha16` came out null), improves
**all ten cells** over `alpha16` with none negative, and repairs `alpha16`'s single regression
(`fo_class` OOD). `number_estimate` 0.3804 vs 0.3530 — the first gen-3.6 arm above A2 ep1's 0.3531,
though the CIs overlap and **that alone is not significant**.

⇒ **The rung 38 NO-GO was the RECIPE, not the backbone — and the connector is a better lever than
the recipe.** Retires the "live axis is the RECIPE" call in the section below.

🔴 **What did NOT change, and must travel with the number.** Against **A2 at matched epochs** the
point estimate moved from −0.0083 to **+0.0206**, but the CI `[−0.0063,+0.0475]` **still contains
zero**. **It is still a tie.** ⇒ *"there is no measured evidence to pick the 27B"* **survives both
arms** — and the 27B costs 3–4× per epoch to **train**. ⚠️ **Do NOT add "and it does not fit the
L40S" to that sentence** (rung 40's `CONTEXT.md:89` does, and it is a retired claim): 52.72 GiB is
the TRAINING peak; FP8 is `PASS`-validated at `capability [8,9]` = Ada = the L40S exactly, ~41 GB,
p99 1.599 s, `timed_out = 0`. **Deployability is not an argument** — see
`decisions/gen36-fails-the-8b-recipe-not-the-backbone-test.md` §1. Being behind A2 ep2/ep3
is **not** a loss and must not be quoted as one: those are 2 and 3 epochs against our 1.

### State of the ladder right now

| | state |
|---|---|
| **rung 40** (recipe + connector) | ✅ **CLOSED — both arms WIN, `conn4e5` wins bigger.** Merged to `main`. |
| **rung 40 arm B cont** (`40_B_connector_ep23_v1`) | 🔵 **RUNNING** — epochs 2+3 continued from Leo's ep1 **merged** model, `num_train_epochs=2`. See the caveat below. |
| **rung 43** (thinking at inference) | 🟡 **STAGED, NOT RUN — CHAINED to arm B's pod.** Work on `task/43-thinking-at-inference`. 🔴 Launch only when `orena-rung40-armbB-ep3` (`5btpl229y7kuar`) has finished and stopped itself. See below. |

### ⛓️ Rung 43 is CHAINED to `orena-rung40-armbB-ep3` — do not launch it before that pod stops

Rung 43 is inference-only, so it is **orthogonal to the training** and does not have to wait
for correctness reasons. It waits on the pod `orena-rung40-armbB-ep3` (`5btpl229y7kuar`, run
`40_B_connector_ep23_v1`) finishing and stopping itself. Full detail in that rung's `PLAN.md`
§"LAUNCH BLOCKERS", on branch `task/43-thinking-at-inference`.

1. 🔻 **Disk — NOT a blocker. An earlier version of this section said it was, and that was
   wrong.** The `min_free_gib = 60` guard lives **inside `merge_arm()`** (`remerge.py:72`), which
   the chain reaches only at **step 3 — after step 2 has deleted 52 GB**. 43 + 52 = ~95 > 60, so it
   passes. **The chain is self-clearing by design**; that is the serial pattern `remerge.py` exists
   to encode. It needs **~8 GiB free at launch, not 60.** Still worth one look (ep2/ep3 write to the
   same ~670 GB quota): confirm **> ~10 GiB** with `remerge.free_gib()` (`du -sx`), 🔴 **never
   `df`** — it reports the MooseFS cluster at 1.4 PB and a chain already died `Disk quota exceeded`
   trusting it.

2. ✅ **RESOLVED 2026-08-15 by `keep_conn_merge`.** The collision was **verified from the running
   job's own log** — it declares `base_model = .../40_B_connector_v1/merged`, the exact path step 2
   deletes, on the same volume, mid-run. But the deletion was never necessary: `du -sx` measures
   **550.1 GiB used of 670 GB ⇒ ~120 GiB free**, not the ~43 GiB assumed, so **both 52 GB merges fit
   at once**. `keep_conn_merge=True` turns step 2 into an echo and merges `alpha16` alongside; the
   only surviving `rm -rf` targets our own `40_A_alpha_v1/merged`. Defaults False ⇒ byte-identical
   for anyone who renders without asking. ⇒ **rung 43 can run in full with zero risk to that job.**
   Original wording kept below for the record:

   > 🔴 **`chain_probe.py` step 2 is `rm -rf` on `conn4e5`'s 52 GB merge, which may be the very
   model that pod is training FROM.** Per `1ab2509`, `40_B_connector_ep23_v1` takes **Leo's ep1
   merged model as its `base_model`**, and the volume is shared across pods. **Not confirmed
   either way** — his run may read a pod-local copy. **Verify, do not assume:** resolve his
   `base_model` against `experiments/40-gen36-recipe-connector/runs/40_B_connector_v1/merged`
   before launching. If they are the same, launching rung 43 **kills an ~11 h training run.**

### 📋 Attempt of 2026-08-15 14:02–14:15 UTC — staged, gated, then the pod was stopped by a user

A pod `rung43` (`hqnyx4oqf9fkwn`, **RTX PRO 6000 Blackwell Server, 96 GB, sm_120**, 1 GPU,
$2.09/h) was rented and prepared. **Stopped by a user at 14:15:23 UTC** — not by the chain and not
by any automation; total spend **~$0.45**. **Nothing destructive ran**: `conn4e5`'s merge is intact
and the ep23 job was never touched.

What the attempt established, and what is reusable next time:

- An **isolated checkout at `/workspace/repo_rung43`** (code only), with
  `experiments/40-gen36-recipe-connector/runs` **symlinked** to `repo_leo`'s. This exists because
  `repo_leo` is on the old branch **and the live ep23 job reads its code from there** — switching
  its branch would break a running job. Do not `git checkout` inside `repo_leo` while that runs.
- **Import gate PASSED** on the pod: `papermill 2.7.0`, `frame.metrics` (incl. `leaderboard_proxy`),
  `eval_arm`, `screen_engine`, `/workspace/orena-data` present, GPU 0 MiB used of 97887.
- The pod has **no git credentials** (`could not read Username for 'https://github.com'`) ⇒ code
  arrives by `scp`/`tar`, not `git fetch`.
- 🔴 **The on-pod self-stop could not be armed.** `chain40`'s trap reads an API key from
  `/workspace/tmp/leo_runpod_key`; staging it was blocked, so `stop_pod_on_exit` cannot work from
  this operator. **Use layer 3 instead** — an off-pod killer with a hard deadline, which
  `experiments_segment/NOW.md` argues is the layer that matters anyway since on-pod layers share a
  failure domain with the pod. A ready one is in the session scratchpad (`killer_rung43.py`:
  deadline + end-marker, retried stops, status read-back).

Nothing is lost by waiting: the adapters are **383 MB** each and regenerate either 52 GB merge in
~10 min (a 135× saving), so the merges are disposable and only the adapters must survive.

3. **Appending rung 43 to that pod's chain is possible but NOT ours to arrange.** Three stop
   layers are armed (`experiments_segment/NOW.md`): the chain's `trap … EXIT`, an on-pod watchdog
   (14 h), and **an independent killer on the operator's machine** (`killer_b200.py`, 15 h hard
   deadline, RunPod API only). The first two are Rodrigo's to move; **the third is not on the pod
   at all** and is deliberately dumb — no log parsing, no liveness heuristics — so **it stops the
   pod at 15 h whether or not rung 43 is mid-run**, and this rung needs 1 h 45 – 2 h 30. 📌 And
   **`pkill` on a chain fires its `EXIT` trap and stops the pod** (`1ab2509`), so there is no
   swap-the-job move either. ⇒ chaining needs all three layers moved by their owner; it is a
   **conversation with Rodrigo, not an action we can take.** A pod rented for someone else's rung
   is read-only for us.

⚠️ **Also closed before launch, and already written on the branch:** latency is now a **declared
read** (`infer_latency_p99_s`, `timed_out` vs the 5.0 s cap). The thinking arm takes
`max_new_tokens` **64 → 512** against a rung-40 p99 of 1.5–1.8 s **at 64** — so this rung can
produce an accuracy WIN that cannot ship, and the rule for calling that is pre-registered.
| **SEGMENT track** | 🔵 Open — rung 01 viability, `experiments_segment/NOW.md` is its own live state. |

⚠️ **The arm B continuation is NOT a 3-epoch run, and the difference is load-bearing** (`1ab2509`).
Leo's cosine annealed to lr 0.0 by the end of ep1, so the continuation starts a **fresh cosine**: one
3-epoch run has ONE schedule over 2,703 steps, this has TWO over 901 + 1,802. They differ from step
one and neither contains the other. ⇒ it is comparable **against Leo's ep1 and against itself**;
comparing it to A2 ep3 (a single 3-epoch cosine) carries this caveat **in writing**. It also inherits
the merge that **dropped the trained connector bias** (declared deviation, rung 40 `PLAN.md` §3d).

---
## 🔴 2026-08-13 (close) — rung 38 is CLOSED: the A2 recipe transplanted to gen-3.6 LOSES. The live axis is the RECIPE.

> ⬆️ **This supersedes the "arm is STAGED" section below.** The arm ran, was evaluated on all 6252
> questions, and was analysed with paired video-clustered CIs. Everything is committed on `main`
> (`d03836f`, `002f01a`, `b06de28`). Full verdict:
> `context/decisions/gen36-fails-the-8b-recipe-not-the-backbone-test.md`.

**Measured, epoch- AND step-matched** against A2 ep1 `checkpoint-901` (proxy 0.4986 / `bucket_mean`
0.5592) — our arm ran 901 steps. Numbers from `experiments/38-gen36-ft-screen/RESULTS.csv`:

| cell | n | Δ (27B − A2 ep1) | 95 % CI | excludes 0 |
|---|---|---|---|---|
| **ALL** | 6252 | **−0.0334** | [−0.0642, −0.0027] | **YES** |
| ID | 2252 | −0.0362 | [−0.0800, +0.0021] | no |
| `proxy_leaderboard` | — | −0.0344 | — | pre-registered read FAILS |
| `margin_OOD` | — | −0.0255 | — | pre-registered read FAILS |

🔑 **The loss is narrower than the headline.** The **ID cell — the one the proxy is made of — does
NOT exclude zero.** All the damage sits in `object_recognition`; `aggregation` is flat and `number`
is a **dead tie**. The arm wins 717 questions A2 gets wrong. **It did not get worse at counting; it
got worse at naming objects.**

🔑 **The finding worth more than the verdict:** given the identical SFT, the 27B **fixes the
frequency prior** (L1 to gold **14.8** vs the 8B's **27.4**; the 8B answers "1" 55.1 % of the time
when gold is 35.2 %) **and gains nothing for it** ⇒ the `number` bottleneck is **perceptual, not
distributional**. Amends [[counting-has-two-failure-modes]].

🔻 **Two deployability claims made while reading this were WRONG** and nearly closed gen-3.6 on
smoke: the **52.72 GiB is the TRAINING peak**, not the serving footprint (FP8 is `PASS`-validated at
capability [8,9] = the L40S exactly, ~41 GB — **it fits**); and the **4.81 s was `max`**, an outlier
among 6252 — **p99 is 1.599 s, `timed_out = 0`**. ⇒ **deployability is not an argument.** Rank 1 on
the leaderboard is a `Qwen3.6 Finetuned` at 0.6235 — somebody serves this inside the budget.

⇒ **This closes the RECIPE TRANSPLANT, not the backbone.** `lr 2e-4` is the **8B's** optimum and was
never ported. 🔴 **Do NOT scale this recipe to 3 epochs** — no resume is possible (cosine is already
at lr 0 by step 901), so it costs ~9.5 h from scratch to scale a recipe already shown to sit wrong.

### State of the ladder right now

| | state |
|---|---|
| **rung 38** (gen-3.6 backbone screen) | ✅ **CLOSED — NO-GO at A2-verbatim.** Trained, evaluated, analysed, committed. |
| **rung 39** (connector LoRA) | 🟢 **GATE RAN AND PASSED on an A100** (`3f932db`, 2026-08-13 17:46). Arm not run yet. |
| **gen-3.6 recipe design** | 🔵 **IN PROGRESS on a branch, not merged** — see below. ⬆️ **SUPERSEDED 2026-08-15: it became rung 40, which is closed and merged.** |

### 🟢 The rung-39 gate PASSED — the connector IS reachable in ms-swift, measured

`3f932db`. Two legs, five real optimiser steps each:

| leg | `freeze_aligner` | total | `n_llm` | `n_vit` | `n_aligner` | orphans |
|---|---|---|---|---|---|---|
| subject | false | **736** | 504 | 216 | **16** | 0 |
| control | true | 720 | 504 | 216 | **0** | 0 |

736 = A2's own 504 + 216 **plus 16 connector tensors** (`merger.linear_fc{1,2}` and
`deepstack_merger_list.{0,1,2}.linear_fc{1,2}`, each with `lora_A` + `lora_B`). Coverage extended,
never replaced. The control returning 0 proves the 16 come from the change, not from the classifier.
`grad_norm` 139 → 14 and loss 2.41 → 0.28 ⇒ **real gradient, not an `rc=0` silent no-op.**

🔑 **This answers the open question in [[the-merger-is-unreachable-by-default]] §4: an explicit
target DOES survive ms-swift's intersection.** ⚠️ **Unsloth remains unmeasured** — which is the gen-3.6
lane, not this one.

⚠️ **`evidence_ep1/` and the rescued `RESULTS_eval_27b_ep1_*` are ONE run, not two** (`results.csv`
byte-identical; the duplicate was removed). **There is no independent replica of this arm, and no
seed has ever been repeated in the campaign.**

### 🔵 Live work that is NOT on `main` — read the branch before reasoning about gen-3.6

> ⬆️ **SUPERSEDED 2026-08-15 — this section is now HISTORY, do not act on it.** The branch
> `task/gen36-recipe-and-connector` was **merged to `main` as `6c2f504`**. Everything it describes
> (the two decision notes, the complete rung 40, the two Unsloth facts) is on `main`; the arms have
> since RUN and both won. There is no longer anything to check out. Read the 2026-08-15 section at
> the top instead.

Desk research on **what recipe to use for the gen-3.6 ladder** (zero GPU) is in progress on branch
**`task/gen36-recipe-and-connector`**, pushed but **deliberately not merged — the task is not
finished**. It carries two decision notes and a section added to `context/39-connector-lora/CONTEXT.md`.

🔴 **If you are about to propose a gen-3.6 recipe, a connector experiment, or a learning-rate change,
`git checkout task/gen36-recipe-and-connector` first.** Those notes retire the `lr 2e-4`-is-too-high
hypothesis and change what the connector work looks like on this backbone. Proposing from `main`
alone will re-derive things that are already settled there.

**As of 2026-08-13 (close) that branch carries a complete rung 40**: pre-registered, **two gates
already RUN** on UNAM with no challenge data and no rented GPU, **both arms smoke-passing**, and
**the eval written before the arms run** — the gap that cost rung 38 a night. What it needs is a
**≥80 GB GPU**, and nothing else.

Two measured facts on that branch bind anyone touching Unsloth here, not just gen-3.6:
**`save_pretrained_merged` keeps a `modules_to_save` weight and silently drops its bias**, and
**`optimizers=` no longer exists in trl 0.24** — `SFTTrainer` swallows it in `**kwargs` with no
error and builds HuggingFace's own optimiser at 5e-5 instead.

---
## 🟢 2026-08-13 — the arm is STAGED on the pod, and the real numbers are better than the estimate

> Everything below is measured on the pod with **real challenge data** — the first time this
> pipeline has touched it. `experiments/38-gen36-ft-screen/RESULTS_smoke_pod_27b.json`.

1. 🟢 **One epoch is ~5.9 h, not ~9.8.** Sustained **23.5 s/it** (step 1 was 230.6 s: compile and
   warm-up), × 900.9 steps. The extrapolation from scaling the 8B was wrong **by excess**, which
   widens the window rather than closing it. ⚠️ One sustained-step measurement; treat as indicative.
2. 🔴 **Peak VRAM 52.64 GiB — the arm does NOT fit a 48 GB card.** It needs ≥80 GB. Measured, not
   estimated, and it decides which GPU to rent.
3. 🟢 **`merger: 0` holds on the real 27B**, not just the 2B — `visual` 108, `language_model` 496,
   62.2 M trainable. [[the-merger-is-unreachable-by-default]] is confirmed at scale.
4. 🔴 **`df` LIES about RunPod volumes.** It reports the MooseFS cluster (314 TB free); the volume
   carries a **quota** (~640 GB) invisible to it. The run died mid-merge on `Disk quota exceeded`
   after I had told legokna there was no space problem — **his instinct to prune was right and I
   talked him out of it.** Fixed: partial merge deleted (47 G) and `qwen3-vl-32b` pruned (63 G,
   NO-GO generator) plus the substitute judge (7.6 G). **631 → 514 GB, ~126 GB free.**
5. 🟡 **Unsloth's own documented install produced a CPU-only torch** here
   (`--torch-backend=auto` → `torch 2.11.0+cpu`, `torchvision 0.2.0`) on a machine with an A100.
   Fixed with an explicit `--torch-backend=cu128`. 📌 Neither install route is reliable on its own —
   what caught it both times was **asserting `torch.cuda.is_available()` and the capability after
   installing**, which is what cell 1 does.
6. 🟢 **Staged and ready:** branch `task/rung38-gen36-screen` in `/workspace/repo_leo`, the 27B
   cached (52 G), data verified (14,415 rows + 15,213 frames), env functional. ⚠️ The env was built
   against an A100 (sm_80); cu128 wheels cover sm_120 too, but that is untested.
7. 📌 **`/workspace/repo` was touched and restored.** It is the *shared* checkout, on
   `task/r3-rung16`; I moved it to main before noticing, then put it back at `6202909` with its
   untracked file returned. Work happens in `repo_leo`, on a branch.

## 🟢 2026-08-12 (close) — rung 38 is BUILT AND READY TO LAUNCH. Nothing technical is unresolved; it needs a pod and someone to press go.

> **Handoff.** legokna had to step away before launching. `experiments/38-gen36-ft-screen/01_gen36_screen.ipynb`
> is the runbook: environment, path check, smoke, arm, FP8, eval. Anyone on the team can run it.

| link in the chain | state |
|---|---|
| trainer (ms-swift → Unsloth) | ✅ forced and resolved — [[ms-swift-cannot-train-gen35]] |
| pipeline load→LoRA→train→merge | ✅ 6/6 stages, `RESULTS_smoke_unsloth.json` |
| eval reads an Unsloth merge | ✅ PASS, `RESULTS_eval_path.json` (2 bugs fixed) |
| **FP8 delivery** | ✅ **PASS**, `RESULTS_fp8_llmcompressor.json` |
| the arm | ⬜ **built, never run** — needs the pod |

🔴 **The one unknown left is the 27B's `s/it`.** Our 8B ran 11.56 s/it; at 30–40 an epoch is
7.5–10 h. Cell 3 of the notebook measures it, and the decision to launch cell 4 should be taken
against that number, not against an estimate.

🟡 **The subject is still open and it is one line.** `Qwen3.6-27B` (chosen — needs the FP8 step,
now validated) vs `Qwen3.5-9B` (~18 GB bf16, **no FP8 step at all**, nearly size-matched to our 8B
so a cleaner read). The evidence pulls both ways: rank 1 is a fine-tuned **3.6**, but its advantage
is concentrated **OOD**, which is what the generation buys and a 9B buys too.

## 🔴 2026-08-12 (night) — ms-swift CANNOT train gen-3.6, Unsloth can, and the connector is unreachable in BOTH. Plus a second machine.

> All of it measured with **zero training GPU**. Details: [[ms-swift-cannot-train-gen35]],
> [[unsloth-is-the-route-to-gen35]], [[the-merger-is-unreachable-by-default]],
> `context/UNAM_SERVER.md`, `experiments/38-gen36-ft-screen/`.

1. 🔴 **`G-VIABILITY` V2 FAILS: ms-swift 4.4.1/4.4.2 have no `MODEL_ARCH_MAPPING` entry for
   `qwen3_5`.** Control `qwen3_vl` returns its three prefixes; the subject returns **no keys at
   all**. Since rung 06 our recipe *is* `--freeze_vit false`, which works by adding `vision_tower`
   to the LoRA targets — from that table. ⇒ **ms-swift cannot express our recipe on gen-3.5/3.6.**
   Not transformers' fault (5.5–5.15 all ship `qwen3_5`), not the version cap (5.9–5.12 satisfies
   both), not size — **every** gen-3.5/3.6 config carries `model_type=qwen3_5`, so the gap is
   generation-wide. 📌 The gate did its job: ~1 h of env build instead of ~3 GPU-h and a 56 GB
   download to hit the same wall.
2. 🟢 **Unsloth can, and the NO-GO is AMENDED — legokna was right, and had proposed it three
   times.** `CONSTITUTION.md`'s first reason (*"lagging Qwen3-VL multimodal support"*) is **false**:
   Unsloth documents SFT and vision RL for Qwen3-VL up to 32B/235B, and is the only known trainer
   for gen-3.5/3.6 with the vision tower. Single-GPU is mitigated. 🔑 **The third reason stands and
   is now the binding one:** rungs 02–35 are ms-swift artifacts, so an Unsloth arm confounds
   framework with everything else. **That binds on ATTRIBUTION, not on a candidate search** — for
   the leaderboard a confounded win is still a win, so it is a reason to *label* the result, not to
   refuse the run.
3. 🟢 **The pipeline is measured, not assumed.** Six-stage smoke on `Qwen3.5-2B` (same class ⇒ same
   code path as the 27B), public CholecT50 frames, zero challenge data: load → LoRA (0.519 %
   trainable) → 2 steps (loss 2.441 → 2.276, peak 4.51 GiB) → adapter 63.2 MB → merge 4.25 GiB.
   **All six passed.**
4. 🔑 **THE FINDING WORTH KEEPING: the ViT→LLM connector is unreachable BY DEFAULT in BOTH
   trainers.** ms-swift: 720 = 504 + 216 + **0 aligner**. Unsloth with everything switched on:
   visual **96**, language **186**, **merger 0**, deepstack **0**. ⚠️ And the "maybe it is named
   differently" objection was checked *before* the claim: a module census of Qwen3.5-2B shows
   `model.visual.merger.linear_fc{1,2}` present under the same name. **The cause is structural** —
   the connector's path carries no `attn`/`mlp` token, and every generic matcher requires one, so
   `all-linear` is literal in neither framework. Unsloth's own docs confirm *"no documented way"*
   to target it. ⇒ **rung 32's null now has a mechanism, switching framework does not unlock the
   connector, and rung 39 must pre-register an explicit `target_modules` plus a blocking
   reachability gate.**
5. 🟢 **Second machine online: `hpclab-RTXA6000` (UNAM).** 2× RTX 6000 Ada 48 GB, 96 cores, 502 GB
   RAM, 17 TB. 🔑 **Compute capability 8.9 — the same architecture as the eval L40S**, so the two
   things the Blackwell dev pod could not do become possible: **FP8 kernels build**, and there is
   finally a **p99 proxy** for the item `THE_MAP.md:252` calls *"the only item BOTH documents
   demand"*. Faithful proxy, not identical silicon. 🔴 **No challenge data goes there** until a
[redacted]
   Encryption-at-rest does not fix it (the data is decrypted for the whole run). 📌 And RunPod is
   not a gold standard either — no DPA, their admins hold root; the gap is paperwork.
6. ⬜ **What can still kill rung 38, in order:** (a) **the eval path** — `screen_engine.py` has
   never been pointed at an Unsloth-merged checkpoint, and without it there is no verdict however
   well training goes; (b) **the FP8 delivery path** — a 27B only fits the eval GPU quantised and
   we have never quantised a merge; (c) the arm itself.
7. 🔑 **AND THE REASON THAT OUTRANKS ALL OF THE ABOVE, which was already in the repo and went
   unquoted all session: rank 1 is `Qwen3.6 Finetuned`, 0.5653** ([[local-eval-vs-judge-calibration]]
   table). We are rank 11 at 0.5288, **+0.0365 behind**. [[backbone-generation-is-not-the-lever]]'s
   amendment named as its first reopening condition *"verified confirmation that a leading entry
   **fine-tunes** a gen-3.6 model"* — **a leaderboard row named after the model, at rank 1, is that
   confirmation.** ⇒ rung 38 is not a speculative screen; it follows the leader's demonstrated
   route. ⚠️ The row does not disclose size, recipe, or whether the vision tower was trained.
8. 🟡 **Recorded, not chosen: Qwen3.5-9B may be the better first arm.** The smallest gen-3.6 is the
   27B, so generation and size move together and delivery needs FP8. A 9B is **nearly size-matched
   to our 8B**, trains on one card (22 GB) and **serves in bf16** — no new delivery path at all.
   And the only competitor known to beat us runs a gen-3.5 **4B** (410 vs 343 on `aggregation`,
   same 2,000 questions). legokna's call: **stay on 3.6 for now, 3.5 is the fallback.**

## 🟢 2026-08-12 (pm) — the lane is RE-RAILED: the axis is fine-tuning, the first rung is 38, and the recipe queue is shared between backbones

> Answers the 🔴 consequence logged below (*"our lane is now empty"*). This is a **plan**, not a
> verdict: nothing here is measured yet. Owner: legokna.
> 📌 **The plan lives in `context/NEXT_STEPS.md`** — the full version with the gates, the control
> numbers and the execution order. This entry is the summary; that file is the source.

1. 🔑 **The axis is fine-tuning, and the record supports it.** Every input-side / external-module
   lever we have paid for came back marginal or negative — 12 (−0.026…−0.056), 14 (NULL, CI
   [−0.0038, +0.0195]), 18 (−0.0003), 24 (CI includes 0), 10 (k=16 **harms** OOD −0.043), 16c
   (collapses to 1 point), 30 GRPO (−0.0094), 35 NTL (own veto fired), 37 overlay (breaks 13, fixes
   1). **The largest move of the campaign was a training flag**: lr 2e-5 → 1e-4, proxy **+0.0480**,
   21 of 30 paired cells significant and all pro-arm. We have swept hyper-parameters; we have never
   worked the training surface deliberately.
2. 🔴 **The ViT↔LLM connection has NEVER been trained — not once, in 30+ rungs, and not by
   decision.** `--freeze_aligner false` was measured in rung 32 and is a **no-op**: the merger's 8
   `Linear` layers (`model.visual.merger.linear_fc{1,2}` + `deepstack_merger_list.{0,1,2}.linear_fc{1,2}`)
   are unreachable through `--target_modules all-linear`, which is **hardcoded** at
   `experiments/06-vit-lora/_models/vit_lora_train.py:103` and has never changed.
   `RESULTS_reachability.csv` measures it: **720 tensors = 504 LLM + 216 ViT + 0 aligner**, both
   legs. In Qwen3-VL the merger is not an ordinary projector — DeepStack wires it into the **first
   3 LLM layers** ([[viT-swap-nogo]]) — so it *is* the connection.
   ⚠️ Honest counter-weight: for `number`, rung 34 points at the **projection into tokens**, not the
   encoder (a linear probe at layer 24 scores 0.5264 against the model's own 0.4680).
3. 🟢 **Rung 38 = the fine-tuned gen-3.6 screen, in two stages.** Not a re-proposal:
   [[backbone-generation-is-not-the-lever]] was **amended 2026-08-09 at Rodrigo's instruction** —
   the NO-GO is **zero-shot only** and stands **OPEN**, with the reopening test named and costed
   there. **Stage 0 = `G-VIABILITY`** (~1 h, blocking, near-zero GPU): which gen-3.6 multimodal
   models exist and which is smallest; does ms-swift register the class; which venv (rung 23a
   measured that the class **cannot load under `transformers` 4.57** — `AutoConfig` →
   `KeyError: 'qwen3_5'`); does LoRA fit (bf16 27B = 55.6 GB, **FP8 has no sm_120 build**);
   `enable_thinking=False` applied (the 23a default template scored **0.0000**). Fail ⇒ stage 1
   never runs and the rung closes as *no route to training* — a publishable result for ~1 h.
   **Stage 1** = one epoch of the A2 recipe, single variable = the backbone, against the free
   already-scored control **`21_lr_2e4_v1/checkpoint-901`**: proxy **0.4986**, `bucket_mean`
   **0.5592**, `margin_ID` **+0.1781**, `margin_OOD` **+0.1643**, `fo_class` macro-F1 ID **0.6112**.
4. 🔑 **THE ARGUMENT THAT REORDERS THIS: the A2 recipe is not known-good on another backbone.**
   `lr 2e-4` is the optimum found *for the 8B over these 14,415 rows*, and the harness's own
   documented failure mode is that **the LR optimum moves with the model and with dataset size**.
   ⇒ the screen at A2-verbatim measures a **floor** of the gen-3.6, not its ceiling. Two
   consequences, written **before** the number exists so they cannot be retrofitted: a **narrow
   NO-GO does not close the axis** (it says *"not for free"*), and a **GO is collected by running
   the queue below on the new backbone**, not by adopting it as-is.
5. 🟢 **The recipe queue is therefore obligatory in BOTH branches — the verdict picks the subject,
   not the list.** NO-GO ⇒ it runs on the 8B; GO ⇒ it runs on the gen-3.6. Ranked by evidence, not
   appetite:

   | # | lever | GPU | what holds it up |
   |---|---|---|---|
   | **1** | **connector**: `--target_modules` reaches the merger, `--freeze_aligner false` | ~10.6 h, **preceded by a zero-GPU gate** | never trained; 720 = 504+216+**0** measured; DeepStack → first 3 LLM layers |
   | **2** | **epochs 3 → 6**, **fresh** cosine | ~17.4 h (5,405 steps) | never swept; `C_epochs` built and **killed mid-run**, never scored; every arm still rising at ep3 |
   | **3** | **`vit_lr` 5e-4** (rung 27's `B_high`) | ~10.6 h | `A3` measured that **cooling** the tower hurts (−0.0278, 4/30 cells, all pro-control); heating it is the one direction A3 does not refute |
   | **4** | **rank 8 → 32** (`B_rank`) | ~11 h | *"PASSES the pre-registration, fails the CI"* — 0/30 cells exclude 0; the README retracts *"rank is dead"* |
   | **5** | **lr above 2e-4** | ~10.6 h | 🔴 **not recommended**: 2e-5→1e-4 bought +0.0480, 1e-4→2e-4 only +0.0203 **and macro-F1 ID fell 0.6906 → 0.5474**. Diminishing returns *with class damage* |

   Default recommendation when the choice comes due (**at rung 38's verdict, not before**): **#1,
   the connector** — the only completely untouched lever, and its `G-REACH` gate is **zero GPU**
   (adapt `experiments/32-aligner-unfreeze/_tools/reachability_smoke.py`, which already
   parameterises `target_modules` and `freeze_aligner` — the engine chain does not — and require
   `n_aligner > 0`, `n_orphans == 0`).
6. ⚠️ **Mechanics for #2 that have sunk controls before:** 3→6 epochs is **not a `resume`**. A2's
   cosine anneals to **lr 0.0** at step 2703; restoring `scheduler.pt` learns nothing and hands you
   a null control that flatters whatever it is compared against. It must be a **fresh** cosine over
   6 epochs — the same trap the team's August plan already flags in red for the paired control of
   Rodrigo's step 9.
7. ⚠️ **Reviving rung 27 is a change of FACTS, not a re-litigation.**
   [[august-plan-closes-the-ladder]] closed it UNRUN and installed *"a rung is revived because the
   plan asks for it, never because it exists."* It is revived here because step 7 died, step 8 was
   cancelled, VCD died in step 4 and the lane went empty on 2026-08-12. Recorded so the rule is not
   quoted against this. 📌 And rung 27's `PLAN.md` is **stale**: it is written against `lr 1e-4`
   while the operative base is 2e-4 — it needs an amendment **before** a verdict, not after.
8. 🟡 **Also being written, zero GPU: `context/FINE_TUNING.md`** — what each flag actually touches
   on *our* model (ViT SigLIP2 → merger ×4 DeepStack → LLM), where LoRA is injected and why
   `all-linear` misses the merger, the training loop, **the operative A2 recipe and its measured
   drift** (`lr 2e-4` is declared **only in prose**; six dataclasses contradict each other and
   **rung 24 trained at 1e-4 against a baseline declared at 2e-4**,
   `experiments/24-geometric-aug/_models/horizontal_flip.py:168`), and the table of what has never
   been trained. Documented, **not** fixed — pinning the recipe in code is a separate change.
9. 🔴 **What does not fit.** ~20 days to pre-eval, week 3 (15–21 Aug) is Rodrigo's GPU for step 9,
   and each full arm is 10–17 h. **The screen plus ONE or TWO of the queue.** Not five.

## 🔴 2026-08-12 — G-BOUNDARY was adjudicated and FAILED. Rung 37 is dead, the SAM block is closed, and our lane is empty

1. 🔴 **The gate FAILS. Step 7 dies with only the export spent, exactly as pre-registered.**
   **B1 = 0.9714** (CI [0.914, 1.000]) **passes**; **B2 = 0.3529** (CI [0.206, **0.529**]) **fails
   with its whole interval under the 0.70 threshold**. Robust to how `unadjudicable` (5/40 = 12.5 %)
   is handled — a mark added to the viewer after pre-registration, so its treatment was never
   fixed; both readings are recorded and neither is load-bearing.
   ⇒ **SAM finds the foreign objects and does not delimit them.** Numbers + the raw per-frame marks:
   `experiments/37-attention-vs-masks/RESULTS_gboundary.json` (committed **outside `runs/`**, which
   is gitignored). Verdict: [[g-boundary-fails-on-precision-not-coverage]].
2. 🔑 **It failed where nobody was looking, and that is the lesson.** **Metallic clips scored
   `covered` 4/4.** The objection that consumed 2026-08-11 — first as the retired `r = −0.17`, then
   as the correction that replaced it — **decided nothing in either form**. 📌 A retired number and
   its replacement can *both* be beside the point. What died instead is this rung's own enabling
   fact, *"SAM does not merge tissue with foreign objects"*; B2 = 0.3529 refutes it directly.
   Worst class `silicone_loop` (clean 0/5), best `specimen_bag` (3/4).
3. 🔑 **The gate never measured PRECISION** — legokna, on reading the adjudicated frames. B1 asks
   whether a mask exists over the object, B2 whether it is sharp; **neither asks how many OTHER
   masks are present**, and SAM emits **13–50 instances per frame** over the whole scene. ⇒ a VLM
   handed these masks gets **references without identity**, which is exactly the overlay's measured
   failure (identity substitution, not blindness). ⚠️ A limitation of the **pre-registration**, not
   of the adjudication: **any future masks-as-input gate needs a precision clause.**
4. ⏸️ **The SAM block is CLOSED, on TIME rather than on merit.** legokna re-derived the whole route
   independently and reached the standing verdict: *"not blocked because SAM has no bearing on the
   model — blocked because doing it properly needs more time than the challenge leaves. It is a
   full CoVT for this problem, worth doing after Sep 8."* Amended into
   [[sam-adaptation-has-no-route-to-points]], including the correction that **leaving ms-swift is
   not required and probably neither is LoRA** (rung 35 shipped a custom-loss adapter with both) —
   the real blockers are latency and gradient share, not the harness. **Do not re-open before
   Sep 8.**
5. 🔴 **CONSEQUENCE NOBODY HAS ACTED ON: our lane is now empty.** Week 2 (8–14 Aug) held step 7 —
   dead. Week 3 (15–21 Aug) is Rodrigo's GPU for step 9 (GRPO + the paired control), and our
   parallel slot reads *"VCD evals"*, but VCD died in step 4 and step 8 was cancelled 06-Aug.
   🔑 And `experiments/27-vit-lr-decouple/` has `PLAN.md`, `_models/` and `_tools/` **and no
   `RESULTS.csv` — pre-registered and never run**, on the ViT, the one perceptual lever with
   measured traction (rung 06 held the campaign's best 17 days; `fo_class` +17.8 pts recall on
   `needle`). **Unevaluated against what we know today.** Not scheduled, not decided.

## 🔴 2026-08-11 — the 36 collided, step 7's export RAN, and its gate fails a dry run

1. 🔴 **TWO rungs 36 reached `main`.** Ours (`36-attention-vs-masks`, pushed 2026-08-09, with
   `NOW.md:11` reading *"do not reassign the number; take 37"*) and Rodrigo's
   `36-clip-sponge-probes`, pushed 2026-08-10 without seeing that line. **Ours was first; legokna
   yielded the number** — his carries committed results, ours carried only a design.
   ⇒ **`experiments/37-attention-vs-masks/`.** The announce step never ran; that is the whole cause.
2. 🟢 **Step 7's export RAN** — `runs/37_masks_v1/`, 45 frames: the N=40 `G-BOUNDARY` sample,
   **5 per class across all eight** foreign-object classes (Clip, Sponge, External drain,
   Specimen, Specimen bag, Silicone loop, Needle, Gallstone — the whole vocabulary, not the Clip
   third C1 used), plus the **5 C2 pairs below the 0.90 floor**. Manifest committed **outside
   `runs/`** as `SAMPLE_37_masks_v1.json`. Nothing adjudicated, no attention read.
   🔑 The five pairs reproduce their persistences **exactly** (0.611, 0.571, 0.615, 0.769, 0.367),
   which re-derives `RESULTS_controls.json` a third time.
3. 🔴 **`G-BOUNDARY` fails a dry run on data we already had, and fails for the wrong reason.**
   Scoring the 8 C1 frames already adjudicated in legokna's mask-reading pass:
   **B1 = 0.625** (fails the ≥0.70 point estimate; CI [0.250, 0.875] fails the >0.50 clause) and
   **B2 = 0.800** (CI [0.400, 1.000] fails it). **All three `covered` failures are metallic clips**,
   and the adjudicator's own note on two of them hedges on whether the object is visible at all
   (*"no sé si es mi sesgo"*, *"si es que los hay"*). Restricted to what that eye resolves (white
   clips, gauze, instruments) `covered` is **5/5**.
   🔻 **CORRECTED same day, before this was acted on:** the argument was first written citing
   **r = −0.17** for the untrained eye, i.e. *"the gate dies from the instrument"*. **That number is
   RETIRED** ([[gold-is-signal-model-underuses-it]]) — range-restricted to gold 3–6, superseded by a
   blind full-range pass scoring **r = +0.7230** with only **2.8 % "cannot tell"**. **The eye reads
   these frames well**, so the expected `unadjudicable` rate is LOW and the gate is in better shape
   than that first reading claimed. The surviving concern is narrow, first-hand and about
   **metallic clips only**. 📌 Textbook [[inference-still-load-bearing]]: a retired number was
   re-cited from `CAMPAIGN_LOG` §10 without checking whether it still stood.
   ⚠️ That sample is clip-heavy by construction and does **not** predict the stratified N=40.
   🔑 **What it does show: the gate's verdict tracks the metallic-clip share of its sample — a
   sampling choice, not SAM's quality.** The gate is NOT amended; the export presumes no outcome.
4. 🟢 **SurgΣ-DB: the repo's "the images are NOT in the package" is FALSE.** Verified against the
   HF API and by streaming `dense_prediction.tar.gz` (3.09 GB, 165,582 files): `desmoke/` holds
   **55,194 real RGB surgical frames** (854×480, smoke-degraded), pixel-aligned by filename with
   **55,194 binary instrument masks** in `seg/`, all CholecT50, 46 videos. `raw_data/` in the
   README is a layout **you must assemble**, not a manifest — it has 0 files in the repo.
   ⇒ **A 55k-pair objective benchmark for SAM exists without CAMMA**, replacing an eye-adjudicated
   gate with IoU against real ground truth. ⚠️ Instruments only, binary, and not our frames.
5. ⚠️ **`/workspace/repo_rodri/.git/config` carries a plaintext GitHub PAT** on a pod shared by
   four checkouts. Rotate it and move the remote to SSH.

## 🟡 2026-08-09 — step 7 is pre-registered as rung 36, and C1 dies to an eye

**Zero GPU, zero pod.** Everything below ran locally or over the S3 gateway.

1. 🔒 **Claimed as rung 36 — 🔻 RENUMBERED TO 37 on 2026-08-11, see below.
   `experiments/37-attention-vs-masks/`.** ⏸️ `G-BOUNDARY`'s formulation is under review by
   legokna. *(Written 2026-08-09 as "PRE-REGISTERED, NOT BUILT"; the export ran 2026-08-11.)*
2. 🔴 **C1 is dead as a clip-count control, and not because it failed.** It passed (separation
   6.475 vs 0.5; the recovered raw data gives bootstrap **CI [+1.05, +11.83]**, Cohen **d 0.52**,
   P(superiority) **0.627** — a medium effect with near-total range overlap, against a threshold
   of 0.5 that never separated the two competing stories). 🔑 **A human eye pass over its own 8
   exported frames killed the interpretation**: in **all three `gold == 1` frames where a clip was
   visible, SAM masked none of them**, and the clips it does catch are the white/plastic ones ⇒
   **the separation is scene complexity, not clip count**
   (legokna's mask-reading pass).
3. 🟢 **And the same eye pass supplies rung 36's enabling fact:** SAM segments instruments, gauze
   and plastic clips well, and **does not merge tissue regions with foreign objects even where it
   over-segments tissue**. That is what makes the masks a usable localization target.
4. ⚠️ **C2 passes as pre-registered and its margin is thin.** 0.9167 against a 0.90 floor, but
   bootstrap **CI [0.8557, 0.9670] contains the floor**, **5 of 30** pairs sit below it, minimum
   **0.367**, and a ~40 ms gap is one breath — it is a floor test, so passing proves little.
   ⇒ rung 36 carries C2′ at **1 s** and exports the five failures.
5. 🟢 **`RESULTS_controls.json` RECOVERED and committed** to `experiments/29-sam2-temporal/`,
   **outside `runs/`**. It had existed only as prose in this file: `run_controls.py:52` writes into
   a gitignored `runs/` dir, and the committed notebook carries `SMOKE = True`, which runs **one
   frame per group** — so the repo as committed did not reproduce the reported n=40+40 / n=30.
   The recovered JSON carries `smoke: false`, `grid: 16`, `seed: 42`. **Same failure shape as the
   ±0.86: a number that moved a decision, computed once, never committed.**
6. 🟢 **The 0.384 is independently re-derived** — a second implementation written from the parquet
   schema, not reusing `label_gap_audit.py`, reproduces **1,946 pairs, n=461, 0.384, 70.1 %,
   2.4 %, max 4, 2.63×**. And it is **robust to the definition**: counting *any* class rather than
   Clip gives 0.498 ⇒ 2.03×, still far from 1.2×. ⚠️ Under that broader definition the largest
   jump is **9**, not 4 — "the gold is stable" is definition-dependent and should name its
   definition.
7. 🔴 **`number` is NOT mostly clips.** `CAMPAIGN_LOG`'s *"83 % of counting is Clip"* is true of
   **per-class** questions, which are 39.3 % of the format. Over all 6,356 counting questions:
   *"how many foreign object **instances**"* **41.3 %**, *"how many **classes**"* **19.4 %**,
   *"how many **Clips**"* **32.6 %**. ⇒ **60.7 % of counting names no class at all**, and both the
   0.384 and C1 were measured on the Clip third.
8. ⚠️ **`max_pixels`, measured on CPU with no weights.** The image-processor class **does** honour
   it (1280×720: 1196 → **1125** visual tokens at the 921,600 cap — exactly what
   `config.py:40-50` predicts; 960×540 and 720×576 are below the cap and unchanged). 🔴 **But
   `engine.py:161` resizes through `qwen_vl_utils.process_vision_info`, and `_messages()` carries
   no `max_pixels` key** — so no-op #4 stands on the path the container actually uses.
   **Untested link:** whether `AutoProcessor.from_pretrained(..., max_pixels=N)` forwards to the
   image processor at all. Needs the checkpoint's config JSONs, no GPU.
9. 🔴 **Upstream bug, `transformers` 4.57.6:** `Qwen2VLImageProcessorFast.__init__` mutates the
   **class-level** `size` dict, so constructing one processor with `max_pixels` changes the default
   for every processor built afterwards in the same process (1280×720 → 72 tokens through a
   *fresh, uncapped* processor). The slow processor does not have it. Rung 31 builds four
   processors in one process — same `max_pixels` in all four, so between-arm comparison holds.

## 🟡 2026-08-10 (pm) — `fo_class` is the new front, and Rodrigo's single-class hypothesis is ON THE TABLE

**Two kill tests are RUNNING autonomously** (`experiments/36-clip-sponge-probes/`, pod
`qsmsncno58eij8`, pushes and stops itself). Nothing to babysit.

### Why `fo_class` at all

`fo_class` is **42.8%** of the eval (2,675 of 6,252), accuracy **0.7473**, and **78.3% of its 676
errors involve `Clip` or `Sponge` being misplaced** — 529 of 676, strict swaps `{Clip}→{Sponge}`
**95×** and `{Sponge}→{Clip}` **53×**. Ceiling if that one confusion were fixed: `fo_class`
0.7473 → **0.9450**, ≈ **+0.0702 headline** — more than double the S1 ship bar, and 3× anything
found in `number` in three days.

Error anatomy: **53% substitutions** (predicted set shares nothing with gold), **47% enumeration**
— and of the "missed an item" cases, **136 of 140 missed exactly ONE**. Accuracy by gold set size:
1 item 0.800, 2 items 0.601, 3 items 0.298, 4 items **0.000**.

### The data audit killed the imbalance story

Not a starved-class problem: Sponge is only ~19% under-supplied relative to its eval load (vs
`External drain` at 1,123 rows scoring F1 0.922), the two classes co-occur **below** chance, the
corpus already carries **144 balanced explicit Clip-vs-Sponge discrimination rows**, and — decisive
— **the model's marginal emission is calibrated**: it predicts Clip 1,099 vs 1,089 gold, Sponge
886 vs 867. A prior-driven imbalance failure does not look like that.

🔴 **What it is instead: `heico` train is `Prokto`+`Rektum`; `heico` eval is `Sigma`, which appears
NOWHERE in training.** 89.5% of heico's `fo_class` errors involve the pair against 59.6% on
`lapchole`; Clip F1 falls **0.858 → 0.761** across that boundary; **half of the 529 errors sit in 5
of 38 videos**, behind only 466 distinct frames. Sharpest datum: video `0025-Heico-Sigma-6` has
gold Clip **90** and gold Sponge **0**, and the model emits Sponge **38 times** — **41 of the 95
Clip→Sponge errors occur in videos where Sponge is never a gold answer at all.**
⚠️ The audit could **not** separate *"Sigma looks different"* (domain shift) from *"a bloodied
sponge and a metal clip are genuinely hard to tell apart"* (fine-grained vision). They are
perfectly confounded — every affected video is Sigma and we own no Sigma training data.

### 🔑 Rodrigo's hypothesis, recorded as his and NOT yet adjudicated

> *"Con que aprendas qué son los clips también es importante. No necesitan estar en el mismo
> dataset para que el LLM lo sepa identificar. No importa si empujan la frontera hacia esponja."*

**Claim: a clips-only segmentation source is sufficient. The two classes do NOT have to appear in
one dataset for the model to learn the distinction, and the asymmetry risk is acceptable.**

This directly overrides the literature sweep's caution, which argued that gauze-only data can only
push the boundary toward `Sponge` — our larger error direction (95 vs 53) — producing a loss
disguised as a wash. **Rodrigo's counter is that `Clip` is where we actually bleed, so a
clip-only source pushes the RIGHT way.** He also accepts dropping gauze entirely (the only real
gauze set is simulator + animal, and the good one is unreleased).

**The dataset that makes it testable, and it corrects a false claim in our own repo:**
🔻 `experiments/19-external-count/README.md:22` states that *"no public dataset annotates applied
surgical clips"*, attributed to three prior sweeps. **That is FALSE and must be amended.**
**HeiSurF** (HeiChole Full Scene Segmentation, EndoVis 2021, Synapse `syn25101790`, public since
**2021** ⇒ clears the 2026-07-15 gate) annotates **applied clips at pixel level in laparoscopic
cholecystectomy**, plus **`Specimen bag`** and **`External drain`**. CC BY-NC-SA, which
[[external-data-policy]] already settled as permitted. Its 720×576 matches our `frames_cache`
exactly, and the challenge design confirms our 170 cholecystectomies are Wellcome Leap SAVE and
**not public**, so there is **zero overlap** with HeiSurF by construction.

**Gate before any training, pre-registered:** count HeiSurF's clip-bearing frames, the instance
distribution, the median clip area as a fraction of frame at our `max_pixels`, and — the one that
matters — what fraction of clip masks include clips **still inside the applier jaws**, which the
ORENA definition explicitly excludes. Die if `n_clip_frames < 300` or if >20% violate the
applier-jaw exclusion (that is target-side noise, priced at ~21% damage by
[[target-noise-is-the-harmful-kind]]).

### The two tests deciding what is fundable, running now

* **KT-C** — ask the **un-fine-tuned base** the 148 strict swaps A2 got wrong. Base right where A2
  is wrong ⇒ **our fine-tune FORGOT it** ⇒ LiNeS/WiSE-FT, **zero further training**. Base wrong too
  ⇒ concept genuinely absent ⇒ concept learning from images (which is Rodrigo's route).
* **KT-A** — for **SETS**, the quantity rung 33 measured for single tokens: is the gold reachable
  by re-ranking? Rung 33's **45.6%** explains all five failed `number` arms; the set analogue has
  never been measured. ≥0.55 licenses candidate-set re-ranking and phrase-level contrastive losses;
  near 0.456 kills them together.
  ⚠️ Smoke at n=12 showed base 41.7% and gold-in-top-2 12/12 — **noise, not a result.**

---

## 🔴 2026-08-10 — the `number` output side is CLOSED on a mechanism, and rung 34 is what survives

Five rungs in three days. Everything below is measured; the GPU is off and nothing is running.

1. 🔴 **Rung 30 (GRPO on `number`) NO-GO** — `aggregation_ID` 0.4932 vs the step-matched SFT
   control's 0.5026. Before that, **the control was training on nothing**: both arms read the
   GRPO-format JSONL, which has no assistant turn, so `swift sft` masked every label and the run
   posted `loss` and `grad_norm` identically **0.0 on 492 of 492 steps** while looking healthy.
   Guards added (`_assert_supervised`, `_assert_learned`). ⚠️ **A checkpoint diff does NOT catch
   this** — AdamW's decoupled weight decay moves all 720 tensors at zero gradient; only
   `sum|Δ|` separates them (1.34 vs 252.2). [[rc-zero-is-not-evidence]]

2. 🔑 **Counting is TWO defects and the campaign had been averaging them.**
   `gold <= 4` is **83%** of the format, bias only **−0.125**, and **77.8%** of its errors are
   adjacent. `gold >= 5` is 17%, bias **−1.841**. The brain's "Spearman 0.49 / bias ≈ −2" replicates
   — **on the tail only**, which is where it was measured.
   [[counting-has-two-failure-modes]]

3. 🟢 **Rung 34 — the count IS in the hidden states, and this is the finding that survives.** A
   linear probe on the last-prompt state, **fitted on `lapchole` (ID) videos and read on the
   `heico` half it never saw**, beats the model's own output: **0.5264 vs 0.4680 at layer 24**.
   Depth profile: below the head through layer 16, **crosses at 18**, peaks **20–24**, and is
   **still there at layer 36**. The model holds the answer and emits a worse one — the loss is in
   the **projection into tokens**. 📌 Layers 18–24 is the band *Counting Circuits* (arXiv
   2603.18523) names on **Qwen3-VL-8B, our backbone**; we hit it from the other side.
   ⚠️ First fit was inadmissible (memorised, `acc_ID_insample` 1.0 everywhere) and was thrown out.
   [[hidden-states-hold-the-count]]

4. 🔴 **Rung 35 (NTL-WAS) NO-GO, and its own veto fired** — `aggregation_ID` **−0.0262**,
   `object_recognition_ID` **−0.0880**, every cell down.
   🔑 **And the autopsy is worth more than the arm.** NTL did **exactly** what it was built to do:
   bias **−0.420 → −0.342**, mean prediction 2.27 → 2.35, off-by-≥3 **14.3% → 13.3%**. Accuracy
   fell anyway. Of the **271** questions A2 got right and NTL lost, **161 moved +1 against 87 that
   moved −1** — it pushed already-correct answers one step off, and the off-by-one rate did not
   budge (65.8% → 65.7%). It is the global "+1" shift (dead at 0.4718 → 0.2197) in soft form.
   ⇒ **Five interventions have failed and all five MOVE PROBABILITY MASS**: shift, LUT, k-voting,
   GRPO, NTL. Rung 33 said why first — when greedy is wrong the gold is the runner-up only
   **45.6%** of the time. **Off-by-one describes the residual, not a fixable displacement**, which
   retires the 0.8185 ceiling. [[moving-the-distribution-cannot-fix-off-by-one]]
   📌 The veto damage was **gradient starvation, not leakage** — 0.00% of `fo_class` answers carry
   a digit in either arm. The pre-registered confound fired as written.

5. 🟢 **The instrument is now the best-controlled thing we own.** A2 was **recomputed** through the
   same eval path, not transcribed, and reproduces rung 21's committed epoch-3 row **to every
   printed digit** (0.490052356 / 0.730709877 / 0.649614088, diff 0.00e+00) across three weeks, a
   different pod and a rebuilt path. So every delta above is the arm, not the harness.

**Nothing shipped changed. We remain 0.5288, rank 11, both official baselines beaten.**

### What this leaves open, honestly

* **Rung 34's branch** — a second READ of the hidden state is a different object from a
  re-weighting of the head's output, and it is the only counting lever with a measured basis. As a
  standalone channel it is worth ≈ **+0.023 headline**: above the readable floor, **below the S1
  bar**.
* **Point supervision as TRAINING** — never run. Gated on measuring SAM 2's pseudo-label error
  rate first ([[target-noise-is-the-harmful-kind]] prices target noise at ~21%).
* 🔴 **And the option nobody has priced: stop spending on `number`.** Five arms, three GPU-days,
  every one negative. `fo_class` is 71% of `object_recognition` and has had far less attention.

---

## 🟢 2026-08-08 (night) — the attention probe, and seven silent no-ops

**Pod RTX PRO 4500 32 GB, ~40 min of GPU total, zero training.** Full artifacts:
`experiments/31-attention-probe/`, `experiments/32-aligner-unfreeze/`, and two viewers —
`docs/viewers/attention_viewer.html` and `docs/viewers/sam2_masks_viewer.html`.

1. 🟢 **Rung 31 — where the model looks, measured across four checkpoints.** 12 questions
   (6 ID / 6 OOD) paired over `base`, `rung02`, `rung06` and `a2`, 231 image tokens on a
   21×11 grid, read at the last prompt token. 🔑 **The headline is legokna's eye observation,
   quantified** (legokna's eye pass): **every arm
   attends the black letterbox**, which is **21.5% of the frame** — and fine-tuning
   progressively removes the bias. `base` **37.7%** (1.75× chance) → `rung02` 33.1% →
   `rung06` 25.2% → **`a2` 20.7%, i.e. at chance.** Correcting for it, content-only visual
   mass runs **0.0324 → 0.1017, a 3.14×** rather than the 2.5× the raw number gave. **A2
   attends more AND better distributed.** And A2 concentrates: top-5 patches 28.8% → 41.1%,
   entropy 0.843 → 0.772. ⚠️ Whether concentration causes undercounting is **inference over
   12 questions, not measured against gold.** ⚠️ `max_pixels` is 512×512 here, not the eval's
   1280×720 — between-arm comparison holds, the absolute level does not transfer.

2. 🟢 **Step 6's two blocking controls RAN, alone, and both PASS.** The closure
   ([[sam2-temporal-probe-closed]]) rested on four measured facts and **none was a measurement
   of SAM 2** — legokna refused to close on that deduction, correctly. C1 separation
   **6.475** against 0.5 (32.0 vs 25.5 masks, n=40+40); C2 persistence **0.9167** against
   0.90 (n=30). **SAM 2 tracks this footage and its instance count carries count information.**
   ⇒ **Step 6 still closes** — its four reasons never depended on SAM — **but it now closes
   with no flank.** 🔴 **And step 7 is REOPENED: closing it "by dependency on step 6" was
   wrong.** It needs SAM masks, not step 6's verdict, and both halves now exist. The margin
   analysis in item 1 is a crude step 7 done with luminance instead of masks.

3. 🔴 **Rung 32 (aligner) FAILS its reachability gate — the rung is not one flag.**
   `--freeze_aligner false` and `true` give **byte-identical** coverage: 720 tensors =
   504 LLM + 216 ViT + **0 aligner**. The arithmetic names the cause: the model has 253 LLM,
   108 vision and **8 merger** `Linear` layers; the adapter holds 252, 108 and **0**. ms-swift's
   `all-linear` does not expand to `model.visual.merger` or `deepstack_merger_list.{0,1,2}`.
   ⇒ the rung needs an explicit `target_modules`, a SECOND variable and a different
   pre-registration. **~10 min of GPU bought that**, against a full arm whose null would have
   read as *"unfreezing the aligner does not help"* with nothing unfrozen. **Parked as an
   optional parallel rung**, to be retried on a larger card or after the server migration.

⚠️ **Seven silent no-ops measured in one session. None raised; all passed with `rc=0`:**
`sdpa` returns `None` for `output_attentions`; mask prompts one at a time
(`obj_with_new_inputs` is assigned, not appended); `obj_ids` drained by aliasing from the
caller's list; **`max_pixels` on the processor does nothing** — the resize obeys only the
in-message key, **and `engine.py:43` sets it the same ineffective way**; four 8B models in one
process (`del`+`empty_cache` is not a teardown); the parquet `ood` column is False on all 2,071
clip-count rows so the OOD half sampled zero; and `--freeze_aligner false` above.

📌 **Recorded, not acted on: legokna could not identify the gold objects by eye** — *"no son
los típicos clips… son como cinchos o bridas de plástico blancos"* (q01/q02/q10). A competent
observer with the class description in hand cannot decide what counts as a `Clip`. Against a
gold that is **stable** (0.384, identical on 70.1% of adjacent pairs) that supports the
**expert-knowledge** reading over the label-noise one, and it is a different hypothesis from
the one step 6 was built to test.

## 🟢 2026-08-08 (pm) — rung 30 is TRAINING: GRPO on `number`, and phase C reopened on arithmetic

**Pod `vugto0zhhtm97z`, RTX 5090, 600 steps at 15.9 s/it (~2h40). 🔴 LEAVE THE POD RUNNING.**
Full state and the resume list: `context/30-grpo-number/CONTEXT.md`.

1. 🔑 **Phase C was closed for the wrong reason.** The August thread retired GRPO as *"diluted —
   it only touches `number`"*. The arithmetic says otherwise: `number` is **80.4% of
   `aggregation`**, `aggregation` is **2 of the 4 populated buckets**, so `number` carries
   **~40% of the headline**. **+4.5pp** on `number` clears S8 on `aggregation_ID`; **+7.5pp**
   clears S1; capturing half of the measured 24pt headroom (pass@8 0.945 vs greedy 0.705) lands
   **0.5770** against today's rank 1 of **0.5653**. It is the largest single lever identified.

2. 🔴 **The smoke caught a trap that would have wrecked the run silently.** `kl` read **4.06 at
   step 1, before any update**, and pinned at exactly **5.0** — one of two completion tokens
   saturating the ±10 per-token clamp (`grpo_trainer.py:964`). Cause: with a PEFT model and no
   explicit `ref_model`, ms-swift's KL reference is `null_ref_context` → `disable_adapter()`
   (`rlhf_mixin.py:186-194`) — **the raw base model, not A2**. The penalty was pulling the policy
   back toward the un-fine-tuned checkpoint, against the **+0.317** the campaign rests on.
   `ref_adapter_name` is not exposed in 4.4.1; `--ref_model <merged A2>` needs 16 GB we do not
   have at 30.4 of 32.6 GiB. ⇒ **`beta = 0`** (`grpo_trainer.py:755` short-circuits it; DAPO and
   Dr.GRPO drop the KL term too). ⚠️ **Cost, stated not hidden:** the S8(c) collapse guard moves
   from the loss to the protocol — `save_steps=100` and `object_recognition` scored at every
   checkpoint. Weaker, and only real if it is actually run.

3. 🟢 **The entropy gate replicates on a third instrument.** `frac_reward_zero_std` averaged
   **0.30** across the smoke against step 5's `zero_advantage` of **0.280/0.270**. Different
   instrument, different slice, same number. 🔻 A first reading of that same smoke reported 0.0
   off a single step and was wrong — three of ten steps carry no gradient at all.

4. 🔻 **Two ms-swift facts that break copied recipes.** `--train_type` **does not exist** on this
   build (renamed `tuner_type`, `base_args.py:93`) — `CLAUDE.md` and the official Qwen3-VL
   best-practice both still document the dead flag. And `model_type` must be pinned: the volume
   weights match three registered types and ms-swift refuses to guess. 205 lines of the cited
   ms-swift source are now vendored at `experiments/30-grpo-number/RESULTS_msswift_source.txt`,
   so no rung-30 claim is unverifiable.

5. 📌 **Side finding, own rung:** 32 counting questions carry golds malformed for `Number`
   (`'2.'`, `'Two.'`, `'Intestine: 1.'`), **all** under the phrasing *"Please provide a single
   integer"*, which appears nowhere else in the corpus. A candidate source of the trailing-period
   habit probe 16a measured at **86.7% ID**. Not fixed here — this arm must not change the corpus.

⚠️ **Not done, and the rung cannot be read without them:** score the six checkpoints (that IS the
collapse guard now), run the **step-matched SFT control with a FRESH cosine** (a resume restores
lr 0.0 and would flatter GRPO for free), and read the 493-row holdout to separate learning from
memorisation-sharpening.

## 🟢 2026-08-08 — the ladder is single-valued again: rung 24 lands, the vit-lr rung becomes 27

**Zero GPU, record-keeping only.** Two items the 31-day thread agreed on and nobody executed.

1. 🟢 **PR #2 is merged — rung 24 (geometric-aug) is on `main` after eight days open.**
   27 files, **17,533 insertions, 0 deletions**: nothing of Yingyu's was dropped. Both conflicts
   were additive and both sides are kept — `context/NOW.md` takes her 2026-07-31 entry in date
   order (after "the teacher cannot see", before the 07-30 recipe sweep), and `context/INDEX.md`
   gains [[flip-narrows-shortcut-not-a-win]] alongside the two 08-05 notes.
   🔑 **The delay had a cost beyond delay.** Because `experiments/24-geometric-aug/` and
   `context/decisions/flip-narrows-shortcut-not-a-win.md` existed only on the branch, a reader
   working from `main` concluded the evidence was **invented**. It was not — it was on the PR the
   whole time. ⚠️ **An unmerged branch is not a record**: anything cited in a cross-team thread
   has to be on `main` before it is cited, or the citation reads as fabrication.

2. 🟢 **`24-vit-lr-decouple` → `27-vit-lr-decouple`.** [[august-plan-closes-the-ladder]]:27 had
   already ruled it (*"Revived ⇒ rung 27"*) and it was never executed, so two rung 24s coexisted.
   Status **unchanged: still `CLOSED-UNRUN`** — `A_low` was answered by `A3_vitlr` as a
   significant negative, `B_high` never ran, no step needs it. All three references follow,
   including the hard `exp_dir` path at `_models/vit_lr_train.py:124`; titles read
   *"Rung 27 (was rung 24)"* so older prose still resolves.

📌 **Still owed from the same thread, not done here:** R9 (Yingyu's multiplicity asymmetry — only
the pre-registered cell may grant a win, any cell may veto one) is **not yet written**;
`significance-rule.md` still stops at R8. And the CAMMA email still has no owner.

## 🔴 2026-08-06 (pm) — step 6 is BUILT and BLOCKED ON ITS OWN PREMISE: there is no sub-second population

**Found by the first smoke, zero GPU spent on a verdict. `experiments/29-sam2-temporal/`.**
🔴 **OPEN DECISION — nothing is decided here and the analysis is deferred to the next session.**

1. 🔴 **Every timestamp in the corpus is `HH:MM:SS` with no fractional part** — checked across
   **all 12 parquets, all three tracks, 40,000 rows: zero fractional values**. The minimum gap
   between two annotated frames is therefore **1 second**, and `frame_index` cannot rescue it
   because `data.py:117` derives it as `round(start_time * base_fps)` from the same coarse field.
   ⇒ **The `≤1 s` band step 6 rests on is not a band, it is a single value.**

2. 🔴 **And that reaches back into three cited numbers.** `ERROR_ANATOMY.md:139-148` reports
   *"≤0.5 s, n=1257"*, *"`8 → 14 → 6` in 690 ms"*, *"`7 → 7 → 1` in 270 ms"* — **not computable
   from this corpus**. So the **31.1% of pairs at ≤1 s** in [[covt-reduced-sam-route]] and the
   **±0.86** attributed to *"frames less than a second apart"* rest on a resolution the data does
   not have. ⚠️ **Neither is refuted** — they may come from an earlier corpus version or a
   derivation not visible in the repo — **but neither is reproducible from what is on the volume
   today.** 📌 **Wording fix owed regardless of the outcome:** the ±0.86 is at best *"at exactly
   one second"*, and it is quoted as *"under a second"* in three documents.

3. **What is buildable** (same template, `number` gold, consecutive within a video), the census
   that replaces the assumption — full table in the experiment's README:

   | gap ≤ | arm J (jump ≥3), gold ≥5 | arm S (Δ=0) |
   |---|---|---|
   | **1 s** | **11** | 68 |
   | 3 s | 24 | 115 |
   | 30 s | 86 | 142 |

   🔴 **n=11 cannot carry the pre-registered 0.10 difference in persistence**, and widening the
   gap is **not** a free parameter change — the entire argument is that at ≤1 s a track is the
   same physical instance **by construction**. At 30 s that claim is gone. **Not widened without
   a decision.** Three options, none taken: run at ≤3 s (n=24, the wording the team's August plan
   itself used) and accept the weaker claim; find the finer-grained source someone computed the
   1,257 from; or close the probe — which also closes step 7, which needs its masks.

🟢 **The code is sound and runnable**, and needs **no new dependency**: SAM 2 ships inside our
pinned `transformers` 4.57.6 as `Sam2VideoModel`, and the checkpoint on the volume declares
exactly that architecture. Meta's package is NOT added. The port has no automatic mask generator,
so seeding is a point lattice with area filters and IoU NMS, in `_models/sam2_probe.py`.
⚠️ **The area cap is load-bearing, not hygiene:** one centre point on a `heico` frame returns
**90% of the image** — SAM 2 is class-agnostic and segments tissue.

⚠️ **Three code defects, all invisible off-GPU, all found by smokes and fixed:** the same-template
test compared **raw questions**, which embed their own `HH:MM:SS` and are therefore unique per row
(rung 08 §3's artifact, and `template_of` exists for it); `obj_ids` must be a **list** or the
processor's `len()` raises a `TypeError` that reads like a shape bug; and **the conditioning frame
must be forward-passed before propagating** — adding masks only registers the prompt, the memory
bank is built by the forward pass.

🔴 **Rodrigo's C1 negative control is restated, not dropped, and the departure is flagged for him.**
His literal test (`K == 1` on a `gold == 1` frame) cannot run on a class-agnostic segmenter. What
is blocking instead is **separation** between `gold ≥ 5` and `gold == 1` frames. A C1 failure
reports **NO VERDICT** rather than falling back to persistence-only — that fallback is deliberately
not pre-registered, because reading it after the fact would be moving the goalposts.

## 🔴 2026-08-06 — week 1's two gates are CLOSED: VCD dies, phase C replicates on `number`

**One pod session, 1× RTX 5090, ~1 h 40 of GPU, zero training.** Both remaining week-1 gates of
the team's August plan ran to a verdict.

1. 🔴 **Step 4 ran and VCD is dead — [[vcd-has-nothing-to-subtract]].** On all **228** frames
   where `Clip` is a false positive, degrading the image makes the model **less** confident in
   `Clip`: p(`Clip`) **0.7794 → 0.6198**, paired delta **−0.1596** against a pre-declared ±0.02
   band. 🟢 **The blocking `manipulation_check` passes by 26×** (0.2615 vs 0.0100) — the control
   is what makes the null readable, because "unchanged" is also what a too-weak corruption looks
   like and would have killed the lever for the wrong reason. 🔑 **Of the two deaths this is the
   informative one: VCD is not a no-op here, it has nothing to subtract** — the model **does** use
   the pixels on the exact error we wanted it to fix. ⇒ **step 8 does not exist**; week 2 keeps
   step 7 only, and nothing else in the plan moves. 🔑 The wider consequence: any lever whose
   mechanism is *"stop it answering blind"* is aimed at a defect this model does not have here —
   the over-enumeration of [[margin-is-vision-not-phrasing]] fails **after** seeing. ⚠️ Not
   unanimous (146 sink / 27 flat / **70 rise**), ID leans on the image harder than OOD (−0.2683
   on n=49 vs −0.1299 on n=179), and the scope is one class, one σ, one checkpoint, first token.

2. 🟢 **Step 5 ran a second time (SEED 43) and the scoping REPLICATES — [[entropy-gate-scopes-phase-c-to-number]].**
   `zero_advantage` **0.280 / 0.270** for `number` against a 0.60 kill line; `binary`
   (0.720/0.735) and `fo_class` (0.660/0.705) dead in both. **GRPO stays scoped to `number`, rung
   22 stays parked.** 🔴 **But only `zero_advantage` is stable (±0.015).** `headroom` is not:
   `fo_class` halved (+0.090 → +0.045) and would have flipped had its verdict rested on that leg
   — **the conjunction saved it, not the margin.** 🔑 **Instrument finding:** greedy on `number`
   moved **10.5 points** between two independent 200-question draws (~2 se) ⇒ **a per-format delta
   below ~0.10 at n=200 is not separable from the draw** — the same class of limit as the video
   jackknife's 0.024 on `acc_OOD`, and it bears on `RULES §S1–S7`. 🟢 Seed 42's `binary` float
   artifact (`0.04999999999999993`) closes on its own: at seed 43 it is 0.040.

⚠️ **Two operational facts, paid for once.** Step 5 costs **~58 min, not 1h48** (papermill's own
cell timings: 56.7 min is the generation cell, all nine evaluator passes are ~38 s; the 1h48
included the smoke and the chain). And **`FrameProvider` never touches `/workspace/frames_cache`**
(`src/frame/data.py:154`) — it reads straight from the source video with decord, so mid-run the
GPU sits near 0% while one CPU thread seeks inside multi-GB AVIs on the network volume. **That
profile looks hung and is not**, and a frame cache that does not grow is not a symptom.
⚠️ `pgrep -f "papermill <nb>"` **matches its own `bash -c` wrapper**, so a wait-loop built on it
never exits and blocks the launch behind it. Cost one silent non-start of step 5. Use `[p]apermill`.

🟢 **The pod checkout `/workspace/repo_leo` is on `main` again** (it sat on `task/covt-sam-route`,
200+ commits back, unable to fetch). Its remote carries no token by design; fetching with the URL
from `/workspace/repo` works. **Nothing was lost**: 7 of its 8 dirty files were byte-identical to
`main` and the eighth (`src/frame/metrics.py`) was an **older** draft of `jackknife_by_video`
missing the RULES §2 assert. Backups in `/workspace/tmp/leo_backup_20260806/`.

🟢 **`assert_decisions_indexed` is GREEN for the first time since it was recorded red** (2026-08-03).
Three of the five offending notes had been fixed at some point; the last two —
`seed-variance-is-small-when-clean` and `target-noise-is-the-harmful-kind` — put prose in `status:`
beginning `LITERATURE (…)`. 🔑 **The closed vocabulary was not missing a member; the field was
carrying the wrong fact.** Provenance already lives in these notes' own `measured_in:` (arXiv
2604.12469v1, named tables), while `status` describes the **standing** of the verdict — and both
verdicts are adopted and operative (one of them installed a rule), so both are **`SETTLED`**.
Nothing is lost: that the evidence is external stays in `measured_in` and in the bodies, which
already state the transfer limits.

## 🟢 2026-08-05 (pm) — step 5 RAN: phase C survives, but only for `number`

**600 train questions, k=8, on A2 ep3. 5,400 generations, 0 errors, 9 evaluator passes, 1 h 48.**
Both thresholds were pre-registered before the run (`10b_entropy_gate.ipynb`).

| format | n | `zero_advantage` | `pass@8` | greedy | headroom | verdict |
|---|---|---|---|---|---|---|
| `binary` | 200 | 0.720 | 0.985 | 0.935 | +0.050 | 🔴 DEAD |
| `fo_class` | 200 | 0.660 | 0.975 | 0.885 | +0.090 | 🔴 DEAD |
| **`number`** | 200 | **0.280** | 0.945 | 0.705 | **+0.240** | 🟢 **ALIVE** |

🔑 **Phase C does not die — it narrows to `number`**, by the rung's own pre-registered rule
(a single live format keeps it, scoped to that format). And `number` does not scrape through:
0.28 against a 0.60 kill line, and **+0.240 of headroom against a +0.05 minimum — nearly 5×**.
Sampling finds the right answer 94.5% of the time where greedy scores 70.5%: **24 points the
model can already reach and does not.** It lands where it pays — `number` is 80.4% of
`aggregation`, one of the four scored buckets. ⇒ **GRPO with a set-F1 reward is scoped to
`number`; rung 22 (loss-mass) stays parked as contingency and is NOT triggered.**
The two dead formats die of **no gradient, not no ceiling** (`mode_share` 0.94 and 0.90).
⚠️ **`binary`'s headroom is a float artifact**: 0.985 − 0.935 = `0.04999999999999993`, so the
strict `<` killed it on a threshold it exactly meets. Verdict unaffected (it also fails
`zero_advantage`), but the comparison needs a tolerance before another arm dies of epsilon.

🔴 **The jackknife over videos is in, and it bites** (`RESULTS_jackknife_by_video.csv`, 40 ID /
38 OOD runs, zero GPU, re-scoring answers that already existed):

```
ID  : max_shift median 0.0086   worst 0.1067   (28 videos)
OOD : max_shift median 0.0242   worst 0.0922   (10 videos)
```

**Dropping ONE video moves `acc_OOD` by a median of 0.024** — larger than `+0.0211`, the
campaign's best and the arm that shipped in submission 02. And it is **the same video**:
`0023 - Heico - Sigma - 4.avi` is worst case in **23 of 38** OOD runs, `0021 - Heico - Sigma - 2`
in 12 — all **Sigma**, the procedure absent from training that *defines* our OOD. ID is fine
(0.0086 over 28 videos), which matches [[local-eval-vs-judge-calibration]] exactly: the ID side
predicts the judge and the OOD side does not.
⚠️ **Read narrowly:** this does NOT falsify the paired OOD deltas — pairing shares the dominant
video across both arms and cancels part of it. It says the **absolute level of `acc_OOD` has less
precision than assumed**, and an OOD delta under ~0.024 is not separable from which video landed.
⇒ `RULES §S4` now rests on our own number instead of borrowed literature.

🟢 **SAM 2 is on the volume** (`models/sam2/sam2.1-hiera-large`, 1.7 GB, verified) — step 6's
missing piece, no GPU spent. 🟢 Migration phase 0 extracted: **4.5 GB / 562 files** of adapters
and results (not <1 GB as estimated — rank-32 adapters are large).
⚠️ **The pod is down; `/workspace` is the network volume, so everything survived** and was
recovered over S3 with no pod. Artifacts written to `/tmp` would not have been.

## 🟢 2026-08-05 — submission 02 is scored, BOTH baselines are beaten, and the local eval is inverted on OOD

**Zero GPU, zero pod.** The whole session ran off two files: the official per-bucket scores
(the platform's returned payload) and 15 `stratified.json` pulled off the volume over S3.

1. 🟢 **Submission 02 scored 0.5288 — rank 11 — and it clears BOTH official baselines.**
   Fine-tuned baseline is rank **13** (0.5189), proprietary rank **30** (0.3883). Submission 01
   was rank 23 (0.4767), so 01 → 02 is **+0.0521**. **The campaign's core value is nominally
   reached.** 🔴 **But the margin over the fine-tuned baseline is +0.0099 and we cannot test
   it** — the adjudicating significance test needs the baseline's per-question answers, held
   only by the organizers; an unpaired estimate gives se ≈ 0.0178 (ratio 0.56 vs the 1.96
   needed), and by `challenge_design.txt:1032-1033` a non-significant delta **collapses to the
   same rank**. Treat it as unconfirmed, not banked. Podium is +0.0258 away, rank 1 +0.0365.
   🟢 Confirmed the headline is the mean of the **four** populated buckets — reproduces the
   reported score **to 1e-9 for all seven entries**; bucket sizes recovered from the rationals
   sum to exactly 2000 (`agg_ID` 553, `obj_ID` 747, `agg_OOD` **188**, `obj_OOD` 512).

2. 🔴 **The local eval is calibrated on ID and inverted on OOD** —
   [[local-eval-vs-judge-calibration]]. Same checkpoint, both sides: `agg_ID` **−0.034**,
   `obj_ID` +0.071, `agg_OOD` +0.080, **`obj_OOD` +0.367**. Local `bucket_mean` **0.6496** vs a
   real **0.5288** ⇒ the local headline overstates by **+0.121**, and the error **grows**
   between submissions (+0.302 → +0.367). 🔑 **The ordering is inverted at both extremes:
   `obj_OOD` is our BEST bucket locally and our WORST on the judge.** Any prioritisation read
   off local buckets pointed at the wrong one — the cheapest explanation yet for why nineteen
   rungs of data work never moved the headline and one LR flag did. ⚠️ Reverses a criticism
   made the same day: the **ID-only `proxy_leaderboard` is the least misleading local number we
   own**; the defect is local `bucket_mean`. 🟢 **Direction survives — 8 of 8 comparisons keep
   their sign, none reversed** (`agg_ID` 0.98×, `obj_ID` 1.48×, `obj_OOD` 2.03×, `agg_OOD`
   3.76×), so past work is usable for the sign and not for the size. ⚠️ **Both calibration
   points are LARGE moves; nothing says a small delta transfers.**
   🔴 **Two things this is NOT:** the OOD split is **not** the defect — `heico`=OOD is the
   organizers' own partition (`RULES §3`) and redefining it fixes nothing (claimed and
   retracted in-session, before it was acted on); and **`kfold_lopo` is not the cheap probe** —
   `split.py:342` costs **one training run per fold**. The cheap thing is a jackknife over
   videos re-scoring predictions that already exist.
   **Best explanation standing:** `acc_OOD` — half the challenge score — rests on **10 videos**
   against 28 for ID, and checkpoints are selected by `idxmax(acc_ood)` over those same 10 and
   reported on a set containing them. Already on record as *selection bias, unmeasured*; it now
   has 0.30–0.37 of evidence. Second: against our own trivial floors the judge puts `agg_OOD`
   at **−0.018** and `obj_OOD` at **+0.019** — at the constant ([[frequency-prior-is-the-failure-shape]]).

3. 🟢 **The significance rule is settled, in legokna's formulation** — [[significance-rule]],
   now `RULES §S1–S7`. **Large ships on sign (≳ 0.03); small does not run automatically, it
   becomes a team call about spending 1 of 8 slots.** The unsigned 2026-08-01 draft is
   superseded: its ε = 0.05 floor was **larger than the entire competitive field**. The seed
   clause survives (**|Δ| < 0.01 unreadable**) because a delta that is noise has no sign to
   transfer. 🔴 The primary cell may **not** be local `bucket_mean`.

⚠️ **Tooling, paid for once:** the S3 endpoint sits behind Cloudflare, which rejects `urllib`'s
TLS fingerprint with **`403 error code: 1010` on signed and unsigned requests alike** — it reads
like a credentials failure and is not. **`curl` passes**; SigV4 signed with stdlib
`hmac`/`hashlib`, transport shelled to `curl`, region `eu-ro-1`, `ListObjectsV2` fine.
🔑 **The two OOD buckets are absent from every `RESULTS_*.csv`** — they store `aggregation_ID`
and `object_recognition_ID` only, while `bucket_mean` silently averages all four. The desglose
exists **only** in `stratified.json`.

🔴 **Still open, all zero-GPU:** the **jackknife over the 10 OOD videos** (if `acc_OOD` swings
±0.05 dropping one video, no OOD comparison in 27 rungs meant anything), **one seed repeat**
(the only thing that retires `RULES §S4`), the **CAMMA e-mail**, and
`assert_decisions_indexed` **still RED** (5 pre-existing notes, unrelated to the two added today).

## 🟢 2026-08-03 — the organizers answered on licences, and the ladder is closed

**Zero GPU, zero pod.** A rules-and-hygiene session. `main` is at `81656bb` and carries everything.

1. 🟢 **External data is PERMITTED, and the licence was never the test** —
   [[external-data-policy]]. The organizers answered in writing: training on public
   `CC BY-NC-SA` / `CC BY-SA` **is allowed**, and *"documenting all data sources in the method
   description is both necessary and sufficient"*. **Three obligations replace the licence
   question** (now RULES §15–19, sourced to `ORena-FOCUS-challenge-design-FRAME-track.pdf` pp. 7–9,
   tracked at the repo root — cite the PDF, never a paraphrase):
   🔴 the dataset must have been **publicly accessible by 2026-07-15** (pre-eval launch — check the
   DATE before the licence, it is the cheaper kill); **all** training data must be specified; and
   annotations we create on **third-party public** data must be **published** with the submission
   while annotations on **challenge** data must **NOT** be (DUA clause 3) — disjoint scope, so the
   two duties never collide. 🔑 **Whether trained weights are a derivative work is unsettled and the
   organizers decline to rule — and it does not gate us:** the award criterion says *"make their
   model … public"* and **names no licence**, so ShareAlike would constrain WHICH licence we release
   under, never whether we may release. ⇒ supersedes the *reasoning* (not the verdict) of the
   2026-07-31 dataset sweep, which rested on a `lapchole` DUA clause the organizers had in front of
   them and still would not confirm. **Closes the open baseline question**: a zero-shot frontier VLM
   + an organizer-fine-tuned open VLM, *"clearly identified as such on the leaderboard"*. And
   co-authorship is capped at **three per team** — exactly our headcount.

2. 🔴 **The ladder is closed: rungs 24 and 25 will never run** — [[august-plan-closes-the-ladder]].
   A sweep of every branch, experiment dir and vault card against the August plan's ten steps found
   three designs marked `todo`/`parked`/*"built, not launched"* being read back as **pending work**.
   They are not. 🔑 **The rule installed: a rung is revived because the plan asks for it, never
   because it exists.** Rung 24's `A_low` was answered by `A3_vitlr` (significant negative);
   **`B_high` is the ladder's ONLY genuinely open arm and still does not enter**, and the `24` slot
   is Yingyu's ⇒ revived it becomes **rung 27**. Rung 25 was parked behind 24, never ran its
   blocking `G-NO-OVERLAP`, and declares its own ceiling — CholecInstanceSeg tops out at **3** where
   our failure lives at **5–12**. Nothing deleted; each `PLAN.md` opens with a CLOSED-UNRUN banner.
   🟢 **Both questions survive, asked cheaper by the plan:** *does the model HAVE the objects?* →
   **step 6**, SAM 2 on our own frames; *does it count or emit a near-constant?* → **step 5**, the
   entropy gate. **Rung 22 (loss-mass) is the only unrun rung that survives**, as the pre-decided
   Plan B if step 5 fails.

3. 🟢 **`main` is the single source of truth again.** Fast-forwarded **+50 commits** (rungs 17, 21,
   24, 25, 26, the metrics gates, submission 02) and **two branches deleted after verifying by
   content, not by history**: `task/covt-sam-route` (0 unique commits, identical trees) and
   `task/enumeration-probe` — whose only delta over `main` was an **older** `07-enumeration/CONTEXT.md`
   reinstating the retracted *"+16.3 for the format"* ([[coa-sft-published-null]]), i.e. merging it
   would have **reintroduced a falsified premise**. Remaining branches are not ours:
   `task/audit-rung12` (Rodrigo, 4 commits of real rung-12c artifacts, his call), `task/r3-rung16`
   (contributes nothing, but it is the pod's checkout — do not switch it), `experiment/geometric_aug`
   (Yingyu, live on rung 24b).

⚠️ **`assert_decisions_indexed` is RED on `main`** — 5 pre-existing notes put prose inside
`status:` (`covt-reduced-sam-route`, `recipe-axis-is-the-learning-rate`,
`seed-variance-is-small-when-clean`, `synthetic-counting-reconciled`, `target-noise-is-the-harmful-kind`).
RULES says this gate RAISES and is never disabled; it has been red and unrun. Five one-line fixes.

⚠️ **Tooling trap, paid for once today:** `grep` in an interactive shell may be a wrapper function
that silently swallows matches — it reported **zero** hits for `public` in a file with 33. It caused
a false claim that the challenge policies were absent from the repo when they had been extracted all
along. **Use `command grep` when a zero result is load-bearing.**

🔴 **Still open, all zero-GPU, all from 2026-08-01:** submission 02 is **not uploaded** (the whole
31-day plan calibrates against an unconfirmed score, 9 of 10 slots free), the **significance rule is
unsigned** (blocks everything), and the **CAMMA e-mail is unsent** (now with no licence or date
blocker in front of it).

## 🔴 2026-07-31 — the teacher cannot see: phase 3 is closed, and the perceptual branch is the only one left

**Three measurements, one direction.** A long session on a shared volume with three checkouts.

1. 🔴 **Rung 17 ran and fired its pre-registered STOP** — [[generator-32b-is-not-a-teacher]].
   Asked the FRAME questions with no gold, Qwen3-VL-32B scores **0.08** where the trivial constant
   scores **0.295** (**margin OOD −0.2150**) — **worse than our own untrained 8B** (−0.191).
   Outputs are well-formed (`Specimen`, `Clip`, `0`), so it is incapacity, not the rung-23a silent
   engine failure. 🔑 **Read it narrowly, as legokna framed it: this does NOT invalidate CoT/CoA.**
   A bigger *general* model is not better than the base we would be teaching, so **there is nothing
   to transfer**. It would change if we fine-tuned the **32B** instead of the 8B — a real escape
   hatch — but that is double the work with no guarantee of better cost/efficiency than the 8B line
   that is actually moving the score. ⚠️ Sampling defect recorded: **200/200 `heico`**, so
   `margin_ID` is NaN and the ID half was never measured; the verdict survives because the rule is a
   conjunction and OOD fails by 21.5 pts. Two bugs left open, neither touching the number:
   `select_blind_probe` does not stratify by dataset, and `register_run()` gets a duplicate
   `run_dir`. 🔴 **The rung had been PRE-REGISTERED AND UNRUN since 2026-07-23 and carried three
   defects that only appear on execution** — `data_root` pointing at a gitignored dir the pod never
   populated, a hard-coded `exp_dir` that would write another checkout, and a scorer expecting the
   Evaluator's schema while the probe emits rung 09's. All three fixed (`b63f1fc`, `644c01f`).

2. 🟢 **Roadmap phase 0.4 CLOSED — the margin is vision, not phrasing** ([[margin-is-vision-not-phrasing]]).
   673 `fo_class` questions, same frame and gold, three phrasings in ONE process. Dropping the
   cardinality premise is **EQUIVALENT** at the pre-declared ε=0.05 (ID +0.0021). ⇒ rung 26 part 1's
   2× margin gap was **difficulty, not exploitation**, and every ladder comparison leaning on
   `object_recognition` — 50% of the headline — **stands**. ⚠️ `epsilon_min` on ID is 0.0417: with
   28 videos no finer margin was reachable. 🔑 **The secondary arm found what we were not looking
   for:** re-asked with the corpus's own *"list all"* template the model must decide the cardinality
   itself and drops **−0.0647** (50 questions flip right→wrong against 11 in OOD). Follow-up
   analysis: **100% of the flips over-enumerate**, keeping the correct class and adding false ones
   (`Clip` the usual intruder). And the gold survives the check — on the **142** stratum frames that
   also carry an independent *"List all"* question, the two golds agree **142/142** and none lists
   more than one class. So it is a **precision** failure, not the gold being incomplete.

3. 🟢 **Rung 21 is closed, and only the learning rate moved anything.** `A2_lr` (2e-4) ep3 is the
   ladder's best and the campaign's first significant win — paired vs arm A: ALL **+0.0211**
   [0.0037, 0.0391], OOD and `fo_class` ID also excluding zero. `B_rank` (32) nearly matches the
   point estimate but its CI touches zero. **`D_clip`** (`max_grad_norm` 1.0 → 10) is a faithful
   **NULL** — no cell excludes zero. **`A3_vitlr`** (`vit_lr` 2e-5 at lr 2e-4) is a **significant
   NEGATIVE** vs A2 (ALL −0.0293, ID −0.0271, OOD −0.0353, all excluding zero) ⇒ slowing the tower
   HURTS ⇒ it is **not saturated** ⇒ the roadmap fork resolves toward the perceptual branch (on ONE arm — do not write it as settled). Arm C (6 epochs) died at 9%.
   ⚠️ Rung 24's `A_low` was effectively answered by `A3_vitlr` rebased on 2e-4; **`B_high` never ran
   and is parked** (legokna's vault card for it). If resumed it goes as **rung 27** — Yingyu
   took 24 for `24-geometric-aug` in `repo_yyy`.

🔴 **Operational lesson, paid for twice today.** `/workspace` carries **three checkouts** and
`repo/` was on Rodrigo's branch. Switching its branch without checking cost a run and left a
colleague mid-rebase; his commit was rescued to `rescue/container-norm` and `repo/` restored to
`task/r3-rung16`. **We now have our own checkout, `/workspace/repo_leo`** — cloned with the token
stripped from its remote. Rung 17's `exp_dir` is self-locating for the same reason.
⚠️ **Chains die with their parent.** Arms C and D lost work that way; only `setsid`/`nohup` from
init survived. ⚠️ **Cold Python imports stall on the FUSE volume** (`WCHAN request_wait_answer`)
under load — papermill works, bare `python -c` hangs.
🔒 **The GitHub PAT is still in plaintext in every checkout's remote URL and is NOT yet rotated.**

4. 🔒 **The teacher line is closed, and for a stronger reason than a bad result.** legokna's
   framing, which is the one written down: rung 17 does **not** invalidate CoT/CoA — it says a
   bigger *general* model is not better than the base we would be teaching, so **there is nothing to
   transfer**. Verified at source the same day: **there is no eligible teacher to switch to.**
   SurgVLM publishes **no weights** (repo + project page checked — the 🤗 icons are placeholders),
   EndoChat no weights repo or licence, Gemma/MedGemma excluded because a model trained on their
   outputs is a *Model Derivative* that would propagate onto our released Apache-2.0 8B. 👀 **SurgVLM
   is the one to watch**: MIT and 9B, so if those weights land, neither licence nor hardware blocks.
   ⇒ [[coa-generator-qwen32b-onpod]] is now **SUPERSEDED IN PART** — its §3 (*"the generator's job is
   not perception, it is anchored writing"*) is the sentence rung 17 falsified.

5. 🆕 **And one new avenue with NO blocker — roadmap phase 4b.** `SurgVLM-DB` **is** partially
   released: 1.81M frames / 7.79M conversations over **23 public datasets**, lap-chole included,
   annotated **`QA Pairs; Bbox`** — boxes, not masks. SurgVLM itself is built on **Qwen2.5-VL**, our
   own family one generation back. 🔴 It annotates **instruments**, which FOCUS excludes — useless as
   task supervision. 🔑 But our measured failure is **over-enumeration** (`Clip` the magnet), and a
   clip resembles a stapler jaw or a grasper tip: **learning instruments is learning not to confuse
   them**. 🔑 And it inverts the SAM 2 relation favourably — SAM 2 is **box-promptable**, so public
   boxes + frozen SAM 2 = **surgical masks at scale**, no annotation, no expert training. Screen
   before building: *what fraction is lap-chole, and how many boxes cover the classes we confuse?*
   Zero GPU. Card: legokna's vault card for it.

🟢 **~51 GB freed** (rungs 06 and 18 `merged/`, both regenerable, adapters verified intact) with
rungs 20/21 untouched. ⚠️ `df -h /workspace` reports the whole MooseFS cluster, not our quota — it
cannot answer "how much room is left".

🆕 **Unregistered finding: 30,000 unused questions.** The challenge corpus has **three** tracks and
we use one — `frame` 20,000 (ours), **`segment` 20,000** and **`procedure` 10,000**, same videos,
same schema, carrying capability leaves that are **zero** in `frame`. ⚠️ They are segment/video-level
aggregates (*"maximum number of Clips at once in a single frame"*), so they are **not** drop-in
frame-level supervision. A rung of its own, not a quick win.

## 🟢 2026-07-31 — rung 24 (geometric-aug) CLOSED: not a win, but a real shortcut mechanism confirmed

`24_flip_p25_v1` (label-aware horizontal flip, `p=0.25`, vs rung 21 arm A) is **NOT A WIN** on
the pre-registered headline at any epoch — `margin_OOD` falls every epoch, and the targeted
871-row check (the actual hypothesis) is non-significant throughout. A separate, unexplained
`fo_class × ID` class-balanced-F1 regression also surfaced.

🔴 **A real bug was found and fixed**: `transform_qa` couldn't supply the row's real
`primary_capability` (not carried by the training JSONL), so the `1e`/situs exclusion never
fired at export time. Quantified: 32/14,415 rows (≈0.22%) plausibly mislabeled — real, but too
small to explain the headline result. Fixed at the source (text-only detection in
`flip_audit.py`, conservative `manual_review` routing).

🟢 **The actual finding**: motivated by ["Your other Left!"](https://arxiv.org/abs/2508.00549)
(MICCAI 2025), built a position-prior probe (`experiments/24-geometric-aug/
24_position_prior_probe.ipynb`) — does FRAME's model answer position questions from a
memorised class→quadrant prior instead of reading the frame, the same shortcut that paper found
in medical VLMs generally? **Yes, significantly, on rung 21 arm A** (atypical accuracy 8.5 pts
below typical, CI excludes zero) — the first on-model confirmation this failure mode transfers
here. **And the flip augmentation significantly narrows it**: a paired interaction test (both
arms score the same 661 questions) gives +0.0608, CI **[+0.0023, +0.1225]**. The augmentation
measurably does what it was designed to do, even though that didn't convert into a net accuracy
win at this dose — reducing reliance on a usually-correct prior doesn't have to raise raw
accuracy to be real progress.

**Verdict: closed, `p=0.50` not pursued as a direct scale-up** (would scale the validated
benefit and the unexplained regression together with no new information). If revisited, the
better-motivated next step is a differently-scoped rung (flip only spatial rows, not the whole
dataset) — a new decision against the campaign's other levers. Full writeup:
[[flip-narrows-shortcut-not-a-win]].


## 🟢 2026-07-30 — the recipe sweep is CLOSED, and we have a new best checkpoint

⚠️ **Adversarially audited the same day; three headlines were corrected.** The corrections are
in place below and collected in the decision note's "What the audit changed" table.

**Read [[recipe-axis-is-the-learning-rate]].** Five arms, one flag each, zero data change in
any of them. **Only the learning rate is real.**

| arm | flag | baseline | Δ proxy @ep3 | paired cells (of 30) |
|---|---|---|---|---|
| `A_lr` | lr 2e-5 → **1e-4** | rung 18 | **+0.0480** | **21 sig, all pro-arm** |
| `A2_lr` | lr 1e-4 → **2e-4** | `A_lr` | **+0.0203** | 3 sig, all pro-arm |
| `B_rank` | r 8→32, α 32→128 | `A_lr` | +0.0193 | **0 sig** |
| `D_clip` | `max_grad_norm` 1.0 → 10.0 | `A2_lr` | −0.0051 | 3 sig, **all at ep1/ep2** |
| `A3_vitlr` | `vit_lr` 2e-4 → 2e-5 | `A2_lr` | **−0.0278** | **4 sig, all pro-CONTROL** |

🟢 **BEST CHECKPOINT OF THE CAMPAIGN — `21_lr_2e4_v1/checkpoint-2703`** (arm A2, epoch 3):
proxy **0.6104**, `bucket_mean` **0.6496**, `margin_OOD` **0.2343**. Rung 06 ep3's 0.5724 had
stood since 13 July; nineteen rungs of data work did not move it and one flag did.

🔻 **Lowering the ViT learning rate costs 0.028 — but the ViT reading is DOWNGRADED.** A3 loses
significantly (4 of 30 cells, all pro-control). ⚠️ **It is a TWO-flag arm**: `--optimizer
multimodal` is emitted if and only if `vit_lr` is set, and **A2 never passed it**, so the two
checkpoints differ in two things. The single-variable gate diffs against a *synthetic* baseline
config and omits `optimizer` from its artifact check. Defensible claim: *lowering `vit_lr`
under the multimodal optimizer costs 0.028.* **NOT** "the tower wants the high LR", and
[[vit-lora-partial]] stays **OPEN**. 🔴 **Open action: a 20-step probe of A2's config with and
without `--optimizer multimodal` at `vit_lr == learning_rate`.**

⚠️ `--vit_lr` is a **silent no-op unless `--optimizer multimodal` is passed** — the trap is real
and still worth more than the arm it guarded.

🔻 **Rank is OPEN — and by the pre-registration it is a WIN.** `PLAN.md:114-116` says a win =
proxy rises AND `margin_OOD` does not fall. B at ep3: **+0.0193 / +0.0155** — it passes, at ep2
and ep3. The CI gate that demoted it was the *noise instrument*, never a decision rule. B and A2
were **never compared to each other**.

🔻 **And A2's edge is not where the leaderboard looks.** The proxy is ID-only; on the ID cell
A2 is +0.0221 **[−0.0017, 0.0449]** and B is +0.0212 **[−0.0039, 0.0468]** — *neither excludes
zero*. A2 ships for being top-scoring and 4× cheaper, not for a significant edge over B.

⚠️ **The cost the headline hides:** A2 appears to give back most of arm A's class-balanced F1 on
ID (0.6906 → **0.5474**) while exact-match rises — the `Clip` attractor. 🔻 **No paired CI on
macro-F1 exists**, so this is a point estimate quoted against the rung's own rule.

⚠️ **No multiplicity correction** anywhere: 150 paired cells at 95% ⇒ ~7.5 false positives
expected. Bears on A2's 3 cells and D_clip's 3, not on arm A's 21 of 30.

**Next:** submit A2 ep3 (9 of 10 slots left), and rebase rung 22 (loss-mass) onto A2 — it was
designed against a 2e-5 recipe whose gradient behaviour it no longer describes.

## 🟢 2026-07-29 (pm) — the brain is reconciled, the ledger dedups, and the SAM 2 probe is unblocked

**Zero GPU of our own; the two rung-21 arms kept running untouched.** A housekeeping session that
turned up four things the records did not have.

1. 🔴 **The rung-21 arm named `A2` is NOT the pre-registered `A2`.** This section and
   [[undertrained-was-real]] both describe **A2 = lr 1e-4 with `vit_lr` held at 2e-5**, the
   diagnostic that separates "the recipe" from "the vision tower". What is running under the name
   `A2_lr` is `--learning_rate 0.0001 → 0.0002` (run `21_lr_2e4_v1`), printed by its own gate. The
   name was reused and **the `vit_lr` diagnostic has never been launched.** It remains the open
   question: at lr 1e-4 our LoRA drives the ViT at the LLM's rate while Qwen3-VL's default puts the
   tower 5–10× lower, and nothing measured says which side the +0.058 came from.
2. 🟢 **The tier-1 ledger de-duplication landed** (`src/frame/ledger.py`), cherry-picked from
   `task/audit-rung12` rather than merging that branch — it sits 112 commits back and its
   `summary.csv` carries 20 rows against main's 38, so a merge would drag the ledger backwards.
   Rebuilt against the pod's full artifacts (22 `stratified.json`, not local's 16): rung 10's
   triple row is gone and rung 12's `arm1_x1`/`arm2_x3` appear once each.
   ⚠️ **The second defect is confirmed and still open.** `02-lora-sft/02_lora_sft_v1` emits
   **twice**, because `_discover_stratified` recurses and the stray
   `runs/02_lora_sft_v1/eval_best/stratified.json` resolves to the same `(experiment, run)` as the
   run-root one. Both rows read 0.548623, so nothing published is wrong today. Not fixed here:
   choosing which file wins is a real decision, not a dedup.
3. 🔴 **`main` was contradicting itself on rung 12, and four orphan numbers are now corrected.**
   The 2026-07-23 audit landed only partially. Fixed: `CAMPAIGN_LOG` latency p99
   **0.196 s / ~25× → 0.352 s / ~14×** (the whole rest of the repo already said 0.352), the 12c row
   **+0.021 ID / +0.0005 OOD → +0.0207 / +0.0040**, and the same delta in
   `12-image-processing/CONTEXT.md`. A **fourth** site the audit branch never covered because it
   postdates it: [[coa-sft-published-null]] had inherited the same 0.196/~25×.
4. 🔓 **The SAM 2 temporal probe is no longer blocked.** [[covt-reduced-sam-route]] closes on
   *"no videos in local"* — but the pod volume carries **253 GB** (`orena-data/heico` 162 GB +
   `lapchole` 91 GB) plus a 1.7 GB `frames_cache`. Step 1 (SAM 2 video-mode over the 1,946
   consecutive pairs) is runnable whenever a GPU is free.

5. 🆕 **Rung 24 is built and pre-registered — `24-vit-lr-decouple` (renumbered to `27-vit-lr-decouple` on 2026-08-08), the other half of the recipe.**
   ms-swift falls back to `vit_lr = learning_rate` (`optimizers/multimodal.py:56`), so the tower has
   trained at the LLM's rate for twenty-two rungs **by default rather than by choice** — and at
   rung 21's 1e-4 it now runs **5× hotter than it ever has**. Control is **arm A itself**, all three
   epochs, already scored: same `train.jsonl` in place (sha256 verified), so the control costs
   **zero**. Two arms ×5 apart in log space, sequential against that shared control — `A_low`
   `vit_lr` **2e-5**, then `B_high` **5e-4**, registered now with a later launch date so a null on
   `A_low` cannot be written up as "`vit_lr` is dead".
   🎯 **Either outcome ranks the two branches we have failed to order for three sessions:** a tower
   that improves when decoupled is still teachable (perceptual — [[covt-reduced-sam-route]]); an
   indifferent one is saturated w.r.t. our 14,415 examples (mapping — [[counting-is-a-mapping-failure]]).
   ⚠️ Note the tension with "the model cannot see": rung 21 lifted Spearman r **+0.059** and
   exact-set `fo_class` ID **0.6478 → 0.688** *without touching perception*. 🔴 **Do NOT cite the
   `+0.174` macro-F1 alongside these** — decomposed 2026-07-29, **82% of it is one `Needle`
   question (`n_gold = 1`, 1/7 of an unweighted macro) flipping**, and `Gallstone` did **not** move
   (recall still 0.036). See the correction in [[undertrained-was-real]].
   🔴 **Three blocking gates, because every failure mode here is silent.** `--vit_lr` switches the
   optimiser to `multimodal` (`trainers/arguments.py:249`), which partitions by prefix — **a
   trainable parameter matching none of the three groups is dropped with no error**. `G-COV`'s
   static leg is already measured off arm A's own adapter (**720 tensors = 504 LLM + 216 ViT +
   0 aligner + 0 orphans**; the 216 matches `CAMPAIGN_LOG`); its runtime leg is a two-smoke
   differential still owed. `G-EQUIV` is needed because arm A used the **default** optimiser
   (`optimizer: None`, read from its `args.json`). And `assert_vit_is_trainable` guards the rung's
   most flattering failure: with a frozen tower `vit_lr` applies to an empty group, both arms train
   the identical model, and the **perfect** null reads like an unusually clean faithful negative.
   **Status: built, not launched** (`3c1fcdc`, `18233d9`) — the notebook and G-COV runtime are owed,
   and the idle pod was stopped to stop paying for it.

6. 🆕 **Rung 25 is built and pre-registered — `25-individuation-probe`, and it re-frames the
   external datasets rather than re-opening rung 19.** Rung 19 closed *"can external counting data
   be **training supervision** for `number`?"* — no, no public instance-annotated corpus reaches our
   **5–12** range. That verdict stands. This rung asks a different question of the same files:
   **not labels for the answer, labels for the intermediate representation.**
   🔑 **We have never been able to measure whether the model HAS the objects**, because our gold is
   a bare integer — *"saw 3, said 2"* and *"saw 2, said 2"* are the same observation to us.
   CholecInstanceSeg's instance masks are the localization gold we lack, so the probe asks the model
   to **point**, checks each point against the masks, and reads **coverage against verbalized count
   accuracy on the same frames**. The discriminating cell — high coverage, low count accuracy — is
   [[counting-is-a-mapping-failure]] measured on **our** checkpoint instead of borrowed from
   Alghisi (whose Qwen2.5-VL-7B answers 32% while individuating **95%**).
   🔴 **Registered kill-only, and the ceiling is declared:** CholecInstanceSeg tops out at **3**, so
   a failure at 1–3 kills the trained-pointing lever for ~2 GPU-h, and a success says **nothing**
   about 5–12. Neither outcome may be written as "pointing works".
   🔴 **G-NO-OVERLAP is blocking:** our `lapchole` split is Laparoscopic Cholecystectomy and
   CholecInstanceSeg is Cholec80-lineage lap-chole. The organizers' video IDs are **anonymised**, so
   a name match cannot settle it — the gate compares *content* (dHash, ≤6 bits, declared before it
   runs). A val-side near-duplicate **aborts the rung**; train-side hits are excluded, not kept.
   **Status: built, not launched** — and deliberately behind rung 24, because measuring on a
   checkpoint we are about to replace is rung 18's mistake again.
   🟢 **One thing IS runnable today with no GPU:** `19-external-count/_tools/medmultipoints_probe.py`.
   MedMultiPoints is the only swept candidate never measured, the only one with a native integer
   count, and the only one with a published counting win on our own model family (MAE 9.86 → 0.26).
   If its counts reach 5–12 it is the first public dataset that does and rung 19's headline needs an
   amendment; if not, that finding is confirmed by a fourth corpus and closes on merit.

⚠️ **The shared volume has a cost, and it is now MEASURED both ways.** The two live arms slowed
**10.9 → ~24 s/it** across this session, monotonically, tracking heavy read traffic from the third
pod (a `find` over 253 GB, the ledger rebuild, ms-swift imports over a network FS) — and recovered
to **11.4 / 11.7 s/it** as soon as that traffic stopped. Not the provider, not coincidence: heavy
I/O on `/workspace` costs real wall-clock to whoever is training. **Rule: while anyone is training,
read the volume over the S3 gateway, not by walking the mounted FS.**

🟢 **And a pod is not required to watch a run.** The volume's S3 API serves the live logs with no
GPU billed: endpoint `https://s3api-eu-ro-1.runpod.io`, bucket = volume id `gf78k60nlt`, creds in
`.secrets.env` / `~/.aws`. `HeadObject` 403s, so `get_object` directly; ranged GETs with `If-Match`
412; `multipart_threshold` must be 6 GB. Measured through it at 23:40: `21_rank32_v1` **81.1%**
(~1 h 40 left), `21_lr_2e4_v1` **70.9%** (~2 h 30) — **rung 24 launches on whichever frees first.**

**Branches swept.** `task/label-noise-ceiling` → **`task/covt-sam-route`** (the name had been
misaligned for three sessions). Deleted after verifying **by content, not by history**, that main
holds the same or better: local `task/data-card` (its one unique hunk, a local timestamp regex, is
superseded by `frame.metrics.template_of`) and `task/enumeration-probe`; remote
`origin/task/image-processing`. ⚠️ The earlier note that those two locals were "already in main"
was false — they held 1 and 9 unmerged commits; the verdict survived for a different reason.

## 🔴 2026-07-29 — rung 21 is the RECIPE now, and it is training

**The two optimiser rungs swapped places.** `21-loss-mass` became `22-loss-mass`; rung 21 is
`21-recipe-sweep`. The argument is the leaderboard proxy, `mean(aggregation_ID,
object_recognition_ID)`: loss-mass moves gradient from `fo_class` (71% of `object_recognition`)
to `number` (80.4% of `aggregation`), so it is **structurally near-zero-sum on exactly the
number that gates co-authorship** — its own pre-registration says the modal outcome is a wash.
The recipe is not a trade: it adds optimisation distance to both buckets. And the ordering
matters, because whether taking gradient from `fo_class` costs anything depends on whether
`fo_class` has saturated, which is what lr and epochs move.

**IN FLIGHT — arm A: `--learning_rate` 2e-5 → 1e-4.** One flag. 3 epochs, 2,703 steps, ~7.6 h
at the measured 10.1 s/it, peak 22,210 MiB of 32,607. Run `21_lr_1e4_v1`, log
`/workspace/tmp/21_full_A.log`. **Control = rung 18's already-scored per-epoch series**
(0.5255 / 0.5488 / **0.5721**) on the **same `train.jsonl`, used in place**, sha256
`180e28f0…8e8b` asserted. Zero data change. Score with `21b_epoch_eval.ipynb -p EPOCH n`.

⚠️ **Named before it ran:** our LoRA reaches the ViT at the LLM's own LR while the Qwen3-VL
default puts the tower 5–10× lower. A collapse in arm A may be the **tower**, not the recipe —
the pre-registered diagnostic is **A2 = lr 1e-4 with `vit_lr` held at 2e-5**. Arm B (rank 8→32,
α 32→128 so α/r stays 4) is **not launched until A is read**.

🟢 **The loss-mass mechanism is now MEASURED, not inferred** (`22-loss-mass/RESULTS_preflight.json`):
the trainer's `num_items_in_batch` equals the token sum of all 16 micro-batches, to the token,
every step. ⚠️ The decision note's original "confirmed at the source" quotation was incomplete —
`seq2seq_trainer.py:196` sits under `if num_items_in_batch is None:` and, had the guard been
open, would have meant training was ALREADY per-sample and the whole 62.4/20.8 table an
artefact. Free findings from the same probe: `max_grad_norm` is **1.0** (never set by us), and
`swift/plugin/` **does not exist** in ms-swift 4.4.1 — but ⚠️ **`swift/loss_scale/` DOES**, at top level
(`base.py`, `mapping.py`), and it was found and **ruled out with a reason**, not missed: `--loss_scale`
multiplies the per-token loss and **leaves the denominator alone**, which IS the ~3.3× magnitude shrink
rung 22 exists to avoid ([[loss-mass-is-token-weighted]] §hooks). The hook is `compute_loss_func`
because it is the only one that receives `num_items_in_batch`.

🟢 **`frame.metrics.class_f1_report`** landed (`0674f02`) — class-balanced F1 on `fo_class`,
validated against probe0 (rung 06 ep3 ID: n=920, exact 0.6391, macro 0.5116). It is what can
see the tail that a 0.6478 headline hides (`Gallstone` recall 0.036).

## 🔴 2026-07-27 (pm) — submission 01's metrics decoded: a 4B beats us, and we were reading the wrong number

**Read [[leaderboard-metric-vs-our-headline]] before quoting any number as "our score".**

🔴 **We were comparing incomparable quantities.** `pre_evaluation_score` is an unweighted mean over
**populated** buckets and only **two** populate on the pre-eval set, both ID — while our
`bucket_mean` averages **four** (ID+OOD). The right local comparator is **mean-ID = 0.5281**, not
0.5667, so the real local↔leaderboard gap is **−0.057**, not −0.096. Now RULES §4b.

🔴 **On identical questions a 4B beats us 0.5163 vs 0.4710.** The exact rationals (`343/754`,
`607/1246` ours; `410/754`, `609/1246` theirs) recover **B = 2000 over 20 videos** and prove the
same set. The whole margin is `aggregation` — **67 questions** — and `object_recognition` is a
**two-question tie**. ⇒ **[[aggregation-is-the-gap]]'s "+14.9 lead on `object_recognition`" is
RETRACTED**; it had compared our local val against their platform score. The `aggregation`
prescription survives and is better supported.

🟢 **Latency is a non-issue and the alarm was arithmetic.** `mean_latency_s × throughput = 20.0`
**exactly** (both teams) — one number, not two. Real cost **0.79 s/question** against a **5.06**
ceiling: **6.4× headroom**. Corroborated by running the container on the organizers' fixture
(0.78 s/q warm, 31.9 s setup of a 120 s allowance).

🟢 **`heico`=OOD is the organizers' own design, not our invention** (RULES §3). Their public
partition puts **Sigmoid Resection — a procedure absent from ALL training** — in `heico` test,
which is exactly the official *"OOD tag with respect to procedure type"*. The empty `ood` column
is a publication choice. ⇒ OOD work is **not** wasted: the final ranking weights ID and OOD
**equally** (`challenge_design.txt:2001`), and it is **Copeland with significance tests**, so a
+0.003 lever buys nothing (RULES §4c).

🔴 **Zero training examples for `event_understanding` and `complex_reasoning`** — two of five
groups, `primary_capability` is only `1a,1c,1d,1e,2a,3a` across all 20,000 public questions
(RULES §4d).

⚠️ **Selection bias, unmeasured.** `val_id ∪ val_ood` IS the whole 6252 set; every rung selects by
`idxmax(acc_ood)` over **10 videos** and then reports on a set containing those same questions.
Biases absolute numbers upward; rung-vs-rung deltas largely cancel. `kfold_lopo` (`split.py:342`)
exists and is unused.

**The submission itself was NOT broken** — an adversarial audit cleared frames, prompt, generation,
offline and Dockerfile. It has since been hardened anyway (`batch.json` layout now read,
case-insensitive frame matching, loud failure, CUDA hard-fail) and has produced a real
`answer.json` against the organizers' fixture for the first time.

🔒 **Open:** we do **not** know where the baselines sit. `challenge_design.txt:453` gates the final
stage on beating **both**, `:375` says they would be "clearly identified" on the leaderboard, and
the visible leaderboard shows 13 rows, all participant teams. Worth asking the organizers.

## 🟢 2026-07-27 — the missing control was run: rungs 14 and 15 are CLOSED, and epoch 3 erases OOD counting

**Rung 06 epoch 3 = `bucket_mean` 0.5724** (`experiments/06-vit-lora/06c_epoch3_eval.ipynb`, RTX
5090, full 6252, protocol identical to `eval_best`, 31 min, **zero training**, ~$1 of pod). All
gates green, gold 6252/6252, hybrid cross-check agrees with the vendor scorer. This is the number
[[epoch-matched-control]] said had to exist before anything else could be read.

🔴 **Both team rungs are closed.** Rung 15's 0.5699 was a win over rung 06's *wrong epoch*, not
over rung 06: against the epoch-matched control it is +0.0017 ID / **−0.0078 OOD** with **0 of 6**
paired video-clustered cells excluding zero. Rung 14 is now the only rung with a significant
cell — `ID ALL −0.0239 [−0.0433, −0.0035]`, i.e. **significantly worse** than the control. The
14+15 fusion is worse-motivated than when proposed: a measured-negative plus a measured-null.

⚠️ **Rung 06 ep3 does NOT become the shipped checkpoint.** It is the ladder's best headline
(+0.0057 over ep2) and a **statistical null** — 0 of 6 paired cells exclude zero. The standard
that closes 14 and 15 closes this too. Submission 01's ep2 checkpoint stands.

🔴 **The finding that outlives the adjudication:** at ep3 `number` on OOD scores
**exactly the trivial floor to 16 digits** (0.46907993966817496 vs floor 0.46907993966817496) —
the signature rung 05 recorded for a **black image**. Across epochs the OOD counting margin decays
monotonically (+0.0151 → +0.0128 → **0.0000**) while ID counting climbs (+0.0781 → +0.0859 →
+0.1068). Training **erases** OOD counting, and epoch 3 is where the erasure completes.

**Two repo defects surfaced, neither fixed here.** (1) `frame.ledger._discover_stratified` globs
`runs/**/stratified.json` recursively and tier 1 does **not** dedup, so two `stratified.json` under
one run dir produce two identical rows — triggered by an untracked stray
`experiments/02-lora-sft/runs/02_lora_sft_v1/eval_best/`, which was parked (not deleted) for the
ledger rebuild. (2) The rung-10/rung-12 phantom rows are **still in the committed ledger** on this
branch (8 extra rows); the root fix lives on `task/audit-rung12`, **unmerged**.

⚠️ Still no seed repeat, ever. Every delta above sits inside a variance band we have never
measured. The user declined seed repeats this session.

## 🔵 2026-07-26 — CoVT read in full, the scoring frame corrected, and the perceptual branch reopened via SAM 2

**Zero GPU, zero pod, zero cost.** A reading session that ended with two decision notes.

1. **[[covt-reduced-sam-route]] — CoVT as published = NO-GO.** Read end-to-end incl. supplementary.
   On a Qwen backbone the gain is **Depth +14.0 / Dist +7.0 / Count +1.2**; the advertised
   **+26.6% BLINK-count is LLaVA-13B vs Aurora**, not Qwen. Blockers: **774.6k-row** general
   corpus, 17K steps, stages 1–2 not skippable (skipping → BLINK 53.8, *below* base), and a
   pipeline **no recipe framework expresses** — ms-swift, LLaMA-Factory and Unsloth alike.
2. 🔴 **The scoring frame was wrong, and it is corrected.** The headline is the **mean of the two
   ID buckets** → `object_recognition × ID` is **50% of the score**, and `1d`/`1e`
   spatial_localization live inside it. We *tie* 1st place there (0.4872 vs 0.4888) — a tie is
   **not a ceiling**. **"The gap is counting" is true about the deficit and false about where
   headline points are available.** Several sessions have been scoped against the wrong denominator.
3. **[[synthetic-counting-reconciled]] — a week-old contradiction closed.** `CAMPAIGN_LOG:240`
   discarded synthetic counting; `count-calibration-dead` called it "the sole remaining lever" —
   survival **by elimination, not merit**. DISCARDED stands. And the missed fact: **rung 15 already
   ran the label-supported half of v05 / point-then-count → null.**
4. **Next front: a bespoke SAM 2 route** (not CoVT). Measured from the parquet: **15,213 frames /
   20,000 questions**, 78.6% single-question, but **979 frames carry `fo_class` AND `number`** —
   identity + cardinality on the same pixels, i.e. pseudo-masks checkable against two gold facts
   **with no clinician**. SAM 2 is *visual*-promptable (text needs a detector in front, licence
   unverified), does **video with streaming memory** — our data *is* video — and was itself built
   by a **model-in-the-loop data engine**.
5. 🎯 **The probe that goes first:** SAM 2 video-mode tracking across the **1,946 consecutive
   same-video pairs** already in `ERROR_ANATOMY` (6.0% jump ≥3 within ≤0.5 s). Steady track +
   jumping gold = **annotation noise**. This decomposes the **±0.86** that was declared
   unmeasurable without clinical adjudication.

🔴 **Blocked:** `external_data/` is parquet only (420 KB) — **no videos, no `frames_cache` in
local.** The probe needs volume access.

## 🟢 2026-07-25 (pm) — the whole repo is consolidated onto `main`, and rung 16 → 17

**`main` is now the single source of truth — it carries EVERYTHING** (merge commit `abdcbcf`,
then `9d663e4`): submission 01 (rung 06), the **CoA line** (rung 09 — `gen_onpod.py`,
`coa_scaffold_gen.py`), and the full **R2 wave** (experiments 13/14/15/16→17 + result CSVs +
robustness analysis + the two literature corpora). No more sibling divergence — **everyone
branches off `main` going forward** (Leo took over the CoA line; Rodrigo owns the rest).

**How it happened.** Two entangled sibling branches existed — `task/r1-coa-sft` (Rodrigo, +12,
held rung 09 alone) and `task/r2-lit-levers` (mostly Leo, +38, held the wave). Neither contained
the other, and Leo had meanwhile moved to committing **directly on `main`** (31 commits incl.
submission 01). Consolidation: FF `main → r1` (brought CoA + all of main), then merged `r2 → main`
(brought the wave). Three brain conflicts were **union-resolved** (INDEX.md, RULES.md kept both
sides; NOW.md kept both dated sections newest-first) — **brain polish is still deferred**, this is
functional-not-pretty. PR #1 auto-marked MERGED.

**Branches after the sweep.** Deleted (local + GitHub): `task/r1-coa-sft`, `task/r2-lit-levers`
(their work is all in `main`), plus 12 fully-merged stale branches earlier the same day. **Kept,
untouched:** `task/audit-rung12` (+32, decision pending), `origin/task/image-processing` (+28,
🔒 do-not-touch per user), `origin/task/enumeration-probe` (+9). Local safety refs `backup/*`
remain, deletable anytime.

**Rung 16 → 17 renumber.** The CoA generator perception probe (`generator-probe`) moved from rung
16 to **rung 17** (`experiments/17-generator-probe/`, notebook `17_generator_probe.ipynb`,
`context/17-generator-probe/`) to **free the 16 slot for the 14/15 continuation work**. Pure
renumber — folder + context + notebook + every internal label/path. **Agent-audited clean:** no
dangling refs, NOW.md inbound links repointed, outbound dep `../09-coa-sft/_tools/gen_onpod.py`
resolves, notebook JSON valid. The ladder now skips 16 (unused), normal for these skip-numbered
ladders. Rung 17 is still **PRE-REGISTERED, UNRUN** (needs ≥80 GB).

**Correction booked this session:** rung 14 (appearance-aug) is a **statistical** null, NOT
"flat/did nothing" — its best checkpoint lifts `margin_OOD` **+0.0078** (OOD `fo_class` **+0.017**)
at an ID cost of −0.0116, a real ID↔OOD robustness trade in the designed direction, just inside
the noise band (CI [−0.0038, +0.0195] crosses 0). rung 15 lifts `number` **on ID only**
(`number_margin_ID` +0.117 vs rung 06's ~0), evaporates OOD. Both are directional signals worth
stacking/dosing, not dead ends — see [[epoch-matched-control]].

## 🟢 2026-07-25 — the first submission is UPLOADING, and the team's two rungs are read.

**Submission 01 is going up** ([[submission-01-rung06]]): algorithm `Qwen3VL-8B-FT-ViT-LLM-v1`,
the merged rung-06 checkpoint (ep2 / step 1720), 17 GB tarball, offline, built from the official
template. **The leaderboard score is the distance-to-baselines number the whole campaign has
lacked** — it decides whether levers of the +0.003 size are worth a 7.5 h run at all.

🔴 **It nearly shipped broken.** The platform's algorithm interface declares `batch-frames` as a
**ZIP at `/input/batch-frames.zip`**, while the organizers' own template documents
**`/input/frames/<qID>.png`** and ships a plain-directory fixture. Our `inference.py` followed the
template; had the ZIP arrived, the per-question `except` would have written a complete
`answer.json` of **empty answers** — a silent zero costing 1 of 10 submissions. The container now
**accepts both layouts** and **logs the `/input` inventory before loading weights**, so a failure
names its own cause. Prompt verified **byte-identical** to the scored engine across all paths.

🔴 **The team's rungs 14 and 15 do NOT beat rung 06 — and the ep3 comparison has no control**
([[epoch-matched-control]]). Rung 14 (appearance-aug) is a faithful **NULL**: every epoch below
rung 06, pre-registered `margin_OOD` +0.0078 with a video-clustered CI **[−0.0038, +0.0195]**.
Rung 15 (count-target) ep3 reads **0.5699 vs 0.5667**, but it is **ID-driven** — `margin_OOD`
**−0.0110**, `number` margin OOD **−0.0053 (below floor)**, its own pre-registered target — and
**0 of 10** paired cells exclude zero. The structured target parsed perfectly (0.0 % malformed),
so the **format worked and the counting did not**. ⚠️ **Correction to the 07-24 note:** the
"rung 15 = 0.5612" reading was **epoch 2 of a still-running eval**, not the rung.

🔴 **The load-bearing gap: rung 06 never evaluated its own ep3** while `eval_loss` was still
falling (0.3204 → 0.2925 → 0.2782). Its `checkpoint-2580` adapter is on the volume, unmerged and
unscored — so **both team rungs compare their ep3 against rung 06's ep2**, which is exactly where
rung 15's only win lives. Closing it is **merge + eval, ~1 h, zero training**, and it settles both
rungs at once. **Do it before any new training run**, especially before the proposed **14+15
fusion** — which would combine two nulls, break single-variable attribution, and inherit the same
missing control. New rule: **RULES §6b — every epoch evaluated, comparisons epoch-matched.**

⚠️ **And the thing none of our tooling can currently answer: no run has EVER been repeated with a
different seed.** We hold **no variance estimate**, so a +0.003 "effect" is formally
indistinguishable from noise. This is the open question legokna raised for the next phase — the
levers may be mis-aimed rather than the training wrong.

## 🔴 2026-07-24 — Wave R2: literature-grounded levers, and two project premises falsified

**The whole session's thesis.** Two paths were on the table — preprocess the image (Leo's rung 12
line) or CoA (change the training target). A literature sweep (87 byte-verified PDFs, new corpora
`literature/preprocessing/` + `literature/vlm-techniques/`) found that **both configurations we were
about to run are published as wrong**, and re-scoped the plan onto the axes the literature says DO
win. See [[coa-sft-published-null]] and [[inference-only-input-tests-biased]].

### Running RIGHT NOW — pod `m7s3xq835y9hd4` (RTX 5090, EU-RO-1), unattended, self-stopping
Driver → `r15` → `r14`; finisher then runs `r16`, rebuilds the ledger, commits+pushes `results/`,
and stops the pod. ~10 h left. Details + the finisher's traps in [[lora-training-in-progress]].
⚠️ GPU note: two prior pods hung in provisioning; the wave first ran on a PRO 6000 Workstation
(27.5 s/step, the slow card — my error, [[train-on-powerful-gpu]] had the measurement) and was cut
over to the 5090 (~12 s/step for ViT+LLM training).

### The four R2 rungs — single-variable vs rung 06 (0.5667), pre-registered before any number

| rung | the one variable | what it attacks | status |
|---|---|---|---|
| **13** `13-wise-ft` | interpolate weights base↔rung06 (no training) | recover `number` erased by training | 🔴 **DONE — NEGATIVE** |
| **15** `15-count-target` | `number` target `"3"` → `{"label":…,"counts":3}` | counting collapse | 🔄 trained (100%), evaluating |
| **14** `14-appearance-aug` | colour/WB augmentation DURING LoRA | the OOD half | ⏳ next |
| **17** `17-generator-probe` | 32B answers with NO gold, scored canonically | is the CoA teacher a real perceiver? | ⏳ needs ≥80 GB — deferred |

**rung 13 result (in the ledger):** all three α NO-WIN — α=0.50→0.5016, 0.70→0.5530, 0.85→0.5645
(vs 0.5667). `number` margin never rises strictly in BOTH distributions and no paired video-clustered
CI excludes 0. Interpolation does not buy back counting for free. Faithful negative, zero training GPU.

### 🔴 The two falsified premises (the session's real product)

1. **CoA-format SFT is a published wash — [[coa-sft-published-null]].** exp 09 asserted "there is NO
   `SFT+CoA, no-RL` row in the source." **It exists**, in Chain-of-Adaptation (arXiv:2603.20116,
   Qwen3-VL-8B, our exact scaffold, verified in the PDF): scaffold-SFT-without-RL scores **62.0 vs
   bare-gold SFT 65.7** on EndoVis2018 — a wash that *loses*; **RLVR-with-no-tags already beats SFT
   (67.4)**; the +18 is RLVR's. The false premise came from `Bloque-A-Modelo.md`, an untracked local
   summary nobody could audit. ⇒ CoA is **re-scoped, not cancelled**: it is a cold start for a later
   **RLVR** rung (the only thing with +18 on our backbone). Rodrigo reserved that decision.
2. **rung 12's image-processing screen measured the wrong thing — [[inference-only-input-tests-biased]]**
   + [[pooled-screening-manufactures-winners]]. It ranked descriptor separability, which Awad 2025
   states does NOT predict downstream gain; and its inference-only test was biased-to-negative by a
   design Jong 2025 / Medeiros 2026 already characterised. The three axes that DO win
   (train-with-transform, geometry, frame-selection) — we tried one, badly, and two never. rung 14
   is the honest version of the first.

### Repo integrity fixed this session (numbers that existed only as prose)
- **rung 12c recovered from the pod volume** — its trained A/B (`+0.021`) was never committed; now
  reproducible. Corrected a mis-reported delta (OOD +0.0005 → **+0.0040**), verdict unchanged (null).
- **Ledger de-poisoned** — rung 12's per-arm CSV injected 8 phantom rows, rung 10 appeared 3×; root
  fix in `frame.ledger._tier1_rows_from_csv` (21→19 rows, dedup within file). Five decision notes +
  six document corrections for Leo's campaign, on branch `task/audit-rung12`.
- 🔴 **No archived result is bit-reproducible — [[archived-results-not-bit-reproducible]].** ~0.5 %
  of stored answers change on a GPU swap (measured: rung 13's α=1.0 gate, 199/200 vs the 5090 while
  the weights were bit-identical). Identity gates must build their control on the same machine.

### Open / next (Rodrigo)
- **rung 17 (32B blind perception probe)** — the gate for the whole CoA/RLVR line. Needs a ≥80 GB pod.
  Everything built + pushed; on a big pod: `papermill 17_generator_probe.ipynb -p SMOKE False`.
- **rung 15 caveat to read with its result:** it replicates Gautam 2025's structured count *format*
  but NOT the *pointing* (coordinates) — our dataset has no boxes. ⚠️ **Corrected 2026-07-27:** the
  pointing did **not** drive the 9.86→0.26; that is v05's **counting-ONLY** arm. Per its Table I the
  joint count+point arm reaches only **1.52**, so dropping pointing is if anything **favourable** for
  counting (different evaluation subsets, so not a strictly paired ablation). Partial replication of
  the *format*, not a weakened copy of the better arm.
- 🔒 **Rotate the GitHub PAT** — it is in plaintext in the pod's git remote URL.

## 🔴 2026-07-20 — rung 10 is CLOSED, and it kills a whole FAMILY of levers.

**Self-consistency is measured dead** ([[self-consistency-dead]]). Three pre-registered arms on the
full 2094 `number`: every one negative, and **k=16 significantly HARMS OOD** (−0.0430, CI excludes
0). Doubling k doubled the harm — the signature of a mode sitting on the wrong value. On OOD the
voted answer falls **below the trivial floor**. It died on **quality, not latency** (k=8 ≈ 1.15 s/q
against a pooled budget that affords it).

🔴 **The generalisation, and it is the important part.** This is the **third independent
measurement** of one fact, after [[count-calibration-dead]] and [[naming-equals-counting]]:
**the deficit is UPSTREAM of the output.** Anything that aggregates, re-encodes, re-ranks, votes on
or remaps what the model has **already emitted** is closed by measurement. Of the six ideas in the
group that owns the gap, **three are now dead** (voting, calibration, enumerate-then-count).
**The next lever must attack perception or supervision — not the output.**

⚠️ **Arm C was a trap the ID-AND-OOD conjunction caught**: +0.0284 in ID, within reach of the bar,
while significantly damaging OOD. **Never relax that conjunction.**

## 🔴 2026-07-19 — the strategy moved. Read these three before anything else.

**A zero-GPU session relocated the target and killed a lever.** Nothing was trained; every number
below came from artifacts already committed.

1. **The gap is the `number` FORMAT, not the classes** ([[the-gap-is-the-number-format]]).
   `aggregation × ID` is **80.4 % `number`**; `fo_class` does not appear in the bucket at all.
   Answering 100 % of `binary` only reaches 0.4492 — **no path to the target avoids lifting
   `number` 0.327 → ~0.482.** ⚠️ This **re-scopes [[class-imbalance-not-counting]]**: the
   `sponge`/`gallstone`/`clip` work measures `object_recognition`, the bucket we **lead by
   +14.9**. It defends the advantage; it does not close the gap.
2. **Post-hoc count calibration is DEAD** ([[count-calibration-dead]], probe 05c). Three
   pre-registered rules fail. Mechanism: true values **2, 3 and 4 share the same modal prediction
   (1)**, so a LUT trades one error for another. Not fixable with more data.
3. **The 5 s cap is POOLED, not per-question** ([[latency-budget-is-pooled]], read from the
   official submission template): `120 s setup + B × 5 s`, and the `latency` we emit is not
   scored. Measured p99 is **0.352 s**. **Self-consistency and higher `max_pixels` are
   affordable** — both had been closed against a ceiling that does not exist as modelled.
   🔴 **The risk inverts to COLD START:** imports + weight load + CUDA-graph capture all eat the
   120 s setup allowance.

Also: **the +77 % secondary-label pool is 89.6 % `fo_class` and 0 % `number`**
([[secondary-labels-are-fo-class]]) — the top-ranked data lever adds nothing to the format that
owns the gap, and survives only as a *multiplicity-transfer* hypothesis judged on `number`.

**New:** 🤖 **`context/MEASURED.md`** — generated (`python -m frame.measured`), answers *"has this
already been measured?"* over four sources. **Read it before proposing an experiment.** It exists
because this session re-derived **four** pieces of already-committed work.

## Live fronts
- **Leo (legokna)** → **submission 01 uploading** ([[submission-01-rung06]]); next, the **CoA rungs
  delegated to us** (`experiments/09-coa-sft`, on the pod volume, not yet in this tree). Then the
  question raised 07-25: **the levers may be mis-aimed rather than the training wrong** — every
  model-side result lands inside a noise band we have never measured (no seed repeat, ever), while
  the ceiling keeps pointing at the DATA (see [[rung12-dominated]] below and
  [[epoch-matched-control]]).
- **Team (RodMed)** → rungs 13–16 on branch `task/r2-lit-levers`. **13 WiSE-FT = NO-WIN** (3 α),
  **14 appearance-aug = NULL**, **15 count-target = no win under the ID-AND-OOD conjunction**,
  17 generator-probe open. A **14+15 fusion** is under discussion **and is theirs to decide** — our
  input is [[epoch-matched-control]]: run the missing ep3 control first, and note that combining
  two nulls breaks single-variable attribution.
- 📕 **CLOSED — rung 12 image processing.** Kept below as the reasoning record; the input-side
  family is **dominated by the ViT-unfreeze**, and its living heir (train-time augmentation) was
  the team's rung 14, now measured NULL. Branch A (unsharp ×3 at inference, on the fine-tuned model) is a **faithful NEGATIVE and monotonic in dose** — −0.056 ID / −0.058 OOD, both significant; the identity gate passed 50/50 byte-identical. 🔴 **The method correction it produced is the important part: any inference-only test of an INPUT intervention is biased toward the negative** on a model fine-tuned without it. That re-scopes branch A itself and, retroactively, rung 11 — it does **not** touch the output family (voting/calibration/enumeration), which died with no mismatch at all. **The rung stays open because branch B decides it** — the same two arms on the **ZERO-SHOT** model (~52 min of 5090, all code exists, only `model_path` changes), far less locked to our frames' appearance. ⚠️ Poor instrument (`bucket_mean` 0.2557, **below floor everywhere**): read it for DIRECTION, never magnitude. 🔴 **But branch B cannot be the fair test either — it is still inference.** The honest test of the whole input-side family is to **TRAIN with the transform**, which is why **idea 2 (proportional subsampled-training harness, `local/hallazgos/ideas-mejora.md` §2) is escalated from convenience to ENABLER**: it takes a run from **7.5 h to ~1.5 h**, costs only CPU hours, and unblocks the input family and the re-scoped rung 11 alike. Subsample **questions within ALL videos** — dropping videos would destroy the effective n of 38. Its known limit: the LR optimum shifts with size, so it serves **relative** comparisons, not absolute values.
  - 🆕 **12d (2026-07-21) — 32 transforms screened for ZERO GPU, and the load-bearing result is methodological** ([[context/12-image-processing/CONTEXT.md]]). `tophat` came **first** pooled (+0.0043) and collapses to **−0.0172 measured inside each video**: its advantage was between videos, because videos containing an object are videos that *look different*. 🔴 **Pooled screening of image transforms manufactures winners** — the same confound as rung 11, and the ≥3-videos gate does not prevent it. `within_video()` is now the standard gate and the primary metric. Against a raw-image separation of 0.2181: **every pipeline ending in an edge operator is strongly negative (−0.018 to −0.054)**, the only two survivors both **preserve** the image (`despec+clahe` +0.0052, `despec+bilateral+unsharp` +0.0031), and `homomorphic` was **mis-calibrated, not dead** (−0.032 → +0.0012 as it softens). **Nothing is validated** — `despec+clahe` is p=0.045 on one comparison out of 13 and fails Bonferroni. Two defects are recorded, not hidden: the negative class is **contaminated** (`scene_inventory` is `partial: true` on **100 %** of frames) and **conditional effects are averaged away**. The screen's record is **four candidates killed, none validated** → it is an instrument of **exclusion, not selection**.
  - ⚠️ **Also measured 12d:** of **8,969 `fo_class` questions, ZERO have gold `none`** although the prompt offers it. The dataset contains **no negative case** — an irreducible ceiling for any perception-side lever, and the reason a human reviewer finds frames with no visible object that the label still asserts.
  - 🆕 **12d bis (2026-07-21) — the "edge family is strongly negative" headline is RETRACTED, and it is a second instrument defect.** `wv_delta` ranks the **best of 18 descriptors** per cell (`idxmax`, `transform_bank.py:483`). Split max from mean: the edge family raises **every** descriptor (`identity` 0.1069 → 0.1245 `bilateral+morphgrad`, 0.1311 `despec+tophat`, 0.1380 `clahe+despec+morphgrad`) while lowering the best one. It does not destroy information — it **redistributes** it onto one axis, and a *maximum* reads compression as loss. Symmetrically the two "winners" barely move the mean (`despec+clahe` 0.1092): they rank first by **preserving** the standout descriptor, not by adding signal. ⇒ the negative stands as a claim about this statistic, **not** about information, and a VLM consumes pixels rather than one descriptor. **Only `homo+sobel` dies on both readings** (mean 0.088 **and** max 0.164, both below identity).
  - **Five transforms selected for the next stage (legokna):** `despec+clahe`, `despec+bilateral+unsharp`, `homo_soft`, `bilateral+morphgrad`, `homo_soft+morphgrad` — the last two are the edge-family rivals the two metrics disagree on, and the pair with `homo_soft` **isolates `morphgrad` as a single variable**. ⚠️ Ranks 4–5 of the screen are the **null anchors**: only 3 of 14 pipelines beat doing nothing. ⚠️ `clahe+despec+morphgrad` leads the *mean* metric (0.1380) and was passed over on the CLAHE vein-noise objection — it is the candidate that reading would pick. ⚠️ `despec+bilateral+unsharp` contains `unsharp`, the only bank member with a real model measurement (branch A **−0.056, monotone**): screen and model **disagree in sign**.
  - 🔴 **12e (2026-07-21) — the conditional hypothesis is a faithful NEGATIVE, and open point ② is closed.** It was the last cheap explanation for the rung: that `wv_delta` averages away a sign-flipping effect (helps on conspicuous objects, hurts on camouflaged ones), which would have explained branch A's −0.056 too. **First, the literal test is NOT MEASURABLE** — splitting 235 cells by the model's right/wrong verdict leaves **14** clearing the gate (19 using every format), because the model was asked about only **4,486 of the 15,213** indexed frames. That negative is recorded, not worked around. **Measured instead:** inside each video, does a transform's descriptor separate frames the model gets right from those it fails (37/38 videos, 3,977 frames, null band permuting the verdict within video, Bonferroni |z|>3.1)? **On the primary max statistic nothing clears the band and `null_jpeg` ranks FIRST** — the textbook signature of no effect. Two transforms clear on the *mean* statistic (`bilateral+morphgrad` +3.91σ, `homo_soft+morphgrad` +3.28σ) **but collapse when restricted to frames containing the class**, so the parsimonious reading is **class/scene composition, not conspicuity**. ⇒ **None of the five selected transforms has evidence of touching what the model actually gets wrong**, which *lowers* the case for spending pod on them as they stand. Does **not** close 12c, the train-with-transform test, or the contamination defect.
  - 🟢 **12c-res (2026-07-21) — the aux view can be sent at HALF resolution for free, zero GPU.** Settling this inside the training A/B would have been fatal (a negative composite arm could not be told from a map crippled by downscaling). `bilateral+morphgrad` Δ mean vs identity: full +0.0176, **half +0.0199**, quarter +0.0187, **1/16 −0.0113**. 🔴 **The control is what makes it readable:** the first run also said halving the *raw photograph* costs nothing (+0.0009), and the descriptors are tile statistics ≈ scale-invariant by construction — so the instrument was suspected blind before it was believed, and retested at an extreme dose. It is **not** blind (monotone to 1/16, where the map collapses): **plateau to 1/4, then a cliff**. ⇒ composite token penalty falls ~2× → **~1.25×**; compute the map at native resolution and **then** shrink (`half_post` +0.0199 vs `half_pre` +0.0165); and the map choice is scale-invariant, supporting `bilateral+morphgrad` alone. Also: the `aux_view` flag is in (`engine.py`/`config.py`, default OFF byte-identical, `gate.aux_view_payload_gate` green without a GPU).
  - **Open, in priority order (2026-07-22):** ① **12c is untested** — the screen *replaces* the image with the edge map; the proposal is image **plus** map, the only configuration in which the edge family can still work. ② **Split by model right/wrong** — does the sign of a transform invert between frames the model already gets right and the ones it fails? Zero GPU, and it would explain the whole rung if appearance help lands where the model already succeeds. ③ **Negative-class contamination** — the most serious defect of the screen, with no known fix short of annotation.
  - **On `task/image-processing`, pushed, deliberately NOT merged.**
- **Closed by Leo, on `main`:** rung 06 ViT-LoRA (🟡 PARTIAL, `bucket_mean` 0.5667, `@ 9d2f1c7` — full account in `experiments/06-vit-lora/README.md` + `context/06-vit-lora/CONTEXT.md`; ⚠️ do NOT read it as "the ViT was not the ceiling", `vit_lr` ran at the LLM's 2e-5 so it cannot separate ceiling from recipe, and [[checkpoint-selection-vs-number]] disconfirmed `vit_lr` as a next move) · rung 10 self-consistency (dead, see above) · rung 11 resolution (**dead at the gate, zero GPU** — the "100%/0%" partition was an n=50 artefact, and the axis is irresolvable anyway: 130 videos, 0 with more than one resolution, so resolution is perfectly confounded with video).
- **Rodrigo** → the MLOps/consistency system (below) + planned **R1 CoA-format SFT** ([[next-move-rodrigo-coa-format]]).

## Done this session (all on `main`, pushed)
- **The brain** — `context/INDEX.md` (map), `context/RULES.md` (DO/DON'T incl. reading rules 10-13), `context/decisions/` (ViT-swap NO-GO, Qwen ladder, Rodrigo CoA front, eval-canonical). `CLAUDE.md` points here every session.
- **Canonical eval** — `src/frame/metrics.py` (`stratified_report` + 5 RAISING gates), wired into `run.py` with a HYBRID SDK cross-check (`ref.ood` stamped from qID → SDK pre_eval becomes correct → asserts our `bucket_mean` ≈ SDK's). `delta.py` de-duplicated. Killed the leaf→group drop-bug + the all-False-`ood` mislabel + the temporal-orphan inflation.
- **Results ledger** — `results/` tiers (summary/detailed/by_run) + root `RESULTS.md`, auto-built by `frame.ledger`, never hand-edited.
- **Data card hooked into the brain** — INDEX + reading rules.
- **Margin/floor enrichment of the ledger (DONE, `task/results-margin`)** — the template-aware floor + normalisation are now ONE implementation in `frame.metrics` (`template_of` / `template_floor`); `build_card.py` (rung 08) imports them, so the card's §4b and `results/` cannot drift. `stratified_report(gold=…)` populates `floor`+`margin` on `by_format`/`by_bucket`/`by_bucket_format` and `floor_ID/OOD`, `margin_ID/OOD` at the top level; `assert_floors_vs_eval_set` now verifies floor∈[0,1] and `margin==acc−floor` (it no longer raises on a below-floor run — that is a finding, not malformed input). `results/summary.csv` gains `margin_ID`/`margin_OOD`, `results/detailed.csv` gains `floor`/`margin`, `RESULTS.md` leads with "read margin, not raw accuracy". Rescored 00-baseline + 02-lora offline (predictions local, gold = `ledger.gold_from_frame_parquets("external_data/orena-data")` from the S3 val parquets). This makes reading-rules 10–13 something `results/` SHOWS, not just says.

## Real numbers (canonical, recomputed offline via S3 — no pod)
- **Epoch 1 of both arms measured on the full 6252 (T7, 2026-07-19): epoch 2 wins everywhere but `number`-OOD.** rung 02 ep1 0.5282 vs ep2 0.5486 · rung 06 ep1 0.5345 vs ep2 0.5667. `acc_OOD` selection was right; `RULES` §6 CONFIRMED, not qualified.
- **rung-06 ViT-LoRA: `bucket_mean` 0.5667** (acc_ID 0.5444, acc_OOD 0.6078) — the ladder's best. **Real skill: margin_ID +0.207, margin_OOD +0.148** (+0.024 / +0.016 over rung 02).
  - ⚠️ **The headline is not the verdict.** The pre-registered target was `dice@2` per cell, and it says **PARTIAL** (one format only, nothing at the +0.10 relevance threshold). Reading 0.5486 → 0.5667 as "the ViT was the ceiling" is the misreading this rung exists to prevent.
  - The single variable, measured from the adapters: **+3,849,984 visual params** (21,823,488 → 25,673,472). Language side byte-identical between arms.
- **rung-02 LoRA: `bucket_mean` 0.5486** (acc_ID 0.5209, acc_OOD 0.5918). The old `pre_eval 0.708` was inflated by one temporal_grounding n=1 question — corrected in RESULTS.csv/README.
  - **Real skill (MARGIN over the template-aware floor): margin_ID +0.184, margin_OOD +0.132** (floors 0.337 ID / 0.460 OOD). By margin the model adds LESS on OOD even though acc_OOD > acc_ID — matches data card §4b.
- **00-baseline zero-shot: `bucket_mean` 0.2557** — **below floor everywhere** (margin_ID −0.088, margin_OOD −0.191): a weak zero-shot model legitimately under the trivial constant.
- rung-05 arms: a0_real 0.550 / a2_shuffled 0.334 / a1_black 0.275 (from committed CSVs).
- 🔴 **`number` per TEMPLATE (new read, 2026-07-20, from rung 10's `RESULTS_templates.csv`).** The
  2094 are not one block: **five templates are already maxed and 1947 questions carry the whole
  fight.** Margin over the template-aware floor:

  | template | n | distinct true | floor | acc | **margin** |
  |---|---|---|---|---|---|
  | *How many **Clips**…* | **681** | 12 | 0.239 | 0.266 | **+0.026** |
  | *…foreign object **instances**…* | **830** | 11 | 0.286 | 0.336 | **+0.051** |
  | *…foreign object **classes**…* | 436 | 4 | 0.608 | 0.665 | +0.057 |
  | *How many Sponges…* | 83 | 2 | 0.904 | 0.928 | +0.024 |
  | Drains / Needles / Bags / Specimens | 64 | 1 | 1.000 | ~0.99 | 0 (degenerate) |

  **`Clips` is the single largest hole in the exam**: 681 questions, 12 distinct true values, and
  the model beats "always answer the mode" by **+2.6 pts**. Read with rung 05 (black image returns
  the `number` floor to 16 digits) the diagnosis is: **we are a good object RECOGNISER that does not
  INDIVIDUATE instances** — and the exam weights individuation at 50 %.

## Key findings baked in (from the data card, rung 08)
- 🔴 **We have never read `secondary_capabilities`** ([[unused-metadata]], 2026-07-19). 89.8% of train questions carry them, and **aggregation appears as a SECONDARY label on 4,238 more train questions (+77%)** — the supervision pool for the bucket we need to lift is **9,762, not 5,524**, at zero annotation cost. Ranking stays primary-only, so this changes what we can TRAIN on, not what we are SCORED on. Also unused: `generation` (automatic 78% / anchor 15% / manual 6.4%, same proportions in train and val). ⚠️ **`clinical_relevance` is all-False in BOTH splits** — a second landmine beside `ood`; never filter on either from public data.
- 🔴 **`aggregation` IS the gap, and it is not a ceiling** ([[aggregation-is-the-gap]], 2026-07-19). First external reference (public leaderboard, a **participant** not a baseline): a **Qwen3.5-4B scores 0.5438 on `aggregation×ID` vs our 0.4188** — 12.5 pts ahead on the bucket worth 50% of the exam — while we lead `object_recognition` by +14.9. The split direction argues against a dataset artifact. **Matching their aggregation alone puts us at 59.1%.** Also: the leaderboard's `pre_evaluation_score` is the mean of **populated** buckets and **every OOD bucket is `null`** — the vara is currently ID-only.
- 🔴 **The class set is OPEN and bigger than our data** ([[open-class-vocabulary]], 2026-07-19). `overview.md:17` says "such as … **and similar objects**"; the organizers' predefined list is **10 classes**, of which **`mesh` and `foreign object` have ZERO examples in train AND val**, and `silicone loop` exists only in train. **Our 8 classes are an artefact of our batch, never a definition of the task — do not optimise the training class mix against val frequencies.** Separately: `overview.md:125` says questions carry a class list, but **70% of ours carry none** → possible train/test regime mismatch, and an argument for open-vocabulary output (FICHAS lever #2, HIGH, untried).
- 🔴 **The FO failure is per-CLASS, not per-count** ([[class-imbalance-not-counting]], 2026-07-19). `gallstone` recall **0.000** in both arms (14 training examples); `needle` 0.511; `sponge` 0.633 **with 558 examples**. Training carries a **phantom class** — `silicone loop`, 435 train examples, **zero in val** — emitted ~27 times as guaranteed false positives. `clip` precision 0.619 (212 of 327 FPs). **68% of omissions are on classes with >400 examples**, so data rebalancing has a low ceiling; the prize is `sponge` perception. Also: **the ViT LoRA raised `needle` recall +17.8 pts** — `bucket_mean` averaged that away.
- **`acc_OOD > acc_ID` is an ARTIFACT** — the OOD floor is ~12 pts higher; read MARGIN over the template-aware floor. By margin the model adds *less* on OOD.
- **`acc_number` is not interpretable** (8 templates, 4 degenerate) — use the hierarchical estimate.
- **Effective n ≈ 38 videos**, not 6252. `procedure_type`/`generation` reach the model but `procedure_type` as a model lever risks OOD (unseen procedures break it) → analysis-only stratifier.

## In progress
- 🔶 **Rung 12 — image processing. OPEN. Branch A closed NEGATIVE; branch B is what decides it.**
  Branch `task/image-processing`, **not merged**. `experiments/12-image-processing/README.md`.
  - **A (fine-tuned base, rung 06 ckpt-1720):** unsharp at ×1 and ×3, inference only, measured on
    `fo_class`. **No arm rises; ×3 harms significantly in ID AND OOD** (−0.0558 [−0.0949,−0.0176] ·
    −0.0582 [−0.0887,−0.0277]) and the damage is **monotonic in dose**. Identity gate 50/50 byte for
    byte; control reused from rung 06 and gated to its canonical `fo_class`.
  - 🔴 **The generalisable part — appearance rarity is real and now quantified.** Rung 05 warned a
    black frame degrades *"by rarity, not only by absence of information"*; this is the first clean
    dose-response of it. ⚠️ **Therefore branch A does NOT show enhancement fails to help
    perception** — it cannot separate that from the fine-tune penalising an unfamiliar appearance.
  - 🔴 **Method consequence that reaches back:** **any inference-only test of an INPUT-side
    intervention is biased toward negative** on a model fine-tuned without it. That applies to
    **rung 11** too. It does **NOT** apply to the output-side family (voting, calibration,
    enumerate-then-count), which died with no train/test mismatch. ⇒ **The honest test of the
    input-side family is to TRAIN with the transform**, which raises the value of a cheap
    subsampled-training harness from convenience to enabler.
  - **B (next): the same arms on the ZERO-SHOT model**, far less locked to our frames' appearance.
    ⚠️ Poor instrument (`bucket_mean` 0.2557, **below floor everywhere**) — read it for direction,
    never magnitude.
  - **Tooling built and reusable:** `_models/build_frame_index.py` — one entry per cached frame with
    its questions, their results in three runs, and photometric statistics. It killed two candidates
    (global white-boost, CLAHE) for **zero GPU** before any arm ran.
- **Rung 10 — self-consistency: CLOSED, FAITHFUL NEGATIVE. Merged to `main` @ `e520dcf`.**
  `experiments/10-self-consistency/` + `context/10-self-consistency/CONTEXT.md`.
  - ✅ **The bit-identity gate PASSED** — `n_samples = 1` reproduces rung 06's `predictions.json`
    **50/50 byte-for-byte across four independent model loads** on the GPU class that produced the
    reference. The shared-code change in `src/frame/{engine,config,parsing}.py` is verified
    flag-off-identical, so the A/B was single-variable and the branch was safe to merge.
  - `src/frame/metrics.py` gained **`paired_delta_ci`** — an A/B on the same questions needs the
    bootstrap of the paired DIFFERENCE; two independent CIs discard the pairing and read far too wide.

## Pending / blocked
- **Rung 06's successor: rung 10 ran and returned a faithful negative.** The two candidates cleared
  on 07-18 (`vit_lr`, epoch-1 checkpoint) stay dead.
- 🔴 **Rung 11 — the resolution axis (6b) is CLOSED too, at the gate, for zero GPU**
  ([[resolution-is-not-the-gap]]). The "100 %/0 % partition" was an artefact of a non-random n=50:
  over the full 15,213-frame cache `lapchole` (ID) has **six** resolutions and its minimum (230k px)
  is **below** `heico`'s uniform 518k. **"OOD gets 56 % of ID's visual tokens" is wrong** — ~66 % by
  mean, inverted in the tails. And **130/130 videos have exactly one resolution**, so resolution is
  **perfectly confounded with video identity** and this dataset cannot answer the question at all.
  Accuracy across resolution cells is non-monotonic. **6b, 11b and 11c die unrun.**
- **Next lever: UNDECIDED, and the constraint has tightened.** It must act upstream of the output
  (rung 10) — and rung 11 plus rung 05 (`number` returns its floor **to 16 digits** on a black
  image) together argue that levers acting on the *image itself* have little to act on for the
  format that owns the gap. ⚠️ **Phase 0 (offline Docker + first leaderboard submission) is still
  open, and every lever is being judged against a score we have never confirmed transfers to the
  organizers' hardware, engine and batching.**
- **Superseded note — the old text of this bullet said:** "Next experiment for rung 06: UNDECIDED. Two candidates were cleared out of the way today, both cheaply: the 7.5 h `vit_lr` re-run (rationale disconfirmed — `number` decays with the ViT frozen too) and the epoch-1 checkpoint switch (**T7 measured it: epoch 2 is better in both arms, we did not own a better checkpoint**). **The two roadmaps are now reconciled in THE_MAP §"What comes next"** — they disagreed for three days and nobody could see it. Its read: **measuring p99 on a real L40S is the only step BOTH documents demand** (Bloque-A makes it a hard gate on the whole capacity branch; no question has ever run on the target hardware). The rank probe is single-sourced. 🔴 **Constrained decoding is measured dead** — `number` is 100% bare integers in all three rungs including zero-shot. See [[checkpoint-selection-vs-number]] **including its retraction**."
  - ✅ **What still holds:** both dead candidates stay dead; constrained decoding stays measured dead.
  - 🔴 **What changed 07-19:** *"measuring p99 on a real L40S is the only step BOTH documents
    demand"* was answered from the **official template instead** — the budget is POOLED, and our
    measured p99 is 0.352 s ([[latency-budget-is-pooled]]). L40S confirmation is now a
    verification, **not a gate**. THE_MAP's capacity branch and its resolution branch were both
    costed against a per-question ceiling that does not exist as modelled; **both need re-costing.**
- **05-bottleneck-audit + 03-prompt-variants rescore** — their predictions are NOT on the volume (only notebook/logs) → stay `needs_backfill`.
- **Phase 0** (offline Docker + first leaderboard submission) — still open. ⚠️ **Jul 15 was the pre-eval OPENING, not a deadline** — the real dates are **Sep 1** (pre-eval closes) and **Sep 8** (final submission). This line used to read "was due Jul 15", which made an open task look overdue.

## Infra / workflow
- **main = shared truth; one branch per task; merge to main when done.** Never work on Leo's `task/vit-lora`.
- **Offline rescoring via RunPod S3** (`get_object`, region eu-ro-1, creds in `.secrets.env`) — no pod needed to recompute metrics from saved predictions.
- **Local-first → push to GitHub.** Never `pull` on a pod while it trains; the shared volume repo is a single checkout (coordinate its branch).
- 🔴 **`stratified.json` is now VERSIONED (`.gitignore` exception, on `main` @ `da56eba`).** Before this, `results/` was committed but rebuilt from files living in gitignored `runs/`, so **`build_results_ledger` on a clone missing another run's artifacts silently downgraded that run's committed row to `needs_backfill=True` with every floor/margin → NaN.** 00-baseline and 02-lora-sft were backfilled and each reproduces its committed row to 1e-9. **Their `source_commit` moved 708a4cb → `da56eba`** (the JSON is newly tracked — the numbers are unchanged).
- ⚠️ **`frame.ledger` treats an `arm` column as a run name** (`ledger.py:67`) — an experiment CSV shaped per-arm injects phantom rows into the shared ledger. Rung 06 works around it by splitting `RESULTS.csv` (ledger-shaped) from `RESULTS_arms.csv`; the edge is still there for the next one.
- ⚠️ **`frame.metrics.template_floor` overstates margin when `gold` is incomplete** — rows without gold leave the numerator but stay in the denominator, warned only via `logger.warning`. Assert gold coverage before reading any margin.
- Pods: all OFF except a read-pod (`eu5j5t7qobk1k2`). Volume `gf78k60nlt` (EU-RO-1) holds data + all run artifacts.
