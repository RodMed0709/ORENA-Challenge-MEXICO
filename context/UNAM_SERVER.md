# context/UNAM_SERVER.md — the university box: what it is, what may run there, what may not

> Second compute environment, online since **2026-08-12**. Not a replacement for the RunPod pod —
> a **complement with a different legal status**. Read the COMPLIANCE section before putting any
> file on it.

## Hardware (measured, not quoted)

| | |
|---|---|
| Host | `hpclab-RTXA6000`, RHEL 9, Linux 5.14 |
| GPU | **2× NVIDIA RTX 6000 Ada, 49,140 MiB each**, driver 595.58.03 |
| **Compute capability** | **8.9 — Ada Lovelace, the SAME architecture as the eval L40S** |
| CPU / RAM | 96 cores · 502 GB |
| Storage | `~/storage` → `/data/uaq_user`, **17 TB volume, 16 TB free** |
| Toolchain | CUDA 12.8 · conda 25.7 · git 2.47.3 · **no `gh`, no sudo, no slurm** |
| Internet | yes — Hugging Face reachable |

🔑 **Why the architecture matters more than the count.** The dev pod is Blackwell (sm_120); the
challenge evaluates on an L40S, which is **Ada sm_89**. This box is the first hardware we have that
matches the target architecture. Two things that were blocked become possible:

1. **FP8.** `backbone-generation-is-not-the-lever.md:87-89` records that the hub's
   `finegrained-fp8` kernels carry **no sm_120 build**, so FP8 could not run on the dev card. They
   build on sm_89.
2. **A p99 proxy.** `THE_MAP.md:252` lists *"measure p99 on the REAL L40S"* as the only open item
   both roadmaps demand, and it gates the size ladder. ⚠️ Faithful **proxy**, not identical
   silicon — RTX 6000 Ada and L40S share AD102/sm_89/48 GB but differ in TDP and memory bandwidth.
   Verify the delta before quoting a number from here as "the L40S figure".

## 🔴 COMPLIANCE — no challenge data, pending a written agreement

The FOCUS DUA (`context/challenge/challenge_design.txt:522-530`, §Data usage agreement p.11)
binds us to *"(2) neither pass it on to a third party nor share it beyond members of the team […]
(5) maintain the data within a protected/secure environment compliant with HIPAA, GDPR, or similar
regulation, and ensure access to the videos are restricted to members of the challenge team only."*

**Status of the case, as of 2026-08-12:**

| fact | bearing |
|---|---|
| `uaq_user` is a **team-only** account (legokna + Rodrigo), SSH-key access | supports (2) |
| `/data/uaq_user` is **`drwx------`** — the other 6 accounts cannot read it | supports (5) |
| ⚠️ `/home/uaq_user` is **`drwxr-xr-x`** — world-readable. **Work in `~/storage`, never `$HOME`** | risk |
| 7 accounts on the box, no scheduler, root held by university staff | against (5) |
| **No written use/data agreement yet** — requested 2026-08-12 | 🔴 **the blocker** |

⇒ **Challenge frames and annotations do NOT go on this machine** until that agreement exists. The
relevant asset would be the ~1.6 GiB `frames_cache`, not the 252 GiB of source video, and
encryption-at-rest does **not** solve it: the data is decrypted for the whole training run, so it
protects against a stolen disk, not against root. If anything is ever staged here, it goes to
**tmpfs** so nothing persists.

📌 And the honest comparison: **RunPod is not a compliance gold standard either** — no DPA signed,
their admins hold root, and data persists on their volume. The university box with a written
agreement would be *more* defensible, not less. The gap is paperwork, not architecture.

## What DOES run here today (zero challenge data)

- Framework probes — `G-VIABILITY`, module/reachability censuses
- Model loading, VRAM measurement, FP8 build tests
- Pipeline smokes on **public** surgical frames (CholecT50, `Voxel51/cholect50` — laparoscopic
  cholecystectomy, same domain as our `lapchole` split)
- Public model weights in `~/storage/hf_cache`

## Layout and access

```
~/storage/ORENA-CHALLENGE/   the repo, rsynced from local (~419 MB)
~/storage/envs/              conda envs (see below)
~/storage/hf_cache/          HF_HOME — public weights
~/storage/smoke/             smoke assets: public frames + jsonl
~/storage/tmp/               TMPDIR, pip cache. PREFIX YOUR FILES: leo-* / rod-*
```

**Git is READ-ONLY here, by construction.** The repo arrives by `rsync` from local, not `clone` —
the repo is private and the GitHub key (`~/.ssh/github`) is deliberately not in the forwarded
agent, per the one-key-one-purpose rule. Two guards are set in the checkout:

```
git config --local user.useConfigOnly true    # a commit FAILS instead of inventing an author
git remote set-url --push origin DISABLED     # a push FAILS with a clear message
```

⇒ **commits and pushes happen on local, with the author's own identity.** This also dissolves the
shared-account attribution problem: nothing is authored here.

## Environments

| env | contents | for |
|---|---|---|
| `orena-gen36` | torch 2.11.0+cu128 · transformers 5.12.1 · ms-swift 4.4.1 | framework probes ([[ms-swift-cannot-train-gen35]]) |
| `orena-unsloth` | torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · transformers 5.5.0 · unsloth 2026.8.15 · peft 0.20.0 · trl 0.24.0 | gen-3.5/3.6 training ([[unsloth-is-the-route-to-gen35]]) |

**Install by stages, verifying torch between each.** Rung 23 measured the trap: installing
`accelerate` pulled torch over the pinned build and desynced torchvision, and the symptom
(`Could not import module 'Qwen3_5ForCausalLM'`) **blames the model**. Both envs above were built
stage-by-stage with an assertion that `torch.__version__` had not moved.

## Traps already paid for

1. **`/etc/ssh/ssh_config` on this box is broken** — it carries `sshd_config` directives
   (`PermitRootLogin`, `MaxAuthTries`, `LoginGraceTime`) from line 57, dated 2026-07-30. **Every
   outbound `ssh`/`git clone` from the machine fails** with *"Bad configuration option"*, for all
   users. Not ours (root-owned, and our account's first login was 2026-08-04). Workaround:
   `ssh -F /dev/null` — a config given on the command line makes ssh ignore the system-wide file.
   Reported to the admins.
2. **No scheduler.** Before any long run: `nvidia-smi --query-compute-apps=...` and `who`. A GPU
   with >1 GB used by another PID is off limits; **always set `CUDA_VISIBLE_DEVICES` explicitly**
   even when both cards are free — our 8B peaks at 22.2 GB of 48, so taking both blocks others for
   nothing. Run inside `tmux`, one session per rung, so `tmux ls` shows teammates what is ours.
3. **Unsloth's documented install is `uv pip install unsloth --torch-backend=auto` in a venv**, and
   their docs say *"Do NOT use this if you have Conda"*. Ours is conda + plain pip, which produced
   a `torchvision::nms` failure that had to be patched. It works, but **use the documented path on
   the pod.**
