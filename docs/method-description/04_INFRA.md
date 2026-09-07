# 04 — Hardware, Compute and Deployment

Feeds the **form §3 hardware questions** and the deployment part of **§5**.

⚠️ Three of the four numbers the form demands are **not directly recorded** in this repository.
Each is given below as an `ESTIMATE` with its arithmetic shown, and each is listed in
`07_OPEN_ITEMS.md` with the exact command that would turn it into a measurement. **Do not submit
an estimate as a measurement.**

---

## 1. Peak VRAM during training — ✅ MEASURED

| | |
|---|---|
| **Peak VRAM** | **22,210 MiB = 21.7 GB** |
| Torch peak | 21.08 GiB |
| GPUs | **1** (single-GPU training) |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 16 |
| Gradient checkpointing | enabled |

Source: `experiments/21-recipe-sweep/RESULTS_vram_A2_lr.json`. The same peak, 22,210 MiB,
reproduces on the epoch-sweep probe (`RESULTS_vram_C_epochs.json`) and on rung 18's own probe
(`experiments/18-count-aug/RESULTS_vram.csv`).

**Answer to the form: `22`.**

Worth reporting as a result in its own right: we probed the batch geometry before training and
`per_device=1, grad_accum=16` was both the **fastest and the lightest** configuration —
`per_device=2` was 12 % slower and 4.2 GB heavier, and `per_device=4` and `6` both OOM'd on a
32 GB card (`experiments/18-count-aug/RESULTS_vram.csv`). An 8B VLM with LoRA on ViT + LLM +
connector fits comfortably in **24 GB**, which is a genuinely useful reproducibility fact for the
paper.

---

## 2. Wall-clock training time of the final model — 🟡 ESTIMATE

The final submitted model is the two-epoch ensemble, whose training log is not in this repo
(`07_OPEN_ITEMS.md` item 1). For **submission 03** (rung 42, 5 epochs on 19,384 rows):

```
measured throughput (rung 18, same recipe, same batch geometry):  0.7224 s / sample
                                        experiments/18-count-aug/RESULTS_vram.csv

19,384 rows × 5 epochs × 0.7224 s = 70,010 s = 19.4 h        ← ESTIMATE
```

Cross-check: rung 19b ran **28.25 h** for five epochs on a larger corpus
(`context/NOW.md:9-12`), and rung 16's README prices a full training run at **7.5 h** for the
smaller corpus. The 19.4 h estimate sits sensibly between them.

**Suggested answer to the form: `19` hours for submission 03 — but confirm against the ensemble's
actual log before submitting.** If the ensemble is two epochs of the same corpus, the figure is
roughly `19.4 × 2/5 ≈ 8 h` per arm.

---

## 3. Total GPU-hours for the FRAME track — 🟡 ESTIMATE, and the weakest number in the dossier

Recorded fragments, each cited:

| item | GPU-hours | source |
|---|---:|---|
| rung 16 probe suite | ~5 | `experiments/16-count-probes/README.md:26` |
| rung 19 external counting | ~2 | `experiments/19-external-count/README.md:130` |
| rung 21 recipe sweep | ~50 (the PLAN's estimate, **not** an artifact) | `context/decisions/recipe-axis-is-the-learning-rate.md:219` |
| enumeration campaign (two RTX 6000 Ada) | ~42 | `context/decisions/enumeration-is-not-fixed-by-output-format.md:12` |
| "undertrained" investigation | ~11 + ~17 | `context/decisions/undertrained-was-real.md:11`, `undertrained-on-both-axes.md:121` |
| rung 19b | 28.25 | `context/NOW.md:9-12` |
| **recorded subtotal** | **≈ 155** | |

On top of that, roughly **15–20 further full training runs** (rungs 02, 06, 12c, 14, 15, 18, 24,
30, 35, 38, 40 A/B, 42, 45, 47, 50 A/B, 54, plus the SEGMENT arm) at 7.5–28 h each, plus dozens of
evaluation passes.

```
155  recorded
+ 15 runs × ~12 h  =  180        ← ESTIMATE, mid-point of the 7.5–28 h observed range
+ evaluation and probe passes    ≈  50
------------------------------------------------
total                            ≈  385 GPU-hours     ← ESTIMATE, range 300–500
```

🔴 **This is the least defensible number in the dossier.** The real figure is recoverable from
the RunPod billing history and the UNAM cluster's job accounting, and Rodrigo has the RunPod
account. See `07_OPEN_ITEMS.md` item 4. Spend twenty minutes on this rather than submitting a
guess.

---

## 4. Hardware actually used

| machine | GPUs | used for |
|---|---|---|
| **RunPod** (EU-RO-1, volume `gf78k60nlt`, 380 GB) | RTX 4090, RTX 5090, A100 80 GB, L40S, RTX 6000 Ada, B200 | most training and evaluation |
| **UNAM cluster** | 2 cards, ≥ 80 GB each (the 27B arm needed ≥ 80 GB — peak 52.64 GiB does not fit a 48 GB card) | the later long runs, the 27B work, the final model |

Sources: `HANDOFF.md` (RunPod volume and pod recipe), `context/NOW.md:1047` (the 52.64 GiB
measurement), `context/NOW.md:291-337` (UNAM operations).

⚠️ **Team constraint, recorded:** `uaq_user` on UNAM is **one tenant, not one card per person**.
The standing rule in the project memory is *one GPU at a time on UNAM*. This is a coordination
rule, not a hardware limit — worth confirming with Rodrigo before it appears in the paper.

**Evaluation hardware is the organizers': 1× NVIDIA L40S 48 GB.**

---

## 5. Inference latency and the compute budget

| | value | source |
|---|---|---|
| Latency, shipped 8B | **0.515 s / question** | `context/NOW.md:148-155` |
| p99 latency, zero-shot baseline | 0.59 s | `ATTACK_LADDER.md:8-13` |
| Budget | **5.0 s / question**, greedy, timeout = wrong answer | challenge rules |
| Container startup, 8B | **31.9 s** against a 120 s allowance | `CLAUDE.md` (serving note) |

The allowance is **pooled per batch**, not per question:

```
allowed per batch of 20 questions   220.0 s   (120 s setup + 20 × 5 s)
us                                   17.17 s   =  7.8 %
rank 1                               42.10 s   = 19.1 %
```

⇒ At 0.515 s/q we have room for roughly **19.7 extra forward passes per question**, and k=15
self-consistency fits inside the budget, measured (`context/NOW.md:148-160`).

🔻 **This corrects a claim the project made for weeks.** Several levers — self-consistency,
`max_pixels`, resolution — were closed on a *cost* that turns out not to exist. They are not
refuted, but their premise changed. Do not write "we could not afford X because of latency"
without checking this table; the honest statement is that we left 92 % of the compute budget
unspent.

⚠️ Capping at 80 % of the allowance matters because `saturation_fraction = 0.2` forfeits the
**whole batch** on a 20 % overrun.

---

## 6. Deployment — Docker, offline

Required by the challenge and worth a short paragraph in the form.

| | |
|---|---|
| Serving | merged LoRA → single checkpoint, `swift export --merge_lora true` |
| Runtime | `transformers` 4.57.*, torch/cu128, bf16 |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; weights `COPY`-ed into the image |
| Image size | 17.5 GB (the compressed artifact is 16.27 GB) |
| Verification | offline smoke with `--network none` |

Source: `submissions/03-rung42-connector-ood/README.md:145-200`.

### 6.1 Packaging defects we hit, and closed — genuinely useful to report

Every one of these was hit *with* network access; offline, each is fatal
(`submissions/04-rung40-conn4e5-ep23/README.md:34-58`):

1. **`llmcompressor` writes no processor files** — quantized output has weights, `config.json`
   and `recipe.yaml`, but no tokenizer, no `preprocessor_config.json`, no chat template. They
   must be copied from the pre-quantization checkpoint.
2. **vLLM needs `ninja`**, whose absence surfaces as a bare `FileNotFoundError` inside the engine
   subprocess with no mention of the package.
3. **The venv's `bin/` must be on `PATH`** — `EngineCore` resolves `ninja` through `PATH`.
4. **`torchvision` is required** by `Qwen3VLVideoProcessor`.
5. **`max_model_len` must be set explicitly** for a large model: the weights leave ~14 GiB for KV
   cache and vLLM's profiling pass, and the default OOMs *after* loading — a model that "fits" and
   still will not start.

### 6.2 A serving result worth reporting (from the 27B branch)

For a quantized 27B on one L40S, `enforce_eager=True` is **correct**, which is the opposite of
the usual advice (`CLAUDE.md`, serving note; `experiments/44-fp8-deployability/RESULTS_fp8_eager.json`):

| | `enforce_eager=false` | **`=true`** |
|---|---|---|
| startup vs the 120 s allowance | 225.6 s (88 % over) | **126.7 s** (5.6 % over) |
| latency vs the 5 s budget | 0.454 s/q | **0.498 s/q** |
| exact match vs gold | 30/50 | **30/50** |

CUDA-graph capture is a **per-process startup cost no container can pre-bake**, proven by four
starts measuring 226.1 / 227.1 / 231.1 / 225.6 s across very different configurations. The
default setting optimises the resource we have 10× spare of, at the cost of the one we were 2×
short on. **This does not apply to the 8B**, whose startup is 31.9 s — keep `false` there.

### 6.3 What was never verified

State this if the form asks about validation:

- **The GPU path inside the image** was never exercised live, for any submission. No local GPU is
  large enough and the RunPod pod cannot run Docker (it is itself a container).
- **Numerics across hardware.** Evaluation ran on an A100; the image runs on an L40S. A GPU swap
  changes ~0.5 % of answers (`context/decisions/archived-results-not-bit-reproducible.md`).
