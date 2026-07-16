# Block A — The Model · strategy, roadmap and pipeline

> **Status: proposal, pending merge into `THE_MAP.md`.** Ported from legokna's working notes
> (2026-07-16) so the team can review it. **Merge instructions at the bottom.**
>
> **Scope:** Block A only — **the product** (what goes into the Docker image). Block B (the lab /
> evaluation harness) appears only at its connection points. The A/B split is in `THE_MAP.md`
> §"The two blocks" (`2e4b0c9`).

## §0 — What governs

**Precedence, in this order:**

1. **`documentacion/overview.md`** (the official challenge page) overrides everything else.
2. **A number we measured** > a citation.
3. **A citation opened and read by hand** > a citation an agent reported.

Every citation in §6 was **opened and verified manually** (legokna, 2026-07-15); verbatim abstracts live
in `documentacion/papers-corroborados.md`. **If a conclusion here contradicts an abstract there, the
abstract wins.**

---

## §1 — The objective, in one line

> **Maximise the mean accuracy over the 4 buckets.**
> **Subject to:** < 5 s/question · 48 GB VRAM · offline · one GPU.

One objective. The constraints are **checked, not maximised**.

### The 4 buckets — ✅ MEASURED (2026-07-16), no longer assumed

Crosstab over **`val` (6252 q)** — val governs, because the score is computed on the evaluated set:

| Group | Total | Real composition in `val` |
|---|---|---|
| **`object_recognition`** | **3421** | `fo_class` 2675 (**78.2%**) · `open_ended` 544 (15.9%) · `multiple_choice` 202 (5.9%) |
| **`aggregation`** | **2830** | `number` 2094 (**74.0%**) · **`binary` 724 (25.6%)** · `open_ended` 12 (0.4%) |
| `temporal_grounding` | **1** | noise → **confirms FRAME has 2 groups → 4 buckets** ✅ |

**Real weight of each format on the score:**

| Format | % of score (val) | Our accuracy |
|---|---|---|
| **`fo_class`** | **~39.1%** | 0.589 |
| **`number`** | **~37.0%** | **0.432** ⚠ the weak one |
| **`binary`** | **~12.8%** | 0.769 |
| `open_ended` | ~8.1% | 0.639 |
| `multiple_choice` | ~3.0% | 0.762 |

> **`fo_class` and `number` are a 1:1 pair.** An earlier draft claimed `aggregation` = `number`, hence
> "`number` is 50%, a point there is worth two in `fo_class`". **False** — `aggregation` is 77% `number`
> + 22% `binary`. **THE_MAP was right**; the "correction" rested on an unverified premise.
> **`binary` is disguised counting** — 100% of it sits in `aggregation` (*"is there any gauze?"* =
> *count > 0*).

### The budget

```
  TIME      ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.59 s of 5 s   → 88% UNSPENT
  MEMORY    ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░  18.01 GB of 48  → 63% UNSPENT
```

**Speed is not a goal — it is budget.** There is no prize for answering in 0.3 s instead of 2 s. The
question is not *"how do I go faster?"* but ***"what do I spend this on to raise the bucket mean?"***

*(Caveat: 0.59 s was measured on an A100. **We have never run a single question on the L40S.**)*

---

## §2 — Where we are: the diagnosis

**What we have:** Qwen3-VL-8B + LoRA (r=8, α=32) merged, bf16. `bucket_mean 0.550` · `raw 0.567` ·
`acc_OOD 0.592 > acc_ID 0.524`.

**What the LoRA did:** the two formats that exploded are the strict-convention ones (`fo_class` +0.42,
`number` +0.29). **The LoRA did not teach the model to see — it taught the task**: the answer convention
and the vocabulary. We froze **ViT and aligner**, so it only touched the language side.

### ✅ RESOLVED by rung 05 (image ablation, 2026-07-16)

Two hypotheses competed: **perception ceiling** vs **text shortcut**. The pre-registered ablation
settled it:

> **NO SHORTCUT. The model looks.** No format triggers the shortcut condition. `fo_class` drops to
> **0.230** when the image is shuffled — **below its 0.269 trivial floor**. Without the image↔question
> link it does *worse than guessing the mode*. **That is real perception.**

**The CoA branch is ruled out by pre-registered data.** Full report:
`experiments/05-bottleneck-audit/README.md`.

### 🔴 But the real finding is narrower and worse

| Format | Weight | Visual signal (real − black) | Margin over trivial floor |
|---|---|---|---|
| `fo_class` | 39.1% | **+51.9 pts** | +32.0 |
| `number` | **37.0%** | **+8.0 pts** | **+8.0** |

**Same model, same frozen ViT, same image.** The vision path delivers object identity richly and almost
nothing usable for **counting**. With a black image, `number` scores `0.3519579751671442` — **the
majority floor to 16 digits**: blind, the model collapses to answering the mode.

**The honest reading of `number` is neither "looks" nor "shortcut" — it is "barely extracts anything".
That is the bottleneck, and it carries 37% of the score.**

**This reframes Test B.** *"Was the ViT the ceiling?"* is too coarse — the ViT plainly sees. The live
question is **"is the ViT the ceiling *for counting*?"**

---

## §3 — The backbone: closed question

**It was a decree, not a derivation** — `CONSTITUTION.md` fixes Qwen3-VL-8B. **But it is now validated
by evidence:**

- **2506.06232:** *"specialized medical VLMs currently **underperform** compared to generalist models"*.
- **2506.17337:** *"efficiently fine-tuned **generalist** VLMs can achieve comparable or even superior
  performance, **particularly when transferring to unseen or rare OOD**"*.

**Generalist + light fine-tuning is correct, and it wins precisely on OOD** — half our exam.

### 🔴 The uncomfortable finding

**We chose the 8B against the POD's VRAM (24–32 GB), not against the EXAM's (48 GB).** Different
constraints; nobody separated them. **The pod is where we train; the L40S is where we are graded.**

Corollary: **"the 32B FP8 wildcard doesn't fit" was concluded against the pod.** In FP8 it is ~32–35 GB
→ **it fits in 48**, and the L40S is Ada with native FP8. **The 32B question is not memory, it is
latency — and nobody has measured it.**

**Recommendation: do not change family.** The unlock is capacity *within* Qwen3-VL, and it hangs on one
measurement we never made.

---

## §4 — The product pipeline

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  BLOCK A — THE PRODUCT                            the only thing shipped  ║
╚═══════════════════════════════════════════════════════════════════════════╝

  ┌──────────────┐
  │ RGB image    │──┐
  └──────────────┘  │   ┌─────────────────────────────────────────────┐
                    ├──►│ [1] PROMPT ASSEMBLY                         │
  ┌──────────────┐  │   │     · system prompt + FO defs (+0.061 meas.)│
  │ question     │──┘   │     · the question ALREADY carries:         │
  │ (text)       │      │       procedure, timestamp, format, FO list │
  └──────────────┘      │     ⚠ does it match the train format?       │
                        │       → if not, THAT is OOD axis #2         │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │ [2] VLM   Qwen3-VL-8B + LoRA (merged)       │
                        │     bf16 · 18.01 GB of 48 · greedy · ≤32 tok│
                        │     ViT frozen · aligner frozen             │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │ [3] CONSTRAINED DECODING     ✗ DOES NOT EXIST│
                        │     aggregation → digits only               │
                        │     fo_class    → enum classes only         │
                        │     binary      → yes/no only               │
                        │     ⚠ constrain ONLY the final answer       │
                        │       (2408.02442: restriction harms CoT)   │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                                   "Sponge"  (≤300 chars)

        NO video · NO decoding · NO frame sampling · NO judge
```

---

## §5 — The roadmap

```
 A0 ── CALIBRATE THE OBJECTIVE ──────────────── ✅ DONE (2026-07-16) · with a surprise
   │   ✅ crosstab done → aggregation = 77% number + 22% binary · 4 buckets confirmed
   │   ✅ selection metric = BUCKET MEAN, already in use in rung 05
   │
   │   🔴 BUT: "pre_evaluation_score IS already the unweighted bucket mean" is TRUE on
   │      paper and FALSE in our runs — see §5.1. A0 was NOT already done.
   │   ⬜ REMAINS: populate `ood` from the split (1 line, no GPU) + drop the orphan question
   ▼
 A1 ── THE TWO TESTS ────────────────────────────── the bifurcation
   │   (a) IMAGE ABLATION ✅ DONE (rung 05) → NO SHORTCUT. The model looks.
   │   (b) LoRA ON THE ViT ⬜ green-lit by the pre-registered rule — but see §7.1
   │       ⚠ Expected value DROPPED. `number-probe` adjudicates first.
   ▼
 A2 ── THE BIFURCATION ────────────────── decided by data, not opinion
   │
   ├── SHORTCUT / DAMAGED GENERALIZATION ──► CoA   ❌ RULED OUT by rung 05
   │
   └── PERCEPTION CEILING ──► CAPACITY
   │      · LoRA on the ViT · a resolution step (max_pixels) · Qwen3-VL-32B FP8
   │      🚦 HARD GATE: measure L40S latency BEFORE training anything
   │
   └── 🆕 OUTPUT FORMAT ──► CoA-as-format (§7.1)
          not the old CoA branch — its premise died with the shortcut hypothesis.
          Two independent papers say +16 lives on the format side, +2 on the visual side.
   ▼
 A3 ── MAKE AN INVALID FORMAT IMPOSSIBLE ──────── independent, in parallel
   │   constrained decoding (xgrammar / vLLM guided_decoding)
   │   ⚠ constrain ONLY the final token(s); free enumeration (2408.02442)
   │   touches no weights · raises MEASURED accuracy
   ▼
 A4 ── PACKAGE ───────────────── partly executable TODAY, partly blocked
       · src/frame/serve package (§8)    ─ TODAY, no GPU
       · measure real p99 on L40S        ─ TODAY, 1 pod session
       · Docker                          ─ ⛔ the template does NOT exist yet
```

### §5.1 🔴 A0 was not already done — `pre_evaluation_score` is broken on our split

**This is the single most important correction in this document.**

The SDK defines `pre_evaluation_score` as *"the unweighted mean over ten buckets — the five capability
groups, each scored independently for in-distribution and out-of-distribution questions"* (official
README; confirmed in `focus/evaluation/evaluator.py:402-462`). **Correct on paper. Broken in our runs:**

1. **`src/frame/data.py:87`** — `ood=bool(row.get("ood", False))`. The organizers' parquet has no `ood`
   column and **we never inject our split's definition**, so `ood` is `False` for all **6252** rows.
   The evaluator sees **only ID buckets** → **10 buckets collapse to 3** → the OOD axis vanishes from
   the official metric.
2. **`frame_ood_v1` carries 1 `temporal_grounding` question** — a group FRAME should not have — which an
   unweighted bucket mean weighs **like a bucket of 3421**.

| Bucket | acc | n | weight |
|---|---|---|---|
| `aggregation` | 0.5170 | 2830 | **1/3** |
| `object_recognition` | 0.6092 | 3421 | **1/3** |
| `temporal_grounding` | **1.0000** | **1** | **1/3** |

**One correctly-answered question carries a third of our headline.** Reconstructed exactly
(`MATCH=True`) on all three ablation arms; the arms confirm the mechanism — when that question is
answered *wrong*, `pre_eval` collapses to 0.187 / 0.234.

The artifact swung **both ways** — 0.0 in zero-shot, 1.0 with LoRA:

| | Reported | Honest (n=1 bucket dropped) |
|---|---|---|
| Rung 00 zero-shot | 0.174 | ~0.251 |
| Rung 02 LoRA | **0.708** | **0.563** |
| **Improvement** | **+0.534** | **+0.312** |

**~43% of the headline gain is one question flipping.**

> ✅ **The LoRA gain is real.** `raw` went 0.262 → 0.566 (**+0.305**), matching the honest +0.312.
> **The model genuinely improved. Only the headline inflated.**

**This was known and lost.** Rung 00 documented it on 2026-07-10 (*"pre_eval 0.174 **FRAGILE** …
**trust raw 0.262**"*) — and **rung 02's ladder used `pre_evaluation_score` as its headline anyway.**
**The fix is the ladder's headline metric, not more prose.** *(Rung 00's attribution to an "all-ID local
slice" is also wrong: the split has 4000 OOD questions and the SDK flags none.)*

**Until `ood` is populated: `pre_eval` is unusable and `bucket_mean` is the only valid metric.**
`bucket_mean` derives ID/OOD from the `qID` prefix and excludes `temporal_grounding`, which is why it
holds where `pre_eval` does not. **Populating `ood` is a one-line, no-GPU atomic — the cheapest and
highest-return item open.**

### Connections to Block B

| Phase A | What it asks of B | What it returns |
|---|---|---|
| **A0** | the parquet + the taxonomy | **the correct metric** → B stops measuring the wrong thing |
| **A1** | the evaluator, unchanged | **a fact**: is our number perception or shortcut? |
| **A2** | selection by bucket mean | Δ per bucket, one variable |
| **A3** | the gates as regression tests | removes an entire class of loss |
| **A4** | nothing — B stays home | the artifact |

---

## §6 — Evidence base · what each paper ACTUALLY supports

> The section that prevents re-litigating. All verified by hand; abstracts in
> `documentacion/papers-corroborados.md`. **Citations the research agents misread are marked 🔧.**

| Paper | What it **DOES** support | What it does **NOT** |
|---|---|---|
| **2506.06232** — Challenging VLMs with Surgical Data *(DKFZ, our videos)* | VLMs do basic perception reasonably; **they collapse when medical knowledge is needed**. **Specialised medical VLMs underperform generalists.** | 🔧 **Gives no comparable counting accuracy.** Cannot calibrate our 0.432. |
| **2506.17337** — Generalist vs Specialist Medical VLMs | **Generalist + efficient FT ≥ specialist**, especially **transferring to OOD** → validates Qwen. | — |
| **2603.20116** — Chain-of-Adaptation ⭐ | Conventional SFT *"can inadvertently alter a model's pretrained multimodal priors, **leading to reduced generalization**"*. **Ablation: SFT 65.7 → RL only 67.4 → RL+format 83.7.** | **Does NOT support RL as the lever.** RL contributes **+1.7**; the **format** contributes **+16.3**. **Its premise (shortcut/damaged priors) was falsified by rung 05** — but the format result stands on its own. |
| **2405.10948** — Surgical-LVLM / **VP-LoRA** ⭐ | **Backbone Qwen-VL — our exact family, one generation back.** **Its ablation is the gold: instruction-FT alone = +16 (72.48→88.53); VP-LoRA adds only +2**; grounding helps mIoU, not accuracy. Own ficha: `literature/FICHAS.md:61`. | 🔴 **Does NOT support "unfreeze the aligner" — it says the opposite.** And **VP-LoRA ≠ unfreezing the aligner**: it is *"Mamba SS2D in LoRA layers"*, an exotic adapter. THE_MAP:164 conflated the two. **Our own ficha already concluded: *"skip VP-LoRA/Mamba/grounding (only +2)"*.** |
| **2504.13837** — Does RL Really Incentivize…? | RLVR **sharpens** what the base model can already do; **does not expand** the frontier. | — |
| **2506.07218** — Perception-R1 | Naive RLVR **does not improve perception**; with a perception-targeted reward, it does. | 🔧 **Does not say RLVR is useless** — that is its *motivation*. **Its reward is not verifiable:** needs **CoT trajectory annotations** + an **LLM judge in the loop**. We have neither. |
| **2603.17326** — FineViT | *"Their visual encoders frequently remain a **performance bottleneck**"*. | 🔧 **Does NOT support "unfreeze the ViT".** It is a **new encoder** trained from scratch. **No freeze-vs-unfreeze ablation.** |
| **2406.09246** — OpenVLA | Efficient VLA fine-tuning with LoRA on consumer GPUs. | 🔧 **Robotics**, not VQA. Transfer not evident. |
| **2408.02442** — Let Me Speak Freely? | *"Significant decline in reasoning **under format restrictions**"*. | **Does not condemn constrained decoding in general.** The damage hits **CoT reasoning**, not short atomic answers → constrain **only the final output**. |
| **2607.06420** — HoloCount | *thinking* mode improves counting **+10.6 to +15.4**. | A **benchmark**, not a method. |
| **2501.02385** — MedVP | Visual prompts improve medical VQA. | Requires **Grounding DINO fine-tuned on medical data** → **bboxes we do not have**. |
| **SurgeNetDINO** *(MIDL 2026)* | **Surgical-domain SSL pretraining helps** (4.7M frames). | **Weights are CC-BY-NC-SA** → would infect our released model. **ViT-S/B/L @224/336 ≠ Qwen3-VL's dynamic-resolution ViT** → not swappable without retraining alignment. |

### ✅ Evidence WE measured (not cited)

| Fact | How | What it changed |
|---|---|---|
| `aggregation` = 77% `number` + 22% `binary` | crosstab, val 6252 | 🔴 Killed "number = 50%". **`fo_class` and `number` are 1:1.** THE_MAP was right |
| FRAME has 2 groups (`temporal_grounding` = 1 q) | crosstab | ✅ Confirms **4 buckets** |
| 100% of `binary` is `aggregation` | crosstab | 🌟 `binary` is disguised counting, worth ~12.8% |
| **NO SHORTCUT — the model looks** | rung 05 ablation, 18,756 inferences | 🔴 **Kills the CoA branch's premise.** `fo_class` falls *below* its trivial floor when ablated |
| **`number` has only +8.0 pts of visual signal vs `fo_class`'s +51.9** | rung 05 | 🌟 **The bottleneck. 37% of the score.** Same encoder |
| **`pre_eval 0.708` is +14.6 pts of one question** | rung 05 verification | 🔴 See §5.1 |

### 🔴 What we have NO evidence for

| Claim | Status |
|---|---|
| **"Unfreezing the ViT unlocks perception"** | 🔴 **Corrected 2026-07-16.** An earlier draft said *"no citation supports it"* — **false: the citation exists and cuts against it.** 2405.10948 measures the visual path at **+2** vs instruction-FT's **+16**. **Still must be measured (Test B), but expected value DROPPED.** See §7.1 |
| Our 0.432 on `number` is below expectations | **Withdrawn.** No comparable number exists |
| 32B FP8 latency on the L40S | **NOT FOUND.** A research agent fabricated it |
| GRPO on 1×L40S | Exists in `ms-swift`, but official recipes **assume 6 GPUs + server mode** |
| How the 5 s are timed | **Still undocumented.** Presumably arrives with the template |

---

## §7 — CoA: format, not RL

> **🔴 Its original premise is DEAD.** CoA was justified by *"SFT altered the priors → the model learned
> a shortcut → preserve generalization"*. **Rung 05 falsified that**: there is no shortcut.
> **The pre-registered rule ruled the CoA branch out.**
>
> **What survives, and it survives independently:** the ablation shows **RL = +1.7, FORMAT = +16.3**
> (SFT 65.7 → RLVR without format 67.4 → RLVR + CoA format 83.7). **The reasoning format never depended
> on the shortcut premise.**

### §7.1 🌟 THE CONVERGENCE — two independent papers say the same thing

| Paper | **Text / format** side | **Visual path** side |
|---|---|---|
| **CoA** (2603.20116) — surgical benchmarks | format **+16.3** | RL **+1.7** |
| **Surgical-LVLM** (2405.10948) — **Qwen-VL backbone, our family** | instruction-FT **+16** | VP-LoRA **+2** |

> **~+16 on the format/instruction side. ~+2 on the visual-adapter side. Twice, with different methods,
> in surgery, one of them in our own backbone family.**

**And it matches what we measured:** the encoder is healthy (`fo_class` +51.9 visual signal) while
`number` barely extracts (+8.0). **A healthy encoder with a poor output points at the format, not the
eyes.**

#### What this changes — and what it does not

- 🔻 **It lowers Test B's expected value.** This was not known when Test B was pre-registered.
- ⚠️ **It does NOT cancel it.** **VP-LoRA (Mamba SS2D) ≠ plain LoRA on the ViT** → the +2 is **not a
  direct prediction**. It is the closest evidence that exists, and it points small. Not proof.
- 🔴 **Do not pivot the roadmap on this without measuring.** That is exactly the error of 2026-07-15
  (§10.5): "correcting" on an unverified premise. **`number-probe` is the adjudicator** — if the model
  **does not count**, the branch is format and this convergence explains it; if it **counts badly**,
  Test B regains its point.
- ⚠️ **The +16 of instruction-FT is already spent** (rung 02: `fo_class` +0.42, `number` +0.29). The
  live question is not *"should we instruction-tune?"* but **"do better-STRUCTURED instruction data buy
  more on top?"** — which is literally CoA.

#### 🔺 Rising in priority: the cheap question nobody has asked

> **How much of the +16.3 survives with SFT on the CoA format, WITHOUT RL?**

**With two papers converging, this is likely the best value/cost item open** — and it needs **no RL and
no new infrastructure**. **Cost is real:** synthesising `<general description>`, `<evidence>`,
`<thought>` for ~13.7k examples is a data project.

**Objections that fall:** the 300-char cap does not bite (CoA is generated **internally**; only
`<answer>` is emitted). "FRAME does not score reasoning" aims at the wrong target — the benefit is
**preserved generalization → OOD → 50% of the exam**.

---

## §8 — The `src/frame/serve/` package

`EXPERIMENT_REPO_STRUCTURE_SPEC` §VIII (binding) requires **ONE package in `src/`**. A sibling package
would violate it. **The solution is a SUBpackage that makes the A/B boundary explicit.**

### The problem, measured

```
src/frame/__init__.py
    from frame.run import run_baseline      ◄── THIS
                              │
                              └─► pandas · focus.evaluation.evaluator
                                  · TransformersJudge · focus.enums

  ⇒ `import frame` DRAGS THE JUDGE AND PANDAS INTO THE DOCKER IMAGE.
```

**The product is almost clean already:** `config.py` is pure; `engine.py` has a single `focus`
dependency; everything else (`data/run/delta/split/qualitative`) is Block B. **The boundary exists de
facto. The only thing breaking it is `__init__.py`.**

```
src/frame/
├── serve/                    ◄── BLOCK A · the ONLY thing in the Docker image
│   ├── __init__.py               (no heavy imports)
│   ├── config.py                 moved as-is
│   ├── prompt.py                 prompt assembly ── today inside engine.py
│   ├── engine.py                 load / predict / unload
│   ├── decoding.py               ✗ NEW — constrained decoding (A3)
│   └── fo_defs.py                local copy of FO_DEFINITIONS_FILE
│                                 └─ kills the product's last `focus` dependency
├── data.py · run.py · delta.py · split.py · qualitative.py   ◄── BLOCK B
└── __init__.py               ── do NOT import run.py from here
```

**The rule that makes it real, checkable with one `grep`:**

> **`serve/` NEVER imports from the lab. The lab MAY import from `serve/`.**
> `grep -rE "from frame\.(data|run|delta|split|qualitative)" src/frame/serve/` → **empty**.

**Gain:** the Docker image copies `serve/` + weights and nothing else. The boundary goes from convention
to **mechanics** — today it is a convention that `__init__.py` already broke unnoticed. **And A and B
keep sharing the same engine**, which is what makes the lab's number mean anything.

**Cost:** it is a refactor of shared infrastructure → **must be agreed with RodMed before starting.**

---

## §9 — Discarded, and why

| Proposal | Why not |
|---|---|
| **RLVR with exact-match reward** | **+1.7 over SFT** (CoA ablation). The value is the format, not the RL |
| **Perception-R1 as-is** | Its reward needs **CoT trajectory annotations + an LLM judge in the loop**. We have neither |
| **SurgeNetDINO as encoder** | **CC-BY-NC-SA** infects our released model (Qwen is Apache-2.0), **and** the swap = retraining the whole alignment |
| **YOLO→ROI / scene-graphs / MedVP** | **No bboxes** in the released data (verified). No public surgical open-vocab detector. A misplaced ROI is worse than none |
| **QLoRA 4-bit on the 8B** | 18 GB of 48. Quantising adds latency and solves nothing. Reserved for the 32B |
| **Unsloth** | Lags on Qwen3-VL. `ms-swift` has native support + `freeze_vit`/`freeze_aligner`/`max_pixels` |
| **SGLang / PagedAttention / speculative decoding** | All solve throughput, batching or long context. **We are batch=1, one image, ≤32 tokens, 18 of 48 GB, 0.59 s of 5 s.** They accelerate what already spares |
| **DPO** | Aligns with **human preferences**. Our metric is exact-match + a hidden judge |
| **Changing backbone family** | No evidence; cost = restart. 2506.06232 + 2506.17337 validate the generalist |

---

## §10 — Corrections on record (so we don't re-litigate)

1. **RLVR was sold as "THE_MAP's biggest hole".** The CoA ablation says **+1.7**. Promoted what does not
   work and rejected what does (the format).
2. **LLaVA-Med curriculum proposed.** With ~14k domain pairs, stage 1 adds little. **Dead.**
3. **"Our 0.432 is below expectations."** No support: no comparable number. **Withdrawn.**
4. **"Unfreeze the aligner"** was repeated, inherited from THE_MAP. The diagnosis points at the **ViT** —
   see #6.
5. 🔴 **"`aggregation` = `number` → number is 50% → a point there is worth two in `fo_class`."**
   The crosstab killed it: **38.6% vs 38.3% — a 1:1 pair. THE_MAP was right and it was "corrected" on a
   premise that was never verified.** *(What saves the episode: verification was made a mandatory task,
   so it fell in hours, not in September.)*
6. 🔴 **Everything was audited EXCEPT the paper that supported the claim being withdrawn.**
   *(Caught by RodMed's review, 2026-07-16.)* §6 said *"unfreeze the ViT"* had **no citation**.
   **False: `THE_MAP.md:164` names it** — Surgical-LVLM / **VP-LoRA (2405.10948)** — and it is in **our
   own corpus** (`literature/INDEX.md:62`, **with a full ficha at `FICHAS.md:61`**). **The papers the
   research agents supplied were audited; our own library was never opened.** Same sin the agents were
   criticised for.
   > **The nuance makes it worse, not better:** the conclusion survives — but **for the opposite reason**
   > than written. Not *"there is no citation"* but ***"the citation exists and says the opposite"***:
   > VP-LoRA = **+2**, instruction-FT = **+16**, and **VP-LoRA is not even aligner-unfreezing**. **Our
   > own ficha already said "skip VP-LoRA".** The strong argument was written at home and went unread.
   > → **§7.1**
7. 🔴 **A broken `pre_eval` was taken at face value throughout.** *(Project-wide, not one person's:
   rung 00 documented it on 10-Jul and rung 02 put it in its ladder anyway.)* See **§5.1**.

**Research-agent errors:** one agent **fabricated papers and data** — including 32B-on-L40S latency with
"HIGH" confidence, the number that carried its recommendation to change backbone, and an **astrophysics**
paper cited as VLM counting. The other misread **Perception-R1**, **FineViT** and **OpenVLA**.
**Forensics: `documentacion/r-adjudicacion.md`.**

---

## §11 — Decisions taken

- [x] **A1 image ablation = the atomic.** ✅ **DONE** → rung 05, no shortcut.
- [x] **The 8B was chosen against the pod, not the exam. The 32B is back on the table**, conditional on
      **measuring the L40S.** → the L40S measurement is the 32B's gate, not optional.
- [x] **`src/frame/serve/` subpackage.** ✅ **APPROVED.** ⚠️ **Prerequisite: agree with RodMed** — shared
      infrastructure. **Do not start without that.**
- [x] **CoA.** Explored **"when the time comes"**, and when it does: **the cheap version first (CoA
      format via SFT, NO RL)**. 🔺 **§7.1 raises its priority** — but it still waits for `number-probe`.
- [ ] **What goes into `THE_MAP.md`?** ⏸️ **This document is that proposal.** See below.

---

## §12 — What is still unmeasured

| Unknown | Why it blocks | Cost |
|---|---|---|
| **Real latency on the L40S** | Blocks **three** decisions at once: 32B, resolution, and CoA's extra tokens. **We have never run one question on the real hardware** | 1 pod session |
| **Does the model count, or emit a constant?** | Decides what Test B means | ~0–15 min (`number-probe`) |
| **Is the ViT the ceiling for counting?** | The bifurcation — and §7.1 says expect little | Test B |
| **How much of CoA's +16.3 is the format alone, without RL?** | Decides if CoA is viable in the remaining weeks | 1 experiment |
| **How are the 5 s timed?** | Undocumented. Arrives with the template | — |
| **Does `Qwen3-VL-32B-Instruct-FP8` exist? Licence?** | The 32B's gate | 5 min |

---

## 📋 Merge instructions — for whoever folds this into `THE_MAP.md`

> **This file is an interim artifact.** It exists so the team can review Block A's strategy before it
> lands in `THE_MAP.md`. **Once merged, delete this file** — two strategy documents that can diverge is
> exactly the failure mode §10.6 describes.

**Merge in this order. Items 1–2 are corrections of things THE_MAP currently states wrongly; they matter
more than the additions.**

1. 🔴 **`THE_MAP.md:164` — the Phase 3 REDIRECT is wrong on its own citation.** It reads: *"we **froze
   the aligner**, but Surgical-LVLM's contribution (**VP-LoRA**, arXiv 2405.10948) is LoRA on the
   *visual-perception path* — we froze exactly what the SOTA surgical VQA model adapts"*, and makes
   **"#1 unfreeze aligner/merger"** the highest-expected-OOD next experiment.
   **Two errors:** (a) **VP-LoRA is not aligner-unfreezing** — it is *"Mamba SS2D in LoRA layers"*, an
   exotic adapter; (b) **the paper's own ablation measures VP-LoRA at +2** while instruction-FT is
   **+16** — and **our own `literature/FICHAS.md:61` already concluded *"skip VP-LoRA/Mamba/grounding
   (only +2)"***. **Demote "#1 unfreeze aligner".** Replace with §7.1's framing.

2. 🔴 **Any headline `pre_evaluation_score` in THE_MAP is inflated.** Fold in **§5.1**. The ladder's
   headline metric must become **`bucket_mean`** until `ood` is populated (`src/frame/data.py:87`).
   **Rung 02's `0.708` is honestly `0.563`, and the gain over zero-shot is `+0.312`, not `+0.534`.**
   **The LoRA gain is real** (`raw` 0.262 → 0.566) — say so, or the correction will read as bad news
   when it is not.

3. **Add the measured bucket weights (§1)** — they replace assumptions THE_MAP made about `fo_class`
   vs `number`. **THE_MAP's original "leverage → fo_class + number" was right**; a later draft
   "corrected" it wrongly and has been reverted.

4. **Add rung 05's verdict (§2)**: no shortcut, the model looks, and **`number`'s +8.0 vs `fo_class`'s
   +51.9 visual signal** is the bottleneck. This retires the shortcut hypothesis THE_MAP carried.

5. **Add §7.1 (the convergence)** as the reason Test B's expected value dropped — **without cancelling
   it**. `number-probe` adjudicates.

6. **Keep §6 verbatim if possible.** It is the only place where each paper's actual claim is separated
   from what agents reported it said. It is what stops this from being re-litigated a third time.

**Do NOT merge:** §10 as-is (it is a working log, not strategy) — but **do carry #5, #6 and #7**, since
each is a live trap someone else can fall into. **§8 (`serve/`) is approved but blocked pending
agreement with RodMed** — do not present it as decided.

**Conflicts:** if anything here contradicts `documentacion/overview.md`, **the overview wins** (§0). If
anything here contradicts a measured number in an `experiments/*/RESULTS.csv`, **the measurement wins**.
