---
question: Does a newer-generation backbone close the gap by itself?
verdict: NO. Three generations of backbone buy +0.036 zero-shot; our own fine-tuning buys +0.317. And the zero-shot 27B sits BELOW the template-aware floor in both halves (margin_ID -0.0666, margin_OOD -0.1367), so that +0.036 is movement beneath the floor, not skill.
status: MEASURED
date: 2026-07-29
measured_in: experiments/23-backbone-screen/RESULTS.csv (runs/23a_qwen36_27b_bf16)
question_derived: false
---
# Finding: the generation story is real, small, and pointed at the right cell

[[judge-swap-is-not-the-gap]] closed the instrument escape, leaving the backbone as the
step-change lever, and [[latency-budget-is-pooled]] had unblocked it. `Qwen/Qwen3.6-27B`
(2026-04-21, Apache-2.0, `Qwen3_5ForConditionalGeneration`) was screened **zero-shot** on the
full 6,252-question eval set — same prompt, same message shape, same greedy decode, same
`max_pixels`, same judge as the ladder. ~$4.40 of pod, no training.

| bucket | rung 00 zs **8B gen-3** | rung 23a zs **27B gen-3.6** | Δ generation | rung 06 ep3 **FT 8B** |
|---|---|---|---|---|
| `aggregation` ID | 0.1885 | 0.1937 | +0.005 | **0.4346** |
| `aggregation` OOD | 0.3003 | 0.3136 | +0.013 | **0.5547** |
| `object_recognition` ID | 0.2932 | 0.3264 | +0.033 | **0.6520** |
| `object_recognition` OOD | 0.2409 | **0.3313** | **+0.090** | **0.6485** |
| **`bucket_mean`** | 0.2557 | **0.2913** | **+0.036** | **0.5724** |

**Three generations and 3.4× the parameters buy +0.036. Our own fine-tuning buys +0.317** —
nearly 9× more. A zero-shot gen-3.6 27B does not reach half of our fine-tuned 8B.

## 🔴 AMENDED 2026-07-29 — the margins are negative, and that is the real verdict

The table above reports raw accuracy. It was written before this rung's canonical
`stratified.json` existed, so it never carried the template-aware floors — the one reading
RULES §10 makes binding. With gold supplied:

| | margin_ID | margin_OOD |
|---|---|---|
| **rung 23a zero-shot 27B** | **−0.0666** | **−0.1367** |
| rung 06 ep3 fine-tuned 8B | +0.2225 | +0.1448 |

**The zero-shot 27B scores BELOW the trivial floor in both halves** — worse than a baseline that
emits each template's modal answer. So the +0.036 "generation lift" is movement *beneath the
floor*, which is not skill, and the +0.090 on `object_recognition` OOD must be read the same way:
better than another sub-floor model, still sub-floor.

This does not change the NO-GO. It makes it stronger, and it removes the one datum that pointed
the other way as evidence of capability. The lesson is procedural: **a rung that ships raw
accuracy without its floor has not been read yet**, and this one shipped a decision note before
its canonical report existed.

⇒ **NO-GO on the migration**, exactly as pre-registered before the run ("lands near rung 00 →
the generation story is wrong for this task and the migration is dead for ~$3"). The price was
never the screen: it is `transformers` 4.57 → 5.x across engine, container and serving
(`CONSTITUTION.md:70` says in so many words *"do NOT jump to 5.x"*), ms-swift support for
`Qwen3_5MoeForConditionalGeneration` that nobody has verified, and an 80 GB QLoRA pod — two to
three of the six remaining weeks, bought with +0.036.

🔒 **23b (`Qwen3.6-35B-A3B-FP8`) does not run.** Its gate was 23a and 23a failed it.

## The one result that argues the other way, and it is worth keeping

The lift is **not uniform**. `object_recognition` **OOD** gains **+0.090** — 2.5× the headline
lift — and that is precisely the cell that collapsed −0.150 on the platform
([[leaderboard-metric-vs-our-headline]]). `aggregation` gains essentially nothing (+0.005 ID).

So the newer generation *is* better at exactly what we are failing at, on unseen centres. What
it is not is better **enough** to skip training. This does not license the migration; it does
say that if a backbone swap ever becomes cheap, `object_recognition` OOD is where it would pay.

## Three things the screen proved for free

1. **Latency is a non-issue at 27B.** 0.585 s/q mean, **p99 0.851 s**, max 1.29 s — a 27B is
   *faster* than our fine-tuned 8B (0.79 s/q) against a 5 s ceiling. Model size is not
   latency-bound for FRAME. Confirms [[latency-budget-is-pooled]] at a new scale.
2. **The zero-shot 27B emits tokens our fine-tuned model cannot.** It answers `0` and names
   `Mesh` unprompted — a numeric zero appears nowhere in our numeric gold
   ([[zero-is-format-localized]]) and `Mesh` has **zero examples anywhere in our data**
   ([[open-class-vocabulary]]). Our supervision cannot teach either. That is a real, narrow
   argument for external data over a bigger backbone.
3. **The 2026 Qwen line is a reasoning model and must be told not to think.** With the default
   template it emits a chain of thought, `max_new_tokens=64` truncates it before the answer,
   and the run scores **0.0000** — a format failure that reads exactly like incapacity. Fixed
   with `enable_thinking=False`, which the CoT-degrades-grounding literature says is the right
   default anyway.

## Deployment caveats recorded, since a later run will hit them

- **FP8 is mandatory to fit the 48 GB eval GPU** — 30.9 GB vs 55.6 GB for bf16
  (`challenge_design.txt:437`). The screen ran **bf16** because the hub's `finegrained-fp8`
  kernels carry **no sm_120 build**, so FP8 cannot run on a Blackwell RTX PRO 6000 at all. bf16
  is the *upper* bound of the FP8 build's score, so the NO-GO is not an artefact of precision.
- Trap for whoever builds the next env: installing `accelerate` pulled `torch 2.13.0+cu130` over
  the image's `2.8.0+cu128` and desynced `torchvision`, whose symptom
  (`Could not import module 'Qwen3_5ForCausalLM'`) blames the model. Uninstall torch/torchvision
  from the venv and let it fall back to the image's matched pair.
