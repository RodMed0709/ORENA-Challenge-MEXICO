# rung 17 — can the CoA generator actually see?

> **Status: PRE-REGISTERED, UNRUN.** Inference only, no training, nothing merged.
> Pre-registration: `context/17-generator-probe/CONTEXT.md`. Read it before the numbers.

## Ladder

| Notebook | Rung | `bucket_mean` (canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.2557 | our 8B zero-shot — **below floor everywhere** |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.5486 | LoRA, LLM only |
| `../06-vit-lora/06_vit_lora.ipynb` | 06 | 0.5667 | LoRA, ViT+LLM — the ladder's best |
| `17_generator_probe.ipynb` | **17** | _pending_ | 🚧 Qwen3-VL-32B, zero-shot, **as a perceiver** |

*Headline = `bucket_mean` (`frame.metrics`), read as **MARGIN over the template-aware floor**
(RULES §10). Floors: 0.337 ID / 0.460 OOD.*

## The question

Rung 09 hands the generator the gold and asks it to write reasoning that *derives* it. **So a
perception failure can never appear as a wrong answer** — it appears as `<evidence>` describing
a scene that is not there, attached to a correct `<answer>`. Nothing downstream catches that:
the answer matches by construction, and the Qwen judge-mirror is a text model that cannot see
the frame either.

This rung removes the gold and asks the generator the question directly. One number decides
whether the 2k cold-start generation is worth running at all.

## What it is NOT

It does **not** test whether CoA-format SFT helps. That is settled in the literature and
recorded in `context/decisions/coa-sft-published-null.md`: scaffold-SFT **without** RL is a
wash (62.0 vs bare-gold SFT's 65.7 on EndoVis2018, same backbone, same scaffold) and the +18
belongs to RLVR. This rung only asks whether the teacher can see.

## Decision rule (fixed before any number)

| margin ID **and** OOD | verdict |
|---|---|
| both > 0 | 🟢 **GO** — generate the 2k cold-start scaffolds |
| CI crosses 0 | 🟡 scaffolds are gold-anchored prose; generate only behind the human eyeball gate |
| either < 0 | 🔴 **STOP** — a below-floor teacher cannot supervise perception |

⚠️ Asymmetric on purpose: a general VLM also loses points to *task format*, so a weak score
under-states perception. **Read a negative as decisive, a positive as directional.**

## Files

| Path | What |
|---|---|
| `_models/probe.py` | engine: GPU-headroom gate, generation (delegates to rung 09), canonical scoring, the coded verdict |
| `17_generator_probe.ipynb` | the only launcher |
| `RESULTS.csv` | ledger-shaped row, written when it scores |

Generation and selection are **rung 09's own** (`../09-coa-sft/_tools/gen_onpod.py`, stage
`blind_probe`). This rung adds the canonical scoring that stage lacked — it reported raw
accuracy, and raw accuracy against a 0.34/0.46 floor is not a statement about perception.

## 🔴 Operational

The 32B is **63 GB in bf16**. It does not co-reside with an 8B LoRA training run on a 96 GB
card. `probe.assert_gpu_headroom` RAISES below 70 GiB free — the failure it prevents is silent
and lands on someone else's multi-hour training, not on this probe.
