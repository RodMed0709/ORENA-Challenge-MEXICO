# 16 — count probes: three cheap measurements before any rung-16 training

## The ladder (where this sits)

| rung | what it changed | headline `bucket_mean` |
|---|---|---|
| 00 baseline | zero-shot Qwen3-VL-8B | 0.2557 (below floor everywhere) |
| 02 lora-sft | LoRA on the LLM | 0.5486 |
| **06 vit-lora ep2** | + LoRA on the ViT | 0.5667 |
| **06 vit-lora ep3** ⬅ current best | the epoch-matched control (`checkpoint-2580`) | **0.5724** |
| 13 wise-ft | weight interpolation, no training | NO-WIN (3 α) |
| 14 appearance-aug | colour/WB augmentation during LoRA | **NULL, and worse than ep3** (ID ALL −0.0239, CI excludes 0) |
| 15 count-target | structured `number` target | NULL under the ID-AND-OOD conjunction |
| **16 (this)** | *nothing is trained here* — three probes that decide what rung 16 becomes | — |

## Why probes instead of a rung

Rung 06 ep3 closed the R2 wave and produced the sharpest number we have:

> **`number_margin_OOD` = 0.0 — exactly the floor.**

Our best checkpoint adds *literally nothing* over "always answer the mode" on OOD counting.
Every remaining idea for `number` (minted zeros, pointing supervision, tiled counting) costs a
7.5 h training run, and a literature sweep (2026-07-27, five parallel readers, ~60 papers
verified) found each of them gated on a question we have never measured. The probes measure
those questions first. Total ≈ 5 GPU-hours against three 7.5 h runs.

| probe | question it settles | cost | kills what if it fails |
|---|---|---|---|
| **16a `zero_probe`** | Can the model emit `0` at all — and did our SFT take that away? | ~1 h, no new data | the whole minted-zeros plan, or redirects it to regularisation |
| **16b `detector_vs_gold`** | Is the gold count something a detector (or a human) can see in the frame? | ~2 GPU-h | pointing **and** every `number` lever, if the gold is not frame-visible |
| **16c `len_points`** | Does deriving the count as `len(predicted_points)` beat verbalising it? | ~2 GPU-h | the largest single published effect available to us (+80.9 pts OOD) |

**A fourth probe was already closed for zero GPU.** The tiling/upsampling family: our
`max_pixels = 921,600` engages on exactly one resolution (lapchole 1280×720, 5,036 of 15,213
frames) at a **2.2% linear** cost. No frame reaches the ViT meaningfully downscaled, so SAHI's
mechanism — recovering detail destroyed by a forced downscale — has nothing to recover. See
`src/frame/config.py:32-52` and `context/decisions/resolution-is-not-the-gap.md`.

## 16a — the zero probe

**The gap it addresses.** Our training set contains **no negative example** in any numeric or
class format: 0 of 2,495 per-class count golds are `0`, 0 of 2,628 total-instance golds are `0`,
0 of 8,969 `fo_class` golds are `none`. Only `binary` carries negatives, and those are
co-occurrence questions about class *pairs*. The model has never seen `0` as an emittable
numeric answer.

**Why probe before minting.** HoloCount (arXiv:2607.06420, Table 3, Null-Target Prompting)
measures **Qwen3-VL-8B at 96.4%** on absent-object questions — ahead of Gemini-3-Flash (62.4%)
and Gemini-3.1-Pro (55.2%), with the verbatim finding that *"larger models become reluctant to
output zero, even when no target exists."* If the capability ships in our backbone and our
checkpoint lost it, the disease is our own SFT's answer prior and the cure is regularisation —
not 37k synthetic rows that would push the count bias further down (we sit at −0.66, and ~8.4%
of minted zeros would be wrong *in the direction of zero*).

**Design.** Paired: same frames, same questions, `base` vs rung-06 **ep3**. Two arms (ABSENT /
PRESENT) × two formats (`number` / `binary`), balanced across ID and OOD, one absent + one
present question per frame so no frame is over-represented.

🔴 **Read `emit_rate`, not `acc`.** The ABSENT label comes from a **closure assumption** — that
the `fo_class` gold names every class present. Measured non-circularly against the 1,008 `no`
co-occurrence binaries: **0.00% false positives**, but **8.4% under-naming** (n=333; a second
cross-check agrees at 10.7%). So ~1 in 12 ABSENT labels is wrong, always in the same direction.
`emit_rate` is a property of the model's *output* and is therefore noise-free; `acc` inherits
the 8.4%. The PRESENT arm is the control that stops "always answer 0" from scoring well, and
the harmonic mean is the two-axis score (POROver discipline, arXiv:2410.12999).

**Pre-registered readings** — written before the run, in the notebook's opening cell:

| outcome | verdict |
|---|---|
| base high, ft ≈ 0 | our SFT destroyed it → regularisation, minting demoted |
| both ≈ 0 | capability absent in this domain → minting becomes the candidate, dosed 3:1–2:1, adversarially sampled |
| both high | no zero problem exists → drop the branch |
| ft high on PRESENT too | a zero *attractor*, not a capability → read the harmonic mean, never `acc_absent` alone |

## Layout

```
16-count-probes/
  README.md               <- this file (opens with the ladder, per the structure spec)
  16a_zero_probe.ipynb    <- the run generator; config inline, SMOKE toggle
  _models/zero_probe.py   <- importable library: inventory, probe set, scoring
  _tools/build_notebook.py<- folder-private; regenerates the notebook so cells diff as text
  runs/                   <- gitignored; each run owns its artifacts
```

## Provenance

Literature that motivated each probe is staged in the session scratchpad pending a
`literature/vlm-techniques/` merge: agentic loops, tiling/counting, negatives/absence,
pointing + dataset audit, pseudo-QA synthesis. The two corrections that sweep produced are
already committed (`b4642e4`): the v05 multi-task claim was inverted at six sites, and the
`~56%` visual-token claim in `config.py` was retracted.
