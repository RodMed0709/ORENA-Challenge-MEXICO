# Experiment 14 — appearance augmentation during LoRA (`14-appearance-aug`)

> Half of `bucket_mean` is OOD, and by MARGIN the model adds **less** there
> (rung 06: `margin_ID` +0.207, `margin_OOD` +0.148). Rung 12 only ever
> transformed pixels at *inference*. This rung transforms them during *training*
> and leaves the evaluation set alone — the axis rung 12 never tested.

## Ladder

| Notebook | Rung | Metric (`bucket_mean`, canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.256 | baseline |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.549 | PASS — LoRA on the LLM |
| `../06-vit-lora/06_vit_lora.ipynb` | 06 | **0.5667** | 🟡 PARTIAL — **the control for this rung** |
| `../12-image-processing/12_composite_pod.ipynb` | 12 | — | NEGATIVE — inference-only transforms (−0.026 → −0.056) |
| `14_appearance_aug.ipynb` | 14 | *pending* | — |

## The single variable

**The training images carry a seeded appearance augmentation. Nothing else moves.**

Same data, same recipe, same seed, same epochs, same checkpoint cadence as rung 06.
`swift sft`'s argv differs from rung 06's in exactly one place — the value of
`--dataset` — and `aug_export.diff_vs_rung06()` proves it by diffing two argv lists
built from rung 06's own `_swift_args`, not by asserting it in prose.

**The evaluation set is untouched.** That is the whole thesis: train for robustness,
measure on the real distribution (Jong 2025; Medeiros 2026). `gate_eval_untouched`
checks that no inference-side appearance flag is set.

## The augmentation, and where every number came from

Colour and illumination **only**. No sharpening (`unsharp` is measured −0.056
monotone on this exact model), no geometric ops, no crops (Ramesh 2023 measured
low-resolution multi-crop *hurting* surgical tasks: −3.5 % / −4.5 % F1), no blur.

Every citation below resolves against **`literature/preprocessing/FICHAS.md`** (ids `p01`–`p42`),
not against the older, unrelated `literature/FICHAS.md` at the repo root:
Jong 2025 = Tier-1 #1 `p01` · Medeiros 2026 = Tier-1 #3 `p03` · Afifi & Brown 2019 = Tier-1 #9
`p09` · Ramesh 2023 = Tier-2 #12 `p12` · Wang 2024 = Tier-2 #19 `p19` · Ali 2019 = Tier-1 #8
`p08` · Nie 2023 = Tier-2 #13 `p13` · Kim 2025 = Tier-2 #27 `p27`.

| Operator | Parameter | Source |
|---|---|---|
| White-balance error (primary) | colour temperature ∈ {2850, 3800, **5500**, 6500, 7500} K, uniform | Afifi & Brown, ICCV 2019 §5 — the five fixed temperatures of the WB-sRGB rendering set. 5500 K is the identity anchor, so 1/5 of rows keep their white balance. |
| WB implementation | Bradford chromatic adaptation in **linear** light | **Ours** — Afifi's learned 3×9 mapping needs a 17,970-image dataset we cannot bundle offline. Declared deviation; see `context/14-appearance-aug/CONTEXT.md`. |
| Brightness | additive on HSV V, `c ∈ {0.10, 0.20}` (severity 1–2) | ImageNet-C constants, via Wang 2024, which states verbatim it uses the `imagecorruptions` severity settings. |
| Dark | the same constants, negated | **Ours** — Wang's taxonomy names *Dark* as a distinct illumination corruption but publishes no constants. Declared deviation. |
| Contrast | scale about the per-image mean, `c ∈ {0.40, 0.30}` (severity 1–2) | ImageNet-C constants, via Wang 2024. |
| Severity draw | uniform over {0, 1, 2}; 0 = uncorrupted | Wang 2024 ("severity 0 signifies the original image remains uncorrupted"); severity 1–2 is the dose `literature/preprocessing/FICHAS.md` → "What to steal" §3(b) prescribes. The *uniform weighting* is ours. |

Realised marginals (measured, not assumed): 1/5 of rows keep their WB, 1/3 take no
illumination corruption, **1/15 are fully untouched**.

## Layout

```
14_appearance_aug.ipynb      the ONLY launcher (inline cfg + SMOKE toggle)
_models/appearance.py        the augmentation library — pure, seeded, no GPU
_models/aug_export.py        rung 06's export/train path, with the augmented JSONL
_models/quality_diag.py      the zero-GPU frame-quality diagnostic (within-video)
_tools/test_appearance.py    the off-pod verification (6 checks, no GPU/data)
_tools/build_notebook.py     regenerates the notebook JSON from plain text
RESULTS.csv                  ledger-shaped, one row (frame.ledger reads it)
RESULTS_epochs.csv           per-epoch margins, incl. `number` at EVERY epoch
RESULTS_paired_ci.csv        paired, video-clustered CIs vs rung 06
RESULTS_quality_diag.csv     the folded-in diagnostic
```

## Gates (all RAISE; none may be disabled)

| Gate | What it protects |
|---|---|
| **G-OFF** 🔴 | flag OFF → our `train.jsonl` is **sha256-identical** to rung 06's. Without it, "single variable" is a claim. |
| **G-SHAPE** 🔴 | the ON JSONL differs from the OFF JSONL in the `images` path and nothing else — same rows, same order, same text. |
| **G-D** | `swift sft` argv differs from rung 06's in `--dataset` only (catches a drifted rank/LR/epoch/`freeze_vit`). |
| **G-EVAL** | no inference-side appearance flag is set. |
| **G-A1/2/3** | flag OFF returns the *same object*; the draw is a pure function of (seed, qID) across processes; no excluded operator family can be smuggled into the policy. |
| **G1** | LoRA reached the ViT and is still LoRA (rung 06's log read). |
| **GATE 0** | every evaluated qID has gold, or the floors understate and the margins inflate. |

## Reading the result

Headline `bucket_mean`; the pre-registered target is **`margin_OOD`**, with a
**paired, video-clustered** CI (effective n ≈ 38 videos, not 6,252 questions).
The full decision rule and the stopping tiers are in
`context/14-appearance-aug/CONTEXT.md`, written before any number existed.

🔴 **Report the `number` margin at every epoch.** rung 06's own trajectory is
+0.015 → +0.013 → **+0.000** on OOD: training erases counting, and `acc_OOD`
selection picks the checkpoint that erased more. Never select last-by-default.

## Folded-in zero-GPU diagnostic (not an arm, no extra training)

Blur (Laplacian variance), specular fraction and luminance per cached frame,
correlated with rung 06's per-question correctness **within video**. Pooled
correlation is confounded by scene — that is precisely how rung 12d manufactured
a winner — so the within-video contrast is the primary statistic and the pooled
number is printed beside it, labelled CONFOUNDED.

⚠️ There is **no inference-side frame-selection lever** in FRAME: the track hands
us one extracted frame (`vendor/orena-focus/src/focus/enums.py:23`). Anything this
diagnostic motivates can only act on training.
