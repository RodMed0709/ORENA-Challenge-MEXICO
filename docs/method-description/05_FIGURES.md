# 05 — Figure plan

**Fig. 1 is mandatory. A submission without a clear main architecture figure is disqualified for
awards.** That makes this the highest-priority item in the whole dossier.

Naming is enforced and automated: `Mexico-Oxford_TEAM_fig_<X>.pdf`, X starting at 1, PDF, ≤10 MB
each, ≤10 files. Reference them in text as `[Fig. 1]`, `[Fig. 2]`, …

---

## Fig. 1 — Main architecture (MANDATORY)

The brief from the organizers: *comprehensively illustrate your complete technical pipeline
end-to-end: from video/image inputs and prompt injection, through baseline adaptations and novel
architectural modules, to the final VQA output.*

Content it must show, left to right:

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  INPUT                                                                  │
 │  batch.json (declared layout)  ──▶  frames/<qID>.png   (ONE frame)      │
 │  question text                                                          │
 │  fo_definitions.json (mounted by the platform, SDK fallback)            │
 └─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  PROMPT ASSEMBLY                                                        │
 │  system = SYSTEM_PROMPT_PREFIX + FO class definitions                   │
 │  user   = [image, question]           (no CoT, no few-shot)             │
 └─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Qwen3-VL-8B-Instruct              ★ = LoRA r=8, α=32, dropout 0.1      │
 │                                                                         │
 │   ┌──────────┐    ┌───────────────────────┐    ┌────────────────────┐   │
 │   │  ViT  ★  │──▶ │  ViT→LLM CONNECTOR ★  │──▶ │      LLM   ★       │   │
 │   │          │    │  visual.merger.fc1/2  │    │                    │   │
 │   │          │    │  deepstack_merger ×3  │    │                    │   │
 │   └──────────┘    └───────────────────────┘    └────────────────────┘   │
 │        max_pixels = 1280×720          ▲                                 │
 │                                       └── unreachable via `all-linear`; │
 │                                           named explicitly (rung 32)    │
 └─────────────────────────────────────────────────────────────────────────┘
                    │  greedy, max_new_tokens = 64
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  ANSWER-BOUNDARY GUARDS                                                 │
 │  normalize_answer      strips the trailing period that auto-fails       │
 │                        Number.verify / Binary.verify                    │
 │  clamp_class_tokens    drops fo_class tokens absent from the SDK        │
 │                        registry, read at RUNTIME (not hard-coded)       │
 │  cap at 300 characters                                                  │
 └─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
                answer.json
```

**Emphasise the starred connector.** It is the one genuinely non-obvious thing we did, and the
figure is where a reviewer will notice it.

**Owner:** Yingyu (drafting), Rodrigo/Leo (technical review).
**Tool suggestion:** draw.io or TikZ; export to PDF. Do not use a raster screenshot.

---

## Fig. 2 — Training pipeline and corpus construction

```
challenge data (heico + lapchole)
        │
        ├── rung 18 minted zero-count rows        667 rows, POPE-style
        │      adversarial sampling weighted by measured per-class counts
        │      + question-surface paraphrase, format-tail dropout
        │      = 14,415 rows
        │
        └── 4,969 rows generated from 30 of 38 public test videos
               ↓
        merged corpus  19,384 rows
               ↓
        LoRA SFT, 5 epochs, lr 2e-4, cosine, bf16, effective batch 16
               ↓
        checkpoint selection on 1,283 held-out questions (8 videos)
        pre-registered memorisation criterion → epoch 4 of 5
               ↓
        swift export --merge_lora  →  17 GB merged checkpoint  →  Docker (offline)
```

Show the held-out split honestly, including the caveat that 8 of 10 `heico` test videos were
promoted into training, so the local OOD axis is *unseen video of a seen procedure*.

**Owner:** Yingyu.

---

## Fig. 3 — Epoch sweep and the memorisation signature

A simple two-line plot: held-out `bucket_mean` per epoch (0.5649 / 0.6242 / 0.6262 / **0.6744** /
0.6592) against training `token_acc` climbing to 0.987. The crossing point is the story: the
model keeps memorising while held-out performance turns over at epoch 4.

Data: `experiments/42-merged-corpus/RESULTS.csv:2-6`.

**Owner:** Rodrigo or Leo (the data is a five-row CSV; this is 20 minutes of matplotlib).

---

## Fig. 4 — Local versus platform calibration

Two panels:

- **(a)** local `bucket_mean` vs public score for the four submissions — three points on a line
  with the right sign and the wrong slope, showing the instrument is ordinal not cardinal.
- **(b)** the four-cell comparison for the same checkpoint, local vs platform, showing
  `object_recognition_OOD` collapsing 0.8285 → 0.4727 while `aggregation_OOD` rises
  0.4086 → 0.6064.

This is the most scientifically interesting figure we have, and it is the kind of thing the
joint publication will actually want: a concrete demonstration that a locally-constructed OOD
proxy (procedure) does not measure the organizers' OOD axis (centre).

Data: `context/NOW.md:109-128`.

**Owner:** Rodrigo.

---

## Fig. 5 (optional) — Failure decomposition

The 84-question gap to rank 1, split by cell: 36 `object_recognition_OOD`, 31
`object_recognition_ID`, 8 `aggregation_OOD`, 9 `aggregation_ID`. Plus the enumeration curve:
exact-set accuracy 0.801 / 0.616 / 0.175 / 0.000 by number of classes in gold.

Data: `context/NOW.md:130-143`, `experiments/51-clip-attractor/README.md:343-352`.

---

## Supplementary PDF (optional, 1 file, ≤10 MB)

Obvious candidate: **the full ablation ledger** (`_ledger/FULL_LEDGER.md` rendered to PDF). Sixty
single-variable experiments with deltas, CIs and verdicts is exactly the kind of material the
form says belongs in supplementary rather than in a form field, and it is strong evidence of
methodological rigour.

---

## Priority order

1. **Fig. 1** — blocking for awards. Do this first.
2. **Fig. 2** — the form asks for the data pipeline explicitly.
3. **Fig. 4** — highest scientific value; likely to be cited in the joint paper.
4. Fig. 3, Fig. 5, supplementary — if time allows.
