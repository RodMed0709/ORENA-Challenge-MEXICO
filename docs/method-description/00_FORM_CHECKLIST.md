# 00 — The form, question by question

**Deadline: 2026-09-16.** One form per track; this is the **FRAME** form.
https://docs.google.com/forms/d/e/1FAIpQLSexxI2SqiYUq4NIHUBR5xcFGdYegFUsiFVBgal0MtSRmfGWhQ/viewform

Status column: **ready** (the answer is below, copy and polish) · **to write** (the facts are
here, the prose is not) · **blocked** (the information does not exist in this repo — see
`07_OPEN_ITEMS.md`).

---

## Section 1 — General Information & Track Selection

| # | field | status | answer / where it comes from |
|---|---|---|---|
| 1.1 | Team Name (must match the platform) | ready | **Mexico-Oxford_TEAM** |
| 1.2 | Title of the Proposed Method | to write | proposals below |
| 1.3 | Short Algorithm / Model Name | to write | proposals below |
| 1.4 | Team Logo (image, ≤10 MB, we must hold the rights) | blocked | nobody has made one — `07_OPEN_ITEMS.md` #6 |

**Method-title proposals** — starting points, not a decision:

- *"Connector-Aware LoRA for Surgical Foreign-Object VQA"* — the ViT→LLM connector is the module
  the frameworks do not reach by default, and naming it is our one real architectural
  contribution.
- *"Reaching the Merger: Full-Stack LoRA Adaptation of Qwen3-VL for Foreign-Object VQA"*
- Short names: `FOCUS-CLoRA`, `Qwen3VL-ConnLoRA`.

Whatever is chosen should be reconcilable with the platform's algorithm name,
"Qwen3VL 8B FT ViT LLM".

---

## Section 2 — Team Members & Authorship

| # | field | status | note |
|---|---|---|---|
| 2.1 | Per author: full name, platform username, email, affiliation(s), ORCID, Google Scholar ID, funding, conflicts of interest | blocked | `07_OPEN_ITEMS.md` #5 |
| 2.2 | Corresponding author email | to write | team decision |
| 2.3 | Three co-author nominations for the joint publication | ready | we are exactly three and the cap is three (`context/decisions/external-data-policy.md`, §Publication policy p.8) |

Known: platform usernames include `Legokna` (Leo). Author **order** is a team decision — the form
records it for authorship, the website and the certificates, so settle it before filling the form
rather than by whoever types it in.

---

## Section 3 — Overall assessment

| # | question | status | answer |
|---|---|---|---|
| 3.1 | Members with prior surgical-domain experience (integer) | blocked | team question |
| 3.2 | Total team hours invested (integer) | blocked | `07_OPEN_ITEMS.md` #7 |
| 3.3 | Allocation across 11 categories, 11 decimals summing to 1.0 | to write | proposal below |
| 3.4 | Strategies **tested** (multi-select) | ready | `03_RESULTS.md` §2 |
| 3.5 | Strategies that **contributed significantly** (multi-select) | ready | `03_RESULTS.md` §3 |
| 3.6 | Peak total VRAM during training, GB (integer) | ready | **22** — `04_INFRA.md` §1 |
| 3.7 | Wall-clock training time of the final model, hours (integer) | estimate | **19** (ESTIMATE) — `04_INFRA.md` §2 |
| 3.8 | Total GPU-hours for this track (integer) | estimate | **≈385** (ESTIMATE) — `04_INFRA.md` §3 |

### 3.3 — Time allocation, a starting proposal

Category order is fixed by the form. This split reflects what the repository actually shows: a
large share in failure analysis and measurement, a small share in architecture.

```
[0.05, 0.15, 0.05, 0.05, 0.08, 0.07, 0.10, 0.15, 0.05, 0.20, 0.05]
```

| # | category | share | basis |
|---|---|---:|---|
| 1 | Literature research | 0.05 | 30-paper corpus, `literature/INDEX.md` |
| 2 | Analysis of data & annotations | 0.15 | rung 08 data card, template floors, shortcut audit |
| 3 | Understanding challenge design | 0.05 | `context/RULES.md`, the Copeland reading, the licence question |
| 4 | Baseline implementation | 0.05 | rung 00 + vendor baseline |
| 5 | Data allocation (new annotations) | 0.08 | rung 18 minted zeros, rung 19b external, rung 42 merge |
| 6 | Data preprocessing / augmentation | 0.07 | rungs 11, 12, 14, 24 — all negative |
| 7 | Model design & architecture | 0.10 | connector reachability, rung 39, the 27B branch |
| 8 | Hyperparameter tuning | 0.15 | rung 21 sweep + the epoch sweeps |
| 9 | Post-processing | 0.05 | the two answer-boundary guards |
| 10 | Failure case analysis | 0.20 | 96 decision notes, most of them post-mortems |
| 11 | Other | 0.05 | packaging, Docker, submissions |

Sums to 1.00. Adjust freely — the one worth keeping large is #10, because it is what we genuinely
did and it is the part a reviewer will find credible.

---

## Section 4 — Method Architecture & VLM Design

| # | question | status | source |
|---|---|---|---|
| 4.1 | Method Abstract & Novelty (max 200 words) | to write | raw material in `01_METHOD.md` §8 |
| 4.2 | Pre-trained VLM / vision backbone | ready | **Qwen series** — Qwen3-VL-8B-Instruct |
| 4.3 | Total parameters, billions, rounded integer | ready | **8** (revisit if the final model is an inference-time ensemble — `07_OPEN_ITEMS.md` #1) |
| 4.4 | Long-context / temporal strategy | n/a | SEGMENT & PROCEDURE only. If a field must be filled: FRAME supplies one pre-extracted frame per question and all FRAME clips have duration zero (`experiments_segment/README.md:42-52`) |
| 4.5 | Prompting / input strategy, **quote text modules verbatim** | ready | `01_METHOD.md` §4 — the system prompt is there character for character |

---

## Section 5 — Training & Data Processing

| # | question | status | source |
|---|---|---|---|
| 5.1 | External datasets used? (or "no") | ready | effectively **no** for the shipped model; CholecT50 disclosed as tested-and-rejected — `02_TRAINING.md` §7 |
| 5.2 | Links to public datasets, plus the **generation approach AND exact structural format** of any annotations we created | to write | the JSON schema is in `02_TRAINING.md` §6.4; the publish-vs-private split by data source is §6.5. This field is one of the form's stated mandatory requirements for FRAME |
| 5.3 | Training strategy in detail | ready | `02_TRAINING.md` §§2–5 |
| 5.4 | Max frames at inference | ready | **1** |
| 5.5 | How the frame count is chosen (multi-select) | ready | **Standardized frame rate (fixed number of frames)** — the platform supplies exactly one frame per question |
| 5.6 | Frame resolution, single integer `(w+h)/2` | ready | **1000** (cap is `max_pixels` 1280×720; no corpus frame exceeds it) |
| 5.7 | Time overlay used? | ready | **No** |
| 5.8 | Core training parameters + how they were optimized | ready | `02_TRAINING.md` §§2–3 |

---

## Section 6 — References and Supplementary

| # | item | status | source |
|---|---|---|---|
| 6.1 | Figure uploads (≤10 PDFs). `Mexico-Oxford_TEAM_fig_1.pdf` is **mandatory** — missing it disqualifies us for awards | blocked | `05_FIGURES.md` |
| 6.2 | Figure captions & descriptions | to write | `05_FIGURES.md` |
| 6.3 | References | to write | `06_REFERENCES.md` |
| 6.4 | Supplementary (1 PDF, ≤10 MB), optional | to write | the ablation ledger is an obvious candidate |
| 6.5 | Challenge experience rating, 1–10 | to write | team call |
| 6.6 | Free-text feedback | to write | team call |

File naming is enforced: `<teamname>_fig_<X>.pdf`, X starting at 1 — e.g.
`Mexico-Oxford_TEAM_fig_1.pdf`. Reference figures in text as `[Fig. X]` and literature as `[1]`.

---

## Section 7 — Declaration

| # | item | status | note |
|---|---|---|---|
| 7.1 | Confirm compliance with challenge rules | ready | one checkbox |

---

## Three things that can still cost us the award

1. **No Fig. 1** — automatic disqualification for awards. Highest-priority deliverable.
2. **Publishing challenge-derived annotations publicly** — a DUA violation. They go privately.
3. **The description not matching the model actually submitted** — the 2026-09-02 ensemble's
   details are not in this repo. `07_OPEN_ITEMS.md` #1.
