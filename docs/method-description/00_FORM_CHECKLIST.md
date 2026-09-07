# 00 — The form, question by question

**Deadline: 2026-09-16.** One form per track; this is the **FRAME** form.
https://docs.google.com/forms/d/e/1FAIpQLSexxI2SqiYUq4NIHUBR5xcFGdYegFUsiFVBgal0MtSRmfGWhQ/viewform

Status key: **✅ ANSWERED** (copy and polish) · **✍️ DRAFT NEEDED** (facts are here, prose is not)
· **🔴 BLOCKED** (information does not exist in this repo — see `07_OPEN_ITEMS.md`)

---

## Section 1 — General Information & Track Selection

| # | field | status | answer / where it comes from |
|---|---|---|---|
| 1.1 | Team Name (must match the platform) | ✅ | **Mexico-Oxford_TEAM** |
| 1.2 | Title of the Proposed Method | ✍️ | Proposal below |
| 1.3 | Short Algorithm / Model Name | ✍️ | Proposal below |
| 1.4 | Team Logo (image, ≤10 MB, we must have rights to it) | 🔴 | Nobody has made one. `07_OPEN_ITEMS.md` #6 |

**Method-title proposals** (pick one; they encode our actual novelty):

- *"Connector-Aware LoRA for Surgical Foreign-Object VQA"* — accurate and specific: the ViT→LLM
  connector is the module the frameworks do not reach by default, and naming it is our one real
  architectural contribution.
- *"Reaching the Merger: Full-Stack LoRA Adaptation of Qwen3-VL for Foreign-Object VQA"*
- Short name: **`FOCUS-CLoRA`** or **`Qwen3VL-ConnLoRA`**.

⚠️ Whatever is chosen must be consistent with the platform's algorithm name
("Qwen3VL 8B FT ViT LLM") or explained.

---

## Section 2 — Team Members & Authorship

| # | field | status |
|---|---|---|
| 2.1 | Per author: full name · platform username · email · affiliation(s) · ORCID · Google Scholar ID · funding · conflicts of interest | 🔴 `07_OPEN_ITEMS.md` #5 |
| 2.2 | Corresponding author email | ✍️ Rodrigo, presumably |
| 2.3 | Three names nominated as co-authors on the joint publication | ✅ We are exactly three — Rodrigo, Leo, Yingyu — and the cap is three (`context/decisions/external-data-policy.md`, §Publication policy p.8) |

Known: platform usernames include **`Legokna`** (Leo) and Rodrigo's own. Author **order** is a
team decision, not a documentation one — settle it before filling the form, because the form
records the official order for authorship, website and certificates.

---

## Section 3 — Overall assessment

| # | question | status | answer |
|---|---|---|---|
| 3.1 | Members with prior surgical-domain experience (integer) | 🔴 team question |
| 3.2 | Total team hours invested (integer) | 🔴 `07_OPEN_ITEMS.md` #7 |
| 3.3 | Allocation across 11 categories, 11 decimals summing to 1.0 | ✍️ **Proposal below** |
| 3.4 | Strategies **tested** (multi-select) | ✅ `03_RESULTS.md` §2 |
| 3.5 | Strategies that **contributed significantly** (multi-select) | ✅ `03_RESULTS.md` §3 |
| 3.6 | Peak total VRAM during training, GB (integer) | ✅ **22** — `04_INFRA.md` §1 |
| 3.7 | Total wall-clock training time of the final model, hours (integer) | 🟡 ESTIMATE **19** — `04_INFRA.md` §2 |
| 3.8 | Total GPU-hours for this track (integer) | 🟡 ESTIMATE **≈385** — `04_INFRA.md` §3 |

### 3.3 — Time allocation proposal

Order is fixed by the form. This allocation reflects what the repository actually shows: an
unusually large share in failure analysis and measurement discipline, and a small share in
architecture.

```
[0.05, 0.15, 0.05, 0.05, 0.08, 0.07, 0.10, 0.15, 0.05, 0.20, 0.05]
```

| # | category | share | why |
|---|---|---:|---|
| 1 | Literature research | 0.05 | 30-paper corpus, `literature/INDEX.md` |
| 2 | Analysis of data & annotations | 0.15 | rung 08 data card, template floors, the shortcut audit |
| 3 | Understanding challenge design | 0.05 | `context/RULES.md`, the metric/Copeland reading, the licence question |
| 4 | Baseline implementation | 0.05 | rung 00 + the vendor baseline |
| 5 | Data allocation (new annotations) | 0.08 | rung 18 minted zeros, rung 19b external, rung 42 merge |
| 6 | Data preprocessing / augmentation | 0.07 | rungs 11, 12, 14, 24 — all negative |
| 7 | Model design & architecture | 0.10 | connector reachability, rung 39, the 27B branch |
| 8 | Hyperparameter tuning | 0.15 | rung 21 sweep + the epoch sweeps |
| 9 | Post-processing | 0.05 | the two answer-boundary guards |
| 10 | **Failure case analysis** | **0.20** | 96 decision notes, most of them post-mortems |
| 11 | Other | 0.05 | packaging, Docker, submissions |

Sanity: sums to 1.00. Adjust as the team sees fit, but keep #10 large — it is what we genuinely
did, and it is the part a reviewer will find credible.

---

## Section 4 — Method Architecture & VLM Design

| # | question | status | source |
|---|---|---|---|
| 4.1 | Method Abstract & Novelty (max 200 words) | ✍️ raw material in `01_METHOD.md` §8 |
| 4.2 | Which pre-trained VLM/vision backbone | ✅ **Qwen series** — Qwen3-VL-8B-Instruct |
| 4.3 | Total parameters, billions, rounded integer | ✅ **8** |
| 4.4 | Long-context / temporal strategy | ➖ **SEGMENT & PROCEDURE only.** If a field must be filled: "not applicable — FRAME supplies one pre-extracted frame per question, and all FRAME clips have duration zero" (`experiments_segment/README.md:42-52`) |
| 4.5 | Prompting / input strategy, **quote text modules verbatim** | ✅ `01_METHOD.md` §4 — the system prompt is there character for character |

---

## Section 5 — Training & Data Processing

| # | question | status | source |
|---|---|---|---|
| 5.1 | External datasets used? (or "no") | ✅ effectively **no** for the shipped model; disclose CholecT50 as tested-and-rejected — `02_TRAINING.md` §7 |
| 5.2 | Links to all public datasets; links to any supplementary annotations | ⚠️ **Read `02_TRAINING.md` §6.4 first.** Our self-made annotations derive from challenge data, so the DUA **forbids** publishing them now — they go to the organizers **privately** and become public when LapChole-FOCUS is released |
| 5.3 | Training strategy in detail: pipeline steps, dataset usage, splits, hyperparameter sequence | ✅ `02_TRAINING.md` §§2–5 |
| 5.4 | Max frames given to the model at inference | ✅ **1** |
| 5.5 | How the frame count is chosen (multi-select) | ✅ **Standardized frame rate (fixed number of frames)** — the platform supplies exactly one frame per question |
| 5.6 | Frame resolution, single integer `(w+h)/2` | ✅ **1000** (cap is `max_pixels` 1280×720; no corpus frame exceeds it) |
| 5.7 | Time overlay used? | ✅ **No** |
| 5.8 | Core training parameters + the strategy used to optimize them | ✅ `02_TRAINING.md` §§2–3 |

---

## Section 6 — References and Supplementary

| # | item | status |
|---|---|---|
| 6.1 | **Figure uploads (≤10 PDFs).** `Mexico-Oxford_TEAM_fig_1.pdf` is **MANDATORY** — missing it **disqualifies us for awards** | 🔴 `05_FIGURES.md` |
| 6.2 | Figure captions & descriptions | ✍️ `05_FIGURES.md` |
| 6.3 | References | ✍️ `06_REFERENCES.md` |
| 6.4 | Supplementary (1 PDF, ≤10 MB) — optional | ✍️ the ablation ledger is an obvious candidate |
| 6.5 | Challenge experience rating, 1–10 | ✍️ team call |
| 6.6 | Free-text feedback | ✍️ team call |

**File naming is enforced:** `<teamname>_fig_<X>.pdf`, X starting at 1 — e.g.
`Mexico-Oxford_TEAM_fig_1.pdf`. Reference figures in text as `[Fig. X]` and literature as `[1]`.

---

## Section 7 — Declaration

| # | item | status |
|---|---|---|
| 7.1 | Confirm compliance with challenge rules | ✅ one checkbox |

---

## The three things that can still cost us the award

1. **No Fig. 1** → automatic disqualification for awards. It is the single highest-priority
   deliverable in this dossier.
2. **Publishing challenge-derived annotations publicly** → DUA violation. They go privately.
3. **The method description not matching the model actually submitted** → the 2026-09-02
   ensemble's details are not in this repo. `07_OPEN_ITEMS.md` #1.
