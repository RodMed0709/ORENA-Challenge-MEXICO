# Experiment 15 — the structured counting target (Qwen3-VL-8B)

> Rungs 02 and 06 train every `number` question on a bare integer (`"3"`). All the gradient
> for a counting question lands on ONE token, so the model can satisfy the objective by
> learning a prior over that token instead of a count — and that is exactly the failure we
> measure. **Is the counting collapse a property of the task, or of the target format?**

## Ladder

| Notebook | Rung | Metric (`bucket_mean`, canonical) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.2557 | baseline |
| `../02-lora-sft/02_lora_sft.ipynb` | 02 | 0.5486 | PASS — LoRA on the LLM |
| `../06-vit-lora/06_vit_lora.ipynb` | 06 | **0.5667** | 🟡 PARTIAL — **the control for this rung** |
| `15_count_target.ipynb` | 15 | *(not yet run)* | — |

*Headline = `bucket_mean` from `frame.metrics.stratified_report`. But the headline is not the
target here: the pre-registered target is the `number` **margin over the template-aware floor**,
per distribution, at every epoch — see `context/15-count-target/CONTEXT.md`.*

## The single variable

**The assistant `content` of `answer_format == "number"` rows, and nothing else.**

`system`, `user`, `images`, `max_pixels`, the LoRA recipe (r=8, α=32, `all-linear`,
`freeze_vit=False`, `freeze_aligner=True`), the LR, the schedule, the seed (42), the epoch count
and the frame cache are all rung 06's, unchanged.

| what | rung 06 (control) | rung 15 |
|---|---|---|
| `number` assistant target | `2` | `{"label": "Clips", "counts": 2}` |
| every other row | — | **byte-identical** (gated) |
| swift argv | — | **identical, 0 flags differ** (`diff_vs_rung06`, gate G-ARGV) |
| inference | bare answer → judge | `parse_count` → bare integer → judge |

Three gates make "single variable" a property of the code rather than a claim:

- **G-OFF** — with `count_target=False` the whole `train.jsonl` must be **byte-identical** to
  rung 06's, proven by sha256 against rung 06's own committed export.
- **G-ON** — with the flag on, every non-`number` line is copied *verbatim* (never
  re-serialised) and both files are re-read to prove it; every `number` line must have changed.
- **G-ARGV** — rung 06's own `_swift_args` is called on both configs and the argv diffed. Empty.

## The target format, and why it is not invented

`literature/vlm-techniques/FICHAS.md` (non-negotiable #9 — every choice cites a paper):

- **v05 Gautam 2025**, *Point, Detect, Count* (arXiv:2505.16647). Qwen2.5-VL-7B + LoRA, ViT
  frozen, 5 epochs: **Count MAE 9.86 → 0.26**. Its count annotations are stored as
  `{"counts": 3, "label": "polyp"}`, and its key trick is *"force a structured, parseable output
  (JSON with explicit `counts` field)"*. **The two field names are taken verbatim.**
- **v14 Qwen3-VL Technical Report** (arXiv:2511.21631): pretraining includes *"normalized
  grounding with box-based, point-based and counting supervision in a [0,1000] coordinate
  system"* — counting is a *pretrained grounding* capability with its own output convention, so
  a bare integer may be fighting the circuit rather than using it.
- **v28 Guo 2025**, *Can't Even Count to 20* (arXiv:2510.04401): counting fails under
  **compositional** scenes; its stated consequence is that *"a per-class count target ('clips: 2,
  sponges: 0') could be more learnable than a single scalar"* — the count must be **bound to a
  named class**. That is why the label is emitted before the count (the one declared deviation
  from v05's key order; the field names are unchanged).

**What we deliberately do NOT copy.** v05's pointing arm and v14's `[0,1000]` convention both
need localization labels. *"The parquets carry `question` / `answer` and metadata only: no
bounding boxes, no masks, no coordinates"* (`context/ERROR_ANATOMY.md`). Point-then-count is not
reproducible on this dataset and is not attempted; this rung tests only the part of v05 our
labels support.

`label` is the question's own noun phrase, **verbatim** — the capture group of
`COUNT_TEMPLATE_RE`. No singularisation, no class vocabulary, no mapping table: the label is a
substring of the question, so the rewrite is deterministic and adds zero invented content.

## 🔴 The parser is a gate, not a detail

`focus.data.formats.Number.verify` requires `text.strip().isdigit()`, and
`Evaluator._evaluate_single` marks anything `fmt.read` rejects as **incorrect**
(`vendor/orena-focus/src/focus/evaluation/evaluator.py:337-343`). A structured answer reaching
the judge is a guaranteed zero. `parse_count` therefore maps the generation back to a bare
integer *inside* `predict`, before `Response.content` exists, and:

1. **round-trips 100 % of the training targets** — gated over the whole exported set
   (`assert_targets_round_trip`), and it RAISES;
2. **never raises** on malformed generation; it degrades to `SAFE_ANSWER = "0"`, which is
   format-valid *and* is not the modal answer of any `number` template — so a malformed
   generation **costs** us instead of quietly inheriting the template floor (`assert_safe_answer_not_modal`);
3. **reports its malformed rate**, next to the published reference: Perek 2026 (v02) measures
   **4.76 %** malformation for vanilla CoT-SFT.

## 🔴 How to read the result

**Never quote `acc_number`.** It averages 8 val templates whose trivial floors run 0.24–1.00,
**four of them degenerate** (`context/RULES.md` §12; data card rung 08 §3). Read:

1. the canonical `number` **margin** from `by_format`, per distribution, **at every epoch**;
2. `RESULTS_number_by_template.csv` — the per-template margin table;
3. the paired, **video-clustered** CI on the delta vs rung 06 (effective n ≈ 38 videos).

Training erases counting (rung 06 `number` margin OOD: +0.015 → +0.013 → **+0.000**) and
`acc_OOD` selection picks the checkpoint that erased more, so **every epoch is saved, merged,
evaluated and reported; nothing is selected last-by-default.**

## Layout

```
15_count_target.ipynb              the ONLY launcher: inline cfg + SMOKE toggle -> main(cfg, stage=…)
_models/count_target.py            the engine: target rewrite, parser, gates, eval wiring, reporting
_tools/test_count_target.py        offline proof of the parser + rewrite gates (no GPU, no swift)
RESULTS.csv                        ledger-shaped, ONE row per run (frame.ledger reads it)
RESULTS_epochs.csv                 per-epoch margins (kept OUT of RESULTS.csv — see below)
RESULTS_number_by_template.csv     per-template `number` margin
runs/<run>/                        gitignored: ckpt/, merged/, train.jsonl, parse_log.jsonl, CSVs
```

⚠️ `frame.ledger` treats an `arm` column as a run name (`src/frame/ledger.py:67`). `RESULTS.csv`
is therefore strictly one row per run; per-epoch and per-template detail live in the sibling CSVs,
exactly as rung 06 split `RESULTS.csv` / `RESULTS_arms.csv`.

## Pre-registration

`context/15-count-target/CONTEXT.md` — arms, decision rule, stopping tiers and known limits,
written **before** any number. It also commits in advance to which of two ceilings — annotation
or model — the eventual result is consistent with, because `number` may be an **annotation**
ceiling: the label moves ±0.86 between frames ≤1 s apart (changing in 56.6 % of pairs) while the
model's MAE is 1.01, and a blind non-clinical human scores r = −0.17 against the gold where the
model scores +0.43 (`context/ERROR_ANATOMY.md`). **A faithful negative here is a valid and
valuable result.**
