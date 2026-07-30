# 26 — de-shortcut eval: how much of our margin does the question hand over?

## Ladder

| run | what it is | `bucket_mean` |
|---|---|---|
| `../00-baseline/` rung 00 | Qwen3-VL-8B, zero-shot | 0.2557 |
| `../02-lora-sft/` rung 02 | LoRA, LLM only — **the scored run this rung re-reads** | 0.5486 |
| `../06-vit-lora/` rung 06 | LoRA, ViT+LLM | 0.5667 |
| **26 part 1** | shortcut exposure + per-stratum margin, **zero GPU** | n/a — a diagnostic, no new score |
| **26 part 2** | paired de-named eval on `fo_class`, 673 × 3 arms, one checkpoint | ✅ **MEASURED** — n/a, a diagnostic |

*This rung produces **no new `bucket_mean`**. It re-reads an existing scored run and asks what its
margin is made of, so there is no `RESULTS.csv` row — the outputs are the two tables. See
`PLAN.md` for the full pre-registration and the decision rule.*

## The one-line result

**40.6% of the val set carries a linguistic shortcut**, and the margin gap it buys is
**heterogeneous by format, not uniform**: `fo_class` gets **~2× the margin** with a shortcut
(+0.22 in both ID and OOD), `number` gets **nothing** (−0.02), and `binary` OOD is **below its
trivial floor** on the shortcut cell (−0.1576 over 311 questions).

🔴 **Do not quote the pooled gap (+0.0493 ID).** It is a format-mix artefact — within format the
sign is not even constant, and `strata_report` refuses to emit a pooled row for that reason.

## Part 2 — and it answers a different question than the one it settles

Full account in `context/decisions/margin-is-vision-not-phrasing.md`. Numbers in
`tables/part2_read.csv`; accuracy `original` 0.8437 · `premise_dropped` 0.8469 · `set_framed` 0.7790.

🟢 **The margin is vision.** Dropping the cardinality premise is **EQUIVALENT** at the pre-declared
ε = 0.05 in ID (+0.0021), OOD (+0.0064) and pooled — so part 1's 2× gap was **difficulty, not
exploitation**, and the ladder comparisons that lean on `object_recognition` stand. ⚠️ Its
`epsilon_min` on ID is 0.0417: with 28 videos no finer margin was reachable.

🔑 **The secondary arm is the finding.** Re-asked with the corpus's own *"list all"* template — same
frame, same gold, still a single class — the model must decide the **cardinality itself** and drops
**−0.0647** (ID −0.0607, OOD −0.0758, every cell excluding zero; in OOD **50 questions flip
right→wrong against 11**). The individuation deficit is **inside `fo_class`**, not only inside
`number`. ⚠️ That arm moves two variables at once, so it localises the cost without attributing it.

## Reproduce part 2

```
papermill 26b_part2_infer.ipynb out.ipynb -p SMOKE False    # 2,019 questions, ~26 min, no training
```

## Reproduce

```python
from shortcut_taxonomy import exposure, strata_report   # experiments/26-deshortcut-eval/_models/
exposure(gold)                    # gold = the two val parquets, + a `distribution` column
strata_report(scored)             # scored = results.csv merged with gold
```

Both tables in `tables/` were produced by exactly those two calls against rung 02's
`eval_best/results.csv` (n = 6,252, asserted).
