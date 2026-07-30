# 26 — de-shortcut eval: how much of our margin does the question hand over?

## Ladder

| run | what it is | `bucket_mean` |
|---|---|---|
| `../00-baseline/` rung 00 | Qwen3-VL-8B, zero-shot | 0.2557 |
| `../02-lora-sft/` rung 02 | LoRA, LLM only — **the scored run this rung re-reads** | 0.5486 |
| `../06-vit-lora/` rung 06 | LoRA, ViT+LLM | 0.5667 |
| **26 part 1** | shortcut exposure + per-stratum margin, **zero GPU** | n/a — a diagnostic, no new score |
| 26 part 2 | paired de-named eval on `fo_class` | 🔒 pre-registered, needs inference |

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

## Reproduce

```python
from shortcut_taxonomy import exposure, strata_report   # experiments/26-deshortcut-eval/_models/
exposure(gold)                    # gold = the two val parquets, + a `distribution` column
strata_report(scored)             # scored = results.csv merged with gold
```

Both tables in `tables/` were produced by exactly those two calls against rung 02's
`eval_best/results.csv` (n = 6,252, asserted).
