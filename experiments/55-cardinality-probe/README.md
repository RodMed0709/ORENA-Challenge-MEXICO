| Notebook | Rung | Metric (primary) | Verdict |
|---|---|---|---|
| `00_cardinality_probe.ipynb` | 55 | probe OOD 0.8849 vs model 0.8504 (+0.0345) | POSITIVE ⭐ |

# Rung 55 — `fo_class` cardinality is more readable than the head emits

| | probe (fit ID → read OOD) | model's own emitted size | n rows | n videos |
|---|---:|---:|---:|---:|
| cardinality accuracy | **0.8849** | 0.8504 | 782 | 10 |

The layer-24 probe was fit on 531 ID rows (28 videos) and scored on 782 untouched OOD
rows (10 videos). Its in-sample ID accuracy is **0.7420**. Rung 34's `number` probe fit
768 ID rows, so this probe has 237 fewer fit examples. The full enumeration-only slice is
1,313 rows over 38 videos.

## Learning curve

| ID fraction | n fit | train accuracy | OOD accuracy |
|---:|---:|---:|---:|
| 25% | 133 | 0.7594 | 0.8747 |
| 50% | 266 | 0.7444 | 0.8862 |
| 75% | 398 | 0.7412 | **0.8887** |
| 100% | 531 | 0.7420 | 0.8849 |

The curve is plateaued by 50–100%; it is not still climbing at 100%. The result is therefore
not data-limited under the approved power-control rule.

## OOD confusion (rows = gold; columns = predicted)

Probe:

| gold | 0 | 1 | 2 | 3+ |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 473 | 32 | 0 |
| 2 | 0 | 58 | 219 | 0 |
| 3+ | 0 | 0 | 0 | 0 |

Model emitted size:

| gold | 0 | 1 | 2 | 3+ |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 463 | 37 | 5 |
| 2 | 0 | 55 | 202 | 20 |
| 3+ | 0 | 0 | 0 | 0 |

OOD contains only gold sizes 1 and 2, so this run does not measure OOD transfer for sizes 0
or 3+. All 782 regenerated model answers were SDK-legal.

## Reuse and choices

- Reused rung 34's last-prompt-token pooling unchanged at layer 24 and persisted `float16`
  features. Fit pipeline stayed `StandardScaler → PCA-128 → LogisticRegression`.
- Exact checkpoint: `/mnt/storage/uaq_user/rung48/adapters/a2_ep3` (`checkpoint-2703`);
  base snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Enumeration split reuses `experiments/51-clip-attractor/_tools/cardinality_by_template.py`;
  gold and emitted sets are parsed only by `frame.metrics.read_fo_class`.
- Choices not fixed by rung 34: `C=1e-4` was frozen from rung 34's layer-24 winner instead of
  re-tuning on OOD; learning subsets are nested, class-stratified, seed 42; `max_new_tokens=64`
  replaces rung 34's number-only value 4 so variable-length sets can finish.
- Runtime: isolated `/mnt/storage/uaq_user/envs/rod-probe` (`transformers 5.12.1`,
  `torch 2.11.0+cu128`, `peft 0.19.1`, `scikit-learn 1.9.0`).

Artifacts: `RESULTS.csv`, `RESULTS_headline.csv`, `RESULTS_learning_curve.csv`,
`RESULTS_confusion_probe.csv`, and `RESULTS_confusion_model.csv`.
