# Rung 56 — the enumeration probe: counting AND fo_class, from the same hidden states

> **Status: MEASURED — faithful negative on both tasks. Neither probe beats the model's own
> token output on the true 8-video held-out set (counting 0.4462 vs 0.4613; `fo_class`
> full-set-identity 0.8143 vs 0.8690), despite 0.8109/0.9224 accuracy in-fold on the 122-video
> fit pool. Decision note: [[layer-24-edge-does-not-clear-a-fresh-holdout]].**

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 34-hidden-state-probe | linear probe on hidden states, ID-fit → OOD-read, established the PCA/scaler discipline | rung 33's token argmax | 🟢 THESIS CONFIRMED — probe beats the head at layers 18–24 |
| 42-merged-corpus | promoted 30 of 38 test videos into training | 21 A2 | shipped — the checkpoint every probe since reads |
| 49-flip-equivariance | same checkpoint, horizontal-flip tracking probe | — | closed, decision note filed |
| 55-cardinality-probe | layer-24 probe on `fo_class` cardinality, A2 ep3 adapter, ID(lapchole)→OOD(heico) | model's own emitted set size | 🟢 POSITIVE, CI excludes zero |
| **56 (this)** | rung 34's method, extended to TWO tasks (`number` counting + `fo_class` full set identity) on rung 42 ep4 **merged**, 122-video fit pool → fresh 8-video held-out | rung 34 (counting only, 92-video ID/OOD split) + rung 55 (cardinality only) | ⚠️ MEASURED — both tasks lose to the token head on the true held-out set |

## Why this rung exists

Rung 34 answered "is the count linearly decodable" for `number` alone, on the 92 `train.parquet`
videos, with an ID/OOD split baked into the fit itself. Two things changed since: rung 42 promoted
30 more videos into scenes the model has seen in training-adjacent form, and Rodrigo's own reading
of the campaign (`[[fo-class-and-number-are-one-front]]`) found `fo_class` and `number` failing
*together* often enough that they read as one front, not two. This rung asks the rung-34 question
for **both** fronts at once, on the larger pool, and adds one thing rung 34 never had a second
task to ask: **if the model's hidden state already encodes which classes are present, does
counting "how many different classes" reduce to `len(that set)`?**

## Result

| task | CV (122-video fit pool) | held-out (8 videos) | held-out token-head | probe wins? |
|---|---:|---:|---:|---|
| counting (`number`, exact value) | 0.8109 (layer **34**, ordinal) | 0.4462 [0.318, 0.585] | 0.4613 [0.343, 0.584] | NO |
| `fo_class` (full set identity) | 0.9224 (layer **24**, C=0.01) | 0.8143 [0.759, 0.871] | 0.8690 [0.807, 0.926] | NO |

Both CV numbers were markedly optimistic relative to the held-out read (36-point and 11-point
drops) — video-grouped CV within one fit pool guards against frame-level leakage, not against the
fit pool being an easier population than genuinely fresh videos. The counting probe's own winning
layer moved from 24 (rung 34/55's peak) to 34, on the same architecture. A small aggregation
comparison (20 exact-frame pairs where an `n_classes` question and an `fo_class` question share
the same frame) hints that `len(fo_class-probe's predicted set)` may recover the count better than
a dedicated counting probe (0.55 vs 0.50, 70% agreement) — consistent in direction with rung 55's
cardinality framing, but n=20 is too small to lean on. Full writeup:
[[layer-24-edge-does-not-clear-a-fresh-holdout]].

## Design decisions (already settled, do not re-litigate)

- **Fit pool = 122 videos**, not 92: `train.parquet`'s 92 (trained on verbatim) **+** the 30
  `test.parquet` videos rung 42 promoted into its corpus (scenes seen via generated training
  questions; these specific rows were never trained on). Final-read = rung 42's own declared
  8-video held-out set, the only videos genuinely untouched by training in any form.
- **`fo_class` fit target = ALL 2,675→8,479 rows** (92→122 videos), not filtered to a "true
  multi-label" subset — because the hidden state being probed is already question-conditioned
  (the question itself projects into the latent feature), so there is no such thing as an
  unconditioned fo_class representation to filter down to.
- **`number` is split into three sub-templates** (`n_classes`, `n_instances_total`,
  `n_instances_per_class`), tagged on every row but **fit as one pooled target** — the
  aggregation comparison below is the only place the split matters.
- Two independent tasks, two independent CV sweeps, two independently-selected best layers
  (`03`). Each task uses its OWN full pool — rows are not restricted to frames carrying both
  question types; that intersection is reserved for the aggregation comparison alone.
- **The aggregation comparison is scoped to `n_classes` questions that share an EXACT frame**
  (same video + same `timestamp_start`) with an `fo_class` question in the final-read set — 20
  such pairs exist locally. It is the only sub-template where `len(fo_class gold set)` equals
  the `number` gold exactly; the other two sub-templates are not comparable this way and are
  never folded into it.
- **Effective n for the final read is 8 VIDEOS**, not the ~1,000 combined rows (RULES §13).
  `04` uses `frame.metrics._hier_bootstrap` (imported, not reimplemented) for every headline CI.

## Pipeline

```
01_manifest.ipynb          zero GPU  — fit/final-read pools, sub-templates, 5-fold CV splits
02_dump_hidden_states.ipynb GPU      — last-prompt-token hidden states, both tasks, one pass/row
03_fit_probes.ipynb        zero GPU  — CV sweep (layer × kind/C), picks the winning config/task
04_final_read.ipynb        GPU       — refit on full fit pool, read ONCE, probe vs token-head,
                                        the aggregation comparison
```

`_tools/`: `manifest.py` (pools/folds), `ordinal_probe.py` (counting — multinomial / balanced /
Frank & Hall ordinal decomposition), `multilabel_probe.py` (fo_class — one-vs-rest), `hidden_dump.py`
(GPU dump, generalizes rung 34's), `token_head.py` (GPU token-head baselines, reuses rung 33's
mechanism for `number`; a documented simplification — joint sequence log-prob — for `fo_class`),
`resolve_frames.py` (manifest row → shared `frames_cache` path, no copying).

All four notebooks have been run for real. `01`: `RESULTS_fit_number_v1.csv`,
`RESULTS_fit_foclass_v1.csv`, `RESULTS_final_read_v1.csv` (18,717 fit rows / 122 videos, 5,838
`number` / 8,479 `fo_class`, 1,283 final-read rows / 8 videos, zero fold leakage). `02`: dumped to
`runs/50_hidden_v1/` (gitignored, pod-local). `03`: `RESULTS_probe_number_cv.csv`,
`RESULTS_probe_foclass_cv.csv`, `RESULTS_best_config.json`. `04`: `RESULTS_final_read.json` +
two per-row detail CSVs — the 8-video held-out set, touched exactly once.

## Running on the pod

Actually used (rung 42 ep4's merged checkpoint lives under rung 49's own run directory, not a
flat `/workspace/models/...` path):

```
papermill 02_dump_hidden_states.ipynb 02_out.ipynb \
  -p MODEL_PATH /workspace/repo_yyy/experiments/49-flip-equivariance/runs/49_flip_pair_v1/merged/checkpoint-4848 \
  -p DATA_ROOT /workspace/orena-data -p FRAMES_CACHE /workspace/frames_cache   # SMOKE=True first, then SMOKE=False
papermill 03_fit_probes.ipynb 03_out.ipynb \
  -p HIDDEN_DIR /workspace/repo_yyy/experiments/56-enumeration-probe/runs/50_hidden_v1
papermill 04_final_read.ipynb 04_out.ipynb \
  -p HIDDEN_DIR /workspace/repo_yyy/experiments/56-enumeration-probe/runs/50_hidden_v1 \
  -p MODEL_PATH /workspace/repo_yyy/experiments/49-flip-equivariance/runs/49_flip_pair_v1/merged/checkpoint-4848 \
  -p DATA_ROOT /workspace/orena-data -p FRAMES_CACHE /workspace/frames_cache   # touches the 8 videos ONCE
```
