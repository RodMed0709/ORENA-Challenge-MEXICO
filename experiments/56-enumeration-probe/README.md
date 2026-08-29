# Rung 56 — the enumeration probe: counting AND fo_class, from the same hidden states

> **Status: BUILT and locally verified (manifests, folds, classifiers, CV/read plumbing all
> exercised against real data or label-correlated synthetic dumps). GPU-dependent steps
> (`02`'s hidden-state dump, `03`'s CV sweep, `04`'s final read and token-head baselines) have
> NOT been run yet — that happens on the pod. Nothing in this README is a result.**

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 34-hidden-state-probe | linear probe on hidden states, ID-fit → OOD-read, established the PCA/scaler discipline | rung 33's token argmax | 🟢 THESIS CONFIRMED — probe beats the head at layers 18–24 |
| 42-merged-corpus | promoted 30 of 38 test videos into training | 21 A2 | shipped — the checkpoint every probe since reads |
| 49-flip-equivariance | same checkpoint, horizontal-flip tracking probe | — | closed, decision note filed |
| **50 (this)** | rung 34's method, extended to TWO tasks (`number` counting + `fo_class` multi-label) on a bigger fit pool | rung 34 (counting only, 92-video ID/OOD split) | built, GPU steps pending |

## Why this rung exists

Rung 34 answered "is the count linearly decodable" for `number` alone, on the 92 `train.parquet`
videos, with an ID/OOD split baked into the fit itself. Two things changed since: rung 42 promoted
30 more videos into scenes the model has seen in training-adjacent form, and Rodrigo's own reading
of the campaign (`[[fo-class-and-number-are-one-front]]`) found `fo_class` and `number` failing
*together* often enough that they read as one front, not two. This rung asks the rung-34 question
for **both** fronts at once, on the larger pool, and adds one thing rung 34 never had a second
task to ask: **if the model's hidden state already encodes which classes are present, does
counting "how many different classes" reduce to `len(that set)`?**

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

`01` has been run for real (`RESULTS_fit_number_v1.csv`, `RESULTS_fit_foclass_v1.csv`,
`RESULTS_final_read_v1.csv` are live, sha256-sidecarred outputs: 18,717 fit rows / 122 videos,
5,838 `number` / 8,479 `fo_class`, 1,283 final-read rows / 8 videos, zero fold leakage). `02`–`04`
are built and verified as far as possible without a GPU (real-data dry runs up to the
model-loading boundary, plus full end-to-end runs against fabricated label-correlated hidden
states to exercise every join, CV loop, and score computation) — run them on the pod next, in
order, each with `SMOKE`/a small row count first.

## Running on the pod

```
papermill 02_dump_hidden_states.ipynb 02_out.ipynb \
  -p MODEL_PATH /workspace/models/rung42_ep4_merged \
  -p DATA_ROOT /workspace/orena-data -p FRAMES_CACHE /workspace/frames_cache   # SMOKE=True first
papermill 03_fit_probes.ipynb 03_out.ipynb \
  -p HIDDEN_DIR .../runs/50_hidden_v1                                          # after the full dump
papermill 04_final_read.ipynb 04_out.ipynb \
  -p HIDDEN_DIR .../runs/50_hidden_v1 -p MODEL_PATH /workspace/models/rung42_ep4_merged \
  -p DATA_ROOT /workspace/orena-data -p FRAMES_CACHE /workspace/frames_cache   # touches the 8 videos ONCE
```
