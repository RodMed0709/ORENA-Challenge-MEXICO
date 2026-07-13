# splits/ — frozen FRAME partitions (shared source of truth)

This folder holds the **committed** train / `val_id` / `ood_test` manifests. It sits
next to the experiments so every notebook can reload the SAME partition instead of
re-deriving it (which would silently drift between runs / machines / teammates).

- **Written by:** `experiments/01-ood-split/01_build_ood_split.ipynb` (calls
  `src/frame/split.py`). Run it on the pod where the data lives, then **commit the CSV**.
- **Reloaded by:** any experiment notebook —
  ```python
  from frame import split as sp
  video_split = sp.load_manifest(REPO / "experiments" / "splits" / "frame_ood_v1.csv")
  train_items = sp.apply_split(items, video_split, "train")
  ood_items   = sp.apply_split(items, video_split, "ood_test")
  ```
- **Committed, NOT gitignored** (unlike `runs/`): the manifest is small and must be
  identical for both teammates and every pod — that is the whole point.
- **Never rename or edit a manifest referenced by another experiment.** Bump to
  `frame_ood_v<N+1>.csv` and record the change in the experiment's RESULTS.csv.

Manifest columns: `dataset, video_id, procedure_type, split, n_questions, seed, ood_procedure`.
