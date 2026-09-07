# Questions for Leo — the final model

The 2026-09-02 leaderboard entry (**0.5813**, listed as "Qwen3VL 8B FT ViT LLM") is the two-epoch
ensemble. It is not in the repository: the last content commit is 2026-08-27, and
`submissions/04-rung40-conn4e5-ep23/` is the Qwen3.6-27B container whose own README says it was
*"BUILT, never asked a question"* and *"not a bid for the leaderboard"*.

The method description has to describe what was actually submitted. These are the answers it
needs. Most are one line each.

---

## 1. The two checkpoints

- Which two runs and which epochs? (run id + epoch, e.g. `42_merged_v1` ep4 and ep5)
- Same training run, different epochs — or two separate runs?

## 2. The ensemble rule — this one changes a mandatory numeric field

How are the two combined?

- **Weight averaging** (the two checkpoints merged into one set of weights, one forward pass at
  inference) → the form's parameter count stays **8**.
- **Answer-level voting or logit averaging** (both models run at inference, each producing a
  candidate) → the form says *"if you used an ensembling or multiple components in the model
  architecture please sum up all together"*, so the answer is **16**.

It also changes the mandatory Fig. 1, which has to show the pipeline end to end.

## 3. The corpus

- Rung 42's merged 19,384-row corpus, or the full ORENA dataset you mentioned training on with
  no internal validation split?
- If it was the full dataset: how many rows, and was anything held out at all?

## 4. Training cost

- Wall-clock hours, summed over both arms — the form asks for the sum.
- Which GPU(s), and how many at once.
- Peak VRAM, if it differs from the 22 GB we measured on the single-GPU recipe.

## 5. Inference

- Did `inference.py` change from `submissions/03-*`? Frames, resolution, prompt,
  `max_new_tokens`, post-processing?
- If it was reused: `inference.py:553` logs the **wrong rung name** at startup. It touches no
  answer and no score, but submission 02's README calls that log line *"the ONLY artifact you get
  back from a run that dies early"*. Worth fixing before the test-phase submission.

## 6. Push it

Please commit it as `submissions/05-<slug>/`, matching the structure of
`submissions/03-rung42-connector-ood/` — Dockerfile, `inference.py`, requirements, README.
That README is the template; it is genuinely good, and the form is asking for exactly the things
it already records.

---

## Also, when you have a moment

- **UNAM job accounting** for the total GPU-hours figure. Our current number (≈385 h) is an
  estimate built from ~155 recorded hours plus ~230 inferred, and the form wants one integer.
- **The UNAM GPU rule.** The project notes record "one GPU, never both" as a standing team rule;
  the 6 September conversation says both can be used. Whichever is right, the method description
  will state a hardware configuration, so it should match reality.
