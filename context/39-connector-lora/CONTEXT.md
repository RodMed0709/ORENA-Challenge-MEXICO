# 39 — connector LoRA · CONTEXT

## Objective

Does putting LoRA on the **ViT→LLM connector** improve the model? Across 30+ rungs the merger has
never received a gradient, and the cause is a regex rather than a decision
([[the-merger-is-unreachable-by-default]]) — rung 39 is the rung that acts on it.

## Setup-config

- **Split:** the frozen `frame_ood_v1` eval set, 6252 questions; ID/OOD from the qID prefix.
- **Base model:** `/workspace/models/qwen3-vl-8b` (`model_type qwen3_vl`), ms-swift 4.4.1.
- **Data:** rung 18's `train.jsonl`, 14,415 rows, sha256-pinned by rung 21, **not re-exported**.
  14,415 / 16 = 901 steps/epoch; ~11.56 s/it ⇒ ~2.9 h.
- **Recipe (A2, pinned and unmoved):** `lora_rank 8` · `lora_alpha 32` · `lora_dropout 0.1` ·
  `learning_rate 2e-4` · `max_grad_norm 1.0` · `cosine` · `warmup_ratio 0.03` · `bfloat16` ·
  `attn_impl sdpa` · `freeze_vit false` · `1 × 16` (effective batch 16) ·
  `gradient_checkpointing true` · `seed 42` · `num_train_epochs 3`.
- **THE ONE VARIABLE:** `--target_modules` gains the eight merger `Linear` layers —
  `model.visual.merger.linear_fc{1,2}` and `model.visual.deepstack_merger_list.{0,1,2}.linear_fc{1,2}`
  — passed as **`all-linear` PLUS the eight names**, nine separate argv values. Coverage extended,
  never replaced.
- **Control:** `21_lr_2e4_v1` arm `A2_lr` **epoch 1** `checkpoint-901` — `object_recognition_ID`
  0.6087962962962963, `proxy_leaderboard` 0.4986389858444832, `bucket_mean` 0.5592175321379278,
  `aggregation_ID` 0.3884816753926701, `margin_OOD` 0.16425. Recomputed from its archived answers,
  never transcribed.
- **Declared primary cell (RULES §S3):** `object_recognition_ID` — the bucket most directly
  downstream of a connector intervention. Not local `bucket_mean`.

## Decisions

- **Schedule (decided 2026-08-13, option A1):** `--num_train_epochs 3`, terminate the trainer once
  `checkpoint-901` is complete. A naive `--num_train_epochs 1` arm would carry a second undeclared
  variable — warmup 27 steps vs the control's 81, and LR exactly 0.0 at step 901 vs mid-cosine
  (`recipe_sweep_train.py`, the `ARMS` comment on `C_epochs`). A1 costs the same ~2.9 h and removes
  the confound. The process is killed rather than the schedule shortened, because shortening the
  schedule **is** the second variable.
- **Targets (decided 2026-08-13):** `all-linear` + the 8 names, and the gate carries an explicit
  **coverage** criterion (`n_llm == 504`, `n_vit == 216`, both > 0) alongside `n_aligner > 0` and
  `n_orphans == 0`, so a pass cannot be satisfied by an adapter that reached the merger and lost the
  LLM and ViT legs.
- **`--freeze_aligner` is branch-selected by the gate's control leg, both branches pre-declared**
  (PLAN.md §5): control `n_aligner == 0` ⇒ arm adds `--freeze_aligner false` (rung 32 is the evidence
  that flag is inert on its own); control `n_aligner > 0` ⇒ arm keeps A2's `true` untouched, and the
  artifact differential is supplied by rung 32 instead of by the control leg.
- **The gate does NOT pass `--adapters`** (rung 32's smoke did). A resumed adapter carries its own
  `target_modules` in `adapter_config.json`, so the gate would measure rung 21's LoRA geometry, not
  rung 39's. It constructs the LoRA fresh, exactly as the arm will.
- **The diff that proves the single variable is LOCAL to rung 39.** Rung 21's `_as_map`
  (`recipe_sweep_train.py:251`) keeps only the first value after a flag — measured on a laptop
  2026-08-13, `_as_map(['--target_modules','m1','m2','m3'])` returns `{'--target_modules': 'm1'}` —
  so it would check 1 of 9 targets and report a clean single-variable diff. It is **not edited**: it
  is the diff behind every already-scored rung-21 arm, and rewriting it re-dates all of them.
  `connector_lora_train._as_map_multi` is the corrected local copy, following the precedent
  `swift_args_21` set.
- **Where the chain lives:** rendered by `_tools/chain.py` into `/workspace/tmp/`, never committed. A
  committed `.sh` is forbidden and a `.py` launcher is forbidden; papermill executing the notebook is
  the sanctioned headless path, and its non-zero exit is the mechanism the blocking gate uses.
- **Motivation, not a prior:** the rung-38 eval closed NO-GO on 2026-08-13 (the fine-tuned gen-3.6
  27B loses on all five cells vs A2 ep1, `proxy_leaderboard` −0.0344), which makes the connector the
  **main live lane** rather than a side bet. ⚠️ That is a floor, not a ceiling — the A2 recipe was
  never ported to that backbone — and it must not inflate any claim about what rung 39 will find.

## Results

**pending — pre-registered 2026-08-13.** The gate has not been run and the arm has not been run.

## Next

Independent read-only review of `PLAN.md` + the code (GO / NO-GO with file:line), then render the
chain onto pod `y6h32tbhwhgxxe` (`/workspace/repo_rodri`) and **run the gate**. The gate decides
whether the arm exists in its declared form; if it fails, the finding is published and the lane
closes for ~$0.50.
