#!/usr/bin/env bash
# Rung 54 — the stacked leaderboard arm, made restartable.
#
# WHY RESTARTABLE: the first attempt ran on a B200 at $6.79/h and RunPod terminated it
# mid-run when the account balance hit zero. Checkpoints are per-epoch, so the work up to
# the last completed epoch survives on the network volume — but only if the launcher looks
# for it instead of starting from scratch. It resumes from the newest checkpoint it finds
# under the SAME output_dir, and says so in the log rather than resuming silently.
#
# It also refuses to start if the corpus is not the gated one: a restart that quietly
# trains on rung 42's original file would be a different rung wearing this rung's name.
set -u
PY=/workspace/envs/infer/bin/python
OUT=/workspace/rung54/runs/full
CORPUS=/workspace/rung54/corpus/train_stack.jsonl
CORPUS_SHA=2b2fc29e9006f8b31a71f80f433e31b70a799b90b03fc35b816e418c2a984564
CONN="model.visual.merger.linear_fc1 model.visual.merger.linear_fc2 \
model.visual.deepstack_merger_list.0.linear_fc1 model.visual.deepstack_merger_list.0.linear_fc2 \
model.visual.deepstack_merger_list.1.linear_fc1 model.visual.deepstack_merger_list.1.linear_fc2 \
model.visual.deepstack_merger_list.2.linear_fc1 model.visual.deepstack_merger_list.2.linear_fc2"

got=$(sha256sum "$CORPUS" | cut -d' ' -f1)
if [ "$got" != "$CORPUS_SHA" ]; then
  echo "REFUSING TO START: corpus sha256 is $got, the gated build is $CORPUS_SHA"; exit 2
fi
mkdir -p "$OUT"

# newest checkpoint by step number, across any v*/ run directory swift may have made
RESUME=$(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
RESUME_ARG=""
if [ -n "$RESUME" ]; then
  CKPT=$(ls -d "$OUT"/v*/checkpoint-"$RESUME" | tail -1)
  RESUME_ARG="--resume_from_checkpoint $CKPT"
  echo "[rung54] RESUMING from $CKPT (step $RESUME of 6060)"
else
  echo "[rung54] no checkpoint found — starting from step 0"
fi

$PY /workspace/rung54/_tools/train_stack.py --ratio 4.0 \
  --evidence "$OUT/RESULTS_loraplus_gate.json" -- sft \
  --model /workspace/models/qwen3-vl-8b --model_type qwen3_vl --tuner_type lora \
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.1 \
  --target_modules all-linear $CONN \
  --freeze_vit false --freeze_aligner false --freeze_llm false \
  --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --torch_dtype bfloat16 --attn_impl sdpa --seed 42 \
  --dataset "$CORPUS" --split_dataset_ratio 0 \
  --optim adamw_torch_fused --weight_decay 0.1 --gradient_checkpointing true \
  --num_train_epochs 5 --save_strategy epoch --save_total_limit 6 --logging_steps 5 \
  $RESUME_ARG --output_dir "$OUT" 2>&1 | tee -a "$OUT/train.log"
RC=${PIPESTATUS[0]}
echo "TRAIN_RC=$RC" | tee "$OUT/TRAIN_RC"

# Merge every saved epoch. rung 42 selected ep4 and ep5 was worse, so nothing is chosen
# last-by-default and all of them have to exist before the card is released.
if [ "$RC" -eq 0 ]; then
  for c in $(ls -d "$OUT"/v*/checkpoint-* 2>/dev/null); do
    n=$(basename "$c")
    [ -d "$OUT/merged/$n" ] && { echo "skip $n (merged)"; continue; }
    echo "===== merging $n"
    $PY -m swift.cli.main export --adapters "$c" --merge_lora true \
        --output_dir "$OUT/merged/$n" >> "$OUT/merge.log" 2>&1 \
        || echo "MERGE_FAILED $n" >> "$OUT/merge.log"
  done
fi
echo DONE_ALL
