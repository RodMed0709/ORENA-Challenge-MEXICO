#!/usr/bin/env bash
# Rung 54 pre-flight. Three short runs on rung 42's UNMODIFIED corpus, to answer three
# questions before a 20-hour run is committed:
#   1. does the LoRA+ registration actually build a two-rate optimizer (the gate), or is it
#      the silent identity that --optimizer lorap would have been?
#   2. does lambda=16 blow the gradient up? (B trains at 16x on a 2e-4 base = 3.2e-3)
#   3. what is s/it on this B200, so the run length is a measurement and not a guess?
set -u
PY=/workspace/envs/infer/bin/python
CORPUS=/workspace/repo_rodri/experiments/42-merged-corpus/runs/42_merged_v1/train.jsonl
MODEL=/workspace/models/qwen3-vl-8b
CONN="model.visual.merger.linear_fc1 model.visual.merger.linear_fc2 model.visual.deepstack_merger_list.0.linear_fc1 model.visual.deepstack_merger_list.0.linear_fc2 model.visual.deepstack_merger_list.1.linear_fc1 model.visual.deepstack_merger_list.1.linear_fc2 model.visual.deepstack_merger_list.2.linear_fc1 model.visual.deepstack_merger_list.2.linear_fc2"

run () {
  local ratio=$1 tag=$2
  local out=/workspace/rung54/runs/probe_${tag}
  mkdir -p "$out"
  echo "===== PROBE ratio=${ratio} -> ${out}"
  $PY /workspace/rung54/_tools/train_stack.py --ratio "$ratio"     --evidence "$out/RESULTS_loraplus_gate.json" -- sft     --model $MODEL --model_type qwen3_vl --tuner_type lora     --lora_rank 8 --lora_alpha 32 --lora_dropout 0.1     --target_modules all-linear $CONN     --freeze_vit false --freeze_aligner false --freeze_llm false     --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03     --per_device_train_batch_size 1 --gradient_accumulation_steps 16     --torch_dtype bfloat16 --attn_impl sdpa --seed 42     --dataset $CORPUS --split_dataset_ratio 0     --optim adamw_torch_fused --weight_decay 0.1 --gradient_checkpointing true     --max_steps 15 --num_train_epochs 1 --save_strategy no --logging_steps 1     --output_dir "$out" 2>&1 | tee "$out/probe.log"
}
run 1.0 control
run 4.0 lp4
run 16.0 lp16
echo ALL_PROBES_DONE
