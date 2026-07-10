#!/usr/bin/env bash
# Set up persistent Python envs ON THE NETWORK VOLUME so they survive pod restarts.
# Run this INSIDE a RunPod GPU pod that has the ORENA-CHALLENGE volume mounted at /workspace.
#   bash scripts/setup_gpu_env.sh
#
# Creates:
#   /workspace/envs/train  (ms-swift LoRA fine-tuning)   <- requirements/train.txt
#   /workspace/envs/serve  (vLLM offline inference)      <- requirements/serve.txt
# Keep them SEPARATE: vLLM and ms-swift pin conflicting torch/flash-attn.
set -e
REPO_DIR="${1:-$(pwd)}"          # path to the cloned repo (has requirements/)
VENV_ROOT=/workspace/envs

python -m pip install -q -U pip virtualenv

echo "=== [1/2] TRAIN env (ms-swift) -> $VENV_ROOT/train ==="
python -m venv "$VENV_ROOT/train"
"$VENV_ROOT/train/bin/pip" install -q -U pip
# torch already present in the RunPod CUDA image; do not reinstall a CPU build
"$VENV_ROOT/train/bin/pip" install -q -r "$REPO_DIR/requirements/train.txt"
echo "train env ready. Activate: source $VENV_ROOT/train/bin/activate"

echo "=== [2/2] SERVE env (vLLM) -> $VENV_ROOT/serve ==="
python -m venv "$VENV_ROOT/serve"
"$VENV_ROOT/serve/bin/pip" install -q -U pip
"$VENV_ROOT/serve/bin/pip" install -q -r "$REPO_DIR/requirements/serve.txt"
echo "serve env ready. Activate: source $VENV_ROOT/serve/bin/activate"

echo "=== DONE. Data is at /workspace/orena-data/{heico,lapchole} ==="
"$VENV_ROOT/train/bin/python" -c "import transformers,swift; print('train ok: transformers', transformers.__version__)" || true
