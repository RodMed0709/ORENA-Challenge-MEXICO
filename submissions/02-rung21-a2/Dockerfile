FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS frame-algorithm-amd64

# torch 2.5.1 + torchvision are pre-installed in the base image (CUDA 12.4).
# requirements.txt only lists additional dependencies.

ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base image ships neither the shared libraries that
# opencv-python (a dependency of orena-focus) needs at import time, nor any
# compiler toolchain. Install system packages HERE, while the build still runs
# as root — everything below runs as the unprivileged 'user'. If one of your
# pip dependencies compiles from source (e.g. flash-attn), add build-essential.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Fail the BUILD, not the submission. `--user` installs to ~/.local, which takes
# precedence over the base image's site-packages — so a transitive dependency that
# drags in its own torch wheel would silently replace the CUDA build we rely on, and
# nothing would notice until the container was already scored. This asserts that torch
# is still CUDA-enabled and still on the 2.5 line, and echoes the versions that a
# future run's build log can be diffed against.
# 2026-07-30, submission 02: this used to assert TORCH ONLY. It printed the transformers
# version without checking it, and never imported qwen_vl_utils, focus or torchvision at
# all -- so a --user wheel shadowing transformers, or a dependency dragging in its own
# torchvision, passed the gate and would only have surfaced at score time, with a
# submission slot already spent. numpy is the sharp one: the requirement allows <3.0,
# which resolves to numpy 2.x against a torch-2.5.1 base predating numpy-2 hardening.
RUN python -c "\
import torch, torchvision, transformers, numpy, qwen_vl_utils, focus;\
print('torch', torch.__version__, 'cuda', torch.version.cuda);\
print('torchvision', torchvision.__version__, 'cuda', torchvision.version.cuda);\
print('transformers', transformers.__version__, '| numpy', numpy.__version__);\
print('qwen-vl-utils', getattr(qwen_vl_utils, '__version__', 'n/a'), '| orena-focus', getattr(focus, '__version__', 'n/a'));\
assert torch.version.cuda, 'torch lost its CUDA build - a --user wheel shadowed the base image';\
assert torch.__version__.startswith('2.5'), f'unexpected torch {torch.__version__}';\
assert torchvision.version.cuda, 'torchvision lost its CUDA build';\
assert transformers.__version__.startswith('4.57'), f'transformers {transformers.__version__} is off the 4.57 line - Qwen3-VL has a HARD floor of 4.57.0';\
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} is 2.x on a torch-2.5.1 base - pin numpy<2 or move the base image'"

# ── Model definition + weights ────────────────────────────────────────────────
# resources/model/ holds the MERGED full checkpoint written by
#   swift export --adapters <ckpt> --merge_lora true --output_dir resources/model
# -- weights AND processor/tokenizer/chat-template, because inference.py calls
# AutoProcessor.from_pretrained(MODEL_PATH). Copying only *.safetensors builds a 17 GB
# image that dies at runtime. The template's model.py / dummy_weights.pt are NOT used.
#
# Submission 02 = rung 21 arm A2: 21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703.
# resources/ is deliberately NOT in git (17 GB). The merge command above regenerates it,
# and it was verified deterministic -- two independent merges, byte-identical.
# OLD COMMENT BELOW IS RETIRED:
# resources/ holds the model definition (model.py) and a small committed dummy
# checkpoint (dummy_weights.pt). For a real submission, replace dummy_weights.pt
# with your trained checkpoint.
#
# ECR layer-size limit: a single image layer must not exceed 50 GB, and each
# COPY instruction produces exactly one layer. A checkpoint large enough to push
# one COPY past that limit must be split into chunks and copied with several COPY
# instructions (one layer each), then reassembled at runtime — e.g. pre-split
# with `split -b 45G weights.pt resources/weights.part-` and:
#     COPY --chown=user:user resources/weights.part-aa /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ab /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ac /opt/app/resources/
# reassembling with `cat resources/weights.part-* > weights.pt` before load.
COPY --chown=user:user resources/ /opt/app/resources/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
