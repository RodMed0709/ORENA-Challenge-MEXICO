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
RUN python -c "\
import torch, transformers, numpy;\
print('torch', torch.__version__, 'cuda', torch.version.cuda);\
print('transformers', transformers.__version__, '| numpy', numpy.__version__);\
assert torch.version.cuda, 'torch lost its CUDA build — a --user wheel shadowed the base image';\
assert torch.__version__.startswith('2.5'), f'unexpected torch {torch.__version__}'"

# ── Model definition + weights ────────────────────────────────────────────────
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
