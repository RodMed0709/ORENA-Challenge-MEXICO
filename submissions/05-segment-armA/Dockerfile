# SEGMENT track — arm A.
#
# Fully qualified on purpose: `docker` defaults to docker.io but `buildah` refuses a short
# name without a TTY, and some of our build hosts have no Docker daemon. Explicit is also
# the safer form anywhere.
#
# 🔴 The base image is torch **2.8.0 / CUDA 12.8**, NOT the 2.5.1/12.4 the FRAME
# containers use. That is deliberate: the SEGMENT path was validated end to end in
# `/workspace/envs/infer` on torch 2.8.0+cu128, and the FRAME image's older torch was
# already a declared divergence from the environment its numbers came from. Matching the
# validated environment removes that divergence instead of inheriting it.
FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS segment-algorithm-amd64

ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base ships neither the shared libraries opencv-python needs at import
# time nor those decord's bundled FFmpeg links against. Install them HERE, while the build
# still runs as root — everything below runs as the unprivileged 'user'.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libgomp1 \
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
# precedence over the base image's site-packages — so a transitive dependency that drags
# in its own torch wheel would silently replace the CUDA build we rely on, and nothing
# would notice until the container was already scored.
#
# The gate asserts CAPABILITIES, not just version strings, for the four things this
# container cannot run without:
#   * torch is still CUDA-enabled and still 2.8 (the validated line),
#   * transformers knows `qwen3_vl` — without it the model will not load at all,
#   * `peft` can be imported — it applies BOTH adapters, and an image that ships the base
#     model alone would answer every question with an untuned 8B and look merely bad,
#   * `decord` imports and finds its bundled FFmpeg — this is the CLIP DECODER, and its
#     absence surfaces at run time, offline, per question, as a caught exception that
#     turns the whole batch into fallbacks. It cost nothing to check here.
RUN python -c "\
import torch, torchvision, transformers, numpy, peft, decord, cv2, focus;\
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES;\
from transformers import Qwen3VLForConditionalGeneration;\
print('torch', torch.__version__, 'cuda', torch.version.cuda);\
print('torchvision', torchvision.__version__, '| numpy', numpy.__version__);\
print('transformers', transformers.__version__, '| peft', peft.__version__);\
print('decord', decord.__version__, '| opencv', cv2.__version__);\
print('orena-focus', getattr(focus, '__version__', 'n/a'));\
assert torch.version.cuda, 'torch lost its CUDA build - a --user wheel shadowed the base image';\
assert torch.__version__.startswith('2.8'), f'unexpected torch {torch.__version__}';\
assert transformers.__version__ == '4.57.6', f'unexpected transformers {transformers.__version__}';\
assert 'qwen3_vl' in CONFIG_MAPPING_NAMES, 'transformers does not know qwen3_vl';\
print('gate OK: cuda + qwen3_vl + peft + decord')"

# ── Model definition + weights ────────────────────────────────────────────────
# resources/ is NOT in git (~17 GB) and is assembled on the build host. Layout:
#
#   resources/model/                                  Qwen3-VL-8B-Instruct base snapshot:
#                                                     config.json, *.safetensors + index,
#                                                     tokenizer*, preprocessor_config.json,
#                                                     video_preprocessor_config.json,
#                                                     chat_template.jinja
#   resources/adapters/01-rung21-a2-checkpoint-2703/   adapter_config.json + adapter_model.safetensors
#   resources/adapters/02-segment-armA-checkpoint-860/ adapter_config.json + adapter_model.safetensors
#
# 🔴 ADAPTERS, NOT A MERGE, AND IN THAT ORDER. Arm A was trained on top of the MERGED
# rung-21 A2 checkpoint; that merged directory was deleted from the volume to reclaim
# disk, and only the two adapters survive. `inference.py` applies A2, merges it into the
# weights IN MEMORY, then applies arm A — which reproduces exactly what arm A saw.
# Loading them as two live PEFT adapters instead composes them additively against the
# BASE, which is a different model that was never trained or scored. The numeric names
# are the application order and `ADAPTER_ORDER` in inference.py is the authority.
#
# ECR layer-size limit: one image layer must not exceed 50 GB and each COPY produces
# exactly one layer. The 8B base is ~17 GB, so one COPY is fine; the adapters get their
# own layer so a re-merge or a checkpoint swap does not re-push the base.
COPY --chown=user:user resources/model/ /opt/app/resources/model/
COPY --chown=user:user resources/adapters/ /opt/app/resources/adapters/

COPY --chown=user:user inference.py /opt/app/

# Offline: no network at inference. Set here as well as in inference.py so the intent is
# visible in `docker inspect` and so any subprocess inherits it.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# The declared A/B of PLAN D2, exposed so it can never be decided by a `dict.get()`:
# `plain` is what the arm was trained on; `overlayed` burns a clock into pixels the model
# has never seen. Build once, run twice, differing by exactly this.
ENV SEG_VARIANT=plain

ENTRYPOINT ["python", "inference.py"]
