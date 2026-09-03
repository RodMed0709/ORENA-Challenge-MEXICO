# Fully qualified on purpose: `docker` defaults to docker.io but `buildah` refuses a
# short name without a TTY, and this image is built with buildah (UNAM has no Docker
# daemon and we are not in its group). Explicit is also the safer form anywhere.
FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS frame-algorithm-amd64

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
# 2026-08-16, submission 04: the assertions are REPLACED, not dropped. Submission 02's
# gate demanded torch 2.5 / transformers 4.57 / numpy 1.x, which is right for the 8B on HF
# and wrong here -- vLLM pins its own torch (2.13+cu130 measured) and needs transformers
# 5.x, because `qwen3_5` does not exist in 4.57 and the model will not load without it.
#
# The gate's PURPOSE is unchanged: catch a dependency that silently replaces something
# critical, at BUILD time, where it is free to fix. What changed is that it now asserts
# CAPABILITIES rather than version strings -- "transformers knows qwen3_5" is the thing we
# actually depend on, and it survives a version bump that a `startswith('5.')` would not.
#
# 🔴 The `ninja` check is new and earns its place: vLLM's EngineCore shells out to it and
# its absence surfaces as an opaque FileNotFoundError INSIDE a worker subprocess, at run
# time, offline. It cost a debugging cycle in rung 44. Here it costs a failed build.
RUN python -c "\
import shutil, torch, torchvision, transformers, numpy, vllm, focus;\
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES;\
print('torch', torch.__version__, 'cuda', torch.version.cuda);\
print('torchvision', torchvision.__version__);\
print('transformers', transformers.__version__, '| vllm', vllm.__version__, '| numpy', numpy.__version__);\
print('orena-focus', getattr(focus, '__version__', 'n/a'));\
assert torch.version.cuda, 'torch lost its CUDA build - a --user wheel shadowed the base image';\
assert 'qwen3_5' in CONFIG_MAPPING_NAMES, f'transformers {transformers.__version__} does not know qwen3_5 - the 27B cannot load (4.57 does NOT have it)';\
assert shutil.which('ninja'), 'ninja is not on PATH - vLLM EngineCore fails at RUNTIME with an opaque FileNotFoundError';\
print('gate OK: cuda + qwen3_5 + ninja + vllm')"

# 🔴 REMOVE torchcodec. Added 2026-09-01, after it broke `from vllm import LLM` in the
# built image. It is a TRANSITIVE dependency of vllm 0.26.0 — never named in
# requirements.txt — and pip resolved 0.16.0, whose only wheel links CUDA 13:
#
#     ldd torchcodec/libtorchcodec_image.so
#       libnvrtc.so.13  => not found
#       libcudart.so.13 => not found
#
# So the deliberate cu128 pin covered torch and torchvision and MISSED the one package
# that reaches CUDA through a compiled .so. vLLM imports it unconditionally on the
# `vllm.multimodal.__init__ -> hasher -> media -> connector -> video` chain, and its own
# guard at `vllm/multimodal/video.py:34` catches `(ImportError, RuntimeError)` — while
# this raises **OSError**. One exception class is the whole bug.
#
# Removing it makes that import raise ImportError, which the guard DOES catch and
# replaces with a PlaceholderModule. torchcodec is a video-decoding backend; this
# container is handed PNG frames and never decodes video, so nothing we use is lost.
# (The alternative — LD_LIBRARY_PATH to the bundled nvidia/cu13/lib — also works, and was
# rejected: it puts a CUDA 13 runtime in the same process as a cu128 torch, on a host
# whose driver we have never seen. That is the exact risk the cu128 pin was chosen to avoid.)
RUN python -c "\
import shutil, glob, site, os;\
bases = site.getsitepackages() + [site.getusersitepackages()];\
pats = ['torchcodec', 'torchcodec.libs', 'torchcodec-*.dist-info', 'torchcodec-*.egg-info'];\
hits = [p for b in bases for pat in pats for p in glob.glob(os.path.join(b, pat))];\
[shutil.rmtree(p, ignore_errors=True) for p in hits];\
print('torchcodec removed:', hits or '(not installed)')"

# 🔴 `from vllm import LLM`, not `import vllm`. Added 2026-09-01. The gate above imported
# the vllm PACKAGE, which succeeds, and never touched the SYMBOL, which is what
# `load_model()` actually needs — and which died on the torchcodec/CUDA-13 mismatch. The
# package import does not walk `vllm.multimodal`; the symbol import does. This one line is
# the difference between catching that at build time and shipping a container that loads
# 33 GB of weights and then answers nothing.
RUN python -c "\
from vllm import LLM, SamplingParams;\
import vllm.multimodal.video as _v;\
print('gate OK: vllm LLM symbol imports, video backend =', type(_v.VideoDecoder).__name__)"

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

# 🔴 vLLM's bin/ must be on PATH: EngineCore looks up `ninja` there, and calling the
# interpreter directly is not enough. Cost one debugging cycle in rung 44.
ENV PATH=/home/user/.local/bin:$PATH

# The thinking switch. Build once, run twice: ENABLE_THINKING=0 and =1 differ by exactly
# this and nothing else, so the challenge's own data can answer what our local proxy
# cannot (heico is 100 % OOD and the leaderboard proxy is ID-only).
ENV ENABLE_THINKING=0

ENTRYPOINT ["python", "inference.py"]
