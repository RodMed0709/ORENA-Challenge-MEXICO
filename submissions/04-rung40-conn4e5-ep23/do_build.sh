#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="frame-algorithm"

# 🔴 PRE-FLIGHT, added 2026-08-27. Two NameErrors shipped in this container in eleven
# days and neither was reachable by any test we ran: the CPU wiring smoke returns from
# `answer_batch` before the second one, and the model itself answered 4,000 questions
# natively through the same vLLM path. The build gate is the cheap catch. Pure stdlib,
# no GPU, no weights, milliseconds.
echo "=+= Pre-flight: every name read in inference.py must be bound"
python3 "${SCRIPT_DIR}/check_undefined_names.py"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  "$SCRIPT_DIR" 2>&1

# The answer-boundary test needs the SDK (`datasets`), which is installed in the image
# and not on the host, and the vendored source, which is in the repo and not in the
# image -- so it runs INSIDE the container with the repo mounted. It checks
# `normalize_answer` and `clamp_class_tokens` against the SDK's real verifiers.
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd )
echo "=+= Answer-boundary test, inside the image, against the vendored SDK"
docker run --rm \
  --platform=linux/amd64 \
  --volume "${REPO_ROOT}":/repo:ro \
  --workdir /repo/submissions/04-rung40-conn4e5-ep23 \
  --entrypoint python \
  "$DOCKER_IMAGE_TAG" test_normalize_answer.py

# And the CALL SITE, which `test_normalize_answer.py` cannot see and the wiring smoke
# cannot reach. This is the test that would have caught the 2026-08-25 failure.
echo "=+= answer_batch post-processing, with a stub vLLM"
docker run --rm \
  --platform=linux/amd64 \
  --volume "${REPO_ROOT}":/repo:ro \
  --workdir /repo/submissions/04-rung40-conn4e5-ep23 \
  --entrypoint python \
  "$DOCKER_IMAGE_TAG" test_answer_batch_postprocess.py
