"""Execute `answer_batch`'s post-processing block — the path NO other test can reach.

    docker run --rm -v "$(git rev-parse --show-toplevel)":/repo:ro \\
      -w /repo/submissions/04-rung40-conn4e5-ep23 \\
      --entrypoint python frame-algorithm test_answer_batch_postprocess.py

🔴 Why this file exists. On 2026-08-25 this container failed all 100 platform cases because
`normalize_answer` was called at line 429 and defined nowhere. Nothing caught it:

  * the CPU wiring smoke returns from `answer_batch` at `if llm is None`, thirty-four lines
    BEFORE the block — structurally unreachable, not merely untested;
  * the model tests answer through vLLM on a GPU with 33 GB of weights, which is why they
    were never run on this file's glue;
  * `test_normalize_answer.py` tests the functions in isolation and would have passed even
    with the call site broken.

The gap is the CALL SITE. This closes it by substituting a fake vLLM that returns canned
strings — enough to run the real block against the real SDK registry, on CPU, in seconds.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

INFERENCE = Path(__file__).with_name("inference.py")

# `from vllm import SamplingParams` inside answer_batch drags in torchcodec, which needs
# libnvrtc and therefore a CUDA runtime. vLLM is not under test here; the block after it
# is. Stub the single symbol it imports, before the module is executed.
_v = types.ModuleType("vllm")
_v.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
sys.modules.setdefault("vllm", _v)

_spec = importlib.util.spec_from_file_location("inference_under_test", INFERENCE)
inf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inf)          # safe: run() is behind the __main__ guard

# 2 format repairs, 1 illegal fo_class token, 1 answer that must come back byte-identical.
CANNED = ["1.", "Yes.", "Clip, Banana", "Clip"]
EXPECT = ["1", "yes", "Clip", "Clip"]


class _Out:
    def __init__(self, text: str):
        self.outputs = [types.SimpleNamespace(text=text)]


class _StubLLM:
    def chat(self, convs, sp, chat_template_kwargs=None):
        assert len(convs) == len(CANNED), f"{len(convs)} conversations for {len(CANNED)} canned"
        return [_Out(t) for t in CANNED]


def main() -> int:
    from focus import load_requests
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="answer-batch-"))
    qids = [f"q-{i:04d}" for i in range(len(CANNED))]
    for q in qids:
        Image.new("RGB", (64, 48), (30, 60, 90)).save(tmp / f"{q}.png")
    (tmp / "request.json").write_text(json.dumps([
        {"qID": q, "videoID": "vid-01", "start_time": 1.0, "end_time": 1.0,
         "procedure_type": "laparoscopic cholecystectomy", "question": f"synthetic {i}"}
        for i, q in enumerate(qids)
    ]))

    requests = load_requests(tmp / "request.json")
    frames = {q: tmp / f"{q}.png" for q in qids}
    legal = inf.legal_class_names()          # read from the SDK, never hard-coded (RULES §8b)

    answers, _ = inf.answer_batch(_StubLLM(), requests, frames, "SYSTEM", legal)

    print("\n  raw -> answer")
    for raw, got, exp in zip(CANNED, answers, EXPECT):
        flag = "ok " if got == exp else "FAIL"
        print(f"  [{flag}] {raw!r:16} -> {got!r:10} expected {exp!r}")
    assert answers == EXPECT, f"MISMATCH: {answers} != {EXPECT}"
    assert all(answers), "an answer came back empty — the DEGRADED breaker would fire"
    print("\nPASS — the call site runs: 2 repairs, 1 illegal token dropped, "
          "1 byte-identical passthrough, 0 empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
