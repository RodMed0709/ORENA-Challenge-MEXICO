"""Rung 50 arm B — rung 47's recipe, rung 47's corpus, ONE variable: the loss.

🔴 The single variable here is stronger than arm A's. ms-swift is launched as a SUBPROCESS with
a CLI argv, so a Python `compute_loss_func` cannot travel as an argument. It travels in the
`swift` shim instead — the same shim rung 19b already needed for a dead shebang — which imports
swift, patches `Seq2SeqTrainer.__init__` to install the hook, and then calls `cli_main()`.
⇒ **the argv is byte-identical to rung 47's. The diff is EMPTY, not one flag.**

The hook is `frame.loss.continuation_weights`: tokens after the first separator in an answer are
up-weighted, everything else keeps weight 1.0, and the mean over supervised tokens stays exactly
1.0 so the loss MAGNITUDE — and therefore the effective learning rate under `max_grad_norm=1.0`
— does not move. Single-class answers have no separator and come out uniform, i.e. identical to
the control: the arm touches only the 1,924 multi-class rows it is aimed at.

Gates B-G1/B-G2 ran offline (`gate_arm_b.py`) and proved both existing weight functions are
arithmetic identities at `per_device=1`, which is how rung 22 actually died.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

STORE = Path("/mnt/storage/uaq_user")
REPO = STORE / "repo_rod"
RUNG = STORE / "rung50"
MODEL = STORE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

CORPUS = STORE / "rung47/corpus/train_mntpaths.jsonl"
R47_SHA = "374c2a236c08e6d5b7b17c978f6c1a2cef3cedd33667691751588257d97dc9e2"
W = float(os.environ.get("RUNG50_W", "2.0"))
SMOKE = os.environ.get("RUNG50_SMOKE", "1") == "1"

os.environ.setdefault("HF_HOME", str(STORE / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── the seam: a swift shim that installs the hook, then runs the ordinary CLI ───────────
SHIM = RUNG / "code" / "bin_armb"
SHIM.mkdir(parents=True, exist_ok=True)
SEP_JSON = RUNG / "code" / "separator_ids.json"
if not SEP_JSON.exists():
    raise SystemExit(f"missing {SEP_JSON} — run gate_arm_b.py first; the ids are derived, "
                     "never hard-coded")
hook_cfg = RUNG / "code" / "armb_hook.json"
MARKER = RUNG / "code" / "armb_hook_calls.txt"
if MARKER.exists():
    MARKER.unlink()
hook_cfg.write_text(json.dumps({"repo_src": str(REPO / "src"),
                                "separator_ids": str(SEP_JSON),
                                "marker": str(MARKER),
                                "continuation_weight": W}, indent=1))

(SHIM / "swift").write_text(f'''#!{sys.executable}
"""rung 50 arm B — swift with `continuation_weights` installed on the trainer.

🔴 It does NOT call `cli_main()`. `swift.cli.main.cli_main` ends in
`subprocess.run([python, sft.py, *argv])` — a FRESH interpreter — so a monkeypatch installed
here would live in the parent and never reach the process that trains. That is exactly how the
first attempt silently produced a control run: the shim executed, the markers never printed,
and the loss hook was a no-op. Measured, then fixed.

Instead it resolves the same `sft.py` through ms-swift's own `ROUTE_MAPPING`, patches the
trainer, and runs that file IN THIS PROCESS with `runpy`. Single GPU means `get_torchrun_args()`
is None and the child would have been a plain `python sft.py` anyway, so this is equivalent to
what the CLI does, minus the fork.
"""
import importlib.util, json, runpy, sys
from pathlib import Path

CFG = json.loads(Path({str(hook_cfg)!r}).read_text())
sys.path.insert(0, CFG["repo_src"])
from frame.loss import continuation_weights, make_compute_loss_func

SEP = set(json.loads(Path(CFG["separator_ids"]).read_text()))
W = CFG["continuation_weight"]
MARK = Path(CFG["marker"])
_seen = {{"calls": 0}}


def _weights(labels, **kw):
    _seen["calls"] += 1
    if _seen["calls"] == 1:
        print(f"[rung50-armB] compute_loss_func FIRED — {{len(SEP)}} sep ids, w={{W}}", flush=True)
    if _seen["calls"] % 200 == 0:
        MARK.write_text(str(_seen["calls"]))
    return continuation_weights(labels, SEP, W)


_fn = make_compute_loss_func(weight_fn=_weights, enabled=True)

from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
_orig = Seq2SeqTrainer.__init__


def _patched(self, *a, **k):
    _orig(self, *a, **k)
    self.compute_loss_func = _fn
    print("[rung50-armB] hook installed on Seq2SeqTrainer", flush=True)


Seq2SeqTrainer.__init__ = _patched

from swift.cli.main import ROUTE_MAPPING

_method = sys.argv[1].replace("_", "-")
_origin = importlib.util.find_spec(ROUTE_MAPPING[_method]).origin
print(f"[rung50-armB] running {{_origin}} IN-PROCESS (no subprocess fork)", flush=True)
sys.argv = [_origin, *sys.argv[2:]]
try:
    runpy.run_path(_origin, run_name="__main__")
finally:
    MARK.write_text(str(_seen["calls"]))
''')
(SHIM / "swift").chmod(0o755)
os.environ["PATH"] = (f"{SHIM}{os.pathsep}{Path(sys.executable).parent}"
                      + os.pathsep + os.environ.get("PATH", ""))

for p in (REPO / "src", REPO / "vendor/orena-focus/src",
          REPO / "experiments/21-recipe-sweep/_models",
          REPO / "experiments/18-count-aug/_models",
          REPO / "experiments/16-count-probes/_models",
          REPO / "experiments/06-vit-lora/_models",
          REPO / "experiments/02-lora-sft/_models"):
    if p.is_dir():
        sys.path.insert(0, str(p))

import recipe_sweep_train as rst  # noqa: E402

cfg = rst.RecipeSweepConfig(
    exp_dir=RUNG,
    run_name="50b_contweight_smoke" if SMOKE else "50b_contweight_v1",
    model_path=MODEL,
    data_root=STORE / "orena-data",
    manifest_path=REPO / "experiments/splits/frame_ood_v1.csv",
    control_train_jsonl=CORPUS,
    control_sha256=R47_SHA,
    learning_rate=2e-4, lora_rank=8, lora_alpha=32, max_grad_norm=1.0, vit_lr=None,
    num_train_epochs=5,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    seed=42, smoke=SMOKE, smoke_max_steps=4,
)
print(f"run_dir {cfg.run_dir}\nsmoke   {SMOKE}\nw       {W}", flush=True)

ds = rst.assert_dataset_is_the_controls(cfg)
assert ds["n_rows"] == 14415, f"expected 14,415 rows, got {ds['n_rows']}"
assert ds["sha256"] == R47_SHA, "arm B must train rung 47's corpus byte-for-byte"
print(f"OK G1 dataset: rung 47's own {ds['n_rows']} rows, sha256 {ds['sha256'][:12]}...", flush=True)

# ── B-G5 — the argv must be IDENTICAL to rung 47's. Not one flag may move. ──────────────
r47_cfg = replace(cfg, smoke=False)
a = rst._as_map(rst.swift_args_21(r47_cfg))
b = rst._as_map(rst.swift_args_21(replace(cfg, smoke=False)))
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
assert not diff, f"B-G5 FAILED: argv is not identical to rung 47's: {sorted(diff)}"
print("OK B-G5 single-variable: argv identical to rung 47; the ONLY difference is the loss",
      flush=True)

cfg.run_dir.mkdir(parents=True, exist_ok=True)
(cfg.run_dir / "argv.json").write_text(json.dumps(rst.swift_args_21(replace(cfg, smoke=False)), indent=1))
(cfg.run_dir / "gates.json").write_text(json.dumps(
    {"corpus_sha256": ds["sha256"], "n_rows": ds["n_rows"], "continuation_weight": W,
     "argv_diff_vs_r47": {}, "shim": str(SHIM / "swift")}, indent=1))
(cfg.run_dir / "pip_freeze.txt").write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

t0 = time.perf_counter()
ckpt = rst._train(cfg)
print(f"\ntrained in {(time.perf_counter() - t0) / 3600:.2f} h -> {ckpt}", flush=True)
