"""Rung 54 — launch ms-swift IN-PROCESS so a monkeypatch survives, and gate before training.

🔴 **`swift.cli.main.cli_main()` ends in `subprocess.run([python, sft.py])`.** Anything
patched in the parent — an optimizer registration, a loss hook — stays in the parent and
the child trains a silent control. Rung 50 arm B lost a full run to exactly this and only
found out because its marker never appeared in the log. So we resolve the route's module
origin ourselves and `runpy.run_path` it in THIS interpreter, after registering.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _install_gate(ratio: float, evidence_path: Path) -> None:
    """Register LoRA+ and wrap the callback so its FIRST optimizer is gated and logged."""
    import loraplus

    loraplus.register()

    cls = loraplus.LoraPlusOptimizerCallback
    original = cls.create_optimizer

    def create_optimizer(self, model=None):
        opt = original(self, model=model)
        target = model if model is not None else self.trainer.model
        ev = loraplus.gate(opt, target)
        ev["loraplus_lr_ratio"] = ratio
        ev["base_lr"] = float(self.args.learning_rate)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(ev, indent=1, default=str), encoding="utf-8")
        print(f"[rung54] LoRA+ GATE PASSED ratio={ratio} groups={ev['n_groups']} "
              f"lrs={ev['distinct_lrs']} fast={ev['fast_group_example']}", flush=True)
        return opt

    cls.create_optimizer = create_optimizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True,
                    help="LoRA+ lambda. 1.0 means run the control: plain optimizer, no patch.")
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("swift_args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    args = list(ns.swift_args)
    if args and args[0] == "--":
        args = args[1:]
    if not args or args[0] != "sft":
        raise SystemExit("expected the swift route as the first positional arg, e.g. `sft ...`")

    if ns.ratio != 1.0:
        os.environ["ORENA_LORAPLUS_RATIO"] = str(ns.ratio)
        _install_gate(ns.ratio, ns.evidence)
        args += ["--optimizer", "loraplus"]
    else:
        print("[rung54] ratio=1.0 — CONTROL: no registration, no --optimizer flag", flush=True)

    from swift.cli.main import ROUTE_MAPPING

    origin = importlib.util.find_spec(ROUTE_MAPPING[args[0].replace("_", "-")]).origin
    sys.argv = [origin, *args[1:]]
    runpy.run_path(origin, run_name="__main__")


if __name__ == "__main__":
    main()
