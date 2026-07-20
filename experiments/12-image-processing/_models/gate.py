"""Rung 12 — the identity gate. Library; a notebook (or a one-off cell) drives it.

Two independent checks, both of which must pass before any arm is worth running:

1. FLAG-OFF IDENTITY — with ``enhance=None`` the engine must reproduce rung 06's
   ``predictions.json`` byte for byte. This is what makes the A/B single-variable.
2. CONTROL RE-SCORE — rung 06's committed predictions must still reproduce their
   canonical `fo_class` accuracy. Never trust an artifact because of its path;
   rung 10 gated its control the same way and that is why its A/B was readable.
"""

from __future__ import annotations

import json
from pathlib import Path


def flag_off_identity(cfg, reference_predictions: str | Path, n: int = 50) -> dict:
    """Run `n` questions with the flag OFF and diff against a reference run.

    `cfg` must already point at the same model/checkpoint that produced the
    reference — comparing across checkpoints (or GPU classes) fails for reasons
    that have nothing to do with the flag.
    """
    from frame.data import FrameProvider, load_frame_items
    from frame.engine import QwenFrameEngine

    assert getattr(cfg, "enhance", None) is None, "gate must run with the flag OFF"

    # predictions.json is a list of {qID, content, latency}
    ref = {r["qID"]: r["content"] for r in json.loads(Path(reference_predictions).read_text())}

    items = load_frame_items(cfg)
    items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    items = [it for it in items if it.request.qID in ref][:n]

    engine = QwenFrameEngine(cfg)
    engine.load()
    provider = FrameProvider(cfg)

    same, diffs = 0, []
    for it in items:
        provider.ensure_reader(it)
        img = provider.get_frame(it)
        got = engine.predict(img, it.request.question)
        want = ref[it.request.qID]
        if got == want:
            same += 1
        else:
            diffs.append({"qID": it.request.qID, "got": got, "want": want})
    provider.close()

    return {"n": len(items), "identical": same, "diffs": diffs[:10], "passed": same == len(items)}


def control_rescore(inspect_csv: str | Path, expected: dict[str, float], tol: float = 1e-4) -> dict:
    """Re-score a committed run and check it still reproduces its canonical numbers.

    `expected` maps e.g. {"fo_class_ID": 0.6120, "fo_class_OOD": 0.6239}.
    """
    import pandas as pd

    df = pd.read_csv(inspect_csv)
    df["distribution"] = df["qID"].str.split("__").str[0].map(
        {"heico": "OOD", "lapchole": "ID"}
    )
    got = {}
    for dist in ("ID", "OOD"):
        sub = df[(df["answer_format"] == "fo_class") & (df["distribution"] == dist)]
        got[f"fo_class_{dist}"] = float(sub["correct"].mean())

    deltas = {k: got[k] - v for k, v in expected.items()}
    return {
        "got": got,
        "expected": expected,
        "deltas": deltas,
        "passed": all(abs(d) <= tol for d in deltas.values()),
    }
