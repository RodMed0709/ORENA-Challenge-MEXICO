"""Rung 50 arm B gates — B-G1 (OFF byte-identical) and B-G2 (ON is NOT an identity).

Pure torch, no ms-swift, no GPU. B-G2 is the gate rung 22 did not have: it proves the weights
actually differ from all-ones AFTER normalisation at `per_device_train_batch_size=1`, which is
the exact condition under which `row_group_weights` and `segment_weights` collapse to the
control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STORAGE = Path("/mnt/storage/uaq_user")
REPO = STORAGE / "repo_rod"
OUT = REPO / "experiments/50-set-enumeration"
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from frame import loss as L  # noqa: E402

SEP_TEXT = ","


def separator_ids(tok) -> set[int]:
    """Every token id whose surface form contains a comma. Derived, never guessed."""
    ids = set()
    for piece, i in tok.get_vocab().items():
        s = tok.convert_tokens_to_string([piece])
        if SEP_TEXT in s:
            ids.add(i)
    return ids


def main() -> None:
    from transformers import AutoTokenizer

    base = next((STORAGE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots").glob("*"))
    tok = AutoTokenizer.from_pretrained(str(base))
    sep = separator_ids(tok)
    print(f"separator ids: {len(sep)}", flush=True)
    # Written to disk so the training shim LOADS them rather than re-deriving them in a
    # process whose tokenizer could differ. The ids are the arm, not an implementation detail.
    sep_path = STORAGE / "rung50/code/separator_ids.json"
    sep_path.parent.mkdir(parents=True, exist_ok=True)
    sep_path.write_text(json.dumps(sorted(sep)))
    print(f"  -> {sep_path}", flush=True)

    # ── B-G1 — OFF returns None, so no code of ours runs ────────────────────────────────
    off = L.make_compute_loss_func(weight_fn=None, enabled=False)
    assert off is None, "B-G1 FAILED: disabled hook is not None"
    print("OK B-G1: enabled=False returns None -> the trainer takes its own path", flush=True)

    # ── B-G2 — ON is NOT an identity at per_device=1, on a REAL multi-class answer ──────
    W = 2.0
    results = {}
    for name, answer in (("multi", "Clip, Sponge"), ("multi3", "Clip, Sponge, Needle"),
                         ("single", "Clip")):
        ids = tok(answer, add_special_tokens=False).input_ids
        labels = torch.full((1, 8 + len(ids)), L.IGNORE_INDEX, dtype=torch.long)
        labels[0, 8:] = torch.tensor(ids)          # prompt masked, answer supervised
        w = L.continuation_weights(labels, sep, W)
        sup = labels.ne(L.IGNORE_INDEX)
        vals = w[sup]
        results[name] = {
            "answer": answer, "n_answer_tokens": len(ids),
            "distinct_weights": sorted({round(float(v), 4) for v in vals}),
            "mean_weight": round(float(vals.mean()), 6),
            "is_identity": bool(torch.allclose(vals, torch.ones_like(vals))),
        }
        print(f"  {name:7s} {answer!r:26s} weights={results[name]['distinct_weights']} "
              f"mean={results[name]['mean_weight']} identity={results[name]['is_identity']}",
              flush=True)

    assert not results["multi"]["is_identity"], (
        "B-G2 FAILED: the weights normalise to all-ones on a multi-class row — this is exactly "
        "how rung 22 died and the arm would be a silent no-op")
    assert not results["multi3"]["is_identity"], "B-G2 FAILED on the 3-class row"
    assert results["single"]["is_identity"], (
        "single-class rows must be untouched; if they are not, the arm changes rows it was not "
        "aimed at and is no longer the declared variable")
    assert abs(results["multi"]["mean_weight"] - 1.0) < 1e-4, (
        "normalisation invariant broken: the mean supervised weight must be 1.0")
    print("OK B-G2: ON differs from the control on multi-class rows and ONLY on those", flush=True)

    # ── the control the existing functions cannot pass, recorded for the note ───────────
    ids = tok("Clip, Sponge", add_special_tokens=False).input_ids
    labels = torch.full((1, 8 + len(ids)), L.IGNORE_INDEX, dtype=torch.long)
    labels[0, 8:] = torch.tensor(ids)
    rg = L.row_group_weights(labels, ["fo_class"], {"fo_class": 2.0})
    sg = L.segment_weights(labels, [(8, 8 + len(ids))], 2.0)
    sup = labels.ne(L.IGNORE_INDEX)
    legacy = {
        "row_group_weights_is_identity": bool(torch.allclose(rg[sup], torch.ones_like(rg[sup]))),
        "segment_weights_is_identity": bool(torch.allclose(sg[sup], torch.ones_like(sg[sup]))),
    }
    print(f"  legacy functions at per_device=1: {legacy}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULTS_arm_b_gates.json").write_text(json.dumps(
        {"n_separator_ids": len(sep), "continuation_weight": W,
         "cases": results, "legacy_at_batch1": legacy}, indent=1))


if __name__ == "__main__":
    main()
