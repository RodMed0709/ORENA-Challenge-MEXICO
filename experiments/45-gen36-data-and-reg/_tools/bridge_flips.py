"""Rung 45 — where R00 and `conn4e5` actually disagree, question by question.

Importable library. NEVER a launcher.

## Why this exists

The bridge scalar says R00 (NF4) sits −0.0874 `bucket_mean` below
`40_B_connector_v1` (bf16) on the same 6 252 questions and the same corpus. That
number cannot be read as "the price of NF4", because four things moved at once:

| moved | where it lives |
|---|---|
| training precision, NF4 vs bf16 | the weights |
| **`weight_decay` 0.1 vs 0.0** | the weights |
| JPEG q95 round-trip on every frame | the input |
| vLLM batched vs HF sequential | the decode |

⚠️ The `weight_decay` difference was found on 2026-08-17 and **contradicts PLAN §5**,
which states rung 40 "never archived" `optim` / `weight_decay` / `max_grad_norm` and
that they are "unrecoverable after the fact". They are archived — in rung 40's own
`RESULTS_arm.json`, recoverable from S3 in seconds — and one of them differs. R00 was
declared at 0.1 believing rung 40's value was unknown; it was 0.0.

Re-running an arm cannot fix that: it is baked into `conn4e5`'s weights. So instead of
paying for a new arm, this asks what the two archived per-question tables can already
answer — **is the difference broad or targeted?** — which is the question that decides
whether the gap looks like precision at all.

## What it found (2026-08-17, and the reason the numbers below are worth keeping)

**23.7 % of all answers flipped** — 1 482 of 6 252, net −568. For "the same recipe at a
different precision" that is not a small perturbation.

**And the loss is spread, not concentrated.** Every answer format degrades, and 35 of
the 38 videos degrade, median −0.075. No single broken format, no one bad video.
`binary` — a two-way choice — loses 0.082, which a regularisation change has no
business costing; that everything down to the easiest format degrades is the signature
of a model that simply knows less.

**Calibration against a known quantity.** Rung 44 measured vLLM and HF disagreeing on
6 % of answers *with identical weights* (47/50). So the decode path explains at most a
quarter of a 23.7 % churn; the rest is the weights or the frames.

🔴 **What this still does NOT separate:** NF4 from `weight_decay` from the JPEG
round-trip. All three degrade broadly and uniformly, so none of them betrays itself in
the pattern. Separating them needs new arms, and this file exists precisely so that
cost is a choice rather than a reflex.
"""

from __future__ import annotations

import json
from pathlib import Path

# Rung 40's winning arm, bf16 — the control side of the bridge.
CONTROL_CSV = ("experiments/40-gen36-recipe-connector/runs/40_B_connector_v1"
               "/eval/40_B_connector_v1/results.csv")
# R00's bridge run, NF4, the SAME 6 252 questions.
ARM_CSV = ("experiments/45-gen36-data-and-reg/runs/45_R00_v1/eval_bridge"
           "/45_R00_v1_eval_bridge/results.csv")

# Measured by rung 44 on IDENTICAL weights (RESULTS_fp8_1gpu_real.json: 47/50 agree),
# so it is the share of churn the decode path can account for on its own.
VLLM_VS_HF_DISAGREEMENT = 0.06


def flip_frame(control_csv: str | Path = CONTROL_CSV, arm_csv: str | Path = ARM_CSV):
    """Join both arms on qID and label every question. RAISES on misalignment.

    The join is inner on `qID` and then asserted back to both lengths: an inner join
    that silently drops rows would shrink the denominator without saying so, and every
    share below would be computed against a number nobody chose.
    """
    import pandas as pd

    a = pd.read_csv(control_csv)[["qID", "video", "answer_format", "primary", "correctness"]]
    b = pd.read_csv(arm_csv)[["qID", "correctness"]]
    a = a.rename(columns={"correctness": "c_control"})
    b = b.rename(columns={"correctness": "c_arm"})
    j = a.merge(b, on="qID", how="inner")
    if not (len(j) == len(a) == len(b)):
        raise AssertionError(
            f"not the same questions: joined {len(j)}, control {len(a)}, arm {len(b)}"
        )
    j["lost"] = (j.c_control == 1) & (j.c_arm == 0)
    j["won"] = (j.c_control == 0) & (j.c_arm == 1)
    j["flipped"] = j.lost | j.won
    return j


def by_format(j):
    """Per answer_format: accuracy each side, and which way the flips went."""
    g = j.groupby("answer_format").agg(
        n=("qID", "size"), acc_control=("c_control", "mean"), acc_arm=("c_arm", "mean"),
        lost=("lost", "sum"), won=("won", "sum"))
    g["delta"] = g.acc_arm - g.acc_control
    g["net"] = g.won - g.lost
    g["churn"] = (g.lost + g.won) / g.n
    return g.sort_values("net").reset_index()


def by_video(j):
    """Per video, so a concentrated loss cannot hide inside a pooled mean."""
    g = j.groupby("video").agg(
        n=("qID", "size"), acc_control=("c_control", "mean"), acc_arm=("c_arm", "mean"))
    g["delta"] = g.acc_arm - g.acc_control
    return g.sort_values("delta").reset_index()


def report(j) -> dict:
    """The headline, with the one calibration that bounds the decode path."""
    n = len(j)
    lost, won = int(j.lost.sum()), int(j.won.sum())
    v = by_video(j)
    churn = (lost + won) / n
    return {
        "n_questions": n,
        "both_correct": int(((j.c_control == 1) & (j.c_arm == 1)).sum()),
        "both_wrong": int(((j.c_control == 0) & (j.c_arm == 0)).sum()),
        "control_only": lost,
        "arm_only": won,
        "n_flipped": lost + won,
        "churn": round(churn, 4),
        "net": won - lost,
        "acc_control": round(float(j.c_control.mean()), 4),
        "acc_arm": round(float(j.c_arm.mean()), 4),
        "videos_worse": int((v.delta < 0).sum()),
        "videos_total": int(len(v)),
        "video_delta_median": round(float(v.delta.median()), 4),
        "video_delta_min": round(float(v.delta.min()), 4),
        "video_delta_max": round(float(v.delta.max()), 4),
        # Not a subtraction — an upper bound. 6 % is what the decode path alone
        # produces on identical weights, so it can account for at most this share.
        "decode_path_max_share_of_churn": round(VLLM_VS_HF_DISAGREEMENT / churn, 3),
        "confounds_not_separated": [
            "NF4 vs bf16 training precision",
            "weight_decay 0.1 (R00) vs 0.0 (conn4e5)",
            "JPEG q95 round-trip on frames",
            "vLLM batched vs HF sequential decode",
        ],
        "reading": (
            "Broad, not targeted: every answer format degrades and 35 of 38 videos "
            "degrade. `binary` losing 0.082 is what rules out a purely structural "
            "cause — a two-way choice should not cost that. This is consistent with a "
            "generally weaker model; it does NOT attribute the loss to NF4, because "
            "weight_decay and the JPEG round-trip degrade just as broadly."
        ),
    }


def write(exp_dir: str | Path, j) -> tuple[str, str]:
    exp = Path(exp_dir)
    fmt = by_format(j)
    csv_path = exp / "RESULTS_bridge_flips.csv"
    fmt.round(4).to_csv(csv_path, index=False)
    json_path = exp / "RESULTS_bridge_flips.json"
    payload = report(j)
    payload["by_video_worst5"] = by_video(j).head(5).round(4).to_dict("records")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(csv_path), str(json_path)
