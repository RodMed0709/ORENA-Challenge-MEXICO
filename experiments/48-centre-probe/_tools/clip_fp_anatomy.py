"""Why the `Clip` cell cannot rank: the false positives are a RAMP, not a constant prior.

v2's negatives are provable: before a video's first `clipper,clip,*` triplet the clip does not
exist yet. In every one of the 15 held-out videos the last negative frame precedes the first
positive one, so the window is clean.

On those 1,332 frames all four models answer the bare string `'Clip'` in ~9 of every 10, and
the rate is NOT flat — it climbs monotonically with proximity to the clipping event:

    first decile of the window   0.57        gap > 300 frames   0.87   (n=808)
    half way                     0.98        gap 100-300        0.98   (n=358)
    last decile                  1.00        gap < 100          1.00   (n=166)

⇒ the models are not emitting a constant. They are reading the SURGICAL PHASE — the dissection,
the applier entering the field — and inferring a clip that has not been placed. That is a
different defect from the frequency prior [[frequency-prior-is-the-failure-shape]] describes,
and it is consistent with step 4's finding that the model DOES look
([[vcd-has-nothing-to-subtract]]): it looks, and the phase prior overrides what it sees.

⚠️ The top of the ramp is also where the LABEL is weakest — a clip deposited slightly before
the first annotated triplet. That band is not interpretable. But `gap > 300` is 61 % of the
negatives, its label is not in doubt, and it still runs 0.81-0.95. The phenomenon survives
without the contaminated band.

🔴 And the reason this stays a RESULT rather than becoming an instrument: clip aggressiveness
does not track the platform. A2 is the LEAST aggressive of the three anchors and rung 42 beats
it by 0.052. Every temporal cut inverts the pair. Read it as anatomy, choose with `bag_f1`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MODELS = ["r06", "a2", "r42", "r47"]
PLATFORM = {"r06": 0.4767, "a2": 0.5288, "r42": 0.5809, "r47": None}


def anatomy(runs: Path, corpus: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame([json.loads(l) for l in corpus.open(encoding="utf-8")])

    # GATE — the negative window really is "before the first clip", per video.
    c = items[items.clip_state.isin(["present", "absent"])]
    for v, g in c.groupby("video"):
        a, p = g[g.clip_state == "absent"].frame, g[g.clip_state == "present"].frame
        if len(a) and len(p) and a.max() >= p.min():
            raise AssertionError(f"{v}: a negative frame at {a.max()} is not before the "
                                 f"first positive at {p.min()} — the window is not clean")

    cuts, bands = [], []
    for m in MODELS:
        pred = pd.read_csv(runs / m / "predictions_v2_full.csv")[["qID", "prediction"]]
        d = items.merge(pred, on="qID")
        neg = d[d.clip_state == "absent"].copy()
        neg["fp"] = neg.prediction.fillna("").str.lower().str.contains("clip")
        neg["q"] = neg.groupby("video").frame.rank(pct=True)
        first = d[d.clip_state == "present"].groupby("video").frame.min()
        neg["gap"] = neg.video.map(first) - neg.frame

        row = {"model": m, "platform": PLATFORM[m], "n_neg": len(neg),
               "fp_rate": neg.fp.mean(), "says_bare_Clip": (neg.prediction == "Clip").mean()}
        for cut in (0.2, 0.3, 0.5, 1.0):
            row[f"fp_q<={cut}"] = neg[neg.q <= cut].fp.mean()
        for lo, hi, lbl in ((300, 10 ** 9, "gap>300"), (100, 300, "gap100_300"), (0, 100, "gap<100")):
            sub = neg[(neg.gap > lo) & (neg.gap <= hi)]
            row[lbl] = sub.fp.mean() if len(sub) else None
        cuts.append(row)

        b = neg.assign(decile=(neg.q * 10).round() / 10).groupby("decile").fp.agg(["size", "mean"])
        bands.append(b.assign(model=m).reset_index())

    return pd.DataFrame(cuts).round(4), pd.concat(bands, ignore_index=True).round(4)


if __name__ == "__main__":
    import sys
    runs, corpus, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cuts, bands = anatomy(runs, corpus)
    cuts.to_csv(out / "RESULTS_clip_fp_anatomy.csv", index=False)
    bands.to_csv(out / "RESULTS_clip_fp_ramp.csv", index=False)
    print(cuts.to_string(index=False))
    print("\nramp (r42):")
    print(bands[bands.model == "r42"][["decile", "size", "mean"]].to_string(index=False))
