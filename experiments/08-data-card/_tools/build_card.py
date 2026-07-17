"""Computes every table and every headline number in README.md — the data card.

Library, not a launcher: the notebook imports `build()` and calls it. It lives here
because the card's claims must be *asserted*, and asserts belong next to the code
that produces the number, not in prose a reader has to trust.

The asserts are the point. Every figure quoted in README.md is checked here; if the
data or the rung-02 artifacts change, this raises instead of letting the card lie.
A failing assert is a FINDING — fix the card or the pipeline, never the expected
value.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import pandas as pd

from focus.data.data_models import Capability
from focus.evaluation.evaluator import Evaluator

# `answer_format == "open_ended"` embeds the timestamp in the question text, so the
# raw string is unique per question and useless as a template key. Normalise it out.
_TS = re.compile(r"\d{2}:\d{2}:\d{2}")


def _template(question: str) -> str:
    return _TS.sub("<TS>", question)


def _load(data_root: Path, split: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(f).assign(ds=Path(f).parents[2].name, split=split)
        for f in sorted(glob.glob(str(data_root / "*" / "data" / "frame" / f"{split}.parquet")))
    ]
    if not frames:
        raise FileNotFoundError(
            f"No {split}.parquet under {data_root}. The QA parquets are 410 KB total and are "
            "gitignored; pull them from the pod — see README.md §Reproducing."
        )
    return pd.concat(frames, ignore_index=True)


def _pre_eval(df: pd.DataFrame) -> float:
    """The SDK's own pre-evaluation score. Not a reimplementation."""
    ev = Evaluator.__new__(Evaluator)  # the method is pure; __init__ wants a judge
    return Evaluator.pre_evaluation_score(ev, df)[0]


def _floor_per_template(df: pd.DataFrame) -> float:
    """Score of answering each TEMPLATE's most common answer, without looking.

    Not the same as the global majority: a template-aware constant is a strictly
    smarter baseline, and measuring against the dumber one inflates our margin.
    """
    hits = sum(g["answer"].value_counts().iloc[0] for _, g in df.groupby("template"))
    return hits / len(df)


def build(data_root: Path, eval_dir: Path, out_dir: Path) -> dict:
    """Compute every table, assert every headline, return the headlines."""
    logging.disable(logging.WARNING)  # the SDK's 3/10-buckets warning is quoted, not silenced
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load(data_root, "train")
    val = _load(data_root, "test")
    for df in (train, val):
        df["qID"] = df["ds"] + "__" + df["id"].astype(str)
        df["template"] = df["question"].map(_template)
        df["group"] = df["primary_capability"].map(lambda c: Capability.from_any(c).group.value)
        df["distribution"] = df["qID"].str.startswith("heico").map({True: "OOD", False: "ID"})
        df["frame"] = df["ds"] + "__" + df["video"].astype(str) + "__" + df["timestamp_start"].astype(str)

    results = pd.read_csv(eval_dir / "results.csv")
    val = val.merge(results[["qID", "correctness"]], on="qID")

    h: dict = {}

    # ---- units. Questions are not independent: they cluster in frames and videos.
    h["n_val"], h["n_train"] = len(val), len(train)
    h["n_frames"], h["n_videos"] = val["frame"].nunique(), val["video"].nunique()
    pd.DataFrame(
        [
            {"unit": "questions", "val": len(val), "train": len(train)},
            {"unit": "frames", "val": val["frame"].nunique(), "train": train["frame"].nunique()},
            {"unit": "videos", "val": val["video"].nunique(), "train": train["video"].nunique()},
        ]
    ).to_csv(out_dir / "units.csv", index=False)

    # ---- schema. All 14 columns, including the three nobody had ever looked at.
    rows = []
    for c in [c for c in train.columns if c not in ("ds", "split", "qID", "template", "group", "distribution", "frame")]:
        try:
            n = train[c].nunique()
        except TypeError:  # secondary_capabilities holds arrays
            n = train[c].map(lambda v: tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v).nunique()
        rows.append({"column": c, "dtype": str(train[c].dtype), "n_distinct": n, "example": str(train[c].iloc[0])[:40]})
    pd.DataFrame(rows).to_csv(out_dir / "schema.csv", index=False)

    # ---- the capability tree, read from the SDK. Never hand-roll this: rungs 05 and
    # 07 did, dropped `instance_matching`, and misfiled it into temporal_grounding.
    pd.DataFrame(
        [
            {"code": c.code, "leaf": c.value, "group": c.group.value if c.group else None}
            for c in Capability
            if c.is_leaf
        ]
    ).sort_values("code").to_csv(out_dir / "capabilities.csv", index=False)

    # ---- templates. The unit of analysis, and the reason `acc_number` is a mixture.
    def catalogue(df: pd.DataFrame, scored: bool) -> pd.DataFrame:
        rows = []
        for (fmt, tpl), g in df.groupby(["answer_format", "template"]):
            top = g["answer"].value_counts()
            r = {
                "answer_format": fmt,
                "n": len(g),
                "n_distinct_answers": g["answer"].nunique(),
                "trivial_floor": top.iloc[0] / len(g),
                "modal_answer": top.index[0][:40],
                "template": tpl[:90],
            }
            if scored:
                r["accuracy"] = g["correctness"].mean()
                r["margin_over_floor"] = r["accuracy"] - r["trivial_floor"]
            rows.append(r)
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    tpl_val, tpl_train = catalogue(val, True), catalogue(train, False)
    tpl_val.to_csv(out_dir / "templates_val.csv", index=False)
    tpl_train.to_csv(out_dir / "templates_train.csv", index=False)

    # The same catalogue split by distribution. Pooling ID and OOD hides that they are
    # different populations: some templates exist only in OOD, and the same template
    # can carry a very different floor on each side. Pooling here would be the exact
    # mistake this card accuses `acc_number` of.
    by_dist = pd.concat(
        [catalogue(val[val["distribution"] == d], True).assign(distribution=d) for d in ("ID", "OOD")],
        ignore_index=True,
    )
    by_dist.to_csv(out_dir / "templates_val_by_distribution.csv", index=False)
    h["n_templates_val"] = len(tpl_val)
    h["n_templates_number"] = int((tpl_val["answer_format"] == "number").sum())
    degenerate = tpl_val[tpl_val["n_distinct_answers"] == 1]
    h["n_degenerate_templates"], h["n_degenerate_questions"] = len(degenerate), int(degenerate["n"].sum())
    below = tpl_val[tpl_val["margin_over_floor"] <= 0]
    h["n_questions_at_or_below_floor"] = int(below["n"].sum())

    # ---- `number`: the mixture, and why its headline margin is Simpson's paradox.
    num = val[val["answer_format"] == "number"]
    h["acc_number"] = num["correctness"].mean()
    h["floor_number_global"] = num["answer"].value_counts().iloc[0] / len(num)
    h["floor_number_per_template"] = _floor_per_template(num)
    h["margin_number_reported"] = h["acc_number"] - h["floor_number_global"]
    h["margin_number_honest"] = h["acc_number"] - h["floor_number_per_template"]

    # ---- ID vs OOD, against a template-aware floor. This is what reverses rung 02's
    # "acc_OOD > acc_ID, no OOD collapse": the OOD slice has a HIGHER trivial floor
    # (its answers are more concentrated), so raw accuracy flatters it.
    h["n_ood"], h["n_id"] = int((val["distribution"] == "OOD").sum()), int((val["distribution"] == "ID").sum())
    for d in ("ID", "OOD"):
        s = val[val["distribution"] == d]
        h[f"acc_{d.lower()}"] = s["correctness"].mean()
        h[f"floor_{d.lower()}"] = _floor_per_template(s)
        h[f"margin_{d.lower()}"] = h[f"acc_{d.lower()}"] - h[f"floor_{d.lower()}"]
    pd.DataFrame(
        [
            {
                "distribution": d,
                "n": h[f"n_{d.lower()}"],
                "accuracy": h[f"acc_{d.lower()}"],
                "trivial_floor_per_template": h[f"floor_{d.lower()}"],
                "margin": h[f"margin_{d.lower()}"],
            }
            for d in ("ID", "OOD")
        ]
    ).to_csv(out_dir / "id_vs_ood.csv", index=False)

    # ---- the ood dossier. Four options, one number each, one run (rung 02 eval_best).
    stamped = results.copy()
    stamped["ood"] = stamped["qID"].str.startswith("heico")
    no_orphan = stamped[stamped["primary"] != "temporal_localization"]
    tmp = no_orphan.copy()
    tmp["group"] = tmp["primary"].map(lambda x: Capability.from_any(x).group.value)

    h["ood_1_untouched"] = _pre_eval(results)
    h["ood_2_stamp_only"] = _pre_eval(stamped)
    h["ood_3_stamp_and_drop"] = _pre_eval(no_orphan)
    h["ood_4_bucket_mean"] = tmp.groupby(["group", "ood"])["correctness"].mean().mean()
    pd.DataFrame(
        [
            {"option": "1 — leave it alone", "pre_eval": h["ood_1_untouched"], "buckets": 3, "note": "the orphan is worth 1/3"},
            {"option": "2 — stamp ood only", "pre_eval": h["ood_2_stamp_only"], "buckets": 5, "note": "TRAP: looks fixed, orphan still worth 1/5"},
            {"option": "3 — stamp + drop orphan", "pre_eval": h["ood_3_stamp_and_drop"], "buckets": 4, "note": "identical to option 4"},
            {"option": "4 — bucket_mean (in use)", "pre_eval": h["ood_4_bucket_mean"], "buckets": 4, "note": "derived from the qID prefix; source untouched"},
        ]
    ).to_csv(out_dir / "ood_dossier.csv", index=False)

    # ---- two estimators. Flat is right for the leaderboard, clustered for "does it
    # generalise". We have been quoting flat for both.
    summary = pd.read_csv(eval_dir / "summary.csv")
    rows = []
    for fmt in val["answer_format"].unique():
        s = summary[(summary["level"] == "answer_format") & (summary["name"] == fmt)]
        if s.empty:
            continue
        rows.append(
            {
                "row": fmt,
                "sdk_hierarchical": s.iloc[0]["accuracy"],
                "ci_low": s.iloc[0]["ci_low"],
                "ci_high": s.iloc[0]["ci_high"],
                "flat": val[val["answer_format"] == fmt]["correctness"].mean(),
            }
        )
    est = pd.DataFrame(rows)
    est["delta"] = est["sdk_hierarchical"] - est["flat"]
    est.to_csv(out_dir / "estimators.csv", index=False)
    n_row = summary[(summary["level"] == "answer_format") & (summary["name"] == "number")].iloc[0]
    h["number_clustered"] = n_row["accuracy"]
    h["number_ci"] = (n_row["ci_low"], n_row["ci_high"])

    # ---- generation and procedure_type: two free stratifiers nobody had looked at.
    val.groupby(["generation", "answer_format"]).agg(n=("qID", "size"), accuracy=("correctness", "mean")).reset_index().to_csv(
        out_dir / "generation.csv", index=False
    )
    both = pd.concat([train, val], ignore_index=True)  # ignore_index: crosstab rejects duplicate labels
    pd.crosstab([both["ds"], both["procedure_type"]], both["split"]).to_csv(out_dir / "procedure_type.csv")

    # ---- the unseen-phrasing axis: templates in val that train never showed.
    unseen = set(val["template"]) - set(train["template"])
    h["n_unseen_templates"] = len(unseen)
    h["n_unseen_questions"] = int(val["template"].isin(unseen).sum())

    _assert_headlines(h)
    return h


def _assert_headlines(h: dict) -> None:
    """Every number quoted in README.md. A failure here is a FINDING.

    Never move an expected value to make this pass — that happened on 2026-07-16 and
    an assert was moved from 0.5503 to 0.5432, papering over a real bug.
    """
    # Counted on the NORMALISED template (`<TS>`), never on the raw question string.
    # On the raw string these read 393 / 330 / 555 / 1158 — inflated, because every
    # open_ended question embeds its own timestamp, so each became a "template" of
    # n=1, and a template of one question is degenerate by construction. The raw
    # count measures the timestamp artifact, not the data.
    exact = {
        "n_val": 6252, "n_train": 13748, "n_frames": 4486, "n_videos": 38,
        "n_templates_val": 188, "n_templates_number": 8, "n_degenerate_templates": 126,
        "n_degenerate_questions": 351, "n_questions_at_or_below_floor": 952,
        "n_ood": 4000, "n_id": 2252,
    }
    for k, want in exact.items():
        assert h[k] == want, f"CARD LIES: {k} = {h[k]}, README says {want}"

    close = {
        "acc_number": 0.4331, "floor_number_global": 0.3520,
        "floor_number_per_template": 0.3840, "margin_number_honest": 0.0492,
        "ood_1_untouched": 0.7079, "ood_2_stamp_only": 0.6389,
        "ood_3_stamp_and_drop": 0.5486, "number_clustered": 0.3803,
    }
    for k, want in close.items():
        assert abs(h[k] - want) <= 5e-4, f"CARD LIES: {k} = {h[k]:.6f}, README says {want}"

    # rung 02 shipped "acc_OOD 0.5918 > acc_ID 0.5209 → no OOD collapse". Both halves
    # are true and the conclusion does not follow: OOD's floor is 12 points higher, so
    # the model adds LESS there. If this ever stops holding, the card's §4b is wrong.
    assert h["floor_ood"] > h["floor_id"], "CARD LIES: the OOD floor is supposed to be the higher one"
    assert h["acc_ood"] > h["acc_id"], "CARD LIES: raw acc_OOD is supposed to beat acc_ID"
    assert h["margin_ood"] < h["margin_id"], "CARD LIES: the OOD margin is supposed to be the smaller one"

    # The dossier's load-bearing claim: bucket_mean IS the stamped version.
    assert abs(h["ood_3_stamp_and_drop"] - h["ood_4_bucket_mean"]) < 1e-12, (
        "CARD LIES: option 3 and option 4 are supposed to be bit-identical"
    )
    # `number`, on the SDK's own estimator, is inside the noise of a constant.
    assert h["number_ci"][0] <= h["floor_number_per_template"] <= h["number_ci"][1], (
        "CARD LIES: the number CI no longer contains the trivial floor"
    )
