"""Score rung 46's three arms through the canonical path. Library — the notebook calls it.

RULES §1: the score comes from `frame.metrics.stratified_report` over the SDK Evaluator's own
`results_df`. Nothing is re-derived inline. The judge is loaded ONCE and reused across the
three arms, which is also what makes them comparable: the same judge, the same seed, the same
process — `archived-results-not-bit-reproducible` bites when those differ.

RULES §13: the paired CI is clustered by VIDEO, not by question. Effective n here is 8 videos
(6 lapchole + 2 heico), so the ID cell has 6 clusters and no OOD CI is readable on 2. That is
declared in PLAN.md, not discovered here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARMS = ("A_alone", "A_selfrevise", "A_debate")


def _score_one(arm: str, answers: list[str], requests, references, judge, out: Path) -> pd.DataFrame:
    from focus.data.data_models import Response
    from focus.evaluation.evaluator import Evaluator

    responses = [Response(qID=r.qID, content=a, latency=0.0)
                 for r, a in zip(requests, answers)]
    ev = Evaluator(judges=[judge], seed=46)
    # track=None on purpose: latency is meaningless for an offline batched re-score, and
    # `Track.FRAME` would mark every answer incorrect on a clock that was never running.
    results_df, _ = ev.run(requests=requests, references=references, responses=responses,
                           output_dir=out / arm, track=None)
    results_df["arm"] = arm
    return results_df


def paired_ci_by_video(res_a: pd.DataFrame, res_b: pd.DataFrame, videos: dict,
                       mask=None, n_boot: int = 2000, seed: int = 46) -> dict:
    """Δ = arm − control, bootstrapped over VIDEOS (the cluster), not questions.

    Resampling questions would treat 1,283 correlated observations as independent and shrink
    the interval to a width the data cannot support (`RULES §13`).
    """
    a = res_a.set_index("qID")["correctness"].astype(float)
    b = res_b.set_index("qID")["correctness"].astype(float)
    q = a.index.intersection(b.index)
    if mask is not None:
        q = q[[mask.get(x, False) for x in q]]
    if len(q) == 0:
        return {"delta": float("nan"), "ci": [float("nan")] * 2, "n_q": 0, "n_videos": 0}
    d = (a[q] - b[q])
    vid = pd.Series([videos[x] for x in q], index=q)
    per_video = d.groupby(vid).mean()
    rng = np.random.default_rng(seed)
    vids = per_video.index.to_numpy()
    boots = [per_video[rng.choice(vids, len(vids), replace=True)].mean() for _ in range(n_boot)]
    return {"delta": float(d.mean()),
            "delta_video_mean": float(per_video.mean()),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_q": int(len(q)), "n_videos": int(per_video.size)}


def score_run(run_dir: str, data_root: str, held_out: list[str], judge_model: str,
              device: str = "cuda") -> dict:
    from focus.evaluation.judges import TransformersJudge
    from frame.metrics import stratified_report, class_f1_report
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from debate import DebateConfig, load_population

    run = Path(run_dir)
    arms = json.loads((run / "arms.json").read_text())
    cfg = DebateConfig(model_a="", model_b="", out_dir=str(run), frames_cache="",
                       data_root=data_root, held_out_videos=held_out)
    items, _ = load_population(cfg)
    assert [i.request.qID for i in items] == arms["qID"], "population drifted since the run"
    requests = [i.request for i in items]
    references = [i.reference for i in items]
    for ref in references:                       # mirrors run.py: the SDK's `ood` is all-False
        ref.ood = ref.qID.split("__", 1)[0] == "heico"

    judge = TransformersJudge(model_name=judge_model, device=device)
    per_arm, reports = {}, {}
    for arm in ARMS:
        df = _score_one(arm, arms[arm], requests, references, judge, run / "scored")
        per_arm[arm] = df
        reports[arm] = stratified_report(df)
        try:
            reports[arm]["class_f1"] = class_f1_report(df)      # RULES §9b — fo_class is scored
        except Exception as e:                                   # never silently omit it
            reports[arm]["class_f1_error"] = repr(e)
        logger.info("%s bucket_mean=%s", arm, reports[arm].get("bucket_mean"))
    del judge

    videos = dict(zip(arms["qID"], arms["video"]))
    is_id = {q: not q.startswith("heico__") for q in arms["qID"]}
    is_ood = {q: q.startswith("heico__") for q in arms["qID"]}
    dis = dict(zip(arms["qID"], arms["B_disagrees"]))

    ci = {}
    for label, (arm, ctrl) in {"debate_vs_selfrevise": ("A_debate", "A_selfrevise"),
                               "debate_vs_alone": ("A_debate", "A_alone"),
                               "selfrevise_vs_alone": ("A_selfrevise", "A_alone")}.items():
        ci[label] = {
            "ALL_ID": paired_ci_by_video(per_arm[arm], per_arm[ctrl], videos, is_id),
            "ALL_OOD": paired_ci_by_video(per_arm[arm], per_arm[ctrl], videos, is_ood),
            "ALL": paired_ci_by_video(per_arm[arm], per_arm[ctrl], videos),
            "disagreements_only": paired_ci_by_video(per_arm[arm], per_arm[ctrl], videos, dis),
        }

    # the oracle bound every protocol is measured against (see PLAN.md)
    corr = {a: per_arm[a].set_index("qID")["correctness"].astype(bool) for a in ARMS}
    b_df = _score_one("B_alone", arms["B_answer"], requests, references,
                      TransformersJudge(model_name=judge_model, device=device), run / "scored")
    b_ok = b_df.set_index("qID")["correctness"].astype(bool)
    a_ok = corr["A_alone"]
    q = a_ok.index
    union = {"A_alone": float(a_ok.mean()), "B_alone": float(b_ok[q].mean()),
             "union_ceiling": float((a_ok | b_ok[q]).mean()),
             "both_right": float((a_ok & b_ok[q]).mean()),
             "only_A": float((a_ok & ~b_ok[q]).mean()),
             "only_B": float((~a_ok & b_ok[q]).mean()),
             "neither": float((~a_ok & ~b_ok[q]).mean())}
    union["oracle_headroom"] = union["union_ceiling"] - max(union["A_alone"], union["B_alone"])

    out = {"reports": reports, "paired_ci": ci, "union": union,
           "n": len(arms["qID"]), "n_videos": len(set(arms["video"]))}
    (run / "RESULTS_scored.json").write_text(json.dumps(out, indent=2, default=str))
    return out
