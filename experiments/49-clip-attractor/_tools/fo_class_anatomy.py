"""Rung 49a — the `fo_class` error anatomy, as a COMMITTED artifact.

The campaign has quoted *"78.3 % of `fo_class` errors involve Clip or Sponge; fixing that
confusion is +0.0702 headline"* since 2026-08-10. Rung 36's README says plainly that the
figure **has no artifact**: it exists as prose in `context/NOW.md` and the `inspect.csv` it
came from is gitignored. It is the number that justifies funding this whole front, so it is
the one number that must be re-derivable. This rebuilds it from the per-question evals we
still hold, for every arm at once.

🔴 Scoring goes through `frame.metrics` and nothing else (RULES §EVAL). The headroom is
computed by flipping `correct` to True on exactly the Clip/Sponge error rows and re-running
`stratified_report` — never by arithmetic on a bucket mean in prose. Set parsing uses
`metrics.read_fo_class`, the byte-for-byte reimplementation of the SDK's own `FOClass.read`,
so an SDK-illegal answer is COUNTED rather than silently treated as an empty set.

🔴 It also asks, on OUR data, the question rung 48 answered on CholecT50: is the spurious
`Clip` a flat prior or a RAMP that climbs toward the moment a clip really appears? Our
corpus has no phase labels, but it has timestamps and golds — so per video the first
Clip-positive timestamp defines the same "cannot exist yet" window, and the same ramp is
measurable. If it ramps here too, the mechanism transfers. If our training corpus ALREADY
contains those negative rows and the model still ramps, then "it never saw the counterexample"
is false for our domain and more negatives cannot be the fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CLIP, SPONGE = "Clip", "Sponge"


def _fo(metrics):
    """The accepted class names, taken from the SDK — never hard-coded (RULES §8b).

    Mirrors `metrics.py:1337` line for line: `FOType` is a descriptor class with a
    `names()` classmethod, not an enum, so it is not iterable. `"none"` is deliberately
    absent — `read_fo_class` handles it before this map is consulted.
    """
    return {n.lower(): n for n in tuple(metrics._load_fotype().names())}


def _sets(df: pd.DataFrame, metrics, valid) -> pd.DataFrame:
    d = df.copy()
    d["gold_set"] = d.ground_truth.map(lambda x: metrics.read_fo_class(x, valid))
    d["pred_set"] = d.our_answer.map(lambda x: metrics.read_fo_class(x, valid))
    return d


def _has(s, c):
    return bool(s) and c in s


def taxonomy(d: pd.DataFrame) -> dict:
    """One arm's fo_class errors, split the way a fix would have to attack them."""
    err = d[~d.correct.astype(bool)]
    n_err = len(err)
    if not n_err:
        return {"n": len(d), "acc": 1.0, "n_err": 0}

    involves = err.apply(
        lambda r: _has(r.gold_set, CLIP) or _has(r.pred_set, CLIP)
        or _has(r.gold_set, SPONGE) or _has(r.pred_set, SPONGE), axis=1)
    strict_cs = err.apply(
        lambda r: r.gold_set == frozenset({CLIP}) and r.pred_set == frozenset({SPONGE}), axis=1)
    strict_sc = err.apply(
        lambda r: r.gold_set == frozenset({SPONGE}) and r.pred_set == frozenset({CLIP}), axis=1)
    missed = err.apply(lambda r: bool(r.pred_set) and bool(r.gold_set)
                       and r.pred_set < r.gold_set, axis=1)
    extra = err.apply(lambda r: bool(r.pred_set) and bool(r.gold_set)
                      and r.pred_set > r.gold_set, axis=1)
    return {
        "n": len(d), "acc": round(float(d.correct.astype(bool).mean()), 4),
        "n_err": n_err,
        "err_involving_clip_or_sponge": int(involves.sum()),
        "share_clip_or_sponge": round(float(involves.mean()), 4),
        "strict_Clip_to_Sponge": int(strict_cs.sum()),
        "strict_Sponge_to_Clip": int(strict_sc.sum()),
        "pred_illegal": int(err.pred_set.isna().sum()),
        "missed_a_class": int(missed.sum()), "added_a_class": int(extra.sum()),
    }


def clip_fp_ramp(d: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Our own version of rung 48's ramp: is the spurious Clip flat, or phase-driven?

    Per video the first timestamp whose GOLD contains Clip opens the window in which a
    placed clip provably does not exist yet. Rows before it whose gold lacks Clip are the
    negatives; `fp` is the model naming Clip anyway.
    """
    d = d[d.gold_set.notna()].copy()
    d["gold_has_clip"] = d.gold_set.map(lambda s: CLIP in s)
    d["pred_has_clip"] = d.pred_set.map(lambda s: bool(s) and CLIP in s)
    first = d[d.gold_has_clip].groupby("video").timestamp_s.min().rename("first_clip_ts")
    d = d.join(first, on="video")
    neg = d[(~d.gold_has_clip) & d.first_clip_ts.notna()
            & (d.timestamp_s < d.first_clip_ts)].copy()
    if neg.empty:
        return pd.DataFrame(), {"n_neg": 0}
    neg["q"] = neg.groupby("video").timestamp_s.rank(pct=True)
    neg["decile"] = neg.q.map(lambda x: min(int(x * n_bins) + 1, n_bins)) / n_bins
    bands = (neg.groupby("decile").pred_has_clip.agg(["size", "mean"])
             .rename(columns={"mean": "fp_rate"}).reset_index().round(4))
    summary = {
        "n_neg": int(len(neg)), "n_videos": int(neg.video.nunique()),
        "fp_rate": round(float(neg.pred_has_clip.mean()), 4),
        "fp_first_decile": round(float(bands.iloc[0].fp_rate), 4),
        "fp_last_decile": round(float(bands.iloc[-1].fp_rate), 4),
        "recall_when_present": round(float(d[d.gold_has_clip].pred_has_clip.mean()), 4),
    }
    return bands, summary


def headroom(results_df: pd.DataFrame, d: pd.DataFrame, metrics, gold) -> dict:
    """`bucket_mean` if the Clip/Sponge errors were correct — through frame.metrics.

    TWO bounds, because the campaign has only ever quoted the loose one and it is close
    to vacuous: with ten classes of which Clip and Sponge are by far the commonest, ~90 %
    of every `fo_class` error *involves* one of them, so "fix that confusion" and "fix
    `fo_class`" are nearly the same sentence. `loose` is therefore a CEILING, not a lever.

    `strict` fixes only the exact substitutions `{Clip}->{Sponge}` and `{Sponge}->{Clip}`
    — the confusion as a confusion, with no other error mixed in. That is the number a
    Clip-vs-Sponge intervention could actually claim.
    """
    base = metrics.stratified_report(results_df, gold=gold)
    err = d[~d.correct.astype(bool)]

    def _inv(r):
        s = (r.gold_set or frozenset()) | (r.pred_set or frozenset())
        return CLIP in s or SPONGE in s

    def _strict(r):
        return {r.gold_set, r.pred_set} == {frozenset({CLIP}), frozenset({SPONGE})}

    def _mid(r):
        """The ONLY thing wrong is the Clip/Sponge membership — nothing else differs.

        `loose` is a ceiling and `strict` only admits singleton swaps, which discards
        every row where the model got a third class right and still missed the Sponge.
        This is the bound an intervention on the Clip-vs-Sponge decision could claim:
        the symmetric difference is a non-empty subset of {Clip, Sponge}.
        """
        if r.pred_set is None or r.gold_set is None:
            return False
        sym = r.gold_set ^ r.pred_set
        return bool(sym) and sym <= {CLIP, SPONGE}

    def _declip(r):
        """A SPURIOUS Clip is the whole error: delete it and the answer is exactly right.

        This is the attractor stated as a lever, and it is not a Clip-vs-Sponge quantity —
        `Clip` is the sink for Needle, Specimen and Gallstone too, so framing the defect as
        a PAIR confusion undercounts it. `mid` cannot see Needle -> Clip; this can.
        """
        if r.pred_set is None or r.gold_set is None:
            return False
        return CLIP in r.pred_set and CLIP not in r.gold_set and (
            r.pred_set - {CLIP} == r.gold_set)

    def _addclip(r):
        """The mirror: a MISSED real clip is the whole error. The attractor's other side."""
        if r.pred_set is None or r.gold_set is None:
            return False
        return CLIP in r.gold_set and CLIP not in r.pred_set and (
            r.pred_set | {CLIP} == r.gold_set)

    def _clip_fp_any(r):
        """EVERY error where the model named Clip and the gold has none.

        The attractor's total mass. `declip` was too narrow — it only admits a Clip added
        ON TOP of an otherwise-right answer, so it misses the substitutions, which are the
        bigger half: `Sponge -> Clip` deletes nothing, it REPLACES. Fixing this class needs
        the model to both drop the Clip and name what was really there, so this is an upper
        bound on the attractor, not a one-token repair.
        """
        if r.pred_set is None or r.gold_set is None:
            return False
        return CLIP in r.pred_set and CLIP not in r.gold_set

    out = {
        "bucket_mean_before": round(float(base["bucket_mean"]), 4),
        "acc_ID_before": round(float(base["acc_ID"]), 4),
        "acc_OOD_before": round(float(base["acc_OOD"]), 4),
    }
    for tag, pred in (("loose", _inv), ("mid", _mid), ("strict", _strict),
                      ("declip", _declip), ("addclip", _addclip),
                      ("clip_fp_any", _clip_fp_any)):
        ids = set(err[err.apply(pred, axis=1)].qID) if len(err) else set()
        fixed = results_df.copy()
        fixed.loc[fixed.qID.isin(ids), "correct"] = True
        after = metrics.stratified_report(fixed, gold=gold)
        out[f"n_fixed_{tag}"] = len(ids)
        out[f"bucket_mean_after_{tag}"] = round(float(after["bucket_mean"]), 4)
        out[f"headroom_{tag}"] = round(
            float(after["bucket_mean"] - base["bucket_mean"]), 4)
    return out


def run(arms: dict, out: Path, metrics, gold=None) -> None:
    valid = _fo(metrics)
    tax, ramps, heads, conf = [], [], {}, []
    for name, insp in arms.items():
        raw = pd.read_csv(insp)
        raw["correct"] = raw.correct.astype(str).str.lower().isin(("true", "1", "yes"))
        fo = _sets(raw[raw.answer_format == "fo_class"], metrics, valid)
        tax.append({"arm": name, "slice": "ALL", **taxonomy(fo)})
        for dist, sub in (("ID", fo[fo.dataset == "lapchole"]),
                          ("OOD", fo[fo.dataset == "heico"])):
            if len(sub):
                tax.append({"arm": name, "slice": dist, **taxonomy(sub)})
        bands, summ = clip_fp_ramp(fo)
        if len(bands):
            ramps.append(bands.assign(arm=name))
        heads[name] = {"clip_fp": summ}
        s = fo[fo.gold_set.map(lambda x: bool(x) and len(x) == 1)]
        for g, grp in s.groupby(s.gold_set.map(lambda x: next(iter(x)))):
            counts = grp.pred_set.map(
                lambda x: ",".join(sorted(x)) if x else "ILLEGAL").value_counts()
            for p, n in counts.items():
                conf.append({"arm": name, "gold": g, "pred": p, "n": int(n)})
        if gold is not None:
            cols = ["qID", "answer_format", "primary", "correct"]
            rdf = raw.rename(columns={"primary_capability": "primary"})
            # `video` and `dataset` feed the video-clustered bootstrap (`_video_key`).
            # Dropping them turned the headroom into a KeyError, not a wrong number.
            cols += [c for c in ("video", "dataset") if c in rdf.columns]
            rdf = rdf[cols]
            try:
                heads[name]["headroom"] = headroom(rdf, fo, metrics, gold)
            except Exception as exc:  # noqa: BLE001
                heads[name]["headroom"] = {"error": f"{type(exc).__name__}: {exc}"}

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tax).to_csv(out / "RESULTS_fo_class_anatomy.csv", index=False)
    if ramps:
        pd.concat(ramps, ignore_index=True).to_csv(
            out / "RESULTS_clip_fp_ramp_ours.csv", index=False)
    if conf:
        pd.DataFrame(conf).sort_values(
            ["arm", "gold", "n"], ascending=[True, True, False]).to_csv(
            out / "RESULTS_fo_class_confusion.csv", index=False)
    (out / "RESULTS_fo_class_headroom.json").write_text(json.dumps(heads, indent=1))
    print(pd.DataFrame(tax).to_string(index=False))
    print()
    print(json.dumps(heads, indent=1))


if __name__ == "__main__":
    sys.exit("import and call run(); this module is a library, not a launcher")
