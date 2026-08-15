"""Build the SEGMENT training corpus for ms-swift, with its blocking gates.

Every constant here is MEASURED, not assumed — see
`experiments_segment/01-viability/RESULTS_probe.md` for the probe that produced them:

  * a visual token is 32x32 px (patch16 x merge2), so a 960x540 cache frame costs 480
  * `temporal_patch_size = 2`, so K frames become ceil(K/2) temporal positions
  * `VIDEO_MAX_TOKEN_NUM` defaults to 768 > 480, so the cap never binds and we pass none
  * ms-swift does NOT validate the `<video>` tag count — it silently prepends a missing
    tag and only warns on an excess one, so the gates below are the only protection

The design decisions this implements are argued in `experiments_segment/01-viability/`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ── measured constants ───────────────────────────────────────────────
TOK_PER_FRAME_960x540 = 480
TEMPORAL_PATCH = 2
BASE_FPS = {"heico": 25, "lapchole": 30}

# ── the routers, both pinned to LITERAL strings ──────────────────────
# NEVER a timestamp regex: that reading scores recall 0.139 with 2,346 false
# positives (e.g. "What types of foreign objects are seen between 00:45:40 and
# 00:46:41?" is fo_class, not time).
_TIME_MARKERS = ("hh:mm:ss",)
_PCT_MARKERS = ("In %", "format xx%")

# `2b` duration_estimation vs `2a` temporal_localization. Both are answer_format
# "time"; only `2a` carries an absolute procedure clock. Separates 261/261 from
# 0/7,384 on the public corpus.
_DURATION_MARKERS = ("for how long", "how much time passes")


def needs_time_grid(question: str) -> bool:
    """True if the question asks for a timestamp or a percentage-of-time."""
    return any(m in question for m in _TIME_MARKERS) or any(m in question for m in _PCT_MARKERS)


def is_duration_question(question: str) -> bool:
    """True for `2b` duration_estimation — an ELAPSED SPAN, never offset."""
    q = question.lower()
    return any(m in q for m in _DURATION_MARKERS)


def threshold_seconds(duration: float) -> float:
    """The SDK's per-question acceptance window.

    Mirrors `FocusDataset._parse_row` (`vendor/orena-focus/.../base_dataset.py`).
    `assert_threshold_matches_sdk` gates the two copies against drift.
    """
    return min(5.0, 1 + duration * (4 / 360))


def frames_for(duration: float, routed: bool) -> int:
    """K frames for a clip.

    Routed rows get the LOCALISATION grid (spacing <= threshold), not merely the
    coverage grid (spacing <= 2*threshold): at coverage density the event is
    bracketed in an interval of width 2*threshold and a perfect detector still
    only scores ~0.51, because it must name the midpoint with no error budget.
    The temporal patch halves the token cost, which is what makes this affordable.
    """
    if not routed:
        return 4
    return int(math.ceil(duration / threshold_seconds(duration))) + 1


def visual_tokens(k: int, tok_per_frame: int = TOK_PER_FRAME_960x540) -> int:
    return math.ceil(k / TEMPORAL_PATCH) * tok_per_frame


def cache_name(dataset: str, video_id: str, frame_index: int) -> str:
    """The shared store's identity key — MUST match `frame.data.frame_cache_name`."""
    vid = re.sub(r"[^A-Za-z0-9]+", "_", str(video_id)).strip("_")
    return f"{dataset}__{vid}__{frame_index}.jpg"


def ts_to_seconds(ts: str) -> float:
    h, m, s = str(ts).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_ts(t: float) -> str:
    """hh:mm:ss, zero-padded, clamped at 0, integer-floored.

    `Time.verify` demands `\\d{2}:\\d{2}:\\d{2}` per comma-separated part with
    0 <= m,s < 60 and RAISES otherwise — and `Evaluator` catches that raise and
    scores the answer INCORRECT behind a `logger.debug`. A malformed timestamp is
    therefore a SILENT zero, which is why every emission goes through this function.
    """
    t = max(0, int(round(t)))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


SYSTEM_PROMPT_HEAD = (
    "You are an expert surgical assistant. You are shown a SEQUENCE OF FRAMES sampled "
    "uniformly from a short clip of a laparoscopic (minimally invasive) surgical video. "
    "The clip's start and end times within the full procedure are given to you. Answer "
    "the question about foreign objects using ONLY the visual evidence in the frames. "
    "When the question asks for a time, answer with a time OFFSET FROM THE START OF THE "
    "CLIP in hh:mm:ss. Respond with the exact answer in the format the question requests "
    "and NOTHING else — no explanation, no full sentences, no extra punctuation.\n\n"
)


def system_prompt() -> str:
    from focus.foreign_objects import FO_DEFINITIONS_FILE

    return SYSTEM_PROMPT_HEAD + FO_DEFINITIONS_FILE.read_text()


@dataclass
class ExportConfig:
    data_root: Path = Path("/workspace/orena-data")
    cache: Path = Path("/workspace/frames_cache")
    out: Path = Path("/workspace/tmp/segment_train.jsonl")
    sdk_base_dataset: Path = Path(
        "/workspace/repo/vendor/orena-focus/src/focus/data/base_dataset.py")
    splits: tuple[str, ...] = ("train",)
    datasets: tuple[str, ...] = ("heico", "lapchole")
    relative_time: bool = True     # flag OFF => absolute golds, byte-identical passthrough
    drop_percentage: bool = False
    smoke: int = 0                 # 0 = full; >0 = that many rows, STRATIFIED
    report: dict = field(default_factory=dict)


# ── gates: each RAISES. A gate that fires is a finding (RULES §7). ───

def assert_threshold_matches_sdk(sdk_base_dataset: Path | str) -> dict:
    """Our acceptance window must be the SDK's — checked against its SOURCE, not itself.

    Three independent copies of this formula now exist (the SDK, `frame.data`, and this
    module). Comparing our function to an inline restatement of itself is tautological
    and proves nothing; this reads the literal out of `base_dataset.py` and evaluates it.
    If the SDK is ever re-vendored with a different window, this RAISES instead of
    silently making our local score stop being the platform's.
    """
    src = Path(sdk_base_dataset).read_text(encoding="utf-8")
    # the inner formula carries its own parentheses, so stop at the closing brace of
    # the dict literal, not at the first ')'
    m = re.search(r"threshold_seconds\"?\s*:\s*min\(\s*([0-9.]+)\s*,\s*(.+?)\)\s*\}", src)
    if not m:
        raise AssertionError(f"could not find the threshold formula in {sdk_base_dataset}")
    cap, form = float(m.group(1)), m.group(2)
    expr = f"{m.group(1)}, {form}"
    for duration in (1.0, 10.0, 29.0, 119.0, 299.0, 300.0, 3600.0):
        sdk = min(cap, eval(form, {"__builtins__": {}}, {"duration": duration}))  # noqa: S307
        if abs(threshold_seconds(duration) - sdk) > 1e-12:
            raise AssertionError(
                f"threshold drift at dur={duration}: ours={threshold_seconds(duration)} "
                f"sdk={sdk} (formula {expr!r})"
            )
    return {"sdk_threshold_formula": expr.strip()}


def assert_router_pure(df: pd.DataFrame) -> dict:
    """The time/percentage router must be exactly pure, or we sample the wrong rows."""
    routed = df.question.map(needs_time_grid)
    gold = df.answer_format.isin(["time", "percentage"])
    fp, fn = int((routed & ~gold).sum()), int((~routed & gold).sum())
    if fp or fn:
        raise AssertionError(f"time/pct router impure: FP={fp} FN={fn}")
    return {"router_tp": int((routed & gold).sum()), "router_fp": 0, "router_fn": 0}


def assert_duration_router_pure(df: pd.DataFrame) -> dict:
    """`2b` must be separable from `2a` by question text alone.

    At inference the container has no `primary_capability`, so this router is the
    only thing standing between a duration gold and a wrongly-added clip offset.
    """
    t = df[df.answer_format == "time"]
    if t.empty:
        return {"dur_router_n": 0}
    pred = t.question.map(is_duration_question)
    gold = t.primary_capability.astype(str).str.startswith("2b")
    fp, fn = int((pred & ~gold).sum()), int((~pred & gold).sum())
    if fp or fn:
        raise AssertionError(f"2a/2b router impure: FP={fp} FN={fn} of n={len(t)}")
    return {"dur_router_n": int(len(t)), "dur_router_2b": int(gold.sum()),
            "dur_router_fp": 0, "dur_router_fn": 0}


def assert_row_wellformed(rec: dict, k: int) -> None:
    """ms-swift will NOT do this for us — measured: 0 tags encodes like 1, 2 only warns."""
    user = [m for m in rec["messages"] if m["role"] == "user"][0]["content"]
    n_tags = user.count("<video>")
    if n_tags != 1:
        raise AssertionError(f"<video> tag count {n_tags} != 1")
    vids = rec["videos"]
    if len(vids) != 1 or len(vids[0]) != k:
        raise AssertionError(f"videos shape {len(vids)}x{len(vids[0])} != 1x{k}")
    if len(set(vids[0])) != k:
        raise AssertionError("duplicate frame paths in one row")
    if "raw_fps" not in rec.get("chat_template_kwargs", {}):
        raise AssertionError("missing per-row raw_fps")


def assert_time_roundtrip(rows: list[dict]) -> dict:
    """post(target, request) must reproduce the ORIGINAL gold, byte for byte.

    And the checked set must be non-empty: rung 15's smoke exported 16 rows with
    zero `number` rows and its round-trip gate passed on an empty set.
    """
    n = 0
    for r in rows:
        meta = r["_meta"]
        if meta.get("time_kind") != "2a_relative":
            continue
        back = seconds_to_ts(ts_to_seconds(r["messages"][-1]["content"]) + meta["clip_start_s"])
        if back != meta["gold_original"]:
            raise AssertionError(
                f"round-trip failed: {r['messages'][-1]['content']} + {meta['clip_start_s']} "
                f"-> {back} != {meta['gold_original']}"
            )
        n += 1
    if n == 0:
        raise AssertionError("round-trip gate checked ZERO rows — it proves nothing")
    return {"roundtrip_checked": n}


def assert_frames_exist(rows: list[dict], cache: Path) -> dict:
    """Every referenced frame must be on disk. One readdir, not N stat calls."""
    import os

    have = set(os.listdir(cache))
    missing = 0
    for r in rows:
        for p in r["videos"][0]:
            if Path(p).name not in have:
                missing += 1
    if missing:
        raise AssertionError(f"{missing} referenced frames are not in {cache}")
    return {"frames_referenced": sum(len(r["videos"][0]) for r in rows)}


# ── the builder ──────────────────────────────────────────────────────

def load(cfg: ExportConfig) -> pd.DataFrame:
    parts = []
    for ds in cfg.datasets:
        for sp in cfg.splits:
            d = pd.read_parquet(cfg.data_root / ds / "data" / "segment" / f"{sp}.parquet")
            d["g_ds"], d["g_split"] = ds, sp
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["g_st"] = df.timestamp_start.map(ts_to_seconds)
    df["g_en"] = df.timestamp_end.map(ts_to_seconds)
    df["g_dur"] = df.g_en - df.g_st
    df["g_routed"] = df.question.map(needs_time_grid)
    df["g_K"] = [frames_for(d, r) for d, r in zip(df.g_dur, df.g_routed)]
    return df


def frame_paths(cfg: ExportConfig, ds: str, video: str, st: float, dur: float, k: int) -> list[str]:
    bf = BASE_FPS[ds]
    out, seen = [], set()
    for i in range(k):
        t = st + (dur * i / (k - 1) if k > 1 else 0.0)
        idx = round(t * bf)
        while idx in seen:      # a degenerate short clip can collide; step forward
            idx += 1
        seen.add(idx)
        out.append(str(cfg.cache / cache_name(ds, video, idx)))
    return out


def build_row(cfg: ExportConfig, r, sysmsg: str) -> dict:
    k = int(r.g_K)
    paths = frame_paths(cfg, r.g_ds, r.video, r.g_st, r.g_dur, k)
    window = f"Clip [{seconds_to_ts(r.g_st)} - {seconds_to_ts(r.g_en)}] of the procedure."
    answer = str(r.answer)
    meta = {"qid": f"{r.g_ds}__{r.id}", "ds": r.g_ds, "video": r.video,
            "clip_start_s": float(r.g_st), "dur_s": float(r.g_dur), "K": k,
            "answer_format": r.answer_format, "primary": str(r.primary_capability),
            "split": r.g_split, "gold_original": answer, "time_kind": None,
            "visual_tokens": visual_tokens(k)}

    if r.answer_format == "time" and cfg.relative_time:
        if is_duration_question(r.question):
            meta["time_kind"] = "2b_elapsed"          # NEVER offset an elapsed span
        else:
            parts = [p.strip() for p in answer.split(",") if p.strip()]
            rel = [seconds_to_ts(ts_to_seconds(p) - r.g_st) for p in parts]
            answer = ", ".join(rel)                    # cardinality preserved: compare needs it
            meta["time_kind"] = "2a_relative"

    # fps the frames were really sampled at — the per-row channel the probe proved live
    raw_fps = (k - 1) / r.g_dur if (r.g_dur > 0 and k > 1) else 1.0
    rec = {
        "messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": f"<video>{window}\n{r.question}"},
            {"role": "assistant", "content": answer},
        ],
        "videos": [paths],
        "chat_template_kwargs": {"raw_fps": round(float(raw_fps), 6)},
        "_meta": meta,
    }
    assert_row_wellformed(rec, k)
    return rec


def build(cfg: ExportConfig) -> dict:
    """Build the corpus, run every gate, write the JSONL. Returns the report."""
    df = load(cfg)
    rep = {"rows_in": int(len(df)), "splits": list(cfg.splits)}
    rep.update(assert_router_pure(df))
    rep.update(assert_duration_router_pure(df))
    rep.update(assert_threshold_matches_sdk(cfg.sdk_base_dataset))

    if cfg.drop_percentage:
        df = df[df.answer_format != "percentage"]
    if cfg.smoke:
        # STRATIFIED, never a prefix slice: "heico" < "lapchole" makes a head() 100% OOD,
        # and a proportional draw expects 0.7 questions in the smallest bucket.
        df["g_stratum"] = (df.g_ds + "|" + df.answer_format + "|"
                          + df.g_dur.astype(int).astype(str))
        per = max(1, cfg.smoke // max(df.g_stratum.nunique(), 1))
        df = (df.groupby("g_stratum", group_keys=False)
                .apply(lambda g: g.head(per)).head(cfg.smoke).reset_index(drop=True))
        rep["smoke_rows"] = int(len(df))
        rep["smoke_strata"] = int(df.g_stratum.nunique())
        rep["smoke_datasets"] = sorted(df.g_ds.unique())
        rep["smoke_formats"] = sorted(df.answer_format.unique())
        rep["smoke_durations"] = sorted(int(x) for x in df.g_dur.unique())

    sysmsg = system_prompt()
    rows = [build_row(cfg, r, sysmsg) for r in df.itertuples(index=False)]

    rep.update(assert_frames_exist(rows, cfg.cache))
    if cfg.relative_time:
        rep.update(assert_time_roundtrip(rows))

    tok = [r["_meta"]["visual_tokens"] for r in rows]
    rep.update({
        "rows_out": len(rows),
        "visual_tokens_mean": int(sum(tok) / len(tok)),
        "visual_tokens_p95": int(pd.Series(tok).quantile(0.95)),
        "visual_tokens_max": int(max(tok)),
        "K_mean": round(sum(r["_meta"]["K"] for r in rows) / len(rows), 2),
        "K_max": max(r["_meta"]["K"] for r in rows),
        "by_format": df.answer_format.value_counts().to_dict(),
    })

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_meta"},
                               ensure_ascii=False) + "\n")
    with open(str(cfg.out) + ".meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r["_meta"], ensure_ascii=False) + "\n")
    rep["out"] = str(cfg.out)
    return rep
