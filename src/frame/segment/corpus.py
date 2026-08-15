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
    video_lengths: Path = Path("/workspace/tmp/video_lengths.json")
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
    """k is the REALISED frame count (len of the grid after clamping/dedup), not the
    requested one — see frame_indices()."""
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
    n = n_multi = 0
    for r in rows:
        meta = r["_meta"]
        if meta.get("time_kind") != "2a_relative":
            continue
        # 1.84% of time golds carry 2-8 comma-separated stamps, and `Time.compare`
        # rejects a cardinality mismatch BEFORE comparing values — so the round-trip
        # must be per-part, exactly like `Time._split`.
        emitted = [p.strip() for p in r["messages"][-1]["content"].split(",") if p.strip()]
        back = ", ".join(seconds_to_ts(ts_to_seconds(p) + meta["clip_start_s"]) for p in emitted)
        if back != meta["gold_original"]:
            raise AssertionError(
                f"round-trip failed: {r['messages'][-1]['content']!r} + {meta['clip_start_s']} "
                f"-> {back!r} != {meta['gold_original']!r}"
            )
        if len(emitted) > 1:
            n_multi += 1
        n += 1
    if n == 0:
        raise AssertionError("round-trip gate checked ZERO rows — it proves nothing")
    return {"roundtrip_checked": n, "roundtrip_multi_stamp": n_multi}


def assert_frames_exist(rows: list[dict], cache: Path) -> dict:
    """Every referenced frame must be on disk. One readdir, not N stat calls."""
    import os

    have = set(os.listdir(cache))
    missing = []
    for r in rows:
        for p in r["videos"][0]:
            if Path(p).name not in have:
                m = r["_meta"]
                missing.append(
                    f"{Path(p).name} (qid={m['qid']} video={m['video']!r} "
                    f"start={m['clip_start_s']} dur={m['dur_s']} K={m['K']})")
    if missing:
        # a gate that reports a COUNT and not the evidence costs three debugging
        # rounds; this one names the rows so the cause is readable at first fire
        raise AssertionError(
            f"{len(missing)} referenced frames are not in {cache}:\n  "
            + "\n  ".join(missing[:10])
            + (f"\n  ... and {len(missing)-10} more" if len(missing) > 10 else ""))
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
    # THE GRID IS A PROPERTY OF THE CLIP, NOT OF THE QUESTION.
    # Two reasons, and the second is the important one:
    #  1. the extractor materialises one grid per clip (the finest any of its questions
    #     needs). A coarser per-question grid is NOT a subset of a finer one — only the
    #     endpoints coincide — so a K=4 row on a clip that also carries a K=53 row
    #     references interior frames nobody wrote. That is the 6-frame failure.
    #  2. if K co-varied with question type, the frame COUNT would leak the question
    #     type into the visual input, and the model could route on it without reading a
    #     pixel. Uniform K per clip removes that shortcut by construction.
    df["g_K"] = df.groupby(["g_ds", "video", "g_st", "g_en"]).g_K.transform("max")
    return df


def frame_indices(ds: str, st: float, dur: float, k: int, n_frames: int | None = None) -> list[int]:
    """THE frame grid. One function, imported by both the extractor and the exporter.

    Two copies of this arithmetic is how a corpus ends up referencing a frame nobody
    wrote: the extractor clamped an index past the end of the video, the exporter did
    not, and the two disagreed on 6 of 257,047 frames. Caught by `assert_frames_exist`.
    """
    bf = BASE_FPS[ds]
    out: list[int] = []
    seen: set[int] = set()
    for i in range(k):
        t = st + (dur * i / (k - 1) if k > 1 else 0.0)
        idx = round(t * bf)
        if n_frames is not None:
            idx = min(idx, n_frames - 1)
        # COLLAPSE duplicates, never step to a neighbour. When a clip runs past the end
        # of its video several grid points clamp onto the same last frame; the extractor
        # deduped them through a set, so inventing idx±1 here references a frame nobody
        # wrote. The row simply gets fewer than k frames, which is the honest outcome.
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def frame_paths(cfg: ExportConfig, ds: str, video: str, st: float, dur: float, k: int,
                n_frames: int | None = None) -> list[str]:
    return [str(cfg.cache / cache_name(ds, video, i))
            for i in frame_indices(ds, st, dur, k, n_frames)]


def build_row(cfg: ExportConfig, r, sysmsg: str, lengths: dict | None = None) -> dict:
    k_req = int(r.g_K)
    nf = (lengths or {}).get(f"{r.g_ds}|{r.video}")
    paths = frame_paths(cfg, r.g_ds, r.video, r.g_st, r.g_dur, k_req, nf)
    k = len(paths)                      # realised, after clamp + dedup
    window = f"Clip [{seconds_to_ts(r.g_st)} - {seconds_to_ts(r.g_en)}] of the procedure."
    answer = str(r.answer)
    meta = {"qid": f"{r.g_ds}__{r.id}", "ds": r.g_ds, "video": r.video,
            "clip_start_s": float(r.g_st), "dur_s": float(r.g_dur),
            "K": k, "K_requested": k_req,
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
    lengths = {}
    if Path(cfg.video_lengths).exists():
        lengths = json.loads(Path(cfg.video_lengths).read_text(encoding="utf-8"))
        rep["video_lengths_known"] = len(lengths)
    rows = [build_row(cfg, r, sysmsg, lengths) for r in df.itertuples(index=False)]

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
