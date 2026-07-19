"""CoA scaffold generator for FRAME LoRA training data (experiment 09, R1).

Pure, importable engine. It restructures the rung-02 training target from the **bare
gold answer** into a **Chain-of-Answer scaffold** that ends in the gold:

    <description>…</description><evidence>…</evidence><thought>…</thought><answer>GOLD</answer>

The scaffold is **reverse-generated** by a text LLM that is handed the KNOWN gold and a
per-frame **sibling fact-sheet** (the other verified (question, gold) pairs on the same
frame) — it never sees the pixels (see ``context/09-coa-sft/CONTEXT.md`` for why). The
single variable vs rung 02 is this assistant target; ``system``/``user``/``images`` stay
byte-identical (``lora_sft_train._export``).

This engine is PURE: it builds prompts and validates completions, but does NOT call any
LLM — the orchestrator drives the deepseek / Claude backends and passes completions back
in. It runs locally with only pandas + stdlib (no ``focus`` / ``transformers`` import;
``SYSTEM_PROMPT`` is resolved lazily and only on the pod).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FRAMES_CACHE = Path("/workspace/frames_cache")
BASE_FPS = {"heico": 25, "lapchole": 30}
ANSWER_FORMATS = ("fo_class", "number", "open_ended", "binary", "multiple_choice")
PLACEHOLDER_SYSTEM_PROMPT = "SYSTEM_PROMPT (resolved from frame.engine on the pod)"

# The reverse-generation instruction handed to the text backend. Gold-anchored, no pixels.
GEN_PROMPT_TEMPLATE = (
    "You are writing a chain-of-answer reasoning scaffold for a surgical VQA training "
    "example about foreign objects in a SINGLE laparoscopic frame. You are given the "
    "QUESTION, the GROUND-TRUTH answer, and a FACT SHEET of other verified facts about the "
    "SAME frame. You CANNOT see the image. Write reasoning a competent surgeon would produce "
    "that DERIVES the ground-truth answer from visual evidence — never assert it as a given.\n\n"
    "Output EXACTLY these four tags, in order, and NOTHING else:\n"
    "<description>1-2 sentences: the surgical scene in general terms, using ONLY facts from "
    "the fact sheet or safe generic anatomy. Do NOT state the answer here.</description>\n"
    "<evidence>The specific visual facts the answer rests on. Derive; do not assert the "
    "answer as a premise.</evidence>\n"
    "<thought>The single reasoning step from evidence to the answer.</thought>\n"
    "<answer>GOLD</answer>\n\n"
    "Rules:\n"
    "- The <answer> content MUST be exactly: {gold}\n"
    "- NEVER state the gold value as a premise in <description> or <evidence> (do not write "
    "the count/class/yes-no before you reason to it). Reason TO it.\n"
    "- Keep it procedure-generic; do not assume gallbladder/clip specifics unless the "
    "procedure is cholecystectomy.\n"
    "- No text before <description> or after </answer>. Total under ~80 words.\n\n"
    "QUESTION: {question}\n"
    "ANSWER FORMAT: {answer_format}\n"
    "PROCEDURE: {procedure_type}\n"
    "GROUND-TRUTH ANSWER: {gold}\n"
    "FACT SHEET (other verified facts about THIS frame; may be 'none'):\n"
    "{siblings}\n"
)

_BACKENDS = ("deepseek", "claude")


@dataclass
class GenConfig:
    """Config for one scaffold-generation run (inline in the notebook)."""

    data_root: Path = Path("external_data/orena-data")  # LOCAL parquets for Stage 1
    datasets: tuple[str, ...] = ("heico", "lapchole")
    base_fps: dict = field(default_factory=lambda: dict(BASE_FPS))
    exp_dir: Path = Path("experiments/09-coa-sft")
    run_name: str = "09_coa_sft_v1"
    frames_cache: Path = FRAMES_CACHE
    seed: int = 42
    sample_n: int = 50
    single_q_frac: float = 0.6  # oversample single-Q frames (62% of the real set has no sibling)

    @property
    def run_dir(self) -> Path:
        return self.exp_dir / "runs" / self.run_name

    @property
    def eyeball_csv(self) -> Path:
        return self.run_dir / "sample_eyeball.csv"


# ── canonical helpers (faithful local copies; the pod uses the real focus/frame ones) ──

def ts_to_seconds(ts: str) -> int:
    """"hh:mm:ss" -> total seconds (mirror of focus.data.formats.ts_to_seconds)."""
    h, m, s = str(ts).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _sanitize_video(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(video_id)).strip("_")


def frame_index_of(dataset: str, timestamp_start: str, base_fps: dict) -> int:
    return round(ts_to_seconds(timestamp_start) * base_fps[dataset])


def frame_cache_name(dataset: str, video_id: str, frame_index: int) -> str:
    """Mirror of frame.data.frame_cache_name: identity-keyed shared-cache filename."""
    return f"{dataset}__{_sanitize_video(video_id)}__{frame_index}.jpg"


# ── data loading + sibling fact-sheets ──

def load_train_rows(cfg: GenConfig) -> pd.DataFrame:
    """Load both train parquets, tag dataset/qID/frame-identity + sibling counts."""
    frames = []
    for ds in cfg.datasets:
        pq = cfg.data_root / ds / "data" / "frame" / "train.parquet"
        df = pd.read_parquet(pq)
        df["dataset"] = ds
        df["qID"] = ds + "__" + df["id"].astype(str)
        # exact grouping key for one physical frame (no ts parsing needed to group)
        df["frameid"] = ds + "__" + df["video"].astype(str) + "__" + df["timestamp_start"].astype(str)
        frames.append(df)
        logger.info("loaded %d rows from %s/train.parquet", len(df), ds)
    out = pd.concat(frames, ignore_index=True)
    out["n_q_on_frame"] = out.groupby("frameid")["qID"].transform("size")
    out["is_single_q"] = out["n_q_on_frame"] == 1
    logger.info("total %d rows | %d frames | %d single-Q rows (%.0f%%)",
                len(out), out["frameid"].nunique(), int(out["is_single_q"].sum()),
                100 * out["is_single_q"].mean())
    return out


def build_factsheets(df: pd.DataFrame) -> dict[str, list[dict]]:
    """frameid -> list of {qID, question, answer, answer_format} for every row on it."""
    sheets: dict[str, list[dict]] = {}
    for fid, grp in df.groupby("frameid"):
        sheets[fid] = grp[["qID", "question", "answer", "answer_format"]].to_dict("records")
    return sheets


def siblings_for(factsheets: dict[str, list[dict]], frameid: str, qID: str) -> str:
    """Render the sibling fact-sheet for qID, EXCLUDING its own row. 'none' if empty."""
    rows = [r for r in factsheets.get(frameid, []) if r["qID"] != qID]
    if not rows:
        return "none"
    return "\n".join(f"- Q: {r['question']} -> A: {r['answer']}" for r in rows)


# ── stratified sampling (deterministic; oversample single-Q frames) ──

def stratified_sample(cfg: GenConfig, df: pd.DataFrame) -> pd.DataFrame:
    """~sample_n rows stratified across answer_format x dataset, oversampling single-Q.

    Deterministic (seed). At least one row per non-empty (answer_format, dataset) cell;
    within the draw, ~single_q_frac of rows have is_single_q==True to stress the case
    that has no sibling fact-sheet (62% of the real train set).
    """
    n_single = round(cfg.sample_n * cfg.single_q_frac)
    n_multi = cfg.sample_n - n_single
    picks: list[pd.DataFrame] = []

    for want, pool in ((n_single, df[df["is_single_q"]]), (n_multi, df[~df["is_single_q"]])):
        if pool.empty or want <= 0:
            continue
        # proportional-ish across (answer_format, dataset), >=1 per non-empty cell
        cells = [g for _, g in pool.groupby(["answer_format", "dataset"]) if len(g)]
        per = max(1, want // max(1, len(cells)))
        got: list[pd.DataFrame] = []
        for g in cells:
            got.append(g.sample(n=min(per, len(g)), random_state=cfg.seed))
        drawn = pd.concat(got, ignore_index=True) if got else pool.head(0)
        # top up / trim to `want` from the remaining pool
        if len(drawn) < want:
            rest = pool.drop(index=pool.index.intersection(drawn.index), errors="ignore")
            extra = rest.sample(n=min(want - len(drawn), len(rest)), random_state=cfg.seed)
            drawn = pd.concat([drawn, extra], ignore_index=True)
        picks.append(drawn.head(want))

    sample = pd.concat(picks, ignore_index=True) if picks else df.head(0)
    sample = sample.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)  # shuffle order
    logger.info("sampled %d rows (%d single-Q / %d multi-Q)", len(sample),
                int(sample["is_single_q"].sum()), int((~sample["is_single_q"]).sum()))
    return sample


# ── prompt + validation ──

def build_prompt(row: pd.Series, siblings: str) -> str:
    return GEN_PROMPT_TEMPLATE.format(
        question=row["question"], answer_format=row["answer_format"],
        procedure_type=row["procedure_type"], gold=str(row["answer"]), siblings=siblings,
    )


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_scaffold(completion: str) -> dict:
    """Extract the four tag bodies (None if missing) + count of <answer> tags."""
    def grab(tag: str):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", completion, re.DOTALL)
        return m.group(1).strip() if m else None
    return {
        "description": grab("description"),
        "evidence": grab("evidence"),
        "thought": grab("thought"),
        "answer": grab("answer"),
        "n_answer_tags": len(re.findall(r"<answer>", completion)),
    }


def validate_scaffold(completion: str, gold, answer_format: str) -> dict:
    """Return flag columns for one scaffold. `valid` ANDs the hard checks."""
    completion = completion or ""
    p = parse_scaffold(completion)
    tags = ["description", "evidence", "thought", "answer"]
    positions = [completion.find(f"<{t}>") for t in tags]
    has_all_tags = all(v is not None for v in (p["description"], p["evidence"], p["thought"], p["answer"])) \
        and all(positions[i] != -1 for i in range(4)) and positions == sorted(positions)
    exactly_one_answer = p["n_answer_tags"] == 1

    ans = p["answer"]
    answer_matches_gold = ans is not None and _norm(ans) == _norm(gold)
    if not answer_matches_gold and ans is not None and answer_format == "number":
        try:
            answer_matches_gold = int(re.sub(r"[^\d-]", "", ans)) == int(re.sub(r"[^\d-]", "", str(gold)))
        except (ValueError, TypeError):
            pass

    m_last = list(re.finditer(r"</answer>", completion))
    nothing_after_answer = bool(m_last) and completion[m_last[-1].end():].strip() == ""
    first = completion.find("<description>")
    no_pretext = first != -1 and completion[:first].strip() == ""

    ctx = _norm((p["description"] or "") + " " + (p["evidence"] or ""))
    gnorm = _norm(gold)
    if answer_format == "number":
        digits = re.sub(r"[^\d]", "", str(gold))
        answer_leak = bool(digits) and re.search(rf"\b{re.escape(digits)}\b", ctx) is not None
    else:
        answer_leak = bool(gnorm) and gnorm in ctx

    if answer_format == "number":
        canonical_answer = ans is not None and re.fullmatch(r"-?\d+", ans.strip()) is not None
    elif answer_format == "binary":
        canonical_answer = ans is not None and _norm(ans) in {"yes", "no"}
    else:
        canonical_answer = ans is not None

    word_count = len(completion.split())
    valid = bool(has_all_tags and exactly_one_answer and answer_matches_gold
                 and nothing_after_answer and no_pretext and not answer_leak and canonical_answer)
    return {
        "has_all_tags": has_all_tags,
        "exactly_one_answer": exactly_one_answer,
        "answer_matches_gold": answer_matches_gold,
        "nothing_after_answer": nothing_after_answer,
        "no_pretext": no_pretext,
        "answer_leak": answer_leak,
        "canonical_answer": canonical_answer,
        "word_count": word_count,
        "valid": valid,
    }


# ── record assembly + eyeball CSV ──

def _system_prompt() -> str:
    try:  # only resolves on the pod, where frame.engine + its deps exist
        from frame.engine import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    except Exception:  # noqa: BLE001 — local Stage-1 gen never needs the real prompt
        return PLACEHOLDER_SYSTEM_PROMPT


def to_sharegpt_record(cfg: GenConfig, row: pd.Series, scaffold_completion: str) -> dict:
    """rung-02-parity ShareGPT record; ONLY the assistant content differs (the scaffold)."""
    idx = frame_index_of(row["dataset"], row["timestamp_start"], cfg.base_fps)
    img = cfg.frames_cache / frame_cache_name(row["dataset"], row["video"], idx)
    return {
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"<image>{row['question']}"},
            {"role": "assistant", "content": scaffold_completion},
        ],
        "images": [str(img)],
    }


def build_eyeball_rows(cfg: GenConfig, sample_df: pd.DataFrame,
                       factsheets: dict[str, list[dict]],
                       completions: dict[str, dict[str, str]]) -> list[dict]:
    """One flat row per sampled question, both backends side by side + per-backend flags.

    `completions[qID] = {"deepseek": <text>, "claude": <text>}` (missing -> "").
    """
    rows: list[dict] = []
    for r in sample_df.to_dict("records"):
        qID = r["qID"]
        sib = siblings_for(factsheets, r["frameid"], qID)
        comp = completions.get(qID, {})
        out = {
            "qID": qID,
            "dataset": r["dataset"],
            "ood_proxy": r["dataset"] == "heico",
            "answer_format": r["answer_format"],
            "is_single_q": r["is_single_q"],
            "n_q_on_frame": r["n_q_on_frame"],
            "procedure_type": r["procedure_type"],
            "question": r["question"],
            "gold": r["answer"],
            "siblings": sib,
            "prompt": build_prompt(pd.Series(r), sib),
        }
        for b in _BACKENDS:
            text = comp.get(b, "") or ""
            out[f"{b}_scaffold"] = text
            for k, v in validate_scaffold(text, r["answer"], r["answer_format"]).items():
                out[f"{b}__{k}"] = v
        rows.append(out)
    return rows


def write_eyeball_csv(cfg: GenConfig, rows: list[dict]) -> Path:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg.eyeball_csv, index=False)
    logger.info("wrote %d eyeball rows -> %s", len(rows), cfg.eyeball_csv)
    return cfg.eyeball_csv


def build_sample(cfg: GenConfig) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Convenience: load -> factsheets -> stratified sample -> per-row prompt specs.

    Returns (sample_df, factsheets, prompt_specs) where each spec is
    {qID, prompt, gold, answer_format, dataset, is_single_q, siblings} — the orchestrator
    feeds `prompt` to each backend, collects completions, then calls build_eyeball_rows.
    """
    df = load_train_rows(cfg)
    factsheets = build_factsheets(df)
    sample = stratified_sample(cfg, df)
    specs: list[dict] = []
    for r in sample.to_dict("records"):
        sib = siblings_for(factsheets, r["frameid"], r["qID"])
        specs.append({
            "qID": r["qID"], "prompt": build_prompt(pd.Series(r), sib),
            "gold": r["answer"], "answer_format": r["answer_format"],
            "dataset": r["dataset"], "is_single_q": r["is_single_q"], "siblings": sib,
        })
    return sample, factsheets, specs
