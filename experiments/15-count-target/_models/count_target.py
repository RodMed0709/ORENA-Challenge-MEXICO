"""Rung 15 — the structured counting target. One variable: the assistant `content`
of `answer_format == "number"` rows.

Importable engine; the notebook calls ``main(cfg, stage=...)``. Nothing here is a
hand-run launcher.

The question
-----------
Rung 02/06 train `number` rows on a bare integer (``"3"``), so the whole gradient for
a counting question lands on ONE token and the model learns a prior over that token
rather than a count. Measured consequence (``context/ERROR_ANATOMY.md``): bias −0.66,
golds 3–4 share modal prediction 2, golds 5–8 share modal prediction 4, accuracy 0.000
at gold ≥7 — and the `number` margin over the template-aware floor decays with epochs
(rung 06 OOD: +0.015 → +0.013 → **+0.000**). Is that the TASK, or the TARGET FORMAT?

Why THIS target format (non-negotiable #9 — nothing here is invented)
---------------------------------------------------------------------
``literature/vlm-techniques/FICHAS.md``:

- **v05, Gautam 2025 — "Point, Detect, Count"** (arXiv:2505.16647). Qwen2.5-VL-7B +
  LoRA, ViT frozen: **Count MAE 9.86 → 0.26**. Its key trick, quoted: *"force a
  structured, parseable output (JSON with explicit `counts` field) and train counting
  jointly with localization, so the count has to be consistent with an enumerable set
  of points."* Its count annotations are stored, quoted, as ``{"counts": 3, "label":
  "polyp"}``. **We take those two field names verbatim.**
- **v14, Qwen3-VL Technical Report** (arXiv:2511.21631). Quoted: pretraining includes
  *"normalized grounding with box-based, point-based and counting supervision in a
  [0,1000] coordinate system"*, and Qwen3-VL *"extends the grounding capacity to
  support counting"*. So counting is a **pretrained grounding** capability with an
  explicit output convention, and a bare integer may be fighting that circuit.
- **v28, Guo 2025 — "Can't Even Count to 20"** (arXiv:2510.04401). VLMs count reliably
  with ONE object type and fail under compositional counting; its stated consequence
  for us is that *"a per-class count target ('clips: 2, sponges: 0') could be more
  learnable than a single scalar"* — i.e. the count must be **bound to a named class**.

**What we CANNOT take, and say so.** v05's second half and v14's [0,1000] convention
both need localization labels. There are none: *"The parquets carry `question` /
`answer` and metadata only: no bounding boxes, no masks, no coordinates"*
(``context/ERROR_ANATOMY.md`` §"There is NO spatial annotation anywhere in the
dataset"). Point-then-count is therefore **not reproducible here** and is not attempted.
This rung tests the part of v05 that our labels support: the structured, parseable,
label-bound count field.

**The one deviation, declared.** v05 writes ``{"counts": N, "label": L}``; we emit
``{"label": L, "counts": N}`` — same field NAMES, label first. Justified by v28: the
count must be bound to an explicit object type, and v28's own example writes the class
before the number. Nothing else about the target is ours.

Target, exactly (one line, no prose, no code fence)::

    {"label": "Clips", "counts": 2}

``label`` is the question's own noun phrase, **verbatim** — the capture group of
``COUNT_TEMPLATE_RE``. No singularisation, no mapping table, no vocabulary: the label is
a substring of the question, so the transform is deterministic and has zero invented
content.

🔴 The parser is a gate, not a detail
-------------------------------------
``focus.data.formats.Number.verify`` requires ``text.strip().isdigit()`` and
``Evaluator._evaluate_single`` marks a response that fails ``fmt.read`` **incorrect**
(``vendor/orena-focus/src/focus/evaluation/evaluator.py:337-343``). A structured answer
reaching the judge is therefore a guaranteed 0. :func:`parse_count` maps it back to a
bare integer BEFORE ``Response.content`` is built, it never raises, and its malformed
rate is measured — Perek 2026 (``literature/vlm-techniques/FICHAS.md`` v02) reports
vanilla CoT-SFT malformation at **4.76 %**,
so this is a measured failure mode, not a hypothetical.

What is verified where
----------------------
Pure and unit-tested offline: :func:`count_label`, :func:`is_count_question`,
:func:`render_count_target`, :func:`parse_count`, :func:`rewrite_train_lines` and every
``assert_*`` gate below. Unverified until the pod SMOKE (they need decord / frames /
swift / CUDA): the frame materialisation inside rung 02's ``_export``, ``swift sft``,
``swift export --merge_lora`` and ``run_baseline``. Those are **delegated verbatim** to
the rung-02/06 modules that already smoked them — never re-implemented here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

_RUNG06_MODELS = Path(__file__).resolve().parents[2] / "06-vit-lora" / "_models"
_RUNG02_MODELS = Path(__file__).resolve().parents[2] / "02-lora-sft" / "_models"
for _p in (_RUNG06_MODELS, _RUNG02_MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Rung 06 IS the control for this rung: same recipe, same seed, same epochs, same
# freeze_vit=False. Imported, never copied, so the training leg cannot drift.
from vit_lora_train import (  # noqa: E402
    ViTLoRAConfig,
    _swift_args,
    _train,
    list_checkpoints,
    merge_checkpoint,
    read_g1,
)
from lora_sft_train import _export as _rung02_export  # noqa: E402

logger = logging.getLogger(__name__)


# ── the count-question template ───────────────────────────────────────────────
# EXACT, anchored, and verified against the data card's committed template tables:
# it matches 10/10 `number` templates in templates_train.csv and 8/8 in
# templates_val.csv, and 0 of the other 426 / 180 templates. The anchoring is
# load-bearing: templates_val.csv also carries the OPEN-ENDED template
#   "How many radiopaque clips are visible in this video? Please provide a single integer."
# which a loose `^How many` regex would swallow, silently rewriting an open_ended row
# and turning a single-variable A/B into two.
COUNT_TEMPLATE_RE = re.compile(
    r"^\s*How many (?P<label>.+?) appear in this frame\?\s*Please provide a number\.\s*$"
)

# The parser's defined degradation. "0" is chosen, not invented, on two grounds:
#   (a) it is format-valid — Number.verify accepts "0" (formats.py:113-124);
#   (b) it is NOT the modal answer of ANY `number` template (data card
#       templates_train.csv / templates_val.csv: every modal answer is 1, 2 or 3), so a
#       malformed generation scores ZERO instead of quietly inheriting the template
#       floor. The malformed rate is therefore a cost we PAY and can see in the number,
#       which is the honest behaviour. `assert_safe_answer_not_modal` re-checks this on
#       the actual eval gold and RAISES if it ever stops being true.
SAFE_ANSWER = "0"

_BARE_INT_RE = re.compile(r"^\s*(\d+)\s*$")
# tolerant to markdown fences, single quotes, missing closing brace, trailing prose
_COUNTS_FIELD_RE = re.compile(r"""["']?counts["']?\s*:\s*["']?(\d+)""", re.IGNORECASE)
_ANY_INT_RE = re.compile(r"\d+")

# Statuses recorded by `parse_count`. A closed set so the malformed rate cannot be
# quietly redefined later.
STATUS_PASSTHROUGH = "passthrough"   # not a count question — returned untouched
STATUS_BARE_INT = "bare_int"         # already a bare integer (the rung-06 shape)
STATUS_STRUCTURED = "structured"     # our target format, parsed from the counts field
STATUS_SALVAGED = "salvaged"         # exactly one integer anywhere in the generation
STATUS_MALFORMED = "malformed"       # nothing parseable -> SAFE_ANSWER
STATUSES = (STATUS_PASSTHROUGH, STATUS_BARE_INT, STATUS_STRUCTURED,
            STATUS_SALVAGED, STATUS_MALFORMED)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class CountTargetConfig(ViTLoRAConfig):
    """Rung 06's config. ONE field is the experiment; the rest are gate inputs.

    Everything that defines the training run — recipe, seed, epochs, max_pixels,
    freeze_vit=False, freeze_aligner=True, target_modules — is inherited unchanged from
    ``ViTLoRAConfig``. ``diff_vs_rung06`` proves mechanically that the swift argv is
    IDENTICAL, so the only difference between this rung and its control is the bytes of
    ``train.jsonl``.
    """

    exp_dir: Path = Path("/workspace/repo/experiments/15-count-target")
    run_name: str = "15_count_target_v1"

    # 🎯 THE VARIABLE. OFF -> `_export` is literally rung 02's `_export` and the
    # resulting train.jsonl is byte-identical to rung 06's (gated by sha256, never
    # asserted in prose — see `assert_flag_off_identical`).
    count_target: bool = False

    # Rung 06's committed run dir, used by the flag-OFF sha256 gate when its
    # train.jsonl is present on the pod. None -> the gate rebuilds rung 06's export
    # into a scratch dir and compares against that instead (slower, same guarantee).
    rung06_run_dir: Path | None = None

    # The eval leg. Only read by the notebook; kept here so the whole run is one config.
    # Wire `parse_count` into the answer path at inference, via
    # `BaselineConfig.answer_postprocess` (src/frame/config.py:46 -> engine.py:129).
    parse_answers: bool = False

    @property
    def base_train_jsonl(self) -> Path:
        """The un-rewritten export, kept beside the rewritten one so the
        every-other-row-is-byte-identical gate has something to compare against."""
        return self.run_dir / "train_base.jsonl"

    @property
    def count_targets_jsonl(self) -> Path:
        """Sidecar: one record per rewritten row {qID, question, label, gold, target}.
        This is the set the 100 % round-trip gate runs over."""
        return self.run_dir / "count_targets.jsonl"

    # NOTE: the parse log is written PER EVAL (one per epoch) to
    # ``run_dir/<eval_tag>/parse_log.jsonl`` by ``count_answer_hook(log_path=...)`` — a
    # single run-level path would have each epoch overwrite the previous one's evidence.


# ── pure target construction ──────────────────────────────────────────────────

def is_count_question(question: str) -> bool:
    """True iff ``question`` is one of the `number`-format counting templates.

    Gated against the real labels by :func:`assert_count_regex_matches_formats` — a
    regex that drifts from `answer_format` would rewrite rows this rung never
    pre-registered, so it is never trusted, always checked.
    """
    return COUNT_TEMPLATE_RE.match(str(question)) is not None


def count_label(question: str) -> str | None:
    """The question's own noun phrase, verbatim (``"Clips"``, ``"different foreign
    object instances"``), or ``None`` when the question is not a counting template.

    Verbatim on purpose: a singularisation rule or a class-name lookup table would be
    invented content (non-negotiable #9). A substring of the question is not.
    """
    m = COUNT_TEMPLATE_RE.match(str(question))
    return m.group("label") if m else None


def render_count_target(label: str, count: int) -> str:
    """The assistant target for one `number` row.

    Field names ``label`` / ``counts`` are ``literature/vlm-techniques/FICHAS.md`` v05
    (Gautam 2025) verbatim; label-first order is the one declared deviation (v28, Guo
    2025 — the count must be bound to a named class). ``json.dumps`` with ``ensure_ascii=False`` so the emitted string is exactly
    what a JSON reader round-trips.
    """
    if int(count) < 0:
        raise ValueError(f"negative count {count!r} — Number format is non-negative")
    return json.dumps({"label": str(label), "counts": int(count)}, ensure_ascii=False)


# ── the parser — a GATE, never a detail ───────────────────────────────────────

@dataclass(frozen=True)
class ParseResult:
    answer: str      # what reaches Response.content (always Number-valid when is_count)
    status: str      # one of STATUSES
    raw: str         # the untouched generation, for the log


def parse_count(raw, *, is_count: bool = True) -> ParseResult:
    """Map a generation back to a bare non-negative integer string. NEVER raises.

    Tiers, in order:
      1. ``bare_int``   — the whole string is already an integer. This is the rung-06
         output shape, so a flag-OFF / control run passes through unchanged.
      2. ``structured`` — the first ``counts: N`` field found anywhere in the string.
         Tolerant to markdown fences, single quotes, a missing closing brace and
         trailing prose, because a 64-token greedy generation can truncate.
      3. ``salvaged``   — the string contains EXACTLY ONE integer run; use it. This
         tier is OURS, not from a paper, and is declared as such in the
         pre-registration. It is deliberately strict: two integers is ambiguous, so it
         degrades rather than guesses. None of the 10 count templates' labels contain a
         digit, so a single integer in a partial structured emission is the count.
      4. ``malformed``  — nothing parseable. Returns :data:`SAFE_ANSWER`.

    ``is_count=False`` returns the generation untouched with ``passthrough`` status:
    that is what keeps every non-`number` answer byte-identical to the control run.
    """
    text = "" if raw is None else str(raw)
    if not is_count:
        return ParseResult(text, STATUS_PASSTHROUGH, text)
    try:
        m = _BARE_INT_RE.match(text)
        if m:
            return ParseResult(str(int(m.group(1))), STATUS_BARE_INT, text)
        m = _COUNTS_FIELD_RE.search(text)
        if m:
            return ParseResult(str(int(m.group(1))), STATUS_STRUCTURED, text)
        ints = _ANY_INT_RE.findall(text)
        if len(ints) == 1:
            return ParseResult(str(int(ints[0])), STATUS_SALVAGED, text)
    except Exception:  # noqa: BLE001 — a parser that can throw is a run that can die
        logger.exception("parse_count failed on %r — degrading to SAFE_ANSWER", text[:80])
    return ParseResult(SAFE_ANSWER, STATUS_MALFORMED, text)


def malformed_stats(records: Iterable[dict]) -> dict:
    """Malformed rate over a parse log. Reported next to every score.

    ⚠️ ``run.py:42`` calls ``predict`` once to warm CUDA graphs BEFORE the item loop, so
    a full log carries at most ONE extra record (``call_index == 0``) that is not a
    scored question. Both rates are returned; they differ by at most 1/n and neither is
    silently preferred.
    """
    recs = list(records)
    counted = [r for r in recs if r.get("status") != STATUS_PASSTHROUGH]
    no_warm = [r for r in counted if r.get("call_index") != 0]

    def _rate(rs: list[dict]) -> float:
        return (sum(r["status"] == STATUS_MALFORMED for r in rs) / len(rs)) if rs else float("nan")

    by_status = {s: sum(r.get("status") == s for r in recs) for s in STATUSES}
    return {
        "n_calls": len(recs),
        "n_count_questions": len(counted),
        "malformed_rate": _rate(counted),
        "malformed_rate_excl_warmup": _rate(no_warm),
        "by_status": by_status,
        # Perek 2026 (vlm-techniques FICHAS v02) measures 4.76 % for vanilla CoT-SFT,
        # reported side by side so
        # our number is read against a published one, not against an intuition.
        "perek2026_vanilla_cot_sft_reference": 0.0476,
    }


# ── the export rewrite (pure over lines — no frames, no decord) ───────────────

@dataclass(frozen=True)
class RowSpec:
    """The alignment key for one exported row, recomputed from the SAME item list
    rung 02's ``_export`` iterates."""
    qID: str
    answer_format: str
    question: str
    answer: str
    image_path: str


def rewrite_train_lines(
    base_lines: list[str], specs: list[RowSpec]
) -> tuple[list[str], list[dict]]:
    """Rewrite the assistant content of `number` rows ONLY. Pure; unit-tested offline.

    Every non-`number` line is **copied verbatim, never re-serialised** — that is what
    makes "every other row is byte-identical" a property of the code rather than a
    promise about ``json.dumps`` key order.

    RAISES on any misalignment between ``base_lines`` and ``specs`` (row count, image
    path, or gold answer). A silent off-by-one would attach `number` targets to the
    wrong questions and the run would look like a result.
    """
    if len(base_lines) != len(specs):
        raise AssertionError(
            f"export misalignment: {len(base_lines)} jsonl lines vs {len(specs)} items — "
            "the rewrite recomputes rung 02's item order and must reproduce it exactly."
        )
    out: list[str] = []
    targets: list[dict] = []
    for i, (line, spec) in enumerate(zip(base_lines, specs)):
        rec = json.loads(line)
        got_img = list(rec.get("images", []))
        if got_img != [spec.image_path]:
            raise AssertionError(
                f"export misalignment at row {i} (qID={spec.qID}): jsonl images={got_img} "
                f"but the recomputed item points at {[spec.image_path]}."
            )
        content = rec["messages"][2]["content"]
        if content != str(spec.answer):
            raise AssertionError(
                f"export misalignment at row {i} (qID={spec.qID}): jsonl assistant "
                f"content={content!r} but the recomputed gold is {str(spec.answer)!r}."
            )
        if spec.answer_format != "number":
            out.append(line)          # verbatim — byte-identical by construction
            continue
        label = count_label(spec.question)
        if label is None:
            raise AssertionError(
                f"row {i} (qID={spec.qID}) has answer_format='number' but its question "
                f"does not match COUNT_TEMPLATE_RE: {spec.question!r}. A new counting "
                "template appeared — extend the regex deliberately, never loosen it "
                "(templates_val.csv carries an open_ended 'How many …' question)."
            )
        try:
            gold = int(str(spec.answer).strip())
        except ValueError as exc:
            raise AssertionError(
                f"row {i} (qID={spec.qID}): `number` gold {spec.answer!r} is not an "
                "integer — Number.verify would reject it too."
            ) from exc
        target = render_count_target(label, gold)
        rec["messages"][2]["content"] = target
        out.append(json.dumps(rec, ensure_ascii=False) + "\n")
        targets.append({"qID": spec.qID, "question": spec.question, "label": label,
                        "gold": gold, "target": target})
    return out, targets


# ── gates (RAISE; never disabled — context/RULES.md §7) ───────────────────────

def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_targets_round_trip(targets: Iterable[dict]) -> dict:
    """🔴 G-PARSE-1. The parser must recover **100 %** of the exported targets.

    Runs over the WHOLE exported set, not a sample. If one target does not round-trip,
    the training signal and the inference path disagree and every `number` number this
    rung produces is meaningless. RAISES with the failing rows.
    """
    ts = list(targets)
    if not ts:
        raise AssertionError("G-PARSE-1: no count targets to round-trip — export first.")
    bad = []
    by_status: dict[str, int] = {}
    for t in ts:
        r = parse_count(t["target"], is_count=True)
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if r.answer != str(t["gold"]) or r.status != STATUS_STRUCTURED:
            bad.append({"qID": t["qID"], "target": t["target"],
                        "gold": t["gold"], "got": r.answer, "status": r.status})
    if bad:
        raise AssertionError(
            f"G-PARSE-1 FAILED: {len(bad)}/{len(ts)} targets do not round-trip through "
            f"parse_count. First 5: {bad[:5]}"
        )
    logger.info("G-PARSE-1 OK: %d/%d targets round-trip (statuses=%s)",
                len(ts), len(ts), by_status)
    return {"n": len(ts), "by_status": by_status}


def assert_parser_never_raises(samples: Iterable[str]) -> dict:
    """G-PARSE-2. Malformed generation must degrade, never throw.

    A parser exception inside ``predict`` would be swallowed by ``run.py:53`` into
    ``"Inference Error: …"`` — a silent zero that looks like a model failure. This gate
    proves the degradation is a defined answer instead.
    """
    seen: dict[str, int] = {}
    for s in samples:
        r = parse_count(s, is_count=True)          # must not raise
        if r.status not in STATUSES:
            raise AssertionError(f"G-PARSE-2: unknown status {r.status!r} for {s!r}")
        if not r.answer.isdigit():
            raise AssertionError(
                f"G-PARSE-2 FAILED: parse_count({s!r}) -> {r.answer!r}, which "
                "focus.data.formats.Number.verify would REJECT (isdigit() is False)."
            )
        seen[r.status] = seen.get(r.status, 0) + 1
    logger.info("G-PARSE-2 OK: every input degraded to a Number-valid answer (%s)", seen)
    return seen


def assert_count_regex_matches_formats(items) -> dict:
    """G-REGEX. ``is_count_question`` must agree with `answer_format` on EVERY item.

    ``items`` = ``frame.data.FrameItem`` list. Two failure directions, both fatal:
    a `number` row we do not rewrite (the variable silently shrinks), and a non-`number`
    row we DO rewrite (a second variable appears). RAISES listing both.
    """
    missed, extra = [], []
    for it in items:
        fmt = str(it.reference._format)
        hit = is_count_question(it.request.question)
        if fmt == "number" and not hit:
            missed.append((it.request.qID, it.request.question))
        if fmt != "number" and hit:
            extra.append((it.request.qID, fmt, it.request.question))
    if missed or extra:
        raise AssertionError(
            f"G-REGEX FAILED: {len(missed)} `number` row(s) not matched by "
            f"COUNT_TEMPLATE_RE (e.g. {missed[:3]}) and {len(extra)} non-`number` row(s) "
            f"matched by it (e.g. {extra[:3]})."
        )
    n_num = sum(1 for it in items if str(it.reference._format) == "number")
    logger.info("G-REGEX OK: %d/%d items are `number` and exactly those match", n_num, len(items))
    return {"n_items": len(items), "n_number": n_num}


def assert_safe_answer_not_modal(gold, results_df) -> dict:
    """G-SAFE. :data:`SAFE_ANSWER` must not be the modal answer of any `number` template.

    If it ever were, a malformed generation would silently score at the template floor
    and the malformed rate would stop costing anything — the rate would still be
    reported, but it would no longer be a cost, and the margin would be flattered.
    """
    g = gold.copy()
    g["template"] = g["question"].map(_metrics().template_of)
    num_qids = set(results_df.loc[results_df["answer_format"].astype(str) == "number", "qID"])
    g = g[g["qID"].isin(num_qids)]
    modal = g.groupby("template")["answer"].agg(lambda s: s.value_counts().index[0])
    hits = [t for t, a in modal.items() if str(a).strip() == SAFE_ANSWER]
    if hits:
        raise AssertionError(
            f"G-SAFE FAILED: SAFE_ANSWER={SAFE_ANSWER!r} is the modal answer of "
            f"{len(hits)} `number` template(s), e.g. {hits[:2]}. A malformed generation "
            "would inherit the template floor instead of costing us — pick a fallback "
            "that is not modal, or the malformed rate stops being a cost."
        )
    logger.info("G-SAFE OK: %r is not modal in any of %d `number` templates",
                SAFE_ANSWER, len(modal))
    return {"n_templates": int(len(modal)), "modal_answers": sorted(set(map(str, modal.values)))}


def diff_vs_rung06(cfg: CountTargetConfig) -> dict[str, tuple]:
    """G-ARGV. Every swift flag where rung 15 differs from rung 06. Must be EMPTY.

    The training recipe is not merely "the same by intention": ``_swift_args`` is rung
    06's own function, called on both configs, and the argv is diffed. If a future edit
    changes a flag here, this gate turns the drift into a failure instead of a footnote.
    """
    def as_map(args: list[str]) -> dict[str, str]:
        out, i = {}, 0
        while i < len(args):
            if args[i].startswith("--"):
                val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else ""
                out[args[i]] = val
                i += 2 if val else 1
            else:
                i += 1
        return out

    control = ViTLoRAConfig(
        model_path=cfg.model_path, model_type=cfg.model_type, attn_impl=cfg.attn_impl,
        lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        learning_rate=cfg.learning_rate, num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        seed=cfg.seed, smoke=cfg.smoke, smoke_max_steps=cfg.smoke_max_steps,
        freeze_vit=cfg.freeze_vit, val_jsonl=cfg.val_jsonl,
        # identical output_dir/dataset paths on purpose: a path difference would show up
        # as a spurious diff and could hide a real one.
        exp_dir=cfg.exp_dir, run_name=cfg.run_name,
    )
    a, b = as_map(_swift_args(control)), as_map(_swift_args(cfg))
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def assert_flag_off_identical(cfg: CountTargetConfig, scratch_dir: str | Path) -> dict:
    """🔴 G-OFF. With ``count_target=False`` the whole ``train.jsonl`` must be
    **byte-identical** to rung 06's — proven by sha256, never asserted in prose
    (wave spec non-negotiable #4).

    Preferred comparand: rung 06's committed ``runs/06_vit_lora_v1/train.jsonl`` on the
    pod (``cfg.rung06_run_dir``). When it is absent, rung 06's exporter is re-run into
    ``scratch_dir`` and compared against that — slower, same guarantee.

    ⚠️ Touches decord + the frame cache, so it only runs on the pod. It is not optional
    there: it is the ONLY thing standing between this rung and an unnoticed second
    variable in the data leg.
    """
    scratch = Path(scratch_dir)
    off = _replace_cfg(cfg, count_target=False, exp_dir=scratch, run_name="g_off")
    off.run_dir.mkdir(parents=True, exist_ok=True)
    _export(off)
    ours = sha256_of(off.train_jsonl)

    ref_path: Path
    if cfg.rung06_run_dir is not None and (Path(cfg.rung06_run_dir) / "train.jsonl").exists():
        ref_path = Path(cfg.rung06_run_dir) / "train.jsonl"
        source = "rung 06 committed export"
    else:
        ctrl = ViTLoRAConfig(
            model_path=cfg.model_path, model_type=cfg.model_type,
            data_root=cfg.data_root, manifest_path=cfg.manifest_path,
            datasets=cfg.datasets, base_fps=cfg.base_fps,
            max_pixels=cfg.max_pixels, seed=cfg.seed, smoke=cfg.smoke,
            smoke_limit=cfg.smoke_limit,
            exp_dir=scratch, run_name="g_ref_rung06",
        )
        ctrl.run_dir.mkdir(parents=True, exist_ok=True)
        _rung02_export(ctrl)
        ref_path, source = ctrl.train_jsonl, "freshly rebuilt rung 06 export"
    ref = sha256_of(ref_path)
    if ours != ref:
        raise AssertionError(
            f"G-OFF FAILED: flag-OFF train.jsonl sha256 {ours} != {source} {ref} "
            f"({off.train_jsonl} vs {ref_path}). Flag OFF is NOT byte-identical to the "
            "control, so this rung is no longer a single-variable A/B."
        )
    logger.info("G-OFF OK: flag-OFF sha256 %s == %s", ours[:16], source)
    return {"sha256": ours, "reference": str(ref_path), "source": source}


def assert_nonnumber_lines_identical(base_path, rewritten_path, specs: list[RowSpec]) -> dict:
    """🔴 G-ON. With the flag ON, every NON-`number` line must still be byte-identical
    to the base export, and every `number` line must have changed.

    This is the flag-ON half of the single-variable claim: G-OFF proves the off path,
    this proves the on path touches exactly the rows the pre-registration named.
    """
    base = Path(base_path).read_text(encoding="utf-8").splitlines(keepends=True)
    new = Path(rewritten_path).read_text(encoding="utf-8").splitlines(keepends=True)
    if len(base) != len(new) or len(base) != len(specs):
        raise AssertionError(
            f"G-ON FAILED: line counts differ (base={len(base)}, rewritten={len(new)}, "
            f"specs={len(specs)})."
        )
    changed_other, unchanged_number = [], []
    n_number = 0
    for i, spec in enumerate(specs):
        if spec.answer_format == "number":
            n_number += 1
            if base[i] == new[i]:
                unchanged_number.append(spec.qID)
        elif base[i] != new[i]:
            changed_other.append(spec.qID)
    if changed_other or unchanged_number:
        raise AssertionError(
            f"G-ON FAILED: {len(changed_other)} non-`number` line(s) changed "
            f"(e.g. {changed_other[:3]}) and {len(unchanged_number)} `number` line(s) did "
            f"not (e.g. {unchanged_number[:3]})."
        )
    logger.info("G-ON OK: %d `number` rows rewritten, %d other rows byte-identical",
                n_number, len(base) - n_number)
    return {"n_rows": len(base), "n_rewritten": n_number,
            "n_untouched": len(base) - n_number}


# ── inference wiring — the parser on the answer path ──────────────────────────
# `run.py:186` builds Response.content straight from `engine.predict(...)`, and
# `Evaluator._evaluate_single` (evaluator.py:337) marks anything Number.verify rejects
# as INCORRECT. So the structured answer must become a bare integer inside `predict`,
# before the Response exists — there is no later hook.
#
# That hook is `BaselineConfig.answer_postprocess` (`src/frame/config.py:46`), applied at
# `src/frame/engine.py:129` — the LAST thing `predict` does, after `predict_samples` has
# produced the generation and BEFORE `Response.content` is built and before the SDK's
# format verification ever sees it. DEFAULT OFF IS BYTE-IDENTICAL: `None` skips the call
# entirely, so a control run takes the identical path it always has.
#
# ⚠️ This replaces an earlier monkeypatch of `frame.run.QwenFrameEngine`. The patch was
# functionally correct — it wrapped the same `predict` at the same point — but it was
# correct by coincidence of module-global resolution order, in the one place where a
# post-processor that silently fails to wire is indistinguishable from a model that
# cannot count (an unparsed structured answer scores 0 by construction). A declared
# config field cannot be defeated by an import order.

PARSE_LOG: list[dict] = []


def make_count_postprocess(log: list[dict] | None = None):
    """Build the ``BaselineConfig.answer_postprocess`` callable: ``fn(answer, question)``.

    Non-count questions take the identical path they always have: :func:`parse_count`
    with ``is_count=False`` returns the generation untouched.
    """
    sink = PARSE_LOG if log is None else log

    def postprocess(answer: str, question: str) -> str:
        res = parse_count(answer, is_count=is_count_question(question))
        sink.append({
            "call_index": len(sink),     # 0 == run.py:42's CUDA warm-up call
            "question": question,
            "raw": res.raw,
            "answer": res.answer,
            "status": res.status,
        })
        return res.answer

    return postprocess


@contextmanager
def count_answer_hook(*, enabled: bool, log_path: str | Path | None = None) -> Iterator[tuple]:
    """Yield ``(answer_postprocess, parse_log)`` for the duration, and flush the log.

    ``enabled=False`` yields ``(None, PARSE_LOG)`` — and ``answer_postprocess=None`` is
    exactly the default, so the control run is byte-identical. That is what lets the same
    notebook produce both arms.

    The caller passes the yielded callable into ``BaselineConfig(answer_postprocess=...)``.
    Nothing is patched, so nothing has to be restored; the ``finally`` exists only to
    write the parse log even when the eval raised.
    """
    if not enabled:
        logger.info("count_answer_hook: DISABLED — answer_postprocess stays None")
        yield None, PARSE_LOG
        return
    PARSE_LOG.clear()
    logger.info("count_answer_hook: ENABLED — BaselineConfig.answer_postprocess = parse_count")
    try:
        yield make_count_postprocess(PARSE_LOG), PARSE_LOG
    finally:
        if log_path is not None:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                for r in PARSE_LOG:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("parse log (%d calls) -> %s", len(PARSE_LOG), p)


# ── reporting: the per-template `number` margin ───────────────────────────────

def _metrics():
    from frame import metrics  # noqa: PLC0415
    return metrics


def number_margin_by_template(results_df, gold):
    """Per-template `number` accuracy, template-aware floor and margin, split ID/OOD.

    🔴 ``acc_number`` is NOT interpretable pooled (``context/RULES.md`` §12, data card
    rung 08 §3): it averages 8 val templates whose floors run 0.24–1.00, **four of them
    degenerate** (floor 1.00 — "How many Needles / Specimens / Specimen bags / External
    drains", modal answer 1). A change confined to the two big templates can be buried,
    or a change in a degenerate one can masquerade as progress. This is the table the
    verdict is read from, next to the canonical ``by_format`` number margin.

    Composed from the canonical primitives — ``metrics.template_of``,
    ``metrics.template_floor`` and ``metrics._dist_from_qid`` — so the floor and the
    ID/OOD rule have exactly ONE implementation in the repo (``context/RULES.md``
    §EVAL rule 1: extend the module, never reimplement beside it). ``_dist_from_qid`` is
    private only by naming; re-deriving "heico == OOD" here would be the defect the rule
    exists to prevent.
    """
    import pandas as pd  # noqa: PLC0415

    m = _metrics()
    g = gold[["qID", "answer", "question"]].drop_duplicates("qID").copy()
    df = results_df.merge(g, on="qID", how="left")
    df = df[df["answer_format"].astype(str) == "number"].copy()
    if df["answer"].isna().any():
        raise AssertionError(
            f"{int(df['answer'].isna().sum())} `number` rows have no gold — the "
            "template floor would be computed on a partial slice and the margin inflated."
        )
    df["_template"] = df["question"].map(m.template_of)
    df["_dist"] = df["qID"].map(m._dist_from_qid)
    df["_correct"] = m._correct_series(df)
    rows = []
    for (tpl, dist), sub in df.groupby(["_template", "_dist"], sort=True):
        acc = float(sub["_correct"].mean())
        floor = m.template_floor(sub, answer_col="answer", template_col="_template")
        rows.append({"template": tpl, "distribution": dist, "n": int(len(sub)),
                     "accuracy": acc, "floor": floor, "margin": acc - floor,
                     "degenerate": bool(floor >= 1.0)})
    return pd.DataFrame(rows).sort_values(["distribution", "n"], ascending=[True, False])


# ── stages ────────────────────────────────────────────────────────────────────

def _replace_cfg(cfg: CountTargetConfig, **changes) -> CountTargetConfig:
    from dataclasses import replace  # noqa: PLC0415
    return replace(cfg, **changes)


def _row_specs(cfg: CountTargetConfig) -> list[RowSpec]:
    """Recompute rung 02's exported item order EXACTLY.

    Same calls, same order, same smoke subsample as ``lora_sft_train._export``
    (lora_sft_train.py:100-146) — load ("train","test") -> apply_split("train") ->
    optional stratified smoke cap -> sort by (dataset, video_id, frame_index). Any drift
    is caught immediately by the image-path and gold checks in
    :func:`rewrite_train_lines`, which is why this is allowed to be a re-derivation.
    """
    from collections import defaultdict  # noqa: PLC0415

    from frame.config import BaselineConfig  # noqa: PLC0415
    from frame.data import frame_cache_name, load_frame_items  # noqa: PLC0415
    from frame import split as sp  # noqa: PLC0415

    bcfg = BaselineConfig(
        data_root=cfg.data_root, model_path=cfg.model_path,
        datasets=cfg.datasets, base_fps=cfg.base_fps,
        max_pixels=cfg.max_pixels, seed=cfg.seed,
    )
    items = load_frame_items(bcfg, splits=("train", "test"))
    vs = sp.load_manifest(cfg.manifest_path)
    train_items = sp.apply_split(items, vs, "train")
    if cfg.smoke:
        by_ds: dict[str, list] = defaultdict(list)
        for it in train_items:
            by_ds[it.dataset].append(it)
        k = max(1, cfg.smoke_limit // max(1, len(by_ds)))
        train_items = [it for ds in by_ds for it in by_ds[ds][:k]]
    train_items.sort(key=lambda it: (it.dataset, it.video_id, it.frame_index))
    return [
        RowSpec(
            qID=it.request.qID,
            answer_format=str(it.reference._format),
            question=it.request.question,
            answer=str(it.reference.answer),
            image_path=str(cfg.frames_dir / frame_cache_name(it)),
        )
        for it in train_items
    ]


def _export(cfg: CountTargetConfig) -> Path:
    """Build ``train.jsonl``.

    Flag OFF: rung 02's ``_export`` verbatim and nothing else — same frames, same
    ordering, same bytes. Flag ON: the same export, then the `number`-row rewrite on
    top of the produced file, with the base kept beside it for G-ON.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _rung02_export(cfg)                       # writes cfg.train_jsonl — the control bytes
    if not cfg.count_target:
        logger.info("count_target OFF — train.jsonl is rung 06's export verbatim (sha256 %s)",
                    sha256_of(cfg.train_jsonl)[:16])
        return cfg.train_jsonl

    base_lines = cfg.train_jsonl.read_text(encoding="utf-8").splitlines(keepends=True)
    specs = _row_specs(cfg)
    new_lines, targets = rewrite_train_lines(base_lines, specs)

    cfg.base_train_jsonl.write_text("".join(base_lines), encoding="utf-8")
    cfg.train_jsonl.write_text("".join(new_lines), encoding="utf-8")
    with open(cfg.count_targets_jsonl, "w", encoding="utf-8") as fh:
        for t in targets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 🔴 Both gates run INSIDE the export, so a broken rewrite cannot reach the GPU.
    assert_nonnumber_lines_identical(cfg.base_train_jsonl, cfg.train_jsonl, specs)
    assert_targets_round_trip(targets)
    logger.info("count_target ON — rewrote %d/%d rows; train.jsonl sha256 %s",
                len(targets), len(new_lines), sha256_of(cfg.train_jsonl)[:16])
    return cfg.train_jsonl


def main(cfg: CountTargetConfig, stage: str) -> Path:
    """stage ∈ {'export', 'train'}.

    ``train`` delegates to rung 06's ``_train`` UNCHANGED — same argv, same run guard,
    same per-epoch ``save_strategy``. Merge stays per-epoch in the notebook via rung 02's
    ``merge_checkpoint``, exactly as rungs 02 and 06 do it.
    """
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return _export(cfg)
    if stage == "train":
        return _train(cfg)
    raise ValueError(
        f"unknown stage {stage!r} (export|train); merge is per-epoch in the notebook "
        "(list_checkpoints -> merge_checkpoint, rung 02's own functions)"
    )


__all__ = [
    "CountTargetConfig", "main",
    "COUNT_TEMPLATE_RE", "SAFE_ANSWER", "STATUSES",
    "is_count_question", "count_label", "render_count_target",
    "parse_count", "ParseResult", "malformed_stats",
    "RowSpec", "rewrite_train_lines",
    "assert_targets_round_trip", "assert_parser_never_raises",
    "assert_count_regex_matches_formats", "assert_safe_answer_not_modal",
    "assert_flag_off_identical", "assert_nonnumber_lines_identical",
    "diff_vs_rung06", "sha256_of",
    "count_answer_hook", "make_count_postprocess", "PARSE_LOG",
    "number_margin_by_template",
    "list_checkpoints", "merge_checkpoint", "read_g1",
]
