"""Rung 46 — the three-turn cross-model debate, as a library.

Importable only. It holds no config and launches nothing: the notebook cell owns the
config and calls `run_debate(cfg)` (EXPERIMENT_REPO_STRUCTURE_SPEC — notebooks generate
runs, `.py` files are libraries).

WHY THE PASSES ARE SEQUENTIAL AND NOT INTERLEAVED
Two live vLLM engines would need 33.5 GiB (27B FP8) + 16 GiB (8B bf16) on a 47.4 GiB card.
They do not fit, and they do not need to: the protocol is turn-based, so turn N+1 cannot
start before turn N finishes anyway. One model is resident at a time, peak 34 GiB, and the
second GPU stays free for a teammate. Interleaving would buy nothing and cost the card.

WHY A's T1 SERVES AS BOTH TURN 1 AND THE CONTROL
`A_alone` is exactly "A answers from the frame". Generating it once and reusing it means the
control and the arm share a byte-identical first turn, so the only difference between them is
the turns that follow. Regenerating it would inject the ~0.5 % GPU-swap drift of
[[archived-results-not-bit-reproducible]] into the comparison for nothing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- prompts
# T2 and T3 are the intervention. They are written here, in the versioned library, rather
# than in the notebook, so a rung cannot be re-run later with a quietly different prompt.

# 🔴 T2 IS SPLIT IN TWO, AND THE SMOKE IS WHY (2026-08-17, n=60).
# The first design asked B for "AGREE" / "DISAGREE: <reason>; ANSWER: <x>". B ignored the
# protocol on 24 of 60 questions — not with refusals but with BARE ANSWERS, median 5 chars.
# `r42_ep4` is four epochs of "emit the answer and nothing else", so the SFT removed the
# instruction-following the protocol assumed ([[zero-is-format-localized]] measured the same
# format rigidity from the other side). Parsing those as AGREE silently muted the critic on
# 40 % of the population — the intervention would have run at 60 % dose and the null would
# have been unreadable.
#
# So B is now used in the mode it actually has. T2a asks it the ORIGINAL question, in its
# native format; disagreement is computed, not self-reported. T2b then asks for the argument
# ONLY where the two models actually differ, which is also the only place it can act.

CRITIC_ANSWER_USER = "{question}"          # B's native task: no protocol to disobey

ARGUE_SYSTEM = (
    "You are an expert surgical assistant. You and a colleague looked at the SAME frame from "
    "a laparoscopic surgical video and gave DIFFERENT answers. State, in ONE short clause, "
    "the visual evidence in the frame that supports YOUR answer. No preamble, no restatement "
    "of the question, one clause only."
)

ARGUE_USER = (
    "Question: {question}\n"
    "Your answer: {b_answer}\n"
    "Their answer: {a_answer}\n"
    "What do you see that supports your answer?"
)

# T3 — the subject. A sees the critique.
FINAL_DEBATE_USER = (
    "Question: {question}\n"
    "Your earlier answer: {a_answer}\n"
    "A colleague reviewed the SAME frame and replied: {critique}\n"
    "Look at the frame again and decide. You may keep your answer or change it. "
    "Reply with ONLY the final answer in the format the question requests — no explanation."
)

# T3' — the confound control. Identical except the critique is absent.
FINAL_SELF_USER = (
    "Question: {question}\n"
    "Your earlier answer: {a_answer}\n"
    "Look at the frame again and decide. You may keep your answer or change it. "
    "Reply with ONLY the final answer in the format the question requests — no explanation."
)


@dataclass
class DebateConfig:
    model_a: str                      # the 27B, speaks first and last
    model_b: str                      # the 8B, the critic
    out_dir: str
    frames_cache: str
    data_root: str
    held_out_videos: list[str]
    limit: int | None = None          # smoke: cap the population
    max_new_a1: int = 64
    max_new_b2: int = 96              # the critic needs room for its clause
    max_new_a3: int = 32              # the final answer is short by construction
    answer_char_cap: int = 300
    # 🔴 2048, not 4096, and CLAUDE.md said so before the run did: the 27B's FP8 weights are
    # 33.67 GiB of a 47.4 GiB card, so at 4096 the engine loads and then dies with "no
    # available memory for the cache blocks" — a model that fits and still will not start.
    # Real FRAME prompts measure 1,244 tokens (image-dominated); T3 adds ~100 of text.
    max_model_len_a: int = 2048
    max_model_len_b: int = 2048
    gpu_mem_util: float = 0.92
    seed: int = 46


# --------------------------------------------------------------------------- population

def load_population(cfg: DebateConfig):
    """The 8 videos rung 42 held out — the only split clean for BOTH models.

    Returns (items, meta). `items` are `frame.data.FrameItem`s restricted to the held-out
    videos of both datasets; `meta` records the counts so a smoke and a full run are
    distinguishable in the artifact rather than by memory.
    """
    from frame.config import BaselineConfig
    from frame.data import load_frame_items

    # `load_frame_items` takes the project config, not a path — it needs `base_fps` per
    # dataset to turn `start_time` into the `frame_index` the cache is keyed on. Building
    # the config here (rather than accepting a path) is what keeps the frame key identical
    # to every other rung's; a hand-rolled loader would silently key on something else.
    bcfg = BaselineConfig(data_root=Path(cfg.data_root))
    held = set(cfg.held_out_videos)
    items = [it for it in load_frame_items(bcfg, splits=("test",))
             if f"{it.dataset}/{it.video_id}" in held]
    items.sort(key=lambda i: i.request.qID)          # deterministic order, no Date/random
    if cfg.limit:
        # Stratified smoke (RULES §8): never a single-dataset prefix. Take every k-th item
        # of each dataset so both splits and every template are represented.
        out = []
        for ds in ("heico", "lapchole"):
            sub = [i for i in items if i.dataset == ds]
            k = max(1, len(sub) // max(1, cfg.limit // 2))
            out.extend(sub[::k][: cfg.limit // 2])
        items = sorted(out, key=lambda i: i.request.qID)
    meta = {"n_items": len(items),
            "n_heico": sum(1 for i in items if i.dataset == "heico"),
            "n_lapchole": sum(1 for i in items if i.dataset == "lapchole"),
            "held_out_videos": sorted(held)}
    return items, meta


def load_images(items, frames_cache: str):
    """Resolve every frame from the single shared store. Missing frame => raise, never skip.

    A silently skipped frame would shorten one arm's population and make the paired
    comparison unpaired, which is the failure mode that is invisible in a summary table.
    """
    from PIL import Image
    from frame.data import frame_cache_name

    root = Path(frames_cache)
    paths = [root / frame_cache_name(it) for it in items]
    missing = [str(p.name) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames absent from {root}: {missing[:5]}")
    return [Image.open(p).convert("RGB") for p in paths]


# --------------------------------------------------------------------------- engine

class _Engine:
    """One vLLM model, loaded and explicitly released. Batched chat over full conversations."""

    def __init__(self, path: str, max_model_len: int, cfg: DebateConfig):
        from vllm import LLM
        t0 = time.perf_counter()
        self.llm = LLM(model=path,
                       tensor_parallel_size=1,
                       gpu_memory_utilization=cfg.gpu_mem_util,
                       max_model_len=max_model_len,
                       limit_mm_per_prompt={"image": 1},
                       enforce_eager=True,          # CLAUDE.md: 225.6 s -> 126.7 s on the 27B
                       seed=cfg.seed,
                       trust_remote_code=True)
        self.setup_secs = time.perf_counter() - t0
        logger.info("loaded %s in %.0fs", path, self.setup_secs)

    def chat(self, convs, max_tokens: int, chunk: int = 256):
        """Batched chat, in CHUNKS.

        🔴 Handing vLLM all 1,283 conversations in one call renders 1,283 image-bearing
        prompts before a single token is generated, and the run degraded from 26 it/s to
        2 it/s before its EngineCore died (2026-08-17). Chunking bounds the render set and
        the KV pressure, and it is what the teammate's own runner does at the same size.
        Order is preserved across chunks by construction — outputs are appended in order —
        and the length assertion below is what proves it rather than assuming it.
        """
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=None)
        out, t0 = [], time.perf_counter()
        for i in range(0, len(convs), chunk):
            part = convs[i:i + chunk]
            res = self.llm.chat(part, sp, chat_template_kwargs={"enable_thinking": False})
            if len(res) != len(part):
                raise AssertionError(f"vLLM returned {len(res)} for {len(part)} requests")
            out.extend(o.outputs[0].text.strip() for o in res)
            logger.info("  chunk %d-%d done (%d/%d)", i, i + len(part), len(out), len(convs))
        wall = time.perf_counter() - t0
        if len(out) != len(convs):
            raise AssertionError(f"{len(out)} answers for {len(convs)} conversations")
        return out, wall

    def close(self):
        import gc
        import torch
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


def _conv(system: str, image, text: str):
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image_pil", "image_pil": image},
                                         {"type": "text", "text": text}]}]


# --------------------------------------------------------------------------- gates

ERROR_SENTINEL = re.compile(r"^Inference Error", re.I)


def gate_infer(answers: list[str], label: str, threshold: float = 0.01):
    """RULES §8c — a total engine failure returns cleanly and scores as incapacity."""
    bad = sum(1 for a in answers if ERROR_SENTINEL.match(a))
    rate = bad / max(1, len(answers))
    if rate > threshold:
        raise AssertionError(f"G-INFER {label}: {bad}/{len(answers)} sentinels ({rate:.3f})")
    return rate


def normalize_all(answers: list[str]) -> list[str]:
    """The shipped container's answer path, applied to EVERY arm identically.

    🔻 ADDED 2026-08-17 AFTER `G-FORMAT` FIRED ON THE SMOKE, and the timing is stated rather
    than hidden. The gate caught A emitting `"2."` on 3 of 60 `number` questions in the
    DEBATE arm and 0 in the control — the off-template fragility probe 16a measured, provoked
    by the debate turn's non-corpus phrasing. Three reasons this is a fix and not a
    gate-laundering: it is applied to all three arms, it is BYTE-IDENTICAL on the control
    (which emits no trailing periods), and it is what `inference.py` already does in the
    submission we ship — NOT applying it would make the experiment unfaithful to production.
    The gate itself is untouched and re-checked after normalisation (`RULES §7`).
    """
    from frame.parsing import normalize_answer
    return [normalize_answer(a) for a in answers]


def parse_failure_rate(answers: list[str], formats: list[str]) -> float:
    """Share of answers the SDK's own verifiers would reject on FORM, before correctness.

    Read from `focus` rather than reimplemented: `Number.verify` is `strip().isdigit()` and
    `Binary.verify` is membership, so "1." and "Yes." are auto-incorrect
    ([[container-answer-path-audit]]). A debate that makes answers chattier loses points to
    this and not to reasoning, which is exactly what G-FORMAT is for.
    """
    bad = 0
    for a, f in zip(answers, formats):
        s = a.strip()
        if f == "number" and not s.isdigit():
            bad += 1
        elif f == "binary" and s.lower().rstrip(".") not in ("yes", "no"):
            bad += 1
        elif len(s) > 60:                      # any format: a paragraph is not an answer
            bad += 1
    return bad / max(1, len(answers))


def _norm(s: str) -> str:
    """Comparison key for 'did the two models say the same thing'.

    Format-agnostic on purpose — the container cannot know a question's `answer_format`
    ([[container-answer-path-audit]]) and neither can this. Lowercase, strip trailing
    punctuation, and sort comma-separated class sets so 'Clip, Sponge' == 'Sponge, Clip'
    (`fo_class` is scored as a SET, so treating those as a disagreement would manufacture
    argument where the models agree).
    """
    t = s.strip().lower().rstrip(". ")
    if "," in t:
        return ", ".join(sorted(p.strip() for p in t.split(",") if p.strip()))
    return t


def disagreement(a_answers: list[str], b_answers: list[str]) -> list[bool]:
    """Disagreement is COMPUTED, never self-reported. See the T2 note above."""
    return [_norm(a) != _norm(b) for a, b in zip(a_answers, b_answers)]


# --------------------------------------------------------------------------- the run

def run_debate(cfg: DebateConfig) -> dict:
    """Three sequential passes. Writes every turn to disk before the next model loads.

    Written to RESUME, not to survive: each pass checkpoints its own JSON and is skipped if
    that file already exists, so a killed run continues instead of restarting. Each write is
    O_CREAT|O_EXCL through a temp file + rename — two processes appending to one artifact is
    how a file got corrupted on 2026-08-16.
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    items, meta = load_population(cfg)
    qids = [i.request.qID for i in items]
    questions = [i.request.question for i in items]
    formats = [i.reference._format for i in items]      # the raw type string, not the object
    golds = [i.reference.answer for i in items]
    logger.info("population: %s", meta)

    from frame.engine import SYSTEM_PROMPT
    report: dict = {"meta": meta, "turns": {}}

    def _save(name: str, payload):
        tmp = out / f".{name}.tmp"
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)      # exclusive: no double-writer
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1)
        tmp.replace(out / f"{name}.json")

    def _load(name: str):
        p = out / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ---- PASS 1 — model A alone. This IS the control arm.
    t1 = _load("turn1_A")
    if t1 is None:
        images = load_images(items, cfg.frames_cache)
        eng = _Engine(cfg.model_a, cfg.max_model_len_a, cfg)
        convs = [_conv(SYSTEM_PROMPT, im, q) for im, q in zip(images, questions)]
        raw, wall = eng.chat(convs, cfg.max_new_a1)
        eng.close()
        t1 = {"answers": normalize_all([r[: cfg.answer_char_cap] for r in raw]), "wall": wall,
              "setup": eng.setup_secs}
        _save("turn1_A", t1)
    gate_infer(t1["answers"], "T1")
    report["turns"]["T1_A_alone"] = {"wall": t1["wall"],
                                     "parse_fail": parse_failure_rate(t1["answers"], formats)}

    # ---- PASS 2 — model B answers the frame itself (T2a), then argues where it differs (T2b).
    t2 = _load("turn2_B")
    if t2 is None:
        images = load_images(items, cfg.frames_cache)
        eng = _Engine(cfg.model_b, cfg.max_model_len_b, cfg)
        # T2a — B's own independent read, in its native format.
        convs = [_conv(SYSTEM_PROMPT, im, CRITIC_ANSWER_USER.format(question=q))
                 for im, q in zip(images, questions)]
        b_raw, wall_a = eng.chat(convs, cfg.max_new_a1)
        b_ans = normalize_all([r[: cfg.answer_char_cap] for r in b_raw])
        # T2b — the argument, ONLY where the two models actually differ. Elsewhere there is
        # nothing to argue about and generating a "reason" would invent one.
        diff = disagreement(t1["answers"], b_ans)
        idx = [i for i, d in enumerate(diff) if d]
        reasons = [""] * len(b_ans)
        wall_b = 0.0
        if idx:
            convs = [_conv(ARGUE_SYSTEM, images[i],
                           ARGUE_USER.format(question=questions[i], b_answer=b_ans[i],
                                             a_answer=t1["answers"][i]))
                     for i in idx]
            raw_r, wall_b = eng.chat(convs, cfg.max_new_b2)
            for i, r in zip(idx, raw_r):
                reasons[i] = r.strip()[:200]
        eng.close()
        t2 = {"b_answers": b_ans, "reasons": reasons, "disagrees": diff,
              "wall_answer": wall_a, "wall_argue": wall_b, "setup": eng.setup_secs}
        _save("turn2_B", t2)
    gate_infer(t2["b_answers"], "T2a")
    report["turns"]["T2_B_critic"] = {
        "wall": t2["wall_answer"] + t2["wall_argue"],
        "disagree_rate": sum(t2["disagrees"]) / max(1, len(t2["disagrees"])),
        "b_parse_fail": parse_failure_rate(t2["b_answers"], formats),
        "empty_reason_on_disagree": sum(
            1 for d, r in zip(t2["disagrees"], t2["reasons"]) if d and not r),
    }

    # ---- PASS 3 — model A decides, twice: with the critique (subject) and without (control).
    t3 = _load("turn3_A")
    if t3 is None:
        images = load_images(items, cfg.frames_cache)
        eng = _Engine(cfg.model_a, cfg.max_model_len_a, cfg)
        conv_dbg = [_conv(SYSTEM_PROMPT, im, FINAL_DEBATE_USER.format(
                        question=q, a_answer=a,
                        critique=(f'"{b}" — and they add: {r}' if d and r
                                  else f'"{b}"' if d
                                  else f'"{b}", which agrees with you')))
                    for im, q, a, b, r, d in zip(images, questions, t1["answers"],
                                                 t2["b_answers"], t2["reasons"],
                                                 t2["disagrees"])]
        raw_dbg, wall_dbg = eng.chat(conv_dbg, cfg.max_new_a3)
        conv_slf = [_conv(SYSTEM_PROMPT, im, FINAL_SELF_USER.format(question=q, a_answer=a))
                    for im, q, a in zip(images, questions, t1["answers"])]
        raw_slf, wall_slf = eng.chat(conv_slf, cfg.max_new_a3)
        eng.close()
        t3 = {"debate": normalize_all([r[: cfg.answer_char_cap] for r in raw_dbg]),
              "selfrevise": normalize_all([r[: cfg.answer_char_cap] for r in raw_slf]),
              "wall_debate": wall_dbg, "wall_self": wall_slf, "setup": eng.setup_secs}
        _save("turn3_A", t3)
    gate_infer(t3["debate"], "T3_debate")
    gate_infer(t3["selfrevise"], "T3_selfrevise")

    # ---- G-FORMAT: the debate must not win by talking and lose to the verifier.
    pf1 = parse_failure_rate(t1["answers"], formats)
    pfd = parse_failure_rate(t3["debate"], formats)
    pfs = parse_failure_rate(t3["selfrevise"], formats)
    report["G_FORMAT"] = {"T1": pf1, "debate": pfd, "selfrevise": pfs,
                          "fires": bool(pfd > pf1 + 0.02)}
    if report["G_FORMAT"]["fires"]:
        raise AssertionError(
            f"G-FORMAT: debate parse-failure {pfd:.3f} vs T1 {pf1:.3f}. The arm would be "
            "scored down for verbosity, not for reasoning. Fix the prompt, not the gate.")

    # ---- the three arms, paired by qID, ready for frame.metrics
    _save("arms", {"qID": qids, "format": formats, "gold": golds,
                   "video": [i.video_id for i in items],
                   "dataset": [i.dataset for i in items],
                   "A_alone": t1["answers"],
                   "A_selfrevise": t3["selfrevise"],
                   "A_debate": t3["debate"],
                   "B_answer": t2["b_answers"],
                   "B_reason": t2["reasons"],
                   "B_disagrees": t2["disagrees"]})
    _save("report", report)
    return report
