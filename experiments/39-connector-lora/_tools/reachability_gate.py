"""Rung 39 — does an EXPLICIT ``--target_modules`` actually put LoRA on the ViT->LLM connector?

Folder-private glue. Importable; a notebook cell calls ``run_both(Config(...))``. Nothing here is
a hand-run launcher.

🔴 **This is NOT the rung.** It is the blocking reachability gate that decides whether the rung
exists at all, and it is deliberately separated so no full arm is paid for before the flag is known
to do something. Adapted from ``experiments/32-aligner-unfreeze/_tools/reachability_smoke.py``;
everything that file learned the hard way is kept verbatim and commented, and only the five changes
listed in ``PLAN.md`` §10 are made.

## Why the question is open

[[the-merger-is-unreachable-by-default]] settled that the merger is unreachable **by default** in
BOTH trainers, and that the cause is structural rather than a defect: ms-swift's ``all-linear`` and
Unsloth's vision regex both require an attention/MLP token in the module path, applied with
``re.fullmatch``, and ``model.visual.merger.linear_fc1`` carries none. Measured:

* ms-swift 4.4.1 — ``720 = 504 llm + 216 vit + 0 aligner`` on BOTH ``--freeze_aligner`` legs
  (``experiments/32-aligner-unfreeze/RESULTS_reachability.csv``);
* Unsloth 2026.8.15 — ``merger 0 / deepstack 0``
  (``experiments/38-gen36-ft-screen/RESULTS_smoke_unsloth.json``).

What is **NOT** measured, in either framework, is whether an EXPLICIT merger name survives the
trainer's own constraining of the target list. That is the one thing this gate establishes.

## 🔴 The landmine this gate is built around

``ms-swift pipelines/train/tuner.py:93``::

    if isinstance(args.target_modules, str):
        return args.target_modules

A **string returns early and is silently ignored**. It works today only because
``--target_modules all-linear`` parses into the LIST ``['all-linear']``. Pass the eight merger names
as one joined string — ``"a,b,c"``, ``"a b c"``, anything a human would naturally type — and training
runs happily, exits ``rc=0``, writes a checkpoint, and adapts NOTHING: the eighth silent no-op in
this repo. Documented at ``experiments/06-vit-lora/_models/vit_lora_train.py:23-26``.

So ``build_argv`` splats the names as **separate argv values** and ``assert_splat`` RAISES unless it
did. That half is provable on a laptop with no GPU. The other half — that the argv shape produced a
real effect — is only provable by running, which is what the legs do.

## What it does, and what it costs

⚠️ **This is NOT zero-GPU.** Each leg runs five real optimiser steps of ``swift sft`` on 32 rows of
A2's own corpus, with A2's recipe inherited verbatim. **Minutes per leg, not seconds** — plus model
load. Cheap against a 2.9 h arm; not free.

Each leg then classifies the produced adapter with rung 27's ``gcov_probe.static_leg`` — already
written, already validated against A2's 720 tensors — and cross-checks the ``adapter_config.json``
ms-swift itself wrote.

**Pre-registered criteria (``PLAN.md`` §4, do not change them to make a number pass — RULES §7):**

* subject leg (``freeze_aligner=False``): ``n_aligner > 0`` AND ``n_orphans == 0`` AND coverage
  PRESERVED (``n_llm == 504``, ``n_vit == 216``, both > 0) AND ``adapter_config.json`` naming all 8
  layers. Sharp expectation ``n_aligner == 16`` (8 Linear layers x (LoRA A + B)); a count that is
  neither 0 nor 16 is a FINDING to read, not something to accept silently.
* control leg (``freeze_aligner=True``): a **BRANCH SELECTOR**, not a pass/fail on ``n_aligner``.
  ``n_aligner == 0`` selects branch ``UNFREEZE`` (the arm adds ``--freeze_aligner false``);
  ``n_aligner > 0`` selects branch ``TARGETS_ONLY`` (the arm keeps A2's ``true``). Both branches are
  pre-declared in ``PLAN.md`` §5. It still fails on a broken trainer, a missing adapter, orphans, or
  lost coverage.

``run_both`` RAISES ``AssertionError`` when the gate does not pass. Raising IS the mechanism:
papermill turns it into a non-zero exit, and the chain reads that exit code and does not train.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

EXP = Path(__file__).resolve().parent.parent

#: 🎯 The eight ``Linear`` layers of the ViT->LLM connector — FOUR merger blocks, not one. DeepStack
#: wires ``deepstack_merger_list.{0,1,2}`` into the first 3 LLM layers, so all four blocks together
#: ARE the ViT<->LLM connection ([[viT-swap-nogo]], [[the-merger-is-unreachable-by-default]]).
#:
#: 🔴 SINGLE SOURCE. ``_models/connector_lora_train.py`` imports this tuple rather than retyping it;
#: two hand-kept copies of eight long dotted names is a typo waiting to become a silent no-op.
MERGER_TARGETS: tuple[str, ...] = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc2",
    "model.visual.deepstack_merger_list.1.linear_fc1",
    "model.visual.deepstack_merger_list.1.linear_fc2",
    "model.visual.deepstack_merger_list.2.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
)

#: What A2 passed. The variable EXTENDS this, it never replaces it (``PLAN.md`` §2).
A2_TARGET_MODULES: tuple[str, ...] = ("all-linear",)

#: A2's own census, from ``experiments/32-aligner-unfreeze/RESULTS_reachability.csv``. The LoRA
#: geometry is data-independent, so a 5-step smoke on 32 rows must reproduce these exactly.
A2_N_LLM = 504
A2_N_VIT = 216

#: 8 Linear layers x (LoRA A + B). The sharp expectation, recorded so a pass is checkable rather
#: than merely non-zero.
EXPECTED_N_ALIGNER = 2 * len(MERGER_TARGETS)

CONTROL_TRAIN = PurePosixPath(
    "/workspace/repo_rodri/experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl"
)

#: The two pre-declared branches the control leg selects between (``PLAN.md`` §5).
BRANCH_UNFREEZE = "UNFREEZE"          # control n_aligner == 0 -> arm adds --freeze_aligner false
BRANCH_TARGETS_ONLY = "TARGETS_ONLY"  # control n_aligner  > 0 -> arm keeps A2's --freeze_aligner true


@dataclass
class Config:
    repo: PurePosixPath = PurePosixPath("/workspace/repo_rodri")
    base_model: PurePosixPath = PurePosixPath("/workspace/models/qwen3-vl-8b")
    model_type: str = "qwen3_vl"
    src: PurePosixPath = CONTROL_TRAIN
    out_root: Path = EXP / "runs"
    run: str = "39_reach"

    #: 🎯 THE VARIABLE. ``all-linear`` PLUS the eight merger names — coverage EXTENDED, never
    #: replaced. Passing the eight names alone would drop the 504 LLM + 216 ViT tensors and produce
    #: a different rung wearing this one's name (``PLAN.md`` §2, hazard H2).
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: A2_TARGET_MODULES + MERGER_TARGETS
    )

    #: 🎯 THE LEG SELECTOR. False = subject leg. True = control leg.
    freeze_aligner: bool = False

    # ── inherited from A2's args.json, do not drift ──────────────────────────
    lora_rank: int = 8
    lora_alpha: int = 32
    #: Declared deviation #5 vs rung 32's smoke, which omitted it: the gate mirrors the ARM's
    #: recipe, not an approximation of it.
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    freeze_vit: bool = False
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 42

    max_steps: int = 5
    rows: int = 32

    #: 🔴 Absolute path, not the bare name. A papermill kernel does not inherit the shell's
    #: activated venv, so ``subprocess.run(["swift", ...])`` raises
    #: ``FileNotFoundError: [Errno 2] ... 'swift'`` seven seconds in, before any GPU work and
    #: before any artifact is written. Measured 2026-08-08 (rung 32).
    #:
    #: ``PurePosixPath`` rather than ``Path`` on purpose: the pod is Linux, and the argv this
    #: builder produces is unit-checked on a Windows laptop where ``Path("/workspace/...")``
    #: stringifies to ``\workspace\...``. The laptop check is the whole proof that the splat is
    #: intact, so the string it checks must be the string the pod will run.
    swift: PurePosixPath = PurePosixPath("/workspace/envs/infer/bin/swift")

    @property
    def leg(self) -> str:
        return "control_frozen" if self.freeze_aligner else "subject_unfrozen"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / f"{self.run}_{self.leg}"


def _slice(cfg: Config) -> Path:
    """A tiny copy of the corpus, so a wiring bug costs a minute rather than an hour."""
    dst = cfg.run_dir / "smoke_train.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(str(cfg.src), encoding="utf-8") as fh:
        dst.write_text("".join(itertools.islice(fh, cfg.rows)), encoding="utf-8")
    return dst


def assert_splat(argv: list[str], n_expected: int) -> list[str]:
    """GATE — ``--target_modules`` must be followed by ``n_expected`` SEPARATE values. RAISES.

    🔴 ``ms-swift pipelines/train/tuner.py:93``::

        if isinstance(args.target_modules, str):
            return args.target_modules

    A string returns early and is **silently ignored**: the whole ``all-linear`` expansion, including
    the ``freeze_vit`` branch that puts LoRA on the vision tower, never runs. Training then completes
    with ``rc=0``, writes a checkpoint, and adapts nothing.

    So this refuses every shape that collapses to one token — a comma-join, a space-join, a Python
    list stringified into one argument — and it refuses a second ``--target_modules`` occurrence,
    which would let argparse keep only the last one. Zero GPU: this is checkable on a laptop, and it
    is checked on a laptop.

    ⚠️ The count check ALONE is not enough, and that hole was found by testing it: a config holding
    ``("a,b,c",)`` declares one target and passes one value, so the arithmetic agrees with itself
    while the value is exactly the string ms-swift discards. The per-VALUE check below is what
    actually closes the landmine; the count check only closes the "someone joined nine into one"
    half.
    """
    _JOINERS = (",", " ", "\t", "[", "]", "'", '"')
    n_flags = argv.count("--target_modules")
    if n_flags != 1:
        raise AssertionError(
            f"--target_modules appears {n_flags} times in the argv; it must appear exactly once "
            "(argparse keeps only the last occurrence, so a duplicate silently discards the first)"
        )
    i = argv.index("--target_modules")
    vals: list[str] = []
    j = i + 1
    while j < len(argv) and not argv[j].startswith("--"):
        vals.append(argv[j])
        j += 1
    if len(vals) != n_expected:
        raise AssertionError(
            f"--target_modules is followed by {len(vals)} value(s), expected {n_expected}: {vals}. "
            "ms-swift's tuner.py:93 returns EARLY on a string target_modules and silently ignores "
            "it, so a joined value trains happily, exits rc=0 and adapts nothing."
        )
    joined = [v for v in vals if any(ch in v for ch in _JOINERS)]
    if joined:
        raise AssertionError(
            f"--target_modules value(s) {joined} carry a separator {list(_JOINERS)} — that is a "
            "JOINED list, which is the exact shape ms-swift's tuner.py:93 discards in silence. "
            "Pass one module name per argv value."
        )
    return vals


def build_argv(cfg: Config, data: Path) -> list[str]:
    """The gate's ``swift sft`` command. Pure — no mkdir, no IO — so it is checkable anywhere.

    Kept verbatim from rung 32 and load-bearing:

    * ``--tuner_type``, NOT ``--train_type`` — renamed on ms-swift 4.4.1 (``base_args.py:93``);
    * the absolute ``swift`` path (see ``Config.swift``).

    Changed, and declared in ``PLAN.md`` §10:

    * ``["--target_modules", *cfg.target_modules]`` — the splat (deviation #1, THE variable);
    * **no ``--adapters``** — the gate builds the LoRA fresh, exactly as the arm will, because a
      resumed adapter carries its own ``target_modules`` in ``adapter_config.json`` and the gate
      would then be measuring rung 21's LoRA geometry instead of rung 39's (deviation #4);
    * ``--lora_dropout`` emitted, so the gate mirrors the arm's recipe (deviation #5).
    """
    argv = [
        str(cfg.swift), "sft",
        "--model", str(cfg.base_model),
        "--model_type", cfg.model_type,
        "--dataset", str(data),
        # 🔻 `tuner_type`, not `train_type` — renamed on ms-swift 4.4.1 (base_args.py:93)
        "--tuner_type", "lora",
        "--lora_rank", str(cfg.lora_rank),
        "--lora_alpha", str(cfg.lora_alpha),
        "--lora_dropout", str(cfg.lora_dropout),
        # 🎯 THE VARIABLE, splatted. NEVER " ".join(...), NEVER ",".join(...), NEVER one value.
        "--target_modules", *cfg.target_modules,
        "--torch_dtype", cfg.torch_dtype,
        "--attn_impl", cfg.attn_impl,
        "--gradient_checkpointing", "true",
        "--freeze_vit", str(cfg.freeze_vit).lower(),
        # 🎯 the leg selector
        "--freeze_aligner", str(cfg.freeze_aligner).lower(),
        "--per_device_train_batch_size", str(cfg.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
        "--max_grad_norm", str(cfg.max_grad_norm),
        "--lr_scheduler_type", cfg.lr_scheduler_type,
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--learning_rate", str(cfg.learning_rate),
        "--max_steps", str(cfg.max_steps),
        "--logging_steps", "1",
        "--seed", str(cfg.seed),
        "--output_dir", str(cfg.run_dir / "ckpt"),
    ]
    assert_splat(argv, len(cfg.target_modules))
    return argv


def _adapter_config_names(ckpt_dir: Path) -> dict:
    """Which of the 8 merger names ms-swift itself recorded in ``adapter_config.json``.

    A reading INDEPENDENT of rung 27's prefix classifier: the classifier answers "where did the
    produced tensors land", this answers "what did the trainer believe it was targeting". Two
    independent readings, because one instrument agreeing with itself is not evidence.

    ms-swift may serialise ``target_modules`` as a list OR as a single expanded regex string
    (``get_multimodal_target_regex``), so the shape is resolved before matching — a shape
    assumption here would turn a real pass into a false alarm.

    🔻 **Corrected 2026-08-13, AFTER the first run and WITHOUT touching the criterion.** The
    original mitigation anticipated the regex shape but searched it as a substring, which cannot
    hit: ms-swift emits an *escaped* pattern, so the literal ``model.visual.merger.linear_fc1``
    does not occur inside ``model\\.visual\\.merger(?=\\.).*\\.(linear_fc2|linear_fc1)``. The
    first run therefore reported 0 of 8 named while the adapter demonstrably carried all 16
    merger tensors — a false negative in the cross-check, not a scientific failure.
    Matching the pattern against each name is the correct reading and a STRICTER one than the
    substring test it replaces: a coincidental substring can pass, ``re.fullmatch`` cannot.
    The criterion is unchanged — ms-swift's own config must cover all 8 merger layers.
    """
    path = Path(ckpt_dir) / "adapter_config.json"
    if not path.exists():
        return {"adapter_config": None, "found": [], "missing": list(MERGER_TARGETS)}
    raw = json.loads(path.read_text(encoding="utf-8")).get("target_modules")
    if isinstance(raw, str):
        found = [name for name in MERGER_TARGETS if re.fullmatch(raw, name)]
    else:
        blob = json.dumps(raw)
        found = [name for name in MERGER_TARGETS if name in blob]
    return {
        "adapter_config": str(path),
        "target_modules_raw": raw,
        "found": found,
        "missing": [name for name in MERGER_TARGETS if name not in found],
    }


def _leg(cfg: Config) -> dict:
    """Run one leg: 5 optimiser steps, then classify the adapter it produced."""
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))
    sys.path.insert(0, str(cfg.repo / "experiments" / "27-vit-lr-decouple" / "_tools"))
    from gcov_probe import static_leg  # noqa: PLC0415 — pod-only dependency

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(cfg, _slice(cfg))
    # written BEFORE the subprocess runs: if the trainer dies, the exact command still exists
    (cfg.run_dir / "argv.txt").write_text(" ".join(map(shlex.quote, argv)), encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # the console script's shebang resolves against PATH; a papermill kernel may not carry
    # the venv's bin, so put it in front explicitly
    env["PATH"] = f"{cfg.swift.parent}:{env.get('PATH', '')}"
    log = cfg.run_dir / "train.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)

    out: dict = {
        "leg": cfg.leg,
        "freeze_aligner": cfg.freeze_aligner,
        "returncode": proc.returncode,
        "run_dir": str(cfg.run_dir),
        "log": str(log),
        "n_target_modules": len(cfg.target_modules),
    }

    # 🔴 the FILE, not the checkpoint directory. `gcov_probe.static_leg` opens what it is given
    # (`_safetensors_keys(Path(adapter))`) and only globs when handed None, so a directory raises
    # `IsADirectoryError` AFTER the training has already been paid for.
    ckpts = sorted((cfg.run_dir / "ckpt").glob("*/checkpoint-*/adapter_model.safetensors"))
    if proc.returncode != 0:
        out["ok"] = False
        out["reason"] = f"trainer failed (rc={proc.returncode}); read {log}"
        return out
    if not ckpts:
        out["ok"] = False
        out["reason"] = "no adapter_model.safetensors written — the leg produced nothing to classify"
        return out

    adapter = ckpts[-1]
    cov = static_leg(adapter)
    out["adapter"] = str(adapter)
    out["gcov"] = cov
    out.update({
        "total_tensors": cov["n_trainable"],
        "n_llm": cov["n_llm"],
        "n_vit": cov["n_vit"],
        "n_aligner": cov["n_aligner"],
        "n_orphans": cov["n_orphans"],
    })
    out["adapter_config"] = _adapter_config_names(adapter.parent)
    out["ok"] = True
    return out


def _apply_criteria(subject: dict, control: dict) -> dict:
    """The PRE-REGISTERED criteria of ``PLAN.md`` §4, applied. Reports; ``run_both`` raises."""
    failures: list[str] = []

    for leg in (subject, control):
        name = leg["leg"]
        if not leg.get("ok"):
            failures.append(f"{name}: {leg.get('reason')}")
            continue
        # structural criteria — both legs, always
        if leg["n_orphans"] != 0:
            failures.append(
                f"{name}: n_orphans={leg['n_orphans']} != 0 — a trainable tensor matching none of "
                "the registered prefixes is dropped from the multimodal optimiser with NO error"
            )
        if leg["n_llm"] != A2_N_LLM or leg["n_vit"] != A2_N_VIT:
            failures.append(
                f"{name}: coverage NOT preserved — n_llm={leg['n_llm']} (A2 {A2_N_LLM}), "
                f"n_vit={leg['n_vit']} (A2 {A2_N_VIT}). The explicit targets were meant to EXTEND "
                "all-linear, not replace it; an adapter that reached the merger and dropped the "
                "LLM and ViT legs is a different rung wearing this one's name (PLAN.md §2)"
            )

    if subject.get("ok"):
        # the reachability criterion itself — subject leg only
        if subject["n_aligner"] <= 0:
            failures.append(
                "subject_unfrozen: n_aligner=0 — the connector is UNREACHABLE even when named "
                "explicitly. That is a real, publishable answer (PLAN.md §9 shape 1), not a bug to "
                "work around: do NOT relax this criterion, record it and stop."
            )
        elif subject["n_aligner"] != EXPECTED_N_ALIGNER:
            failures.append(
                f"subject_unfrozen: n_aligner={subject['n_aligner']}, expected "
                f"{EXPECTED_N_ALIGNER} (8 Linear layers x LoRA A+B). A count that is neither 0 nor "
                f"{EXPECTED_N_ALIGNER} is a FINDING to be READ before the arm launches, not "
                "something to accept silently (RULES §7)"
            )
        missing = subject.get("adapter_config", {}).get("missing", list(MERGER_TARGETS))
        if missing:
            failures.append(
                f"subject_unfrozen: adapter_config.json does not name {len(missing)} of the 8 "
                f"merger layers: {missing}. The prefix classifier and ms-swift's own serialised "
                "config must agree; one instrument agreeing with itself is not evidence"
            )

    branch = None
    if control.get("ok"):
        branch = BRANCH_UNFREEZE if control["n_aligner"] == 0 else BRANCH_TARGETS_ONLY

    return {
        "passed": not failures,
        "failures": failures,
        # 🔑 the control leg is a BRANCH SELECTOR, not a pass/fail on n_aligner. Both branches are
        # pre-declared in PLAN.md §5, so neither can become a post-hoc choice.
        "branch": branch,
        "arm_freeze_aligner": (False if branch == BRANCH_UNFREEZE else True) if branch else None,
        "sharp_expectation_met": (
            subject.get("n_aligner") == EXPECTED_N_ALIGNER if subject.get("ok") else None
        ),
        "coverage_preserved": (
            subject.get("n_llm") == A2_N_LLM and subject.get("n_vit") == A2_N_VIT
            if subject.get("ok") else None
        ),
    }


RESULTS_CSV = EXP / "RESULTS_reachability39.csv"
RESULTS_JSON = EXP / "RESULTS_reachability39.json"


def run_both(cfg: Config | None = None) -> dict:
    """Run BOTH legs, apply the pre-registered criteria, persist, and RAISE on failure.

    Both legs in ONE call — and one notebook — so the chain needs a single papermill invocation and
    reads a single exit code (declared deviation #6; rung 32 ran one leg per notebook run).

    The results land at the **experiment root**, OUTSIDE ``runs/``: ``runs/`` is gitignored, and two
    earlier pushes were lost to committing into it. Those two files are what the chain commits.

    RAISES ``AssertionError`` when the gate does not pass. That is the mechanism, not an accident:
    papermill turns the raise into a non-zero exit and the chain reads it and does not train
    (RULES §7 — a gate that fires is a FINDING; never disable one to get past it).
    """
    import pandas as pd  # noqa: PLC0415 — pod-only dependency

    cfg = cfg or Config()
    subject = _leg(replace(cfg, freeze_aligner=False))
    control = _leg(replace(cfg, freeze_aligner=True))
    verdict = _apply_criteria(subject, control)

    rows = []
    for leg in (subject, control):
        rows.append({
            "leg": leg["leg"],
            "freeze_aligner": str(leg["freeze_aligner"]).lower(),
            "returncode": leg["returncode"],
            "n_target_modules": leg["n_target_modules"],
            "total_tensors": leg.get("total_tensors"),
            "n_llm": leg.get("n_llm"),
            "n_vit": leg.get("n_vit"),
            "n_aligner": leg.get("n_aligner"),
            "n_orphans": leg.get("n_orphans"),
            "adapter_config_named": len(leg.get("adapter_config", {}).get("found", [])),
            "ok": leg.get("ok", False),
            "note": leg.get("reason", ""),
        })
    df = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    result = {
        "passed": verdict["passed"],
        "branch": verdict["branch"],
        "arm_freeze_aligner": verdict["arm_freeze_aligner"],
        "sharp_expectation_met": verdict["sharp_expectation_met"],
        "coverage_preserved": verdict["coverage_preserved"],
        "failures": verdict["failures"],
        "expected_n_aligner": EXPECTED_N_ALIGNER,
        "a2_census": {"n_llm": A2_N_LLM, "n_vit": A2_N_VIT, "n_aligner": 0},
        "target_modules": list(cfg.target_modules),
        "legs": {"subject": subject, "control": control},
    }
    RESULTS_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(df.to_string(index=False), flush=True)
    print(f"\nbranch: {verdict['branch']}  ->  arm --freeze_aligner "
          f"{str(verdict['arm_freeze_aligner']).lower()}", flush=True)
    print(f"wrote {RESULTS_CSV}\nwrote {RESULTS_JSON}", flush=True)

    if not verdict["passed"]:
        raise AssertionError(
            "RUNG 39 REACHABILITY GATE FAILED — the arm does NOT run.\n  - "
            + "\n  - ".join(verdict["failures"])
            + f"\nEvidence: {RESULTS_CSV} and {RESULTS_JSON}. Commit them; they are the result."
        )
    return result


__all__ = [
    "A2_N_LLM",
    "A2_N_VIT",
    "A2_TARGET_MODULES",
    "BRANCH_TARGETS_ONLY",
    "BRANCH_UNFREEZE",
    "Config",
    "EXPECTED_N_ALIGNER",
    "MERGER_TARGETS",
    "RESULTS_CSV",
    "RESULTS_JSON",
    "assert_splat",
    "build_argv",
    "run_both",
]
