"""Rung 54 held-out scoring harness.

This is an importable library.  A notebook or a short ``python -c`` cell calls
``score_models``; the module is not a launcher.  It deliberately composes rung 47's
held-out instrument and rung 15's count parser instead of growing another evaluator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

TARGET_42_EP4 = 0.6744
VALIDATION_TOLERANCE = 0.01
PARSE_STATUSES = ("bare_int", "structured", "salvaged", "malformed")

REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_import_paths(repo_root: Path = REPO_ROOT) -> None:
    """Expose the vendored SDK and the three experiment-private libraries we reuse."""
    paths = (
        repo_root / "src",
        repo_root / "vendor" / "orena-focus" / "src",
        repo_root / "experiments" / "15-count-target" / "_models",
        repo_root / "experiments" / "23-backbone-screen" / "_tools",
        repo_root / "experiments" / "45-gen36-data-and-reg" / "_tools",
        repo_root / "experiments" / "47-epochs-vs-corpus" / "_tools",
    )
    for path in paths:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _count_target():
    ensure_import_paths()
    import count_target

    return count_target


def _rung47():
    ensure_import_paths()
    import eval_arm47

    return eval_arm47


def _rung45():
    ensure_import_paths()
    import eval_arm45

    return eval_arm45


def make_logged_count_postprocess(records: list[dict]):
    """Apply rung 15's parser only when rung 15 identifies a count question."""
    ct = _count_target()

    def postprocess(answer: str, question: str) -> str:
        result = ct.parse_count(answer, is_count=ct.is_count_question(question))
        records.append(
            {
                "call_index": len(records),
                "question": question,
                "raw": result.raw,
                "answer": result.answer,
                "status": result.status,
            }
        )
        return result.answer

    return postprocess


def parse_status_histogram(records: Iterable[dict]) -> dict[str, int]:
    """Return the four count statuses; non-count ``passthrough`` calls stay outside."""
    rows = list(records)
    return {status: sum(row.get("status") == status for row in rows) for status in PARSE_STATUSES}


def assert_control_parser_identity(histogram: dict[str, int]) -> None:
    """The rung-42 control must already emit bare integers for every count question."""
    if histogram.get("bare_int", 0) <= 0 or any(
        histogram.get(status, 0) for status in PARSE_STATUSES if status != "bare_int"
    ):
        raise AssertionError(
            "control parser is not an identity: expected count statuses to be all bare_int, "
            f"got {histogram}"
        )


@dataclass(frozen=True)
class ModelSpec:
    label: str
    merged_dir: Path

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        if "=" not in value:
            raise ValueError(f"model spec must be LABEL=MERGED_DIR, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError(f"model spec must be LABEL=MERGED_DIR, got {value!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"merged checkpoint is not a directory: {path}")
        if not (path / "config.json").is_file():
            raise AssertionError(f"merged checkpoint has no config.json: {path}")
        return cls(label=label.strip(), merged_dir=path)


@dataclass
class HarnessConfig:
    repo_root: Path = REPO_ROOT
    work_root: Path = Path("/mnt/storage/uaq_user/rung54/harness")
    data_root: Path = Path("/mnt/storage/uaq_user/orena-data")
    frames_cache: Path = Path("/mnt/storage/uaq_user/frames_cache")
    control_42_csv: Path = Path(
        "/mnt/storage/uaq_user/rung47/runs/47_a2_ep5_v1/controls/42_ep4_results.csv"
    )
    max_pixels: int = 1280 * 720
    seed: int = 42

    @property
    def split_json(self) -> Path:
        return self.repo_root / "experiments" / "42-merged-corpus" / "RESULTS_split_42.json"

    def eval_config(self, spec: ModelSpec):
        r45 = _rung45()
        return r45.EvalConfig(
            arm="R0",
            merged_dir=str(spec.merged_dir),
            out_dir=str(self.work_root / "eval"),
            run_name=f"54_{spec.label}_heldout",
            eval_set=r45.PRIMARY_EVAL,
            data_root=str(self.data_root),
            repo_root=str(self.repo_root),
            split_json=str(self.split_json),
            frames_cache=str(self.frames_cache),
            max_pixels=self.max_pixels,
            seed=self.seed,
            use_vllm=False,
        )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _assert_parser_is_only_inference_change(cfg_eval) -> None:
    if cfg_eval.answer_postprocess is None:
        raise AssertionError("count parser is not wired into the prediction path")
    if cfg_eval.n_samples != 1 or cfg_eval.enhance is not None or cfg_eval.aux_view is not None:
        raise AssertionError("an inference control besides the required count parser moved")


def _run_predictions(cfg: HarnessConfig, spec: ModelSpec, split: dict) -> tuple[Path, list[dict]]:
    """Run rung 47's HF/GenericVLM path with rung 15's parser at ``predict``."""
    r45 = _rung45()
    eval_cfg = cfg.eval_config(spec)
    results_path = Path(eval_cfg.out_dir) / eval_cfg.run_name / "results.csv"
    parse_path = Path(eval_cfg.out_dir) / eval_cfg.run_name / "parse_log.jsonl"

    if results_path.exists():
        if not parse_path.exists():
            raise AssertionError(
                f"{results_path} exists without {parse_path}; refusing an unproven parser run"
            )
        log.info("%s already answered -> %s (no GPU)", spec.label, results_path)
        return results_path, _read_jsonl(parse_path)

    records: list[dict] = []
    cfg_eval = r45.build_baseline_config(eval_cfg)
    cfg_eval.answer_postprocess = make_logged_count_postprocess(records)
    _assert_parser_is_only_inference_change(cfg_eval)

    # This is the exact non-vLLM engine selected by eval_arm45.score, which rung 47
    # calls.  The only intentional difference is the required answer_postprocess.
    from screen_engine import GenericVLMEngine
    from frame.run import run_baseline

    cfg_eval.engine_factory = GenericVLMEngine
    try:
        run_baseline(cfg_eval, video_filter=split["held_videos"])
    finally:
        _write_jsonl(parse_path, records)
    return r45.arm_results_csv(eval_cfg), records


def write_summary(
    path: Path,
    *,
    label: str,
    model_dir: Path,
    bucket_mean: float,
    histogram: dict[str, int],
    target: float = TARGET_42_EP4,
    tolerance: float = VALIDATION_TOLERANCE,
) -> dict:
    delta = float(bucket_mean) - target
    payload = {
        "label": label,
        "model_dir": str(model_dir),
        "bucket_mean": float(bucket_mean),
        "target_42_ep4": target,
        "delta_vs_target": delta,
        "tolerance": tolerance,
        "verdict": "GREEN" if abs(delta) <= tolerance else "RED",
        "parse_status_histogram": histogram,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@contextmanager
def rung42_epoch4_step(r47_module):
    """Let rung 47's reusable merge helpers address rung 42's real ep4 step."""
    previous = r47_module.EPOCH_STEPS[4]
    r47_module.EPOCH_STEPS[4] = 4848
    try:
        yield
    finally:
        r47_module.EPOCH_STEPS[4] = previous


def reclaim_model_dir(path: Path) -> None:
    """Delete one explicitly supplied merged checkpoint after its answers are durable."""
    import shutil

    path = Path(path)
    if not (path / "config.json").is_file():
        raise AssertionError(f"refusing to reclaim a directory without config.json: {path}")
    shutil.rmtree(path)
    log.info("reclaimed merged checkpoint %s", path)


def score_models(
    model_args: Iterable[str],
    *,
    control_labels: Iterable[str] = (),
    cfg: HarnessConfig | None = None,
    reclaim_models: bool = False,
) -> list[dict]:
    """Score ``LABEL=MERGED_DIR`` models on rung 42's held-out 1,283 questions."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise AssertionError("CUDA_VISIBLE_DEVICES must be exactly '0' for rung 54 scoring")

    cfg = cfg or HarnessConfig()
    ensure_import_paths(cfg.repo_root)
    r45, r47, ct = _rung45(), _rung47(), _count_target()
    specs = [ModelSpec.parse(value) for value in model_args]
    if not specs:
        raise ValueError("at least one LABEL=MERGED_DIR model is required")

    probe_eval = cfg.eval_config(specs[0])
    split = r45.held_out_split(probe_eval)
    r45.assert_cache_covers(probe_eval, split["held_items"])
    ct.assert_count_regex_matches_formats(split["held_items"])

    from frame import ledger

    gold = ledger.gold_from_frame_parquets(cfg.data_root)
    controls = set(control_labels)
    summaries: list[dict] = []
    for spec in specs:
        try:
            results_csv, records = _run_predictions(cfg, spec, split)
            histogram = parse_status_histogram(records)
            if spec.label in controls:
                assert_control_parser_identity(histogram)

            scored = r47.score_on_heldout(
                probe_eval,
                results_csv,
                split["held_qids"],
                gold,
                spec.label,
            )
            ct.assert_safe_answer_not_modal(gold, scored["res"])
            summary = write_summary(
                cfg.work_root / "summaries" / f"{spec.label}.json",
                label=spec.label,
                model_dir=spec.merged_dir,
                bucket_mean=scored["strat"]["bucket_mean"],
                histogram=histogram,
            )

            if cfg.control_42_csv.is_file():
                ci = r47.paired_ci_vs_42(
                    results_csv, cfg.control_42_csv, split["held_qids"], n_boot=4000
                )
                ci_path = cfg.work_root / "summaries" / f"{spec.label}_paired_vs_42.csv"
                ci_path.parent.mkdir(parents=True, exist_ok=True)
                ci.to_csv(ci_path, index=False)
                summary["paired_ci_csv"] = str(ci_path)
                (cfg.work_root / "summaries" / f"{spec.label}.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
            summaries.append(summary)
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            if reclaim_models:
                reclaim_model_dir(spec.merged_dir)
    return summaries


def validate_control(
    *,
    adapter_dir: Path = Path("/mnt/storage/uaq_user/rung19/r42_ep4_adapter"),
    base_model: Path = Path(
        "/mnt/storage/uaq_user/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
        "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    ),
    cfg: HarnessConfig | None = None,
) -> dict:
    """Merge and score the archived rung-42 epoch-4 adapter, then reclaim our merge."""
    adapter_dir = Path(adapter_dir)
    base_model = Path(base_model)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"control adapter is not a directory: {adapter_dir}")
    for name in ("adapter_config.json", "adapter_model.safetensors", "args.json"):
        if not (adapter_dir / name).is_file():
            raise AssertionError(f"control adapter is incomplete: no {adapter_dir / name}")
    if not (base_model / "config.json").is_file():
        raise FileNotFoundError(f"base model is not loadable: {base_model}")

    args = json.loads((adapter_dir / "args.json").read_text(encoding="utf-8"))
    recorded_out = str(args.get("output_dir", ""))
    if "experiments/42-merged-corpus/runs/42_merged_v1/ckpt" not in recorded_out:
        raise AssertionError(
            "the archived adapter does not identify rung 42's training output: "
            f"{recorded_out!r}"
        )

    cfg = cfg or HarnessConfig()
    r47 = _rung47()
    validation_root = cfg.work_root.parent / "harness-validation"
    ckpt_root = validation_root / "ckpt"
    checkpoint = ckpt_root / "checkpoint-4848"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or checkpoint.is_symlink():
        if checkpoint.resolve() != adapter_dir.resolve():
            raise AssertionError(
                f"validation checkpoint points at {checkpoint.resolve()}, not {adapter_dir.resolve()}"
            )
    else:
        checkpoint.symlink_to(adapter_dir.resolve(), target_is_directory=True)

    merge_cfg = r47.Rung47Config(
        ckpt_root=str(ckpt_root),
        work_root=str(validation_root),
        repo_root=str(cfg.repo_root),
        data_root=str(cfg.data_root),
        frames_cache=str(cfg.frames_cache),
        hf_home=str(cfg.work_root.parent.parent / "hf_cache"),
        split_json=str(cfg.split_json),
        base_model=str(base_model),
        keep_merged=False,
    )
    with rung42_epoch4_step(r47):
        try:
            merged = r47.merge_checkpoint(merge_cfg, ckpt_root, 4)
            summary = score_models(
                [f"control42_ep4={merged}"],
                control_labels=("control42_ep4",),
                cfg=cfg,
            )[0]
            if summary["verdict"] != "GREEN":
                raise AssertionError(
                    f"rung 54 harness validation is RED: {summary['bucket_mean']:.6f} vs "
                    f"{summary['target_42_ep4']:.4f} "
                    f"(delta {summary['delta_vs_target']:+.6f})"
                )
            return summary
        finally:
            r47.reclaim_merged(merge_cfg, 4)


__all__ = [
    "HarnessConfig",
    "ModelSpec",
    "TARGET_42_EP4",
    "VALIDATION_TOLERANCE",
    "assert_control_parser_identity",
    "ensure_import_paths",
    "make_logged_count_postprocess",
    "parse_status_histogram",
    "reclaim_model_dir",
    "rung42_epoch4_step",
    "score_models",
    "validate_control",
    "write_summary",
]
