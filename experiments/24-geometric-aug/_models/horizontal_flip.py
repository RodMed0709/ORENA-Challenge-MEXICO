"""Label-aware horizontal-flip export and training engine for experiment 24.

This is a training-time augmentation, materialised once per training row.  The
Bernoulli draw is deterministic from ``(seed, frozen-control-row)`` so a given probability
produces reproducible JSONL, frames, and QA transformations.  It is deliberately
not an inference transform: evaluation continues to use untouched images.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve()
_EXPERIMENTS = _HERE.parents[2]
for _path in (
    _HERE.parent,
    _EXPERIMENTS / "21-recipe-sweep" / "_models",
    _EXPERIMENTS / "06-vit-lora" / "_models",
    _EXPERIMENTS / "02-lora-sft" / "_models",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import flip_audit  # noqa: E402
from recipe_sweep_train import RecipeSweepConfig, swift_args_21  # noqa: E402


@dataclass(frozen=True)
class HorizontalFlipPolicy:
    """The sole experimental variable. ``probability=0`` is an exact no-op."""

    probability: float = 0.25
    seed: int = 42
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"flip probability must be in [0, 1], got {self.probability}")


def should_flip(policy: HorizontalFlipPolicy, row_key: str) -> bool:
    """A stable Bernoulli draw from the policy seed and frozen JSONL row key."""
    digest = hashlib.blake2b(
        f"{policy.seed}|horizontal_flip|{row_key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 < policy.probability


def _safe_name(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", qid).strip("_")


def _parse_jsonl_row(line: str) -> dict:
    record = json.loads(line)
    user = record["messages"][1]["content"]
    if not user.startswith("<image>"):
        raise ValueError(f"user content must start with '<image>', got {user[:64]!r}")
    return {
        "record": record,
        "question": user[len("<image>"):],
        "answer": str(record["messages"][2]["content"]),
        "image": Path(record["images"][0]),
    }


def transform_qa(question: str, answer: str) -> tuple[str, str, str]:
    """Return the label-correct QA pair for a horizontally mirrored image.

    Non-spatial and excluded-situs rows are invariant. Any unexpected
    ``manual_review`` disposition raises rather than creating guessed labels.
    """
    import pandas as pd

    row = pd.Series({
        "question": question,
        "answer": answer,
        # The camera-relative templates are recognised by exact text patterns.
        # A non-spatial code is intentional: it leaves all other rung-18 rows
        # (including minted and paraphrased number rows) byte-for-byte invariant.
        "primary_capability": "3a",
    })
    rule = flip_audit.classify_row(row)
    if rule.disposition == "manual_review":
        raise ValueError(f"unrecognised spatial template: {question!r}")
    if rule.disposition != "transformable":
        return question, answer, rule.rule
    transformed = flip_audit.flipped_example(row)
    return str(transformed["question"]), str(transformed["answer"]), str(transformed["rule"])


def rewrite_jsonl_with_flips(
    base_jsonl: Path,
    out_jsonl: Path,
    flip_frames_dir: Path,
    policy: HorizontalFlipPolicy,
    manifest_csv: Path,
    *,
    max_rows: int | None = None,
) -> dict:
    """Write a mixed original/flipped JSONL and materialise only flipped JPEGs.

    Rung 21's control is rung 18's final JSONL, which contains minted and
    paraphrased rows. Therefore its frozen digest plus line number is the row
    identity—not a re-derived raw-data qID. This preserves every upstream data
    intervention and avoids silently dropping the 667 minted rows.
    """
    if policy.probability == 0:
        raise ValueError("probability=0 is the control path; do not create a copied JSONL")
    flip_frames_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    control_sha = hashlib.sha256(base_jsonl.read_bytes()).hexdigest()
    with base_jsonl.open(encoding="utf-8") as source, out_jsonl.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            parsed = _parse_jsonl_row(line)
            row_key = f"{control_sha}:{line_no}"
            flip = should_flip(policy, row_key)
            question, answer, rule = parsed["question"], parsed["answer"], "not_flipped"
            image_path = parsed["image"]
            if flip:
                question, answer, rule = transform_qa(question, answer)
                image_path = flip_frames_dir / f"{_safe_name(row_key)}.jpg"
                with Image.open(parsed["image"]) as image:
                    image.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
                        image_path, quality=policy.jpeg_quality
                    )
                parsed["record"]["messages"][1]["content"] = f"<image>{question}"
                parsed["record"]["messages"][2]["content"] = answer
                parsed["record"]["images"] = [str(image_path)]
            target.write(json.dumps(parsed["record"], ensure_ascii=False) + "\n")
            rows.append({
                "row_key": row_key,
                "flipped": flip,
                "rule": rule,
                "src_frame": str(parsed["image"]),
                "train_frame": str(image_path),
                "question_changed": question != parsed["question"],
                "answer_changed": answer != parsed["answer"],
            })

    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"n_rows": len(rows), "n_flipped": sum(bool(row["flipped"]) for row in rows),
            "jsonl": out_jsonl, "manifest": manifest_csv, "control_sha256": control_sha}


@dataclass
class HorizontalFlipConfig(RecipeSweepConfig):
    """Rung-21 arm A plus one training-data policy; OFF is byte-identical to arm A."""

    exp_dir: Path = Path("/workspace/repo/experiments/24-geometric-aug")
    run_name: str = "24_flip_p25_v1"
    learning_rate: float = 1e-4
    flip: HorizontalFlipPolicy | None = None
    flip_smoke_rows: int = 64

    @property
    def flip_frames_dir(self) -> Path:
        return self.run_dir / "flip_frames"

    @property
    def flip_train_jsonl(self) -> Path:
        return self.run_dir / "train_flip.jsonl"

    @property
    def flip_manifest_csv(self) -> Path:
        return self.run_dir / "flip_manifest.csv"

    @property
    def train_jsonl(self) -> Path:
        return self.control_train_jsonl if self.flip is None else self.flip_train_jsonl


def diff_vs_rung21(cfg: HorizontalFlipConfig) -> dict[str, tuple[str | None, str | None]]:
    """The on-policy training argv may differ from rung 21 arm A only in ``--dataset``."""
    from recipe_sweep_train import _as_map
    from dataclasses import replace

    control = replace(cfg, flip=None)
    before, after = _as_map(swift_args_21(control)), _as_map(swift_args_21(cfg))
    return {key: (before.get(key), after.get(key)) for key in set(before) | set(after)
            if before.get(key) != after.get(key)}


def _assert_control_sha256(cfg: HorizontalFlipConfig) -> str:
    if not cfg.control_train_jsonl.is_file():
        raise FileNotFoundError(f"missing rung-21 control dataset: {cfg.control_train_jsonl}")
    digest = hashlib.sha256(cfg.control_train_jsonl.read_bytes()).hexdigest()
    if cfg.control_sha256 and digest != cfg.control_sha256:
        raise AssertionError(
            f"control JSONL sha256 {digest} != declared {cfg.control_sha256}; "
            "the baseline data changed, so the A/B is no longer defined"
        )
    return digest


def gate_single_variable(cfg: HorizontalFlipConfig) -> dict:
    """GATE: flag OFF is control bytes; flag ON changes the trainer argv only in dataset."""
    control_digest = _assert_control_sha256(cfg)
    if cfg.flip is None:
        if cfg.train_jsonl != cfg.control_train_jsonl:
            raise AssertionError("flip OFF must use the rung-21 control JSONL in place")
        return {"mode": "off", "dataset": str(cfg.train_jsonl), "sha256": control_digest, "diff": {}}
    diff = diff_vs_rung21(cfg)
    if set(diff) != {"--dataset"}:
        raise AssertionError(f"flip arm changed flags besides --dataset: {diff}")
    if not cfg.flip_train_jsonl.exists():
        raise FileNotFoundError(f"export stage did not create {cfg.flip_train_jsonl}")
    return {"mode": "on", "probability": cfg.flip.probability,
            "control_sha256": control_digest, "diff": diff}


def export(cfg: HorizontalFlipConfig) -> Path:
    """Create the policy-on dataset; policy OFF returns the control dataset unchanged."""
    if cfg.flip is None:
        return cfg.control_train_jsonl
    return Path(rewrite_jsonl_with_flips(
        cfg.control_train_jsonl, cfg.flip_train_jsonl, cfg.flip_frames_dir, cfg.flip,
        cfg.flip_manifest_csv,
        max_rows=cfg.flip_smoke_rows if cfg.smoke else None,
    )["jsonl"])


def train(cfg: HorizontalFlipConfig) -> Path:
    """Delegate training to rung 06's guarded trainer with rung 21's argv builder."""
    import vit_lora_train as rung06

    gate_single_variable(cfg)
    original = rung06._swift_args
    rung06._swift_args = swift_args_21
    try:
        return rung06._train(cfg)
    finally:
        rung06._swift_args = original


def main(cfg: HorizontalFlipConfig, stage: str) -> Path:
    """Notebook entrypoint: ``export`` or ``train``; merge/evaluation stay in notebooks."""
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "export":
        return export(cfg)
    if stage == "train":
        return train(cfg)
    raise ValueError(f"unknown stage {stage!r}; expected 'export' or 'train'")
