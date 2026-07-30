"""Horizontal-flip QA audit for experiment 24.

The module is deliberately conservative.  It permits automatic transformations
only where the public FRAME schema proves the relation is camera-relative.  A
row outside a known template is retained in the audit as ``manual_review``;
it is never silently transformed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


_QUADRANT_RE = re.compile(r"\b(top|bottom)/(left|right)\b", re.IGNORECASE)
_FIXED_QUADRANT_CLASS_RE = re.compile(
    r"^What class is the foreign object located in the "
    r"(?:top|bottom)/(?:left|right) relative to the image center\?",
    re.IGNORECASE,
)
_CENTER_LOCATION_RE = re.compile(
    r"^Where is the center of the .+ located relative to the image center",
    re.IGNORECASE,
)
_POSITION_LIST_RE = re.compile(
    r"^At timepoint .+ provide all relative central positions of foreign objects",
    re.IGNORECASE,
)
_CLOSEST_CENTER_RE = re.compile(
    r"^Which of the visible foreign objects has its centre closest to the centre of the image",
    re.IGNORECASE,
)


def _swap_quadrant(match: re.Match[str]) -> str:
    vertical, horizontal = match.group(1), match.group(2)
    swapped = "right" if horizontal.lower() == "left" else "left"
    if horizontal.isupper():
        swapped = swapped.upper()
    elif horizontal[0].isupper():
        swapped = swapped.capitalize()
    return f"{vertical}/{swapped}"


def swap_left_right_quadrants(text: str) -> str:
    """Swap left/right in every ``top|bottom`` quadrant token in *text*."""
    return _QUADRANT_RE.sub(_swap_quadrant, str(text))


@dataclass(frozen=True)
class FlipRule:
    """A conservative disposition for one FRAME QA row."""

    disposition: str
    rule: str
    transform_question: bool = False
    transform_answer: bool = False


def classify_row(row: pd.Series) -> FlipRule:
    """Classify a QA row without inferring anatomical laterality.

    Dispositions:
    - ``transformable``: a complete deterministic image/QA transformation exists;
    - ``invariant``: the image can flip while question and answer stay valid;
    - ``excluded_situs``: anatomy-relative semantics are intentionally not flipped;
    - ``manual_review``: a camera-relative row has an unrecognised template.
    """
    capability = str(row["primary_capability"])
    question = str(row["question"])

    if capability == "1e":
        return FlipRule("excluded_situs", "anatomy_relative")
    # The public primary-capability label is helpful but not complete: the
    # fixed-quadrant class templates are tagged 1a in the supplied Parquets.
    # Template semantics therefore take precedence for camera-relative flips.
    if _FIXED_QUADRANT_CLASS_RE.match(question):
        return FlipRule("transformable", "fixed_quadrant_class", transform_question=True)
    if _CENTER_LOCATION_RE.match(question):
        return FlipRule("transformable", "object_center_quadrant", transform_answer=True)
    if _POSITION_LIST_RE.match(question):
        return FlipRule("transformable", "all_object_positions", transform_answer=True)
    if _CLOSEST_CENTER_RE.match(question):
        return FlipRule("invariant", "closest_to_image_center")
    if capability != "1d":
        return FlipRule("invariant", "non_spatial")
    if not _QUADRANT_RE.search(question) and not _QUADRANT_RE.search(str(row["answer"])):
        # The source has a few 1d rows whose text is actually grasping or
        # occlusion recognition. They are label-invariant under a reflection.
        return FlipRule("invariant", "nonspatial_mislabeled_1d")
    return FlipRule("manual_review", "unrecognised_1d_template")


def flipped_example(row: pd.Series) -> dict[str, object]:
    """Return the QA text after the horizontal image reflection.

    This function transforms only rows classified as ``transformable``.  It
    raises for every other disposition so a future exporter cannot accidentally
    apply a guessed mapping.
    """
    rule = classify_row(row)
    if rule.disposition != "transformable":
        raise ValueError(
            f"Row {row.get('id', '<unknown>')} is {rule.disposition} ({rule.rule}), "
            "not automatically transformable."
        )
    question = str(row["question"])
    answer = str(row["answer"])
    return {
        "question": swap_left_right_quadrants(question) if rule.transform_question else question,
        "answer": swap_left_right_quadrants(answer) if rule.transform_answer else answer,
        "rule": rule.rule,
    }


def audit_dataframe(frame: pd.DataFrame, *, split: str, dataset: str | None = None) -> pd.DataFrame:
    """Create a row-level audit; this does not modify the input dataframe."""
    required = {"id", "question", "answer", "answer_format", "primary_capability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FRAME dataframe is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rule = classify_row(row)
        flipped_question, flipped_answer = str(row.question), str(row.answer)
        if rule.disposition == "transformable":
            transformed = flipped_example(row)
            flipped_question = str(transformed["question"])
            flipped_answer = str(transformed["answer"])
        rows.append({
            "split": split,
            "dataset": dataset,
            "id": row.id,
            "primary_capability": row.primary_capability,
            "answer_format": row.answer_format,
            "disposition": rule.disposition,
            "rule": rule.rule,
            "question_changed": rule.transform_question,
            "answer_changed": rule.transform_answer,
            "question": row.question,
            "answer": row.answer,
            "flipped_question": flipped_question,
            "flipped_answer": flipped_answer,
        })
    return pd.DataFrame(rows)


def audit_parquet(
    path: str | Path, *, split: str, dataset: str | None = None,
) -> pd.DataFrame:
    """Read one FRAME Parquet and return its horizontal-flip audit."""
    return audit_dataframe(pd.read_parquet(path), split=split, dataset=dataset)


def audit_frame_root(data_dir: str | Path) -> pd.DataFrame:
    """Audit FRAME Parquets in either supported local directory layout.

    ``<root>/train.parquet`` + ``test.parquet`` is the original flat layout.
    ``<root>/<dataset>/train.parquet`` + ``test.parquet`` is the current
    dataset-nested layout. The latter retains the dataset name in every row.
    """
    root = Path(data_dir)
    audits: list[pd.DataFrame] = []
    for split in ("train", "test"):
        flat = root / f"{split}.parquet"
        if flat.is_file():
            audits.append(audit_parquet(flat, split=split))
            continue
        nested = sorted(root.glob(f"*/{split}.parquet"))
        if not nested:
            raise FileNotFoundError(
                f"No {split}.parquet found directly under {root} or one directory below it."
            )
        audits.extend(
            audit_parquet(path, split=split, dataset=path.parent.name) for path in nested
        )
    return pd.concat(audits, ignore_index=True)


def write_audit(audit: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """Write inspectable rows and a compact disposition summary to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "flip_audit_rows.csv", index=False)
    summary = (
        audit.groupby(
            ["split", "dataset", "primary_capability", "answer_format", "disposition", "rule"],
            dropna=False,
        )
        .size()
        .reset_index(name="n")
        .sort_values(["split", "primary_capability", "disposition", "rule"])
    )
    summary.to_csv(out / "flip_audit_summary.csv", index=False)
    return summary
