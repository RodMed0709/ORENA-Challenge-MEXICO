"""Build ``context/MEASURED.md`` — the answer to *"has this already been measured?"*.

Four sessions in a row re-derived work that was already committed and well written
(the per-class verdict, rung 05's image ablation, the ``by_bucket_format`` cross, and a
complete probe spec with its run closed). None was lost to poor documentation. They
were lost because the repo has six orientation documents and no way to *ask* it.

So this index is **generated, never maintained** — the same contract as
``frame.ledger``. A hand-kept seventh document would become the fifth thing that goes
stale; ``context/NOW.md`` went stale within hours of the session that wrote it.

It reads **four** sources, because a decisions-only index would have caught exactly one
of the four losses:

===================  =====================================================
source               what it contributes
===================  =====================================================
``decisions/*.md``   settled verdicts (frontmatter)
experiment READMEs   the ladder — which rung changed what, and its verdict
``RESULTS*.csv``     runs and probes that produced a number
``stratified.json``  **which cuts are already computed** (the quiet one)
===================  =====================================================

The last is the least obvious and the highest value: ``by_bucket_format`` had been
committed for days before anyone read it.

Run: ``python -m frame.measured``. Zero GPU, pure file reads.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "Measurement",
    "SourceKind",
    "Status",
    "assert_decisions_indexed",
    "build",
    "render",
]

REPO = Path(__file__).resolve().parents[2]


class Status(StrEnum):
    """Standing of a verdict. Closed vocabulary — an unknown value raises at parse
    time rather than silently entering the table."""

    MEASURED = "MEASURED"
    SETTLED = "SETTLED"
    ACTIVE_PLAN = "ACTIVE_PLAN"
    RETRACTED = "RETRACTED"
    RE_SCOPED = "RE_SCOPED"
    NO_GO = "NO_GO"


class SourceKind(StrEnum):
    DECISION = "decision"
    LADDER = "ladder"
    RESULTS = "results"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class Measurement:
    question: str
    verdict: str
    where: str
    source_kind: SourceKind
    status: Status | None = None
    date: str = ""
    withdrawn: str = ""
    amended_by: tuple[str, ...] = ()
    question_derived: bool = False

    @property
    def needs_warning(self) -> bool:
        """A verdict that has been narrowed or partly withdrawn must never render as
        plain. Reading the per-class note without its re-scoping is precisely the
        failure this index exists to prevent."""
        return bool(self.withdrawn or self.amended_by)


# ── frontmatter ───────────────────────────────────────────────────────────────
# Hand-rolled on purpose: the schema is flat keys plus short lists, and `pyyaml` is
# present in the dev env only transitively. A 20-line parser beats an undeclared
# dependency in a repo whose Constitution pins its deps.

_LIST_RE = re.compile(r"^\[(.*)\]$")


def _parse_frontmatter(text: str) -> dict[str, str | list[str]]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    out: dict[str, str | list[str]] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if (m := _LIST_RE.match(value)) is not None:
            out[key.strip()] = [x.strip() for x in m.group(1).split(",") if x.strip()]
        else:
            out[key.strip()] = value.strip('"')
    return out


# ── sources ───────────────────────────────────────────────────────────────────


def read_decisions(repo: Path = REPO) -> list[Measurement]:
    rows = []
    for path in sorted((repo / "context" / "decisions").glob("*.md")):
        fm = _parse_frontmatter(path.read_text())
        if not fm:
            continue
        rows.append(
            Measurement(
                question=str(fm.get("question", "")),
                verdict=str(fm.get("verdict", "")),
                where=f"context/decisions/{path.name}",
                source_kind=SourceKind.DECISION,
                status=Status(str(fm["status"])) if "status" in fm else None,
                date=str(fm.get("date", "")),
                withdrawn=str(fm.get("withdrawn", "")),
                amended_by=tuple(fm.get("amended_by", []) or ()),
                question_derived=str(fm.get("question_derived", "")) == "true",
            )
        )
    return rows


_LADDER_ROW = re.compile(r"^\|\s*\**([\w\-.]+)\**\s*\|(.+?)\|(.+?)\|(.+?)\|\s*$")


def read_ladders(repo: Path = REPO) -> tuple[list[Measurement], list[str]]:
    """Parse each experiment README's ``## Ladder`` table.

    Returns the rows plus the list of READMEs whose ladder could not be parsed —
    reported in the output rather than swallowed, so a drifted README is visible
    instead of quietly absent.
    """
    rows, skipped = [], []
    for readme in sorted((repo / "experiments").glob("*/README.md")):
        text = readme.read_text()
        block = re.search(r"^## Ladder\n(.*?)(?=\n## )", text, re.S | re.M)
        if block is None:
            skipped.append(str(readme.relative_to(repo)))
            continue
        found = False
        for line in block.group(1).splitlines():
            m = _LADDER_ROW.match(line)
            if not m or set(m.group(2).strip()) <= {"-", ":"}:
                continue
            rung, changed, _baseline, verdict = (g.strip(" *") for g in m.groups())
            if rung.lower() in {"rung", ""}:
                continue
            found = True
            rows.append(
                Measurement(
                    question=f"What did rung {rung} change, and what came of it?",
                    verdict=f"{changed} → {verdict}",
                    where=str(readme.parent.relative_to(repo)),
                    source_kind=SourceKind.LADDER,
                )
            )
        if not found:
            skipped.append(str(readme.relative_to(repo)))
    # a rung appears in every downstream README's ladder; keep the first mention only
    seen, unique = set(), []
    for r in rows:
        if r.verdict in seen:
            continue
        seen.add(r.verdict)
        unique.append(r)
    return unique, skipped


def read_results(repo: Path = REPO) -> list[Measurement]:
    rows = []
    for csv_path in sorted((repo / "experiments").glob("*/RESULTS*.csv")):
        try:
            with csv_path.open() as fh:
                header = next(csv.reader(fh), [])
                n_rows = sum(1 for _ in fh)
        except OSError:
            continue
        rows.append(
            Measurement(
                question=f"What numbers does {csv_path.parent.name} report?",
                verdict=f"{n_rows} row(s); columns: {', '.join(header[:6])}"
                + ("…" if len(header) > 6 else ""),
                where=str(csv_path.relative_to(repo)),
                source_kind=SourceKind.RESULTS,
            )
        )
    return rows


def read_artifacts(repo: Path = REPO) -> list[Measurement]:
    """Which stratified cuts are already computed and sitting on disk.

    ``runs/`` is gitignored, so on a fresh clone this degrades to an explicit
    "not available locally" row — never to silence. A silently-missing artifact is
    the exact bug that versioning ``stratified.json`` was introduced to fix.
    """
    paths = sorted((repo / "experiments").glob("*/runs/*/stratified.json"))
    if not paths:
        return [
            Measurement(
                question="Which stratified cuts are already computed?",
                verdict="NOT AVAILABLE LOCALLY — `runs/` is gitignored; "
                "fetch a run's artifacts to populate this row",
                where="experiments/*/runs/*/stratified.json",
                source_kind=SourceKind.ARTIFACT,
            )
        ]
    rows = []
    for path in paths:
        try:
            keys = sorted(json.loads(path.read_text()).keys())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            Measurement(
                question=f"Which cuts are precomputed for {path.parent.name}?",
                verdict=", ".join(f"`{k}`" for k in keys),
                where=str(path.relative_to(repo)),
                source_kind=SourceKind.ARTIFACT,
            )
        )
    return rows


# ── gate ──────────────────────────────────────────────────────────────────────


def assert_decisions_indexed(repo: Path = REPO) -> None:
    """RAISING gate: every decision note must carry valid, resolvable frontmatter.

    A new note without frontmatter must break the build, not slip into the repo
    unindexed — otherwise this index decays exactly like the documents it replaces.
    """
    notes = sorted((repo / "context" / "decisions").glob("*.md"))
    slugs = {p.stem for p in notes}
    problems: list[str] = []

    for path in notes:
        fm = _parse_frontmatter(path.read_text())
        if not fm:
            problems.append(f"{path.name}: no frontmatter")
            continue
        for required in ("question", "verdict", "status"):
            if not fm.get(required):
                problems.append(f"{path.name}: missing `{required}`")
        if (status := fm.get("status")) and str(status) not in set(Status):
            allowed = ", ".join(sorted(s.value for s in Status))
            problems.append(f"{path.name}: status `{status}` not in [{allowed}]")
        for link_field in ("amends", "amended_by"):
            for target in fm.get(link_field, []) or []:
                if target not in slugs:
                    problems.append(
                        f"{path.name}: {link_field} → `{target}` does not exist"
                    )

    if problems:
        raise AssertionError(
            "decision notes are not indexable:\n  " + "\n  ".join(problems)
        )


# ── render ────────────────────────────────────────────────────────────────────

_ORDER = [SourceKind.DECISION, SourceKind.LADDER, SourceKind.RESULTS, SourceKind.ARTIFACT]

_SECTION_TITLE = {
    SourceKind.DECISION: "Settled verdicts",
    SourceKind.LADDER: "The ladder — what each rung changed",
    SourceKind.RESULTS: "Runs and probes that produced a number",
    SourceKind.ARTIFACT: "Cuts already computed (do not recompute these)",
}


@dataclass
class Index:
    measurements: list[Measurement] = field(default_factory=list)
    skipped_ladders: list[str] = field(default_factory=list)


def build(repo: Path = REPO) -> Index:
    assert_decisions_indexed(repo)
    ladders, skipped = read_ladders(repo)
    return Index(
        measurements=[
            *read_decisions(repo),
            *ladders,
            *read_results(repo),
            *read_artifacts(repo),
        ],
        skipped_ladders=skipped,
    )


def render(index: Index) -> str:
    out = [
        "# MEASURED — has this already been measured?",
        "",
        "> 🤖 **GENERATED by `python -m frame.measured`. Never hand-edit** — your changes",
        "> will be overwritten, and a hand-kept index is the failure this file exists to fix.",
        ">",
        "> **Read this BEFORE proposing an experiment.** Four sessions in a row re-derived work",
        "> that was already committed. Nothing was lost to poor documentation — it was lost",
        "> because there was no way to ask.",
        ">",
        "> ⚠️ **A row marked ⚠️ has been narrowed or partly withdrawn. Read its note, not this",
        "> row.** Reading a verdict without its amendment is how this repo has been bitten.",
        "",
    ]
    for kind in _ORDER:
        rows = [m for m in index.measurements if m.source_kind == kind]
        if not rows:
            continue
        out += [f"## {_SECTION_TITLE[kind]}", ""]
        if kind is SourceKind.DECISION:
            out += ["| | Question | Verdict | Status | Where |", "|---|---|---|---|---|"]
            for m in sorted(rows, key=lambda m: (m.date, m.where), reverse=True):
                flag = "⚠️" if m.needs_warning else ""
                q = m.question + (" *(derived)*" if m.question_derived else "")
                status = str(m.status or "")
                if m.withdrawn:
                    status += f"<br>withdrawn: {m.withdrawn}"
                if m.amended_by:
                    status += "<br>amended by: " + ", ".join(
                        f"[[{a}]]" for a in m.amended_by
                    )
                out.append(f"| {flag} | {q} | {m.verdict} | {status} | `{m.where}` |")
        else:
            out += ["| Question | Answer | Where |", "|---|---|---|"]
            for m in sorted(rows, key=lambda m: m.where):
                out.append(f"| {m.question} | {m.verdict} | `{m.where}` |")
        out.append("")

    if index.skipped_ladders:
        out += [
            "## ⚠️ Not indexed",
            "",
            "These READMEs have no parseable `## Ladder` table, so their rungs are missing",
            "from this index. Reported rather than dropped silently:",
            "",
            *[f"- `{p}`" for p in sorted(index.skipped_ladders)],
            "",
        ]
    return "\n".join(out)


def main(repo: Path = REPO) -> Path:
    target = repo / "context" / "MEASURED.md"
    target.write_text(render(build(repo)))
    return target


if __name__ == "__main__":  # pragma: no cover
    print(f"wrote {main()}")
