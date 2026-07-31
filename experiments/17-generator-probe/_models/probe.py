"""Rung 17 — can the CoA generator actually see?

Importable engine. The notebook is the launcher; nothing here is hand-run.

The generator (`Qwen3-VL-32B-Instruct`) writes reasoning that DERIVES a gold it is handed,
so a perception failure never surfaces as a wrong answer — it surfaces as evidence
describing a scene that is not there, attached to a correct answer. Nothing downstream
catches that: `<answer> == gold` by construction, and the Qwen judge-mirror is a text model
that cannot see the frame either. So we ask the generator the FRAME question directly, with
no gold and no scaffold, and score it exactly like any other model in this repo.

Reuses `experiments/09-coa-sft/_tools/gen_onpod.py` (stage ``blind``) rather than standing up
a second generator: a probe that loads the model differently from the thing it is probing
measures the probe, not the generator. What this rung ADDS is canonical scoring — the
existing stage reported raw accuracy, and raw accuracy against a template-aware floor of
0.34/0.46 is not a statement about perception.

🔴 63 GB in bf16. Never co-resident with an 8B LoRA training run on a 96 GB card.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_EXPERIMENTS = Path(__file__).resolve().parents[2]
for _p in (_EXPERIMENTS / "09-coa-sft" / "_tools", _EXPERIMENTS / "09-coa-sft" / "_models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@dataclass
class ProbeConfig:
    """Inline in the notebook. One model, one sample, no training."""

    model_id: str = "/workspace/models/qwen3-vl-32b"
    quantization: str | None = None            # bf16; the volume copy is not prequantized
    max_new_tokens: int = 32                   # a direct answer, not a scaffold
    max_pixels: int = 1280 * 720               # == BaselineConfig: probe pixels == serve pixels

    # 🔴 The pod keeps the corpus at /workspace/orena-data, NOT inside a checkout. This field
    # used to read `/workspace/repo/external_data/orena-data` — the repo's LOCAL convention
    # (`external_data/` is gitignored, so every checkout populates it by hand and the pod's
    # never was). Three of the four paths here were pod-correct and this one was not, which is
    # the signature of a config written for a rung that has never run. Same layout either way:
    # <dataset>/data/frame/{train,test}.parquet.
    data_root: Path = Path("/workspace/orena-data")
    frames_cache: Path = Path("/workspace/frames_cache")
    # Self-locating rather than hard-coded: the checkout that OWNS this file is the one whose
    # runs/ the results belong in. A literal `/workspace/repo/...` writes another person's
    # checkout when the pod carries more than one — which it does.
    exp_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    run_name: str = "17_generator_probe_v1"
    datasets: tuple[str, ...] = ("heico", "lapchole")
    base_fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})

    probe_n: int = 200
    seed: int = 42
    smoke: bool = False
    smoke_n: int = 8

    # 🔴 Refuses to load if the card is busy. The wave-R2 trainings own this GPU first, and
    # a 63 GB load beside them OOMs the training, not the probe — i.e. the cheap job kills
    # the expensive one. Measured headroom, not a hope.
    min_free_gib: int = 70

    @property
    def run_dir(self) -> Path:
        return self.exp_dir / "runs" / self.run_name

    @property
    def n(self) -> int:
        return self.smoke_n if self.smoke else self.probe_n


def assert_gpu_headroom(cfg: ProbeConfig) -> dict:
    """RAISES unless the card has room for a 63 GB model. Never a warning: the failure this
    prevents is silent and lands on someone else's 8-hour run."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — this rung is pod-only")
    free_b, total_b = torch.cuda.mem_get_info()
    free, total = free_b / 2**30, total_b / 2**30
    rep = {"free_gib": round(free, 1), "total_gib": round(total, 1), "required_gib": cfg.min_free_gib}
    if free < cfg.min_free_gib:
        raise RuntimeError(
            f"GPU headroom gate: {free:.1f} GiB free of {total:.1f}, need {cfg.min_free_gib}. "
            "Something else is on this card — almost certainly a wave-R2 training run. "
            "Loading a 63 GB generator beside it OOMs the TRAINING, not the probe. Wait."
        )
    logger.info("GPU headroom OK: %.1f GiB free of %.1f", free, total)
    return rep


def generate(cfg: ProbeConfig) -> Path:
    """Run the blind probe through rung 09's own driver and return its CSV.

    Selection is rung 09's ``select_blind_probe`` over the TRAIN rows — deliberately, on both
    counts. Train is the population the generator will actually caption, so probing it there
    measures the thing we are about to trust. And reusing rung 09's selector and its model
    load means a difference between probe and generation cannot come from the harness.
    """
    import gen_onpod  # noqa: PLC0415  (experiments/09-coa-sft/_tools)

    on = gen_onpod.OnPodConfig(
        model_id=str(cfg.model_id),
        quantization=cfg.quantization,
        max_new_tokens=cfg.max_new_tokens,
        max_pixels=cfg.max_pixels,
        data_root=cfg.data_root,
        frames_cache=cfg.frames_cache,
        datasets=cfg.datasets,
        base_fps=dict(cfg.base_fps),
        exp_dir=cfg.exp_dir,
        run_name=cfg.run_name,
        seed=cfg.seed,
        eyeball_n=cfg.n,          # blind_probe reads eyeball_n as its sample size
    )
    out = gen_onpod.run_stage(on, "blind_probe")
    logger.info("blind probe generated -> %s", out)
    return Path(out)


def score_canonically(cfg: ProbeConfig, results_csv: Path, gold) -> dict:
    """The whole point of this rung. RULES §1: score ONLY via frame.metrics, read MARGIN.

    Raw accuracy for a zero-shot generator is unreadable — the template-aware floor is 0.34
    ID / 0.46 OOD, so a model can look respectable and be at the trivial constant.
    """
    import glob  # noqa: PLC0415

    import pandas as pd
    from focus.taxonomy import Capability  # noqa: PLC0415
    from frame import metrics

    df = pd.read_csv(results_csv)

    # 🔴 `blind_probe.csv` is rung 09's shape, not the Evaluator's. It carries
    # qID/answer_format/gold/correct — `correct` is already an accepted alias for
    # `correctness` (metrics.py:148) — but NOT `video` (the clustered bootstrap keys on it,
    # RULES §13: effective n is videos) nor `primary` (leaf→group, RULES §1). Without them
    # stratified_report dies on KeyError, which is what a rung that had never been RUN could
    # not have known. Both come back from the very parquets the probe sampled, on the same
    # qID namespacing, so nothing is invented here.
    meta = []
    for f in sorted(glob.glob(str(Path(cfg.data_root) / "*" / "data" / "frame" / "train.parquet"))):
        ds = Path(f).parents[2].name
        p = pd.read_parquet(f, columns=["id", "video", "primary_capability"])
        meta.append(p.assign(qID=ds + "__" + p["id"].astype(str)))
    meta = pd.concat(meta, ignore_index=True)
    # Store the LEAF VALUE, as data.py:69 does, so this run's leaf rows are named like every
    # other run's rather than carrying the raw "1a" codes.
    meta["primary"] = meta["primary_capability"].map(
        lambda raw: (c.value if (c := Capability.from_any(raw)) is not None else None)
    )

    df = df.merge(meta[["qID", "video", "primary"]], on="qID", how="left")
    if (n_bad := int(df["primary"].isna().sum() + df["video"].isna().sum())):
        raise KeyError(
            f"{n_bad} of {len(df)} probe rows failed to join their parquet metadata — the "
            "probe and the gold disagree on qID namespacing, and scoring would silently drop rows."
        )

    metrics.assert_no_dup_qid(df)
    metrics.assert_ood_from_qid(df)
    rep = metrics.stratified_report(df, gold=gold)
    metrics.assert_floors_vs_eval_set(rep)
    return rep


def verdict(rep: dict) -> str:
    """The pre-registered rule, in code so it cannot be reinterpreted after the fact.
    See context/17-generator-probe/CONTEXT.md."""
    m_id, m_ood = rep.get("margin_ID"), rep.get("margin_OOD")
    if m_id is None or m_ood is None:
        return "INDETERMINATE — a margin is missing; gold coverage is incomplete"
    if m_id < 0 or m_ood < 0:
        return ("🔴 STOP — below the trivial floor on at least one distribution. A teacher that "
                "cannot beat the modal answer cannot supervise perception.")
    if m_id > 0 and m_ood > 0:
        return ("🟢 GO — positive margin on both. Directional, not a magnitude: a general VLM also "
                "loses points to task format, so this under-states perception.")
    return ("🟡 INDETERMINATE — indistinguishable from the trivial constant. Scaffolds would be "
            "gold-anchored prose; generate only behind the human eyeball gate.")
