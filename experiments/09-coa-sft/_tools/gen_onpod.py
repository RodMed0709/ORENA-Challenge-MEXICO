"""ON-POD scaffold generation driver for R1 (experiment 09) — RUN ON THE POD ONLY.

⚠️ This module loads a large VLM and reads frames from the volume — it CANNOT run on the
local dev box (no GPU, DUA keeps data on-pod). It was authored off-pod and is UNTESTED on a
live GPU; run the `eyeball` stage first and inspect before trusting the `pilot` stage.

Pipeline (both stages reuse the pure engine `coa_scaffold_gen` + `frame.data`):
  stage="eyeball" : select_eyeball_frames(50) -> generate VISION scaffolds -> sample_eyeball.csv
                    (the human kill-gate: does <evidence> contradict the frame on multi-object
                    frames? Do NOT proceed to pilot until these pass.)
  stage="pilot"   : select_pilot(2000) -> generate scaffolds -> train_coa.jsonl  AND the matched
                    control train_bare.jsonl (bare gold, SAME 2000 qIDs) -> both feed identical
                    LoRA runs; the kill-gate reads CoA-2k − bare-2k (single variable).

Compliance: the generator runs entirely on-pod; no frame/annotation leaves
([[no-external-api-for-challenge-data]]). Model is a local downloadable-weights VLM.

Pod usage (in Jupyter or a shell on the pod, from the repo root):
    import sys; sys.path.insert(0, "experiments/09-coa-sft/_tools")
    import gen_onpod
    cfg = gen_onpod.OnPodConfig(model_id="Qwen/Qwen2.5-VL-72B-Instruct")  # 32B: Qwen/Qwen3-VL-32B-Instruct, quantization=None
    gen_onpod.run_stage(cfg, stage="eyeball")     # inspect runs/.../sample_eyeball_vision.csv, THEN:
    gen_onpod.run_stage(cfg, stage="pilot")
Or from a shell:  python experiments/09-coa-sft/_tools/gen_onpod.py eyeball
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

_MODELS = Path(__file__).resolve().parents[1] / "_models"
if str(_MODELS) not in sys.path:
    sys.path.insert(0, str(_MODELS))
import coa_scaffold_gen as g  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class OnPodConfig:
    # generator — 72B (max perception, teacher-only, needs FP8 to fit one 80GB card) is the
    # default per Rodrigo; flip model_id to "Qwen/Qwen3-VL-32B-Instruct" for the Apache /
    # same-family / faster option (bf16 fits 80GB). BOTH are data-gen teachers; we ship only the 8B.
    model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    quantization: str | None = "fp8"      # None for bf16 (32B on 80GB); "fp8" for the 72B
    max_new_tokens: int = 160             # short scaffold; generator only (NOT the 5s serve path)
    max_pixels: int = 1280 * 720          # == BaselineConfig — train pixels == serve pixels
    # data / paths (POD layout)
    data_root: Path = Path("/workspace/repo/external_data/orena-data")
    frames_cache: Path = Path("/workspace/frames_cache")
    base_fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})
    datasets: tuple[str, ...] = ("heico", "lapchole")
    exp_dir: Path = Path("/workspace/repo/experiments/09-coa-sft")
    run_name: str = "09_coa_sft_v1"
    seed: int = 42
    eyeball_n: int = 50
    pilot_n: int = 2000

    @property
    def gen_cfg(self) -> g.GenConfig:
        return g.GenConfig(data_root=self.data_root, datasets=self.datasets,
                           base_fps=dict(self.base_fps), exp_dir=self.exp_dir,
                           run_name=self.run_name, frames_cache=self.frames_cache, seed=self.seed)

    @property
    def run_dir(self) -> Path:
        return self.exp_dir / "runs" / self.run_name


# ── VLM wrapper (transformers; mirrors src/frame/engine.py load path, model-agnostic) ──

class VLMGenerator:
    def __init__(self, cfg: OnPodConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM  # generic (4.57+)
        except ImportError:  # pragma: no cover
            from transformers import AutoModelForVision2Seq as _AutoVLM
        logger.info("loading generator %s (quant=%s) …", self.cfg.model_id, self.cfg.quantization)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id, max_pixels=self.cfg.max_pixels)
        kw = dict(dtype=torch.bfloat16, device_map="auto")
        if self.cfg.quantization == "fp8":
            # requires a prequantized -FP8 checkpoint OR compressed-tensors/torchao support;
            # if this errors, use an -FP8 model_id (e.g. Qwen/Qwen2.5-VL-72B-Instruct-FP8) or serve
            # via vLLM (--quantization fp8) instead. Left explicit so the failure is legible.
            kw["dtype"] = torch.bfloat16
        self.model = _AutoVLM.from_pretrained(self.cfg.model_id, **kw).eval()
        logger.info("generator ready.")

    def generate(self, image, prompt_text: str, system: str | None = None) -> str:
        import torch
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt_text}]})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:  # noqa: BLE001
            image_inputs, video_inputs = [image], None
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def _frame_for(provider, item, frames_cache: Path):
    """Return the PIL frame for one item, materializing into frames_cache if missing."""
    from frame.data import frame_cache_name
    fc = frames_cache / frame_cache_name(item)
    if fc.exists():
        from PIL import Image
        return Image.open(fc).convert("RGB")
    provider.ensure_reader(item)
    img = provider.get_frame(item)
    fc.parent.mkdir(parents=True, exist_ok=True)
    img.save(fc, quality=95)
    return img


def _iter_rows_with_frames(cfg: OnPodConfig, sample_df):
    """Yield (row_dict, PIL_image, siblings_str) for a selected sample, frames grouped by video."""
    from frame.config import BaselineConfig
    from frame.data import FrameProvider, load_frame_items
    gcfg = cfg.gen_cfg
    df = g.estimate_frame_objects(g.load_train_rows(gcfg))
    factsheets = g.build_factsheets(df)
    # build FrameItems keyed by qID so we can fetch the right frame per selected row
    bcfg = BaselineConfig(data_root=cfg.data_root, model_path=Path("/workspace/models/qwen3-vl-8b"),
                          datasets=cfg.datasets, base_fps=dict(cfg.base_fps), max_pixels=cfg.max_pixels)
    items = {it.request.qID: it for it in load_frame_items(bcfg, splits=("train", "test"))}
    provider = FrameProvider(bcfg)
    rows = sample_df.to_dict("records")
    rows.sort(key=lambda r: (r["dataset"], str(r["video"]), r["timestamp_start"]))  # video-grouped reads
    try:
        for r in rows:
            it = items.get(r["qID"])
            if it is None:
                logger.warning("no FrameItem for %s — skipped", r["qID"])
                continue
            sib = g.siblings_for(factsheets, r["frameid"], r["qID"])
            yield r, _frame_for(provider, it, cfg.frames_cache), sib
    finally:
        provider.close()


def run_stage(cfg: OnPodConfig, stage: str) -> Path:
    """stage='eyeball' -> sample_eyeball_vision.csv ; stage='pilot' -> train_coa.jsonl + train_bare.jsonl."""
    assert stage in ("eyeball", "eyeball_grouped", "blind_probe", "pilot"), stage
    gcfg = cfg.gen_cfg
    df = g.estimate_frame_objects(g.load_train_rows(gcfg))
    if stage == "eyeball_grouped":
        sample = g.select_eyeball_grouped(gcfg, df, n_questions=cfg.eyeball_n)
    elif stage == "eyeball":
        sample = g.select_eyeball_frames(gcfg, df, n=cfg.eyeball_n)
    elif stage == "blind_probe":
        sample = g.select_blind_probe(gcfg, df, n=cfg.eyeball_n)
    else:
        sample = g.select_pilot(gcfg, df, n=cfg.pilot_n)
    gen = VLMGenerator(cfg); gen.load()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    if stage == "blind_probe":
        # 32B answers the FRAME question directly, NO gold, NO scaffold — pure perception on the
        # target bucket. Uses the baseline system prompt so it mirrors real inference.
        import pandas as pd
        sysp = g._system_prompt()
        rows = []
        for r, image, _sib in _iter_rows_with_frames(cfg, sample):
            ans = gen.generate(image, str(r["question"]), system=sysp)
            correct = g.blind_match(ans, r["answer"], r["answer_format"])
            rows.append({"qID": r["qID"], "dataset": r["dataset"], "answer_format": r["answer_format"],
                         "n_objects_est": r["n_objects_est"], "question": r["question"],
                         "gold": r["answer"], "model_answer": ans, "correct": correct})
        out = cfg.run_dir / "blind_probe.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        acc = sum(x["correct"] for x in rows) / max(1, len(rows))
        logger.info("BLIND PROBE: %d Q, acc %.2f (32B perception, no gold) -> %s", len(rows), acc, out)
        return out

    if stage in ("eyeball", "eyeball_grouped"):
        rows = []
        for r, image, sib in _iter_rows_with_frames(cfg, sample):
            scaffold = gen.generate(image, g.build_vision_prompt(__import__("pandas").Series(r), sib))
            flags = g.validate_scaffold(scaffold, r["answer"], r["answer_format"])
            rows.append({"qID": r["qID"], "frameid": r["frameid"], "n_q_on_frame": r["n_q_on_frame"],
                         "dataset": r["dataset"], "ood_proxy": r["dataset"] == "heico",
                         "answer_format": r["answer_format"], "n_objects_est": r["n_objects_est"],
                         "is_single_q": r["is_single_q"], "question": r["question"], "gold": r["answer"],
                         "siblings": sib, "vision_scaffold": scaffold,
                         **{f"vlm__{k}": v for k, v in flags.items()}})
        out = cfg.run_dir / f"sample_{stage}_vision.csv"
        __import__("pandas").DataFrame(rows).to_csv(out, index=False)
        n_ok = sum(r["vlm__valid"] for r in rows)
        logger.info("EYEBALL: %d scaffolds, %d valid — INSPECT %s before the pilot", len(rows), n_ok, out)
        return out

    # stage == "pilot": write CoA train.jsonl + matched bare-gold control (SAME qIDs)
    coa_p, bare_p = cfg.run_dir / "train_coa.jsonl", cfg.run_dir / "train_bare.jsonl"
    import pandas as pd
    n = 0
    with open(coa_p, "w", encoding="utf-8") as fc, open(bare_p, "w", encoding="utf-8") as fb:
        for r, image, sib in _iter_rows_with_frames(cfg, sample):
            scaffold = gen.generate(image, g.build_vision_prompt(pd.Series(r), sib))
            flags = g.validate_scaffold(scaffold, r["answer"], r["answer_format"])
            if not flags["valid"]:
                logger.warning("drop invalid scaffold %s (%s)", r["qID"],
                               {k: flags[k] for k in ("has_all_tags", "answer_matches_gold", "answer_leak")})
                continue
            fc.write(json.dumps(g.to_sharegpt_record(gcfg, pd.Series(r), scaffold), ensure_ascii=False) + "\n")
            # matched control: identical record, assistant = bare gold
            fb.write(json.dumps(g.to_sharegpt_record(gcfg, pd.Series(r), str(r["answer"])), ensure_ascii=False) + "\n")
            n += 1
    logger.info("PILOT: wrote %d matched pairs -> %s (CoA) + %s (bare control)", n, coa_p, bare_p)
    return coa_p


if __name__ == "__main__":  # pod convenience: `python gen_onpod.py eyeball`
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stage = sys.argv[1] if len(sys.argv) > 1 else "eyeball"
    run_stage(OnPodConfig(), stage=stage)
