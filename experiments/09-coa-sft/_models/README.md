# `_models/` — engines only (09 CoA-format SFT)

Importable engines. **Never launchers** — the notebook (`../09_coa_sft.ipynb`) constructs a
config and calls `main(cfg, stage=...)`. No `run_*.py`, no `.sh`.

- `coa_scaffold_gen.py` — the CoA scaffold **generator**: loads the train parquets, builds
  per-frame sibling fact-sheets, stratified-samples, constructs the reverse-generation
  prompt (gold-anchored, no pixels), validates a returned scaffold (format / answer-match /
  answer-leak), and assembles the rung-02-parity ShareGPT record. Backend is **pluggable**
  (a `prompt -> completion` callable) so the ~50 sample can be filled by deepseek / Claude
  this session and the full 13.7k by an API / VLM-hybrid model later — a config flip, not a
  rewrite. The actual generation calls (deepseek MCP / Claude) are driven by the
  orchestrator; the engine is pure and testable.
