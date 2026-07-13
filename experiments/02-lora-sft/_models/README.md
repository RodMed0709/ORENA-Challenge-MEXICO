# _models/ — engines only

| Engine | Used by rung | Notebook | Latest run |
|---|---|---|---|
| `lora_sft_train.py` | 02 | `02_lora_sft.ipynb` | `02_lora_sft_v1` |

`lora_sft_train.py` — ms-swift LoRA fine-tune engine. Importable; the notebook calls
`main(cfg, stage=...)` for stages `export → train → merge`. Eval reuses
`frame.run.run_baseline` on the merged checkpoint; the Δ vs zero-shot uses `frame.delta`.
Not a hand-run launcher.
