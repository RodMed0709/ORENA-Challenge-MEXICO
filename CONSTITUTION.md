# CONSTITUCIÓN DEL PROYECTO — ORENA FOCUS · FRAME Track

> **Fuente de verdad de hechos duros y reglas no-negociables.** Todo spec, plan o código debe respetar esto. Verificado contra el repo oficial `IMSY-DKFZ/orena-focus` (pip `orena-focus`, módulo `focus`) + PDFs del challenge. Si algo aquí choca con una suposición en otro doc, **gana esta constitución**.

---

## I. Hechos verificados del repo oficial (NO inventar sobre esto)

### I.1 Naturaleza del repo
- Es un **SDK/toolkit**, NO un esqueleto de submission. Paquete pip: `orena-focus`. Import: `import focus`.
- **NO trae** Dockerfile, config de grand-challenge, ni submission template. Esos vienen de la plataforma (challenge website), no del repo.
- El pipeline de referencia baja pesos en vivo de HF (`Qwen/Qwen3-VL-4B-Instruct`). Para submission real hay que **empaquetar offline** (ver §IV).

### I.2 Contrato de submission (la interfaz que implementamos)
- No hay clase base `Model` que heredar. El contrato de facto (de `examples/inference.py`):
  ```python
  class <NuestroEngine>:
      def load(self): ...
      def predict(self, sample: VideoSample) -> str: ...
  ```
- **Entrada:** `sample.request.question` (str) + `sample.video_path` (mp4 temporal en disco) + `sample.fps`.
  - ⚠️ **El input visual es VIDEO, no imagen suelta.** FRAME rebana un clip corto; igual recibimos un path de video + fps. Nuestro código muestrea frame(s) del clip.
- **Salida:** el texto crudo del modelo, recolectado como:
  ```python
  Response(qID=req.qID, content=prediction, latency=latency)
  ```
  Campos: `qID: str`, `content: str` (texto crudo del VLM), `latency: float` (segundos).

### I.3 Schema de datos (VERBATIM — nombres reales de campo)
Fila cruda de HuggingFace (parseada en `base_dataset._parse_row`):
```
id, video, timestamp_start, timestamp_end, procedure_type,
question, primary_capability, secondary_capabilities (list),
answer_format, answer, clinical_relevance (bool)
```
- Timestamps = strings `"hh:mm:ss"`.
- Se parsea en dos dataclasses:
  - **`Request`**: `qID:str, videoID:str, start_time:float, end_time:float, procedure_type:str, question:str`
  - **`Reference`**: `qID:str, primary:Capability, _format:str, answer:str, format_kwargs:dict, secondaries:tuple, ood:bool=False, clinical:bool`
- ❌ NO existen los campos `image`, `question_type`, ni `metadata`. No asumirlos.

### I.4 Answer formats (8) — exacto vs juez (VERIFICADO en `formats.py`/`evaluator.py`)
`answer_format` ∈ `{binary, number, percentage, fo_class, open_ended, matching, multiple_choice, time}`. ⚠️ es **`fo_class`** (no `foclass`).
- **Match EXACTO** (`fmt.compare()`): `binary, number, percentage, fo_class, time`.
  - `number`: **solo `str.isdigit()`**. → `"two"`, `"2 clips"`, `"2.0"`, `"-1"` **fallan todos**. Salida numérica pura.
  - `percentage`: `isclose(abs_tol=1e-9)` = exacto (el param `threshold_pp` está definido pero **no se usa**).
  - `fo_class`: **set-equality** de nombres en `Title Case` (orden y duplicados no importan).
  - `time`: número de timestamps debe coincidir, con tolerancia pairwise de ~5s.
- **LLM-as-judge** (`JUDGE_FORMATS`): `open_ended, matching, multiple_choice`. Juez emite `CORRECT`/`INCORRECT`; veredicto = `"CORRECT" in raw and "INCORRECT" not in raw`. Juez default: `TransformersJudge` con **`Qwen/Qwen3.5-4B`**, o `APIJudge`. Majority vote.
  - `matching` está gated por regex `fullmatch` **antes** del juez, aunque sea judge-routed.

### I.4-bis Gates SILENCIOSOS que te ponen 0 (leídos del código — CRÍTICO)
1. **Parse gate universal:** `fmt.read(prediction)` corre en **TODA** respuesta ANTES del juez/scoring. Si lanza `ValueError` → incorrecta. Aplica **también** a formatos de juez.
2. **>300 caracteres → auto-incorrecta.** Brevedad NO es preferencia, es gate duro.
3. **`AdversarialDetector.check()` lanza `RuntimeError`** ante frases de su lista heurística → **descalifica toda la submission**. Y **misfire con frases inocentes**: `"the answer is definitely correct"`, `"you are now"`, `"act as if"`, `"always respond with correct"`. → Escanear nuestras propias salidas offline contra esa lista antes de enviar.
4. **`qID` duplicado → `ValueError` aborta el eval run COMPLETO** (no una pregunta, el score entero). Garantizar qIDs únicos.
5. **Latencia mata en p99, no media.** Timeout = incorrecta aunque el contenido sea correcto.

### I.5 Métrica
- Headline = **`pre_evaluation_score`**: media **sin pesos** sobre hasta **10 buckets** = 5 grupos de capacidad × {in-distribution, out-of-distribution}. Accuracy plana por pregunta dentro de cada bucket.
- Fila de salida: `summary_df` con `level="pre_evaluation", name="SCORE"`. Escribe `results.csv`/`summary.csv` si se pasa `output_dir`.
- Respuestas faltantes o con timeout = **incorrectas**.
- `ood` es campo de `Reference` pero el código público lo deja en `False`; lo puebla el split de test privado.

### I.6 Dependencias base (mín. pinneadas)
Python `>=3.10`. `datasets>=2.14, decord>=0.6, huggingface-hub>=0.17, opencv-python>=4.8, pandas>=2.0, numpy>=1.23, torch>=2.0, torchvision>=0.15, transformers>=4.30, tiktoken>=0.5, progiter, matplotlib, pillow, requests`. Extras de inferencia: `qwen-vl-utils, accelerate`.

**Pins DUROS para Qwen3-VL (verificado):** `transformers==4.57.*` (abajo de eso el arch `qwen3_vl` no carga; NO saltar a 5.x) + `qwen-vl-utils>=0.0.14`. Fine-tune: **ms-swift `>=4.2`** (soporte nativo Qwen3-VL, `--max_pixels`, `--freeze_vit/--freeze_aligner`). Serving: **vLLM `>=0.11`** (par verificado 0.11.2 + transformers 4.57). **Dos envs Python separados** (train: ms-swift+bitsandbytes+flash-attn; serve: vLLM) — pins de torch/flash-attn chocan; el artefacto de handoff = pesos LoRA ya merged. Cuant: **bf16 LoRA** pal 8B (cabe en 80GB, sin pérdida NF4); QLoRA NF4 solo pal wildcard 32B; en L40S (Ada CC 8.9) el lever es **FP8 w8a8** si p99 aprieta.

---

## II. Restricciones de la plataforma (de los PDFs — no del repo)

- **Hardware de evaluación (lo corren ELLOS):** 1× NVIDIA **L40S 48GB**.
- **Latencia FRAME = 5.0 s/pregunta** (`TRACK_MAX_LATENCY = {FRAME:5.0, SEGMENT:15.0, PROCEDURE:30.0}`). Pasarse = respuesta incorrecta.
- **Docker OFFLINE**: sin internet en inferencia. Pesos y deps empaquetados en el contenedor.
- **Submission** vía grand-challenge (clon DKFZ).
- Se permite data pública externa + modelos preentrenados públicos, **documentando** y **liberando** anotaciones extra.
- **Modelo final debe liberarse open-source** para optar a premio.

---

## III. Objetivo y línea roja competitiva

- **Piso (lo que importa):** vencer **los 2 baselines** → coautoría **Nature BME**. Baseline #1 = frontier zero-shot; **baseline #2 = Qwen3-VL-4B fine-tuneado por ellos**.
- **No sobreajustar a colecistectomía.** OOD pesa igual que ID en el score. Un modelo que brilla ID y colapsa OOD pierde.
- **Ser parejo en las 5 capacidades.** Media sin pesos sobre buckets: ser malísimo en un bucket hunde el score global.

---

## IV. Reglas duras de ingeniería (no-negociables)

1. **Siempre existe una submission válida > 0.** El zero-shot empaquetado es el piso; nunca romperlo persiguiendo mejoras.
2. **Toda mejora se valida contra OOD, no solo ID.** Reportar acc-ID y acc-OOD en cada checkpoint. Descartar el que gane ID pero pierda OOD.
3. **Split por `video`/`videoID`, NUNCA por frame.** Frames del mismo video = fuga de datos.
4. **Docker offline de verdad:** `COPY` pesos al layer, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, versiones pinneadas, wheels vendored. Probar el contenedor **con la red desconectada** antes de subir.
5. **Formato de salida canónico.** Para formatos de match exacto, la respuesta debe calzar el formato del dataset. Para formatos de juez: cortas, sin hedging, sin repetir la pregunta, sin disclaimers.
6. **NUNCA prompt-injection al juez.** Hay `AdversarialDetector` → detección = descalificación inmediata.
7. **Medir latencia p99 en L40S**, no la media, antes de congelar. `max_new_tokens` bajo, greedy (sin beam), cap de resolución/tokens visuales.
8. **Reproducibilidad:** seeds fijas, configs versionadas, `requirements` pinneado desde el día 1. Bus factor > 1: el pipeline debe poder correrlo cualquiera del equipo.

---

## V. Reglas de trabajo colaborativo (git / sync)

Repo privado: `RodMed0709/ORENA-Challenge-MEXICO`. Sincroniza **local + RunPod + compu del amigo**.

1. **`main` siempre funcional.** Trabajo en ramas (`feat/`, `exp/`, `fix/`). Merge a `main` solo lo que corre.
2. **`git pull` antes de empezar, `push` al terminar.** Evitar divergencias entre las 3 máquinas.
3. **NUNCA commitear:** `.secrets.env`, tokens, datos crudos, pesos, checkpoints, videos. (Ya bloqueado en `.gitignore`.)
4. **Datos y pesos viven en RunPod / HF**, no en git. El repo lleva **código, specs, configs, docs**.
5. **Commits atómicos** con mensaje claro. Un experimento = una rama + su config versionada.
6. **Specs antes que código** (SDD): cada feature arranca de un spec en `specs/`, no de teclear directo.

---

## VI. Reglas de equipo

- 3 autores Nature: tú (lead), Leonardo (AI Scientist, Xantolo AI Lab), 3er ejecutor técnico. Gilberto = advisor (acknowledgments).
- Solo existe **Nature BME** como publicación. No hay LNCS proceedings ni workshop.
- Dinero de Leonardo y reparto de premio: ver `PLAN.md` §1 y `PROPUESTA-LEO.docx`.

---

## VII. Reglas de evidencia y literatura

1. **Corpus:** la literatura vive en `literature/` — `INDEX.md` rankeado por tiers (versionado), PDFs en `literature/pdfs/` (local, gitignored). ~30 papers indexados; los paywalled los provee el usuario.
2. **Recomendaciones = con cita.** Cuando se pida una recomendación técnica/de método, NO opinar al aire: buscar en el **corpus + web**, citar qué hizo cada investigación y si funcionó, y marcar confianza. Motor: MCP **`RAG-Research`** (`verify_claim_against_corpus`, `tier_papers`, `build_bibliography`, `draft_methods`, `draft_section`, `export_docx`).
3. **Paper writing** es entregable del proyecto. Mínimo obligatorio: la **descripción de método** del submission (`SUB-01`). Opcional: paper de método propio (autores a discreción del lead), redactado con el MCP RAG-Research contra el corpus.
4. **Trazabilidad:** toda afirmación que entre al paper debe verificarse contra el corpus (`verify_claim_against_corpus`) antes de incluirse.

---
*Constitución v1 — grounded en repo oficial clonado + PDFs + corpus de literatura. Se actualiza si cambia el repo o abre la pre-evaluación (15 jul).*
