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

### I.4 Answer formats (8) — exacto vs juez
`answer_format` ∈ `{binary, number, percentage, foclass, open_ended, matching, multiple_choice, time}`.
- **Match EXACTO** (`fmt.compare()`): `binary, number, percentage, foclass, time`. → El formato canónico de la respuesta es **crítico**; un `2` vs `two` vs `2 clips` puede fallar. Aprender el formato exacto del dataset.
- **LLM-as-judge** (`JUDGE_FORMATS`): `open_ended, matching, multiple_choice`. Juez emite exactamente `CORRECT`/`INCORRECT`; veredicto = `"CORRECT" in raw and "INCORRECT" not in raw`.
- Juez default: `TransformersJudge` con **`Qwen/Qwen3.5-4B`** (`DEFAULT_JUDGE_MODEL`), o `APIJudge` (endpoint OpenAI-compatible). Majority vote sobre lista de jueces.

### I.5 Métrica
- Headline = **`pre_evaluation_score`**: media **sin pesos** sobre hasta **10 buckets** = 5 grupos de capacidad × {in-distribution, out-of-distribution}. Accuracy plana por pregunta dentro de cada bucket.
- Fila de salida: `summary_df` con `level="pre_evaluation", name="SCORE"`. Escribe `results.csv`/`summary.csv` si se pasa `output_dir`.
- Respuestas faltantes o con timeout = **incorrectas**.
- `ood` es campo de `Reference` pero el código público lo deja en `False`; lo puebla el split de test privado.

### I.6 Dependencias base (mín. pinneadas)
Python `>=3.10`. `datasets>=2.14, decord>=0.6, huggingface-hub>=0.17, opencv-python>=4.8, pandas>=2.0, numpy>=1.23, torch>=2.0, torchvision>=0.15, transformers>=4.30, tiktoken>=0.5, progiter, matplotlib, pillow, requests`. Extras de inferencia: `qwen-vl-utils, accelerate`.

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

*Constitución v1 — grounded en repo oficial clonado + PDFs. Se actualiza si cambia el repo o abre la pre-evaluación (15 jul).*
