# PLAN MAESTRO — ORENA SAVE FOCUS · FRAME Track (MICCAI 2026)

> Documento vivo. Estrategia + técnica + timeline + equipo. Base para los 3 del equipo.
>
> ⚠️ **Hechos duros verificados están en `CONSTITUTION.md`** (schema real, interfaz de submission, answer_formats, métrica, latencia). Si algo aquí choca con la constitución, gana la constitución. Correcciones clave del repo oficial: input es **clip de video** (no imagen suelta), baseline ref = **Qwen3-VL-4B**, **8 answer_formats** (no 2), métrica = media sin pesos de 10 buckets.

---

## 0. Resumen ejecutivo

- **Qué:** VQA de imagen única quirúrgica. 1 frame laparoscópico + pregunta → respuesta corta. Solo 2 tipos: **recognition** y **counting**.
- **Meta mínima (el piso que importa):** vencer **los 2 baselines** → coautoría en **Nature Biomedical Engineering**. No hay que quedar en podio para el paper.
- **Meta stretch:** top-3 FRAME (premio ~$12k USD: 1º $6k / 2º $3.6k / 3º $2.4k).
- **Backbone:** Qwen3-VL-8B (Apache-2.0, es el del repo oficial). Backup Qwen2.5-VL-7B. Wildcard 32B FP8.
- **Compute:** dev en RunPod (A100/L40S 80GB, ~$60-120 USD total). Inferencia la corren ELLOS (L40S 48GB, 5s/pregunta, Docker offline).
- **Ventana:** ~8 semanas (9 jul – 8 sep). Pre-eval abre 15 jul. Final 8 sep.
- **Eje que decide el resultado:** ser **parejo en recognition Y counting** y **robusto en OOD** (no sobreajustar a colecistectomía). Copeland castiga lo desbalanceado.

---

## 1. Equipo, roles y dinero

**3 personas en el paper Nature** (cap de 3, +excepción posible por "reasonable request"):

| Persona | Rol | Paper |
|---|---|---|
| **Tú** | Lead / orquestador. Data, prompt eng, análisis de errores, evaluación, submission, dirige agentes Claude, paper. | Autor 1 |
| **Leonardo** | AI Scientist pagado (vía Xantolo AI Lab). Motor de fine-tuning, pipeline, dockerización, RunPod. | Autor 2 |
| **3er ejecutor técnico (PhD/VLM)** | Apoya fine-tuning, experimentos, robustez OOD. | Autor 3 |
| Gilberto Ochoa | Advisor de dominio quirúrgico (ligero). NO en el equipo de 3. | Acknowledgments |

**Publicación:** solo existe **Nature BME**. NO hay MICCAI LNCS proceedings (los organizadores dijeron "No"), NO hay workshop. Autoría = regalo por vencer baseline, no por seniority.

**Dinero:** los términos de compensación y el reparto del premio son **privados** — viven fuera de este repo compartido (ver docs locales de negocio, no versionados). Este PLAN solo documenta roles y trabajo.

---

## 2. Enfoque técnico

### 2.1 Fine-tuning de Qwen3-VL-8B (LoRA/QLoRA)

Partimos de **QLoRA 4-bit (NF4, double quant)** para caber en la L40S/A100 80GB de dev con margen. Config de arranque:

- **rank r=16, alpha=32** (ratio 2:1), `lora_dropout=0.05`.
- **Target modules:** solo capas lineales del LLM (`q_proj, k_proj, v_proj, o_proj, gate/up/down_proj`). **Congelar el vision encoder y el merger/projector** en la primera pasada; si tras evaluar el cuello de botella es percepción (no lenguaje), desbloqueamos LoRA sobre el visual con lr menor.
- **lr=1e-4** (cosine, warmup 3%), **1–2 epochs** sobre 20k pares (más epochs sobre-ajusta y daña OOD). **batch efectivo 32–64** vía `per_device=2–4` + grad accumulation. `bf16` cómputo, `gradient_checkpointing=on`.
- **Resolución de imagen:** fijar `min_pixels/max_pixels` (p. ej. 256×28×28 a 1280×28×28) para no explotar tokens visuales y respetar los 5 s/pregunta en inferencia.
- Guardar checkpoints por epoch y elegir por accuracy en validación con **el mismo LLM-judge** que usarán ellos (aproximado).

### 2.2 Prompt / formato de entrada

Plantilla chat unificada con **system prompt fijo** que define el dominio y las reglas de exclusión (instrumentos conectados al exterior NO son foreign objects). Estructura:

```
<system> Eres asistente quirúrgico. Responde SOLO sobre objetos
extraños retenibles (esponjas, agujas, clips, drenajes, specimen
bags, material biológico). Graspers/tijeras/trocars NO cuentan.
Responde en 1 frase breve, sin explicación.
<image> [frame RGB]
<user> [pregunta] Procedimiento: {cholecystectomy}. Fase: {ts}.
```

Metadata (procedimiento, timestamp) se inyecta como texto corto en el turno de usuario: ayuda a desambiguar sin contaminar la respuesta.

### 2.3 Formato de SALIDA (blindaje anti-judge)

El juez premia coincidencia semántica con la referencia: **respuestas cortas, canónicas, sin hedging ni justificación**. Normalizamos el target de entrenamiento a la forma canónica del dataset (mismo vocabulario, minúsculas, singular/plural consistente).

- **Recognition — bueno:** `A surgical sponge, retained, in the lower right quadrant.` — **malo:** `It looks like there might be some gauze-like material, possibly a sponge, though I'm not fully certain...`
- **Counting — bueno:** `2` (o `Two clips.` si la referencia usa palabra). — **malo:** `I can count approximately two, maybe three clips visible in the image.`

Regla: **nunca** frases de incertidumbre, disclaimers, ni repetir la pregunta. Entrenar exactamente en ese estilo hace que el modelo lo emita por defecto.

### 2.4 Recognition vs Counting

**Un solo modelo** (evita duplicar riesgo OOD), pero con **datos balanceados** por tipo y system prompt guiando el formato. Para counting, curriculum con énfasis en ejemplos numéricos y evaluación separada de accuracy exacta (error típico = off-by-one). Si counting queda débil tras epoch 1, **oversampling** de esos casos. Dos modelos solo si la brecha es grande.

### 2.5 Herramientas

**ms-swift (ModelScope Swift)** principal: soporte nativo de Qwen3-VL, QLoRA, control de `max_pixels` y freezing del visual. **LLaMA-Factory** de respaldo. Evitar `trl` puro (más plumbing). Inferencia final en **vLLM** dentro del Docker offline para cumplir los 5s.

---

## 3. Pipeline de datos

### 3.1 Carga (HF) — schema REAL (ver CONSTITUTION §I.3)

`orena-dkfz/heico-focus-vqa` + `orena-dkfz/lapchole-focus-vqa`. Campos reales de fila: `id, video, timestamp_start, timestamp_end, procedure_type, question, primary_capability, secondary_capabilities, answer_format, answer, clinical_relevance`. Se parsea en dataclasses `Request` / `Reference` del SDK `focus`. **Reusar el data loader del repo (`src/focus/data/`), no reinventarlo.** El input visual es un **clip de video** (`sample.video_path` + `fps`); nuestro código muestrea el/los frame(s). No hay campo `image`.

### 3.2 Split anti-sobreajuste (OOD) — crítico

Riesgo: 170/200 videos son chole; split aleatorio infla accuracy ID.
- **Split por `video_id`, nunca por frame** (frames del mismo video → fuga).
- **Val-OOD:** reservar HeiCo-FOCUS completo (30 videos, otros procedimientos/centros) como validación OOD.
- **Val-ID:** apartar ~15-20 videos de LapChole (por centro si el metadata lo permite) como ID held-out.
- **Reportar SIEMPRE acc-ID y acc-OOD**, optimizar hacia el mínimo/promedio (replica el peso igual ID/OOD de Copeland). Estratificar por `qtype`.

### 3.3 Preprocesamiento Qwen3-VL

Resolución dinámica (native-res + tiling). Mantener **aspect ratio original**; laparoscopía es circular sobre fondo negro → **recortar el letterbox negro** antes de tiling para no gastar tokens en píxeles vacíos. Cap `min/max_pixels` (~1280px lado mayor). No forzar 224/336 cuadrado (destruye detalle de agujas/clips pequeños).

### 3.4 Taxonomía y metadata en prompt

Inyectar la **lista cerrada de clases** y la regla dura (instrumentos externos NO cuentan). Incluir `procedure` como contexto. Ancla el vocabulario de salida al que espera el juez.

### 3.5 Augmentation

Seguro quirúrgico: brillo/contraste/gamma, specular-highlight jitter, blur leve, ruido, humo/CO₂ sintético, flip horizontal. **Nunca** rotaciones que cambien "arriba/abajo/izquierda" en preguntas de localización. Counting: mosaicos controlados + copy-paste de instancias para balancear conteos altos, regenerando `answer` numérica.

### 3.6 Data externa (documentar + liberar)

Candidatos: Cholec80/CholecT50, EndoVis/SurgVisDom, PSI-AVA, EndoVis-VQA, SSG-VQA. Uso: pretraining de dominio + balanceo de counting. Requisito: documentar procedencia y **liberar anotaciones extra** (auto-etiquetado + revisión de Gilberto) para elegibilidad al premio.

---

## 4. Estrategia para vencer los baselines

### 4.1 Anatomía

- **#1 frontier zero-shot (GPT/Gemini):** el difícil. Fuerte justo en imagen única, prior visual enorme, robusto a fraseo. 
- **#2 open-source fine-tuneado por ellos:** el alcanzable. Mismo terreno (Qwen-class); su única ventaja es su receta sobre los mismos datos. Igualamos datos + mejor ingeniería.

### 4.2 Nuestra ventaja

El frontier **nunca vio la taxonomía exacta**: qué cuenta como foreign object, la exclusión de instrumentos externos, estados (retenida vs en uso), y el vocabulario de respuesta corta del juez. Fine-tuning in-domain codifica esas reglas que ningún zero-shot infiere.

### 4.3 Medir antes del 15 jul

Correr **Qwen3-VL-8B zero-shot sobre HeiCo** con el evaluador oficial (`orena-focus`, `examples/inference.py`) → número base honesto con la métrica real. Sin ese ancla no sabemos qué aporta cada cambio.

### 4.4 Copeland + peso igual ID/OOD

Premia ser competitivo en **todos** los buckets, no dominar uno. Optimizar recognition **y** counting a la par. Peso igual ID/OOD castiga memorizar chole → mezclar HeiCo, augmentation, validar en held-out.

### 4.5 Plan de submissions (no quemar intentos)

- **Intento 0 (local, gratis):** zero-shot base + valida el pipeline Docker offline.
- **Intento 1:** LoRA in-domain balanceado ID/OOD; confirma que vence al #2.
- **Intento 2:** mejora del bucket más débil (probablemente counting) vía datos.
- Cada submission solo tras ganar en validación interna. Nunca a ciegas.

---

## 5. Riesgos y restricciones

| # | Riesgo | Impacto | Mitigación |
|---|--------|---------|------------|
| 1 | **5s/pregunta en L40S 48GB** — no cabe / no responde a tiempo | Alto (timeout = 0) | Qwen3-VL-8B bf16 (~18GB) holgado. FP8/AWQ solo si sube a 32B. vLLM. Cap resolución + `max_new_tokens ≤32`. Medir p99, no media. Greedy sin beam. |
| 2 | **LLM-judge penaliza formato** verboso aunque sea correcto | Alto (pérdida silenciosa) | System prompt de respuesta corta/literal. Post-proceso recorta prefacios. Calibrar replicando el juez. NUNCA jailbreak (=descalificación). |
| 3 | **Sobreajuste a chole → colapso OOD** (50% del peso) | Muy alto | No entrenar solo LapChole. Mezclar HeiCo + externa. Rank moderado + early stopping por acc-OOD. Validar OOD cada checkpoint. |
| 4 | **Docker offline** — build que llama a HF en runtime falla | Alto | `COPY` pesos al layer, no `from_pretrained` remoto. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Versiones pinneadas. Probar sin red. Partir del Dockerfile oficial. |
| 5 | **Equipo sin exp. VLM; bus factor 1 en Leo** | Alto | Documentar pipeline (scripts/configs/seeds) desde día 1. 3er ejecutor replica el flujo. Zero-shot como fallback. |
| 6 | **Fechas apretadas** (15 jul, 8 sep) | Medio-alto | Congelar submission zero-shot para el 15 jul. Freeze código 1 sep; última semana solo Docker + validación offline. |
| 7 | **No vencer baseline #1 → sin premio ni Nature** | Crítico | Medir brecha temprano vs GPT/Gemini. Plan B: escalar a 32B FP8 + ensemble + prompting. Atacar buckets donde el frontier es débil (counting, dominios raros). |

**Regla de oro:** mantener SIEMPRE una submission válida y > 0 (zero-shot empaquetado) antes de perseguir mejoras. Toda mejora se valida contra OOD, no solo ID. Riesgos 1/4/7 = bloqueantes de submission.

---

## 6. Timeline y reparto

### 6.1 Roadmap semana por semana

| Semana | Fechas | Objetivo | Hito |
|---|---|---|---|
| 0 | 9–14 jul | Registro GC, clonar `orena-focus`, bajar HeiCo, EDA taxonomía, harness de eval con LLM-judge | — |
| 1 | 15–21 jul | Zero-shot Qwen3-VL-8B + prompt eng; número base en leaderboard | **15 jul: pre-eval abre** |
| 2 | 22–28 jul | RunPod, pipeline LoRA/QLoRA, primer fine-tune HeiCo; bajar LapChole | — |
| 3 | 29 jul–4 ago | Iterar fine-tune, análisis de errores por bucket, tuning de datos | — |
| 4 | 5–11 ago | Fine-tune HeiCo+LapChole, ablations 8B vs 32B FP8 | — |
| 5 | 12–18 ago | Cerrar mejor checkpoint, validar vs ambos baselines | **15 ago: vencer 2 baselines** |
| 6 | 19–25 ago | Robustez OOD, augmentation, conteo | — |
| 7 | 26 ago–1 sep | Dockerización offline, latencia <5s, submission de prueba | **1 sep: cierra registro** |
| 8 | 2–8 sep | Buffer + freeze, descripción método, submission final, liberar modelo | **8 sep: final** |

Buffers: semana 8 completa de colchón; semana 6 absorbe slippage.

### 6.2 Responsabilidades (R=responsable, A=apoya, C=consultado)

| Área | Tú | Leo | 3º |
|---|---|---|---|
| Data / EDA | **R** | C | C |
| Prompt engineering | **R** | — | C |
| Pipeline fine-tune + RunPod | C | **R** | A |
| Dockerización offline | C | **R** | C |
| Análisis errores / eval | **R** | C | A |
| Robustez OOD / ablations | C | A | **R** |
| Submission GC | **R** | A | — |
| Paper | **R** | C | C |

### 6.3 Claude Code

Delegar boilerplate: dataloaders, scripts de eval/parsing, Dockerfile offline, docs del método, análisis de errores por categoría. **Descarga horas de Leo** → él solo en el motor de fine-tune y arquitectura.

---

## 7. Acciones ESTA semana (semana 0)

1. **Registrar** el equipo en grand-challenge (clon DKFZ) + aceptar términos.
2. **Clonar** `IMSY-DKFZ/orena-focus`, montar entorno, bajar `orena-dkfz/heico-focus-vqa`.
3. **Zero-shot** Qwen3-VL-8B sobre HeiCo local → número de referencia antes del 15 jul.
4. **Harness de eval** con LLM-as-judge replicando la métrica oficial (Copeland por buckets).
5. Reunión con Gilberto (advisor) — mañana.
6. Cerrar propuesta con Leonardo (`PROPUESTA-LEO.docx`).

---

*Plan generado con orquestación multi-agente. Ajustable conforme abra la pre-evaluación y confirmemos el esquema real de datos.*
