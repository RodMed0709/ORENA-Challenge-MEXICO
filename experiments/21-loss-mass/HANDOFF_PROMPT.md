# Handoff prompt — paste this into the next session

---

Continúo el proyecto ORENA. Vas a construir y correr **rung 21**. Trabaja autónomo, no me
preguntes cada cosa.

## Lee primero, en este orden

1. `experiments/21-loss-mass/PLAN.md` — el plan completo, empieza en "Build order"
2. `context/decisions/loss-mass-is-token-weighted.md` — el hallazgo que este rung ataca
3. `context/decisions/undertrained-on-both-axes.md` — el hallazgo hermano, que es rung 22
4. `context/18-count-aug/CONTEXT.md` + `experiments/18-count-aug/RESULTS.csv` — el control
5. `context/INDEX.md` y `context/RULES.md` — el mapa y las reglas
6. `context/MEASURED.md` — **antes de proponer nada nuevo**

Rama: `task/r3-rung16`, todo commiteado y pusheado.

## POD

```
ssh -p [redacted-port] root@[redacted-host]
repo:   /workspace/repo
python: /workspace/envs/infer/bin/python
GPU:    RTX 5090, 32 GB     frames: /workspace/frames_cache (15,213)
export HF_HOME=/workspace/hf_cache      # el juez Qwen3-4B vive ahí, NO en ~/.cache
PYTHONPATH=/workspace/repo/src:/workspace/repo/vendor/orena-focus/src:<experiment>/_models
```

Si el pod está muerto, crea uno y móntale el volumen. Úsalo a fondo — pero lee la sección
"batch" más abajo antes de decidir cómo.

## QUÉ PASÓ LA SESIÓN ANTERIOR (rung 18, cerrado)

Rung 18 metió dos palancas de datos sobre `number`: ceros minados (L1) y parafraseo + dropout
de la cola de formato (L2). Corrió 8.55 h, tres épocas, todas puntuadas.

**Resultado: negativo fiel con mecanismo completamente diagnosticado.**

| | rung 06 ep3 | rung 18 ep3 |
|---|---|---|
| `bucket_mean` | 0.5724 | 0.5721 (**−0.0004**) |
| Spearman r (Clips) | 0.5866 | **0.6003** (+0.0137) |
| `margin_OOD` | 0.1448 | 0.1455 (+0.0007) |
| `number_margin_OOD` | 0.000000 | **−0.0098** |

Cumple la condición pre-registrada (r sube, `margin_OOD` no baja) con márgenes minúsculos.
**Pero las sondas dicen que las palancas funcionaron espectacularmente:**

| sonda 16a, `number` | rung 06 ep3 | rung 18 ep3 |
|---|---|---|
| emite `0` cuando el objeto NO está (ID) | 0.0083 | **0.9000** |
| ídem OOD | 0.0167 | **0.8333** |
| hmean ID | 0.017 | **0.803** (47×) |
| respuestas SDK-ilegales (`"1."`) ID/OOD | 0.87 | **0.0000** |

Arreglamos la capacidad 47× y el score local no se movió, porque **el eval local solo pregunta
las plantillas del corpus**, dominadas por `Clip`, donde la respuesta siempre es un entero
positivo. ⚠️ **Eso NO significa que no valga**: el test oculto son 200 videos con el generador
de ELLOS, y `mesh` y otra clase tienen cero ejemplos en todo nuestro corpus — justo donde el
`"1."` disparaba al 100%. La robustez comprada es del tipo que solo se cobra en el hidden.
No la podemos acreditar localmente; eso es distinto de que no exista.

Costos honestos: falsos ceros en PRESENT 0.0000 → 0.2750 (ID), e ilegales en `binary` 0.1458 →
0.3792 (OOD) en un formato que ninguna palanca tocó.

## LO QUE SE MIDIÓ Y CERRÓ (no lo re-litigues)

- **El `r = 0.43` del modelo era rango restringido.** En los mismos frames del humano el modelo
  saca **0.8303** contra su 0.7230. El comparador correcto en la plantilla `Clips` completa es
  **0.5866**. → `model-out-ranks-the-blind-human.md`, RULES §13b/§13c.
- **`4 × 4` no cabe en la tarjeta y batch-1 es más rápido.** Ver "batch" abajo.
- **El atractor de `clip` NO existe en `fo_class` ID** (354 emisiones vs 406 golds, ratio 0.87
  — sub-emite). Una corrección de prior de clase apuntaría a un fantasma. Muerta por 0 GPU.
  ⚠️ Sí existe en **OOD** y en la plantilla de posiciones `open_ended` (ver rung 20).
- **Multi-frame temporal: muerto.** `timestamp_start == timestamp_end` en el **100%** de las
  13,748 filas. El clip de FRAME tiene duración cero.
- **Los tracks `segment` y `procedure`** (30,000 preguntas en disco) son **temporales**,
  inservibles para FRAME. Cierra la fantasía de "hay más datos".
- **`instance_matching` (`1b`)**, hoja de `object_recognition`: tenemos **cero** ejemplos y no
  hay forma de conseguirlos. Riesgo abierto sin remedio, no pendiente.
- **CoA / SAM2:** no construir todavía. Nulo publicado en nuestro backbone exacto, y el trace
  diluye el gradiente de la respuesta **70–100×**. Rung 17 (la sonda del generador) sigue
  sin correr.
- **El colapso de la cola es real:** `Gallstone` (n=28) recall 0.036, `External Drain` (n=24)
  0.208. macro-F1 0.5166 contra accuracy 0.6478.

## EL HALLAZGO QUE ESTE RUNG ATACA

Confirmado **en el código fuente de ms-swift**, no inferido:

```
swift/trainers/seq2seq_trainer.py:196   num_items_in_batch = (labels != -100).sum()
swift/trainers/seq2seq_trainer.py:202   loss = outputs.loss.sum() / num_items_in_batch
```

`labels != -100` son exactamente los tokens de respuesta no enmascarados del batch efectivo.
La reducción es **suma de losses por token ÷ total de tokens de respuesta** = media por token.
Ninguna rama que pudiera re-normalizar se toma en nuestra config (`compute_loss_func` es None,
`label_smoother` es None).

⇒ **La contribución de una fila al gradiente es proporcional a la longitud de su respuesta.**

Medido sobre el `train.jsonl` real (`experiments/18-count-aug/RESULTS_loss_mass.txt`):

| formato | filas | % filas | tokens | **% gradiente** | tok/fila |
|---|---|---|---|---|---|
| `fo_class` | 7,343 | 50.9% | 29,564 | **62.4%** | 4.03 |
| `number` | 4,929 | 34.2% | 9,865 | **20.8%** | 2.00 |
| `open_ended`/MC | 913 | 6.3% | 5,492 | 11.6% | 6.02 |
| `binary` | 1,230 | 8.5% | 2,460 | 5.2% | 2.00 |

`number` está **infra-pesado 1.64×**, y es el **80.4% del bucket `aggregation`**, uno de los
únicos dos buckets que puebla el pre-eval del leaderboard. Diecinueve rungs metieron datos de
conteo contra una cuota de gradiente que nadie había medido.

## LA TAREA: rung 21

**Una variable: normalización por MUESTRA en vez de por token.** Cada pregunta pesa igual, sin
importar cuántos tokens tenga su respuesta.

**Control y base de datos: rung 18 ep3** (`checkpoint-2703`), `train.jsonl` idéntico, tres
épocas ya puntuadas. NO rung 06 — el offset del leaderboard (**−0.0571**, de proxy local
mean-ID a `pre_evaluation_score`) es propiedad de *nuestro val contra sus 20 videos ocultos* y
aplica a cualquier checkpoint. Encadenar a rung 06 no compra nada y tiraría la robustez de
rung 18. ⚠️ El offset descansa en **n = 1** submission.

### 🔴 La trampa que haría ilegible la corrida

Pesar cada token por `1/len(respuesta)` mientras el denominador sigue siendo
`num_items_in_batch` **encoge la magnitud total del loss ~3.3×**. Menos loss = menos gradiente
= **un cambio de learning rate disfrazado de normalización** — y **rung 22 ES el rung de LR**.
Confundirlos destruye los dos.

**Mitigación, y es un gate, no una nota:** reescala para preservar la magnitud del loss del
batch, y **asserta** que el train loss del primer paso queda a ±5% del control. Si la magnitud
se movió, el brazo está midiendo el LR y la corrida es nula.

### Build order

1. Lee `swift/plugin/loss_scale/loss_scale.py` **antes de escribir una línea**.
   `compute_loss_func` se checa ANTES de la rama por defecto (`seq2seq_trainer.py:189`), así
   que es el punto de inserción más limpio.
2. `_models/sample_norm_loss.py` — la función de pesos. Flag OFF debe ser aritmética
   **byte-idéntica**, no un número parecido.
3. `_tools/verify_gradient_share.py` — recalcula la cuota de gradiente por formato **bajo la
   nueva ponderación** y asserta que `number` llega a su cuota de filas (34.2%) ±2 pts.
   **No asumas que el arreglo funcionó; mide que funcionó**, igual que rung 18 midió su dosis.
4. `21_loss_mass.ipynb` — celda `parameters` con literales crudos + celda `derive` DEBAJO
   (la trampa de papermill), 6 épocas, eval CADA época (RULES §6b).

### Gates que RAISE

- Flag-off byte-idéntico al `train.jsonl` de rung 18 y a su loss del primer paso.
- **Magnitud del loss preservada** — ±5% contra el control. El anti-confound.
- Cuota de gradiente realizada — `number` en 34.2% ±2 pts.
- G1 — el LoRA llega al ViT, params entrenables < 500M.

### Métricas y qué cuenta como win

Lee en **los dos** buckets puntuados. Esto es una **reasignación, no comida gratis**: el
gradiente que va a `number` sale de `fo_class`, que es el 71% de `object_recognition`.

- `aggregation_ID` y `object_recognition_ID` por separado, **y su media = el proxy del
  leaderboard**, que es lo que la plataforma realmente puntúa.
- `bucket_mean`, `margin_OOD`, Spearman r en `Clips` (contra 0.6003 de rung 18 ep3).
- 🆕 **F1 balanceado por clase en `fo_class`** — todavía NO está en `frame.metrics`.
  **Añádelo antes de la corrida, no después.** Dos notas distintas lo llevan pidiendo.
- ⚠️ Vigila la plantilla de posiciones en OOD por daño colateral (ver rung 20 y
  `judge-swap-is-not-the-gap.md`).

**Pre-registrado:** un win exige que **suba el proxy del leaderboard (mean-ID)** Y que
`margin_OOD` no baje. Un alza de `aggregation_ID` pagada íntegramente por
`object_recognition_ID` es un **empate en la métrica que decide la co-autoría** y se reporta
como tal.

## 🔴 ANTES DE CONSTRUIR: agentes adversariales

Lanza **agentes adversariales en paralelo** a atacar esta propuesta antes de gastar GPU:

1. **Ataca el mecanismo.** ¿La normalización por muestra realmente mueve `aggregation`, o el
   efecto de 1.64× es demasiado chico para superar el ruido? ¿Qué dice la literatura sobre
   normalización de loss por muestra vs por token en SFT de VLM con respuestas cortas? Busca
   ablaciones con tamaños de efecto y citas concretas.
2. **Ataca el diseño.** ¿La normalización por muestra es mejor que un peso por formato? ¿El
   gate de magnitud de loss realmente descarta el confound con el LR? ¿Hay una tercera
   interpretación de las dos líneas de `seq2seq_trainer.py`?
3. **Ataca la prioridad.** Dado todo lo medido (arriba), ¿es rung 21 el mejor uso de ~9 GPU-h,
   o hay algo con mejor valor esperado? Que revise `context/MEASURED.md` y diga explícitamente
   qué candidatos verificó contra él.

**Si concuerdan en que vale la pena, córrelo — aunque el efecto esperado sea +0.1%.** El
usuario quiere intentarlo si la idea es buena. Si lo tumban con evidencia, dilo claro y
propón lo que sí. No hagas de profeta: reporta la evidencia y deja que decida.

Restricción DUA (RULES §14, vinculante): **ningún frame, pregunta o gold del challenge sale a
ninguna API externa.** Los agentes leen papers públicos y archivos locales, nada más.

## BATCH — lee esto antes de decidir cómo exprimir el pod

Medido ayer, emparejado, dentro de una sola corrida, sobre el `train.jsonl` real, en esta
tarjeta (`experiments/18-count-aug/RESULTS_vram.csv`):

| per_device × grad_accum | pico GPU | s/it (batch efectivo 16) | 3 épocas |
|---|---|---|---|
| **1 × 16** | **22,210 MiB** | **11.56** | **8.68 h** |
| 2 × 8 | 26,370 MiB | 13.16 | 9.88 h |
| 4 × 4 | **OOM** (32,076) | — | — |
| 6 × 2 | **OOM** (31,002) | — | — |

Esta tarjeta tiene 32 GB, no "mucho espacio". `per_device` 4 y 6 **no caben**, y batch-1 salió
**12% más rápido** y 4.2 GB más liviano que 2 — las secuencias de FRAME están dominadas por un
número variable de tokens visuales, así que un micro-batch de 2 hace padding hasta la muestra
más larga y el desperdicio supera la ganancia de paralelismo.

Además: **el batch efectivo 16 es lo que mantiene válida la comparación** con rungs 06 y 18.
Cambiarlo es una segunda variable en un rung que es de una sola.

⇒ **Para exprimir el pod, corre brazos EN PARALELO, no subas el micro-batch.** Si consigues una
tarjeta más grande (A100/L40S 80 GB), re-mide antes de asumir que un batch mayor ayuda — el
mecanismo del padding no depende de cuánta VRAM haya. Y si aun así quieres batch efectivo
mayor, eso es rung 22 (receta), no rung 21.

## REGLAS (no las rompas)

- Todo sobre **rung 18 ep3** (`checkpoint-2703`). ⚠️ `18b` borra el merge para liberar 17 GB;
  hay que re-mergear (`swift export --merge_lora`, ~7 min).
- **NO propongas resubir rung 06 ep3 al leaderboard. NO propongas correr semillas para
  varianza.** El usuario dijo que no a las dos, en firme.
- **NO toques `experiments/20-judge-swap/`** — otra sesión trabaja ahí.
- **Evalúa CADA época** (RULES §6b) y compara epoch-matched.
- Los gates RAISE, nunca warn. Aborta antes de cargar un modelo de 17 GB.
- Repo en inglés. **NUNCA me pongas a Claude como contribuidor de git.**
- Notebooks generan corridas; los `.py` son librerías, nunca lanzadores. Nada de `.sh` en el
  repo. Temporales al scratchpad, borrados al terminar.
- Todo resultado → `RESULTS.csv` + el ledger + una nota en `context/decisions/` si cambia un
  veredicto.

## TRAMPAS YA DESCUBIERTAS

- **papermill** inyecta su celda DESPUÉS de la celda tagged `parameters`. Los valores derivados
  van en una celda `derive` DEBAJO, o `-p SMOKE False` no toma efecto y la corrida "completa"
  es el smoke otra vez. Pon un gate que compare el ARTEFACTO realizado contra el modo
  declarado, no la variable.
- **El SDK expone el formato como `Reference._format`, NO `answer_format`.** Leer el atributo
  equivocado devuelve `""` en silencio y el dataset sale vacío sin reventar.
- **`HF_HOME` debe apuntar a `/workspace/hf_cache`** antes de las banderas offline, o el juez
  no se encuentra — y revienta DESPUÉS del merge de 17 GB y de toda la inferencia. Hay un gate
  de pre-vuelo en `18b_epoch_eval.ipynb`; cópialo.
- **`_stop` pisa un método interno de `threading.Thread`.** Si escribes un poller, no uses ese
  nombre.
- **Un probe fallido no tiene velocidad.** Si lees s/it de un run que hizo OOM, agarras un
  valor viejo del tqdm y proyectas una corrida en 0.71 h.
- **Un `pgrep -f` dentro de un script matchea su propio proceso padre** si la línea de comando
  del padre contiene el texto del script (heredoc). Escribe los scripts con `scp`, no heredoc.

Empieza por los agentes adversariales, luego el paso 1 del Build order, y ve commiteando.
