# Context 05 — Bottleneck Audit

## Objective
Determinar si el modelo con LoRA del rung 02 aprendió a "ver" o simplemente aprendió un "atajo lingüístico" para responder preguntas visuales.

## Setup-config
- **Baseline**: Rung 02 (LoRA).
- **Split**: `frame_ood_v1` completo (2252 ID, 4000 OOD).
- **The ONE variable**: La imagen de entrada (`a0_real`, `a1_black`, `a2_shuffled`).

## Data insight (T0)
- **El crosstab revela que `aggregation` NO es solo `number`.**
  - **En `train`**: `aggregation` se compone de 77.1% `number` y 22.3% `binary`. (Además, el total de `binary` ocupa el 8.9% de todo el split).
  - **En `val`**: `aggregation` se compone de **74.0% `number` y 25.6% `binary`**. (Y el total de `binary` pesa mucho más en `val`: 11.6%).
  Este volteo de pesos altera las asunciones iniciales de la distribución del bucket.
- Nuestro LoRA (rung 02) saca **+3.8 puntos sobre el mejor clasificador ciego posible** (55.9%) y **+15.6 sobre el que repetiría el prior de train** (44.1%).
- **Qué significa: pendiente.** Esto solo demuestra que el modelo usa *alguna* señal — no prueba que use la *imagen*. Memorizar un rasgo en la formulación de la pregunta produciría el mismo resultado. **Lo decidirá el control (`a2_shuffled`), aún sin correr.**

## Decisions
- Usar barajado de imágenes entre distintos videos (`a2_shuffled`) como brazo principal de control visual, porque preservar las estadísticas de la imagen es mejor que dar una imagen negra (OOD para ViT).
- Almacenar los tensores leídos de disco (en formato PNG sin pérdida) antes del loop de inferencia para no colapsar la caché de decord y garantizar que el brazo de control opere en condiciones byte-idénticas al baseline.
- Evaluar usando la exactitud de validación frente a la mayoría absoluta por tipo de respuesta, con una métrica comparativa `trivial_floor_val_majority_{fmt}` contra `trivial_floor_train_prior_{fmt}`.
- Reportar `bucket_mean` exclusivamente con los cuatro sub-grupos formados por `object_recognition` y `aggregation` combinados con `ID` y `OOD`.

## Results
- A la espera de la ejecución en pod.

## Next
- Correr el notebook completo.
- Tomar decisión según la tabla de contingencia de T0, validando la regla de bifurcación `A_ablada ≤ A_trivial + 5 pts`.
