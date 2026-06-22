# Optimizar la detección de régimen: reducir el retardo del HMM — EXPLORATORIO (NO canónico)

> ⚠️ **Esto es trabajo en pruebas, NO documentación válida de la tesis.** Nada de aquí debe tomarse
> como referencia para la memoria del TFG mientras no se valide y Raquel lo apruebe. No tocar
> `MANUAL.md`, `RESULTADOS_OBJETIVO.md`, `DECISIONES_ESENCIALES.md`, los capítulos de `tesis/` ni la
> BITACORA canónica con estas cifras. Todo lo relativo a las variantes de estimación de régimen y su
> retardo vive AQUÍ hasta nuevo aviso.
>
> Rama: `feat/quant-validation-panel`. Fecha: 2026-06-22.

---

## 0. Qué es esto

El valor de STRATA se concentra en el **canal régimen** (detector RAM = HMM gaussiano de 3 estados).
Una sospecha razonable: el régimen tiene **retardo** —cuando el HMM declara Crisis, la caída ya lleva
días— y eso podría degradar la señal. Este experimento pregunta, de forma falsable:

> **¿Se puede reducir el retardo de la estimación de régimen sin look-ahead, y eso mejora la estrategia?**

**Aclaración previa que enmarca todo.** La inferencia de régimen **ya es causal**: se usa
`predict_proba_filtered` (algoritmo forward, solo `x_{1:t}`; `core/hmm.py:178`, verificado por
`tests/test_hmm.py::test_filtered_no_lookahead`). El retardo **no es un bug de look-ahead** —eso ya
está resuelto— sino retardo *real de reactividad*, con dos fuentes atacables:

1. **La feature de volatilidad** `RV^{21}_t = std(r_{t-20:t})·√252`: una ventana móvil de 21 días
   introduce ~10 días de lag medio (`core/features.py:realized_vol_annualized`).
2. **La inercia del HMM**: la diagonal alta de la matriz de transición hace que el `argmax` del
   posterior tarde en voltear.

Se prueban **8 variantes** que atacan una u otra fuente, manteniendo causalidad estricta, y se mide
(a) el **lag** sobre la historia completa (2000→2026, capta GFC-2008/COVID-2020/bear-2022) y (b) el
**downstream** sobre el OOS (2024-10→2026-06) con la estrategia **"Régimen" pura** (posición = signo
del régimen dominante data-driven, sin agente ni vol-target: aísla la calidad del régimen) en los
**13 activos**, con todas las métricas y baselines (B&H, ZeroR).

Artefactos (reproducibles, cifras desde JSON):
- `experiments/optimizar_deteccion_regimenes.py` → `outputs/experiments/optimizar_deteccion_regimenes.json`.
- `core/features.py` — añadida `ewma_vol_annualized` (RiskMetrics, causal); no altera ningún pipeline.
- **NO** toca `cache/models/*.pkl`: los HMM se recalibran en memoria por activo (≤ 2024-09, sin fuga).

Las 8 variantes (todas K=3, covarianza `full`, 10 seeds, `seed=42`, calibración ≤ 2024-09):

| Variante | Feature(s) `[r, vol(, extra)]` | Regla de estado | Ataca |
|---|---|---|---|
| **V0_rv21** (control) | `[r, rv21]` | argmax | — (canónica) |
| **V1a_rv10** | `[r, rv10]` | argmax | lag de la ventana |
| **V1b_rv5** | `[r, rv5]` | argmax | lag de la ventana |
| **V2_ewma094** | `[r, EWMA(λ=0.94)]` | argmax | lag de la ventana (RiskMetrics) |
| **V2b_ewma097** | `[r, EWMA(λ=0.97)]` | argmax | lag de la ventana |
| **V3_multi** | `[r, rv21, EWMA(0.94)]` | argmax | reactividad sin mover la etiqueta |
| **V4a_th04** | `[r, rv21]` | Crisis si `crisis_prob>0.4` | inercia del argmax |
| **V4b_th03** | `[r, rv21]` | Crisis si `crisis_prob>0.3` | inercia del argmax |

Métricas de lag: **onset_lag** (días bursátiles hasta declarar Crisis tras una entrada objetiva en
drawdown, definida como el cruce del drawdown por debajo de −10% desde el máximo móvil),
**detección** (fracción de esos onsets capturados en 63 días), **whipsaw/año** (transiciones de
estado), **crisis_precision** (fracción de días-Crisis con drawdown real ≤ −5%) y **xcorr_kstar**
(`corr(sev_t, |dd_{t+k}|)`; convención: `k>0` ⇒ el régimen anticipa el drawdown).

---

## 1. Resultado de cabecera (promedios cross-activo, 13 activos)

| Variante | onset_lag | detección | whipsaw/año | crisis_prec | xcorr k* | Sharpe OOS | acc OOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **V0_rv21** | 9,58 | 0,543 | **7,87** | 0,934 | +2,7 | 0,345 | 0,527 |
| V1a_rv10 | 9,31 | 0,636 | 14,9 | 0,932 | +4,2 | 0,376 | 0,524 |
| **V1b_rv5** | **6,23** | **0,705** | 30,9 | 0,937 | +8,1 | **0,504** | **0,533** |
| V2_ewma094 | 11,2 | 0,527 | 6,87 | 0,942 | +2,5 | 0,380 | 0,527 |
| V2b_ewma097 | 8,08 | 0,436 | **4,06** | 0,927 | +1,2 | 0,480 | 0,525 |
| V3_multi | 9,27 | 0,472 | 6,37 | 0,920 | +1,2 | 0,322 | 0,526 |
| V4a_th04 | 9,31 | 0,545 | 7,90 | 0,934 | +2,7 | 0,295 | 0,526 |
| V4b_th03 | 8,85 | 0,557 | 7,93 | 0,933 | +2,7 | 0,290 | 0,526 |

Tres lecturas:

**(1) El retardo NO está en la regla de decisión, está en la feature.** Las variantes V4 (umbral en
`crisis_prob`, misma feature rv21) dan onset_lag 9,3/8,8 ≈ V0 (9,6), **mismo** whipsaw (7,9) y **mismo**
xcorr (+2,7); y el downstream incluso empeora (Sharpe 0,29 vs 0,35). Adelantar el disparo del estado
Crisis no reduce el lag: solo añade algo de ruido. **Refuta** la hipótesis de que la inercia del
argmax es el cuello de botella. El que manda es el **suavizado de la volatilidad**.

**(2) Acelerar la feature es un trade-off lag↔whipsaw estricto.** `rv5` logra el mejor lag (6,2 vs
9,6), la mejor detección (0,70 vs 0,54) y la mayor anticipación (k*=+8,1) — **pero** cuadruplica el
whipsaw (30,9 vs 7,9/año). `EWMA(0,97)` está en la esquina opuesta: el más suave (4,1) pero el más
lento en detectar (0,44). `rv21` (canónica) es un punto de Pareto intermedio sensato.

**(3) La precisión de Crisis es alta y estable (~0,92–0,94) en todas.** Cuando el HMM dice Crisis, el
precio está en un drawdown ≥5% el ~93% de las veces, independientemente de la velocidad. El lag es de
**oportunidad**, no de **falsas alarmas**: las variantes rápidas no pierden precisión, pero parten la
Crisis en más episodios y más cortos (más whipsaw).

---

## 2. ¿Mejora el downstream? Tests honestos contra V0

Pooled cross-activo clusterizado por fecha (`core.validation.panel_pooled_test`, bloques circulares
√(nº fechas); respeta autocorrelación serial y correlación transversal):

| Variante | Δacc vs V0 | IC95 | p | Δpnl p | Sharpe>V0 | lag<V0 |
|---|---:|---:|---:|---:|:--:|:--:|
| V1a_rv10 | −0,0023 | [−0,011, +0,005] | 0,71 | 0,38 | 3/13 | 7/13 |
| **V1b_rv5** | **+0,0065** | [−0,001, +0,013] | **0,038** | 0,097 | 6/13 | 5/13 |
| V2_ewma094 | +0,0007 | [−0,005, +0,007] | 0,43 | 0,14 | 4/13 | 5/13 |
| V2b_ewma097 | −0,0014 | [−0,009, +0,005] | 0,66 | 0,12 | 6/13 | 7/13 |
| V3_multi | −0,0011 | [−0,007, +0,005] | 0,65 | 0,30 | 3/13 | 6/13 |
| V4a_th04 | −0,0002 | — | 0,63 | 0,92 | 1/13 | 3/13 |
| V4b_th03 | −0,0009 | — | 0,75 | 0,88 | 1/13 | 6/13 |

**Ningún variante bate a V0 de forma robusta.** El único candidato es `rv5`: Δacc pooled +0,0065
(p=0,038) y el mejor Sharpe medio (0,50 vs 0,35). Pero:

- Es **1 de 7** pruebas → Bonferroni ×7 ⇒ p≈0,27. **No sobrevive a la multiplicidad.**
- La mejora de accuracy es **diminuta** (0,533 vs 0,527) y el Δpnl pooled no llega a significativo
  (p=0,097).
- El **whipsaw ×4** implica que, con costes de transacción reales, la ganancia marginal de Sharpe se
  erosiona (el turnover OOS también se cuadruplica).
- Por activo es mixto: mejor que V0 en 6/13, peor en 7/13.

**Dónde se concentra lo poco que mejora.** La ganancia de Sharpe de `rv5` aparece **exactamente** en
los nombres volátiles/bajistas con cambios de régimen reales en el OOS: SMCI (−0,46→+0,08), MSTR
(−1,57→−0,42), XLE (+0,67→+1,11), BAC (+0,70→+0,90). En los **índices/ETF amplios de leverage fuerte**
(SPY, QQQ, DIA, IWM) —donde el régimen *debería* mandar según la tesis— el régimen **no voltea** en
este OOS alcista, así que reducir el lag **no cambia nada** downstream: SPY y DIA quedan idénticos a
V0, QQQ/IWM cambian de forma marginal. La estrategia Régimen ahí coincide con B&H.

---

## 3. Interpretación

- **El retardo es reducible** (rv10/rv5 detectan los onsets de 2008/2020/2022 antes y capturan más),
  pero la palanca es la **feature de volatilidad**, no la regla de decisión (V4 lo refuta), ni un
  EWMA (V2 no mejora el lag de la ventana).
- **Reducir el lag no fabrica cambios de régimen que no están.** Coherente con el límite central del
  proyecto ([[raquel-confusiones-io-m8-m10]] y Decisión #16): el canal régimen solo aporta donde el
  agente discrepa de un régimen que **voltea** y el OOS **contiene** ese volteo. En un OOS con
  tendencia (2024-10→2026-06), el régimen de los índices no voltea y el lag es irrelevante para el
  P&L; donde sí hay volteo (acciones volátiles bajistas) la mejora existe pero es marginal y no
  sobrevive a multiplicidad + costes.
- **`rv21` (canónica) es un punto de Pareto defendible.** `rv10` sería una subida de reactividad leve
  con whipsaw moderado (14,9), pero no mueve el titular. **No justifica cambiar el pipeline canónico.**

**Veredicto honesto: el retardo del régimen se puede recortar, pero NO es la causa de que la
estrategia no concluya. El cuello de botella no es la latencia del detector, es la ausencia de
cambios de régimen explotables en el OOS y la fuerza del leverage effect por activo.**

---

## 4. Preguntas abiertas / próximos pasos

1. **Condicionar por tramos.** El downstream se midió sobre todo el OOS. ¿Y si se aísla a los tramos
   de drawdown reales (pocos en 2024-10→2026-06)? Ahí es donde el lag *podría* importar; el problema
   es la potencia (n muy pequeño).
2. **OOS con crisis.** El lag de onset se mide sobre 2008/2020/2022 (dentro de calibración: compara
   reactividad del filtro, no es OOS). Un test limpio exigiría un OOS que contenga una crisis — no
   existe en esta ventana (el agente LLM solo vive post-2024-09).
3. **¿Propagar a STRATA-U / M8?** Aquí se aisló el régimen puro. Si se quisiera el efecto sobre la
   estrategia que usamos, habría que reinyectar `rv10`/`rv5` en `build_states` y re-correr STRATA-U;
   pero dado que el régimen puro ya no mejora robustamente, no parece prioritario.
4. **xcorr_kstar** es la métrica de lag más ambigua (el whipsaw de rv5 infla correlaciones líder
   espurias). Apoyarse en onset_lag + detección + whipsaw, que son más limpias.
