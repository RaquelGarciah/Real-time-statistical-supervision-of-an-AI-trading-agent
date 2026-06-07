# Decisiones clave de STRATA

Documento didáctico con las **decisiones metodológicas críticas** del proyecto, pensado para entender y explicar cómo y por qué funciona STRATA. No sustituye a la BITACORA (cronológica, exhaustiva); destila las decisiones que **hay que poder defender ante el tribunal**.

Cada decisión sigue el mismo formato:

- **Qué.** La decisión en una línea.
- **Por qué.** El problema empírico o teórico que la motiva.
- **Cómo se implementa.** Detalle técnico mínimo.
- **Qué cambia.** Efecto práctico observable.
- **Cómo se defiende.** Una o dos frases listas para la defensa oral.

---

## 1. Activo central: SPY (justificado por el *leverage effect*)

**Qué.** El experimento empírico se ejecuta sobre **SPY** (el ETF que replica el S&P 500), no sobre acciones individuales como NVDA.

**Por qué.** La arquitectura de STRATA asume que el régimen de mercado detectado por el HMM **correlaciona con la dirección esperada** del activo (Calma → drift alcista, Crisis → drift bajista). Esta asunción se sostiene en índices agregados gracias al **leverage effect** (Black 1976; Christie 1982): correlación fuerte y negativa entre retornos y volatilidad, en torno a -0,7 sobre periodos largos. En stocks individuales —especialmente growth tech— el leverage effect es débil o incluso positivo en fases de noticias buenas (un growth stock puede tener volatilidad alta en pleno rally por earnings beat), y la asunción direccional de RAM se rompe.

**Cómo se implementa.** `config.TICKER_PRIMARY = "SPY"`. Todos los experimentos del cuerpo central usan este ticker; el ensayo previo sobre NVDA (BITACORA 2026-05-15) queda como hallazgo metodológico secundario en la discusión.

**Qué cambia.** Mantener SPY garantiza la consistencia teórica del marco. El experimento sobre NVDA, conservado en `outputs/experiments/nvda/`, ilustra qué pasa cuando se viola la asunción y sirve como contraejemplo en la memoria.

**Cómo se defiende.**
> "STRATA es académicamente correcto sobre índices agregados con leverage effect documentado y conceptualmente problemático sobre activos individuales con leverage effect débil. La elección de SPY no es pragmática, está teóricamente fundamentada."

> **Matiz (2026-05-21).** Tras corregir la extensión NVDA para que use su **HMM propio** (ver §9), el contraejemplo se afina: la supervisión sí transfiere a NVDA (M8 +0,66, como en SPY); lo que no transfiere son los *priores direccionales* de RAM, calibrados al leverage effect del índice. SPY sigue siendo el activo canónico; NVDA pasa de "STRATA falla en *stocks*" a "los priores de RAM habría que re-signarlos por activo".

---

## 2. Inyección de contexto macro a las personalidades del agente

**Qué.** Antes de cada llamada al LLM, `agent/wrapper.py` antepone al prompt una **`SystemMessage` con un snapshot macro/sentimiento** del día (VIX, TNX, retornos sectoriales 1M vs SPY, momentum 1M/3M/YTD del índice). Sustituye la lectura por defecto del submódulo de fundamentales empresariales.

**Por qué.** SPY es un ETF: carece de fundamentales empresariales en sentido estricto (no tiene management, ni moat individual, ni insider trades). Las cinco personalidades de AI Hedge Fund consultan por defecto Financial Datasets API con el ticker `SPY` y reciben listas vacías → responden sistemáticamente *"insufficient data on fundamentals"* y devuelven `size = 0` con confianza muy baja (0,12-0,32). El agente opera "a ciegas", con hit rate 32,9 % sobre los 404 días originales (BITACORA 2026-05-16).

**Cómo se implementa.** Tres piezas:
1. `core/macro_features.py::build_macro_snapshot(date, ticker)` construye el dict con valores reales del día (yfinance para VIX/TNX/ETFs sectoriales; precios SPY).
2. `agent/_macro_patch.py` monkey-patchea `src.utils.llm.call_llm` del submódulo para anteponer una `SystemMessage` con el snapshot a cada prompt.
3. `agent/wrapper.py` fija el contexto con `set_macro_context()` antes de llamar al agente y lo limpia con `clear_macro_context()` en un `finally`.

**Qué cambia.** Smoke test 2024-10-15, comparativa antes/después:

| | Antes | Después |
|---|---|---|
| Confianza personalidades | [0,12; 0,32] | [0,50; 0,85] |
| *"Insufficient data"* | 5/5 personalidades | 0/5 |
| Decisión Portfolio Manager | `short -0,17 conf 0,32` | `short -0,20 conf 0,85` |
| Citas a métricas concretas | ninguna | "VIX 20,64 percentil 0,93", "YTD +23,82 %", "XLK lidera" |

**Cómo se defiende.**
> "Como SPY no tiene fundamentales empresariales, el agente original quedaba mudo. Inyectamos un contexto macro (VIX, tipos, rotación sectorial) para que las personalidades razonen sobre el mercado agregado, manteniendo su lógica de personalidad pero alimentándolas con la información que sí aplica a un ETF."

---

## 3. Features del HMM: `realized_vol_21d` en lugar de `log(VIX)`

**Qué.** El HMM de tres estados se ajusta sobre `[log_return, realized_vol_21d × √252]`. La segunda feature es la **volatilidad realizada anualizada a 21 días bursátiles**, no `log(VIX)`.

**Por qué.** El VIX mide volatilidad **implícita** (forward-looking), contaminada por la prima de riesgo: en periodos de incertidumbre (elecciones, Fed) sube aunque la volatilidad realizada sea baja. Sobre 21 días de octubre 2024 con el HMM viejo: 20/21 días clasificados como Estrés porque VIX rondaba 20-23, mientras que la volatilidad realizada de SPY era apenas 10-12% (mercado en realidad tranquilo). RAM no podía intervenir porque su trigger es la masa en Calma/Crisis, no Estrés.

**Cómo se implementa.**
- `core/features.py::realized_vol_annualized(returns, window=21)` = `rolling.std × √252`.
- `core/hmm.py::RegimeHMM.fit` con **10 inicializaciones** (`random_state = SEED..SEED+9`), `n_iter=1000`, conserva la corrida con `model.score(X)` máximo (evita mínimos locales del Baum-Welch).
- **Estandarización por columna** dentro del módulo: `fit` calcula `feature_means_` y `feature_stds_` y entrena sobre `(X − μ)/σ`; `predict_states` y `predict_proba` reaplican el mismo escalado. Es **imprescindible** porque `ret_log` (std ≈ 0,01) y `rv_21_ann` (std ≈ 0,11) difieren ~10× en escala: sin escalar, la covarianza emisora full queda dominada por la dimensión vol, Crisis y Estrés se vuelven indistinguibles, y la matriz de transición se hace excesivamente persistente (diagonales > 0,99). Con escalado las diagonales bajan a 0,98 / 0,97 / 0,98 y la transición directa Crisis → Calma se hace exactamente 0 (paso obligado por Estrés, financieramente realista). Ver entrada BITACORA 2026-05-19 "Estandarización de features en el HMM".
- Ordenamiento de estados por **media de la segunda columna** (`realized_vol`) ascendente → Calma=0 (vol baja), Estrés=1 (media), Crisis=2 (alta). Determinista por construcción y robusto a la estandarización (es monótona por columna).

**Qué cambia.** Mismas 21 fechas de octubre 2024:

| Régimen | HMM viejo (log_vix) | HMM nuevo (rv_21d) |
|---|---:|---:|
| Calma | 0/21 | **19/21** |
| Estrés | 20/21 | 1/21 |
| Crisis | 1/21 | 1/21 |

Con la clasificación corregida, RAM detecta el desajuste del agente (short en Calma) y STRATA empieza a intervenir: M7 Sharpe pasa de -0,90 a +4,75 con 17 intervenciones.

**Cómo se defiende.**
> "El régimen direccional del leverage effect se refiere a volatilidad observada, no implícita. El VIX contiene prima de riesgo que distorsiona la clasificación en periodos de incertidumbre sin volatilidad real (como pre-elecciones). Sustituimos por volatilidad realizada a 21 días anualizada, siguiendo la receta documentada en `replicar_regimen_mercado.md`."

---

## 4. Política simétrica de RAM con el *leverage effect*

**Qué.** El detector RAM penaliza al agente cuando la dirección de su posición es contraria al drift implícito del régimen detectado:

- **Calma** → penaliza **short** (drift alcista, short es contrarian).
- **Estrés** → no penaliza nada (régimen indefinido direccionalmente).
- **Crisis** → penaliza **long** (drift bajista, long es contrarian).

**Por qué.** La regla original era *"Crisis → flat (long y short ambos inconsistentes)"*. Sobre el día 2024-10-01 (Crisis P=1,0 según HMM, agente short -0,14, SPY cayó -0,94 %), RAM forzó el size a 0 y eliminó el rendimiento positivo del agente que había acertado. El short en Crisis era **coherente** con el leverage effect (Crisis ≈ caída), no inconsistente. La regla original era averse-to-risk, no profit-maximizing, y contradecía la justificación teórica de SPY.

**Cómo se implementa.** Cambio puntual en `strata/detectors.py::ram_detector`:

```python
# Antes:
inconsistency = 0.0
if agent_sign < 0:
    inconsistency += regime_probs.get("Calma", 0.0)
if agent_sign != 0:   # cualquier no-cero
    inconsistency += regime_probs.get("Crisis", 0.0)

# Ahora:
inconsistency = 0.0
if agent_sign < 0:
    inconsistency += regime_probs.get("Calma", 0.0)
if agent_sign > 0:   # solo long
    inconsistency += regime_probs.get("Crisis", 0.0)
```

**Qué cambia.** Política simétrica respecto a la dirección del régimen. En 2024-10-01: short en Crisis → score 0 (consistente) en lugar de score 1.0 (incoherente). RAM ya no anula posiciones short bien posicionadas en mercados bajistas.

**Cómo se defiende.**
> "El leverage effect en índices agregados implica Crisis ↔ drift bajista. La acción coherente en Crisis es el opuesto direccional de Calma (donde se permite long), no flat. La política simétrica `Calma → penaliza short, Crisis → penaliza long, Estrés permisivo` alinea la regla de RAM con la justificación teórica del activo."

---

## 5. Umbrales de PSA y GSO recalibrados por percentiles

**Qué.** Los umbrales de severidad `(low, medium, high)` de PSA y GSO se fijan a los percentiles `(P95, P99, max)` de la distribución de scores que produciría una política de referencia (sizing GARCH × HMM óptimo) sobre el periodo de calibración 2000-2024-09. RAM mantiene los defaults uniformes `(0,2 / 0,4 / 0,7)` porque su score es una masa de probabilidad sobre regímenes, no datos.

**Por qué.** Los umbrales originales eran uniformes y arbitrarios (`high = 0,7` para los tres detectores). PSA mide `cp_prob` de BOCPD, que en práctica rara vez supera 0,02 sobre series financieras estables → **PSA nunca dispara con umbral 0,7**. GSO mide exceso relativo a la banda GARCH, que puede llegar a 10× en periodos de baja volatilidad → con umbral 0,7 dispara constantemente sin información. Resultado: detectores efectivamente desactivados o saturados.

**Cómo se implementa.**
1. `experiments/recalibrate_strata_thresholds.py` calcula la distribución de cada score sobre la calibración (6 025 obs), persiste percentiles en `cache/models/strata_thresholds.json`.
2. `strata/detectors.py::_load_thresholds` lee ese JSON si existe y asigna `(max, P99, P95)` como umbrales `(high, medium, low)` para PSA y GSO.
3. Para PSA: `low = 0,023` (P95), `medium = 0,648` (P99), `high = 1,0` (max).
4. Para GSO: `low = 2,37` (P95), `medium = 5,57` (P99), `high = 10,3` (max).

**Qué cambia.** Los detectores tienen sensibilidad **calibrada a los datos reales**, no a un número mágico. Frecuencia de activación esperada: ~5 % en severidad `low`, ~1 % en `medium`, ~0,1 % en `high`.

**Cómo se defiende.**
> "Los umbrales por defecto del diseño preliminar eran arbitrarios y no reflejaban la distribución empírica de scores. Recalibramos por percentiles de la distribución observada en 24 años de calibración, lo que garantiza que la severidad `high` corresponda a anomalías genuinamente raras (top 0,1 %), no a saturación constante o silencio total."

---

## 6. Comparativa unificada de 9 configuraciones (M1–M9)

**Qué.** La validación empírica del TFG es **un único experimento** con nueve configuraciones evaluadas sobre el mismo OOS (2024-10-01 → cierre):

| # | Categoría | Configuración |
|---|---|---|
| M1 | Baseline | Buy & Hold puro |
| M2 | Estadística pura | B&H + sizing GARCH × HMM |
| M3 | ML naive | H2O AutoML con **KFold** (réplica del sesgo a denunciar) |
| M4 | ML + estadística | H2O AutoML con **CPCV** + regime conditioning + sizing |
| M5 | IA pura | AI Hedge Fund (5 personalidades, DeepSeek V3) |
| M6 | IA + STRATA warn | M5 + supervisión sin intervenir |
| M7 | IA + STRATA reduce | M5 + atenuación proporcional |
| M8 | IA + STRATA override | M5 + sustitución por banda GARCH |
| M9 | ML + IA | H2O AutoML combinado con salidas de las personalidades |

**Por qué.** El diseño anterior tenía dos experimentos separados (OS1 motivador con 4 estrategias sobre 2022-2024 + STRATA con 5 sobre 2024-2026) ejecutados sobre tickers y métodos heterogéneos. Esto dificulta la comparación directa y deja huecos: no había configuración ML+IA. El experimento unificado es **académicamente más limpio** (un solo OOS, un solo activo, 9 configuraciones comparables dos a dos) y permite testear hipótesis específicas, p.ej. "¿M3 KFold sobreestima Sharpe respecto a M4 CPCV?" o "¿M9 mejora a M3 al añadir features del agente?".

**Cómo se implementa.** Nueve scripts `experiments/m{1..9}_*.py` + `ablation_strata.py` + `statistical_tests.py` que produce la matriz pareada 9×9 de Diebold-Mariano. `viz/comparison.py` genera 14 figuras + 2 tablas distribuidas en 4 bloques (comparativo, explicativo, STRATA, arquitectural).

**Qué cambia.** Una sola tabla resumen 9×7 con todas las métricas comparables; matriz 9×9 de significancia; figuras que muestran las 9 curvas simultáneamente. La memoria pasa de tener dos capítulos de resultados a uno solo bien estructurado.

**Cómo se defiende.**
> "El experimento unificado permite contrastar las tres familias de técnicas (estadística clásica, ML supervisado, IA generativa) sobre la misma ventana OOS con métricas directamente comparables. Cada hipótesis del TFG corresponde a un par o subconjunto explícito de configuraciones."

---

## 7. Dirección continua para los modelos ML (M3, M4, M9)

**Qué.** La traducción de la probabilidad `p1 = P(retorno_t+1 > 0)` que emiten los modelos H2O AutoML a una orden direccional usa una **función lineal continua**:

```
direction = 2 × p1 − 1   ∈ [-1, +1]
```

No una binarización con banda muerta (`+1 si p1>0,55, -1 si p1<0,45, 0 sino`).

**Por qué.** El modelo de ML emite probabilidades graduales que reflejan **cuánta confianza tiene** en cada dirección. Binarizar con threshold descarta esa información: un día con `p1 = 0,54` (modelo ligeramente alcista) y un día con `p1 = 0,46` (modelo ligeramente bajista) reciben el mismo trato (flat) con la regla binaria, perdiendo la información de confianza. Además, el threshold `0,55` (o cualquier otro) es un **hiperparámetro arbitrario sin justificación teórica**: cambiar 0,55 → 0,52 → 0,60 da resultados distintos sin criterio para elegir.

**Cómo se implementa.** Función `_direction_continuous(p1)` en `experiments/m4_ml_strata.py` y `experiments/m9_ml_ai.py`:

```python
def _direction_continuous(p1: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * p1 - 1.0, -1.0, 1.0)
```

Después se compone con la magnitud `(TARGET_VOL / σ_t).clip(0,1) × regime_factor`, idéntica a M2.

**Qué cambia.**

| `p1` | Regla anterior (binaria) | Regla nueva (continua) |
|---:|---|---|
| 0,46 | banda muerta → flat | -0,08 (short pequeño) |
| 0,50 | banda muerta → flat | 0,00 (flat exacto) |
| 0,55 | banda muerta → flat | +0,10 (long pequeño) |
| 0,60 | +1 (long pleno) | +0,20 (long modesto) |

- M4 y M9 ya **no se quedan flat por banda muerta** (antes 43/90 días). Toman posiciones pequeñas proporcionales a la confianza.
- M3, M4 y M9 ahora son **comparables**: usan la misma regla de traducción. La diferencia entre ellos es lo importante (KFold vs CPCV vs CPCV+features-agente), no un threshold distinto.
- Si el modelo no tiene señal predictiva (probabilidades ≈ 0,5), la regla continua lo expone: posiciones pequeñas en direcciones casi-aleatorias. La regla anterior **enmascaraba** esa falta de señal con flats artificiales.

**Cómo se defiende.**
> "El modelo de machine learning emite una probabilidad continua. La traducción anterior usaba un umbral arbitrario (0,55) para binarizar la señal, descartando la información de confianza. He cambiado a una regla lineal `2·p − 1` que mapea directamente la probabilidad a una dirección en [-1, +1], preservando la confianza y eliminando un hiperparámetro sin justificación teórica."

---

## 8. Por qué el sizing no puede arreglar una dirección equivocada (ablación M3/M4)

**Qué.** Sobre los 404 días, M3 (KFold, sin sizing) sale Sharpe -1,46 y M4 (CPCV + sizing GARCH × HMM) sale Sharpe -2,28. Aparente paradoja: añadir sizing parece empeorar. **No es paradoja**: M3 y M4 cambian dos cosas a la vez (esquema de validación H2O **y** sizing), y el efecto dominante es el primero, no el segundo. Esta decisión documenta el principio matemático que subyace y la ablación 2 × 2 que lo confirma.

### Principio 1 — Sizing escala magnitud; dirección fija el signo

La regla de trading es:

```
weight_t        = direction_t · magnitude_t
return_strat_t  = weight_t · ret_market_t
                = direction_t · magnitude_t · ret_market_t
```

- `direction_t ∈ [-1, +1]` es la **apuesta direccional** (señal del modelo, ±1 binario o continuo).
- `magnitude_t ∈ [0, 1]` es el **tamaño** que se le da (sizing: GARCH, regime, factor target-vol/σ, etc.).

Si `magnitude_t = c` constante (uniforme entre días), entonces `Sharpe(return_strat) = Sharpe(c · direction · ret) = Sharpe(direction · ret)` porque Sharpe es invariante a escala positiva: `Sharpe(c·X) = c·E[X] / (c·σ(X)) = E[X]/σ(X) = Sharpe(X)`. **Escalar uniformemente no cambia Sharpe**.

El sizing solo puede mover Sharpe si **no es uniforme** — es decir, si la magnitud varía entre días de forma correlacionada con los retornos. Eso es exactamente lo que hacen GARCH (más sizing cuando σ es baja) y regime conditioning (sizing=0 en Crisis): un sizing condicional al riesgo.

Ahora bien:

- Si la dirección es **mayoritariamente correcta**, modular el tamaño por riesgo es net-positivo: reduces exposición justo cuando el riesgo aumenta, sin perder demasiada media. → M2 vs M1.
- Si la dirección es **ruido** (AUC ≈ 0,5), el sizing no tiene materia prima con la que trabajar: estás escalando una caminata aleatoria. Cualquier ganancia o pérdida del sizing es atribuible al azar correlación-vol con retornos en ese sample concreto.

### Principio 2 — La comparación limpia del sizing es M1 vs M2, no M3 vs M4

**M1 vs M2** mantiene la **misma dirección** (`direction = +1` permanente, B&H) y solo cambia el sizing:

| | Dirección | Sizing | Sharpe 404 d | MaxDD 404 d |
|---|---|---|---:|---:|
| M1 | +1 fijo | 1,0 fijo | **+0,98** | **-19,2 %** |
| M2 | +1 fijo | GARCH × HMM | **+1,13** | **-5,4 %** |

Con dirección controlada (B&H), añadir sizing **mejora Sharpe en +15 % y reduce drawdown 71 %**. Este es el resultado limpio y defendible que prueba el valor empírico del sizing. La narrativa académica de la memoria descansa aquí.

**M3 vs M4** cambia **dos ejes simultáneamente**:

1. Esquema de validación del H2O AutoML (KFold convencional vs CPCV-purged) → líder H2O distinto → serie `p1` distinta → **dirección distinta cada día**.
2. Sizing on/off (M3 no aplica sizing; M4 sí).

Comparar M3 directamente con M4 conflate las dos cosas. Para aislar el sizing hay que controlar la dirección, y para eso hace falta la matriz 2 × 2.

### Principio 3 — La matriz 2 × 2 que aísla cada efecto

Construida por `experiments/diagnose_m3_m4_sizing.py` sobre los **mismos 403 días OOS** (reusando los `p1` cacheados de ambos JSON, sin volver a entrenar H2O):

| | **Sin sizing** (`weight = direction`) | **Con sizing M4** (`weight = direction × magnitude`) |
|---|---:|---:|
| **Dir KFold** (líder M3) | -1,456 (M3 actual) | **-1,262** (M3 + sizing) |
| **Dir CPCV** (líder M4) | -2,170 (M4 sin sizing) | -2,283 (M4 actual) |

Deltas por componente:

| Δ Sharpe | Valor | Interpretación |
|---|---:|---|
| Δ(fold scheme \| sin sizing) = SR(KFold) − SR(CPCV) | **+0,714** | Pasar de KFold a CPCV cuesta 0,7 Sharpe por **cambio de leader**. |
| Δ(sizing \| dir KFold) = SR(M3 actual) − SR(M3 + sizing) | **−0,194** | Con dirección KFold, **añadir sizing MEJORA** Sharpe (+0,19 puntos). |
| Δ(sizing \| dir CPCV) = SR(M4 sin sizing) − SR(M4 actual) | +0,113 | Con dirección CPCV, sizing es prácticamente neutro. |
| Δ(regime Crisis→0 \| dir CPCV) | −0,085 | Anular 28 días Crisis es casi indiferente. |

**Conclusión**: el delta visible M3 − M4 (~0,83 Sharpe) está dominado por el ruido del fold scheme (0,71 de 0,83), no por el sizing (0,11). Es decir, en problemas sin señal direccional real (AUC 0,52 — moneda casi pura), elegir KFold o CPCV genera dos líderes distintos cuyas series `p1` están descorrelacionadas; sobre 404 días esa diferencia de ruido es mayor que cualquier mejora o degradación del sizing.

### Principio 4 — Por qué este resultado es **el esperado**, no un fallo

Tres observaciones que se conjugan:

1. **El AUC de los líderes H2O es ~0,52** sobre la calibración (50 % es moneda, 100 % es oráculo perfecto). Esto significa que las features técnicas del mercado **no contienen señal direccional aprovechable** sobre SPY a horizonte 1 día. Es coherente con la hipótesis de mercado eficiente en su forma débil: la información histórica de precios no permite predecir la dirección del día siguiente.

2. **Si la dirección es ruido, el sizing no puede convertir ruido en señal**. Por el principio 1, el sizing solo modula magnitud; no puede invertir el signo de `direction × ret`. Si `direction · ret` es centrada en 0 con std positivo (ruido), entonces `magnitude · direction · ret` también lo es, escalada. Sharpe permanece cerca de 0 — fluctuará por azar a positivos o negativos según el sample.

3. **El bloque ML (M3, M4, M9) no es donde el TFG demuestra el valor del sizing.** Es el bloque que **denuncia el sesgo metodológico del KFold** (M3 +1,74 sobre 90 d → -1,46 sobre 404 d cuando la ventana corta enmascara el ruido) y muestra los límites del AutoML sin features fundamentales reales (M4, M9 cercanos a 0 o negativos). El sizing se demuestra en M1 vs M2 (estadística pura) y en M5 vs M7/M8 (IA pura supervisada por STRATA, que también es sizing condicional, esta vez condicional al desajuste régimen-acción que detecta RAM).

### Cómo se implementa

- `experiments/diagnose_m3_m4_sizing.py` — script de análisis read-only que recompone las 4 celdas reusando los `p1` cacheados en `outputs/experiments/m{3,4}_*.json`. Persiste `outputs/experiments/m3_m4_sizing_ablation.json`.
- `viz/comparison.py::tabla_c_ablacion_sizing` — render PNG/HTML de la tabla 2 × 2 + deltas para incluir en la memoria como ablación auxiliar.

M3 y M4 **no se modifican**: mantienen su identidad teórica (M3 = ML naive con KFold; M4 = ML honesto con CPCV + sizing). La ablación 2 × 2 entra en la memoria como tabla auxiliar para desambiguar la lectura del bloque ML.

### Qué cambia

Antes de este análisis, alguien podría leer la tabla principal y concluir "el sizing empeora porque M4 < M3". Tras la ablación, la lectura correcta es:

- **El sizing es neutro o ligeramente positivo** manteniendo la dirección fija (matemáticamente esperable).
- **El fold scheme domina el delta entre M3 y M4** cuando el modelo subyacente no tiene señal direccional.
- **El valor del sizing se prueba en M1 vs M2**, donde la dirección está controlada (+1 fijo) y la única variable es el sizing GARCH × HMM.

### Cómo se defiende

> "El sizing es matemáticamente invariante de Sharpe bajo escalado uniforme; solo cambia el resultado cuando varía día a día de forma correlacionada con los retornos. La comparación M3 vs M4 confunde dos cambios — fold scheme y sizing — así que no aísla el efecto sizing. La ablación 2 × 2 muestra que con la dirección fija (sea KFold o CPCV), el sizing es neutro: el delta visible entre M3 y M4 viene del ruido del leader H2O, no del sizing. La prueba limpia del valor del sizing está en M1 vs M2: dirección B&H idéntica, solo sizing, Sharpe +15 % y MaxDD −71 %. Y de fondo: si la dirección es ruido, ningún sizing puede convertirla en señal — el sizing modula magnitud, no signo."

---

## 9. Régimen HMM por activo y prior de RAM re-signado por activo (multi-activo completo)

**Qué.** Cuando STRATA se aplica a un activo distinto de SPY, **todos** sus parámetros son de ese activo: el **HMM de régimen se entrena sobre su propia serie** (igual que el GARCH y los umbrales PSA/GSO) **y el prior direccional de RAM se deriva de sus datos de calibración**. No se reutiliza ni el HMM del S&P ni la tabla de priores del índice. Las extensiones NVDA y BAC del notebook usan por tanto `hmm_nv`/`hmm_bc` y `RDIR_nv`/`RDIR_bc`, no los de SPY.

**Por qué.** Hasta el 2026-05-21 la extensión NVDA reutilizaba el régimen del S&P mientras GARCH/σ/umbrales sí eran de NVDA — una **asimetría incoherente**. El argumento del *leverage effect* que justifica la política de RAM es sobre la correlación retorno-volatilidad **del propio activo**, así que el régimen tiene que ser el del activo. Con el régimen del S&P, RAM aplicaba a NVDA la dirección implícita de un mercado al que NVDA se desacopla → penalizaba por construcción, no por una propiedad real de NVDA.

**Cómo se implementa.** Dos piezas en el notebook: (1) `supervised_sizes(...)` admite `hmm_src/feats_src/proba_src` para inyectar el HMM del ticker; (2) `ram_detector(..., regime_dir)` recibe el sentido favorable por régimen, que `regime_dir_from_calib(hmm, feats_calib)` calcula como el **signo del retorno medio de calibración** de cada régimen (Estrés neutro). En §12/§13 se entrenan `hmm_nv`/`hmm_bc`, se recalibran sus umbrales y se derivan `RDIR_nv`/`RDIR_bc`, que se pasan a `supervised_sizes`. El cross-check de SPY queda intacto: su prior re-signado `RDIR` es idéntico al leverage (Δ 3e-7), y las firmas son retrocompatibles.

**Qué cambia.** Sobre los 403 días del OOS de NVDA, las dos correcciones reordenan el bloque supervisado en cascada: del régimen del S&P al propio, **M8 sube de −0,46 a +0,66**; añadiendo el prior re-signado, **M8 sube de +0,66 a +0,95** y M7 de −0,04 a +0,19 (M2 +0,99, Calmar 1,35). NVDA M8 +0,95 es el M8 más alto del trío y sigue por debajo de su M2. En SPY y BAC el prior derivado de datos reproduce el leverage clásico (Crisis bajista) → resultados **idénticos al default**. Tabla completa de las tres variantes en [hallazgos_strata.md](hallazgos_strata.md) §3.

El porqué: el prior derivado de datos revela que el *leverage effect* de NVDA está **invertido** —su régimen de máxima volatilidad (Crisis) es el más alcista (+17 bps), frente a −4 bps en SPY y −6 bps en BAC—. Por eso RAM se activa **más** con el régimen correcto (~61 % vs ~36 %): el detector funciona y, al voltear «Crisis ⇒ short» a «Crisis ⇒ long» solo para NVDA, su intervención apunta en la dirección real del activo.

**Cómo se defiende.**
> "Todos los parámetros de STRATA son del activo: el régimen (HMM propio), la volatilidad (GARCH), los umbrales y también el *prior direccional de RAM*, que derivamos sin look-ahead del signo del retorno medio por régimen en calibración. Con eso la supervisión deja de depender de asumir el *leverage effect* del índice: en SPY y BAC (Crisis bajista) el prior sale «Crisis ⇒ short» y los resultados no cambian; en NVDA, cuya alta volatilidad es alcista, se voltea solo a «Crisis ⇒ long» y el *override* sube de +0,66 a +0,95. STRATA se auto-adapta a cada activo, sin un único modelo para todos."

---

## 10. *Volatility targeting* y comparación de retornos a riesgo común (`Ret@σBH`)

**Qué.** Las configuraciones cuantitativas (M2, M4, M8) dimensionan la posición por *volatility targeting* a una vol anual objetivo fija (`TARGET_VOL = 0,10`), con tope a exposición plena y **sin apalancamiento**: `peso = clip(target_vol / σ_t, 0, 1)`. Por eso operan muy por debajo de la exposición de Buy & Hold (100 %) y su **retorno bruto es menor**. La comparación entre estrategias se hace **ajustada por riesgo** (Sharpe, ya invariante a escala) y, para que el retorno sea directamente comparable, con la columna **`Ret@σBH`**, que reescala cada estrategia a la volatilidad del propio B&H del activo.

**Por qué.** (1) El retorno **bruto no es comparable** entre estrategias que corren a vol distinta: escalar (apalancar/desapalancar) por una constante `k` multiplica retorno y vol por `k` pero **deja el Sharpe igual** → el nivel de exposición es una palanca *independiente* de la calidad de la estrategia. (2) Un vol-target común (~10 %) pone todos los activos (SPY ~17 %, XLE ~24 %, TSLA ~60 %) y todas las configuraciones en el **mismo plano de riesgo**, lo que hace honestas las comparaciones cross-activo y el DSR. (3) Es coherente con el mandato de STRATA —disciplina de riesgo, no maximizar retorno—; el tope a 1,0 (sin margen) es deliberado.

**Cómo se implementa.** `TARGET_VOL = 0,10` (congelado). El *sizing* GARCH×HMM y la banda del detector GSO usan `clip(target_vol/σ_t, 0, 1)`. Comparación a riesgo común: helper `ret_at_vol(r, target_vol) = (1 + (target_vol/σ)·r).prod() − 1` (escalado lineal, preserva el Sharpe) y columna **`Ret@σBH`** en las cinco tablas de métricas con `σ = vol anual del B&H del activo` (B&H queda con `k = 1` y conserva su retorno real). Una nota junto a la tabla de §5 lo explica una vez para todas.

**Qué cambia.** Disuelve la aparente paradoja "M8 rinde menos que B&H": el retorno bruto bajo es **solo menor exposición**. Ejemplo TSLA: M8 corre al 8 % de vol vs 60 % de B&H → bruto +15,1 %, pero **+122,1 % a riesgo de B&H**, por encima del +53,7 % de B&H (porque su Sharpe 1,14 > 0,75). En SPY y XLE el M8 reescalado sigue por debajo de B&H (Sharpe menor → **techo de supervisión**). En la columna `Ret@σBH` el orden de retornos coincide con el del Sharpe.

**Decisión abierta (diferida, 2026-05-22).** Queda **pendiente** decidir para la entrega final si se **sustituye** la columna "Retorno" bruta por la escalada a riesgo común (`Ret@σBH`) —evitando dos columnas de retorno y dejando la comparación directa de un vistazo— o si se **mantienen ambas**. Por ahora conviven en el notebook con la nota explicativa.

**Cómo se defiende.**
> "El retorno bruto no es comparable entre estrategias con distinta volatilidad. Operamos a un *vol-target* del 10 % por disciplina de riesgo y para comparar todos los activos en el mismo plano; pero como el escalado preserva el Sharpe, la comparación justa es el Sharpe —o el retorno reescalado a riesgo común (`Ret@σBH`)—. Una estrategia solo supera a Buy & Hold si su Sharpe es mayor, no por tener más exposición."

---

## Resumen de una página (para la diapositiva de defensa)

| # | Decisión | Una línea |
|---|---|---|
| 1 | Activo SPY | Leverage effect garantiza régimen ↔ dirección. NVDA = contraejemplo (matiz §9). |
| 9 | Todo por activo (HMM + prior RAM) | Cada ticker deriva su HMM, GARCH, umbrales y prior direccional de RAM (signo del retorno de calibración). NVDA M8 +0,95; SPY/BAC intactos (leverage clásico). |
| 2 | Macro context al agente | SPY no tiene fundamentales empresariales; inyectamos macro/sentimiento. |
| 3 | HMM con `realized_vol_21d` | Vol realizada captura régimen direccional; VIX está contaminado por prima de riesgo. |
| 4 | RAM simétrico | Calma penaliza short, Crisis penaliza long. Coherente con leverage effect. |
| 5 | Umbrales por percentiles | PSA P95=0,023 ; GSO P95=2,37. Sensibilidad calibrada a la distribución empírica. |
| 6 | 9 configuraciones unificadas | Un único experimento OOS comparando estadística vs ML vs IA con y sin STRATA. |
| 7 | Dirección continua ML | `direction = 2p−1`. Sin threshold arbitrario; preserva confianza del modelo. |
| 8 | Sizing no salva una dirección de ruido | Sharpe es invariante a escalado uniforme. M3 vs M4 confunde fold scheme + sizing. Valor del sizing probado en M1 vs M2 (Sharpe +15 %, MaxDD −71 %). |
| 10 | Vol-target y retorno a riesgo común | Escalar no cambia el Sharpe; el retorno bruto no es comparable. `Ret@σBH` reescala a la vol de B&H (TSLA M8 +122 % > B&H +54 %). **Decisión diferida:** ¿sustituir la columna "Retorno"? |

---

## Nota sobre el historial

La cronología completa, con causas inmediatas, diagnósticos día a día, errores documentados y resultados numéricos, está en `BITACORA.md` ordenada por fecha en orden cronológico inverso. Este `decisiones.md` es la destilación didáctica de las decisiones **que hay que entender para entender STRATA**; cualquier matiz o detalle se busca en BITACORA.
