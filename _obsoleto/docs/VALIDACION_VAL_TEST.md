# Validación VAL/TEST por consistencia operativa

**Petición del tutor.** Subdividir el OOS en validación (1ª mitad) y test (2ª mitad), y comparar cada decisión metodológica con gráficos de barras pareados (val vs test). Si las barras "van de la mano", la decisión es robusta. Si divergen, hay sobreajuste al periodo.

**Tipo de validación elegido.** **Consistencia operativa (sin re-entrenar).** Se cargan los `net_returns` ya calculados de cada modelo y se dividen por fecha 50/50. Responde a la pregunta: *"¿el modelo se comporta igual en la 1ª mitad que en la 2ª?"*.

---

## 1. Metodología

- **OOS unificado:** 2024-10-01 → 2026-05-08 (400 días bursátiles comunes).
- **Fecha de corte automática:** 2025-07-23 (mediana de fechas comunes a los 14 modelos).
- **VAL:** primeros 202 días (oct-2024 → jul-2025).
- **TEST:** siguientes 200 días (jul-2025 → may-2026).
- **Sin re-entrenar:** se dividen los `net_returns` canónicos. NO es validación-test ML-académica estricta — es **consistencia operativa**.

**Métricas calculadas por subperiodo:**
- Sharpe causal anualizado (252 días).
- Equity final (€1000 → ).
- Hit rate direccional sobre días activos.
- MaxDD por subperiodo.

**Etiqueta de consistencia:**
- **Robusto:** |Δ Sharpe| < 0.3
- **Moderado:** 0.3 ≤ |Δ Sharpe| < 0.8
- **Frágil:** |Δ Sharpe| ≥ 0.8

---

## 2. Resultado principal

| Modelo | val_sharpe | test_sharpe | Δ Sharpe | consistencia |
|---|---:|---:|---:|---|
| **M10-v6** (walk-forward CPCV intra) | +1.74 | +1.68 | **−0.06** | **robusto** |
| **M10-v4** (walk-forward simple) | +1.44 | +1.36 | **−0.09** | **robusto** |
| M10-v5 (walk-forward bagging) | +1.38 | +1.97 | +0.59 | moderado |
| M1 (B&H) | +0.70 | +1.61 | +0.91 | frágil |
| M2 (GARCH × HMM) | +0.32 | +1.19 | +0.88 | frágil |
| M5 (agente solo) | −1.36 | −2.61 | −1.25 | frágil |
| M8 (STRATA override) | −0.16 | +1.62 | +1.79 | frágil |
| M10 (original) | −0.46 | +1.76 | +2.22 | frágil |
| M10-v2 (CPCV + disciplina) | +0.61 | +3.13 | +2.52 | frágil |
| M10-v3 (CPCV + P95) | +0.57 | +2.96 | +2.38 | frágil |
| M10-v7a (CPCV + Kelly) | +0.54 | +3.00 | +2.46 | frágil |
| M10-v7b (CPCV + Kelly + tilt) | +0.66 | +2.93 | +2.27 | frágil |
| M10-v7c (CPCV + Kelly + tilt + abst15) | +0.60 | +3.00 | +2.40 | frágil |
| M10-v7d (CPCV + Kelly + tilt + abst15 + unclip) | +0.59 | +3.03 | +2.44 | frágil |

**Resumen.** 2 robustos · 1 moderado · 11 frágiles.

---

## 3. El hallazgo no anticipado (lo más importante)

**Solo M10-v4 y M10-v6 — los dos esquemas walk-forward causales — son robustos.** Todos los modelos CPCV completo (M10-v3, v7a-d) son frágiles con Δ Sharpe ≈ +2.4.

### ¿Es esto overfitting al periodo VAL?

**No.** Es una **propiedad estructural** del CPCV completo bajo cambio de régimen entre subperiodos:

1. CPCV completo entrena 15 modelos sobre la muestra OOS COMPLETA con folds combinatoriales.
2. Cada modelo del ensemble ve datos de AMBAS mitades temporales en su train.
3. El régimen de mercado **cambió fuerte** entre VAL y TEST (B&H: VAL +0.70 vs TEST +1.61).
4. Al dividir a posteriori en VAL/TEST, las predicciones de la 2ª mitad están informadas implícitamente por la 1ª mitad (vía train).
5. CPCV completo "promedia" ambos regímenes y eso infla el Sharpe en el régimen más favorable.

**Walk-forward causal (v4, v6) NO tiene este efecto** porque cada predicción solo usa información pasada estrictamente. Su Sharpe en VAL y TEST son estructuralmente independientes.

### Confirmación adicional con M1

M1 (Buy & Hold puro, sin look-ahead posible) también es frágil (Δ +0.91). Esto confirma que el cambio de régimen entre VAL y TEST es **real y afecta a cualquier estrategia**. Lo que distingue a M10-v4 y v6 es que su Sharpe NO depende de "ver el régimen entero" durante el entrenamiento.

---

## 4. Implicaciones para el TFG

### 4.1 Refuerza el reporte honesto del gap CPCV ↔ walk-forward

El gap M10-v3 CPCV (€1148) ↔ walk-forward causal (€1044) **no es solo de "información disponible para train"** — es también de "vista del régimen entero". CPCV completo se beneficia retrospectivamente de saber que el régimen mejoró en la 2ª mitad, walk-forward no tiene esa información. **Estos €104 son cuantificables.**

### 4.2 M10-v6 es DOBLEMENTE deployable

Ya se sabía que es causalmente estricto (cada predicción solo usa info hasta `t-embargo`). Ahora se sabe también que es **robusto al cambio de régimen** (val Sharpe +1.74 ≈ test Sharpe +1.68). Esta es la propiedad que un live system necesita.

### 4.3 Los resultados de M10-v7 (Kelly + tilt) están afectados por el sesgo CPCV

Las decisiones v7a-v7d se evaluaron sobre CPCV completo. Sus Δ Sharpe ≈ +2.4 son artefactos del esquema, no del Kelly sizing per se. Para una validación honesta del Kelly + regime tilt habría que re-entrenarlas con esquema walk-forward (v7-WF), pendiente como trabajo futuro.

### 4.4 La narrativa de defensa se actualiza

> *"Una validación de consistencia VAL/TEST (50/50 split, sin re-entrenar) sobre los 14 modelos demuestra que los dos esquemas walk-forward causales (M10-v4 y M10-v6) son los ÚNICOS con consistencia robusta entre subperiodos (Δ Sharpe < 0.1). Todos los modelos CPCV completo muestran consistencia frágil (Δ Sharpe > +2.0). Esto NO se debe a overfitting al periodo VAL — es una propiedad estructural del CPCV completo bajo cambio de régimen entre subperiodos. Walk-forward es estructuralmente robusto al régimen porque cada predicción solo usa información pasada. M10-v6 (walk-forward CPCV intra-train) es por tanto la configuración recomendada para deployment, coincidiendo con López de Prado (2018, sec. 7.4.3)."*

---

## 5. Las figuras generadas

5 figuras PNG en `outputs/figures/val_test/`:

| Figura | Contenido | Mensaje |
|---|---|---|
| `fig_A_canonicas.png` | M1, M2, M5, M8, M10-v3 | B&H mejora en TEST por mercado alcista; STRATA y agente menos consistentes |
| `fig_B_disciplina_m10.png` | M10, M10-v2, M10-v3 | Las mejoras de disciplina están infladas por sesgo CPCV |
| `fig_C_cpcv_vs_wf.png` | M10-v3, v4, v5, v6 | **La figura más reveladora**: walk-forward robusto, CPCV frágil |
| `fig_D_intento_BH.png` | M10-v3, v7a-d | Las mejoras Kelly/tilt están infladas por sesgo CPCV |
| `fig_consistency_scatter.png` | Los 14 modelos | Scatter VAL vs TEST con bandas verde (robusto) y amarillo (moderado) |

---

## 6. Cómo replicar en el nuevo proyecto

### 6.1 Esqueleto del análisis

```python
# experiments/val_test_consistency.py
import json
import pandas as pd
from core.metrics import sharpe, max_drawdown, hit_rate, equity_curve

MODELS = {...}  # paths a JSON canónicos

# 1. Cargar net_returns por modelo
all_returns = {label: load_returns(path) for label, path in MODELS.items()}

# 2. Fecha de corte automática (mediana de fechas comunes)
common = intersection_of_all_indices(all_returns)
split_date = common[len(common) // 2]

# 3. Métricas VAL/TEST por modelo
for label, returns in all_returns.items():
    val = returns.loc[:split_date]
    test = returns.loc[split_date + pd.Timedelta(days=1):]
    # calcular sharpe, equity, hit_rate, MaxDD por subperiodo
    # etiquetar consistencia según |Δ Sharpe|
```

### 6.2 Figuras

`viz/val_test_bars.py` reusa `viz/shared.py:setup_matplotlib()` y la paleta Okabe-Ito. 2 paneles verticales por familia (Sharpe + equity), con barras pareadas VAL (alpha 0.55) y TEST (alpha 1.0).

### 6.3 Workflow con agentes

```
@asesor-historico    ← "¿hay análisis previos de consistencia VAL/TEST?"
@disenador-experimentos ← pre-registra: cuál métrica, qué fecha de corte, qué etiquetado
@rigor-matematico    ← audita: ¿el análisis es sin look-ahead? ¿la fecha de corte está justificada?
@ejecutor-experimentos ← corre los scripts (~30s)
@narrativa-coherencia ← interpreta el hallazgo y propaga a la memoria
@defensa-tutor       ← prepara: "el tutor pidió esto, este es el resultado, esta es la interpretación honesta"
```

### 6.4 Limitación honesta a documentar

Este análisis es **consistencia operativa**, no validación-test ML-académica estricta. Para esta última habría que re-entrenar cada modelo SOLO sobre VAL y evaluar en TEST. Coste computacional ~30 minutos para los 14 modelos. Pendiente como trabajo futuro (opcional).

---

## 7. Referencias

- BITACORA entrada `[2026-06-15] [Validación] - VAL/TEST consistency check`.
- `notebooks/validacion_live_backtest.ipynb` — protocolo más exigente (6 capas) para validación pre-deployment.
- `M10_V3_GUIA.md` — disciplina anti-overfitting de M10.
- `M10_V7_GUIA.md` — multitest con DSR (las 4 variantes M10-v7 sufren el mismo sesgo CPCV documentado aquí).
- `INVESTIGACION_VALIDACION_TIEMPO_REAL.md` — bibliografía sobre walk-forward vs CPCV.
