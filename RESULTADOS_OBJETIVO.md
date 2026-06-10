# Resultados objetivo — cifras de referencia (no objetivo único)

**ESTRUCTURA:** §1 = cifras canónicas del nuevo proyecto (fuente única de verdad para la memoria LaTeX). §2 = tabla del proyecto anterior (referencia histórica, no canónica).

---

## §1. Cifras canónicas — nuevo proyecto (notebook strata_canonical.ipynb, 2026-06-08)

**Fuente:** `notebooks/strata_canonical.ipynb` (K=3, τ=0.5, OOS 2024-10-01→2026-06, N=401). Estas son las cifras que van a la memoria LaTeX.

### Tabla maestra canónica (SPY OOS, N=401 días)

| Estrategia | Accuracy | Sharpe | McNemar vs M5 | Lectura |
|---|---:|---:|---:|---|
| B&H (referencia pasiva) | 0.569 | ≈ B&H | — | Techo del problema |
| M5 (agente solo) | 0.384 | −1.82 | — | Perdedor direccional (sign test p=4·10⁻⁶) |
| M7 (reduce) | 0.384 | −1.41 | trivial (b=c=0) | Reduce daño en P&L (DM p=0.095), no accuracy |
| M8 (STRATA override C) | 0.436 | +0.67 | **p=0.069 (τ=0.5) / 0.088 (τ=0.40)** | Rescata accuracy; Sharpe frágil (DSR≈0.10) |
| M10 (XGBoost CPCV) | **0.539** | +0.64 | DM M10 vs M8: p=0.67 | Mejor accuracy; equivalente a M8 en P&L |
| M2 (régimen×GARCH, sin agente) | — | +~0 | DM M8 vs M2: p=0.44 | Ablación sin agente |

**Métrica primaria: accuracy direccional** (el Sharpe es ilustración frágil; Deflated Sharpe M8 ≈ 0.10).
**Rescate condicional:** walk-forward §13 muestra ΔSharpe invertido en el tramo bajista (−3.92, n=123 ≥ 60) — falsificación pre-registrada disparada. "STRATA-SPY = disciplina de riesgo condicional al alza."

### Tests pareados canónicos

| Test | Comparación | p-valor | Lectura |
|---|---|---:|---|
| McNemar (τ=0.5) | M8 vs M5 | **0.069** | Rechaza H0 a α=0.10 |
| McNemar (τ=0.40 default) | M8 vs M5 | **0.088** | Rechaza H0 a α=0.10 (blindaje dual) |
| Block permutation | M8 vs M5 | **0.044** | Controla autocorrelación |
| Diebold-Mariano | M10 vs M8 P&L | **0.67** | Equivalentes en P&L |
| Walk-forward B-conf | mediana ΔSharpe IC95 | [−0.21, +5.71] | Incluye 0 — rescate no robusto multi-ventana |

---

## §2. Tabla del proyecto anterior (referencia histórica — NO usar en memoria LaTeX)

Las cifras que el proyecto anterior produjo al cierre (2026-06-07). El nuevo proyecto las ha replicado con mejor rigor (K=3 fijo, τ=0.5, walk-forward, accuracy-first). Si en la memoria aparece una cifra, debe venir de §1.

**Fuente primaria:** `_archivo_proyecto_anterior/outputs_canonicos/m{5,8,10}*.json` y `statistical_tests.json`.

| Estrategia | Accuracy | AUC | Log-loss | Brier | MCC | Sharpe | €1000→ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline "always long" | 0.566 | — | — | — | — | ≈ B&H | 1323 |
| M5 (agente solo) | 0.407 | 0.481 | 0.756 | 0.281 | −0.106 | **−1.83** | **903** |
| M8 (STRATA override C) | 0.460 | 0.471 | 1.640 | 0.312 | −0.090 | **+0.66** | **1064** |
| M10 (XGBoost CPCV) | 0.530 | 0.504 | 0.785 | 0.284 | +0.022 | **+0.69** | **1035** |

*Diferencias respecto al canónico actual: K no fijo vs K=3; τ=0.40 vs τ=0.5; N=400 vs 401; M10 accuracy 0.530 vs 0.539; M8 Sharpe 0.66 vs 0.67; DM p=0.75 vs 0.67. Las diferencias son menores y esperadas por el rediseño.*

### Métricas auxiliares del backtest (JSON canónico)

| Métrica | M5 | M8 | M10 |
|---|---:|---:|---:|
| Sortino | −2.37 | +0.99 | +1.09 |
| Max drawdown | −9.7% | −6.8% | −4.4% |
| Calmar | −0.64 | +0.59 | +0.48 |
| Profit factor | 0.70 | 1.15 | 1.17 |
| Hit rate (backtest) | 0.375 | 0.418 | 0.441 |
| Turnover | 0.139 | 0.224 | 0.174 |
| `n_obs` | 400 | 402 | 417 |

**Nota.** Los `n_obs` difieren ligeramente porque M10 usa CPCV-within-OOS y tiene splits que reciclan ciertos días; M5/M8 son secuenciales. Documentar en el notebook.

---

## Tests pareados

| Test | Comparación | Estadístico | p-valor | Lectura |
|---|---|---|---:|---|
| **McNemar** | M8 vs M5 (decisiones) | χ² pareado | **≈ 0.088** | STRATA rescata al agente con significancia borderline |
| **Diebold-Mariano** | Sharpe M10 vs M8 | DM | **≈ 0.75** | M10 y M8 indistinguibles |
| **Sign test** | M5 contra 0.5 | binomial | **< 0.001** | Agente solo peor que el azar |
| **Bootstrap estacionario** | Δ Sharpe(M10−M8) | IC95% Politis-Romano | **[−1.80, +1.15]** | El intervalo contiene cero |
| **Bootstrap** | P(M10 > M8) | porcentil | **≈ 0.543** | Moneda |

---

## SHAP global de M10 — top 5 features

Calculado con TreeSHAP nativo de XGBoost (`booster.predict(X, pred_contribs=True)`). Eficiencia de Shapley verificada: `max |sum(SHAP_j) + base − logit(p)| < 1e-6`.

| # | Feature | SHAP medio |
|---|---|---:|
| 1 | `ram_score` | **0.527** |
| 2 | `psa_score` | **0.428** |
| 3 | `garch_sigma` | 0.346 |
| 4 | `stress_prob` | 0.342 |
| 5 | `calm_prob` | 0.324 |

**Las 3 features STRATA + 2 de régimen ocupan el top 5. Ninguna personalidad del agente llega.**

Top features de cada personalidad (Buffett, Wood, Druckenmiller, Burry, Ackman) caen por debajo del puesto 10 en SHAP. Esto es el **argumento empírico** de que la regla a mano de STRATA captura la señal informativa.

---

## Análisis condicional (SPY OOS)

| Subset | N | M5 acc | M8 acc | M10 acc | Lectura |
|---|---:|---:|---:|---:|---|
| Régimen Crisis | ~85 | 35.7% | 35.7% | **60.7%** | XGBoost útil en crisis |
| Régimen Calma | ~250 | 41.1% | **57.0%** | 51.9% | STRATA-override óptimo en calma |
| RAM-flag high | 107 | 41.1% | **58.9%** | — | **M8 − M5 = +17.8 pp** cuando RAM dispara high |

Este último número (RAM-high → +17.8 pp) es **el dato más limpio para defensa**.

---

## Estabilidad temporal del umbral XGBoost

| Umbral `p1` | Sharpe mitad-1 (train) | Sharpe mitad-2 (test) |
|---:|---:|---:|
| 0.42 | +0.20 | **+1.07** |
| 0.50 | +0.41 | +0.09 |
| **0.565 (óptimo train)** | **+0.76** | +0.14 |
| 0.60 | +0.40 | −0.04 |

**Conclusión:** umbral aprendido por XGBoost no estable. STRATA con umbrales fijos calibrados ex-ante (RAM 0.20/0.40/0.70, PSA/GSO P95/P99) es estable por construcción. Argumento de **interpretabilidad como ventaja defendible** frente a meta-learners.

---

## Panel multi-activo (decision-level, 10 tickers, ~1406 intervenciones)

Datos en `_archivo_proyecto_anterior/outputs_canonicos/decision_level/`.

### Atribución de P&L por detector

| Detector | % P&L | bps totales |
|---|---:|---:|
| **RAM** | **98%** | +9218 |
| PSA | 2% | +185 |
| GSO | 0% | 0 |

### P&L atribuible por activo (sign test sobre mediana 0)

| Ticker | P&L (bps) | sign test p | McNemar p vs M5 |
|---|---:|---:|---:|
| SPY | +1740 | < 0.10 | ≈ 0.088 |
| XLE | +1840 | < 0.10 | — |
| NVDA | (+) | — | — |
| BAC | (+) | — | < 0.05 |
| TSLA | (+) | — | — |
| UNG | (+) | — | — |
| MSTR | (−) | — | (contraejemplo) |
| **SMCI** | (−) | — | **0.011 (contra M8)** |
| ROKU | (+) | — | — |
| MARA | (+) | — | — |

Hit rate M5 vs M8 **mejora en 8/10 activos**. Sign test panel `p ≈ 0.109` (borderline).

### Hallazgos del panel

- **GSO no dispara con severidad medium+ en NINGÚN activo del panel.** Hallazgo metodológico negativo.
- **SMCI = contraejemplo McNemar contra M8** (agente con información direccional complementaria al prior).
- **MSTR = `prior-flip` clásico** (signo calibración ≠ signo OOS).

---

## La narrativa de cierre (frase canónica — usar §14 del notebook)

> *"Un agente LLM perdedor direccional (38.4%, < azar, sign test p<0.001) es rescatado por supervisión estadística clásica: la accuracy sube 0.384 → 0.436 (regla M8) → 0.539 (XGBoost M10 sobre features STRATA), y regla a mano y caja negra son equivalentes en P&L (DM p=0.67). La señal informativa es la de STRATA: ablación sin features de régimen/RAM/PSA/GSO cae a Sharpe +0.21. Ningún sistema bate B&H pasivo (0.569 accuracy) — STRATA reduce el daño, no genera alfa. El rescate es condicional al régimen alcista (walk-forward §13: ΔSharpe = −3.92 en el tramo bajista n=123 ≥ 60, falsificación pre-registrada disparada); el modelo K=3 sí generaliza inter-época (15/16 orígenes). La aportación es un protocolo de supervisión estadística interpretable que recupera accuracy direccional de un agente perdedor y delimita honestamente dónde funciona y dónde no."*

---

## Cómo verificar estas cifras desde el kit

```bash
cd /Users/Raquel/Desktop/STRATA_kit/
python3 -c "
import json
for f in ['m5_agent_alone', 'm8_strata_override', 'm10_ml_meta']:
    d = json.load(open(f'_archivo_proyecto_anterior/outputs_canonicos/{f}.json'))
    m = d['metrics']
    print(f'{f}: Sharpe={m[\"sharpe\"]:.3f}, equity={d[\"equity_final\"]:.4f}, hit={m[\"hit_rate\"]:.3f}, n={d[\"n_obs\"]}')
"
```

Output esperado:
```
m5_agent_alone:  Sharpe=-1.831, equity=0.9031, hit=0.375, n=400
m8_strata_override: Sharpe=+0.659, equity=1.0639, hit=0.418, n=402
m10_ml_meta: Sharpe=+0.692, equity=1.0353, hit=0.441, n=417
```

Si el nuevo proyecto produce cifras **dentro de ±10%** de estas, valida el resultado. Si difieren más, hay que entender por qué.
