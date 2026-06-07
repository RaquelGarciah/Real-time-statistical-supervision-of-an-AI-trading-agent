# Resultados objetivo — cifras de referencia (no objetivo único)

Las cifras que el proyecto anterior produjo al cierre (2026-06-07). El nuevo proyecto debe **replicarlas o superarlas con mejor rigor**. Si las cambia por un diseño más limpio, hay que justificar la diferencia.

**Fuente primaria:** `_archivo_proyecto_anterior/outputs_canonicos/m{5,8,10}*.json` y `statistical_tests.json`.
**Fuente secundaria** (accuracy/AUC/Brier/MCC contra ground truth binario): notebook `strata_final.ipynb` del proyecto anterior, sesión 2026-06-02 → 2026-06-07. Esas cifras se recalculan en el nuevo notebook canónico desde los net_returns y weights del JSON.

---

## Tabla maestra (SPY OOS, N ≈ 400 días)

| Estrategia | Accuracy | AUC | Log-loss | Brier | MCC | Sharpe | €1000→ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline "always long" | 0.566 | — | — | — | — | ≈ B&H | 1323 |
| M5 (agente solo) | 0.407 | 0.481 | 0.756 | 0.281 | −0.106 | **−1.83** | **903** |
| M8 (STRATA override C) | 0.460 | 0.471 | 1.640 | 0.312 | −0.090 | **+0.66** | **1064** |
| M10 (XGBoost CPCV) | 0.530 | 0.504 | 0.785 | 0.284 | +0.022 | **+0.69** | **1035** |

Sharpe y €1000→ en negrita = directamente de los JSON canónicos. Accuracy/AUC/log-loss/Brier/MCC del notebook (recalcular).

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

## La narrativa de cierre (frase a memorizar)

> *"El agente LLM sin supervisar pierde dinero (€903 sobre €1000) y acierta direccionalmente menos del 50% (sign test p<0.001). STRATA lo rescata con significancia pareada (McNemar p≈0.088, Δ€161). Un meta-learner XGBoost validado con CPCV llega al mismo techo (€1035) sin ser distinguible estadísticamente de la regla a mano (DM p≈0.75), y SHAP confirma que las features informativas son exactamente las que STRATA codifica explícitamente. Ningún sistema bate B&H pasivo (+32%) sobre 400 días de SPY — resultado coherente con la literatura sobre eficiencia direccional de índices agregados. La aportación del TFG es un protocolo de supervisión estadística que rescata a un agente LLM perdedor, no un sistema que bate al mercado."*

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
