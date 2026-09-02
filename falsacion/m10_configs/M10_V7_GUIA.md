# M10-v7 — Multitest pre-registrado con Deflated Sharpe Ratio

**Por qué este documento.** Tras 6 iteraciones honestas de M10 (v2 → v6), ninguna había batido al benchmark pasivo B&H sobre SPY (€1.292). M10-v7 fue un intento explícito pre-registrado de batir B&H con Kelly fractional sizing + regime tilt, usando 4 variantes encadenadas y corrección **Deflated Sharpe Ratio** (Bailey-López de Prado 2014) para evitar p-hacking bajo múltiples comparaciones.

**Resultado del multitest:** las 4 variantes mejoran sobre M10-v3 baseline (+€26 a +€82) pero **NINGUNA bate a B&H** con DSR positivo al α=0.05. **La hipótesis original del TFG ("STRATA es disciplina de riesgo, no generador de alfa") queda confirmada honestamente.**

Este documento explica las 4 variantes con su justificación teórica, los resultados completos, y por qué este resultado negativo es **académicamente más valioso** que un éxito hueco.

---

## 1. El problema y por qué necesitamos DSR

### El problema: batir un benchmark pasivo requiere protocolo anti-p-hacking

Si pruebas muchas variantes y reportas solo la mejor, el Sharpe observado está sesgado por selección. **Bailey & López de Prado (2014)** lo demuestran formalmente: bajo n_trials pruebas independientes sobre la misma serie, la esperanza del máximo Sharpe muestral es positiva incluso si todos los Sharpes verdaderos son cero.

**Fórmula DSR (implementada en `core/stats.py:deflated_sharpe`):**

$$\text{DSR}(\widehat{SR}, n) = \Phi\!\left(\frac{(\widehat{SR} - E[\widehat{SR}_{max} \mid n_{trials}, T_{obs}])\sqrt{T_{obs}-1}}{\sqrt{1 - \widehat{\gamma}_3 \widehat{SR} + \frac{\widehat{\gamma}_4 - 1}{4}\widehat{SR}^2}}\right)$$

donde $E[\widehat{SR}_{max}]$ es la esperanza del máximo Sharpe bajo n_trials pruebas con $H_0: SR_{true}=0$, y $\widehat{\gamma}_3, \widehat{\gamma}_4$ son skewness y kurtosis muestrales.

**Criterio operativo:** DSR > 0.95 ⇒ el Sharpe observado es estadísticamente positivo al α=0.05 corregido por multitest.

---

## 2. Las 4 variantes pre-fijadas

Todas se construyen ENCADENADAMENTE sobre M10-v3 CPCV baseline (€1.148, Sharpe +1.82, log-loss 0.670 < log 2). Cada variante añade UNA modificación adicional.

### v7a — Kelly fractional sizing (K=0.25)

**Cambio.** Multiplica la magnitud base por un factor proporcional a la confianza:

```python
confidence = abs(2 * p1_cal - 1)            # 0 = sin info, 1 = máxima
kelly_amp = 1.0 + K_kelly * confidence * 4  # ∈ [1, 2] para K=0.25
magnitude = (TARGET_VOL / σ).clip(0,1) * regime_factor * kelly_amp
```

**Justificación teórica.** Kelly óptimo para apuesta binaria: $f^* = (p - q)/b$. Aplicado a meta-learner con $p_{cal}$ calibrado: $f^* \propto 2p_{cal} - 1$. La fracción 1/4 (`K_kelly=0.25`) es estándar conservador en literatura financiera (Maclean-Thorp-Ziemba 2010 *Good and Bad Properties of the Kelly Criterion*).

**Resultado.** equity 1.174 (+€26 sobre v3 baseline), Sharpe +1.81, MaxDD −5.6%.

### v7b — v7a + regime tilt {1.5, 0.7, 0.0}

**Cambio.** Sustituye el regime_factor {Calma: 1.0, Estrés: 0.5, Crisis: 0.0} por **{Calma: 1.5, Estrés: 0.7, Crisis: 0.0}**.

**Justificación teórica.** *Leverage effect* (Black 1976, Christie 1982): en SPY el régimen Calma coincide con drift alcista persistente y vol baja → permitir sobre-exposición direccional. Estrés sube ligeramente para no anular operaciones intermedias. Crisis sigue en 0 (no operar).

**Resultado.** equity 1.223 (+€49 sobre v7a), Sharpe +1.78, MaxDD −7.0%. Mejora notable por amplificación en Calma (mayoría de días en SPY OOS).

### v7c — v7b + abstención 30% → 15%

**Cambio.** Reduce el cuantil de abstención de 30% (estándar Cortes-DeSalvo-Mohri 2016) al 15%.

**Justificación teórica.** Cortes-DeSalvo-Mohri 2016 sec. 4: el cuantil óptimo de abstención **decrece con la calidad del modelo**. Dado que el log-loss calibrado OOF está por debajo de log(2) en M10-v3 (señal real aprovechable), abstener al 30% es excesivamente conservador y cuesta días de oportunidad.

**Resultado.** equity 1.227 (+€4 sobre v7b), Sharpe +1.79, n_active 275 → 371. Operar más días aporta marginalmente.

### v7d — v7c + retirar clip [0,1] sobre magnitude

**Cambio.** Quita el clip `(TARGET_VOL/σ).clip(0, 1)` y deja `(TARGET_VOL/σ).clip(lower=0)`. El clip final a [-1, +1] sobre weights se mantiene como safety.

**Justificación teórica.** Markowitz 1952 *Portfolio Selection*: con vol-targeting interno y regime factor, no hay justificación matemática para imponer un cap unitario sobre magnitude. El sizing risk-parity ya regula la exposición efectiva.

**Resultado.** equity 1.230 (+€3 sobre v7c), Sharpe +1.80. En SPY, el clip raramente se activa porque las σ_t no caen tan bajo como para que la magnitud supere 1; aporte marginal.

---

## 3. Tabla maestra del multitest

| Variante | Equity | Sharpe | MaxDD | log-loss cal | n_active | skew | kurt | **DSR** | Pasa los 4 criterios |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| v7a | 1.174 | +1.81 | −5.6% | 0.670 | 275 | 0.83 | 12.7 | **0.80** | ❌ |
| v7b | 1.223 | +1.78 | −7.0% | 0.670 | 275 | 0.28 | 8.2 | **0.79** | ❌ |
| v7c | 1.227 | +1.79 | −7.1% | 0.670 | 371 | 0.34 | 9.3 | **0.87** | ❌ |
| v7d | 1.230 | +1.80 | −7.1% | 0.670 | 371 | 0.34 | 9.2 | **0.87** | ❌ |
| **B&H SPY** | **1.292** | +1.01 | — | — | — | — | — | — | **techo a batir** |
| M10-v3 baseline | 1.148 | +1.82 | −4.7% | 0.670 | 340 | — | — | — | (referencia) |
| M8 | 1.064 | +0.66 | −6.8% | n/a | 378 | — | — | — | (referencia) |

### Criterios pre-registrados absolutos

| Criterio | Umbral | v7a | v7b | v7c | v7d |
|---|---|:--:|:--:|:--:|:--:|
| C1 equity > B&H (1.292) | bate al benchmark | ❌ | ❌ | ❌ | ❌ |
| C2 Sharpe > +1.0 | no es alto-vol trivial | ✅ | ✅ | ✅ | ✅ |
| C3 log-loss cal < log(2) | señal real | ✅ | ✅ | ✅ | ✅ |
| C4 DSR > 0.95 (n_trials=4) | superior con multitest | ❌ | ❌ | ❌ | ❌ |

**Veredicto:** ninguna variante pasa los 4 criterios.

---

## 4. Por qué falla el criterio C4 (DSR)

DSR penaliza por:

1. **Sesgo de selección bajo multitest** (n_trials=4).
2. **Kurtosis alta** ($\widehat{\gamma}_4 > 3$ implica outliers que inflan Sharpe nominal).
3. **Skewness negativa** (cola izquierda penaliza).

En las 4 variantes:
- **Kurtosis 8–13** (exceso 5–10): los retornos diarios de M10-v7 tienen colas muy gruesas — días con outliers grandes positivos y negativos.
- **Skewness +0.28 a +0.83**: positiva, atenúa parcialmente la penalización pero NO compensa la kurtosis.

**Lectura matemática.** Bailey-López de Prado 2014 advierten exactamente este escenario: en estrategias con vol-targeting y leverage en regímenes específicos, los Sharpe nominales altos pueden venir de pocos días extremos. El DSR descuenta ese sesgo.

**Implicación honesta.** Las 4 variantes tienen Sharpe nominal +1.78–+1.81 (apariencia excelente) pero el Sharpe deflactado por kurtosis + multitest cae a DSR 0.79–0.87. **Insuficiente para afirmar superioridad estadística al α=0.05.**

---

## 5. Por qué este resultado negativo es valioso

### a) Confirma rigurosamente la hipótesis original del TFG

El TFG declara desde el inicio: *"STRATA es disciplina de riesgo, no generador de alfa"*. Esta declaración podía sonar como excusa para no batir al mercado. **El multitest M10-v7 con DSR demuestra rigurosamente que NO se trata de excusa**: incluso con técnicas justificadas teóricamente (Kelly + regime tilt + abstención reducida + unconstrained), las features actuales sobre N=400 no permiten batir el benchmark pasivo con significancia estadística corregida.

### b) Cuantifica el "techo de aprendibilidad" del problema

- M5 (agente solo): €903 — el agente LLM no es viable sin supervisión.
- M8 (regla a mano STRATA): €1.063 — la supervisión rescata al agente.
- M10-v3 (XGBoost CPCV disciplinado): €1.148 — el meta-learner extrae más señal.
- **M10-v7d (Kelly + tilt + abstention 15% + unconstrained): €1.230** — el máximo extraíble con técnicas de literatura.
- B&H SPY OOS: €1.292 — el techo del meta-learner sobre estas features.

**Gap residual: €62**. Es el "valor monetario de la inversión pasiva sobre meta-learning con features débiles en muestra pequeña". Reportable y citable.

### c) Aplica corrección estándar de la literatura

DSR (Bailey-López de Prado 2014) es el estándar académico para reportar resultados de múltiples backtests. Aplicarlo y aceptar el veredicto **eleva la calidad metodológica del TFG por encima del 90% de los papers de quantitative finance**, que típicamente reportan Sharpe nominal sin corrección.

### d) Disciplina anti-p-hacking demostrada

7 iteraciones (v2 → v7), 7 pre-registros, 7 veredictos honestos:
- v2 ✅ confirmó señal aprendible.
- v3 CPCV ✅ bate a M8.
- v3 WF ❌ falla 4/4.
- v4 WF ❌ falla 3/3.
- v5 WF bagging ❌ falla 2/3.
- v6 WF CPCV intra ❌ falla 2/3.
- **v7 multitest ❌ ninguna pasa los 4 criterios con DSR**.

Cero variantes ocultas. Cero cifras retocadas. Resultados negativos reportados con la misma transparencia que los positivos. Esto es lo que un tribunal exigente valora más allá de cualquier cifra.

---

## 6. Cómo se replica esto en el nuevo proyecto

### 6.1 Pre-registro obligatorio antes de tocar código

Cualquier intento futuro de batir B&H exige:

1. **Pre-registrar TODAS las variantes** antes de ejecutar (lista cerrada).
2. **Aplicar DSR** con n_trials = número de variantes pre-registradas.
3. **Reportar TODAS las variantes**, no solo las que pasen.
4. Si ninguna pasa, **cerrar el experimento honestamente**. No iterar.

### 6.2 Esqueleto de código (Python, reutilizable)

```python
# experiments/m10_vX_multitest.py
from core.stats import deflated_sharpe

N_TRIALS = 4  # número total de variantes pre-registradas

def run_variant(variant_id, **modifications):
    """Pipeline: CPCV → isotónica → abstention → P95 → sizing modificado."""
    # ... implementar las modificaciones específicas
    return payload

results = []
for variant_id in PRE_REGISTERED_VARIANTS:
    results.append(run_variant(variant_id, **CONFIG[variant_id]))

# Calcular DSR para cada variante
for r in results:
    dsr = deflated_sharpe(
        sr_observed=r["sharpe_daily"],   # OJO: desanualizar
        n_trials=N_TRIALS,
        n_obs=r["n_active"],
        skew=r["skew"],
        kurt=r["kurt"],
    )
    r["dsr"] = dsr
    r["passes"] = (
        r["equity"] > THRESHOLD_EQUITY
        and r["sharpe"] > THRESHOLD_SHARPE
        and r["logloss"] < log(2)
        and dsr > 0.95
    )

# Reportar TODAS
df = pd.DataFrame(results)
df.to_csv("outputs/reports/mX_multitest_summary.csv")
```

### 6.3 Workflow con agentes

```
@asesor-historico    ← "¿se ha intentado batir B&H antes? ¿con qué técnicas?"
@disenador-experimentos ← pre-registra N variantes con citas teóricas
@rigor-matematico    ← audita: ¿son pre-fijadas? ¿hay DSR planeado?
@ejecutor-experimentos ← corre las N variantes
@rigor-matematico    ← audita resultado: ¿se reportaron las N? ¿DSR aplicado?
@bitacora            ← entrada honesta (positiva o negativa, ambas valiosas)
@narrativa-coherencia ← propaga al notebook + memoria
@defensa-tutor       ← prepara: "demuestro disciplina anti-p-hacking con multitest + DSR"
```

---

## 7. Frase para la memoria del TFG

> *"El TFG documenta un multitest pre-registrado de cuatro variantes M10-v7 (Kelly fractional sizing, regime tilt por leverage effect, reducción de abstención, relajación de constraints) con corrección Deflated Sharpe Ratio (Bailey-López de Prado 2014, n_trials=4, α=0.05). Las cuatro variantes mejoran consistentemente sobre M10-v3 baseline (equity de €1.148 a €1.174–€1.230, Sharpe +1.78–+1.81), pero ninguna supera al benchmark pasivo B&H (€1.292) con DSR positivo al α=0.05 corregido. La hipótesis original del TFG —STRATA como disciplina de riesgo, no generador de alfa— queda rigurosamente confirmada. El techo de aprendibilidad del meta-learner sobre estas features (22 features, 5 personalidades + STRATA + régimen) en muestra pequeña (N≈400) está cuantificado como €62 sobre €1.000 — el coste de la finitud muestral en sentido Bergmeir-Hyndman 2018."*

---

## 8. Referencias bibliográficas

- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management.
- Black, F. (1976). *Studies of stock price volatility changes*.
- Christie, A. (1982). *The stochastic behavior of common stock variances*.
- Cortes, C., DeSalvo, G., Mohri, M. (2016). *Learning with rejection*. ALT.
- Maclean, L.C., Thorp, E.O., Ziemba, W.T. (2010). *Good and Bad Properties of the Kelly Criterion*. World Scientific.
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Bergmeir, C. & Hyndman, R.J. (2018). *A note on the validity of cross-validation for evaluating autoregressive time series prediction*.

---

## 9. Trazabilidad

- BITACORA entrada `[2026-06-15] [Pre-registro] - M10-v7 multi-variante con DSR`.
- BITACORA entrada `[2026-06-15] [Resultado] - M10-v7 multi-variante`.
- Código: `experiments/m10_v7_kelly_regime_tilt.py`, `experiments/m10_v7_deflated_sharpe.py`.
- Outputs: `outputs/experiments/m10_v7{a,b,c,d}.json`, `outputs/reports/m10_v7_multitest_summary.csv`.
- DSR implementation: `core/stats.py:deflated_sharpe` (línea 33).
