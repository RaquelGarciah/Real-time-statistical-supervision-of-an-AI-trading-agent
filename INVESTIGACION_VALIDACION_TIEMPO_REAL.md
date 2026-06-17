# Investigación: ¿existe algún método tan eficaz como CPCV pero compatible con tiempo real?

**Resumen ejecutivo.** Tras revisar la literatura actualizada (2018-2025), **no existe ningún método que iguale exactamente la eficiencia estadística de CPCV en tiempo real**, porque CPCV obtiene su eficiencia precisamente del uso de información futura (cada día se ve en múltiples folds). Sin embargo, **hay tres familias de métodos que se acercan operativamente**, y una de ellas — **bagging walk-forward (Inoue & Kilian 2008)** — es directamente aplicable a M10 y debería cerrar parcialmente el gap con CPCV. Esta es la propuesta concreta a pre-registrar como M10-v5.

---

## ⚡ Actualización (post-experimentos v5/v6/v7) — 2026-06-15

Las propuestas de este documento se ejecutaron como M10-v5 (bagging walk-forward) y M10-v6 (CPCV intra-train walk-forward). **Convergen al mismo punto operativo** (Δ equity 0.01% sobre SPY) confirmando que el gap CPCV↔WF es **estructural**, no algorítmico. Detalle en `M10_V7_GUIA.md` y BITACORA 2026-06-15.

Posteriormente, M10-v7 (intentar batir B&H con multitest pre-registrado + DSR) **falla en los 4 criterios** ⇒ la hipótesis original del TFG ("STRATA es disciplina de riesgo, no generador de alfa") queda **rigurosamente confirmada**. Detalle completo en `M10_V7_GUIA.md`.

Para implementar el **modo live deployable**, ver:
- `notebooks/validacion_live_backtest.ipynb` — protocolo de 6 capas de validación documentado.
- `M10_V7_GUIA.md` — guía técnica completa del multitest M10-v7.

---

## 1. ¿Por qué CPCV es estadísticamente más eficiente que walk-forward?

**CPCV** (López de Prado 2018, cap. 7) genera 15 modelos sobre 15 combinaciones distintas de bloques temporales como train y test. Cada día observado aparece en ~5 folds distintos como muestra de test, y su predicción final es el promedio de esos 5. **Beneficio:** la varianza del estimador se reduce por un factor √5 ≈ 2.2 frente a una sola observación.

**Walk-forward genuino** entrena un modelo cada vez (o cada semana) y predice cada día UNA sola vez. La varianza del estimador es la del modelo único + ruido del refit. **Coste estadístico:** ~√5 más varianza por observación.

Esa es la diferencia matemática del gap CPCV ↔ walk-forward que observamos en M10-v3 (€1148 CPCV vs €1031 walk-forward).

**Cita central:** López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 7.4. Comparado contra walk-forward en [Backtest overfitting in the machine learning era: a comparison of OOS testing methods](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) (ScienceDirect 2024), donde CPCV se confirma superior pero a costa de no replicar el despliegue operativo.

---

## 2. Las tres familias de métodos que se acercan a CPCV en tiempo real

### Familia A — Bagging walk-forward (Inoue & Kilian 2008)

**Idea.** En cada refit del walk-forward, en lugar de entrenar UN modelo sobre la ventana `[start, t-embargo]`, generar **K bootstraps** (típicamente K=20-50) de esa ventana con *moving-block bootstrap* (preserva dependencia temporal) y entrenar K modelos. Promediar las K predicciones para el día t.

**Por qué imita a CPCV.** CPCV agrega 15 modelos vía split combinatorio; bagging walk-forward agrega K modelos vía bootstrap. Ambos reducen la varianza del estimador por agregación. **Inoue-Kilian (2008) demuestran reducción del MSE out-of-sample comparable** a métodos más complejos en series temporales con N moderado.

**Es estrictamente causal:** cada bootstrap usa solo datos hasta `t-embargo`. No mira al futuro.

**Coste computacional:** K × n_refits modelos. Con K=20 y refit semanal (~80 refits) ≈ 1600 XGBoosts. Asumible (~2-3 min total sobre SPY).

**Por qué creo que funcionará sobre M10.** Nuestro diagnóstico de M10-v3-walkforward identificó que la **inconsistencia día-a-día** del modelo refittado degrada la calibración. Bagging suaviza esa inconsistencia promediando sobre múltiples modelos del mismo periodo. Es el "abstracto de CPCV" aplicable causalmente.

**Cita central:**
- Inoue, A. & Kilian, L. (2008). *How useful is bagging in forecasting economic time series? A case study of US CPI inflation*. Journal of the American Statistical Association.
- Bergmeir, C. & Hyndman, R. J. (2018). *A note on the validity of cross-validation for evaluating autoregressive time series prediction*.

### Familia B — Online Conformal Prediction (Bates et al. 2023, Angelopoulos 2024)

**Idea.** Wrap a cualquier predictor con una capa de calibración que **garantiza cobertura empírica** sobre cualquier ventana temporal, incluso bajo distribution shift. La calibración se actualiza online cada día.

**Qué resuelve.** El problema diagnosticado en M10-v3-walkforward de que la isotónica refittada cada día con datos crecientes produce mapeos inestables. OCP da una calibración garantizada que se actualiza incrementalmente sin saltos.

**Qué NO resuelve.** OCP da garantías de COBERTURA (intervalos de confianza correctos), NO mejora la accuracy del modelo subyacente. Es complementario a las mejoras de M10-v3, no sustituto.

**Aplicabilidad inmediata a M10:** moderada. La isotónica de v3 ya es bastante buena en CPCV; el problema del walk-forward es la INESTABILIDAD del modelo, no la calibración de probabilidades per se. OCP probablemente aporta menos que bagging en este caso concreto.

**Cita central:**
- Bates, S., Candès, E., Lei, L., Romano, Y. & Sesia, M. (2023). *Testing for outliers with conformal p-values*. Annals of Statistics.
- Angelopoulos, A. et al. (2024). *Online conformal prediction with decaying step sizes*. ICML 2024.
- Recientes 2025: ["Conformal Prediction for Time-series Forecasting with Change Points"](https://arxiv.org/html/2509.02844v3) (NeurIPS 2025).

### Familia C — Adaptive walk-forward con purga + bootstrap interno

**Idea.** Estructura walk-forward pero dentro de cada ventana de refit aplica purga + embargo + bootstrap para intervalos de confianza. Es el camino del paper [arxiv 2512.12924 (Dec 2025)](https://arxiv.org/pdf/2512.12924).

**Qué resuelve.** Ofrece intervalos de confianza honestos y reduce información redundante entre refits. Mejor para REPORTAR robustez estadística que para mejorar el resultado puntual.

**Aplicabilidad inmediata a M10:** baja para mejorar equity; alta para reportar IC honestos en la memoria del TFG.

**Cita central:** El paper arxiv 2512.12924 demuestra que walk-forward con embargo + purga + bootstrap CIs alcanza **"comparable statistical power"** a CPCV bajo condiciones específicas, especialmente con `n ≥ 500`. Para n ≈ 400 (nuestro caso) el gap persiste pero es menor.

---

## 3. Síntesis: qué hace cada método

| Método | Mejora del estimador | Coste computacional | Causalidad estricta | Aplicabilidad a M10 |
|---|---|---|---|---|
| **CPCV** (López de Prado) | √5 reducción varianza por obs | bajo | NO (look-ahead in-sample) | baseline |
| **Bagging walk-forward** (Inoue-Kilian) | √K reducción varianza por obs | medio-alto | SÍ | **alta** |
| **Online conformal prediction** | calibración garantizada | bajo | SÍ | media (complementario) |
| **Adaptive WF + purga** | bootstrap CIs | medio | SÍ | media (reporting) |
| **Walk-forward simple** (lo que tenemos) | sin agregación | bajo | SÍ | baja |

**Conclusión:** *la opción que probablemente cierra el gap CPCV↔walkforward sobre M10 es el bagging walk-forward (Inoue-Kilian)*. Voy a desarrollarlo en la siguiente sección.

---

## 4. Propuesta concreta — M10-v5: bagging walk-forward

### Por qué tiene sentido para M10

Los problemas concretos diagnosticados en walk-forward fueron:

1. Modelos entrenados con muestras pequeñas (los primeros con n=60-120) producen predicciones ruidosas.
2. La calibración (isotónica o Platt) refittada con datos inestables propaga ruido.
3. Los percentiles de abstención y P95 acumulados se contaminan con el ruido inicial.

Bagging walk-forward ataca el problema raíz: **la varianza del estimador en cada refit**. Al promediar K modelos independientes (vía bootstrap), reduces ese ruido por factor √K, lo que estabiliza:
- Las predicciones `p1` día a día (menos saltos).
- La calibración (datos más coherentes).
- Los percentiles acumulados (distribuciones más limpias).

### Diseño técnico de M10-v5 (pre-registro propuesto)

```python
# En cada refit semanal (t cada 5 días):
K = 30  # número de bootstraps
predictions = []
for k in range(K):
    # Moving-block bootstrap del periodo de train (preserva dependencia)
    boot_idx = moving_block_bootstrap(
        n=last_train_idx + 1,
        block_size=10,  # ~2 semanas, captura autocorrelación
        seed=42 + k,
    )
    X_boot = X_np[boot_idx]
    y_boot = y_np[boot_idx]
    clf = XGBClassifier(**XGB_V3_PARAMS)
    clf.fit(X_boot, y_boot)
    predictions.append(clf)

# Para cada día t intra-semana, predicción agregada:
p1_t = np.mean([clf.predict_proba(X_np[t:t+1])[0,1] for clf in predictions])

# Resto del pipeline igual: Platt calibration semanal, abstención 30%, P95.
```

**Parámetros pre-fijados (no se tunean):**
- K = 30 bootstraps (estándar Inoue-Kilian 2008).
- Block size = 10 días bursátiles (≈ 2 semanas, captura autocorrelación típica en retornos diarios SPY).
- Resto idéntico a M10-v4 (burn-in 120, refit semanal, Platt, abstención 30%, P95).

**Coste computacional estimado:** ~80 refits × 30 modelos = 2400 XGBoosts. Sobre SPY ≈ 3 min de cómputo. Coste asumible.

### Criterios de éxito pre-registrables

- log-loss cal OOF < log(2)
- equity > 1.064 (M8)
- Sharpe > +1.5

Mismos criterios que v4. Si no pasa, cierre del experimento sin iterar.

### Por qué podría no funcionar

**Honestidad metodológica:** bagging reduce varianza pero no añade información. Si las features son irreducibly débiles (lo que ya documentamos: log-loss raw walk-forward ≈ log(2) en M10-v4), bagging no inventa señal. Lo más optimista a esperar:
- log-loss baja de 0.713 → ~0.685 (acercándose a log(2)).
- Sharpe sube de +1.37 → ~+1.6-1.8.
- Equity sube de 1.035 → ~1.05-1.08 (potencial empate con M8).

**Si esto pasa:** experimento exitoso, M10-v5 walk-forward iguala a M8 operativamente.
**Si no pasa:** el gap CPCV↔walkforward es estructural en este problema y se reporta como límite teórico. La memoria del TFG citará bagging como mejora aplicada pero insuficiente, lo que **refuerza la honestidad del trabajo**.

---

## 5. Sobre el modo live del TFG

**Lo crítico para tu TFG.** El modo live de STRATA ya está implementado en `live/daily_run.py`. **No usa CPCV ni walk-forward; ejecuta el agente + STRATA + M8 cada día.** Es decir, **el modo live del TFG ya es walk-forward de facto para M5/M8** — no hay desfase.

El desfase aparece SOLO cuando intentas hacer M10 (XGBoost meta-learner) en live. Ahí tienes dos opciones honestas:

### Opción 1 — Modo live solo con M8

Es lo que el TFG ya tiene. M5 + M8 funcionan idénticamente en backtest y en live. M10 solo se reporta para el periodo OOS cerrado (CPCV) como apéndice de robustez ML, sin desplegarlo en live. **La memoria explica que M10 con CPCV es válido como evaluación in-sample mientras que en live se mantiene M8 por estabilidad.**

Ventaja: no añades complejidad al modo live, y M8 es robusta en walk-forward (lo demostramos: M8 vs M10-v4-walkforward son indistinguibles en equity, M10 mejor en Sharpe pero menor magnitud).

Honestidad: declaras la limitación en la memoria. No mientes, no escondes.

### Opción 2 — Modo live con M10-v5 bagging walk-forward

Si M10-v5 (bagging) funciona y bate o iguala a M8 en walk-forward, lo despliegas en live como variante experimental. **Esto sería lo más ambicioso y defendible** pero requiere que el experimento M10-v5 lo confirme primero.

---

## 6. Recomendación final

**Para defender el TFG ahora**, lo más limpio:

1. **Memoria del TFG documenta CPCV como esquema canónico** (López de Prado 2018) y reporta M10-v3 sobre SPY con CPCV (€1148, Sharpe +1.82, log-loss < log 2). Eso es perfectamente defendible académicamente.

2. **Apéndice de robustez** reporta M10-v3 y M10-v4 con walk-forward genuino, mostrando que: (a) el rescate del agente sobrevive en ambos esquemas (DM p<0.02), (b) la superioridad sobre M8 no es robusta al esquema, (c) M8 mantiene techo en walk-forward.

3. **Mencionas el bagging walk-forward (Inoue-Kilian 2008) como trabajo futuro** o (si tienes tiempo y ánimo) ejecutas M10-v5 y lo añades.

4. **El modo live se queda con M8** como configuración operativa, citándolo como decisión deliberada por robustez cross-validation-schemes.

---

## 7. Referencias bibliográficas completas

**CPCV y comparaciones:**
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 7.4. Wiley.
- López de Prado, M. (2018). *The 10 Reasons Most Machine Learning Funds Fail*. [GARP white paper](https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf).
- ScienceDirect 2024 — [Backtest overfitting in the ML era: comparison of OOS testing methods](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110).

**Bagging y walk-forward:**
- Inoue, A. & Kilian, L. (2008). *How useful is bagging in forecasting economic time series? A case study of US CPI inflation*. JASA. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=872856).
- Bergmeir, C. & Hyndman, R. J. (2018). *A note on the validity of cross-validation for evaluating autoregressive time series prediction*.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.).
- arxiv 2512.12924 (Dec 2025) — [Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework](https://arxiv.org/pdf/2512.12924).

**Online conformal prediction:**
- Bates, S. et al. (2023). *Testing for outliers with conformal p-values*. Annals of Statistics.
- Angelopoulos, A. et al. (2024). *Online conformal prediction with decaying step sizes*. ICML.
- NeurIPS 2025 — [Conformal Prediction for Time-series Forecasting with Change Points](https://arxiv.org/html/2509.02844v3).
- arxiv 2024 — [Online Conformal Inference for Multi-step Time Series](https://www.monash.edu/business/ebs/research/publications/ebs/2024/wp20-2024.pdf).

**Bias-variance en rolling windows:**
- [The bias of IID resampled backtests for rolling window models (2025)](https://www.tandfonline.com/doi/pdf/10.1080/10293523.2025.2552592).
- [Investment Model Validation — CFA Institute](https://rpc.cfainstitute.org/sites/default/files/-/media/documents/article/rf-brief/investment-model-validation.pdf).

**Calibración probabilística (ya en M10-v3):**
- Platt, J. (1999). *Probabilistic outputs for SVMs*.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. ICML.

---

## 8. Trazabilidad

- BITACORA entradas relevantes: `[2026-06-15] [Pre-registro/Resultado] - M10-v3-walkforward`, `[Pre-registro/Resultado] - M10-v4-walkforward`.
- Documentos del kit: `M10_V3_GUIA.md` (técnica M10-v3 CPCV), este documento (investigación validación tiempo real), `DECISIONES_ESENCIALES.md` (decisión #10 CPCV).
- Pre-registro M10-v5 PENDIENTE — se ejecuta solo si Raquel lo aprueba como siguiente paso.
