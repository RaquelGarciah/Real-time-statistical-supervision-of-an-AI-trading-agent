# Instrucciones para el Cap. 3 (marco teórico) — qué AÑADIR, su importancia, dónde se usa, y qué QUITAR

> Para la sesión de redacción. La **parte matemática de los modelos está COMPLETA y rigurosa — NO se toca**
> (preliminares, HMM forward/filtrado/Baum-Welch/Viterbi/K, GARCH(1,1)-t, BOCPD, los 3 detectores RAM/PSA/GSO,
> capa M8, aprendizaje supervisado: árboles/boosting/XGBoost/M10/stacking/AutoML). Lo que falta es **toda la caja
> de validación** (de `\section{Métricas de evaluación}` en adelante, líneas ~737–805 del `.tex`): hoy son
> **subsecciones-cabecera vacías**. Abajo, qué escribir en cada una, por qué importa y en qué resultado del Cap. 4
> se usa, más lo que hay que **quitar**.

Principio: el Cap. 3 explica **exactamente** las herramientas que usa el Cap. 4, ni una más (un test enumerado y
no usado es relleno que el tribunal castiga). Cada concepto, con su **cita** en el docstring/nota.

---

## A. AÑADIR — métricas (`\section{Métricas de evaluación}`)

| Subsección | Qué escribir | Importancia / dónde se USA en Cap. 4 | Cita |
|---|---|---|---|
| `sec:economia-sharpe` (Sharpe) — **vacía** | Definición del ratio de Sharpe (media/desv. de los retornos, anualizado ×√252). | **Crítica.** Es la métrica de riesgo titular: tabla de 6 estrategias SPY (§3), **pooled-10 ΔSharpe** (resultado duro O2), ΔSharpe por activo/grupo, confirmatorio. | Sharpe 1966, 1994 |
| `sec:economia-otros` — **vacía** | **maxDD** (máxima caída pico-valle) y **Calmar** (retorno/maxDD). **QUITAR "Sortino"** (no se usa). | maxDD/Calmar salen en la tabla SPY y en **ΔmaxDD pooled** (rescate de riesgo). | Magdon-Ismail (maxDD); Young 1991 (Calmar) |
| **NUEVA: AUC** (hoy solo se menciona 1 vez en AutoML) | Curva ROC y AUC como métrica de ranking de probabilidad. | AutoML elige su *leader* por **AUC** (Purged K-fold) y se reporta la columna AUC en la tabla SPY. | Hanley-McNeil 1982; Fawcett 2006 |
| **NUEVA: accuracy direccional + matriz de confusión** (0 menciones) | accuracy = aciertos/total sobre el signo de $r_{t+1}$; matriz 2×2 (TP/FP/FN/TN), precision y recall del lado largo. | **Es la métrica TITULAR de accuracy** (M5/M8/M10/AutoML); las **matrices de confusión** de §3 (SPY) y §4 (panel) cuelgan de aquí. | estándar (clasificación binaria) |

---

## B. AÑADIR — validación (`\section{Validación}`)

| Subsección | Qué escribir | Importancia / dónde se USA | Cita |
|---|---|---|---|
| `sec:cpcv-causalidad` (signal\_lag=1) — **ya escrita** | — | — | — |
| `sec:cpcv-walkforward` — **ya escrita** | — | — | — |
| `sec:cpcv-purga` (Purga y embargo) — **vacía** | Qué es purgar (quitar solapes etiqueta-feature) y el embargo; por qué el horizonte=1 fija **embargo=1**. | El protocolo desplegable de **M10/AutoML** usa embargo=1; es decisión #4. | López de Prado 2018 (cap. 7) |
| `sec:cpcv-cpcv` (CPCV / por qué no KFold) — **vacía/stub** | Por qué el KFold convencional filtra en series temporales; qué es la **Purged K-Fold** (la que usa AutoML). Breve. | AutoML valida con **Purged K-fold**; principio "no KFold" del proyecto. | López de Prado 2018 (sec. 7.4) |

---

## C. AÑADIR — contraste de hipótesis (`\section{Contraste de hipótesis}`)

| Subsección | Qué escribir | Importancia / dónde se USA | Cita |
|---|---|---|---|
| `sec:tests-sign` (sign test binomial) — **vacía** | Test del signo / binomial vs 0,5. | **O1**: el agente acierta < 0,5 (sign test); tabla de sign-tests SPY. | binomial estándar |
| `sec:tests-mcnemar` (McNemar) — **vacía** | McNemar pareado sobre aciertos correlacionados (tabla 2×2 de discordancias; exacto binomial si pocos pares). | **El test central de accuracy**: rescate del agente (sup vs M5) en §3/§4 y por régimen. | McNemar 1947 |
| **NUEVA: block-permutation** (0 menciones) | Test de permutación por bloques (blinda a McNemar frente a autocorrelación serial). | Robustez del rescate por **sub-ventana** (§7) y **por régimen** (PARTE B). | (permutación por bloques; LdP 2018) |
| `sec:tests-tost` (equivalencia / **TOST**) — **vacía** | Two One-Sided Tests: equivalencia si el IC$(1-2\alpha)$ ⊂ $(-\delta,\delta)$; margen $\delta$ pre-registrado. | **¿el aprendiz redescubre o bate a la regla?** (§4): TOST da superior-no-equivalente en accuracy. | Schuirmann 1987 |
| `sec:tests-bootstrap` (bootstrap estacionario) — **vacía** | Bootstrap estacionario/por bloques (Politis-Romano); IC de la mediana de ΔSharpe respetando dependencia. | **El caballo de batalla del riesgo:** IC del **pooled-10 ΔSharpe**, estratos, DiD, cota Bonferroni. | Politis-Romano 1994 |
| `sec:tests-dsr` (Deflated Sharpe) — **vacía** | DSR = P(Sharpe verdadero > 0) tras descontar E[máx] de $n_{\text{trials}}$ Sharpes (haircut por multiplicidad). | **PARTE B confirmatoria** (§7): DSR AutoML-SPY 0,92; M5/M8/M10 reprueban. | Bailey & López de Prado 2014 |
| **NUEVA: corrección por multiplicidad (Holm / Bonferroni)** (0 menciones) | Holm-Bonferroni; cota de Bonferroni para IC de familia. | $p_{\text{Holm}}$ por régimen y **cota Bonferroni** del confirmatorio de Sharpe (§7). | Holm 1979; Bonferroni 1936 |
| **NUEVA: correlación (Pearson / Spearman)** (0 menciones) | Coef. de Pearson y Spearman, su $t$ y dependencia de $n$. | **La ley del leverage** (§5): $r=-0{,}56$, $p=0{,}093$, $n=10$ (la $p$ depende de $n$ — clave para explicar α=0,10). | estándar |

---

## D. AÑADIR — dos bloques que NO existen como sección (se usan mucho)

| Bloque nuevo | Qué escribir | Importancia / dónde se USA | Cita |
|---|---|---|---|
| **Interpretabilidad: SHAP + permutation importance** | Valores de Shapley para atribución; TreeSHAP (exacto en árboles); permutation importance. | **Toda la universalidad (O4):** cuota SHAP de STRATA 0,66 (>0,5 en 10/10); §3 SHAP dependency; ablación. | Lundberg-Lee 2017; Lundberg 2020 (TreeSHAP); Breiman 2001 (perm. imp.) |
| **Análisis no supervisado: clustering + PCA** | KMeans, Ward (aglomerativo), GMM (+BIC), Spectral; **silhouette**, **índice de Rand ajustado**; **PCA** (reducción 2D). | **Todo §6** (naturaleza→canal): clustering de los 10, consenso Rand=1,0, PC1≈leverage. Sin esto §6 no tiene base teórica. | MacQueen 1967; Ward 1963; Schwarz 1978 (BIC); Hubert-Arabie 1985 (Rand); Jolliffe (PCA) |
| **Definición de *pooled* (bootstrap de panel)** | Apilar los días de los activos en una muestra; su caveat (correlación cruzada → n efectiva < nominal). Puede ir dentro de `sec:tests-bootstrap`. | El **pooled-10** (riesgo) y la frase "la significancia vive en el pooled". | — |

---

## E. QUITAR (enumerados pero NO se usan en el enfoque actual)

- **Sortino** (subsección `sec:economia-otros`): no se usa; se reportan Sharpe/maxDD/Calmar. → fuera.
- **Diebold-Mariano** (`sec:tests-dm`): era del enfoque viejo (P&L equivalente). Hoy la universalidad se argumenta con **SHAP + TOST**, no con DM. → fuera (o una línea como "alternativa no usada").

---

## F. REORGANIZAR la sección de tests (por la pregunta que responde — más claro que una lista)

- **Dirección/accuracy:** sign test · McNemar · block-permutation.
- **Riesgo/retorno:** bootstrap estacionario (ΔSharpe IC) · DSR.
- **Multiplicidad:** Holm · Bonferroni.
- **Equivalencia:** TOST.
- **Asociación:** Pearson/Spearman.
- Cerrar con el encuadre **nominal vs contrastado** (accuracy vs trivial = nominal; rescate = contrastado).

**Orden de redacción sugerido:** métricas → SHAP → clustering/PCA → tests. Cada cifra del Cap. 4 debe poder
señalar la herramienta del Cap. 3 que la sostiene (y al revés: ninguna herramienta del Cap. 3 sin uso en el Cap. 4).
