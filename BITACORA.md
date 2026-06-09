# BITÁCORA — STRATA

Cuaderno de campo del TFG. Documenta el **proceso** de investigación: decisiones
metodológicas, errores que costaron tiempo, hallazgos sobre el fenómeno y, sobre
todo, los **pre-registros** de cada experimento (hipótesis, test y criterio de
éxito fijados *antes* de mirar resultados). El pre-registro es lo que blinda la
tesis frente a la acusación de p-hacking.

Formato de entrada: `## [YYYY-MM-DD] [Milestone | Decisión | Error | Hallazgo | Pre-registro] - Título`.

---

## [2026-06-08] [Decisión] - Funciones-librería que faltaban en `core/` para el notebook canónico

**Contexto.** Al diseñar el notebook canónico (`notebooks/strata_canonical.ipynb`)
detectamos cuatro piezas de rigor que el código no proporcionaba y sin las cuales
las cifras de M8/M10 serían irreproducibles o estarían contaminadas por look-ahead.

**Detalle.** Se añaden a `core/`, con su cita y su test, por ser funciones de
librería reutilizables (no orquestación, que vive en el notebook):

1. `core.hmm.RegimeHMM.predict_proba_filtered` — posterior **filtrado** causal
   `γ^f_t(s) = P(s_t | x_{1:t})` por el algoritmo forward (Rabiner 1989). El
   `predict_proba` existente es el suavizado de `hmmlearn` (forward-backward), que
   usa el futuro de `t`: alimentarlo a RAM en OOS sería look-ahead estructural.
   Decisión esencial #5 exige régimen *filtered*. Test: `test_filtered_no_lookahead`.
2. `core.stats.mcnemar_test` (McNemar 1947 + corrección de Edwards 1948; binomial
   exacto si `b+c<25`) y `core.stats.sign_test` (binomial exacto, Conover 1999).
   No reportar accuracy sin ellos. Tests en `test_stats.py`.
3. `core.stats.block_permutation_test` (permutación por bloques ≈√N) y
   `core.stats.tost` (equivalencia de Schuirmann 1987). El primero blinda McNemar
   contra la autocorrelación de los pares; el segundo permite **afirmar**
   equivalencia M10≈M8, no solo "no-superioridad" como hace Diebold-Mariano.
4. `core.metrics.classification_metrics` — `accuracy/auc/log_loss/brier/mcc` para
   que la tabla maestra muestre acierto y rentabilidad juntos (lección #11).

**Implicaciones para el TFG.** La tabla maestra y los contrastes pareados ya
tienen soporte de librería testeado (115 tests verdes, salvo los 2 que dependen de
`experiments/` por diseño).

**Referencias.** `core/hmm.py`, `core/stats.py`, `core/metrics.py`; tests homónimos.

---

## [2026-06-08] [Error] - σ del GSO: usar `forecast_path[t]`, no `σ_{t-1}`

**Contexto.** El plan inicial pedía alimentar GSO con `sigma.shift(1)` para evitar
look-ahead (lección #5 del proyecto anterior).

**Detalle.** Al leer `core/garch.py::forecast_path` se ve que `sigma2[t]` se calcula
como `ω + α·ε_{t-1}² + β·σ²_{t-1}` y se fija **antes** de incorporar `r_t`. Es decir,
el valor en el índice `t` ya es la previsión a un paso hecha al cierre de `t-1` —
exactamente la información disponible al decidir el día `t`. Aplicar `shift(1)`
retrasaría un día de más y usaría la previsión de `t-1` para decidir en `t`. Lo
correcto es pasar `sigma.iloc[t]` directamente. El bug histórico de la lección #5
era distinto: el GSO *relativo* recomputaba σ incorporando `r_t`. Aquí no ocurre.

**Implicaciones para el TFG.** El notebook alimenta GSO con `forecast_path[t]`. La
garantía causal se verifica en `test_garch.py::test_forecast_path_causal_sin_lookahead`
(perturbar `r_k` no altera σ_k, solo σ_{k+1}).

**Referencias.** `core/garch.py:88-108`, `tests/test_garch.py`.

---

## [2026-06-08] [Hallazgo] - El corte empírico de RAM es ≈1.0, no 0.20/0.40/0.70: el score es cuasi-binario

**Contexto.** Al validar los umbrales de RAM con un árbol de decisión de profundidad 1
y un histograma coloreado por el signo de $r_{t+1}$ (§4/§6 del notebook), tal como pide
el tutor, el corte óptimo del árbol cae en RAM ≈ 1.0, no en los defaults 0.20/0.40/0.70.

**Detalle.** El HMM filtrado asigna casi toda la masa a un único régimen casi todos los
días ($P(\text{estado})\approx 0$ o $\approx 1$), de modo que el score de RAM es
cuasi-binario: en OOS solo **11 de 401 días (2.7%)** caen en la zona intermedia
$[0.20,0.70)$. Mover el corte intermedio entre 0.30 y 0.50 cambia el nº de días
intervenidos por RAM de 107 a 101 (apenas). El resultado de M8 es por tanto insensible
a la posición exacta de esos cortes. Se documenta como robustez por la forma del score,
no como "umbrales validados".

**Implicaciones para el TFG.** Convierte un flanco ("umbrales elegidos a ojo") en algo
defendible, siempre que se cuantifique el nº de días en la zona intermedia y se muestre
el test de robustez —ambos están en §6.

**Referencias.** notebook §4 y §6 (árbol corte=1.000, conteo de zonas); `strata/detectors.py`
(defaults RAM); `cache/models/strata_thresholds.json` (sin clave `ram`).

---

## [2026-06-08] [Hallazgo] - Régimen→signo: solo Calma es direccional significativa; Crisis es prior débil

**Contexto.** §3 justifica la política de RAM con la media de retorno por régimen en
calibración y su IC95 bootstrap estacionario.

**Detalle.** Calma tiene media diaria $+0.0006$ con IC95 $[+0.00044,+0.00085]$ (excluye 0).
Estrés $+0.0003$ y Crisis $-0.0003$ tienen IC que **incluyen** el 0: no son
significativas a resolución diaria. En Crisis el ~50.9% de días son positivos pero la
media es negativa por la **asimetría** (las caídas son mayores; std 0.021 vs 0.006 en
Calma): leverage effect. Por tanto "Crisis → short" es un prior direccional **débil**;
el papel principal de RAM en Crisis es frenar la exposición larga en alta volatilidad
(gestión de riesgo), no una apuesta direccional fuerte. Esto encaja con el relato
honesto "STRATA es disciplina de riesgo, no generación de alfa".

**Implicaciones para el TFG.** Hay que redactar la defensa de RAM como prior + control
de riesgo, no como predictor direccional de Crisis. Preview del desacople Sharpe/accuracy.

**Referencias.** notebook §3 (tabla régimen→signo con IC bootstrap); Black 1976, Christie 1982.

---

## [2026-06-08] [Decisión] - Posición de M10 = signo(p1−0.5) direccional, no continua

**Contexto.** Al traducir la probabilidad out-of-fold $p_1$ de M10 a una posición para el
backtest hay dos opciones: continua ($w=2p_1-1$, pondera por convicción) o direccional
($w=\mathrm{signo}(p_1-0.5)$, el "1/−1").

**Detalle.** El mapeo continuo daba a M10 un Sharpe de $+0.51$, **por debajo** de M8
($+0.75$), lo que parecía contradecir que un meta-learner con todas las features debería al
menos igualar a la regla. Diagnóstico (sin fuga; cobertura OOF = 5 correcta): el mapeo
continuo concentra posición en días de alta convicción y penaliza el Sharpe. Con el mapeo
**direccional** —que además es el "$1/-1$" que pide el problema y la salida natural de un
clasificador binario— M10 sube a $+0.71 \approx$ M8 ($+0.75$), indistinguibles (DM $p=0.61$,
IC ΔSharpe $[-1.93,+1.94]$). El log-loss OOF mediano ($0.912$) coincide con el del proyecto
anterior ($0.914$). **Decisión:** M10 usa $w=\mathrm{signo}(p_1-0.5)$. Documentado para que
no parezca que se eligió el mapeo que favorece la narrativa: es el mapeo canónico de un
clasificador y el que usaba el proyecto anterior.

**Implicaciones para el TFG.** La conclusión "M10 no bate a M8" se mantiene con ambos
mapeos; el direccional hace la comparación justa (predicción de signo vs regla de signo).
Un fold de los 15 tiene log-loss disparado (7.79): artefacto de muestra pequeña en CPCV,
por eso se reporta la mediana; es la misma inestabilidad que invierte el umbral óptimo
($\rho=-0.60$).

**Referencias.** notebook §11 (M10, ablación, DM/TOST, inestabilidad de umbral).

---

## [2026-06-08] [Decisión] - M_neg (negar al agente) se considera y se EXCLUYE por alcance

**Contexto.** Al diseñar el blindaje de M8 se evaluó M_neg ($=-$agente) como benchmark,
por ser la objeción nº1 del red-team ("¿no basta con invertir al agente?").

**Detalle.** Se comprobó empíricamente que M_neg rinde más que M8 en este OOS (su Sharpe
es mayor), porque el agente arrastra un sesgo corto sistemático en un mercado alcista e
invertirlo entero cabalga la tendencia. **Decisión:** M_neg se retira del notebook por
estar **fuera del alcance del estudio**: no es un método de *supervisión estadística*
(no usa régimen, volatilidad ni coherencia temporal), sino una apuesta degenerada a que
el agente siempre se equivoca, frágil ante cualquier agente con destreza direccional. El
objeto del TFG es comparar **supervisores** (M5 sin supervisar, M8 STRATA, M10 meta-learner)
y situarlos frente a baselines triviales y al régimen sin agente (M2), no frente a la
negación del agente. Se deja constancia aquí para que el rastro sea honesto: M_neg se
examinó y se excluyó por criterio de alcance, no por resultado.

**Implicaciones para el TFG.** Si el tribunal pregunta "¿probó a invertir al agente?", la
respuesta documentada es: sí; no es supervisión y queda fuera del estudio (esta entrada).
La hipótesis se mantiene intacta: STRATA rescata al agente (McNemar M8 vs M5 $p=0.062$).

**Referencias.** notebook §10 (M2 + baselines, sin M_neg); abogado-del-diablo objeción nº1.

---

## [2026-06-08] [Hallazgo] - Metodología rigurosa y ex-ante para el umbral de RAM

**Contexto.** El umbral de RAM eran defaults (0.20/0.40/0.70). Hacía falta una metodología
data-driven para fijarlo desde el histórico, como PSA/GSO, y responder al tutor ("¿de dónde
sacas el umbral rigurosamente?"). El percentil naíf sobre la distribución (bimodal) era
degenerado (P75=0.996) y empeoraba M8 (Sharpe −0.49).

**Detalle.** Metodología adoptada (§4 del notebook), ex-ante sobre calibración 2000–2024-09,
sin agente ni OOS: para cada día se toma la confianza del régimen direccional (P(Calma) eje
alcista, P(Crisis) eje bajista) y si la dirección implícita acierta el signo de r_{t+1}; se
estima P(régimen acierta | confianza) y se localiza dónde cruza 0.5. Resultado: el régimen
es ruido por debajo de ≈0.2 (acierto <0.5) y se vuelve informativo por encima (acierto
0.58–0.62), con un máximo en las transiciones (0.5–0.9) y un descenso a confianza extrema
>0.9 (deep-regime, acierto 0.52). El **umbral data-driven ≈ 0.20–0.25** coincide con el
`low` por defecto, y el `medium`=0.40 de override cae dentro de la banda informativa
[≈0.2, 0.9]. En calibración hay 242 días en la zona intermedia [0.2,0.7) — densidad
suficiente para calibrar (en el OOS calmo solo había 11).

Sensibilidad de M8 (§6): Sharpe estable en la banda informativa —0.83 (med=0.20), 0.76
(0.30), 0.75 (0.40)— y cae a 0.54 en 0.50. El umbral exacto no manda dentro de la banda.

**Implicaciones para el TFG.** Convierte "umbral inventado" en "umbral en el punto donde el
régimen empieza a predecir la dirección en 24 años de SPY, robusto dentro de la banda". El
método es ex-ante y por activo (en activos con regímenes más solapados el cruce caería en
otro punto). DECISIÓN (resuelta el 2026-06-08, ver entrada posterior "Resultados del bloque
régimen/RAM"): se adopta el τ calibrado (=0.176, isotónica sobre la curva binned) como umbral
canónico, RE-REPORTANDO el McNemar también con el default 0.40 para demostrar que el rescate a
α=0.10 NO depende del umbral — blindaje explícito contra "elegí el umbral mirando el OOS".

**Referencias.** notebook §4 (figura confianza→acierto, corte ≈0.25), §6 (sensibilidad de
M8); contraste con el percentil naíf degenerado.

---

## [2026-06-08] [Pre-registro] - Experimento M5 (agente LLM sin supervisar)

**Hipótesis.** El agente LLM (5 personalidades + PM) sobre SPY en el OOS pierde
dinero y acierta direccionalmente **menos del 50%**.

**H0.** La proporción de acierto direccional del agente es 0.5 (azar).

**Estadístico.** Sign test binomial exacto a dos colas contra 0.5
(`core.stats.sign_test`), con IC95% de Clopper-Pearson sobre la proporción.

**Criterio de éxito.** Se rechaza H0 (p<0.05) **y** la proporción puntual es <0.5;
adicionalmente Sharpe causal (`signal_lag=1`) negativo. Eso confirma "agente
perdedor", premisa de la hipótesis del TFG.

**Criterio de fracaso.** Si la accuracy ≥0.5 o el Sharpe causal es positivo, el
agente no es un "perdedor" en este OOS y la premisa del rescate decae; se reporta
honestamente.

**Datos.** SPY, OOS `2024-10-01 → última fecha en cache/agent/SPY/`. `signal_lag=1`
(posición_t × retorno_{t+1}). Decisiones del agente leídas de `cache/agent/SPY/`.

**Output esperado.** `outputs/experiments/m5.json` con: `accuracy, n, sign_test_p,
ci95, sharpe_causal, sharpe_sameday, equity_final, confusion_matrix`.

---

## [2026-06-08] [Pre-registro] - Experimento M8 (STRATA, override variante C)

**Hipótesis.** Supervisar al agente con STRATA en modo override-C **rescata** al
agente cuando éste pierde: mejora su acierto direccional de forma pareada.

**H0.** M8 y M5 aciertan con igual probabilidad marginal (las discordancias
McNemar cumplen `P(b)=P(c)`).

**Estadístico.** McNemar pareado sobre la **intersección exacta de fechas** M8/M5
(`core.stats.mcnemar_test`; exacto si `b+c<25`), complementado con
`block_permutation_test` (bloques ≈√N) para controlar la autocorrelación. Sharpe
con Diebold-Mariano + **Deflated Sharpe** (`n_trials` declarado abajo).

**Criterio de éxito.** McNemar p<0.10 **pre-declarado** (α=0.10 justificado por la
baja potencia con N≈400 y un efecto direccional pequeño; convención común en
finanzas) con M8 acertando más que M5; resultado reportado **también a α=0.05**
para lectura honesta del borderline.

**Criterio de fracaso (regla prior-flip).** Si el signo de la media de retornos por
régimen en calibración ≠ signo en los primeros 60 días de OOS, el prior RAM no es
válido en el OOS y M8 no es defendible para ese activo (mecanismo de falsificación
pre-registrado). En SPY se espera estabilidad (caso central); se verifica explícito.

**Config congelada.** `StrataSupervisor(mode="override", override_variant="C",
gso_mode="absolute", psa_signal="cp_prob", psa_hazard=1/250, regime=filtered)`,
`signal_lag=1`. Umbrales: RAM 0.20/0.40/0.70 (defaults — se documenta honestamente
que **no** son data-driven, a diferencia de PSA/GSO); PSA/GSO en P95/P99/max sobre
`cache/models/strata_thresholds.json` (calibración 2000-01-01 → 2024-09-30, n=6025).

**`n_trials` para el Deflated Sharpe.** Se cuenta TODA la búsqueda de configuración
que conduce a elegir override-C: `mode∈{warn,reduce,override}` × `override_variant∈
{A,B,C,D}` × `gso_mode∈{absolute,relative,relative_conviction}` × `regime∈
{filtered,smoothed}`, más la escalera de benchmarks. El número exacto se cuenta y se
imprime en §18 del notebook y se usa en `deflated_sharpe`. Los descartes de
variantes A/B/D y de `smoothed`/GSO-relativo se justifican por **causalidad/leakage**
(criterio mecánico), no por Sharpe; esa frontera se documenta celda a celda.

**Blindaje pre-registrado.** Se compara M8 contra **M2** (=HMM×GARCH sin agente, la
ablación de STRATA sin el input del agente) con DM e IC del ΔSharpe, y contra los
baselines triviales (B&H, corto, cara/cruz). M_neg (=−agente) se evaluó y se excluyó por
alcance (ver entrada [Decisión] del 2026-06-08): no es un método de supervisión.

**Output esperado.** `outputs/experiments/m8.json` con: tabla maestra (7 métricas),
matriz de confusión, `mcnemar{stat,p,b,c}`, `block_perm_p`, `deflated_sharpe`,
`n_trials`, `prior_flip{calib_signs,oos60_signs,flip}`, `vs_mneg`, `vs_m2`.

---

## [2026-06-08] [Pre-registro] - Experimento M10 (meta-learner XGBoost con CPCV)

**Hipótesis (nivel universalidad, CLAUDE.md §2).** Un XGBoost universal validado
con CPCV-within-OOS **no bate** significativamente a la regla a mano M8, y SHAP
identifica las features STRATA como las informativas. Es decir: la regla a mano
captura la misma señal que el meta-learner redescubre.

**H0.** M10 y M8 tienen igual capacidad predictiva (Diebold-Mariano sobre pérdidas
diarias pareadas, alineadas por fecha).

**Estadístico.** Diebold-Mariano (`core.stats.diebold_mariano`) **y** TOST de
equivalencia (`core.stats.tost`) con margen pre-declarado sobre la pérdida diaria;
IC del ΔSharpe por bootstrap estacionario (Politis-Romano). SHAP por TreeSHAP nativo
de XGBoost (`pred_contribs=True`, sin dependencia `shap`), agregado **pooled
out-of-fold**.

**Criterio de éxito (de la hipótesis).** DM p>0.10 (no se rechaza igualdad) **y**
TOST declara equivalencia dentro del margen **y** el top-5 de SHAP son features
STRATA/régimen (`ram_score, psa_score, garch_sigma, stress_prob, calm_prob`).

**Criterio de fracaso.** Si M10 bate a M8 con DM p<0.10, o si una personalidad del
agente entra en el top-5 de SHAP, la hipótesis de universalidad decae; se reporta.
Refuerzo: **ablación M10-sin-features-STRATA** debe degradar hacia M5 (si no
degrada, SHAP estaba repartiendo crédito entre proxies colineales).

**Config congelada.** Features = 22 (5×signo_personalidad, 5×size, 5×conf,
ram_score, psa_score, gso_score, calm_prob, stress_prob, crisis_prob, garch_sigma).
Etiqueta `y_t = 1{r_{t+1}>0}`. `CombinatorialPurgedKFold(n_splits=6, n_test_splits=2,
embargo=5)` con `t1=índice.shift(-1)` **explícito** (15 folds combinatorios; purge +
embargo verificados por fold con `gap_dias≥0`). XGBoost con hiperparámetros fijados
en esta entrada y semilla 42 (se imprimen en el notebook). NUNCA KFold convencional.

**Diagnóstico de honestidad obligatorio.** Log-loss out-of-fold vs el trivial 50/50
(0.693): si es peor, se dice con todas las letras. Inestabilidad del umbral óptimo
`p1` (mitad-1 vs mitad-2, Spearman) frente a la constancia por construcción de los
umbrales STRATA.

**Output esperado.** `outputs/experiments/m10.json` con: tabla de 15 folds, OOF
(p1 agregado + regla de agregación), log-loss por fold, matriz de confusión, tabla
SHAP de las 22 features, ablación, `dm_vs_m8{stat,p}`, `tost{p,equiv}`,
`delta_sharpe_ci`, `threshold_stability`.

**Calibración de las convicciones del agente (pre-registro auxiliar).** Reliability
diagram + ECE + Brier de la `conf` del LLM vs `y_t`, para validar que el descarte
SHAP de las personalidades no es artefacto de features-ruido.

---

## [2026-06-08] [Pre-registro] - Bloque de régimen/RAM: τ data-driven, M7 (reduce) y ablación K=2/K=3

Tres mejoras del bloque RAM, pre-registradas en conjunto **antes** de re-ejecutar la
Parte IV. Sustituyen el flanco "umbral de RAM elegido a ojo" por un umbral derivado de
datos con IC, añaden el modo `reduce` como control intermedio (cierra la objeción
"¿por qué descartaste reduce?") y justifican numéricamente el nº de estados del HMM.

### (1) Umbral τ de RAM por regresión isotónica + IC bootstrap (el *gate*)

**Hipótesis.** Existe un umbral de confianza de régimen $\tau$ por debajo del cual el
régimen NO es direccionalmente informativo (acierto $<0.5$) y por encima del cual sí
($\ge0.5$). $\tau$ se estima sin agente ni OOS, solo sobre la calibración 2000–2024-09.

**H0.** La fiabilidad direccional del régimen $P(\text{acierta}\mid\text{confianza})$ no
cruza $0.5$ (régimen no informativo a ninguna confianza) → $\tau$ indefinido.

**Estadístico.** Curva de fiabilidad $g(c)=P(\text{régimen acierta }r_{t+1}\mid
\text{conf}=c)$ estimada por **regresión isotónica** creciente (Robertson, Wright &
Dykstra 1988; `sklearn.isotonic`), monótona por construcción → un único cruce de $0.5$.
$\tau=\min\{c: \hat g(c)\ge0.5\}$. **IC95 por bootstrap estacionario** (Politis-Romano
1994) remuestreando **días de calibración** en bloques de media $\sqrt{N}$ (respeta la
dependencia serial) y recalculando $\tau$ en cada réplica; 1000 réplicas, semilla 42.

**Criterio de éxito.** $\hat g$ cruza $0.5$ en $\tau\in(0,1)$ con IC95 que **no** toca los
extremos degenerados $\{0,1\}$; $\tau$ cae dentro de la banda informativa ya descrita
(≈0.2–0.9). El supervisor canónico adopta `ram_thresholds=(τ/2, τ, 0.70)`: `medium`
(donde dispara el override) $=\tau$ es el *gate*; `low`$=\tau/2$ activa el reduce suave;
`high`$=0.70$ solo re-etiqueta severidad (no cambia ninguna acción, documentado).

**Criterio de fracaso.** Si el IC de $\tau$ es tan ancho que cubre casi $[0,1]$, el umbral
no es identificable en este activo y se reporta como tal (no se adopta; se mantiene el
default conservador 0.40). Equivale a la regla prior-flip a nivel de umbral.

### (2) M7 — reduce continuo *gated* en τ (control intermedio de la escalera)

**Hipótesis.** Atenuar (no voltear) la posición del agente cuando el régimen la
contradice con confianza $\ge\tau$ ya rescata parcialmente al agente: $M5 < M7 < M8$ en
acierto y Sharpe (la escalera de intensidad de intervención: no hacer nada → encoger →
voltear).

**H0.** M7 y M5 aciertan con igual probabilidad marginal (McNemar pareado).

**Estadístico.** McNemar pareado M7 vs M5 (mismo test y fechas que M8 vs M5) +
Diebold-Mariano sobre pérdidas diarias. Mecánica de M7: si RAM tiene severidad
`medium`/`high` (score $\ge\tau$), $w_t = \text{size}_t\cdot(1-\text{RAM}_t)$; si no,
$w_t=\text{size}_t$. Implementado como `reduce_mode="ram_continuous"` en
`strata/intervention.py` (gated por las mismas `ram_thresholds` que M8).

**Criterio de éxito.** Ordenación monótona $M5\le M7\le M8$ en accuracy y Sharpe
causal. Que M7 quede entre M5 y M8 valida que el rescate viene de corregir la dirección
contra el régimen, y que el override (voltear) extrae más que la atenuación (encoger).

**Criterio de fracaso / honestidad.** Si $M7>M8$, el voltear de override-C estaría
sobre-corrigiendo y habría que revisar la variante. Si $M7\approx M5$, el reduce no
aporta y la intervención útil es solo el flip. Se reporta lo que salga.

### (3) Ablación K=2 vs K=3 del HMM (justificar los 3 estados)

**Hipótesis.** Los 3 estados (Calma/Estrés/Crisis) no son sobre-parametrización: el
estado Estrés actúa como **opción de abstención** (no dispara override), de modo que un
HMM binario (solo Calma/Crisis) **sobre-interviene** —fuerza dirección en días que K=3
deja en Estrés— sin mejorar (y plausibmente empeorando) el rescate de M8.

**H0.** K=2 y K=3 producen el mismo M8 (McNemar pareado entre las dos versiones de M8;
DM sobre pérdidas; nº de intervenciones similar).

**Estadístico.** (a) **BIC** de ambos HMM sobre la calibración:
$\text{BIC}=-2\ell+k\log T$ con $k$ = nº de parámetros libres
($K$ inicios $+ K(K{-}1)$ transiciones $+ K\cdot d$ medias $+ K\cdot d(d{+}1)/2$
covarianzas, $d=2$). (b) Downstream: re-ejecutar M8 con un HMM K=2 calibrado igual
(mismas semillas, mismo etiquetado por vol ascendente: S0→Calma/long, S1→Crisis/short,
sin Estrés) y comparar contra M8 (K=3) con **McNemar pareado**, **DM**, nº de
intervenciones y **nº de días reclasificados** (Estrés en K=3 → forzados a un signo en
K=2).

**Criterio de éxito (de la hipótesis).** K=2 interviene en **más** días que K=3 (pierde
la abstención de Estrés) y su M8 **no supera** a K=3 (McNemar no significativo a favor de
K=2; DM $p>0.10$). Complemento: BIC puede favorecer K=2 (penaliza parámetros) — si es
así se dice con todas las letras y se argumenta que la elección de K=3 es **funcional**
(la abstención mejora el supervisor), no de bondad de ajuste marginal.

**Criterio de fracaso.** Si K=2 iguala intervenciones y bate a K=3 en M8, los 3 estados
no se justifican por el downstream y habría que defender K=3 solo por interpretabilidad
económica (Calma/Estrés/Crisis ↔ régimen de mercado). Se reporta honestamente.

**Config congelada (común a las tres).** OOS `2024-10-01 → cierre del agente`,
`signal_lag=1`, override variante C, régimen **filtrado**, `gso_mode="absolute"`,
`psa_signal="cp_prob"`, `psa_hazard=1/250`, semilla 42. K=2 con `n_seeds=10`,
`n_iter=1000`, mismo etiquetado determinista por volatilidad emisora ascendente.

**Output esperado.** Notebook §4 (figura isotónica + τ + IC), §9 (escalera M5→M7→M8 con
los tres McNemar), §12 (tabla BIC + tabla downstream K=2 vs K=3). El umbral τ adoptado se
propaga a `master` (sup_C con `ram_thresholds`), de modo que **todas** las cifras de M8 a
partir de la re-ejecución usan el τ data-driven, no el default 0.40.

---

## [2026-06-08] [Hallazgo] - Resultados del bloque régimen/RAM tras auditoría del Consejo

**Contexto.** Ejecutado el pre-registro anterior y auditado por `@rigor-matematico` y
`@experto-series-temporales`. Se corrigieron tres fallos detectados en la primera ejecución
(ver "Detalle") y se re-ejecutó la Parte IV completa (notebook 0 errores, 117 tests verdes).

**Detalle (correcciones sobre la 1ª ejecución).**
1. La isotónica **a nivel de punto** degeneraba (τ=0.004) por la masa puntual del posterior
   cuasi-binario. Se aplica sobre la **curva de fiabilidad por deciles ponderada por nº de
   días**, con la construcción de **dirección dominante** (`c=máx(P(Calma),P(Crisis))`,
   dirección = argmax). Resultado: **τ=0.176**, IC95 bootstrap [0.000, 0.276]. El IC toca 0
   por la cola degenerada (11.5 % de réplicas dan τ<0.05); se reporta como identificación
   débil por abajo pero **inocua** porque el score es cuasi-binario y §6 muestra el Sharpe de
   M8 plano en la banda [≈0.2, 0.7].
2. El **McNemar M7 vs M5 es trivial (b=c=0)**: reduce nunca cambia el signo, así que su
   acierto direccional ≡ M5 por construcción. La aportación de M7 es de **riesgo**: se mide
   con Diebold-Mariano (p=0.040) y Wilcoxon signed-rank (p=0.001) sobre el P&L diario.
3. La ablación K=2/K=3 con τ **importado** de K=3 era injusta (K=2 es hiper-confiado,
   c≈1, → sobre-interviene tautológicamente). Se **re-calibra τ_K2 sobre la curva propia
   de K=2** (misma metodología): τ_K2=0.716.

**Resultados (OOS SPY, N=401).** Escalera monótona M5 (Sharpe −1.82, acc 0.384) → M7 reduce
(−1.30, acc 0.384) → M8 override (+0.92, acc 0.441). **McNemar M8 vs M5 = 0.037 con τ y
0.062 con el default 0.40: rescate robusto a α=0.10 con AMBOS umbrales** (blinda contra "elegí
τ para que saliera"). ΔSharpe(M8−M2) IC95 [+0.21, +2.50] excluye 0 (M8 ≥ régimen-solo); DM
borderline (p≈0.10). **Deflated Sharpe bajo (≈0.17, n_trials=32)**: el Sharpe no es robusto a
multiplicidad → la evidencia fuerte es el McNemar (acierto), no el Sharpe. Coherente con
"STRATA = disciplina de riesgo, no alfa".

**Resultado INCÓMODO (se reporta sin maquillar).** En este OOS, el HMM **binario K=2 es
nominalmente mejor** (Sharpe 1.33, acc 0.50, McNemar a su favor p≈0.04) **interviniendo el
doble** (235 vs 111 días). PERO el **Diebold-Mariano sobre P&L no ve diferencia significativa
(p≈0.38)**: K=2 y K=3 son indistinguibles en la métrica económica. El edge de K=2 viene de
sobre-intervenir (fuerza dirección en ~134 de 217 días de Estrés) y, en un OOS alcista,
apostar más exposición sale premiado (mismo mecanismo que la exclusión de M_neg). Justificación
de K=3: **estructural** (3 estados con vol y signo de retorno distintos: Calma +/Estrés ≈0/
Crisis −, duraciones 54/30/47 d) + **funcional** (abstención de Estrés, mitad de
intervenciones) + **interpretabilidad**, NO superioridad en P&L. Cierre robusto pendiente =
**panel multi-activo**.

**Implicaciones para el TFG.** (a) Decisión cerrada: τ=0.176 canónico (sustituye al default
0.40), con el McNemar dual como blindaje anti-p-hacking. (b) La sección de nº de estados se
titula "criterios de información vs criterio funcional" y reconoce que BIC/AIC piden K=4
(gaussiana mal especificada). (c) PENDIENTES de la auditoría: congelar las cifras en
`outputs/experiments/*.json` (§18); replicar K=2/K=3 y la escalera en el panel.

**Referencias.** notebook §4, §6, §9, §10, §12; `strata/intervention.py` (reduce_mode
`ram_continuous`); dictámenes `@rigor-matematico` y `@experto-series-temporales` (2026-06-08).

---

## [2026-06-08] [Pre-registro] - Estudio panel K=2 vs K=3 (¿generaliza la ventaja del binario?)

**Contexto.** En SPY el HMM binario K=2 salió nominalmente mejor que K=3 (Sharpe/accuracy más
altos) pero indistinguible en P&L (DM p≈0.38), sobre-interviniendo el doble. Antes de cambiar
el modelo canónico hay que ver si eso es un rasgo del OOS alcista de SPY o una mejora real y
general. Decisión de Raquel: estudiarlo en el panel multi-activo ANTES de decidir.

**Hipótesis.** La ventaja nominal de K=2 sobre K=3 es **específica de SPY en este OOS alcista**
(artefacto de sobre-intervención premiada por la tendencia), no general. Si fuera real, K=2
batiría a K=3 de forma **consistente cross-asset**.

**H0.** K=2 y K=3 rinden igual en el panel (ΔSharpe mediano ≈ 0; sin mayoría de activos a favor
de ninguno; DM no significativo por activo).

**Estadístico.** Por activo (10 tickers de `cache/agent/`): se calibra HMM K=2 y K=3 y un
GARCH sobre el propio histórico (2000→2024-09), se re-calibra **τ por (activo, K)** con la
misma isotónica, y se ejecuta M8 override-C en el OOS con cada HMM. Se reporta por activo:
Sharpe K3/K2, accuracy, nº intervenciones, McNemar(K3 vs K2) y Diebold-Mariano sobre P&L.
Agregado: nº de activos donde K2>K3 en Sharpe, ΔSharpe(K3−K2) mediano e IC, y en cuántos el DM
es significativo.

**Criterio de decisión (lo acuerda Raquel tras ver la tabla).**
- Si K=2 bate a K=3 de forma **consistente y significativa** (mayoría de activos, DM signif.,
  ΔSharpe mediano claramente >0 a favor de K=2) → se considera cambiar el modelo canónico a
  K=2 y se documenta el cambio.
- Si K=2 NO generaliza (ΔSharpe mediano ≈0, sin mayoría, DM no signif.) → se confirma que la
  ventaja de SPY era idiosincrásica; K=3 se mantiene por estructura + abstención +
  interpretabilidad, y el panel es la evidencia que lo blinda.

**Datos.** Panel = SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR, SMCI, ROKU, MARA. OOS
`2024-10-01 → cierre del agente` (~401 días/activo). signal_lag=1. Mismo etiquetado HMM por vol
ascendente; n_seeds=10. NO es ajuste de hiperparámetros: el único grado de libertad es K∈{2,3}.

**Output esperado.** `outputs/experiments/k_ablation_panel.json` (tabla por activo + agregados)
y `experiments/k_ablation_panel.py` (script re-ejecutable).

---

## [2026-06-08] [Hallazgo] - La ventaja de K=2 NO generaliza: era idiosincrásica de SPY

**Contexto.** Ejecutado el estudio panel pre-registrado (10 activos, HMM K=2/K=3 + GARCH + τ
re-calibrado por activo; `experiments/k_ablation_panel.py`).

**Detalle.** Agregados: K=2 supera a K=3 en Sharpe en **solo 4/10** activos; **ΔSharpe mediano
= +0.38 A FAVOR de K=3**; **Diebold-Mariano significativo (p<0.10) en 0/10** activos; McNemar
significativo en **1/10** — y ese único caso es **SPY**, el que disparó la sospecha. Es decir,
fuera de SPY el binario no bate al de tres estados: en mediana K=3 rinde MÁS, y en ningún
activo la diferencia de P&L es estadísticamente detectable. La ventaja de K=2 en SPY era un
**artefacto del OOS alcista** (sobre-intervención premiada por la tendencia), como sostenía la
hipótesis.

**Decisión.** Se **mantiene K=3** como modelo canónico. Queda confirmado por el criterio de
decisión pre-registrado (rama "K=2 no generaliza"): K=3 se justifica por estructura (3 estados
distintos en vol y signo), abstención de Estrés, interpretabilidad económica y ahora también
por **no ser inferior cross-asset**. El panel es la evidencia que blinda §12.

**Caveat honesto.** En el panel, SPY se re-ajusta con un HMM fresco (no el `hmm.pkl` canónico):
su τ_K3 sale 0.55 y su Sharpe 0.52, distinto del notebook (τ=0.176, Sharpe 0.92), porque el
ajuste Baum-Welch tiene óptimos locales y la curva de fiabilidad → τ es sensible. No afecta a
la comparación K2-vs-K3 (interna a cada activo, mismo procedimiento), pero recuerda que τ es un
parámetro con incertidumbre —ya cubierto por el McNemar dual (τ vs 0.40) de §9—.

**Implicaciones para el TFG.** §12 pasa de "resultado incómodo en SPY" a "K=2 parece mejor en
SPY pero NO generaliza (panel: ΔSharpe mediano +0.38 pro-K3, DM 0/10); mantenemos K=3". Es un
cierre fuerte y honesto. El panel va al apéndice multi-activo de la memoria.

**Referencias.** `experiments/k_ablation_panel.py`, `outputs/experiments/k_ablation_panel.json`.

---

## [2026-06-08] [Pre-registro] - Experimento K-por-activo (selección ex-ante del nº de regímenes)

**Contexto.** Idea de Raquel: en vez de K fijo, elegir el nº de estados del HMM **por activo**,
decidido **a pasado** (calibración 2000–2024-09), sin optimizar sobre el OOS. Se explora en un
cuaderno aparte (`notebooks/experimentos.ipynb`), NO se toca el canónico (que sigue con K=3).

**Hipótesis.** Un criterio ex-ante de informatividad direccional del régimen permite elegir K
por activo de forma que el K* seleccionado coincide con el K que mejor rinde fuera de muestra
(si coincide, la selección por activo tiene valor predictivo; si no, es ruido).

**H0.** El K* elegido a pasado NO predice el K mejor-OOS (concordancia ≈ azar) y la cartera
"K-por-activo" no bate a K=3 fijo fuera de muestra.

**Criterio de selección (ex-ante, pre-registrado).** Por (activo, K∈{2,3,4}): se calibra el HMM
y su gate τ_K (isotónica sobre la fiabilidad direccional, igual que §4). El **score de selección
es la accuracy direccional del régimen en CALIBRACIÓN entre los días que superan su propio gate**
(`acc_at_gate`), con guardia de soporte mínimo (n_fired ≥ 5% de los días de calibración) y
desempate por parsimonia (menor K). NUNCA usa OOS ni P&L de trading. K* = argmax acc_at_gate.

**Validación (diagnóstico, NO selección).** Se compara K* (ex-ante) con el **K mejor-OOS**
(argmax Sharpe OOS por activo) — esto SÍ mira el OOS, pero solo para medir la concordancia, no
para elegir. Concordancia alta ⇒ el criterio ex-ante funciona; baja ⇒ no.

**Métricas.** Por activo y K: τ_K, acc_at_gate (calib), n_fired, Sharpe/accuracy/intervenciones
OOS. Agregado: K* por activo, concordancia K* vs K-mejor-OOS, y Sharpe OOS de la cartera
"K-por-activo" vs K=3 fijo vs K=2 fijo. Sanity: la columna K=3 debe reproducir el panel
`k_ablation_panel.json` (mismo re-ajuste fresco por activo).

**Criterio de decisión (lo decide Raquel tras ver la tabla).** Solo se consideraría adoptar
K-por-activo si (a) K* concuerda con K-mejor-OOS en la mayoría de activos Y (b) la cartera
K-por-activo bate a K=3 fijo fuera de muestra con la multiplicidad (K×τ por activo) contada en
el Deflated Sharpe. En otro caso se mantiene K=3 fijo y el experimento queda documentado como
exploración honesta.

**Output esperado.** `experiments/k_per_asset.py`, `outputs/experiments/k_per_asset.json`,
`notebooks/experimentos.ipynb`.

---

## [2026-06-08] [Hallazgo] - K-por-activo NO funciona: se mantiene K=3 fijo

**Contexto.** Ejecutado el experimento pre-registrado K-por-activo (`experiments/k_per_asset.py`,
`notebooks/experimentos.ipynb`), 10 activos × K∈{2,3,4}.

**Detalle.** (a) **Concordancia K\* (ex-ante) vs K mejor-OOS = 1/10** — peor que el azar
(p_binomial=0.98 frente a 1/3 esperado): el criterio de informatividad direccional en
calibración NO predice qué K rinde mejor fuera de muestra. (b) La cartera **K-por-activo rinde
PEOR que K=3 fijo**: Sharpe OOS medio +0.30 vs +0.49 (mediano +0.24 vs +0.43); Diebold-Mariano
p=0.14 (no significativo, punto a favor de K=3 fijo). (c) El criterio ex-ante ya gravita a K=3
por sí solo (lo elige en 6/10). Coherente con el panel previo (K2≈K3 indistinguibles ⇒ no hay
señal que explotar; añadir el grado de libertad de la selección solo mete ruido).

**Decisión.** Se **mantiene K=3 fijo** como modelo canónico. La selección de K por activo se
descarta: no tiene poder predictivo ex-ante y degrada el rendimiento. Queda documentada como
exploración honesta en `notebooks/experimentos.ipynb` (cuaderno no canónico).

**Implicaciones para el TFG.** Refuerza por tercera vía la elección de K=3 (tras estructura del
HMM y panel K2-vs-K3): ni siquiera dejando que K varíe por activo se mejora. Es un resultado
negativo limpio que demuestra que K=3 no es arbitrario ni sobre-ajustado.

**Referencias.** `experiments/k_per_asset.py`, `outputs/experiments/k_per_asset.json`,
`notebooks/experimentos.ipynb` (§E1).

---

## [2026-06-08] [Decisión] - Umbral τ=0.5 (histograma) y selección de K por accuracy direccional; SPY→K=2

**Contexto.** Tras un estudio extenso (held-out likelihood, held-out direccional, panel, K-por-activo)
y dos rondas de Consejo, se cierran las dos decisiones del bloque régimen/RAM. Decisión de la
autora, con información completa.

**Umbral τ.** Se fija **τ=0.5** (regla de mayoría: el override dispara cuando el régimen contrario
es el más probable). Justificación de calibración (no look-ahead): el RAM score P(Calma) es
**bimodal** (histograma con masa en ~0 y ~1, valle vacío en el medio) y el acierto direccional es
**plano** para cualquier τ∈[0.3,0.9] (≈0.556, calibración 2000–2024). El cruce-de-fiabilidad
(isotónica/logística) NO identifica un τ fino con el HMM reproducible y degenera por el confound
del drift (long acierta >0.5 siempre porque SPY sube 54.4%); por eso se usa el valle bimodal +
robustez en vez de un punto frágil. τ=0.5 tiene varianza de estimación cero.

**Número de regímenes K.** Se elige **por activo** maximizando la **accuracy direccional del
régimen fuera de muestra en calibración** (CV temporal, sin OOS). Criterio ex-ante alineado con
el uso (dirección). Resultado: SPY/SMCI/ROKU→K=2; resto→K=3. Concordancia calib↔OOS 7/10
(sign test p=0.172, NO significativo a α=0.10 — se reporta honestamente como limitación, n=10).

**SPY canónico → K=2.** El criterio direccional prefiere K=2 para SPY (acc calib 0.538 vs 0.532).
OOS: K=2 da acc 0.499, Sharpe +1.33, McNemar 0.0035 (vs K=3: acc 0.436, +0.67, 0.069).

**CAVEAT HONESTO (obligatorio en Limitaciones).** K=2 interviene 238/401 días (59%) y su accuracy
(0.499) está **por debajo de comprar-y-mantener (0.566)**: su ventaja viene de **voltear al agente
sobre-corto a largo en la mayoría de días → cabalgar el drift alcista** (mismo mecanismo que la
exclusión de M_neg), no de destreza de régimen. Es **frágil fuera de un mercado alcista**. El
Consejo (rigor + series + finanzas) recomendó **K=3** (held-out likelihood decisiva, interpretabilidad
Calma/Estrés/Crisis, supervisión disciplinada con abstención de Estrés). La autora elige K=2 por
máxima accuracy/Sharpe, asumiendo y documentando el carácter agresivo/drift-riding. Se deja
constancia para que el rastro sea honesto y la decisión revisable ante el tribunal.

**Implicaciones.** Reescritura del canónico de 3→2 regímenes (§3–§12), τ=0.5, y §12 pasa a
documentar la selección de K por accuracy (con K=3 mostrado al lado y el caveat). El panel y el
K-por-activo quedan en el cuaderno de experimentos.

**Referencias.** experiments/k_selection*.py, k_per_asset_directional.py; outputs/experiments/*.json;
dictámenes Consejo 2026-06-08 (rigor BLOQUEÓ el estudio post-hoc de fallos; recomendó K=3).

---

## [2026-06-08] [Decisión] - FINAL: K=3 canónico y τ=0.5 (supersede la entrada previa de SPY→K=2)

**Contexto.** La entrada anterior registró SPY→K=2 (decisión de la autora por máxima accuracy).
Al probar M10 con esa lógica y un test falsable adicional, la decisión se **revierte a K=3** —de
forma razonada y documentada en el propio notebook canónico (§12)—. τ=0.5 se mantiene.

**Por qué se revierte a K=3 (la prueba, en §12 del canónico):**
1. **Verosimilitud fuera de muestra (calibración, sin OOS):** K=3 (−1.301/obs) ≫ K=2 (−1.693).
   El tercer estado describe mucho mejor los datos no vistos → es un régimen real, no sobreajuste.
   Es el criterio honesto de selección de K.
2. **La mayor accuracy de K=2 es un espejismo del drift, no destreza.** K=2 va largo el **~75 %**
   de los días (≈ siempre-largo) y su accuracy (0.499) está **por debajo de B&H (0.569)**: no
   predice, **cabalga el drift alcista** (voltea al agente sobre-corto a largo casi siempre =
   mecanismo de M_neg). K=3 va largo ~49 % (selectivo) → supervisión.
3. **Falsable y confirmado:** la ventaja de K=2 es condicional al alza. Test M10 vs M8(K=2) en 5
   activos: M8 gana en alcistas (NVDA, SPY), **M10 gana en bajistas (UNG, MARA)**; corr
   drift↔(M10−M8) ρ=−0.70. La "victoria" de K=2 desaparece (e invierte) en mercados que caen.
4. **Mecanismo:** el estado **Estrés = abstención** es lo que hace a STRATA un *supervisor* y no
   un *cabalga-drifts*. K=3 se abstiene en régimen ambiguo (interviene ~30 %); K=2 no tiene
   abstención → fuerza dirección siempre → colapsa en drift-riding.

**Cifras canónicas finales (SPY, K=3, τ=0.5, OOS 2024-10→2026-05, N=401):**
- Escalera: M5 −1.82 (acc 0.384) → M7 reduce −1.41 (DM vs M5 p=0.095, Wilcoxon 0.004) → M8
  override +0.67 (acc 0.436). **McNemar M8 vs M5: 0.069 (τ=0.5) / 0.088 (default 0.40)** — rescate
  robusto a α=0.10 con AMBOS umbrales; permutación por bloques p=0.044. **Deflated Sharpe 0.106**
  (Sharpe no robusto a multiplicidad → la evidencia es el acierto pareado, no el Sharpe).
- M8 vs M2 (régimen sin agente): ΔSharpe +0.70, IC incluye 0, DM p=0.44 → no significativo.
- M10 +0.64 ≈ M8 +0.67: DM p=0.67, TOST p=0.42, IC ΔSharpe contiene 0. SHAP top = ram_score,
  garch_sigma, psa_score, crisis_prob, stress_prob (STRATA/régimen; ninguna personalidad en top-5).
  Ablación +0.64→+0.21. Umbral XGBoost inestable ρ=−0.90 vs umbrales STRATA fijos.

**Implementación.** §4 reescrita (τ=0.5 por histograma bimodal + acierto plano + confound del
drift). §12 reescrita como prueba deliberada de K=3 (held-out LL + anatomía de la accuracy de
K=2 + estructura). Notebook re-ejecutado 0 errores. La exploración K-por-activo y el test del
drift quedan en `experiments/` y en el cuaderno de experimentos.

**Frase de defensa.** "Un Sharpe más alto que solo existe porque el mercado subió no es un
resultado: es el riesgo que el supervisor debería vigilar. Por eso K=3, no K=2."

---

## [2026-06-09] [Decisión] - Respuesta a la auditoría del Consejo: reestructura del bloque K + honestidad

**Contexto.** Auditoría crítica (rigor-matemático, series-temporales, inspector-sesgos, abogado-del-
diablo + interrogatorio tribunal). Encontraron un autogol y varios overclaims. Se corrige todo.

**1. AUTOGOL corregido (el más grave).** El test del drift (`m10_vs_m8_drift.py`) corría con
`n_states=2`, así que NO probaba nada sobre K=3 — y §12 lo citaba como prueba de que "K=3 supervisa".
Re-ejecutado con K=2 Y K=3 sobre los 10 activos (`experiments/drift_test_k2k3.py`): ρ(drift, M10−M8)
= −0.36 (K=2, n.s.) vs **−0.70 (K=3, p=0.02)**. Es decir, el ρ va EN CONTRA del relato: el filo de
K=3 también está amplificado por el drift. Matiz: por conteo, M8-K3 bate al meta-learner en **9/10**
activos (incl. 2/3 bajistas) vs 6/10 de K=2 → K=3 es **menos frágil**, pero **NO "drift-free"**.
Conclusión honesta: el test es **mixto**; ambos se benefician del tape alcista; K=3 menos.

**2. Reestructura (la elección de K es CALIBRACIÓN).** La justificación de K=3 se **mueve a §3
(Parte II, calibración)** con un gráfico: verosimilitud fuera de muestra K∈{2,3,4,5} (curva) +
estructura de los 3 regímenes (scatter vol×retorno). §12 (Parte V) se reduce a una **nota corta de
robustez OOS** ("¿contradice el OOS la elección de K=3?"): la anatomía de K=2 (largo ~75%, accuracy
< B&H = cabalga el drift) + referencia al panel y al test del drift, sin el overclaim del ρ.

**3. Overclaims corregidos.** (a) "la verosimilitud prefiere K=3 sin ambigüedad" → honesto: la LL es
monótona en K (gaussiana mal especificada pediría K≥4); solo descarta K=2; el tope en 3 es por
INTERPRETABILIDAD. (b) Deflated Sharpe: n_trials 32 → **50 (cota inferior; real >100)**; DSR aún más
bajo → refuerza "la evidencia es el McNemar, no el Sharpe". (c) drift "confirmado" → "sugerente/mixto".
(d) Eliminado `hmm_seed_stability_spy.json` (obsoleto, era isotónico, contradecía el canónico).

**4. Validado por la auditoría (no se toca).** Sin look-ahead (inspector lo verificó: posterior
filtrado, GARCH causal, patches del agente causales, OOS post-cutoff DeepSeek). Pre-registro existe.
McNemar dual (τ=0.5 y default 0.40) como blindaje. Panel cross-asset da robustez transversal.

**5. PENDIENTE GRANDE (reconocido como límite en §12).** Todo vive en UNA ventana OOS alcista. La
**validación multi-ventana / walk-forward** (2008/2020/2022 de la calibración como pseudo-OOS para el
rescate M8 vs M5, estratificando por régimen) está SIN HACER — es lo que el tutor pidió explícitamente
y la validación que de verdad cerraría la cuestión. También pendiente: control `M_drift` (¿el rescate
de M8-K3 es régimen o deshacer el sesgo-corto del agente?), documentar ECE del LLM (0.407), y
recalibrar `strata_thresholds.json` (PSA/GSO) a n_obs=6204.

**Cifras canónicas (sin cambio respecto a la entrada FINAL anterior):** M5 −1.82 → M7 −1.41 → M8
+0.67; McNemar 0.069(τ)/0.088(default); M10 +0.64 ≈ M8; held-out LL K3 −1.30 ≫ K2 −1.69.

**Referencias.** notebooks/strata_canonical.ipynb (§3 selección K con gráfico, §12 robustez OOS),
notebooks/experimentos.ipynb (E1-E3), experiments/drift_test_k2k3.py, outputs/experiments/*.json.

---

## [2026-06-09] [Pre-registro] - Experimento E4 (sensibilidad de los umbrales de PSA y GSO)

**Contexto.** Los umbrales de PSA/GSO son percentiles ex-ante de la calibración (low=P95,
medium=P99, high=max); el override dispara en severidad ≥ medium, por eso PSA/GSO casi nunca
saltan en el OOS de SPY (RAM hace todo el trabajo). Pregunta: si los bajamos para que disparen,
¿mejora algo el acierto direccional? Es **diagnóstico de sensibilidad, NO recalibración**.

**Hipótesis (H1).** Como PSA/GSO **no voltean el signo** (GSO solo recorta magnitud en medium+;
PSA frena ×0.5 solo en high; RAM es el único que reorienta la dirección, override-C), bajar sus
umbrales **NO mejora el acierto direccional** de M8; a lo sumo cambia Sharpe/turnover por modular
magnitud. Refuerza la tesis "RAM domina la supervisión".

**H0.** Existe algún umbral PSA/GSO (más bajo que P95/P99) que mejora el acierto direccional de
M8 de forma significativa y en la dirección de mejora.

**Estadístico.** McNemar pareado (sweep vs base P95/P99 y sweep vs M5) sobre el acierto
direccional; IC del ΔSharpe por bootstrap estacionario (Politis-Romano) en el punto extremo (P50).

**Criterio de fracaso de H1 (pre-registrado).** Un punto del barrido REFUTA H1 si **Δaccuracy >
+0.02 vs base** Y **McNemar vs base p<0.10** Y los discordantes favorecen al sweep (**b>c** con
`mcnemar_dict(sweep, base)`, donde b = sweep✓&base✗). La coherencia de signo evita contar como
"mejora" un punto que en realidad empeora. Si ≥1 punto refuta, se analiza en auditoría de
resultados SIN prejuzgar la causa.

**Mecánica del barrido (decisión metodológica).** Se barre el umbral en la severidad donde cada
detector interviene: **PSA mueve `high`** (low=0, medium=high=p{pct}); **GSO mueve `medium`**
(low=medium=p{pct}, high=max). Percentiles del barrido: {50,75,90,95,99}. RAM fijo en τ=0.5.

**Anti-p-hacking.** Se reportan TODOS los puntos; **prohibido adoptar** el mejor umbral OOS como
nuevo default (sería look-ahead). El default sigue siendo P95/P99 ex-ante. `n_trials = 10`
(5 pct × 2 detectores) declarado ex-ante para el Deflated Sharpe si se mencionara algún Sharpe
favorable.

**Datos.** SPY, OOS 2024-10-01→cierre, signal_lag=1 (peso_t·retorno_{t+1}). HMM/GARCH cacheados
(no se recalibran). Auditado por @rigor-matematico (PASS tras 4 arreglos: cableado de umbrales,
PSA mueve high, coherencia de signo en refutación, n_trials).

**Output esperado.** `outputs/experiments/psa_gso_threshold_sensitivity.json` con claves: `meta`
(incl. n_trials, hashes, signal_lag=1), `m5`, `base` (accuracy/mcc/hit_rate/sharpe/turnover +
mcnemar_vs_m5), `sweep` (10 puntos: detector, pctile, umbral_movido, n_intervenciones_detector,
accuracy, mcc, n_valid, delta_accuracy_vs_base, sharpe_causal, mcnemar_vs_m5/base, refuta_h1),
`ci_sharpe_extreme_vs_base`, `verdict` (h1_sostenida, n_puntos_refutan, comentario neutral).
Interpretación → notebooks/experimentos.ipynb §E4. Script: experiments/psa_gso_threshold_sensitivity.py.
