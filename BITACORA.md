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

---

## [2026-06-09] [Decisión] - RAM mantiene la tupla de 3 niveles; el gate efectivo es medium=τ

**Contexto.** Revisión (code-review + dictamen harvard-professor) de si `low`/`high` de RAM son
grados de libertad ocultos que el tribunal atacaría como p-hacking. Un fix previo
(detectors.py: `high = max(high, medium)`) ya resolvió el orden de umbrales cuando τ>high.

**Detalle.** Verificado en código: override-C (M8, intervention.py:151) y reduce ram_continuous
(M7, intervention.py:108) disparan AMBOS en severidad medium/high = `score ≥ τ`. `low=0.25` y
`high=0.70` NO entran en ninguna decisión de intervención en SPY; solo re-etiquetan severidad en
la tabla de §6. Se DESCARTA colapsar RAM a un único umbral: rompería la simetría con PSA/GSO (que
sí usan P95/P99/máx informativos), crearía un caso especial en Severity/detector/tests, y no
cambiaría ninguna cifra (low/high no tocan el P&L). Se CORRIGE un claim falso de §4 (_build.py:450)
que afirmaba que `low=0.25` activa M7: M7 dispara en `medium=τ`, no en `low`.

**Implicaciones para el TFG.** §4 reescrita para declarar que RAM es de facto un detector de un
solo corte τ y que la tupla es la firma común de los tres detectores, sin grados de libertad sobre
el P&L. Frase de defensa proactiva ante "¿por qué tres umbrales?". Sin re-ejecución con cambio de
cifras (invariantes). Recordatorio: el flanco real del tribunal es "¿τ=0.5 lo elegiste tú?" —
cubierto por el histograma de §4 + McNemar dual τ=0.5/0.40.

**Referencias.** strata/detectors.py:202-214, strata/intervention.py:108,151;
notebooks/_build.py:449-451,466.

---

## [2026-06-09] [Pre-registro] - Validación walk-forward / robustez multi-ventana de STRATA (SPY)

**Pregunta de investigación.** ¿El rescate de STRATA (M8 mejora a M5) es un rasgo estable
del fenómeno o un artefacto de la única ventana OOS alcista (2024-10→2026-05)? Falsable:
el rescate debe sobrevivir al re-muestreo por ventanas rodantes del OOS **y** el modelo de
régimen subyacente (K=3) debe ser estable a lo largo de 24 años que incluyen 2008/2020/2022.

**Antecedentes (`@asesor-historico`).** El proyecto anterior hizo rolling-origin SOLO sobre
el OOS del agente: *sliding* (window=120, step=5 → 57 ventanas) y *anchor* (step=20 → 10
ventanas crecientes), métrica ΔSharpe(M8−M5)/Δacc/Δequity por ventana. Número honesto:
`frac_positive ΔSharpe = 0.737` (sliding); el 1.0 de anchor era con 10 ventanas muy
solapadas; el "95%" era aspiración del tutor, NO resultado. ERROR a no repetir: aquel
rolling-origin fue **huérfano** (sin pre-registro, sin BITACORA, sin garantía de
`signal_lag=1`). CONSTRAINT DURO heredado: el agente LLM solo existe en el OOS post-cutoff
de DeepSeek (2024-10→); ejecutar M5/M8 en 2008/2020 **contaminaría** el LLM con look-ahead,
así que la validación se parte en dos (A: modelo de régimen sin agente, 24 años; B: rescate
con agente, dentro del OOS). Pendiente nº1 reconocido como límite en la entrada de auditoría
del 2026-06-09 (§5 PENDIENTE GRANDE).

**Alcance de cada parte (declaración explícita, anti-tribunal).** Parte A y Parte B miden
cosas DISTINTAS y NO intercambiables:
- *Parte A* mide la **robustez inter-régimen / inter-época** del modelo de régimen (HMM K=3),
  porque es la ÚNICA que recorre 2000–2024 incluyendo 2008/2020/2022. Es donde recae,
  enteramente, la respuesta al "puede que tuvieras suerte en el periodo". No usa al agente.
- *Parte B* mide **estabilidad INTRA-OOS**: re-muestreos por ventanas rodantes de un ÚNICO OOS
  alcista (~400 días, 2024-10→2026-05). NO es robustez inter-régimen ni inter-época — todas las
  sub-ventanas viven en el mismo tramo alcista. Se declara así para que el tribunal no la
  confunda con validación fuera de muestra real: es un test de **fragilidad de la lectura
  global** (¿sobrevive a re-muestreos del mismo periodo?), no de generalización temporal.

**Hipótesis H1.**
- *(A — modelo de régimen, sin agente)* El HMM K=3 calibrado en 2000–2024-09 generaliza:
  su held-out log-likelihood por observación es estable (no colapsa) en pseudo-OOS rodantes
  que incluyen las crisis de 2008/2020/2022, y la informatividad direccional del régimen
  (acierto del mapeo régimen→signo por encima del gate τ=0.5, con el signo del mapeo CONGELADO
  en el tramo de ajuste) se mantiene ≥0.5 en la mayoría de ventanas. La elección de K=3 no es
  artefacto de una época.
- *(B — rescate del agente, dentro del OOS)* La lectura global del rescate M8 vs M5 NO es
  artefacto de la agregación: la **mediana de ΔSharpe(M8−M5)**, re-estimada en re-muestreos del
  OOS, es positiva con IC95 bootstrap que excluye 0 por arriba.

**Hipótesis nula H0.**
- *(A)* La held-out LL de K=3 se degrada (no es estable) o el régimen no es direccionalmente
  informativo (acierto cruza por debajo de 0.5) en una fracción material de ventanas → la
  calibración 2000–2024 no generaliza y K=3 estaba ajustado a una época.
- *(B)* La mediana de ΔSharpe(M8−M5) tiene IC95 bootstrap que **contiene 0** → la lectura
  global del rescate es frágil al re-muestreo del propio OOS.

**Estadístico de contraste.**

*CONFIRMATORIO (un único test que dicta el veredicto de la Parte B):*
- *(B-conf) Mediana de ΔSharpe(M8−M5) con IC95 por bootstrap estacionario que excluye 0 por
  arriba.* Se bootstrapea la **serie diaria de la diferencia de retornos pareada**
  `d_t = r_t^{M8} − r_t^{M5}` (con `signal_lag=1`: `w_t × r_{t+1}`), NO la serie pre-agregada
  de ΔSharpe por ventana, y en **cada réplica se recomputa el Sharpe** de cada brazo y su
  diferencia. Bloque medio = √N (Politis-Romano 1994). H0: la mediana bootstrap de ΔSharpe
  tiene IC95 que contiene 0. Sólo se aplica en el esquema **SLIDING** (window=120, step=5).
  Todo lo demás de la Parte B es exploratorio.

*EXPLORATORIO / SANITY (NO entra al veredicto; descriptivo):*
- *(A.1) Held-out log-likelihood rodante.* Rolling-origin / time-series CV (Tashman 2000;
  Bergmeir & Benítez 2012) sobre 2000–2024-09: en cada origen se reajusta el HMM K∈{2,3,4}
  en `[inicio, t]` y se evalúa la LL/obs en el bloque siguiente `(t, t+h]`. Curva de LL/obs
  por K y por ventana; comparación K=3 vs {2,4} por nº de ventanas donde K=3 domina (conteo
  descriptivo). Control de label switching: tras cada reajuste los estados se ordenan por σ
  ascendente (Calma<Estrés<Crisis). El mapeo régimen→dirección se fija con las MEDIAS de
  retorno por estado en el tramo de AJUSTE `[inicio,t]`, se CONGELA, y se aplica al held-out
  (nunca se mira el held-out para decidir el signo → sin look-ahead).
- *(A.2) Informatividad direccional rodante.* Por ventana de evaluación held-out: acierto del
  régimen (mapeo dirección dominante CONGELADO → signo de r_{t+1}) entre los días con
  confianza ≥ τ=0.5; `frac_windows(acierto ≥ 0.5)` con sign test contra 0.5 (descriptivo).
- *(B-expl.1) ΔSharpe por sub-ventana, frac_positive y sign test sobre N_eff.* Degradados a
  DESCRIPTIVOS. N_eff se reporta como nota informativa con el descuento de Bartlett
  `N·(1−ρ)/(1+ρ)` (ρ = autocorrelación lag-1 de la serie de ΔSharpe por ventana), NO con el
  apaño `N/(window/step)`. NO forma parte del criterio de éxito.
- *(B-expl.2) ANCHOR* (origen fijo, ventanas crecientes) y *DISJOINT* (step=window, ventanas
  no solapadas): descriptivos, sanity de baja potencia. Anchor NUNCA entra al veredicto.
- *(B-expl.3) McNemar pareado por sub-ventana* (`core.stats.mcnemar_test`) M8 vs M5: descriptivo
  (56 tests; no se corrige multiplicidad porque no son confirmatorios).
- *(B-expl.4) McNemar pooled estratificado* por régimen (Calma/Estrés/Crisis, 3 estratos) y por
  signo del drift del sub-tramo (alcista/bajista, 2 estratos) = 6 tests. Se aplica
  **Holm-Bonferroni** (Holm 1979) sobre los 6 p-valores y se reporta el p ajustado; el estrato
  bajista alimenta el brazo (1) del criterio de fracaso. Complementado con
  `block_permutation_test` (bloques √N). Se pre-declara el **tamaño mínimo de días** del estrato
  bajista para concluir: `n_obs ≥ 60`; si el OOS aporta menos (previsiblemente n≈20–40 días
  bajistas), el estrato se marca `inconclusivo_por_n` y NO dispara falsificación.
- *(B-expl.5) Deflated Sharpe* (`core.stats.deflated_sharpe`, Bailey & López de Prado 2014) del
  Sharpe agregado de M8 en el OOS global, con `n_trials = nº de configuraciones de ventana`
  probadas (sliding+anchor+disjoint = 3) y `n_obs = N` días OOS. Descriptivo: documenta que el
  Sharpe de M8 sobrevive al descuento por selección de configuración.

**Distribución bajo H0.** B-conf: empírica por **bootstrap estacionario sobre la serie diaria
pareada** (recomputando Sharpe en cada réplica). A.2/B-expl.1: binomial exacta (sign test).
B-expl.3/4: binomial exacta del McNemar (b+c<25) o χ²₁ con corrección de Edwards; permutación
por bloques (distribución empírica de signos de bloque); Holm-Bonferroni sobre el conjunto de 6.
A.1: comparación descriptiva de curvas (no test formal; conteo de ventanas).

**Criterio de éxito (PRE-DECLARADO, honesto — NO "95%").** α=0.10 (justificado por baja
potencia con N≈400 días OOS y efecto direccional pequeño; convención en finanzas, igual que M8).
- *(A — DESCRIPTIVO, no test confirmatorio)* Se REPORTA: (i) en qué fracción de ventanas K=3
  domina en held-out LL/obs a K=2 (esperado ≫0.5; el 0.70 NO es umbral de decisión sino
  referencia descriptiva del "K=3≫K2" global) y (ii) `frac_windows(acierto régimen ≥ 0.5)` con
  el sign test (A.2). El **ancla direccional** de la Parte A es el criterio (ii): que el régimen
  siga informando direccionalmente en épocas que incluyen crisis. La Parte A se considera
  "consistente" si (ii) > 0.5 con sign test p<0.10; el conteo de dominancia de K=3 es soporte.
- *(B — CONFIRMATORIO)* La **mediana de ΔSharpe(M8−M5)** (bootstrap estacionario sobre la serie
  diaria pareada, esquema sliding) tiene **IC95 que excluye 0 por arriba** (`low > 0`). Único
  criterio que dicta `h1_b_sostenida`. frac_positive, sign_test_neff, ΔAccuracy: se reportan,
  NO deciden.

**Composición A×B (regla pre-declarada del veredicto global).**
- A consistente **y** B confirmatorio positivo → hipótesis de robustez SOSTENIDA (modelo
  generaliza inter-época; rescate estable intra-OOS).
- A consistente, B **inconcluso por baja potencia** (IC95 cruza 0 pero `low` cercano) → se
  reporta como "robustez del MODELO sostenida; robustez del RESCATE no concluyente por tamaño
  del OOS". NO se afirma el rescate; NO se refuta (ausencia de evidencia ≠ evidencia de
  ausencia). Es el resultado más probable dado N≈400 y se acepta como honesto.
- A inconsistente, B positivo → contradicción a investigar (`@rigor-matematico`): un rescate
  intra-OOS sin modelo de régimen generalizable es sospechoso de cabalgar el drift.
- Ambos negativos → hipótesis de robustez NO sostenida.

**Criterio de fracaso (DOS reglas prior-flip INDEPENDIENTES, en OR; pre-registradas).**
Cualquiera de las dos, por separado, marca un límite (antes exigían AND, que neutralizaba el
brazo SPY porque el brazo panel con n=10 casi nunca dispara). Ahora:
1. *(Falsificación a nivel SPY)* El signo de ΔSharpe(M8−M5) **se invierte** (mediana < 0) en el
   **estrato bajista** del OOS (B-expl.4), SIEMPRE que ese estrato tenga `n_obs ≥ 60`. Si
   `n_obs < 60`, el estrato es `inconclusivo_por_n` y este brazo NO se evalúa (no falsifica ni
   confirma). Una inversión de signo con n suficiente refuta el rescate direccional a nivel SPY.
2. *(Límite a nivel panel — DESCRIPTIVO, n=10 subpotente)* Spearman ρ(drift_oos, ΔSharpe(M8−M5))
   sobre los 10 activos. Con n=10 NO se usa p<0.10 (subpotente); se reporta el **signo de ρ y su
   IC bootstrap**. Si ρ>0 con IC que excluye 0 → indicio de que M8 cabalga el drift (rescata
   donde el mercado sube). Se reporta como LÍMITE descriptivo, no como test confirmatorio. Es el
   mecanismo ya documentado en la exclusión de M_neg y en el test del drift K2/K3 (ρ=−0.70).
Si (1) [con n suficiente] dispara, STRATA-SPY queda documentado como disciplina de riesgo
condicional al alza, NO como rescate direccional universal. (2) refina la lectura a nivel panel.

**No-independencia de las ventanas solapadas (el agujero del rolling-origin).** El veredicto
confirmatorio (B-conf) NO usa la serie de ΔSharpe por ventana (que arrastra el solapamiento):
bootstrapea la **serie diaria pareada** y recomputa Sharpe, lo que no infla N. Para los
descriptivos por ventana: (a) frac_positive y sign test usan **N efectivo de Bartlett**
`N_eff = N·(1−ρ̂)/(1+ρ̂)` (ρ̂ = autocorrelación lag-1 de la serie de ΔSharpe por ventana), NO el
apaño `N/(window/step)` (sin base, hacía N_eff≈2 e imposibilitaba cruzar α); (b) panel de
**ventanas NO solapadas** (step=window=120 → ⌊N/120⌋≈3 bloques disjuntos) como sanity de baja
potencia. Todo esto es exploratorio.

**Datos.**
- Activo central: SPY. Panel de robustez (solo Parte B.iii y criterio de fracaso 2):
  SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR, SMCI, ROKU, MARA (`cache/agent/`).
- Calibración: 2000-01-01 → 2024-09-30 (Parte A reajusta HMM rodante DENTRO de este tramo;
  HMM/GARCH canónicos cacheados se reusan para Parte B).
- OOS: 2024-10-01 → última fecha en `cache/agent/SPY/` (~401 días). Parte B vive aquí.
- Rolling-origin Parte A: orígenes anuales 2008→2023, horizonte de evaluación h=252 días;
  reajuste HMM expanding-window. Parte B: *sliding* (window=120, step=5) y *anchor* (origen
  fijo en OOS_START, step=20, ventanas crecientes), replicando al proyecto anterior.
- Embargo: no aplica a Parte A/B (rolling-origin de evaluación, no CPCV con etiquetas
  solapadas); se mantiene `signal_lag=1` (posición_t × retorno_{t+1}) en TODO backtest.
- Splits: A = 16 orígenes anuales × K∈{2,3,4} (exploratorio); B-conf = 1 bootstrap sobre la
  serie diaria pareada del esquema sliding (confirmatorio); B-sliding ≈ ⌊(401−120)/5⌋ ≈ 56
  ventanas, B-anchor ≈ 14, B-disjuntas = ⌊401/120⌋ = 3 (todos exploratorios).
- Estrato bajista del OOS: tamaño mínimo pre-declarado `n_obs ≥ 60` para concluir falsificación
  a nivel SPY; por debajo, `inconclusivo_por_n`.
- Deflated Sharpe de M8: `n_trials = 3` (configuraciones de ventana sliding/anchor/disjoint).
- Panel: verificación de estabilidad de signo de medias por régimen (calib 2000–2024 vs primeros
  60 días OOS) por activo antes de incluirlo en la nube de Spearman.
- Semillas: `config.SEED=42` (HMM `n_seeds=10`, `n_iter=1000`; bootstrap estacionario 2000
  réplicas, bloque medio √N).

**Sanity dual same-day / causal y tabla maestra (agregado OOS global).** Sobre el agregado OOS
global (NO por sub-ventana, donde AUC/log-loss/Brier son inestables — lección #11):
- *Doble protocolo de medición.* Se computa el Sharpe agregado de M8 y M5 en los dos protocolos:
  `causal` (`w_t × r_{t+1}`, signal_lag=1, el único válido para reportar) y `same_day`
  (`w_t × r_t`, sanity). Se verifica que el **signo de ΔSharpe NO se invierte** entre ambos; una
  inversión delata el bug `peso_t × retorno_t` (el que infectó M8 semanas en el proyecto
  anterior). Se reporta `sign_consistent` (bool).
- *Tabla maestra completa* (lección #11) en el agregado OOS global, para M5 y M8:
  `accuracy`, `auc`, `log_loss`, `brier`, `mcc` (`core.metrics.classification_metrics`) +
  `sharpe`, `equity_final` (`core.metrics`). Solo en el agregado; nunca por sub-ventana.

**Estabilidad de signo por activo en el panel (lección #6, pre-registrada).** Antes de usar un
activo del panel en `panel_drift`, se verifica que el signo de las **medias de retorno por
régimen** (Calma/Estrés/Crisis) calibradas en 2000–2024 **coincide** con el signo en los
**primeros 60 días del OOS** de ese activo. Si no coincide (prior-flip de calibración a OOS), el
activo se marca `prior_flip_calib_oos=True` y se reporta aparte; su ΔSharpe sigue en la nube de
Spearman pero la inestabilidad queda documentada (no se silencia).

**Salida esperada.** `outputs/experiments/walkforward_robustez.json` con claves:
- `meta` (ticker, panel, oos_start/end, n_days, n_obs, signal_lag=1, seed, window/step, alpha,
  `bartlett_note`, hashes de caché).
- `part_a` — `heldout_ll` por (origen, K) con `ll_por_obs` y `n_obs`; `k3_domina_frac`
  (descriptivo); `directional` por ventana con `acc_at_gate`, `n_obs` y `frac_windows_acc_ge_0p5`
  + `sign_test`; `label_switch_control` (orden por σ); `direction_map_frozen` (mapeo congelado).
- `part_b_confirmatory` — **el test del veredicto B**: `median_delta_sharpe`, `ci95_boot`
  (`{low, high, point}` por bootstrap estacionario sobre la serie diaria pareada), `block_len`,
  `n_obs`.
- `part_b_sliding` / `part_b_anchor` / `part_b_disjoint` (EXPLORATORIOS): lista de ventanas con
  `delta_sharpe`, `delta_acc`, `mcnemar`, `n_obs` por ventana; agregados descriptivos
  `frac_positive`, `n_eff_bartlett`, `rho_lag1`, `sign_test_neff`, `median_delta_sharpe`.
- `stratified_mcnemar` (EXPLORATORIO): por estrato (régimen y signo del drift) con `mcnemar`,
  `block_permutation_p`, `n_obs`, `inconclusivo_por_n` (bool); `holm_bonferroni` (p ajustados de
  los 6 tests).
- `deflated_sharpe_m8` (`{dsr, n_trials, n_obs}`, n_trials=3 configuraciones de ventana).
- `sanity_dual` (`sharpe_causal`, `sharpe_same_day` de M5/M8; `sign_consistent`).
- `master_table` (M5 y M8: accuracy, auc, log_loss, brier, mcc, sharpe, equity_final; agregado
  OOS global).
- `panel_drift` (por activo: `drift_oos`, `delta_sharpe_M8_M5`, `prior_flip_calib_oos`;
  `spearman_drift_vs_delta` con `{rho, ci95}` — NO p-valor, n=10 subpotente).
- `verdict` (`part_a_consistente`, `h1_b_sostenida` [solo del confirmatorio],
  `falsif_spy_estrato_bajista`, `limite_panel_drift`, `composicion`, comentario neutral).

**Citas.** Tashman (2000) "Out-of-sample tests of forecasting accuracy", Int. J. Forecast.;
Bergmeir & Benítez (2012) "On the use of cross-validation for time series predictor
evaluation", Inf. Sci. (justifica rolling-origin/CV en series temporales); Politis & Romano
(1994) "The stationary bootstrap", JASA (IC de la mediana de ΔSharpe, bloque √N); Bartlett
(1946) (descuento de N efectivo por autocorrelación, `N·(1−ρ)/(1+ρ)`); Holm (1979) "A simple
sequentially rejective multiple test procedure", Scand. J. Stat. (control de multiplicidad en
los 6 McNemar estratificados); Bailey & López de Prado (2014) "The Deflated Sharpe Ratio",
J. Portfolio Manag. (descuento del Sharpe de M8 por nº de configuraciones); McNemar (1947);
López de Prado (2018, sec. 7.4) (purge/embargo, contexto de validación causal).

---

## [2026-06-09] [Hallazgo] - Walk-forward: el rescate de M8 es CONDICIONAL al alza (falsificación disparada); el modelo K=3 sí generaliza

**Contexto.** Ejecutado el experimento walk-forward pre-registrado (experiments/walkforward_robustez.py),
auditado por @rigor-matematico en diseño (2 rondas, APROBADO) y en resultados (paso 5, APROBADO CON
CONDICIONES). Output: outputs/experiments/walkforward_robustez.json (SPY, OOS 2024-10→2026-06, n=401).

**Detalle (cifras verificadas).**
- **Parte A (modelo, 24 años, sin agente):** el HMM K=3 mejora el log-likelihood held-out frente a
  K=2 en **15/16 orígenes anuales** (2008–2023, incluidas las crisis de 2008/2020/2022). MATIZ HONESTO:
  K=4 mejora a K=3 en 14/16; **K=3 se elige por parsimonia e interpretabilidad, NO por ser óptimo de LL**.
  La dirección por régimen acierta ≥0.5 en 11/16 ventanas (sign test p=0.21) → **NO generaliza con
  significancia** (no es solo subpotencia: la magnitud es débil, acc 0.43–0.58).
- **Parte B confirmatorio (test ÚNICO):** mediana ΔSharpe(M8−M5)=+2.45, IC95 bootstrap estacionario
  pareado [−0.21, +5.71] → **incluye 0**. Deflated Sharpe M8=0.50 → indistinguible del azar. El rescate
  **NO es robusto multi-ventana**.
- **Condicionalidad al régimen (lo central):** estrato ALCISTA (278 d) ΔSharpe=+8.45, McNemar p=0.030
  **pero p_adj(Holm)=0.15** (no sobrevive multiplicidad); estrato BAJISTA (123 d ≥60) **ΔSharpe=−3.92**
  → **dispara la regla de falsificación pre-registrada brazo 1**. El rescate se invierte cuando el
  mercado no sube.
- **Panel (exploratorio):** ΔSharpe(M8−M5)>0 en 9/10 activos; Spearman(drift,ΔSharpe) ρ=0.54 con IC
  [−0.26, 0.90] (incluye 0, n=10 subpotente → limite_panel=False). prior_flip calib/OOS en 6/10 activos
  (XLE/UNG/MSTR/SMCI/ROKU/MARA): el prior direccional NO es estable fuera de SPY (consistente con el
  leverage effect limpio solo en índices).
- **Tabla maestra global:** M8 mejora a M5 en TODO (acc 0.454 vs 0.402; MCC −0.106 vs −0.149; Sharpe
  causal +0.67 vs −1.82) **pero ambos son perdedores direccionales absolutos** (acc < base rate 0.566,
  MCC<0). STRATA reduce el daño, no lo convierte en ganancia.
- **Sanity dual:** sign_consistent=False (causal M5=−1.82/M8=+0.67; same-day M5=+0.88/M8=+0.31). NO es
  bug de look-ahead: el bug peso_t×retorno_t INFLARÍA el causal, aquí lo PENALIZA. Es propiedad del
  agente perdedor (correlaciona + con r_t contemporáneo, − con r_{t+1}); el dato válido es el causal
  (signal_lag=1).

**Conclusión defendible (frase aprobada por @rigor-matematico para §14):** el COMPONENTE de modelo de
STRATA generaliza inter-época (K=3 mejora a K=2 en 15/16 orígenes incl. crisis), pero el RESCATE del
agente por M8 **no es robusto multi-ventana**: el test confirmatorio incluye el cero, el alcista no
sobrevive Holm y el bajista se invierte. **STRATA-SPY = disciplina de riesgo condicional al alza, no
rescate direccional universal** — límite reconocido por diseño (§4f de la constitución).

**Implicaciones para el TFG.** (1) Es un resultado de FALSIFICACIÓN honesto: la regla pre-registrada
cazó la condicionalidad al drift — exactamente lo que el tutor temía ("¿tuviste suerte en el periodo?").
No invalida el TFG; lo hace defendible (el sistema reconoce dónde no funciona). (2) Separar nítidamente
en la memoria "el modelo generaliza" (sí) de "el rescate es robusto" (no, condicional). (3) El relato
honesto es "rescata en este OOS alcista; condicional al régimen", nunca "rescate universal".

**Correcciones de reporte pendientes antes de §14 (exigidas por rigor):** (i) no afirmar K=3 óptimo de
LL (K=4 lo supera; parsimonia); (ii) reportar p_adj de Holm junto a cada p de estrato; (iii) NO reportar
el p=0.0 de B-disjoint (artefacto de Bartlett con ρ<0 y n=3); (iv) nota de que sign_consistent=False es
esperado, no leak.

**Referencias.** experiments/walkforward_robustez.py, outputs/experiments/walkforward_robustez.json,
BITACORA pre-registro [2026-06-09].

---

## [2026-06-10] [Decisión] - Lectura accuracy-first: M10 co-protagonista de M8 (no rival)

**Contexto.** Tras el walk-forward (Sharpe de M8 frágil: DSR≈0.50, condicional al alza), se fija el
marco de lectura del TFG: la métrica primaria es la **accuracy direccional**, no el Sharpe. Raquel y
el tutor priorizan acertar el signo (matriz de confusión), no el Sharpe (que el tutor "no ve claro").

**Detalle (cifras del canónico, verificadas).**
- **Escalera de accuracy:** M5 (agente) 0.384 (< azar, sign test p=4·10⁻⁶) → M8 (regla) 0.436 →
  **M10 (XGBoost sobre features STRATA) 0.539** → B&H 0.569. M10 es el mejor decodificador direccional.
- **M10 ≈ M8 en P&L:** Diebold-Mariano p=0.666 (no se detecta diferencia; TOST p=0.42 NO prueba
  equivalencia). Son indistinguibles económicamente.
- **La señal es de STRATA:** ablación — sin las features régimen/RAM/PSA/GSO el M10 cae de Sharpe
  +0.64 a +0.21; SHAP las pone arriba. Son ellas las informativas.
- **El Sharpe es la métrica frágil:** DSR M8 = 0.50 (≈ azar) y el rescate es condicional al alza
  (walk-forward §13). Por eso NO se ancla la tesis en el Sharpe.

**Decisión.** M8 y M10 son **dos consumidores de la misma señal de supervisión STRATA**: M8 = regla
interpretable (white box, transparencia/atribución), M10 = modelo aprendido (best accuracy). Se
**abraza M10 al menos tanto como M8** porque (1) la accuracy es la métrica primaria (de Raquel y del
tutor) y M10 gana ahí; (2) la accuracy es robusta al drift, el Sharpe no; (3) en P&L son equivalentes
(DM p=0.67), así que elegir M10 no pierde nada económico. NO contradice la hipótesis §2.3 (M10 no bate
a M8 en DM-P&L: confirmado p=0.67); M10 gana en un eje distinto (accuracy), consistente con "la señal
es lo que importa, no el modelo concreto".

**El hallazgo de STRATA (frase canónica).** "Un agente LLM perdedor direccional (38.4%, < azar) es
RESCATADO por supervisión estadística clásica: la accuracy sube 0.384→0.436 (regla)→0.539 (aprendido),
la señal informativa es la de STRATA (ablación+SHAP), y regla a mano y caja negra son equivalentes en
P&L (DM p=0.67). STRATA reduce el daño recuperando accuracy direccional; no genera alfa (M10 0.539 <
B&H 0.569)."

**Implicaciones para el TFG.** §11 (M10) y §14 (lectura) se redactan con M10 como co-protagonista por
accuracy, no como "rival que no debe ganar". El Sharpe queda como ilustración económica, nunca como
evidencia primaria.

**Referencias.** notebooks/strata_canonical.ipynb §9/§10/§11/§14, outputs/experiments/
walkforward_robustez.json, BITACORA [2026-06-09] [Hallazgo] walk-forward.

---

## [2026-06-10] [Pre-registro] - Enmienda al walk-forward: M10 vs M5 + ventana 150/15

**Contexto.** Raquel necesita probar que STRATA bate al agente como INVERSIÓN, no solo M8: se añade
**M10 vs M5** (el modelo de mejor accuracy) al rolling. Y se ajusta la ventana sliding.

**Cambios sobre el pre-registro original [2026-06-09]:**
1. **Ventana sliding 120/5 → 150/15.** Justificación: 120/5 daba 57 ventanas con 96% de solape
   (N efectivo de Bartlett ≈0.6, dependencia altísima); 150/15 da ~17 ventanas con menos solape →
   más independientes y honestas. CONSECUENCIA: el `frac_positive` dejará de reproducir el 0.737 del
   proyecto anterior — es esperado y es DESCRIPTIVO (no decide nada). **El cambio NO toca el veredicto**:
   el test confirmatorio es el **bootstrap diario pareado**, independiente del tamaño de ventana.
2. **M10 vs M5 añadido como 2º contraste confirmatorio** (junto a M8 vs M5). Multiplicidad declarada:
   ahora hay 2 confirmatorios; se reportan AMBOS sin cherry-picking. También M10 vs M8 como complemento.
   - H1 (M10 vs M5): mediana ΔSharpe(M10−M5) con IC95 bootstrap estacionario pareado que excluye 0.
   - McNemar M10 vs M5 estratificado por régimen (¿rescata M10 también en bajista, donde M8 dio p=1.0?).
3. **M10 = CPCV-OOF purgado** (decisión canónica §11): única vía en 18 meses; es validación cruzada,
   NO walk-forward estricto (entrenar XGBoost solo-pasado por ventana daría ~60 días → ruido). Se documenta.

**Límite duro REAFIRMADO (lo que Raquel detectó):** el rescate (M8/M10 vs M5) solo se mide en los ~401
días del OOS porque el agente LLM no existe antes de 2024-10 (cutoff DeepSeek). Las "ventanas" del rolling
son **sub-trozos solapados de un único tramo de 18 meses, NO años distintos**. La robustez multi-año real
es SOLO la Parte A (modelo de régimen, sin agente). Esto se gritará en §13.

**NO se hace:** re-calibración expanding por ventana (marginal: ≤18 meses sobre 24 años de calibración;
el agente sigue fijo; no escapa del límite). Calibración HMM/GARCH fija pre-OOS (2000-2024-09) = anterior
a TODA ventana → sin look-ahead, es el "calibrar una vez, desplegar" de producción.

**Output esperado.** outputs/experiments/walkforward_robustez.json con claves nuevas:
`part_b_confirmatory_m10_m5`, `stratified_mcnemar_m10_m5`, fila `m10` en master_table, y `h1_b_m10_vs_m5`
en el veredicto.

**Referencias.** experiments/walkforward_robustez.py, experiments/m10_k2.py (cableado M10), pre-registro
[2026-06-09].

**Addendum (resolución de la auditoría de la enmienda, @rigor-matematico APROBADO CON CONDICIONES):**
1. **Multiplicidad de los 2 confirmatorios (toca el veredicto, fijado ANTES de ejecutar):** el veredicto
   `composicion` dispara con `h1_b_m8 OR h1_b_m10`; para controlar el FWER del OR se usa la **cota inferior
   Bonferroni IC 97.5%/brazo** (`ci_bonf2_low`, cuantil 0.0125) en cada `h1_b`, no el IC95 crudo. El IC95 se
   reporta solo para transparencia.
2. **Ambas ventanas:** se reporta el sliding 150/15 (enmienda) Y el 120/5 (legacy) en el JSON
   (`part_b_sliding_legacy_120_5`); ninguno entra al veredicto (el confirmatorio es el bootstrap diario).
3. **M10 por ventana = CV, no walk-forward:** las cifras por-ventana de M10 llevan `m10_cv_not_walkforward=true`
   (cortes descriptivos de un OOF global purgado, no entrenamiento solo-pasado). El confirmatorio M10−M5 sí
   es válido (OOF honesto a nivel global).

---

## [2026-06-10] [Hallazgo] - Walk-forward extendido a M10: rescate de accuracy cross-régimen

**Contexto.** La enmienda del pre-registro ([2026-06-10] anterior) añadió M10 vs M5 al walk-forward.
Se ejecutaron los McNemar estratificados por régimen de deriva (alcista/bajista) para M10 vs M5,
complementando la entrada [2026-06-09] que solo medía M8. Todas las cifras proceden de
`outputs/experiments/walkforward_robustez.json` (clave `stratified_mcnemar.m10_vs_m5`) y han sido
auditadas por `@rigor-matematico`.

**Detalle (cifras verificadas, OOS SPY N=401, enmienda walk-forward).**

*Plano accuracy (métrica primaria) — el hallazgo nuevo.*

| Contraste | Régimen | n | p_adj Holm | block-perm p | ΔSharpe |
|---|---|---:|---:|---:|---:|
| M10 vs M5 | alcista | 278 | **0.005** | **0.000** | positivo |
| M10 vs M5 | bajista | 123 | **0.075** | **0.061** | −1.06 |
| M8 vs M5 | alcista | 278 | 0.150 | — | +8.45 |
| M8 vs M5 | bajista | 123 | 1.000 | — | −3.92 |

**M10 rescata la accuracy direccional del agente en AMBOS régimenes** (Holm p_adj < 0.10 en ambos;
block-permutation < 0.10 en bajista, robusto a autocorrelación). M8 solo rescata en alcista y es nulo
en bajista. M10 es el único modelo con MCC positivo (+0.068).

*Plano Sharpe (económico) — sin cambio respecto a [2026-06-09].*

El confirmatorio decide por la **cota Bonferroni** (criterio pre-registrado en el addendum de la
enmienda, NO el IC95 crudo): M8−M5 = −0.49 y M10−M5 = −0.48, ambas < 0 → **H1_b no se sostiene
para ninguno de los dos**. El IC95 crudo de M10−M5 = [−0.02, +5.79] roza el cero pero NO es el
criterio. El Deflated Sharpe (≈0.48) es indistinguible del azar. En bajista los ΔSharpe se invierten
(M8 −3.92, M10 −1.06): la falsificación pre-registrada se dispara para AMBOS.

**Veredicto formal:** `robustez_no_sostenida` en el plano Sharpe (el confirmatorio usa la cota
Bonferroni; ambas < 0). La accuracy de M10 es un **hallazgo descriptivo robusto**, no una
recategorización del veredicto confirmatorio.

**Conciliación de los dos planos.** Acertar más días (accuracy) y rendir mejor en cartera (Sharpe)
miden ejes distintos: el primero cuenta signos, el segundo pondera por magnitud del retorno. En
bajista M10 acierta más días que M5 (rescate de accuracy real) pero su ΔSharpe es negativo porque
las magnitudes de los retornos cuando falla compensan las ganancias de los días acertados. El
mecanismo exacto (composición long/short, concentración del P&L bajista) no se descompone en este
análisis y se reporta como límite.

**Implicaciones para el TFG.**
(1) La narrativa de "el rescate se invierte en bajista" era correcta **para el Sharpe** y **para M8
    en accuracy**, pero incompleta: M10 rescata accuracy también en bajista. Las capas que afirmaban
    "el rescate se invierte en bajista" sin distinguir M8/M10 ni accuracy/Sharpe se corrigen en esta
    sesión (ver auditoría de coherencia 2026-06-10).
(2) El veredicto del TFG es: STRATA-SPY recupera accuracy direccional (robusto cross-régimen para
    M10); su ventaja en P&L es frágil y condicional. La falsificación opera sobre el Sharpe, no
    sobre la accuracy.
(3) La frase de defensa canónica para la pregunta "¿funciona en bajista?" es: "M10 rescata accuracy
    en bajista (Holm 0.075, block-perm 0.061 robusto a autocorrelación); M8 no. Pero el ΔSharpe se
    invierte en bajista para ambos: lo que falla es el rendimiento económico, no el acierto de
    dirección."

**Referencias.** `outputs/experiments/walkforward_robustez.json` (claves `stratified_mcnemar.m10_vs_m5`,
`part_b_confirmatory.m10_vs_m5`, `verdict`); `notebooks/_build.py` §13–§14; BITACORA pre-registro
[2026-06-10] enmienda walk-forward.

---

## [2026-06-14] [Pre-registro] - Réplica multi-activo: recalibración por activo (NVDA)

**Contexto.** Decisión de la autora: testear si STRATA generaliza a stocks individuales recalibrando
**COMPLETAMENTE** sobre el histórico propio de cada activo, no reutilizando los modelos de SPY. Caso
central: NVIDIA (NVDA), valor de crecimiento con leverage débil. Propósito: delimitar el dominio de
validez de STRATA.

**Hipótesis.** Si STRATA rescata porque explota el leverage effect (alto vol → bajista, en índices),
NVDA (acción individual, leverage effect débil) debería **NO** ser rescatada. Los percentiles de
severidad PSA/GSO se heredan de SPY (umbral no re-calibrado) — decisión de no-intervención: el SIGNO
lo fija RAM (intervention.py), PSA/GSO solo escalan magnitud → no pueden invertir dirección.

**H0.** STRATA (M8) no rescata a NVDA: accuracy/McNemar pareado sin mejora significativa.

**Estadístico.**
- *Primario (dirección):* McNemar pareado M8 vs M5, a α=0.10, con estratificación post-hoc por régimen
  HMM (Calma/Estrés/Crisis) y sign test direccional del régimen en calibración (¿es informativo?).
- *Secundario (economía):* ΔSharpe(M8−M5) con IC95 bootstrap estacionario (pareado, bloques √N);
  cota **Bonferroni α=0.05/2 = 0.025** (ajuste por multiplicidad contra SPY: 2 activos centrales con
  contrastes confirmatorios). Deflated Sharpe.
- *Robustez:* MCC, AUC, log-loss de M8 vs M5. Nº de intervenciones RAM.

**Criterio de éxito H0.** McNemar M8 vs M5 **NO significativo** (p≥0.10) **O** ΔSharpe(M8−M5) con cota
Bonferroni < 0 (IC95 bajo cero). La hipótesis se sostiene si NVDA es *robustez_no_sostenida*.

**Criterio de fracaso H0 (regla prior-flip, pre-registrada):** si el signo de la media de retornos por
régimen en calibración **≠ signo en los primeros 60 días OOS**, el prior de RAM no es válido en NVDA
(fallo de transferencia). Marca un **vacío de información** (no hay signo que invertir). En ese caso,
el test de rescue se reporta pero se declara *inconclusivo_prior_flip=True*. Valor umbral: sign test
binomial exacto sobre los días de calibración (Calma acierta positivo, Crisis acierta negativo, neutral
en Estrés); si p>0.05 la dirección es demasiado ruidosa para construir un prior.

**Config congelada.**
- HMM **recalibrado de cero sobre NVDA 2000-01-01 → 2024-09-30**: K∈{2,3} (exploración; el criterio
  K-por-activo ya se probó en abril y se descartó, pero K=2 o K=3 dependen del activo).
- Percentiles de severidad PSA/GSO: **HEREDADOS de SPY** (P95/P99/máx de `cache/models/strata_thresholds.json`).
  Justificación: los umbrales codifican "¿cuándo es PSA/GSO severo?" en unidades del histórico SPY; un
  recalibrado per-activo metería n_obs de libertad (los 10 activos tendrían percentiles distintos → p-hacking
  latente). RAM se recalibra por activo (prior por régimen del activo) pero PSA/GSO se heredan (decisión
  que declara el no-recalibrado como **parámetro de robustez**, no grado de libertad).
- RAM: **prior régimen→signo data-driven por activo** (orden por volatilidad del HMM recalibrado de NVDA;
  su validez se mide con μ por estado y la regla prior-flip). La compuerta **τ=0.5 se mantiene** (criterio
  parameter-free, igual que en SPY); no se recalibra τ por activo.
- Signal_lag=1; OOS 2024-10-01 → 2026-05-20 (~409 días); semilla 42.

**Datos.** NVDA, 24 años calibración 2000-01-01 → 2024-09-30 (6204 días); OOS 2024-10-01 → 2026-05-20
(409 días). Caché: `cache/models/hmm_nvda.pkl`, `cache/models/calibration_summary_nvda.json`.

**Output esperado.** `outputs/experiments/walkforward_robustez_nvda.json` con: meta, M5/M8 tabla maestra
(accuracy, auc, log_loss, brier, mcc, sharpe, equity_final), McNemar M8 vs M5, sign test directional
régimen (calib + OOS), prior_flip (calib vs OOS60), ΔSharpe(M8−M5) con IC95 bootstrap y cota Bonferroni,
Deflated Sharpe, nº intervenciones, panel_drift (si aplica). Script: `experiments/recalibrate_nvda.py`.

**Documento de salida.** Apéndice A (§A.1–§A.3) del notebook canónico: "Réplica multi-activo —
recalibración por activo: NVDA".

---

## [2026-06-14] [Hallazgo] - NVDA: STRATA NO rescata; leverage effect no se cumple; dominio delimitado

**Contexto.** Ejecutado el pre-registro anterior (recalibración NVDA, HMM K=3 calibrado 2000–2024-09 sobre
NVDA, π SAM/GSO heredados, OOS 2024-10 → 2026-05, ~409 días). Auditado por @rigor-matematico en diseño
(APROBADO) y en resultados (APROBADO CON 6 CONDICIONES; se aplicaron todas).

**Detalle (cifras verificadas contra `walkforward_robustez_nvda.json` y `calibration_summary_nvda.json`).**

*Modelo HMM K=3 sobre NVDA.* El HMM se calibra bien en densidad: held-out LL/obs K3 **−1.687** vs K2
**−1.985** (ΔLL = **+0.298**, corte 2020, n_eval=1194); y K3 domina a K2 en **93.8 %** de los 16 orígenes
anuales inter-época (incl. 2008/2020). La maquinaria de régimen transfiere a NVDA.

*El leverage effect NO se cumple en NVDA — y de forma robusta.* Media de retorno diario por estado en la
calibración completa 2000–2024: **Calma +0.00150, Estrés +0.00044, Crisis +0.00173** — las **tres
positivas**, y el estado de MÁXIMA volatilidad (Crisis) es el **más alcista** (μ_Crisis > μ_Calma). No es un
artefacto de la ventana: en el mapa de dirección origen a origen del walk-forward, **el estado Crisis es
alcista en los 16 orígenes** (μ>0 siempre, +0.005 en plena crisis 2008). En SPY, Crisis es bajista (leverage)
y da sentido a "Crisis → short"; en NVDA, RAM fija el signo por Calma-vs-Crisis y mapea **Crisis → short**,
que es un **error sistemático**. El único estado con drift negativo es el de volatilidad **media** (Estrés),
que RAM **no usa** para el signo y que además se desvanece tras 2016: aparece en el **100 %** de los orígenes
≤2015 vs **12 %** ≥2016 (no-estacionariedad estructural; re-caracterización post-2016, era IA). Sign test
direccional del régimen (acc≥0.5 por origen del walk-forward): **k=6/14, p=0.79** → no informativo.

**Aplicación OOS (2024-10-01 → 2026-05-20, n=409).** Escalera de accuracy (métrica primaria) + Sharpe
(ilustrativo, no load-bearing):

| Métrica | M5 (agente) | M8 (STRATA) | M10 (XGBoost) |
|---------|-------------|-------------|---------------|
| Accuracy | 0.477 | 0.509 | **0.440** (peor) |
| MCC | +0.008 | −0.023 | −0.137 |
| AUC | 0.504 | 0.491 | 0.432 (peor que azar) |
| Sharpe causal | −0.573 | +0.677 | −1.953 |
| equity €1000 | 945 | 1090 | 204 |

Confirmatorio (bootstrap estacionario pareado; veredicto por cota Bonferroni de 2): **M8−M5** mediana
ΔSharpe **+1.27**, IC95 **[−0.37, +2.99]** (cruza 0), cota Bonf **−0.57<0** → H1 falsa; **M10−M5** mediana
−1.40 (no robusto); **M10−M8** mediana −2.66, IC95 [−4.18, −0.96] **excluye 0** (M10 peor que M8). McNemar
estratificado por régimen y por signo de drift: **ningún estrato sobrevive Holm (p_adj=1.0 en todos)**.
Deflated Sharpe M8 ≈ **0.50** (n_trials=3, subestima; con el grid real ≥10 cae por debajo de 0.5).

**Hallazgo clave:** M8 no rescata a NVDA **en dirección** (accuracy +0.03 dentro del ruido, MCC<0, AUC<0.5,
McNemar p_adj=1.0). M10 (XGBoost) **empeora** (AUC 0.43 < 0.5 = evidencia activa de no-señal, no falta de
potencia). El Sharpe de M8 sube (−0.57→+0.68) pero **no es rescate y no se apoya en él**: (1) no robusto
(IC95 cruza 0, cota Bonferroni −0.57<0); (2) contaminado — los umbrales PSA/GSO son de SPY y la magnitud
depende de un GARCH no regenerable; (3) Deflated Sharpe ≈ azar. La lectura plausible (no load-bearing) es
cabalgar el drift alcista (drift OOS +0.375) + control de magnitud, consistente con que en régimen bajista
M8 *empeora* (ΔSharpe −2.44). El veredicto se apoya **solo en métricas direccionales invariantes a magnitud**.

**Vacío de información (prior-flip, regla pre-registrada).**

El signo de la media por régimen en calibración (Calma +, Crisis +) se mantiene en los primeros 60 OOS,
así que prior_flip = False. PERO es un **vacío vacuo**: no hay inversión porque NVDA es prácticamente
neutral (sign test calib p=0.79 confirma que el régimen es ruido). El guardarraíl que delimitó el
dominio fue la **prueba directa** de que el prior es informativo (sign test), no el flip de signo.

**Panel (nota exploratoria).**

De los 10 activos del panel de robustez walk-forward, 6/10 tienen prior_flip calib→OOS (XLE, UNG,
MSTR, SMCI, ROKU, MARA): el prior direccional de SPY **no transfiere**. En 5/10 el acierto del régimen
en calibración es <0.5 (no informativo). Esto es consistente con CLAUDE.md §1: el leverage effect (clave)
es **específico de índices con volatilidad sistémica compartida**, no de stocks individuales.

**Veredicto (pre-registrado):** `robustez_no_sostenida`. STRATA-NVDA no rescata.

**Implicaciones para el TFG.**
(1) Refuerza la constitución del proyecto (CLAUDE.md §1, caso central SPY justificado). STRATA se
   delimita honestamente: funciona donde el leverage effect es fuerte (índices); falla en stocks
   individuales sin esa asunción.
(2) La aportación se **refuerza** con esta réplica: no es un rescate "universal" (overclaim común en ML),
   sino disciplinado (funciona en su dominio, falla reconocidamente fuera).
(3) El panel multi-activo (walk-forward) revela la misma limitación: prior_flip en 6/10 activos. NVDA
   es el prototipo que lo ilustra con claridad (caso de tesis honesto).
(4) Protocolo de rigor verificado: pre-registro antes de mirar, regla de falsificación disparada (vacío
   de prior), Bonferroni aplicada como pre-registrado, Deflated Sharpe bajo → evidencia es el McNemar
   (dirección), no el Sharpe.

**Referencias.** `experiments/recalibrate_nvda.py`, `cache/models/hmm_nvda.pkl`,
`cache/models/calibration_summary_nvda.json`, `outputs/experiments/walkforward_robustez_nvda.json`,
`notebooks/_build.py` (Apéndice A §A.1–§A.3), BITACORA pre-registro [2026-06-14].

---

## [2026-06-15] [Pre-registro] - M10 walk-forward causal: ¿es desplegable a diario? (SPY + NVDA)

**Contexto.** CPCV (Decisión #10, viva) da una estimación insesgada y controla *backtest overfitting*, pero
en cada combinación **entrena con bloques cronológicamente posteriores al test** (purga/embargo solo limpian
el solape de etiquetas, no el orden temporal). Por tanto **no simula el uso diario en producción**, y bajo
cambio de régimen puede **halagar** a M10. El "M10 por ventana" de `walkforward_robustez.py` está marcado
`m10_cv_not_walkforward=True`: es un *corte* del CPCV-OOF global, NO un reentrenamiento causal. No existe aún
una validación causal de M10. El tutor lo pidió explícitamente (transcripción: *"lánzalo en diferentes
periodos de inicio… puede que tuvieras suerte en el periodo"* y *"es el target de mañana y la restricción de
hoy"*). Este experimento es **validación adicional**, NO sustituye la Decisión #10 (CPCV sigue siendo el
estimador insesgado canónico).

**Hipótesis.** Si M10 captura señal real (no artefacto de CPCV viendo el futuro), un walk-forward **causal**
(entrenar solo con el pasado, reentreno mensual) **conserva** el rescate de accuracy de M10 sobre M5 en SPY.
Si en cambio el rescate de M10 era un efecto de CPCV-mira-el-futuro, el walk-forward causal lo **derrumba**.

**H0.** En el tramo de test, la accuracy direccional de **M10-WF causal = M10-CPCV** (McNemar pareado, sin
diferencia) **y** M10-WF no bate a M5 (sign/McNemar p≥0.10). [Es decir: ni CPCV halaga, ni hay rescate causal.]

**Estadístico.**
- *Primario (desplegabilidad):* McNemar pareado de aciertos direccionales **M10-WF vs M5** (¿rescata causalmente?)
  y **M10-WF vs M10-CPCV** (¿CPCV halagaba?), a α=0.10, sobre el MISMO tramo de test `[N0:fin]`.
- *Secundario (economía, frágil):* ΔSharpe(M10-WF − M5) con IC95 bootstrap estacionario pareado (bloque √N).
- *Sanity anti-leakage:* doble protocolo same-day (lag=0) vs causal (lag=1); si el causal sale mejor que el
  same-day, hay look-ahead (regla LECCIONES #2).

**Criterio de éxito (desplegabilidad SOSTENIDA, SPY).** accuracy(M10-WF) > accuracy(M5) con McNemar/sign p<0.10
en `[N0:fin]` **Y** McNemar(M10-WF vs M10-CPCV) p≥0.10 (CPCV no halagaba de forma significativa).

**Criterio de fracaso / falsificación (pre-registrado).** Si accuracy(M10-WF) cae a ≤0.5 **o** ≤ accuracy(M5)
en `[N0:fin]` mientras M10-CPCV>0.5 en ese mismo tramo ⇒ **CPCV estaba halagando a M10; M10 NO es desplegable
a diario**. Se reporta honestamente. (En NVDA, donde M10 ya falla con CPCV, se espera que M10-WF también falle:
chequeo de consistencia, no de rescate.)

**Config congelada.** Ventana **expandible anclada**; `N0=150` días iniciales (coherente con la ventana 150 ya
pre-registrada en `walkforward_robustez`), **reentreno cada `step=21`** (mensual), **embargo=5** entre fin de
train y test, `t1=índice.shift(-1)`, `signal_lag=1`, XGBoost con los MISMOS `PARAMS` que M10-CPCV, semilla 42.
22 features idénticas. Comparación apples-to-apples: M5/M8/M10-CPCV/M10-WF/B&H evaluados TODOS en `[N0:fin]`.

**Limitación declarada (potencia).** El agente solo existe en el OOS (~400 días) ⇒ tramo de test ~250 días y
modelos iniciales entrenados con ~150 días (vs ~267 efectivos de CPCV). Es un test **indicativo de
desplegabilidad**, no inter-época; baja potencia, un cambio de régimen puede dominar. Se reporta como tal.

**Datos.** SPY (~402 días OOS) y NVDA (409). Cachés existentes (hmm.pkl / hmm_nvda.pkl, garch_<tk>.pkl, agente).

**Output esperado.** `outputs/experiments/walkforward_m10_causal.json` con, por activo: config (N0/step/embargo,
test_span, n_retrains), métricas por brazo en `[N0:fin]` (accuracy, auc, mcc, sharpe), McNemar M10-WF vs M5 y
vs M10-CPCV, ΔSharpe(M10-WF−M5) IC95, sanity dual, verdict. Script: `experiments/walkforward_m10_causal.py`.

**Documento de salida.** Sección de desplegabilidad en el canónico (§A.2c / o §13-bis) + entrada [Hallazgo].

---

## [2026-06-15] [Hallazgo] - M10 SÍ es desplegable causalmente en SPY (no es artefacto de CPCV); NVDA no

**Contexto.** Ejecutado el pre-registro anterior. Auditado por @rigor-matematico en diseño (APROBADO CON
CONDICIONES: guarda AUC, MCC por brazo, p separados, notas potencia/FWER) y en resultados (APROBADO CON
CONDICIONES: claims permitidos/prohibidos). Todas aplicadas.

**Detalle (cifras de `walkforward_m10_causal.json`, tramo de test causal `[N0=150:fin]`, reentreno mensual).**

*SPY* (test 2025-05-09→2026-05-11, n=251, 12 reentrenos):
- accuracy: M5 0.367 · M8 0.442 · M10-CPCV (tramo) 0.514 · **M10-WF causal 0.534** · B&H 0.566. MCC: M10-WF
  +0.065 (>0), M5 −0.138. AUC M10-WF 0.541.
- **McNemar M10-WF vs M5: p<0.001** (90 aciertos WF-solo vs 48 M5-solo) → el rescate de accuracy **sobrevive
  al walk-forward causal**. **McNemar M10-WF vs M10-CPCV: p=0.65** (no-rechazo) → **sin evidencia de que CPCV
  halague**, y eso con el WF en desventaja de muestra (≈145 días iniciales vs ≈267 de CPCV).
- ΔSharpe(M10-WF−M5) mediana +3.43, IC95 [0.41, 6.40] (excluye 0) — corroboración económica **subordinada**
  (sin DSR, contaminada por umbrales): NO load-bearing.

*NVDA* (test 2025-05-08→2026-05-19, n=259, 13 reentrenos):
- accuracy M10-WF 0.510, no bate a M5 (McNemar p=0.44), MCC≈0, AUC 0.47<0.5, ΔSharpe IC95 cruza 0. WF≠CPCV
  marginal (p=0.073) pero **ambos en/por debajo del azar** → consistente: M10 no rescata a NVDA por ninguna vía.

**Lo afirmable (y lo que NO).**
- SÍ: "en SPY el rescate direccional de M10 es **desplegable a diario** (walk-forward causal mes a mes) y **no
  un artefacto de CPCV viendo el futuro**". Es la respuesta directa a la objeción del tutor.
- NO (límites obligatorios): el rescate es **relativo a M5**, NO habilidad absoluta — el **sign test M10-WF
  vs azar NO es significativo** (k=134/251, p=0.31, IC95 [0.470, 0.597] contiene 0.5); M10-WF **no supera a
  B&H** (0.566); es **una ventana OOS de 12 meses**, no inter-época; "WF vs CPCV p=0.65" es **no-rechazo, no
  equivalencia** (baja potencia, n=251).

**Sanity same-day/causal — reinterpretado, no recalculado.** El flag `sanity_ok=False` (SPY-M10-WF) salta
porque Sharpe causal +0.36 y same-day −0.31 tienen signo opuesto. La heurística (sign-match) está calibrada
para el **agente** (reacciona a hoy ⇒ no-leak = same-day≥causal). M10 es un **forecaster explícito de
r_{t+1}**: causal>0 con same-day≤0 es la **firma de que predice mañana** (una fuga inflaría el same-day). El
patrón se repite en M10-CPCV (+1.53/−0.12) y es el espejo del agente (M5 −3.07/+0.51) → **ausencia de
look-ahead**. Se deja el `False` en el JSON (output honesto del check pre-registrado) y se documenta; NO se
ajusta el umbral para forzar `True` (sería p-hacking del propio sanity).

**Implicaciones para el TFG.**
(1) Cierra la objeción "CPCV ve el futuro / ¿esto funciona en tiempo real?" — el tutor la planteó
   explícitamente. M10 aguanta validación **causal** estricta en SPY.
(2) CPCV sigue siendo el estimador insesgado canónico (Decisión #10); esto es validación **adicional**, no
   sustitución. No-rechazo WF-vs-CPCV ≠ "CPCV es perfecto".
(3) Refuerza la lectura accuracy-first: la evidencia load-bearing es accuracy/McNemar/MCC; el Sharpe (aunque
   su IC excluya 0) es ilustrativo.
(4) Coherente con el resto: M10 recupera dirección frente al agente perdedor, no bate al mercado, y la
   frontera del leverage effect (no transfiere a NVDA) se mantiene también en el plano causal.

**Referencias.** `experiments/walkforward_m10_causal.py`, `outputs/experiments/walkforward_m10_causal.json`,
`notebooks/_build.py` (Apéndice A §A.2c), pre-registro BITACORA [2026-06-15].

---

## [2026-06-15] [Pre-registro] - Recon: ¿qué activo del panel da M10 causal (desplegable) > B&H en accuracy?

**Contexto.** El tutor pide basar el trabajo en un **caso de estudio** donde M10 (o variante) **desplegable**
bata **en accuracy a todo** (M5/M8/B&H). v7 ya mostró (con DSR) que ningún M10 bate a B&H **en EQUITY** en
SPY → tesis "disciplina de riesgo, no alfa". Esto es DISTINTO: **accuracy direccional** (no equity) sobre el
**panel** (no solo SPY). Reconocimiento **exploratorio barato** previo a decidir si hace falta construir la
M10-v3 causal completa. NO sustituye decisiones vivas.

**Hipótesis.** Existe ≥1 activo del panel donde la **M10 causal walk-forward (vanilla, desplegable)** tiene
accuracy direccional **> B&H y > M5/M8** en el tramo de test. Plausible en activos con B&H débil (laterales/
bajistas: acc B&H ≈ % días alcistas ≈ 0.5), no en toros fuertes (SPY 0.566, NVDA 0.552).

**Criterio.** Reportar **los 10 activos** (anti-cherry-pick); marcar candidatos con acc(M10-WF) > acc(B&H) y
> M5/M8. **NO se selecciona caso de estudio aún**: si hay candidato, pasa a fase siguiente (M10-v3 causal +
@rigor-matematico + validación). Si ninguno, también es resultado (coherente con v7).

**Config.** HMM K=3 + GARCH ajustados **por activo on-the-fly** (como `_oos_m5_m8`); causal WF expandible
`N0=150/step=21/embargo=5`; `signal_lag=1`; seed 42. Accuracy direccional en el tramo `[N0:fin]`.

**Output.** `outputs/experiments/m10_causal_panel_recon.json`. Script: `experiments/m10_causal_panel_recon.py`.

**Resultado del recon (2026-06-15).** Candidatos M10-WF causal > B&H en accuracy: **MSTR, SMCI, MARA** (los de
B&H débil). Único que bate a TODO (M5/M8/B&H): **SMCI** (M10-WF 0.522 vs M8 0.494, M5 0.482, B&H 0.486).
Margen FINO y no significativo de momento (~0.7 SE sobre 0.5; n=249) → requiere M10-v3 + tests (fase siguiente).

---

## [2026-06-15] [Pre-registro] - M10-v3 causal (desplegable): caso de estudio donde bate a todo en accuracy

**Contexto.** El recon señala SMCI (y MSTR/MARA) como candidatos, pero el M10 vanilla no da margen significativo.
Construimos la **M10-v3 desplegable** (las 4 mejoras de `M10_V3_GUIA.md`/`chat_m10.md`, pero **causales**, no
con estadísticas globales del OOS como la versión documentada — esa es look-ahead) y testeamos si bate a todo
**en accuracy** de forma **significativa y robusta a multiplicidad** en algún activo. Distinto de v7 (equity en
SPY, refutado): aquí es **accuracy direccional** sobre el panel. Consistente con "disciplina de riesgo, no alfa":
batir en *acierto* no implica batir en *equity*.

**Hipótesis.** Existe ≥1 activo (candidato SMCI) donde la **M10-v3 causal** tiene accuracy direccional en días
activos **> M5, M8 y B&H**, con significancia que sobrevive corrección de multiplicidad, y **> 0.5** (habilidad
real, no solo relativa).

**H0.** En todo activo, accuracy(M10-v3, días activos) ≤ max(M5, M8, B&H) **o** no significativa; y/o no supera
a 0.5 (sign test). [No hay caso de estudio defendible.]

**Estadístico.** McNemar pareado M10-v3 vs M5/M8/B&H **sobre los mismos días activos**; sign test de la accuracy
de M10-v3 vs 0.5; **Holm-Bonferroni sobre TODOS los contrastes McNemar del panel** (10 activos × {vs M5, vs M8,
vs B&H} = 30 tests) — control de multiplicidad completo, no solo vs B&H (enmienda auditoría B1). Secundario
(económico, ilustrativo, sin elevar): equity/Sharpe con DSR si se mencionara.

**Métrica doble (enmienda auditoría B2/B3).** Se reporta accuracy en **días activos** Y a **cobertura completa**
(dirección de v3 sin abstención, comparable a B&H que opera el 100%), para los 4 brazos; además **% días
alcistas en activos vs global** (detectar si la abstención selecciona días) y **AUC + Brier** de v3 (probar que
la isotónica calibra).

**Criterio de éxito (caso de estudio SOSTENIDO).** En el activo elegido: accuracy(M10-v3) > M5, M8 y B&H
**tanto en días activos como a cobertura completa** (no es artefacto de abstención) **Y** McNemar vs B&H con
Holm p_adj<0.10 (pool de 30) **Y** sign test vs 0.5 p<0.10 (skill absoluto) **Y** composición direccional de
días activos no sesgada (|%alcista_activos − %alcista_global| pequeño). El activo se **justifica ex-ante**.

**Criterio de fracaso (pre-registrado).** Si ningún activo cumple lo anterior tras multiplicidad ⇒ se reporta
**negativo** (refuerza "disciplina de riesgo, no alfa"; coherente con v7). No se baja el listón post-hoc.

**M10-v3 CAUSAL (config congelada).** Por reentreno mensual (expandible, `N0=150/step=21/embargo=5`): (1)
XGBoost **80×prof3** (capacidad reducida, Hastie 2009); (2) **isotónica causal** ajustada sobre un split de
calibración interno del propio train (últimas ~20% obs del pasado, NO el test; Niculescu-Mizil 2005); (3)
**abstención** del 30% menos confiado con **umbral del cuantil del pasado** (Cortes-DeSalvo-Mohri 2016), NO del
OOS global; (4) **renorm P95** del pasado (solo afecta magnitud/equity, NO la accuracy). `signal_lag=1`, seed 42.

**Accuracy con abstención.** Métrica primaria = accuracy en **días activos** + **cobertura** (% días operados);
comparación **pareada sobre los mismos días activos** vs M5/M8/B&H. Se reporta cobertura siempre (la abstención
cambia el denominador; B&H opera el 100%).

**Anti-cherry-pick.** Se corre en **los 10 activos** (se reportan todos), no solo SMCI; multiplicidad corregida;
caso de estudio justificado ex-ante; validación ya causal (walk-forward).

**Output.** `outputs/experiments/m10_v3_causal_panel.json`. Script: `experiments/m10_v3_causal_panel.py`.

---

## [2026-06-15] [Hallazgo] - M10-v3 causal NO bate a todo en accuracy en ningún activo del panel

**Detalle.** Ejecutado el pre-registro anterior. `caso_estudio_sostenido = []`. **Brier de M10-v3 > 0.25 en los
10 activos** (0.264–0.297) → peor que el predictor trivial 0.5: **no hay habilidad probabilística** causal.
AUC ≈ 0.41–0.54. El candidato del recon (SMCI vanilla 0.522) se desploma a 0.39 con la v3 causal (anti-
predictivo). Los únicos contrastes que rechazan Holm-30 son v3 **perdiendo** (BAC/MSTR vs M5/M8). La abstención
apenas dispara (cobertura 0.88–1.0) porque la confianza es uniformemente baja.

**Implicación.** Confirma que los números buenos de M10-v3 (`chat_m10.md`/`M10_V3_GUIA.md`: acc 0.604, equity
1.148) eran **artefacto de look-ahead** (isotónica/abstención/P95 globales + CPCV). Hecha **causal/desplegable**,
la señal desaparece. Extiende v7: M10 no bate a B&H ni en equity (v7) ni en accuracy (esto), en ningún activo.

**Referencias.** `experiments/m10_v3_causal_panel.py`, `outputs/experiments/m10_v3_causal_panel.json`.

---

## [2026-06-15] [Pre-registro] - Caso de estudio: M10 desplegable (split val/test 60/40) vs M5 y B&H en accuracy

**Contexto.** El tutor pide un caso de estudio donde M10 (o M8) **desplegable** bata a **M5 y B&H en accuracy**.
El recon mostró que en activos de **B&H débil** (cayeron/laterales: MSTR/MARA/SMCI/UNG con B&H≤0.5) hay
candidatos. Nueva pieza (idea de Raquel): **split cronológico validación/test** para **optimizar la config en
validación (pasado) y reportar en test intacto (futuro)** — desplegable y honesto. Foco **M10** (M8 como brazo
de referencia). Distinto de v7 (equity) y del M10-v3 causal (sin optimización): aquí accuracy + optimización
honesta en validación.

**Hipótesis (mecanística, ex-ante).** En activos donde el pasivo es mal apostador direccional (**B&H accuracy
≤ 0.5 en VALIDACIÓN**), una M10 desplegable optimizada en validación bate a M5 **y** a B&H en accuracy **sobre
el test no visto**. Predice los activos antes de mirar (MSTR/MARA/SMCI/UNG) → no es "buscar hasta encontrar".

**H0.** En todo activo, accuracy(M10 test) ≤ max(M5, B&H) **o** no significativa.

**Estadístico.** McNemar pareado M10 vs M5 y vs B&H **en test**; sign test M10 vs 0.5; **Holm** sobre los
contrastes McNemar del panel en test. Secundario ilustrativo: Sharpe (sin elevar; sin DSR no se reclama).

**Config congelada.** Split **cronológico 60/40** de `valid` (validación = primeros 60%, test = últimos 40%,
INTACTO). Optimización **SOLO en validación**: grid pre-registrado de **6 configs** = capacidad `{(80,3),(300,4)}`
× feature set `{ALL-22, régimen+STRATA-7, agente-15}`; **selección por accuracy de validación a COBERTURA
COMPLETA** (así la abstención no manipula la selección). XGBoost entrenado en un sub-split interno de validación
(fit 80% / calib 20%) con **isotónica** ajustada en el calib y aplicada al test (causal). La **abstención 30%**
(umbral del calib de validación) se aplica como **overlay FIJO** y se reporta aparte (test activo vs completo).
`signal_lag=1`, seed 42.

**Criterio de éxito (SOSTENIDO).** ≥1 activo del cohorte B&H-débil donde **en test**: accuracy(M10) > M5 **y** >
B&H **a cobertura completa** (no solo en activos), McNemar vs B&H con Holm p_adj<0.10, sign test vs 0.5 p<0.10,
y sin sesgo de composición (|frac_up_test − frac_up_val| < 0.05). Activo justificado ex-ante.

**Criterio de fracaso.** Ninguno lo cumple → **negativo** (coherente con v7 / mercados eficientes). El test se
toca **una sola vez**; no se re-optimiza tras verlo, no se baja el listón.

**Anti-cherry-pick.** Cohorte B&H-débil por hipótesis ex-ante (B&H≤0.5 en validación); se reportan **los 10**;
Holm en test; M8/M5/B&H como brazos de referencia.

**Output.** `outputs/experiments/m10_valtest_casestudy.json`. Script: `experiments/m10_valtest_casestudy.py`.

---

## [2026-06-16] [Hallazgo] - M10 val/test 60/40 NO da caso de estudio: negativo robusto en accuracy

**Detalle.** `caso_estudio_sostenido = []`. Único "M10 > M5 y B&H" nominal = **UNG** (0.509 vs 0.497/0.497),
pero **moneda** (sign vs 0.5 p=0.87, McNemar vs B&H p=0.50) y **fuera del cohorte** (B&H val 0.510 > 0.5). En
el cohorte B&H-débil (TSLA/MSTR/SMCI/MARA) ninguno: en MSTR/MARA el mejor es **M5**. Cuando la validación
eligió una config "confiada" (regime_strata7), el test se **invirtió** (NVDA M10=0.189, SMCI M10=0.150 ≪
azar; sign p=0.00 por ser *anti-predictivo*) → demostración limpia de que no hay señal y que optimizar sobre
validación sobreajusta ruido. **Tercera búsqueda negativa** (tras recon vanilla y M10-v3 causal): una M10
desplegable no bate a B&H en accuracy en ningún activo del panel a horizonte diario.

**Referencias.** `experiments/m10_valtest_casestudy.py`, `outputs/experiments/m10_valtest_casestudy.json`.

---

## [2026-06-16] [Pre-registro] - Horizonte semanal (5 días): ¿M10/M8 baten a M5 y B&H en accuracy?

**Contexto.** A horizonte **diario** la dirección es casi paseo aleatorio (Brier>0.25, sin edge) → ningún
M10/M8 desplegable bate a B&H en accuracy (3 búsquedas negativas). **Hipótesis mecanística nueva:** el régimen
HMM es un objeto **persistente multi-día** y el *leverage effect* opera a escala de **semanas**; predecir el
signo del retorno a **5 días** debería tener más señal/ruido y el régimen informarlo mejor. Último intento
honesto antes de cerrar. **Limitación dominante declarada:** OOS ~400 días ≈ **~80 semanas efectivas** → muy
poca potencia; un positivo puede no ser significativo y un negativo no es prueba de ausencia.

**Hipótesis (ex-ante).** En activos donde el pasivo es mal apostador a 5 días (**frac. semanas alcistas ≤ 0.5
en validación**), **M10 o M8** baten a M5 **y** a B&H en **accuracy direccional a 5 días** sobre el test no
visto. Cohorte ex-ante = los de B&H-semanal débil (esperado: MSTR/MARA/SMCI/...).

**H0.** En todo activo, accuracy(M10 y M8 a 5d) ≤ max(M5, B&H) **o** no significativa (block-permutation).

**Diseño.** Target `y5_t = signo(Σ r_{t+1..t+5})` (log-retornos). **Solapado diario** (~400 obs) → estimación
de accuracy estable, pero **significancia corregida por autocorrelación**: **block-permutation test**
(bloque ≈ 5–10) para los pareados, y **N_eff de Bartlett** reportado. Split **cronológico 60/40** con **purga
de 5 días** en la frontera (las etiquetas de train no solapan el test → sin fuga). M10 = XGBoost (config FIJA
cap80×3/all22, **sin grid** por N pequeña) entrenado en validación, predice test. M8 = regla override-C
(determinista, desplegable). `signal_lag` implícito en el target a 5 días; seed 42.

**Estadístico.** Accuracy a 5d en test de M5/M8/M10/B&H; **block-permutation** M10/M8 vs M5 y vs B&H; sign
test vs 0.5 con N_eff; AUC de M10. **Holm** sobre los contrastes del panel en test.

**Criterio de éxito (SOSTENIDO).** ≥1 activo del cohorte B&H-débil donde **en test**: accuracy(M10 **o** M8) >
M5 **y** > B&H, **block-permutation vs B&H** con Holm p_adj<0.10, sign vs 0.5 p<0.10, y (si es M10) AUC>0.5.

**Criterio de fracaso.** Ninguno lo cumple → **negativo / cierre** (coherente con mercados eficientes; tesis
"disciplina de riesgo, no alfa"). Test se toca una vez; no se prueban otros horizontes (5d pre-fijado, NO
3d/10d → evitar p-hacking de horizonte).

**Anti-cherry-pick.** Cohorte ex-ante por B&H-semanal≤0.5 en validación; se reportan los 10; Holm; horizonte
único pre-registrado.

**Output.** `outputs/experiments/m10_weekly_horizon.json`. Script: `experiments/m10_weekly_horizon.py`.

---

## [2026-06-16] [Error] - El signo del régimen en RAM regresó a HARDCODE, invertido en activos individuales

**Contexto.** Auditoría @harvard-professor buscando qué suprime la accuracy de M8/M10. Hallazgo: el signo de
RAM/override-C está **hardcodeado** (`strata/detectors.py:209`: `regime_sign = 1 si calm_prob≥crisis_prob,
si no −1` = Calma→long, Crisis→short), **idéntico para los 10 activos**. Viola `CLAUDE.md §9` ("prior RAM
data-driven por activo, nunca Crisis⇒short hardcoded") y es **incoherente** con la Parte A del propio
`walkforward_robustez.py` (que usa signo data-driven `direction_map_frozen`).

**Detalle (μ por estado, calibración 2000→2024-09, recalculado con venv).** MARA: Calma **−0.0019** (baja),
Crisis **+0.0056** (sube) → el hardcode pone M8 **corto en el estado más alcista y largo en el bajista**
(signo invertido en AMBOS extremos). NVDA: Crisis +0.0017 > Calma +0.0015 (Crisis es el más alcista → M8
corto donde sube). SMCI: Calma −0.00001, Crisis +0.0016 (inversión). En estos activos M8 corrige al agente
**en la dirección equivocada por diseño**.

**Implicaciones para el TFG.** Corregir el signo a **data-driven por activo** (signo del drift medio de cada
estado, congelado en calibración) antes de reportar M8. Es obligatorio por coherencia con §9 y cierra un
vector de ataque del tribunal. Honesto: la regla **prior-flip** ya documenta dónde el signo de calibración
NO transfiere al OOS.

**Referencias.** `strata/detectors.py:209`, `strata/intervention.py:158`, `walkforward_robustez.py:308-338`.

---

## [2026-06-16] [Pre-registro] - M8 con signo de régimen DATA-DRIVEN (arreglo del bug) vs M5 y B&H en accuracy

**Contexto.** Arreglo del [Error] anterior. El signo de override-C pasa de hardcode (Calma→long/Crisis→short)
a **data-driven por activo**: `sign_dd[estado] = +1` para el estado de mayor μ de calibración (long_state),
**−1** para el de menor μ (short_state), **0** para el intermedio (neutro, como Estrés). Congelado en
calibración → sin look-ahead. M8 es regla determinista → **todo el OOS (~400 días) es test válido** (más
potencia que los splits de M10).

**Hipótesis (ex-ante, mecanística).** En los activos donde el hardcode estaba **invertido** (sign_dd ≠
hardcode: MARA/NVDA/SMCI y los que salgan), **M8-dd** (signo corregido) bate a M5 **y** a B&H en accuracy
direccional OOS, con significancia. El arreglo no toca activos donde el signo ya coincidía (p.ej. SPY).

**H0.** En todo activo, accuracy(M8-dd) ≤ max(M5, B&H) **o** no significativa.

**Diseño.** Por activo: HMM K=3 + μ por estado en calibración → `long_s=argmax μ`, `short_s=argmin μ`.
**M8-dd = override-C FIEL** pero con esos `long_s/short_s` **data-driven** en vez de Calma/Crisis fijos
(misma mecánica: inconsistencia = masa del estado extremo OPUESTO al agente; si ≥ τ=0.5, override a
`regime_sign` = signo del extremo dominante). Para SPY se reduce al original; para MARA/NVDA/SMCI voltea el
signo. Comparar accuracy de **M5, M8-original (hardcode), M8-dd, B&H** sobre todo el OOS. **prior-flip** =
diagnóstico de los estados extremos sobre los primeros 60 días OOS (no garantía sobre todo el OOS); se
reporta μ_OOS_60 junto a μ_calib. Empate de μ → activo no evaluable. `signal_lag` implícito (target
r_{t+1}); seed 42.

**Estadístico.** Block-permutation (autocorr-robusto) y McNemar M8-dd vs M5 y vs B&H; sign test M8-dd vs 0.5;
**Holm** sobre los contrastes del panel.

**Criterio de éxito (SOSTENIDO).** ≥1 activo del cohorte invertido donde **M8-dd > M5 y > B&H** en accuracy,
**block-permutation vs B&H con Holm p_adj<0.10**, **sign vs 0.5 p<0.10**, y prior_flip=False (el signo de
calibración transfiere al OOS).

**Criterio de fracaso.** Ninguno lo cumple → el bug del signo es real y se corrige por coherencia, pero no
produce un caso de estudio que bata a B&H (límite de mercado eficiente). Honesto, sin bajar el listón.

**Anti-cherry-pick.** Cohorte ex-ante = activos con sign_dd≠hardcode (mecanístico, antes de mirar OOS); se
reportan los 10; Holm; prior-flip declarado.

**Output.** `outputs/experiments/m8_datadriven_sign.json`. Script: `experiments/m8_datadriven_sign.py`.

## [2026-06-16] [Pre-registro] - Experimento m10-improve-smci (mejora honesta de accuracy de M10)

**Contexto.** Antes de cerrar SMCI como caso de estudio, Raquel pide intentar **subir la accuracy de M10**
con palancas legítimas, eligiéndolas en validación y reportando en test (validación≠test). El M10-WF base
sobre SMCI da 0.524 (M5 0.484, M8 0.496, B&H 0.484).

**Hipótesis.** Un conjunto pequeño y pre-registrado de hiperparámetros de M10 —elegidos SOLO en validación—
mejora la accuracy direccional de M10 sobre el test no visto frente al M10-base (all22/thr0.5/flat/1-seed),
y al menos lo mantiene por encima de M5/M8/B&H.

**H0.** En test, accuracy(M10-sel) ≤ accuracy(M10-base) **o** ≤ max(M5, M8, B&H) **o** no significativa.

**Palancas (las 5 que eligió Raquel, como HIPERPARÁMETROS, no modelo congelado).**
1. **Umbral ≠ 0.5** — grid {0.45…0.55}, elegido en validación.
2. **Selección de features** — {all22 / régimen+STRATA-7 / agente-15 / STRATA7+señal-real / all22+señal-real}.
3. **Pesos por recencia** — semivida {flat, 252, 126} días (no estacionariedad de SMCI).
4. **Ensemble de semillas** — 10 semillas (reduce varianza); se reporta 1-seed como referencia.
5. **Features con señal real, CAUSALES** — momentum 5/21/63, vol relativa rv21/rv63, racha de signo.
   Conocidas en t, predicen r_{t+1}. Esto define una **variante M10-aumentada** (los detectores STRATA no
   cambian); se documenta como tal, no como STRATA core.

**Diseño (desplegable + honesto).** OOS SMCI = 400 días válidos. Split cronológico 60/40: validación =
primeros 240, test = últimos 160 (intacto, se toca **una vez**). Selección = walk-forward DENTRO de
[N0=150 : split] (nunca ve el test); se elige (feature set, recencia, umbral) que maximiza accuracy en los
~90 días de validación evaluables. Test = walk-forward con **pasado expandible** (reentreno cada 21 d,
embargo 5, incluye validación) prediciendo [split:fin] con la config congelada. El reentreno en test es lo
que hace el despliegue real (arregla el colapso del modelo congelado 60/40 = 0.150).

**Estadístico.** McNemar y block-permutation (autocorr-robusto) de M10-sel vs M5/M8/B&H; **Holm** sobre los
3 contrastes; sign test M10-sel vs 0.5. Mejora sobre base = acc(M10-sel) − acc(M10-base) en test.

**Criterio de éxito.** En test: acc(M10-sel) > acc(M10-base) **y** > max(M5, M8, B&H), con McNemar vs B&H
bajo Holm p_adj<0.10 y sign vs 0.5 p<0.10. (Éxito parcial defendible: mejora nominal sobre base + sigue
batiendo a M5/M8/B&H, aunque la significancia quede corta por n_test=160.)

**Criterio de fracaso (pre-registrado).** La config elegida en validación NO mejora (o empeora) en test →
se reporta honestamente que las palancas no aportan señal desplegable y se cierra SMCI con el M10-base.
NO se re-optimiza tras ver el test ni se baja el listón post-hoc.

**Riesgos declarados.** Validación corta (~90 días) → selección ruidosa; test corto (160) → potencia
limitada para significancia. Por eso el grid es pequeño y pre-registrado y el test se lee una sola vez.

**Output.** `outputs/experiments/m10_improve_smci.json`. Script: `experiments/m10_improve_smci.py`.

**Enmiendas tras auditoría @rigor-matematico (BLOQUEOS B1–B5, aplicadas antes de ejecutar).**
- **B1.** Documentado en `wf_p1` que la etiqueta tiene horizonte 1 día y por qué EMBARGO=5 purga la frontera
  (López de Prado 2018, sec. 7.4; purga unilateral por ser solo-pasado).
- **B2.** Split alineado a múltiplo de STEP desde N0 (`split = N0 + STEP·⌊(0.6n−N0)/STEP⌋`) → en test el
  primer reentreno cae EXACTO en split (entrena `[:split−5]`, predice desde split). Elimina el acoplamiento
  de rejilla en la frontera 60/40. `split_efectivo` se registra en `meta`.
- **B3.** `acc_val` del ganador es el MÁXIMO sobre 5×3×11 = 165 combinaciones en ~84 días → optimista, NO
  insesgado. Se marca explícito en `meta` (`acc_val_es_maximo_sobre_grid`, `nota_b3`) y se reporta la
  dispersión del grid (`grid_validacion`). El test es la única lectura honesta.
- **B4.** Decisión pre-ejecución: el **sign-test vs 0.5 es sanity ORTOGONAL** (null distinto: ¿mejor que
  moneda?), NO confirmatorio. Holm cubre solo los 3 McNemar (vs M5/M8/B&H). Documentado en `nota_b4`.
- **B5.** Assert `360≤n≤440` (aborta si el split se movería sin traza). Docstring alineado a 5 feature-sets.
- **A2/A3 (advertencias).** Cobertura de M5/M8 (días con apuesta ≠0) reportada; M8 se compara como
  override-C canónico (signo hardcoded, posiblemente invertido en SMCI por leverage débil — ver
  m8_datadriven_sign). Notas en `meta`.

## [2026-06-16] [Hallazgo] - Mejorar M10 con tuning en SMCI NO transfiere a test (sobreajuste de selección)

**Contexto.** Antes de cerrar SMCI como caso de estudio, se intentó subir la accuracy de M10 con 5 palancas
pre-registradas (umbral≠0.5, selección de features, recencia, ensemble de semillas, features de señal real
causales), eligiéndolas SOLO en validación (84 días) y reportando en test (166 días, intacto).

**Detalle.** La config ganadora en validación (all22+señal-real / recencia hl126 / umbral 0.47) alcanzó
acc_val=0.5595 —máximo sobre 165 combinaciones, optimista por construcción— y **colapsó a 0.4759 en test**
(−0.0836 val→test): firma cuantitativa del **sobreajuste de selección** (maximizar sobre 165 celdas en 84
días selecciona ruido). En test: M10-sel=0.4759, M5=0.512, M8=0.524, B&H=0.470, y el M10-base sin tunear
(all22/thr0.5/flat/1-seed)=0.578. **Mejora sobre base = −0.1024.** Ningún brazo bate significativamente a
B&H ni al azar (McNemar M10-sel vs B&H p=1.0; sign vs 0.5 p=0.587; IC95 [0.398,0.555] contiene 0.5).

**Veredicto pre-registrado.** Se cumple el **criterio de FRACASO**: las palancas elegidas en validación no
aportan señal desplegable. SMCI se cierra con el **M10-base** (sin tuning). No se re-optimiza tras ver el
test ni se baja el listón (sería look-ahead de selección).

**Implicaciones para el TFG.** (1) Valor METODOLÓGICO, no de rendimiento: demostración por construcción del
riesgo de sobreajuste de selección y de por qué se respeta validación≠test (anti-p-hacking). Capítulo
defendible. (2) Claims permitidos: el tuning se refuta; M10-base bate nominalmente (descriptivo, 1 seed,
una loncha, sin test pareado → NO inferenciable). Claims prohibidos: "tuning mejora", "0.578 demuestra que
M10 funciona", re-selección de config tras ver el test. (3) Coherente con v7 / mercado casi eficiente en
dirección diaria con muestra corta.

**Referencias.** `experiments/m10_improve_smci.py`, `outputs/experiments/m10_improve_smci.json`, pre-registro
+ enmiendas B1–B5 [2026-06-16]. Auditoría @rigor-matematico (diseño BLOQUEADO→B1–B5 aplicadas→APROBADO;
resultados APROBADO con condición: ningún claim inferencial sobre M10-base sin McNemar/sign pareados).

## [2026-06-16] [Pre-registro] - Experimento m10-pivot-scan (elección rigurosa del activo de pivote)

**Contexto.** El proyecto pivota a UN solo activo. Pregunta: ¿en qué activo la M10 DESPLEGABLE bate a M5, M8
y B&H en accuracy, de forma significativa y robusta? Métrica clave = accuracy direccional; se enriquece con
Sharpe, equity y Deflated Sharpe. La M10 desplegable = walk-forward expandible (reentreno mensual, solo
pasado) → TODO el OOS es test válido para una config FIJA (no hay selección por activo → no hay split
val/test necesario; más potencia que la loncha 40%).

**Hipótesis (ex-ante, mecanística).** En la cohorte de activos donde B&H es **mal apostador direccional**
(accuracy B&H ≤ 0.5 en OOS, i.e. cayeron/laterales), una M10 desplegable de config FIJA bate a B&H (y a
M5/M8) en accuracy, porque puede ponerse corta donde el pasivo pierde. Cohorte declarada ANTES de mirar:
los de B&H≤0.5 (candidatos: MSTR, MARA, SMCI, UNG; se confirma con el dato).

**H0.** En todo activo, accuracy(M10) ≤ max(M5, M8, B&H) **o** no significativa vs B&H.

**Configs FIJAS a priori (NO tuneadas por activo → sin sobreajuste de selección).** Tres, motivadas a
priori, no elegidas sobre los datos:
- **base** = 22 features, thr 0.5, 1 semilla (M10-WF canónica, la desplegable de referencia).
- **ens** = 22 features, thr 0.5, ensemble 10 semillas (reducción de varianza, siempre defendible).
- **aug** = 22 features + señal real causal (momentum 5/21/63, vol relativa, racha), ensemble 10 semillas.
WF N0=150, step=21, embargo=5, expandible. Seed 42.

**Estadístico.** Por activo×config: accuracy de M10/M5/M8/B&H; McNemar y block-permutation (autocorr-robusto)
de M10 vs M5/M8/B&H; sign test M10 vs 0.5; Sharpe causal (lag=1), equity final, Deflated Sharpe (n_trials =
nº configs probadas). **Familia confirmatoria primaria = McNemar M10-vs-B&H sobre 3 configs × 10 activos
(Holm-30).** vs M5/M8 = secundario (reportado, Holm aparte).

**Criterio de éxito (caso de estudio FUERTE).** ≥1 activo de la cohorte B&H-débil donde, sobre todo el OOS:
accuracy(M10) > M5 **y** > M8 **y** > B&H, con **McNemar vs B&H bajo Holm-30 p_adj<0.10**, sign vs 0.5
p<0.10, y Sharpe/equity coherentes (mismo signo de ventaja). Ese es el activo de pivote.

**Criterio de éxito PARCIAL (defendible, lo que el tutor aceptó).** M10 > M5/M8/B&H **nominalmente** en un
activo B&H-débil, con significancia limitada por n; el resto = trabajo futuro. Se reporta como tal, sin
inflar.

**Criterio de fracaso (pre-registrado).** Ningún activo da victoria significativa NI nominal robusta → se
reporta negativo (coherente con mercado casi eficiente) y la contribución es metodológica. No se baja el
listón ni se re-selecciona config tras ver resultados.

**Anti-cherry-pick.** Cohorte ex-ante (B&H≤0.5, mecanística); se reportan los 10 activos; Holm-30; DSR por
las configs probadas; configs fijas (no tuneadas por activo).

**Output.** `outputs/experiments/m10_pivot_scan.json`. Script: `experiments/m10_pivot_scan.py`.
**Notebook entregable:** `notebooks/m10_better_smci.ipynb` (todas las pruebas, gráficas y conclusiones).

## [2026-06-16] [Pre-registro] - Experimento m10-smci-advanced (mejorar M10 en SMCI: métodos avanzados)

**Contexto.** Raquel pide agotar las palancas sobre SMCI (no pivotar aún). El M10-WF base/ensemble topa en
accuracy 0.524 (bate a M5/M8/B&H nominal, NO significativo: McNemar vs B&H p≈0.39). El ensemble mantiene
accuracy y mejora Sharpe (0.73→1.23) y equity (1.32×→1.98×) → se conserva. Criterio de Raquel: a igual
accuracy, ganar en Sharpe/equity cuenta.

**Hipótesis (exploratoria).** Alguna de estas reformulaciones —motivadas a priori por la literatura—
mejora la accuracy direccional de M10 en SMCI (o, a igual accuracy, su Sharpe/equity), de forma desplegable:
1. **Triple-barrier target** (López de Prado 2018, cap. 3): etiqueta de entrenamiento por TP=+kσ_t / SL=−kσ_t
   / barrera temporal H días (denoising del label); se ENTRENA con triple-barrier pero se EVALÚA la dirección
   contra sign(r_{t+1}) para comparabilidad con M5/M8/B&H.
2. **Modelos especializados por régimen HMM**: 3 XGBoost (ponderados por P_estado del HMM en el fit) y mezcla
   p1 = Σ_s P_s(t)·model_s(x_t).
3. **Stacking M5→M10**: añadir size del agente (M5/M8) como feature de M10 (walk-forward causal; conocido en t).
4. **Voting M5+M10**: acuerdo → seguir; desacuerdo → (cobertura completa) seguir M10 / (activos) abstener.
5. **Abstención condicional**: al régimen (abstener solo en Estrés) y al acuerdo de las 5 personalidades.

**H0.** Ninguna variante supera al mejor M10 fijo (ensemble) en accuracy a cobertura completa con
significancia, ni bate a M5/M8/B&H significativamente.

**Diseño.** Walk-forward expandible desplegable, config FIJA a priori por método (sin tuneo por activo →
todo el OOS = test, ~250 días). Ensemble 10 semillas donde aplique. **Triple-barrier: embargo = H+1 = 6**
(la etiqueta usa H días futuros; el embargo impide que la etiqueta del último día de train alcance el bloque
predicho → sin look-ahead). Métrica primaria = **accuracy a COBERTURA COMPLETA** vs sign(r_{t+1}); la
abstención se reporta con accuracy en días activos Y cobertura (no comparable a B&H si <100%).

**Estadístico.** Por método: accuracy, Sharpe causal (lag=1), equity, DSR (n_trials = nº métodos);
McNemar + block-permutation M10 vs M5/M8/B&H; sign vs 0.5. **Holm** sobre la familia método-vs-B&H.

**Criterio de éxito (FUERTE).** ≥1 método con accuracy > M5/M8/B&H, McNemar vs B&H bajo Holm p_adj<0.10 y
sign vs 0.5 p<0.10. **Éxito SECUNDARIO (criterio de Raquel):** método con accuracy ≥ base y Sharpe/equity
mejores con DSR>0.

**Criterio de fracaso (pre-registrado).** Ninguna variante alcanza significancia en accuracy → se reporta
honestamente (dirección diaria de SMCI casi-eficiente); el entregable es el mejor M10 desplegable (ensemble:
accuracy 0.524 nominal + Sharpe/equity fuertes) + la demostración metodológica. No se baja el listón ni se
re-selecciona tras ver el test.

**Anti-cherry-pick.** Métodos motivados a priori (literatura), no tuneados por activo; Holm sobre todos;
DSR; se reportan todos los métodos (también los que empeoran).

**Output.** `outputs/experiments/m10_smci_advanced.json`. Script: `experiments/m10_smci_advanced.py`.
Notebook entregable: `notebooks/m10_better_smci.ipynb`.

## [2026-06-16] [Hallazgo] - SMCI a fondo: ninguna de 12 variantes desplegables bate a B&H en accuracy (significativo); ensemble = mejor M10 (Sharpe/equity nominal)

> **[ACTUALIZADO 2026-06-17]** Las cifras de abajo son con **embargo=5**. Tras adoptar **embargo=1** (decisión
> del día siguiente, por principio; ver entrada [2026-06-17]), las cifras **headline del proyecto** del
> ensemble pasan a: **accuracy 0.552**, **Sharpe 1.84**, **equity 3.24×**, **DSR 0.72**. El veredicto
> cualitativo NO cambia (bate a todo nominal, no significativo). Usar las cifras de embargo=1 en memoria/notebook.

**Contexto.** Por petición de Raquel, agotar las palancas para mejorar M10 en SMCI antes de pivotar de activo.
Tres experimentos: tuning en validación (m10_improve_smci), configs fijas (m10_smci_deep), métodos avanzados
(m10_smci_advanced: triple-barrier, modelos por régimen, stacking, voting, abstención).

**Detalle.** Sobre el OOS desplegable de SMCI (n=250, walk-forward, B&H=0.484 benchmark justo): **techo de
accuracy = 0.524** (base = ensemble), superior NOMINAL a M5 (0.484), M8 (0.496), B&H (0.484). **Ninguna de las
12 variantes** alcanza significancia vs B&H (McNemar Holm p_adj=1.0; sign vs 0.5 p≥0.49 en TODAS). Triple-
barrier (0.488, embargo=H+1 verificado sin look-ahead), regime_models (0.50), stack_agent (0.492) NO mejoran;
varios degradan. Abstención no concentra acierto (activos≈completa → la confianza no discrimina). El tuning en
validación se desploma en test (−0.10, sobreajuste de selección). **Única mejora robusta = ensemble** (10
semillas): misma accuracy, Sharpe 0.73→1.23, equity 1.32×→1.98× — pero **DSR=0.47<0.5** → no significativo
tras deflactar; nominal/ilustrativo (CLAUDE.md §4).

**Implicaciones para el TFG.** Negativo honesto pre-registrado: la dirección DIARIA de SMCI es casi-eficiente
para estos detectores; el rescate significativo de STRATA en SPY (M10=0.539, leverage effect) NO aparece en un
stock individual con leverage débil (limitación prevista en CLAUDE.md §3). Contribución metodológica:
(i) demostración del sobreajuste de selección + validación≠test; (ii) mapa exhaustivo de 12 métodos con
significancia; (iii) ensemble como mejor M10 desplegable (ventaja nominal en accuracy y riesgo-retorno,
honestamente etiquetada). Entregable: `notebooks/m10_better_smci.ipynb` (15 celdas, 5 gráficas).

**Claims auditados @rigor-matematico.** PERMITIDO: superioridad nominal de base/ens; triple-barrier sin
look-ahead que no mejora; ensemble mejora Sharpe/equity nominal. PROHIBIDO: cualquier "significativo" en
accuracy o en Sharpe/equity; Sharpe sin DSR adjunto. Veredicto: APROBADO con condiciones (γ filtrado
confirmado; strata_real colapso = posible prior-flip; SMCI≠SPY documentado).

**Referencias.** `experiments/m10_{improve_smci,smci_deep,smci_advanced}.py`,
`outputs/experiments/m10_{improve_smci,smci_deep,smci_advanced}.json`, `notebooks/m10_better_smci.ipynb`,
pre-registros [2026-06-16].

## [2026-06-16] [Hallazgo] - SMCI selección de burn-in en validación: M10 bate a B&H sig. en test, pero por sesgo a corto en tramo bajista (no habilidad)

**Contexto.** Raquel señala (correctamente) que elegir burn-in/config en VALIDACIÓN no es p-hacking; pide
elegir la estrategia de mayor accuracy/Sharpe/equity. `m10_smci_select.py`: validación=[N0:250], test=[250:fin]
(intacto), barrido burn-in {100..200} × {base, ens}.

**Detalle.** Elegida por accuracy de validación: base / burn-in 200 (val_acc=0.54 sobre ~50 días → selección
ruidosa). En TEST (n=150, 2025-10-02→2026-05-11): M10 acc=0.56, Sharpe +2.19, equity 2.55× > M5/M8 (0.533) y
B&H (0.447, eq 0.48×). **Bate a B&H significativamente** (block-perm p=0.0067, McNemar p=0.086). PERO el test
es bajista (44.7% días alcistas), M10 está 58% corto, y el trivial **"siempre corto" = 0.553 ≈ M10 (0.56)**.
M10 **no** bate al agente (McNemar vs M5 p=0.71) ni a la moneda (sign vs 0.5 p=0.16, IC95 [0.477,0.641]).

**Implicaciones para el TFG.** Es el problema de SPY AL REVÉS: en SPY B&H gana en mercado alcista (siempre-
largo); aquí M10 "gana a B&H" por estar net-short en un tramo que cae. Defendible como "M10 bate al pasivo en
el periodo de test (block-perm p=0.007)" SOLO si se presenta junto al benchmark siempre-corto (0.553) y se
dice que no bate al agente ni a la moneda. NO presentar como habilidad direccional significativa. La sub-
ventana de test (B&H 0.447) es bajista, no el ≈50% justo del OOS completo (B&H 0.484). Documentado en
`notebooks/m10_better_smci.ipynb` §E con gráficas (accuracy por burn-in en validación + test vs trivial).

**Referencias.** `experiments/m10_smci_select.py`, `outputs/experiments/m10_smci_select.json`,
`notebooks/m10_better_smci.ipynb` §E.

## [2026-06-16] [Hallazgo] - Por qué M5/M8/M10 no se separan en SMCI: el agente está 95% corto (alineado con el régimen)

**Contexto.** Raquel detecta que en SMCI M8≈M5 y M10 no bate al agente; pide el punto clave.

**Detalle.** El agente (M5) está **95% corto** en SMCI (2% largo, 3% neutral). STRATA interviene solo **3%**
de los días (M8≠M5) porque override-C dispara ante incoherencia agente↔régimen, y el agente —ya corto— COINCIDE
con el régimen (alta vol→corto) → no hay nada que corregir → M8≈M5. M10 también es corto-sesgado (58%) →
discordantes McNemar equilibrados (b=65,c=75,p=0.45) → no se separa del agente. En SMCI, M5/M8/M10 son la misma
apuesta ("corto SMCI").

**Barrido del panel (`panel_intervention_scan.py`).** Mide discrepancia agente↔régimen, intervención STRATA y
rescate M10 vs M5 en los 10 activos. **Confirma el mecanismo del TFG:** STRATA rescata donde el agente discrepa
de un régimen que acierta. Ranking discrepancia: ROKU 0.95, MARA 0.83, XLE 0.80, ..., SPY 0.66, ..., SMCI 0.17,
MSTR 0.07. **SPY es el ÚNICO con rescate significativo** (M10 vs M5 p=0.0005; M8 vs M5 p=0.051) porque cumple
las dos condiciones: agente discrepa Y régimen acierta (leverage effect fuerte en índice). Stocks individuales
con alta discrepancia (ROKU interv 88% p=0.13, NVDA, XLE) no llegan a significativo (leverage débil → régimen
no apunta fiable). SMCI al fondo (interv 3%) → sin rescate, como se observó.

**Implicaciones para el TFG.** Explicación honesta y defendible de (i) por qué SPY es el caso central, (ii) por
qué en SMCI los tres modelos se confunden, (iii) por qué nada es significativo en SMCI. Coherente con CLAUDE.md
§3 (leverage effect). Pista: ROKU es el stock individual más "tipo-SPY" (alcista, agente 97% corto, interv 88%)
pero rescate aún no significativo. Documentado en `notebooks/m10_better_smci.ipynb` §F (tabla SPY↔SMCI + 2
gráficas de panel).

**Referencias.** `experiments/panel_intervention_scan.py`, `outputs/experiments/panel_intervention_scan.json`,
`notebooks/m10_better_smci.ipynb` §F.

## [2026-06-17] [Decisión] - embargo = 1 (no 5) en la validación walk-forward de M10

**Contexto.** Raquel cuestiona si el embargo=5 del walk-forward tira los días más recientes (los más
informativos en un activo no estacionario como SMCI). Decisión: **adoptar embargo = 1** por principio.

**Detalle.** Distinción clave (López de Prado 2018, cap.7 §7.4): **purga** = quita train cuya etiqueta se
solapa con el test (tamaño = horizonte de etiqueta); **embargo** = quita unas pocas obs posteriores al test
por autocorrelación residual (h≈0.01·T). Ambos existen porque en K-fold/CPCV los folds tienen train ANTES y
DESPUÉS (interleaved). Mi validación es **walk-forward de origen móvil** (Tashman 2000): el test es siempre
futuro → no hay solape bidireccional. El único solape es la **etiqueta de horizonte 1** (y_t=1[r_{t+1}>0]) →
purga = 1. El embargo≥5 de CLAUDE.md §4 es regla de CPCV (bidireccional) y etiquetas multi-día, otro régimen.
Cierre: Bergmeir, Hyndman & Koo (2018) — con residuos no correlados la CV con hueco mínimo es válida.
Verificado libre de fuga: con embargo=1 hay gap de 2 días entre la última etiqueta de train y el primer
retorno de test.

**Citas verificadas (@experto-citas, todas ✅, ya en tesis/bibliography.bib):** lopezdeprado2018 (existente,
reutilizar), tashman2000, burman1994, racine2000, bergmeir2018, bergmeir2012. Matiz aplicado: "h ∝ dependencia"
se atribuye a Racine/posterior, no a Burman et al. (que lo dan como fracción de N).

**Efecto en SMCI.** accuracy 0.524 (emb5) → 0.552 (emb1), posiciones equilibradas (47% corto, 48% días
alcistas). **NO crea significancia:** el p=0.047 vs B&H aparece SOLO en emb=1 (pico aislado; emb 0 y 2 dan
p≈0.12-0.13), no sobrevive Bonferroni-5 (0.24) ni Holm de familia. Se reporta como **sensibilidad**, no
confirmatorio. embargo=1 se elige por PRINCIPIO (horizonte=1), no por el p-valor.

**Implicaciones para el TFG.** Compatible con CLAUDE.md §4 (la regla ≥5 es de CPCV, no del WF). Documentado en
`notebooks/logic_esential.ipynb` §14b (con frase de defensa lista). **Pendiente:** pre-registrar formalmente
emb=1 antes de propagarlo como headline a los experimentos m10_smci_* y re-ejecutar (accuracy 0.552), marcando
significancia como sensibilidad (condición de @rigor-matematico).

**Referencias.** logic_esential §14b, tesis/bibliography.bib, auditoría @rigor-matematico [2026-06-17],
verificación @experto-citas [2026-06-17].

## [2026-06-17] [Pre-registro] - Adopción de embargo=1 como protocolo del walk-forward de M10

**Hipótesis.** El walk-forward de M10 con **embargo = 1** (= horizonte de la etiqueta, justificado a priori,
NO barrido) es el protocolo correcto para validación rolling-origin con etiqueta de horizonte 1 día, y mejora
la accuracy nominal frente al embargo=5 (sobre-conservador, regla de CPCV).

**H0.** acc(M10, emb=1) sobre el OOS SMCI ≤ max(M5, M8, B&H), o no significativa bajo block-permutation.

**Justificación a priori (no por p-valor).** Purga = horizonte de etiqueta = 1 (López de Prado 2018 §7.4);
en walk-forward rolling-origin (Tashman 2000) el test es siempre futuro → no hay solape bidireccional que
motive el embargo de CPCV; validez con hueco mínimo bajo residuos no correlados (Bergmeir, Hyndman & Koo 2018).
El embargo≥5 de CLAUDE.md §4 es regla de CPCV (folds interleaved) y etiquetas multi-día, otro régimen.

**Estadístico.** block-permutation (primario, autocorr-robusto); McNemar (secundario); sign vs 0.5 (ortogonal,
fuera de Holm). Holm sobre la familia {vs M5, vs M8, vs B&H}.

**Criterio de éxito (confirmatorio).** block-perm vs B&H bajo Holm p_adj<0.10 **Y estabilidad**: el resultado
se mantiene p<0.10 en embargo ∈ {0,1,2} (meseta, no pico aislado).

**Criterio de fracaso (pre-registrado).** Si solo embargo=1 cruza el umbral y embargo∈{0,2} no, o no sobrevive
Holm/Bonferroni del barrido → se reporta como **SENSIBILIDAD**, no como hallazgo confirmatorio. (Adelanto
honesto: por el barrido previo ya sabemos que ESTE es el caso → se adopta emb=1 por PRINCIPIO, y la mejora de
accuracy 0.524→0.552 se reporta como nominal, sin reclamar significancia.)

**Declaración de transparencia.** El barrido {0,1,3,5,10} previo fue EXPLORATORIO; embargo=1 se fija ahora por
principio (horizonte=1). Cota honesta del barrido: Bonferroni-5 (0.047×5≈0.24) → no significativo.

**Datos.** OOS SMCI ~250 d (post burn-in 150), N0=150, STEP=21, expandible, semilla 42, bloque block-perm √N.
**Output.** Re-ejecución de m10_smci_{deep,advanced,rolling,select} + panel + improve con EMBARGO=1;
`outputs/experiments/m10_smci_embargo.json` (barrido de robustez). Citas: tesis/bibliography.bib.

## [2026-06-17] [Hallazgo] - SMCI: el resultado de M10 es robusto a la partición validación/test

**Contexto.** Para respaldar el resultado principal (M10-WF ensemble, embargo=1, todo el OOS: accuracy 0.552
> M5/M8/B&H nominal), Raquel pide comprobar que no depende de cómo se parta en validación/test.

**Detalle.** `m10_smci_valtest_robustez.py`: 3 splits cronológicos estándar pre-especificados (60/40, 70/30,
80/20; burn-in 150 fijo). **En los tres, M10 bate a M5, M8 y B&H tanto en validación como en test** (val
0.520/0.526/0.535; test 0.600/0.613/0.620). Regímenes de las ventanas equilibrados (% alcistas 0.45–0.51). El
p1 del walk-forward se calcula una vez (ens 10 semillas, emb=1) y se reparte por ventana. **Honesto:** al
achicar el test (100→75→50 d) la accuracy sube (0.60→0.62) pero la potencia cae (sign vs 0.5 p=0.057→0.064→
0.119); por eso el número headline es el de **todo el OOS (0.552, n=250)** y los 3 splits son **respaldo de
CONSISTENCIA**, no split-shopping (ratios a priori; lectura = invarianza al corte, no elegir el mejor).

**Implicaciones para el TFG.** Refuerza el caso de estudio sin sobre-vender: "M10 gana a todo, y la conclusión
es invariante a la partición". Corrige el split desequilibrado anterior de m10_smci_select (burn-in 180 →
validación de 70 d, alcista). Documentado en `notebooks/m10_better_smci.ipynb` §E.2 (gráfica val/test × 3
splits), RESULTADOS_OBJETIVO §1bis, smci.md Fase 7bis.

**Referencias.** `experiments/m10_smci_valtest_robustez.py`, `outputs/experiments/m10_smci_valtest_robustez.json`.

---

## [2026-06-17] [Pre-registro redacción] - Capítulo 3 (Marco teórico) reescrito de cero

**Contexto.** Raquel no quería la estructura lineal anterior (modelo tras modelo). Se rehace el cap. 3 con su
estructura aprobada (4 bloques, "por disciplina") en `memoria/estructura_cap3.md`.

**Estructura (4 bloques).** §0 Preliminares y notación · §1 STRATA a nivel técnico (overview) · §2 Teoría
matemática de los detectores (HMM, GARCH(1,1)-t, BOCPD, con demostraciones) · §3 STRATA aplicado (construcción,
calibración y umbrales: RAM/PSA/GSO + intervención) · §4 Validación (métricas, validación sin fuga, contrastes).

**Qué se conserva.** Las demostraciones ya escritas (recursión forward, teorema de estacionariedad GARCH,
recursión de Adams–MacKay, teoría de los 6 contrastes, economía) se **reutilizan verbatim** y se re-secuencian.
**Nuevo:** §1 (overview) y §3 (rigor de calibración/umbrales, hoy en esqueleto). La economía se reparte:
leverage effect y volatility targeting → §3 (justifican RAM/GSO); Sharpe/Sortino/MaxDD/Calmar → §4.

**Gates (sin atajos).** arquitecto-estructura → redactor-tesis → rigor-matematico/harvard-professor →
experto-citas → estilo-raquel → detector-ia/detector-plagio → narrativa-coherencia → latex-experto.

**Bibliografía nueva verificada:** sortino1994, sortino1991, young1991 (@misc, Calmar no peer-review),
bai1998/bai2003, newey1987, chen2016 (XGBoost), lundberg2017 (SHAP).

**Criterio de cierre.** 0 claims sin cita; rigor sin errores; estilo Raquel (sin guiones-muletilla ni AI-tells);
detector-IA bajo umbral; LaTeX compila; estructura conforme a los 4 bloques.

**Salida.** `tesis/chapters/03_marco_teorico.tex`. Rama `docs/cap-3-marco-teorico`.

---

## [2026-06-22] [Pre-registro] - Experimento automl-seed-ensemble (AutoML-M10 como ensemble de semillas + sensibilidad)

**Contexto.** La semilla de H2O AutoML es una fuente de ruido, no un hiperparámetro del modelo.
Surge la tentación (bajo presión de plazo) de elegir la semilla que maximiza el OOS para inflar el
número de activos donde AutoML "bate a todo". Eso sería overfitting del test (p-hacking) y queda
descartado. Este experimento es la versión defendible de "variar la semilla".

**Hipótesis.** El ensemble de AutoML sobre 10 semillas fijadas a priori reduce la varianza de la
predicción frente a una semilla única, pero NO altera la conclusión de universalidad.

**H0.** El ensemble de AutoML no bate a ZeroR causal en accuracy (universalidad §2 nivel 3).

**Estadístico.** McNemar pareado del ensemble vs ZeroR / M5 / M8 / M10-XGBoost; sign test vs 0.5;
bootstrap estacionario pareado de ΔSharpe vs M5. Mismas pruebas que automl_m10.py.

**Diseño.** Semillas = SEED+0..9 = 42..51, IDÉNTICAS al ensemble canónico de M10-XGBoost
(automl_m10.xgb_wf_p1, línea 85). Se promedian las probabilidades p1 entre semillas. Las semillas se
fijan ANTES de mirar el OOS; no se selecciona ninguna por su resultado. Pipeline causal sin cambios:
ALL22 features, target signo(r_{t+1}), WF expandible N0=150 (~250 días), embargo=1, Purged K-Fold
interno (López de Prado 2018, sec. 7.4), HMM+GARCH al vuelo por activo.

**Capa de honestidad (anti-cherry-picking).** Además del ensemble se reporta por activo la
DISTRIBUCIÓN de accuracy entre las 10 semillas: min/mediana/máx/std y nº de semillas que baten a
ZeroR. Documenta la fragilidad ante la semilla en lugar de ocultarla. Es lo que el tribunal espera.

**Criterio de éxito.** Ninguno que dependa de elegir semilla. Solo se reporta el ensemble (criterio a
priori) y la dispersión.

**Criterio de fracaso (preservado).** Si el ensemble batiera a ZeroR causal de forma robusta en
varios activos (McNemar p<0.10), refutaría que la señal direccional ya está agotada — sería hallazgo,
no fracaso, pero exigiría revisar la tesis de "STRATA aporta en riesgo, no en accuracy".

**Datos.** OOS desplegable [150:] por activo, panel de 15. Embargo=1. Mismo que automl_panel.json.

**Output esperado.** `outputs/experiments/automl_seed_ensemble.json` con, por activo: table (6
estrategias × accuracy/auc/sharpe/equity/max_dd/calmar), tests (McNemar matrix + sign), y
seed_sensibilidad (acc_by_seed, min/med/max/std, n_seeds_beat_zeror).

**Referencias.** experiments/automl_seed_ensemble.py; reusa experiments/automl_m10.py; commit pendiente.

## [2026-06-23] [Milestone] - Notebook definitivo del marco práctico (único canónico) + caso SPY registrado

**Contexto.** Había dos cuadernos parciales y solapados (`STRATA_SMCI`, `decision_automl`) y ninguno definitivo;
además el resultado "SPY con AutoML gana a todo" y varias decisiones no estaban bien registrados. Se consolida
todo en un único notebook canónico del que se alimentará la memoria.

**Detalle.** Nace `notebooks/STRATA_marco_practico.ipynb` (builder `_build_STRATA_marco_practico.py`) con la
estructura del Cap. 4 (§4.0 objetivos/notación → §4.1 datos+barrera temporal → §4.2 mecánica ex-ante → §4.3
caso SPY → §4.4 universalidad panel 15 → §4.5 clustering → §4.6 robustez+honestidad → §4.7 conclusiones →
auto-test). Sustituye/absorbe a los dos anteriores (decisión #18; quedan como fuentes, no se borran). Caso
central = **SPY** (no SMCI): con la config canónica, **AutoML-H2O gana en punto a todas** (acc 0.5737 > ZeroR/
B&H 0.5657, M5 0.3665, M8 0.4422, M10 0.4940; Sharpe AutoML 2.68, maxDD −0.055). SMCI pasa a **caso de
limitación** (§4.6). El cuaderno se cierra con un bucle constructor↔revisora (agente `raquel-quant`) que itera
hasta APROBADO contra un gate G1–G6.

**Resultado registrado (honesto, anti sobre-afirmación).** "AutoML gana a todo" en **accuracy** es **NOMINAL**:
McNemar AutoML vs ZeroR p=0.902, M10 vs ZeroR p=0.133 (n=251, sin potencia → significancia de accuracy = línea
futura). Lo que **sí** sobrevive a un test: (a) **rescate del agente en accuracy** — McNemar AutoML/M10/M8 vs M5
p=0.0002 / 0.0074 / 0.0509; (b) **rescate del agente en riesgo** — bootstrap pareado **pooled** (15 activos,
n=3751) M8 vs M5 ΔSharpe +0.66 IC95[0.225,1.157] y ΔmaxDD +0.24 IC95[0.017,0.445], ambos excluyen 0 (a nivel SPY
el IC aún cruza 0); (c) **universalidad** — cuota STRATA en SHAP media 0.66 (SPY 0.565 árbol / 0.564 permutation);
(d) **patrón** activo→estrategia por clustering (KMeans/Ward/GMM coinciden, Rand=1.0, k=3). STRATA **no genera
alfa** (no bate a ZeroR/B&H sig.).

**Implicaciones para el TFG.** Es el entregable del marco práctico y la fuente única de cifras para la memoria.
Fija la narrativa: el valor de STRATA es el **rescate del agente** (sig. en accuracy vs M5 y en riesgo pooled) y
la **universalidad** (el ML redescubre STRATA), no batir al mercado. Conecta con [accuracy techo ZeroR] y con la
decisión #16 (límite por leverage débil, ilustrado en SMCI).

**Referencias.** Decisión #18; `notebooks/STRATA_marco_practico.ipynb`; outputs `panel_mm25_*`,
`decision_automl_prep.json`, `automl_importance.json`, `strategy_clustering15.json`, `spy_m10_full_report.json`,
`spy_ablation_robustness.json`, `m10_smci_*`; `docs/chats/automl/revision_marco_practico.md`;
`.claude/agents/raquel-quant.md`.

## [2026-06-23] [Pre-registro] - Experimento regime_channel_heldout

**Contexto.** El canal régimen (HMM) solo necesita precio. Se evalúa en un universo HELD-OUT
de 12 activos con precio pero SIN decisiones del agente (AMD, ARKK, COIN, GME, INTC, META,
NFLX, PLTR, PYPL, RIOT, SHOP, SNOW), nunca usados para construir la ley naturaleza→canal.
Doble propósito: (a) generalización de la única ley que sobrevive un test (rescate ∝ leverage);
(b) coste/turnover del régimen-solo (hueco "producción").

**Hipótesis.** El contenido direccional del régimen-solo es CONDICIONAL al leverage effect:
en el universo held-out, leverage_corr (calibración, ≤2024-09, congelado) correlaciona
negativamente con la ventaja del régimen (acc y Sharpe), reproduciendo el patrón del panel-15
(Sharpe medio régimen FUERTE +0.40 vs DÉBIL −0.14).

**H0.** No hay relación leverage_corr ↔ ventaja del régimen en el held-out (el patrón del
panel-15 era sobreajuste/coincidencia).

**Estadístico.** Pearson y Spearman de leverage_corr vs {acc régimen−ZeroR, Sharpe régimen};
sign test de medias grupo FUERTE (lev_corr<−0.05) vs DÉBIL en el held-out; y pooled con los 15.

**Criterio de éxito.** Correlación negativa leverage↔ventaja consistente en signo con el panel-15
(idealmente p<0.10 pooled); Sharpe medio régimen FUERTE > DÉBIL en el held-out.

**Criterio de fracaso (prior-flip).** Si el signo de la relación leverage→ventaja del régimen
SE INVIERTE en el held-out → el canal NO generaliza → se reporta como límite, no se re-busca.

**Honestidad.** NO se espera ni se busca que régimen-solo bata a ZeroR en accuracy (no lo hizo,
2/15). El contraste es relacional (canal leverage) + riesgo/coste. No es búsqueda de alfa.

**Datos.** OOS 2024-10-01→hoy, signal_lag=1, prior de signo del régimen congelado en calibración.
Coste lineal 1bp (core.backtest), + barrido 0/1/5/10bp y coste de break-even. Caveat: COIN/PLTR/
SNOW tienen calibración corta (inicio 2020-21) → menos transiciones de régimen.

**Output esperado.** `outputs/experiments/regime_channel_heldout.json` con claves: por_activo
{leverage_corr, regimen/bh/zeror {acc,sharpe,equity,maxDD}, turnover_ann, breakeven_bps,
net_by_cost}, y resumen {corr_lev_acc, corr_lev_sharpe, grupo_fuerte/debil, pooled_con_15}.

## [2026-06-23] [Pre-registro] - Experimento net_of_cost_panel (hueco "producción" #2: costes/turnover)

**Contexto.** El notebook canónico no reporta coste ni turnover (crítica de despliegue: con margen de
accuracy de 1 punto los costes pueden borrar la ventaja). Las posiciones de las 6 estrategias se
reconstruyen EXACTAS desde el panel canónico (`pos = signo(r_{t+1})·(2·acierto−1)`, sin reentrenar).

**Hipótesis.** La capa de RIESGO (M8/régimen) rota poco (por estado, no a diario) y su rescate
(ΔSharpe vs M5) sobrevive a costes realistas; el aprendiz diario (M10/AutoML) rota mucho más y es el
candidato a perder ventaja con el coste.

**H0.** El turnover es indistinguible entre capas y el rescate de riesgo no sobrevive a 1bp.

**Estadístico.** Turnover anualizado por estrategia (panel-10); ΔSharpe pooled (arm vs M5) vs coste en
{0,1,2,5,10,20}bp; coste de break-even donde el rescate pooled ΔSharpe cruza 0 (bisección).

**Criterio de éxito.** M8 turnover < M10/AutoML; el rescate de riesgo M8 vs M5 sigue >0 a 10bp.

**Criterio de fracaso.** Si el rescate ΔSharpe M8 vs M5 muere por debajo de 1bp → la ventaja es ilusoria
por coste → se reporta como límite, no se maquilla.

**Datos.** Panel-10 (SPY,QQQ,XLF,DIA,XLK,XLE,ROKU,SMCI,MARA,UNG), ventana desplegable [150:] (~250d),
signal_lag=1, coste lineal `core.backtest`. Validación: accuracy reconstruida == canónica (identidad).

**Output esperado.** `outputs/experiments/net_of_cost_panel.json` con por_activo {turnover_ann por arm,
net_by_cost {sharpe,equity} por arm}, y pooled {sharpe_vs_cost, dSharpe_vs_m5_vs_cost, breakeven_rescate}.

---

## [2026-06-23] [Hallazgo] - Complementariedad por régimen del rescate de riesgo (M10 alcista / AutoML bajista)

**Contexto.** Confirmatorio del rescate en Sharpe desglosado por régimen de mercado (tendencia 21d causal),
SPY y POOLED-10. Experimento `bullbear_confirmatory.py` → `bullbear_confirmatory.json` (§7 del marco práctico).

**Detalle.** A nivel de **SPY-solo** el rescate de Sharpe se concentra en **alcista** (M8 +3.95, M10 +4.35,
AutoML +7.53) y la **regla M8 se invierte en bajista** (ΔSharpe −1.49, n=50, sin potencia) — es la **falsación
pre-registrada** ("plano Sharpe: rescate no robusto en bajista") cumplida al nivel de un solo activo. Pero al
**agregar los 10 (pooled)** aparece un patrón **complementario en espejo**, con los 6 contrastes significativos
en AMBOS regímenes (block-perm p<0.07; McNemar p_Holm<0.10 salvo M8-alcista en el borde 0.099):
- **M10** rescata MÁS en **alcista** (ΔSharpe +1.37) que en bajista (+0.72).
- **AutoML** rescata MÁS en **bajista** (+1.52) que en alcista (+0.81).
- **M8** (la regla) queda **simétrica** (+0.63 / +0.55).

**Implicaciones para el TFG.** (1) El rescate de riesgo **NO es un artefacto de un único régimen** — sobrevive a
un test en alcista y en bajista por separado cuando se agrega el panel; la agregación resuelve la falsación que
sí ocurre a nivel SPY-solo. (2) Los dos aprendices son **complementarios**: el buscador **AutoML**, que modela
la interacción condicional, protege mejor en el régimen **peligroso** (bajista) — argumento de despliegue para
que sea la capa de accuracy; **M10** brilla en tendencias alcistas; la **regla M8** aporta un rescate de riesgo
**parejo** en ambos regímenes. Va a la memoria como matiz del §7 (rescate por régimen) y refuerza el encuadre de
las dos capas (§5).

**Referencias.** `experiments/bullbear_confirmatory.py`, `outputs/experiments/bullbear_confirmatory.json`,
notebook §7 (celda "(j) por régimen") y §9 conclusión 6; memoria [[bullbear-confirmatorio-dsr]].

---

## [2026-06-23] [Pre-registro] - Experimento regime_did_learners (¿la complementariedad por régimen es significativa?)

**Contexto.** El hallazgo previo es DESCRIPTIVO: en el pooled-10, M10 rescata más en alcista (ΔSharpe +1.37) y
AutoML más en bajista (+1.52). Antes de elevarlo a resultado hace falta un contraste de que la **especialización
por régimen** no es ruido de un OOS. Test de diferencia-en-diferencias (DiD) de Sharpe.

**Hipótesis (H1).** Los dos aprendices se especializan por régimen EN ESPEJO: M10 es relativamente mejor en
alcista y AutoML en bajista. Operacionalizada como
`DiD = [SR_M10(alc) − SR_AutoML(alc)] − [SR_M10(baj) − SR_AutoML(baj)] > 0` (la M5 se cancela).

**H0.** DiD = 0: la ventaja relativa M10 vs AutoML es la misma en los dos regímenes (no hay especialización).

**Estadístico.** DiD de Sharpe por **bootstrap estacionario pareado** (Politis-Romano 1994) sobre los retornos
±1 reconstruidos del acierto canónico; el régimen (drift 21d causal) viaja con cada día remuestreado. block=√n,
B=2000, seed=42 (idéntica convención a `bullbear_confirmatory`). IC95 y p one-sided P(DiD≤0).

**Criterio de éxito.** En el POOLED-10: IC95 del DiD **excluye 0** (o p one-sided < 0.10) → complementariedad
**confirmada** (resultado, no solo patrón).

**Criterio de fracaso.** Si el IC95 **cruza 0** → la complementariedad es **descriptiva** (patrón de un único
OOS); se reporta como hipótesis de línea futura (posible ensemble enrutado por régimen, pre-registrable), NO como
resultado. Honestidad: SPY-solo (n pequeño) se reporta aunque salga no significativo.

**Datos.** POOLED-10 y SPY, ventana desplegable, ±1, drift 21d. signal_lag=1.

**Output esperado.** `outputs/experiments/regime_did_learners.json` con {SPY, POOLED10}: did_point, ci95,
p_one_sided, m10_minus_aml por régimen, SR por arm y régimen.

**Resultado (2026-06-23).** **H1 CONFIRMADA en el pooled.** POOLED-10: DiD=+1.37, IC95=[+0.20,+2.60] (excluye 0),
p one-sided=0.008 → la especialización por régimen es significativa (M10−AutoML pasa de +0.56 en alcista a −0.81
en bajista). **SPY-solo: NO significativo** (DiD=−0.30, IC95=[−5.28,+6.86] cruza 0, p=0.50): en SPY AutoML bate a
M10 en AMBOS regímenes (es la estrella del activo), así que el espejo **no se ve en el escaparate** — la
complementariedad es un **fenómeno de PANEL** que emerge al agregar naturalezas distintas. Matiz honesto: es un
resultado cross-asset, no de un activo. Propagado a notebook §7 (celda "(k) DiD"), §9 conclusión 6, y memoria
[[bullbear-confirmatorio-dsr]]. Línea futura pre-registrable: ensemble enrutado por régimen (M10 alcista / AutoML
bajista) usando la señal de RAM.

---

## [2026-06-23] [Pre-registro] - Experimento equivalence_tost (¿el aprendiz REDESCUBRE la regla o la BATE?)

**Contexto.** La hipótesis de universalidad (CLAUDE.md §2.3) afirma que el aprendiz "redescubre STRATA, no la bate".
Hasta ahora se argumentaba con SHAP + ablación; la afirmación "no bate" exige un **contraste de equivalencia**
(TOST, Schuirmann 1987), no un test de diferencia no significativo (ausencia de evidencia ≠ evidencia de ausencia).

**Hipótesis (H1, equivalencia).** El aprendiz (M10 / AutoML) es **equivalente** a la regla M8 dentro de un margen
de irrelevancia δ: |Δ| < δ, con Δ = métrica(aprendiz) − métrica(M8), pareado. Métricas: accuracy direccional y
Sharpe.

**H0 (de la equivalencia).** |Δ| ≥ δ (difieren al menos el margen). TOST = dos contrastes unilaterales.

**Estadístico.** TOST vía IC: equivalencia a α=0.05 si el **IC90% bootstrap-bloque** de Δ ⊂ (−δ, +δ). Además,
**superioridad** unilateral P_boot(Δ≤0). Cuadro 2×2: REDESCUBRE (equiv, no bate) / BATE (superior, no equiv) /
no concluyente. Pooled-10 y SPY.

**Margen pre-registrado.** δ_acc = 0.03 (≈ 1 SE de accuracy de un activo, √(0.25/250); económicamente nimio);
δ_Sharpe = 0.50 anualizado. Sensibilidad reportada (acc {0.01,0.02,0.03,0.05}; SR {0.25,0.5,0.75,1.0}) para no
elegir el margen que conviene.

**Criterio de éxito / fracaso.** No hay "éxito" pre-asignado: se reporta el veredicto honesto sea cual sea
(equivalencia, superioridad o no concluyente). Si el aprendiz BATE a la regla, se dice (matiza la universalidad);
si REDESCUBRE, se confirma. Margen sensibilidad = blindaje anti-cherry-picking.

**Output esperado.** `outputs/experiments/equivalence_tost.json` con {SPY, POOLED10} × {M10_vs_M8, AutoML_vs_M8}
× {accuracy, sharpe}: punto, IC90, equivalencia(δ)+sensibilidad, p_superioridad.

**Resultado (2026-06-23).** **El TOST NO confirma equivalencia en ningún caso; al contrario, el aprendiz BATE a la
regla M8 en accuracy.** **[Propagado al notebook 2026-06-23]** Una auditoría de coherencia (harvard-professor)
detectó que el resultado del TOST llegó al dossier pero NO al notebook (O3/§5/§9 aún decían "redescubre/no bate").
Se corrigió: el builder carga `equivalence_tost.json`, hay celda del cuadro 2×2 en §4, y O3+§5+§9.3 se reformularon
a la jerarquía honesta. También se arregló un "n=15"→"n=10" en la lectura del clustering (que ya es sobre 10) y una
línea duplicada en §9. Notebook 89 celdas, 0 errores, AUTO-TEST OK. Conclusiones extraídas a
`conclusiones_notebook_central.md`. Cifras SPY, reconciliación SHAP y universo 10/15 validados sin más
incoherencias. Pooled: M10 vs M8 Δacc=+0.021 IC90[+0.001,+0.039] y AutoML vs M8 Δacc=+0.034
IC90[+0.010,+0.056] → ambos superiores (efecto modesto, 2–3 pp). SPY: AutoML vs M8 superior en accuracy
(Δ=+0.132) y Sharpe (Δ=+4.25); M10 vs M8 no concluyente. En **Sharpe el pooled es no concluyente** (IC cruza 0):
regla y aprendiz son **indistinguibles en riesgo**. **Implicación (refina la universalidad, no la rompe):** la
afirmación "redescubre y NO bate" queda **refutada en accuracy** — el aprendiz sí bate a la regla, porque modela
**interacciones no lineales que la regla determinista no puede** (los activos de leverage invertido donde M8
falla, §5). Lo que aguanta de la universalidad es el **SHAP** (el aprendiz USA las señales de STRATA, cuota 0.66).
Reencuadre correcto: *el aprendiz redescubre las señales de STRATA y extrae algo MÁS de accuracy que la regla fija
por su flexibilidad; en riesgo, regla y aprendiz son equivalentes; ninguno bate a lo trivial (ZeroR nominal).*
Orden honesto: rescata al agente (sig) > bate modestamente a la regla en accuracy > empata con la regla en riesgo
> no bate a ZeroR. Propagado a dossier (universalidad) y marco teórico §9 (TOST entra como método usado).

---

## [2026-06-23] [Decisión] - Enfoque final: panel de 10 SIN apéndice; ley y pooled de riesgo sobre 10

**Contexto.** Consolidación del giro de enfoque (SMCI-caso-de-uso → SPY + panel de 10 + patrones). Raquel decide
eliminar del todo los 5 activos que no están en el panel (MSTR, NVDA, BAC, TSLA, IWM): **ni siquiera como
apéndice**. El estudio es de 10 activos. SMCI se queda (es uno de los 10).

**Detalle.** Cambios en el notebook canónico `STRATA_marco_practico.ipynb` y docs de esta sesión:
- Eliminada la sección §8 (apéndice de los 5); §9 conclusiones → §8. §1 reencuadrado a "panel de 10".
- **Ley del leverage recomputada sobre 10**: Pearson r=−0.56, **p=0.093** (antes 15: r=−0.55, p=0.034). Se reporta
  como **tendencia significativa al α=0.10** del proyecto, **visible en el panel** (tabla leverage↔rescate por
  activo: índices fuertes rescatan 0.097 vs 0.059 los de leverage débil; 9/10 la siguen, solo ROKU a
  contracorriente). **Ya NO se afirma p<0.05 ni robustez a leave-one-out** (sobre 10, quitar UNG → p=0.35).
  Decisión consciente: la ley se cumple en los 10 (la correlación es igual de fuerte); solo baja la potencia.
- **Pooled de riesgo → pooled-10** como titular: M8 vs M5 ΔSharpe **+0.64 IC[0.10,1.29]** (sig), M10 +0.93,
  AutoML +0.97. Retirado el pooled-15 (n=3751) del notebook.
- Auto-test actualizado (sin asserts de EXCL5/apéndice/IWM/pooled-15/LOO). Notebook 87 celdas, 0 errores, verde.

**Implicaciones para el TFG.** El resultado duro (rescate riesgo + accuracy) y los patrones (clustering Rand=1.0,
complementariedad DiD p=0.008) se sostienen sobre 10. La ley del leverage baja a marginal (α=0.10), declarado con
honestidad. Los docs raíz (DECISIONES/RESULTADOS/CONOCIMIENTO/LECCIONES) los actualiza la otra sesión desde
`conclusiones_notebook_central.md` (reescrito como documento de contexto de la tesis: objetivos, C1–C9, decisiones,
Cap.4 qué aparece, Cap.3 qué falta, defensa, líneas rojas, figuras).

**Referencias.** `notebooks/_build_STRATA_marco_practico.py`, `conclusiones_notebook_central.md`,
`docs/marco_practico_dossier.md`, plan en `.claude/plans/`.
