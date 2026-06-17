# Decisiones esenciales — vivas a 2026-06-17

Las 12 decisiones que sobreviven al pivot final del proyecto anterior, **más 4 del pivot a caso de estudio
SMCI (#13–#16, 2026-06-17)**. Cada una con: **Qué + Por qué + Dónde está justificada en el archivo + Estado**. Léelas antes de cuestionar cualquier hiperparámetro.

Decisiones **descartadas** explícitamente: Nemotron como LLM principal (reemplazado por DeepSeek V3 / gpt-oss-120b); NVDA como caso central (vuelto a SPY por leverage effect); GSO relativo con look-ahead (reemplazado por GSO absoluto causal); plan original de 9 configuraciones M1..M9 (reducido a 3 canónicas M5/M8/M10 más M1/M2 como baselines).

---

## 1. SPY-only en el cuerpo central del TFG

**Qué.** El caso central de validación es SPY. El panel multi-activo de 10 tickers va como apéndice de robustez, no como experimento principal.

**Por qué.** El **leverage effect** (Black 1976; Christie 1982) implica que en índices agregados existe correlación negativa fuerte (~−0.7) entre retornos y volatilidad. Sobre SPY, esto hace que el régimen de alta volatilidad coincida con régimen bajista, y el HMM de 3 estados actúa **implícitamente como detector direccional** (Crisis ≈ caídas, Calma ≈ subidas). La política RAM ("Crisis penaliza long agresivo") es entonces empíricamente justificable. En stocks individuales con leverage débil o positivo, esa asunción se rompe — y por tanto STRATA no es académicamente válido sobre ellos.

**Dónde está justificada.**
- `_archivo_proyecto_anterior/docs/marco_teorico.md` (sección leverage effect)
- `_archivo_proyecto_anterior/docs/hallazgos_strata.md` §3
- `_archivo_proyecto_anterior/BITACORA.md` entrada 2026-05-19 (pivot del NVDA-como-principal)

**Estado.** Viva. NO cuestionar.

---

## 2. OOS unificado 2024-10-01 → cierre del TFG

**Qué.** Evaluación oficial sobre `[2024-10-01, hoy]`. Mismo periodo para todas las estrategias.

**Por qué.** Posterior al rango estimado del **cutoff de conocimiento de DeepSeek V3** (julio-octubre 2024). Garantiza que el LLM no ha visto los datos del OOS durante su entrenamiento. Sin esta barrera, los resultados del agente estarían contaminados por look-ahead específico del LLM.

**Dónde está justificada.** `_archivo_proyecto_anterior/BITACORA.md` entradas 2026-05-14 / 2026-05-15.

**Estado.** Viva.

---

## 3. Calibración 2000-01-01 → 2024-09-30

**Qué.** HMM, GARCH y BOCPD calibrados sobre ese rango único, una vez, sin re-entrenamiento durante el OOS.

**Por qué.** 24 años de datos diarios. Suficiente para estabilidad estadística. No solapa con el OOS (barrera temporal estricta). El **test de Bai-Perron** sobre la calibración confirma estabilidad estructural suficiente para una sola calibración.

**Dónde está justificada.** `_archivo_proyecto_anterior/BITACORA.md` (Fase 1).

**Estado.** Viva. Los pickles entrenados están en `cache/models/`.

---

## 4. HMM features = `[log_return, realized_vol_21d]`

**Qué.** Features de entrada del HMM: log-retorno diario y volatilidad realizada anualizada de 21 días. **No** `log(VIX)`.

**Por qué.** La volatilidad realizada es observada, no implícita; coherente con la formulación matemática del HMM gaussiano. El VIX implícito introduce información forward-looking del mercado de opciones que no encaja con la estructura del modelo de régimen.

**Dónde está justificada.** `_archivo_proyecto_anterior/docs/marco_teorico.md` (sección HMM features) + BITACORA 2026-05-?.

**Estado.** Viva.

---

## 5. M8 = STRATA `override C` + régimen `filtered` + `signal_lag=1` causal

**Qué.** La configuración canónica de la regla a mano es:
- Modo: `override C` (sustituye sizing por el más próximo dentro de la banda admisible).
- Régimen: probabilidades `filtered` (no smoothed) — solo información hasta `t`.
- `signal_lag = 1`: la posición en `t` multiplica al retorno en `t+1`.

**Por qué.** El protocolo de medición dual (`same-day` para sanity + `causal` para reportar) reveló que las variantes A, B, D y el GSO relativo introducían look-ahead. Override C + régimen filtered es la **única** combinación que sobrevive al test causal con Sharpe positivo. **Cifra canónica actual: +0.67** (K=3, τ=0.5, notebook canónico 2026-06-08); el +0.66 citado en la justificación del proyecto anterior corresponde a K sin fijar y τ=0.40 — misma decisión, cifra actualizada.

**Dónde está justificada.**
- `_archivo_proyecto_anterior/docs/hallazgos_strata.md` (techo supervisión, protocolo dual)
- `_archivo_proyecto_anterior/docs/known_issues.md` (bug `signal_lag`)
- `_archivo_proyecto_anterior/BITACORA.md` entradas 2026-05-?.

**Estado.** Viva. Look-ahead corregido y verificable en código con `test_no_leakage.py`.

---

## 6. Prior RAM data-driven por activo, re-signado por signo de medias calibradas

**Qué.** El prior RAM ("Crisis penaliza long" o "Crisis permite long") no se hardcodea. Se calcula del signo de las medias de retorno por régimen en el periodo de calibración, **por activo**.

**Por qué.** En SPY/BAC/XLE las medias en Crisis son negativas (leverage estándar): prior es `Crisis ⇒ short permitido`. En NVDA las medias en Crisis son **positivas** (leverage no estándar): prior es `Crisis ⇒ long permitido`. Hardcodear el signo rompe la lógica de RAM en activos con leverage débil o invertido.

**Dónde está justificada.**
- `_archivo_proyecto_anterior/docs/hallazgos_strata.md` (per-asset prior)
- `_archivo_proyecto_anterior/docs/chats/organize_notebook.md`
- `_archivo_proyecto_anterior/BITACORA.md` (NVDA per-asset).

**Estado.** Viva. Generalizable al panel.

---

## 7. Política RAM simétrica con leverage

**Qué.** En activos con leverage estándar:
- **Calma** ⇒ long permitido; short es inconsistente (`RAM = P(Calma)` si agente short).
- **Estrés** ⇒ ambos permitidos (sin penalización).
- **Crisis** ⇒ short permitido; long es inconsistente (`RAM = P(Crisis)` si agente long).

El `RAM_score` es **literalmente una masa de probabilidad** sobre regímenes donde la acción es inconsistente.

**Por qué.** Simétrica con el leverage effect. El score tiene interpretación probabilística clara y comparable entre activos.

**Dónde está justificada.** `_archivo_proyecto_anterior/docs/marco_teorico.md` (política RAM) + CLAUDE.md original §2.

**Estado.** Viva.

---

## 8. Umbrales STRATA fijos calibrados ex-ante

**Qué.**
- RAM: **low 0.25 / medium (τ) 0.50 / high 0.70**. El gate operativo que dispara M7 (reduce) y M8 (override) es `medium = τ = 0.5` (regla de mayoría: el régimen contrario es el más probable). `low` y `high` solo re-etiquetan severidad — no afectan al P&L (BITACORA [2026-06-09]). Se mantiene el blindaje dual: el McNemar se reporta con τ=0.5 **y** con el default conservador 0.40 para demostrar que el rescate no depende del umbral elegido.
- PSA: **P95 / P99** sobre el periodo de calibración.
- GSO: **P95 / P99** sobre el periodo de calibración.

Guardados en `cache/models/strata_thresholds.json`.

**Por qué.** El RAM score es cuasi-binario (masa en ~0 y ~1, valle vacío en el medio): el acierto direccional es plano para cualquier τ∈[0.3, 0.9] ≈ 0.556 en calibración, lo que hace τ=0.5 identificable sin ajuste. El cruce isotónico fino degeneraba por confound del drift (SPY sube 54.4% del tiempo). τ=0.5 tiene varianza de estimación cero y es la regla de mayoría natural. Calibración ex-ante = sin look-ahead. Defendible: los umbrales fijos de STRATA son estables por construcción; el umbral aprendido de XGBoost no (estabilidad temporal §11).

**Dónde está justificada.** `_archivo_proyecto_anterior/docs/decisiones.md` + `cache/models/strata_thresholds.json`.

**Estado.** Viva.

---

## 9. M7 reduce = PSA `cp_prob_delta` + hazard 1/60

**Qué.** El modo `reduce` (M7) usa PSA con feature `cp_prob_delta` (incremento de probabilidad de cambio de régimen, no la prob absoluta) y prior con hazard rate `1/60`.

**Por qué.** Variantes con `cp_prob` absoluta sobre-disparaban en regímenes ya cambiados. `cp_prob_delta` detecta el momento del cambio, no el estado posterior. El hazard `1/60` se calibró por sensibilidad sobre el periodo de calibración.

**Dónde está justificada.** `_archivo_proyecto_anterior/BITACORA.md` línea 1401 + `docs/decisiones.md`.

**Estado.** Viva. Nota: M7 da Sharpe causal negativo en SPY (−0.95); útil como **control de daños**, no como rescate. Mantener para ablación pero no proponer como rescate principal.

---

## 10. M10 validado con CPCV-within-OOS

**Qué.** XGBoost meta-learner sobre 22 features, validado con Combinatorial Purged Cross-Validation **dentro** del OOS unificado:
- `n_splits = 6, n_test_splits = 2` → 15 folds combinatorios.
- `embargo = 5` días entre train y test.
- `t1 = índice.shift(-1)` (etiqueta del día t se asocia al retorno t+1).
- Hiperparámetros XGBoost fijos pre-registrados.

**Por qué.** López de Prado (2018, sec. 7.4). Es la metodología defendible para series temporales financieras con muestra pequeña (~400 días). Evita el sesgo de KFold y el desperdicio de datos de un walk-forward simple.

**Dónde está justificada.** `_archivo_proyecto_anterior/docs/chats/need_mathematic_rigor.md` (diseño completo) + BITACORA pre-registro 2026-06-02.

**Estado.** Viva.

---

## 11. Pre-registro obligatorio en BITACORA antes de mirar resultados

**Qué.** Todo experimento nuevo se pre-registra en BITACORA con:
- Hipótesis falsable explícita.
- Hipótesis nula H0.
- Estadístico de contraste.
- Criterio de éxito numérico (p<α, IC contiene/no contiene cero, Δ métrica > umbral).
- Criterio de fracaso (regla `prior-flip` o equivalente).

**Solo después** se ejecuta el experimento. El JSON output incluye los hash del pre-registro.

**Por qué.** Blindaje contra acusaciones de p-hacking. El tutor exige rigor — esto es lo que el rigor parece desde fuera.

**Dónde está justificada.** `_archivo_proyecto_anterior/docs/chats/need_mathematic_rigor.md` ("Criterio de éxito pre-registrado en BITACORA antes de mirar resultados").

**Estado.** Viva. **Obligatoria en el nuevo proyecto sin excepción** — y auditada por `@rigor-matematico` antes de cada ejecución.

---

## 12. Caché por activo: `cache/agent/` en git, `cache/llm/` solo local

**Qué.**
- `cache/agent/<TICKER>/<TICKER>_<date>.json` = decisión final del Portfolio Manager por (activo, fecha). **Versionado en git.**
- `cache/llm/<sha256>.json` = inferencia individual de cada personalidad por prompt. **No versionado** (156M, ruidoso); se preserva en disco local.
- `cache/models/` = HMM/GARCH/BOCPD pickles + thresholds + calibración. **Versionado en git.**

**Por qué.** Reproducibilidad de backtest = `cache/agent/`. Reproducibilidad de detalle (qué dijo cada personalidad un día concreto) = `cache/llm/` local. Modelos calibrados = `cache/models/` para que cualquier máquina arranque sin recalibrar 24 años.

**Dónde está justificada.** `_archivo_proyecto_anterior/BITACORA.md` (estrategia de caché Fase 1) + `CLAUDE.md` §8.

**Estado.** Viva.

---

---

# Decisiones del pivot a caso de estudio de UN activo (2026-06-17)

El tutor pide centrar el TFG en **un activo** donde **M10 (o variante) bata en accuracy a M5, M8 y B&H**.
Estas decisiones salen de esa búsqueda, toda pre-registrada en BITACORA y auditada (@rigor-matematico,
@experto-citas). Conviven con las 12 anteriores; donde matizan a una, se indica.

## 13. Caso de estudio = SMCI (benchmark B&H justo)

**Qué.** El activo del caso de estudio es **SMCI**. B&H ≈ 0.48 (≈ moneda) → benchmark **justo** (el tribunal
no puede tumbarlo con "una estrategia trivial gana", como sí pasa en SPY donde B&H ≈ 0.57).

**Por qué.** Barriendo el panel de 10, **SMCI es el ÚNICO activo donde acc(M10) > M5, M8 y B&H** a la vez
(nominal). No es casualidad: el agente LLM es **corto-sesgado en los 10 activos** (71–100 %); en los que caen
(B&H batible) el agente ya acierta yendo corto → M10 no se separa; en los que suben, M10 rescata pero B&H
gana. La casilla "activo cae + agente equivocado" está **vacía**. SMCI es el caso umbral.

**Dónde está justificada.** `notebooks/m10_better_smci.ipynb` §F/§F.2; `outputs/experiments/panel_intervention_scan.json`;
BITACORA 2026-06-16.

**Estado.** Viva. Matiza la decisión #1 (SPY sigue siendo el caso central del *método*; SMCI es el caso de
estudio que pide el tutor para "batir a todo en accuracy").

## 14. M10 desplegable = walk-forward ensemble (no CPCV)

**Qué.** El M10 **operativo/desplegable** es **walk-forward expandible** (burn-in 150, reentreno mensual),
**ensemble de 10 semillas** sobre las 22 features STRATA, umbral 0.5. El **M10-CPCV** (decisión #10) se
conserva **solo como contraste** (ve bloques futuros → no desplegable; en SMCI da 0.448, peor).

**Por qué.** El ensemble, a **igual accuracy** que la base (0.552 con embargo=1), mejora Sharpe (→1.84) y
equity (→3.24×) por reducción de varianza (criterio de Raquel: a igual accuracy, Sharpe/equity cuenta). Las
palancas probadas **NO** mejoran la accuracy y se descartan: tuning en validación (sobreajuste, validación≠test),
features de señal real (momentum/vol-rel/racha), recencia, triple-barrier, modelos por régimen, stacking
M5→M10, voting y abstención condicional.

**Dónde está justificada.** `notebooks/m10_better_smci.ipynb` §A–§D; `experiments/m10_smci_{deep,advanced}.py`;
BITACORA 2026-06-16.

**Estado.** Viva.

## 15. Embargo = 1 en el walk-forward desplegable (no 5)

**Qué.** En el walk-forward de M10 el embargo es **1 día**, no 5.

**Por qué.** Es validación **rolling-origin** (Tashman 2000), no CPCV: el test es siempre futuro → no hay
solape bidireccional que motive el embargo. La etiqueta tiene **horizonte 1** (y_t=1[r_{t+1}>0]) → purga = 1
(López de Prado 2018 §7.4). El **embargo≥5 de la decisión #10 / CLAUDE.md §4 es regla de CPCV** (folds
interleaved, etiquetas multi-día), otro régimen. Validez con hueco mínimo bajo residuos no correlados:
Bergmeir, Hyndman & Koo (2018). Sube la accuracy de SMCI **0.524 → 0.552** (nominal). **La significancia NO
sobrevive** (el p=0.047 vs B&H es un pico aislado en emb=1; Bonferroni-5 = 0.28) → se reporta como
sensibilidad; el embargo=1 se elige **por principio**, no por el p-valor.

**Dónde está justificada.** `notebooks/logic_esential.ipynb` §14b; `tesis/bibliography.bib` (tashman2000,
bergmeir2018, lopezdeprado2018, burman1994, racine2000, bergmeir2012); BITACORA 2026-06-17; auditoría
@rigor-matematico + @experto-citas 2026-06-17.

**Estado.** Viva. Matiza la decisión #10 (embargo 5 sigue para CPCV; 1 para el WF desplegable).

## 16. Límite honesto: STRATA rescata donde el agente discrepa del régimen

**Qué.** STRATA/M10 aporta valor (bate al agente) **solo donde el agente va a contracorriente de un régimen
que acierta**. En SPY ocurre (M10 vs M5 McNemar p=0.0005, leverage effect fuerte). En SMCI **no**: el agente
ya está 95 % corto (alineado con el régimen) → STRATA interviene el 3 % → M5/M8/M10 son la misma apuesta corta
y ninguno se separa. Ningún M10 desplegable bate a B&H/M5/M8 de forma **significativa** en SMCI.

**Por qué importa.** Define dónde funciona el método (su frontera), coherente con el leverage effect (decisión
#1). La ventaja de M10 sobre B&H en SMCI es **sesgo a corto en un activo que cae**, no habilidad direccional
fina (se reporta con el benchmark "siempre-corto" al lado).

**Dónde está justificada.** `notebooks/m10_better_smci.ipynb` §F/§G; `experiments/panel_intervention_scan.py`,
`experiments/m10_smci_rolling.py`; BITACORA 2026-06-16.

**Estado.** Viva (hallazgo de cierre, honesto).

---

## Tabla resumen

| # | Decisión | Categoría | Estado |
|---|---|---|---|
| 1 | SPY-only cuerpo central | Universo | Viva |
| 2 | OOS 2024-10-01 → cierre | Universo | Viva |
| 3 | Calibración 2000-01-01 → 2024-09-30 | Universo | Viva |
| 4 | HMM features `[log_ret, realized_vol_21d]` | Modelo | Viva |
| 5 | M8 = override C + filtered + signal_lag=1 | STRATA | Viva |
| 6 | Prior RAM data-driven por activo | STRATA | Viva |
| 7 | Política RAM simétrica con leverage | STRATA | Viva |
| 8 | Umbrales fijos ex-ante (RAM 0.25/τ=0.5/0.70, PSA/GSO P95/P99; blindaje dual τ=0.5 y 0.40) | STRATA | Viva |
| 9 | M7 reduce = PSA cp_prob_delta + hazard 1/60 | STRATA | Viva (control de daños) |
| 10 | M10 con CPCV-within-OOS, embargo 5, n_splits 6 | Validación | Viva |
| 11 | Pre-registro en BITACORA obligatorio | Metodología | **Viva, obligatoria** |
| 12 | Caché `cache/agent/` git, `cache/llm/` local | Reproducibilidad | Viva |
| 13 | Caso de estudio = SMCI (B&H justo ≈0.48; único que bate a todo nominal) | Universo | Viva |
| 14 | M10 desplegable = WF ensemble 10 semillas (CPCV solo contraste) | Validación | Viva |
| 15 | Embargo = 1 en el WF desplegable (5 solo para CPCV) | Validación | Viva |
| 16 | Límite: STRATA rescata donde agente discrepa del régimen (SMCI no) | STRATA | Viva |

---

## Cómo consultar si una nueva idea contradice una de estas decisiones

Pregunta a `@asesor-historico`: *"Voy a usar [feature/método/setup]. ¿Contradice alguna decisión esencial?"*. El asesor lee este fichero + BITACORA y responde con cita.

Si una decisión necesita revisarse, hay que: (i) documentar en BITACORA el motivo, (ii) auditar con `@rigor-matematico` el coste de cambiarla, (iii) actualizar este fichero. **Nunca cambiar silenciosamente.**
