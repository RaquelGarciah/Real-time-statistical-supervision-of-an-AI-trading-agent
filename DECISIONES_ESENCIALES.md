# Decisiones esenciales — vivas a 2026-06-07

Las 12 decisiones que sobreviven al pivot final del proyecto anterior. Cada una con: **Qué + Por qué + Dónde está justificada en el archivo + Estado**. Léelas antes de cuestionar cualquier hiperparámetro.

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

**Por qué.** El protocolo de medición dual (`same-day` para sanity + `causal` para reportar) reveló que las variantes A, B, D y el GSO relativo introducían look-ahead. Override C + régimen filtered es la **única** combinación que sobrevive al test causal con Sharpe positivo (+0.66 sobre SPY).

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
- RAM: **low 0.20 / medium 0.40 / high 0.70** (defaults razonables, robustos por construcción del score como probabilidad).
- PSA: **P95 / P99** sobre el periodo de calibración.
- GSO: **P95 / P99** sobre el periodo de calibración.

Guardados en `cache/models/strata_thresholds.json`.

**Por qué.** Calibración ex-ante = sin look-ahead. Los percentiles sobre 24 años son estables. Defendible: 6 hiperparámetros fijos vs los ~4500 splits de un XGBoost.

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
| 8 | Umbrales fijos ex-ante (RAM 0.2/0.4/0.7, PSA/GSO P95/P99) | STRATA | Viva |
| 9 | M7 reduce = PSA cp_prob_delta + hazard 1/60 | STRATA | Viva (control de daños) |
| 10 | M10 con CPCV-within-OOS, embargo 5, n_splits 6 | Validación | Viva |
| 11 | Pre-registro en BITACORA obligatorio | Metodología | **Viva, obligatoria** |
| 12 | Caché `cache/agent/` git, `cache/llm/` local | Reproducibilidad | Viva |

---

## Cómo consultar si una nueva idea contradice una de estas decisiones

Pregunta a `@asesor-historico`: *"Voy a usar [feature/método/setup]. ¿Contradice alguna decisión esencial?"*. El asesor lee este fichero + BITACORA y responde con cita.

Si una decisión necesita revisarse, hay que: (i) documentar en BITACORA el motivo, (ii) auditar con `@rigor-matematico` el coste de cambiarla, (iii) actualizar este fichero. **Nunca cambiar silenciosamente.**
