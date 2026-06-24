# Decisiones esenciales — vivas a 2026-06-17

> **⚠️ ENFOQUE ACTUAL (2026-06-24) — fuente de verdad práctica: [`MARCO_PRACTICO_CONTEXTO.md`](MARCO_PRACTICO_CONTEXTO.md).**
> Universo = **panel de 10** (SPY central + QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG), elegido **ex ante por
> naturaleza, SIN apéndice**; los demás activos no existen en el TFG. SMCI es **uno de los 10**, no el caso central.
> Riesgo = **pooled-10** (M8 vs M5 ΔSharpe +0,64). Notebook canónico = `STRATA_marco_practico.ipynb`. Decisiones,
> jerarquía de valor y cifras canónicas, en CONTEXTO. **Lo que abajo trate a SMCI como caso central (#13–#16) o
> hable de "10 de 15 / apéndice" está obsoleto** (queda como registro histórico del pivot).

Las 12 decisiones que sobreviven al pivot final del proyecto anterior, **más 4 del pivot a caso de estudio
SMCI (#13–#16, 2026-06-17)**. Cada una con: **Qué + Por qué + Dónde está justificada en el archivo + Estado**. Léelas antes de cuestionar cualquier hiperparámetro.

Decisiones **descartadas** explícitamente: Nemotron como LLM principal (reemplazado por DeepSeek V3 / gpt-oss-120b); NVDA como caso central (vuelto a SPY por leverage effect); GSO relativo con look-ahead (reemplazado por GSO absoluto causal); plan original de 9 configuraciones M1..M9 (reducido a 3 canónicas M5/M8/M10 más M1/M2 como baselines).

---

## Mapa de documentos del proyecto (para no perder el hilo)

Dónde vive cada cosa, de lo general a lo concreto:

| Documento | Para qué | Estado |
|---|---|---|
| **`CLAUDE.md`** | Constitución del proyecto (qué es STRATA, reglas, workflow) | Vivo |
| **`DECISIONES_ESENCIALES.md`** (este) | Las 16 decisiones vivas con su porqué | Vivo |
| **`decisiones_respaldadas_literatura.md`** | Decisiones con respaldo bibliográfico verificado (embargo, ensemble, detectores, tests, abstención) | Vivo |
| **`RESULTADOS_OBJETIVO.md`** | Cifras canónicas para la memoria: §1 SPY (método), **§1bis SMCI (caso de estudio)** | Vivo |
| **`BITACORA.md`** | Cuaderno de campo cronológico (pre-registros + hallazgos) | Vivo |
| **`CONOCIMIENTO_ACUMULADO.md`** | Síntesis de hallazgos | Vivo |
| **`docs/chats/decision_activo/smci.md`** | **Recorrido completo de la elección de SMCI** (resumen + decisiones en orden + fases) | Vivo |
| **`notebooks/m10_better_smci.ipynb`** | **Entregable** del caso de estudio SMCI (pruebas + gráficas + conclusiones) | Vigente |
| **`notebooks/decision_activo.ipynb`** | Registro de la *elección* del activo (protocolo inicial emb=5; ver banner) | Histórico |
| **`notebooks/strata_canonical.ipynb`** | Notebook canónico del método (SPY) | Vivo |
| **`notebooks/logic_esential.ipynb`** | Didáctico: conceptos esenciales (§14b embargo, §14d ensemble) | Vivo |

**Dos casos, no confundir:** **SPY = caso central del método** (donde STRATA rescata significativamente, decisión #1); **SMCI = caso de estudio del tutor** (donde M10 desplegable bate a todo nominal, decisión #13).

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

**Dónde está justificada.** **Recorrido completo: `docs/chats/decision_activo/smci.md`** (resumen + decisiones
en orden + narrativa fase por fase, desde "no sé qué activo" hasta consolidado). Además:
`notebooks/m10_better_smci.ipynb` §F/§F.2; `outputs/experiments/panel_intervention_scan.json`; BITACORA
2026-06-16.

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
que acierta**. En SPY ocurre (M10 vs M5 McNemar p=0.0041 en el panel todo-OOS, embargo=1; p=0.0005 en el
walk-forward causal n=251 — significativo en ambas muestras; leverage effect fuerte). En SMCI **no**: el agente
ya está 95 % corto (alineado con el régimen) → STRATA interviene el 3 % → M5/M8/M10 son la misma apuesta corta
y ninguno se separa. Ningún M10 desplegable bate a B&H/M5/M8 de forma **significativa** en SMCI.

**Por qué importa.** Define dónde funciona el método (su frontera), coherente con el leverage effect (decisión
#1). La ventaja de M10 sobre B&H en SMCI es **sesgo a corto en un activo que cae**, no habilidad direccional
fina (se reporta con el benchmark "siempre-corto" al lado).

**Dónde está justificada.** `notebooks/m10_better_smci.ipynb` §F/§G; `experiments/panel_intervention_scan.py`,
`experiments/m10_smci_rolling.py`; BITACORA 2026-06-16.

**Estado.** Viva (hallazgo de cierre, honesto).

---

## 17. Robustez a la ventana de calibración: la completa (pre-registrada) es la más robusta

**Qué.** A petición del tutor, se prueba recalibrar HMM+GARCH con ventanas más cortas (inicio 2007→2022, fin
fijo 2024-09-30, sin fuga) y recomputar el walk-forward de M10 sobre el **mismo OOS** (`experiments/smci_calib_window.py`).
Resultado: **(a)** acortar **no** vuelve direccional al régimen — la media de Crisis se mantiene **positiva** y
crece (en SMCI el pasado reciente es el *boom* de IA: alta volatilidad con subidas), refutando la hipótesis de
que "el pasado lejano no aporta"; **(b)** la accuracy de M10 **degrada al acortar** (0.552 con la completa →
~0.48, el nivel del agente). La ventaja de M10 **vive en las features de régimen calibradas sobre la historia
larga** (coherente con la ablación: agente-15 0.468 → +STRATA 0.552).

**Por qué importa.** (1) Responde al tutor con evidencia y **no es p-hacking**: la ventana completa era la
pre-registrada (decisión #3), no se elige por el número, y de hecho cualquier ventana más corta es peor.
(2) Es a la vez una **dependencia honesta** que se reporta: el resultado necesita la historia larga.
(3) Conecta con el límite de SMCI: el régimen **separa por volatilidad pero no por dirección** (Crisis con media
positiva), por lo que en el drawdown de verano 2025 M10 sigue largo aunque el régimen ya marque Crisis (la causa
del −34 % es la no-direccionalidad, no el rezago de la RV²¹).

**Dónde está justificada.** `notebooks/STRATA_SMCI.ipynb` §8c (robustez calibración), §2 (régimen→signo
honesto), §5 (drawdown); `experiments/smci_calib_window.py` → `outputs/experiments/smci_calib_window.json`.

**Estado.** Viva (robustez + hallazgo honesto). Nota: el entregable definitivo es `notebooks/STRATA_SMCI.ipynb`,
que **sustituye a `strata_canonical.ipynb`**; el Sharpe se reporta como **P(Sharpe>0)** (0.976 sin corregir /
0.72 corregida por multiplicidad), no como "DSR".

---

## 18. Notebook definitivo del marco práctico = único canónico; caso central SPY (con AutoML)

**Qué.** El marco práctico (Cap. 4) se consolida en **un único notebook canónico**,
`notebooks/STRATA_marco_practico.ipynb` (builder `_build_STRATA_marco_practico.py`), del que se alimenta la
memoria. **Sustituye y absorbe** a `STRATA_SMCI.ipynb` y `decision_automl.ipynb`, que quedan como fuentes/archivo
(no se borran; ver Lección #9, un único canónico). Estructura: §1 datos/protocolo → §2 mecánica ex-ante (HMM K=3
justificado por verosimilitud held-out, GARCH-t, BOCPD, leverage honesto, intervención/atribución por detector) →
§3 **caso SPY** → §4 **panel de 10** → §5 **mecanismo por activo** → §6 **clustering que afirma naturaleza→canal**
→ §7 robustez (equity por activo, accuracy rodante, val/test, rescate alcista/bajista) → §8 apéndice (5 excluidos)
→ §9 conclusiones → auto-test.

**Panel = 10 casos de aplicabilidad de los 15** (SPEC §6.1; el "15" visible, 5 en apéndice de límite):
cuerpo = SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG; apéndice = MSTR, NVDA, BAC, TSLA, IWM. Selección
**ex-ante por mecanismo** (cohorte donde el agente pierde y un canal STRATA lo rescata), no por significancia
per-activo. (Punto más blando del split: UNG, donde el agente ya bate a las triviales — encuadrado como caso ML.)

**Dos supervisores / dos canales (núcleo del Cap. 4).** Regla M8 (canal régimen) vs aprendiz M10/AutoML (canal
ML). Discriminante medible: `crisis_mean<0` → régimen direccional → la **regla** rescata; `crisis_mean>0`
(leverage invertido) → el **aprendiz** rescata. En el panel: 5 régimen (SPY,QQQ,XLF,DIA,XLK) / 5 ML
(XLE,ROKU,SMCI,MARA,UNG). El clustering por naturaleza **afirma** que esa naturaleza causa el canal ganador.

**Robustez (no es suerte):** rescate del agente persiste en accuracy **rodante** (>50% de ventanas en 8/10),
**val/test** (3 particiones) y **en alcista Y bajista** — McNemar pooled M10/AutoML vs M5 significativo en ambos
regímenes. Riesgo pooled: canónico = **pooled-15 (n=3751)**, ΔSharpe M8 vs M5 +0.66 IC[0.23,1.16]; pooled-10
(n=2493) como sensibilidad (M8 +0.64, M10 +0.93, AutoML +0.97, todos sig).

**Caso central = SPY** (no SMCI). En SPY el **leverage effect** es fuerte (régimen direccional) y, con la
configuración canónica, **AutoML-H2O gana en punto a TODAS las estrategias** (acc 0.5737 > ZeroR/B&H 0.5657; M5
0.3665) — resultado real que se **registra**. SMCI pasa a ser el **caso de limitación** (§4.6): leverage débil,
régimen no direccional (Crisis con media positiva), poco margen de rescate (matiza la decisión #13).

**Encuadre honesto cableado (línea roja).** "AutoML gana a todo" en **accuracy** es **nominal**: McNemar AutoML
vs ZeroR p≈0.90 (n≈251 → sin potencia; significancia de accuracy = línea futura). Lo que **sí** sobrevive a un
test: **(a)** rescate del agente en accuracy — McNemar AutoML/M10/M8 vs M5 p≈0.0002 / 0.0074 / 0.051;
**(b)** rescate del agente en riesgo — bootstrap pareado **pooled** (15 activos) M8 vs M5 ΔSharpe +0.66
IC95[0.23,1.16] y ΔmaxDD +0.24 IC95[0.02,0.44], ambos excluyen 0; **(c)** universalidad — cuota STRATA en SHAP
media ≈ 0.66; **(d)** patrón activo→estrategia por clustering. STRATA **no genera alfa** (no bate a ZeroR/B&H sig.).

**Proceso.** El notebook se cierra con un bucle constructor↔revisora (agente `raquel-quant`, quant senior +
matemática) que itera hasta APROBADO contra un gate G1–G6 (estructura/objetivos, rigor, honestidad, coherencia,
reproducibilidad, pitch). Ver `docs/chats/automl/revision_marco_practico.md`.

**Dónde está justificada.** `notebooks/STRATA_marco_practico.ipynb`; outputs `panel_mm25_*`,
`decision_automl_prep.json`, `automl_importance.json`, `strategy_clustering15.json`, `spy_m10_full_report.json`,
`spy_ablation_robustness.json`, suite `m10_smci_*`, y los **nuevos**: `mechanism_panel.json` (diagnóstico de
canal por activo), `panel_robustness.json` (rodante/val-test/bull-bear), `automl_net_returns.json` (serie AutoML
reconstruida), `detector_analysis_{SPY,XLE,MARA}.json`, `k_selection.json`/`k_ablation_panel.json` (K=3). Log de
revisión: `docs/chats/automl/revision_marco_practico.md`. **Estado.** Viva.

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
| 17 | Robustez a la ventana de calibración: la completa (pre-registrada) es la más robusta; M10 depende de la historia larga | Validación | Viva |
| 18 | Notebook definitivo del marco práctico = único canónico (`STRATA_marco_practico.ipynb`); caso central SPY (AutoML gana a todo nominal); SMCI → caso de limitación | Entregable | Viva |

---

## Cómo consultar si una nueva idea contradice una de estas decisiones

Pregunta a `@asesor-historico`: *"Voy a usar [feature/método/setup]. ¿Contradice alguna decisión esencial?"*. El asesor lee este fichero + BITACORA y responde con cita.

Si una decisión necesita revisarse, hay que: (i) documentar en BITACORA el motivo, (ii) auditar con `@rigor-matematico` el coste de cambiarla, (iii) actualizar este fichero. **Nunca cambiar silenciosamente.**
