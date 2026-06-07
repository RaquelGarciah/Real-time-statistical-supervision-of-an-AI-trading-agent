# Lecciones aprendidas — errores cometidos y cómo NO repetirlos

12 errores reales del proyecto anterior. Cada lección: **Qué pasó + Síntoma + Causa raíz + Cómo evitarlo en el nuevo proyecto + Quién audita**.

---

## 1. Confusión predicción vs supervisión (terminológica)

**Qué pasó.** Durante semanas, Raquel describió STRATA como *"predictor de retorno direccional"*. El tutor entendió que ella prometía predicción y la juzgó como modelo de pronóstico — pero STRATA no predice, **supervisa**. La angustia ("no entiendo qué estoy haciendo") venía de este lío conceptual.

**Síntoma.** Reunión bloqueante con el tutor en la que Raquel no supo responder *"¿qué hace tu sistema?"*.

**Causa raíz.** Ausencia de glosario en CLAUDE.md. La palabra "predicción" se usaba intercambiablemente con "decisión", "filtro", "supervisión".

**Cómo evitarlo.**
- Glosario obligatorio al inicio del nuevo `CLAUDE.md`:
  - **predicción** = `f: x_t → y_{t+1}` (lo que hace el agente y el XGBoost de M10).
  - **supervisión** = `f: tupla_agente × estado_mercado → tupla_supervisada` (lo que hace STRATA).
  - **ground truth** = `1{r_log(t+1) > 0}` binario.
- Cada vez que aparezca "predicción" en un texto sobre STRATA, sustituir por "supervisión" o "filtro".
- En la primera diapositiva de defensa: *"STRATA NO predice retornos. Es una capa de supervisión estadística sobre un agente que sí decide."*

**Quién audita.** `@narrativa-coherencia`.

---

## 2. Look-ahead `signal_lag` (peso_t × retorno_t)

**Qué pasó.** Durante semanas, el backtest calculaba `pnl_t = peso_t × retorno_t`. Eso es **look-ahead**: la posición de hoy no podía haber capturado el retorno de hoy (la decisión se toma al cierre). Las cifras de Sharpe en M8 estaban infladas (+1.59 same-day vs +0.66 causal). El bug se reportó cuando se descubrió un Sharpe sospechoso en GSO relativo.

**Síntoma.** Sharpe demasiado bueno para ser verdad (+1.5 sobre 400 días). Diferencia grande entre same-day y causal.

**Causa raíz.** Ausencia de test automático que verificara `posicion_t.index + 1 == retorno_usado_t.index` en el motor de backtest.

**Cómo evitarlo.**
- `test_no_leakage.py` corre en CI (ya incluido en `tests/`).
- **Toda nueva métrica** se reporta en doble formato: `same-day` (sanity check) + `causal` (lo válido). Si difieren mucho, hay look-ahead en alguna parte.
- En `core/backtest.py`: la posición lleva atributo `.signal_date` y el motor verifica `signal_date + 1_BDay <= return_date`.

**Quién audita.** `@rigor-matematico` antes de cada experimento; `test_no_leakage.py` en CI.

---

## 3. KFold convencional en series temporales

**Qué pasó.** En el proyecto anterior, M3 usa KFold deliberadamente (para **denunciar** el sesgo) pero el riesgo es que un experimento futuro lo use por descuido y se cuele en la memoria como válido.

**Síntoma.** Cualquier `from sklearn.model_selection import KFold` sin contexto explícito.

**Causa raíz.** Falta de marcador semántico que distinga "uso pedagógico" de "uso por descuido".

**Cómo evitarlo.**
- Solo se permite KFold convencional en scripts con sufijo `_naive` o `_bias_demonstration`. El resto debe usar `WalkForwardSplit` o `CPCVSplit` (de `core/cpcv.py`).
- `@rigor-matematico` audita esto explícitamente en cada experimento.
- `tests/test_no_leakage.py` incluye check: `from sklearn.model_selection import KFold` solo aparece en `m3_*` o `*_naive.py`.

**Quién audita.** `@rigor-matematico` + `@disenador-experimentos`.

---

## 4. RAM no re-signado en NVDA (prior hardcoded)

**Qué pasó.** Durante el ensayo multi-activo, el prior RAM "Crisis ⇒ short" se asumió universal. En NVDA las medias en Crisis son **positivas** (durante el OOS, NVDA subió en crisis tech). RAM volteaba decisiones correctas del agente. Sharpe NVDA cayó de +0.95 a negativo antes de corregirlo.

**Síntoma.** STRATA empeoraba al agente en NVDA mientras lo mejoraba en SPY/BAC.

**Causa raíz.** Hardcode `"Crisis ⇒ short" if regime == "Crisis"` en el detector RAM, sin chequeo del signo de las medias calibradas por activo.

**Cómo evitarlo.**
- El prior RAM se calcula al inicio: para cada régimen, `sign(mean_return[regime])` sobre el periodo de calibración del activo. El detector recibe el prior como argumento, nunca lo hardcodea.
- Test: `tests/test_detectors.py::test_ram_prior_per_asset` verifica que NVDA tiene prior distinto que SPY.

**Quién audita.** `@rigor-matematico` + `@panel-multiactivo`.

---

## 5. GSO relativo con look-ahead

**Qué pasó.** La banda GSO se calculaba con `sigma_t` calculado en `t` (incluyendo el retorno de ese día). Eso filtraba información futura al detector. Sharpe +1.59 same-day, −0.92 causal: claramente look-ahead.

**Síntoma.** GSO sobre-activaba en días de cambio brusco de volatilidad.

**Causa raíz.** El GARCH(1,1) se actualizaba con `r_t` antes de evaluar el sizing del agente de ese día.

**Cómo evitarlo.**
- La banda GSO siempre usa `sigma_{t-1}` (volatilidad condicional con datos hasta `t-1`). El detector recibe `sigma_t_minus_1` como argumento.
- Verificable con `tests/test_detectors.py::test_gso_no_lookahead`.

**Quién audita.** `@rigor-matematico`.

---

## 6. MSTR prior-flip (signo calibración ≠ signo OOS)

**Qué pasó.** En MSTR, las medias por régimen en la calibración (2000–2024-09) tenían signo opuesto al OOS (2024-10 →). El prior RAM aprendido en calibración llevaba al detector a votar al revés en OOS. Sharpe M8 sobre MSTR fue negativo.

**Síntoma.** STRATA empeora al agente en MSTR con McNemar p<0.05.

**Causa raíz.** Cambio estructural del régimen entre calibración y OOS — el activo cambió de comportamiento.

**Cómo evitarlo.**
- **Test ex-ante de coherencia de signo:** comparar `sign(mean_return[regime])` en calibración vs primeros 60 días del OOS. Si discrepa, abortar o documentar como apéndice (caso de fallo conocido, no se incluye en panel de robustez).
- Regla operativa para la memoria: "El prior RAM asume continuidad estructural entre calibración y OOS. Cuando esa continuidad se rompe (caso MSTR), STRATA puede degradar al agente."

**Quién audita.** `@panel-multiactivo` antes de añadir activo nuevo al panel.

---

## 7. SMCI sobre-intervención (agente con info complementaria al prior)

**Qué pasó.** En SMCI, el agente quería short en 279 días (acertaba 53%). RAM con severity high lo volteaba sistemáticamente a long (acertaba 46%). RAM-override **descarta información direccional buena del agente**. McNemar significativo *en contra* de M8.

**Síntoma.** Hit rate M8 < M5 en SMCI con todos los flips en la misma dirección.

**Causa raíz.** El modo `override` asume que el agente está sistemáticamente mal cuando RAM dispara. En SMCI no era el caso: el agente tenía señal direccional buena que RAM no sabía evaluar.

**Cómo evitarlo.**
- **Regla operativa:** si en un activo el agente tiene `hit_rate ≥ 0.50` en una dirección antes de la intervención, **no voltearle** sistemáticamente. Quizás `reduce` (atenuar) en vez de `override` (sustituir).
- Documentar en la memoria como "segundo tipo de fallo distinto del prior-flip clásico: agente con información complementaria al prior".

**Quién audita.** `@panel-multiactivo` + `@rigor-matematico`.

---

## 8. Proliferación de scripts de tuning sin BITACORA

**Qué pasó.** El proyecto anterior acumuló 12 scripts en `experiments/tuning/` (prior_variants, fine_tune_tsla, diagnose_*, screen_*, etc.). Nadie sabía si eran canónicos, intermedios o exploraciones. Costó horas decidir qué borrar.

**Síntoma.** `experiments/tuning/` con muchos scripts y poca documentación.

**Causa raíz.** Permitir exploraciones sin pre-registro y sin protocolo de cierre.

**Cómo evitarlo.**
- **Prohibida la carpeta `experiments/tuning/`** en el nuevo proyecto.
- Toda exploración va a una **branch** `explore/<nombre>`, no a `main`.
- Al cerrar la exploración: o se promueve a canónica (entra a `experiments/` con BITACORA) o se elimina (merge denegado). No hay tercera opción.

**Quién audita.** `@bitacora` antes de cada merge.

---

## 9. Notebooks duplicados

**Qué pasó.** El proyecto anterior acabó con 4 notebooks: `strata_tfg.ipynb` (130 celdas), `strata_final.ipynb` (72 celdas), `notebook_strata_Resultados.ipynb` y `tutor_audit.ipynb`. Cada uno con su versión de la verdad. Confusión al actualizar cifras.

**Síntoma.** Carpeta `notebooks/` con varios `.ipynb` y nadie sabe cuál es el bueno.

**Causa raíz.** Versionar variantes en lugar de iterar sobre el canónico.

**Cómo evitarlo.**
- **Un solo notebook canónico** vive en `notebooks/strata_canonical.ipynb`.
- Las variantes (auditorías, exploraciones) viven en branches dedicadas, nunca en `main`.
- Si una auditoría debe quedar versionada, va como apéndice del canónico (sección nueva), no como notebook separado.

**Quién audita.** `@narrativa-coherencia`.

---

## 10. BITACORA con ruido

**Qué pasó.** La BITACORA llegó a 1520 líneas con entradas tipo "completé X y voy a empezar Y" mezcladas con decisiones metodológicas reales. Encontrar la decisión sobre prior RAM costaba minutos.

**Síntoma.** BITACORA enorme con ratio decisión/ruido bajo.

**Causa raíz.** Ausencia de filtro al escribir entradas. Cualquier cosa entraba.

**Cómo evitarlo.**
- `@bitacora` audita cada propuesta de entrada nueva antes de escribirla.
- Criterios para entrar: (i) decisión metodológica con impacto en cifras; (ii) error con tiempo perdido + solución; (iii) hallazgo del fenómeno relevante para la memoria; (iv) cierre de milestone.
- Criterios para NO entrar: progreso trivial, mensajes de estado, "voy a empezar X".

**Quién audita.** `@bitacora`.

---

## 11. Confusión Sharpe vs Accuracy en la defensa

**Qué pasó.** A mitad de presentar resultados, Raquel se confundió viendo que M8 tenía accuracy 46% (peor que baseline "always long" 56.6%) pero Sharpe +0.62. Pensó que se contradecían. **No se contradicen:** una estrategia puede acertar 46% y ganar dinero si acierta en días de retorno grande y falla en pequeños.

**Síntoma.** Confusión propia ("antes parecía todo tan bonito y ahora ya no") en mitad de la defensa.

**Causa raíz.** Tablas que reportaban Sharpe sin accuracy o accuracy sin Sharpe. Sin ambos juntos, no se entiende el desacoplo.

**Cómo evitarlo.**
- **Toda tabla maestra reporta SIEMPRE Accuracy + AUC + log-loss + Brier + MCC + Sharpe + equity_final juntos.** Eso es el formato canónico para `@narrativa-coherencia`.
- En la memoria, una sección explícita "Cuando la accuracy y el Sharpe desacoplan": un clasificador con acc 46% que gana dinero indica que las pérdidas direccionales caen en días de retorno pequeño.

**Quién audita.** `@narrativa-coherencia` + `@defensa-tutor`.

---

## 12. No haber tenido un agente-asesor desde el principio

**Qué pasó.** Cada sesión nueva con Claude empezaba sin contexto. Decisiones tomadas en una sesión se contradecían en la siguiente. La BITACORA estaba ahí pero requería leerla entera cada vez.

**Síntoma.** Mismas preguntas respondidas distinto en sesiones distintas. Tiempo perdido recontextualizando.

**Causa raíz.** El contexto persistente vivía en `BITACORA.md` + `docs/` pero ningún agente lo consultaba sistemáticamente.

**Cómo evitarlo.**
- **`@asesor-historico` es OBLIGATORIO** al inicio de cada nueva pregunta de investigación (paso 1 del workflow en `CLAUDE.md` §5).
- El asesor lee de `_archivo_proyecto_anterior/` (BITACORA, decisiones, chats, transcripciones) y cita explícitamente la fuente de cada afirmación.
- Si el asesor no encuentra antecedente, lo dice. Si encuentra antecedente contradictorio con la pregunta actual, también lo dice.

**Quién audita.** `@asesor-historico` se invoca a sí mismo, pero `@rigor-matematico` verifica que se consultó antes de aprobar el diseño.

---

## Tabla resumen — pre-flight check para cualquier experimento nuevo

| ✓ | Check | Lección |
|---|---|---|
| ☐ | ¿He preguntado a `@asesor-historico`? | #12 |
| ☐ | ¿Mi documento usa "supervisión" no "predicción"? | #1 |
| ☐ | ¿Mi backtest pasa `test_no_leakage.py`? | #2, #5 |
| ☐ | ¿Mi splitter es CPCV/WalkForward (no KFold)? | #3 |
| ☐ | ¿Mi prior RAM es data-driven por activo? | #4 |
| ☐ | ¿He verificado coherencia de signo calib vs OOS? | #6 |
| ☐ | ¿Mi régimen `override` no degrada agentes con hit_rate≥0.5? | #7 |
| ☐ | ¿Mi exploración va a branch, no a `experiments/tuning/`? | #8 |
| ☐ | ¿Estoy iterando sobre el notebook canónico, no creando uno nuevo? | #9 |
| ☐ | ¿Mi entrada de BITACORA pasa el filtro de `@bitacora`? | #10 |
| ☐ | ¿Mi tabla reporta accuracy + Sharpe + log-loss juntos? | #11 |
| ☐ | ¿Mi pre-registro está en BITACORA antes de ejecutar? | #11 + decisión esencial #11 |

Imprime esta lista mentalmente antes de cada experimento.
