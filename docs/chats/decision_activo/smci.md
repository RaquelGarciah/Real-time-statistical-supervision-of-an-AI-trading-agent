# Elección de SMCI como activo principal del caso de estudio — registro completo

> Registro ordenado de todo el proceso por el que SMCI se eligió como activo del caso de estudio del TFG, las
> "manipulaciones" (experimentos, configuraciones, validaciones) que se probaron sobre él, y cómo quedó
> consolidado. Desde el punto de partida (no sabía qué activo coger) hasta el estado actual. Toda cifra está
> auditada (`@rigor-matematico`, `@experto-citas`) y pre-registrada en BITACORA.

---

## RESUMEN

El tutor pidió centrar el TFG en **un activo** donde el meta-learner **M10 (o variante) batiera en accuracy a
M5 (agente), M8 (regla) y B&H (comprar-y-mantener)**, siendo **desplegable**. Tras descartar honestamente que
exista un M10 que bata a B&H **significativamente** en ningún activo del panel (los buenos números previos eran
*look-ahead* de CPCV), se buscó un activo con **B&H ≈ 50 %** (benchmark justo, para que el tribunal no lo tumbe
con "una estrategia trivial gana"). **SMCI es el ÚNICO activo del panel donde M10 bate a M5, M8 y B&H a la vez
(nominal)** y donde el M10 **desplegable** (walk-forward, solo pasado) gana — no por look-ahead.

Se intentó **mejorar la accuracy de M10 en SMCI** con muchas palancas; la **única** que aporta es el
**ensemble de 10 semillas** (accuracy 0.552, Sharpe 1.84, equity 3.24×, embargo=1). El resto (tuning en
validación, features de señal real, triple-barrier, modelos por régimen, stacking, abstención) **no mejora**.
La ventaja de M10 **no es estadísticamente significativa** tras corrección honesta — y entendemos **por qué**:
en SMCI el agente ya está **95 % corto** (alineado con el régimen), así que M5/M8/M10 son la **misma apuesta
corta** y STRATA no tiene nada que corregir. STRATA rescata al agente **solo donde éste va a contracorriente de
un régimen que acierta** (eso es SPY, p=0.0041; no SMCI). SMCI se cierra como **caso de estudio honesto**: M10
desplegable bate a todo **nominalmente**, con la significancia plena como **trabajo futuro** (muestra corta).

---

## DECISIONES EN ORDEN

1. **Criterio del tutor (y su relajación).** Activo donde M10/variante bata en accuracy a M5, M8 y B&H,
   desplegable. El tutor lo relaja: elegir activo con **B&H ≈ 50 %**, que M10 lo bata **nominalmente**, el
   resto = trabajo futuro.
2. **Hallazgo honesto previo.** **Ningún** M10 desplegable bate a B&H **significativamente** en el panel. Los
   números buenos de `chat_m10`/guías eran **look-ahead** (CPCV ve bloques futuros; isotónica/abstención/P95
   sobre el OOS global). → no usables como resultado desplegable.
3. **Elección de SMCI.** Único activo del panel donde **acc(M10) > M5, M8 y B&H** (nominal), con B&H ≈ 0.484
   (benchmark justo) y M10 **desplegable** (walk-forward, solo pasado) por encima incluso de su propia versión
   CPCV (que ve el futuro).
4. **Tuning en validación → DESCARTADO.** Elegir la mejor de 165 combinaciones en validación se desploma en
   test (sobreajuste de selección). Disciplina **validación≠test**.
5. **Métodos avanzados → DESCARTADOS.** Triple-barrier (López de Prado), modelos por régimen HMM, stacking
   M5→M10, voting y abstención condicional **no mejoran** la accuracy; varios la degradan.
6. **Ensemble de semillas → ADOPTADO.** 10 XGBoost que solo difieren en la semilla, promediados. Mejora
   accuracy (0.52→0.552) **y** Sharpe (0.85→1.84) y equity (1.45×→3.24×). Lícito (bagging, Breiman 1996;
   seed-averaging, Dietterich 2000); sin look-ahead, sin cherry-pick.
7. **Embargo = 1 → ADOPTADO (por principio).** Para etiqueta de horizonte 1 en walk-forward rolling-origin, el
   embargo mínimo correcto es 1 (Tashman 2000; López de Prado 2018 §7.4; Bergmeir, Hyndman & Koo 2018). El ≥5
   es regla de CPCV. Sube accuracy 0.524→0.552 (nominal); la significancia no sobrevive (pico aislado).
8. **M10 final desplegable.** XGBoost (300×4) **ensemble 10 semillas** sobre las **22 features STRATA**,
   walk-forward expandible (burn-in 150, reentreno mensual 21 d, **embargo 1**), posición = signo(p1−0.5),
   cobertura 100 %.
9. **Límite honesto (hallazgo de cierre).** En SMCI el agente es 95 % corto → STRATA interviene 3 % → M5/M8/M10
   no se separan; ninguno bate a B&H/M5/M8 **significativamente**. STRATA rescata significativamente solo donde
   el agente discrepa de un régimen que acierta = **SPY** (M10 vs M5 p=0.0041; leverage effect). Coherente con
   CLAUDE.md §3.

---

## EL RECORRIDO, PASO A PASO

### Fase 0 — Punto de partida: ¿qué activo?

El tutor pide un **caso de estudio de un activo** donde M10 (o una variante) bata en accuracy a todo lo demás
y sea **desplegable**. Inicialmente no estaba claro qué activo. El tutor matiza (a ~10 días de la entrega):
elegir un activo con **B&H ≈ 50 % acierto** para que el tribunal no pueda decir "una estrategia trivial es
mejor", mostrar que el modelo lo bate **nominalmente**, y dejar lo demás como trabajo futuro.

### Fase 1 — ¿Se puede batir a B&H de verdad? (lo honesto primero)

Antes de elegir, se estableció con rigor que:
- **Ningún M10 desplegable** bate a B&H **significativamente** en accuracy en el panel de 10 activos (varias
  búsquedas pre-registradas).
- Los números buenos previos (`chat_m10.md`, guías M10-v3/v7) eran **look-ahead**: CPCV entrena con bloques
  cronológicamente futuros, y la isotónica/abstención/P95 se calibraban sobre el OOS global. → No sirven como
  resultado **desplegable**.
- En activos donde **B&H es débil** (cayeron/laterales) sí hay candidatos a "batir a B&H", porque "siempre
  largo" es mala apuesta ahí.

### Fase 2 — Elección de SMCI

Barriendo el panel con la **M10 desplegable** (config fija, walk-forward), **SMCI** resultó el **único** activo
donde:
- **acc(M10) > M5, M8 y B&H** simultáneamente (nominal): M10 0.524 (luego 0.552 con embargo=1) > M8 0.496 > M5
  0.484 = B&H 0.484.
- **B&H ≈ 0.484** → benchmark **justo** (≈ moneda), lo que pedía el tutor.
- El M10 **walk-forward** (entrenado solo con el pasado) gana por encima incluso de su versión **CPCV** (que ve
  el futuro; en SMCI CPCV da 0.448, peor) → el buen resultado **no** viene de mirar el futuro.

Registrado en `notebooks/decision_activo.ipynb` y BITACORA.

### Fase 3 — Intentar mejorar la accuracy de M10 en SMCI

Raquel pidió agotar las palancas **antes** de cerrar SMCI. Tres experimentos pre-registrados:

**(A) `m10_improve_smci.py` — tuning en validación → FRACASA.**
Split OOS en validación/test; elegir la mejor de **165 combinaciones** (umbral, features, recencia, ensemble,
señal real) **en validación**. La config ganadora se **desploma en test** (mejora −0.10): firma del
**sobreajuste de selección**. Lección: validación≠test; por eso no se hace p-hacking.

**(B) `m10_smci_deep.py` — configs fijas a priori → techo 0.552.**
Cinco configs motivadas a priori (base, ensemble, +señal real, etc.) sobre todo el OOS. **Techo = ensemble**
(0.552 con embargo=1; 0.524 con embargo=5). Las features de señal real (momentum/vol-rel/racha) y la recencia
**no suben** la accuracy. Ninguna significativa.

**(C) `m10_smci_advanced.py` — métodos de la literatura → ninguno mejora.**
- **Triple-barrier** (López de Prado 2018, cap. 3; embargo=H+1, sin look-ahead): 0.488 → empeora.
- **Modelos por régimen HMM** (3 XGBoost ponderados por P_estado): ~0.50–0.536 → no aporta.
- **Stacking M5→M10** (size del agente como feature): ~0.50 → no aporta.
- **Voting M5+M10** y **abstención condicional** (por régimen / por acuerdo de las 5 personalidades): no suben
  la accuracy (ver Fase 8).
- **Única mejora robusta: el ENSEMBLE** (misma accuracy o mejor, Sharpe 0.85→1.84, equity 1.45×→3.24×).

### Fase 4 — El embargo (de 5 a 1)

Raquel preguntó cómo se calculaba el **0.524** (todo el OOS menos burn-in 150 = 250 días) y si el **embargo=5**
estaba tirando los días recientes (los más informativos en un activo no estacionario).

- Barrido de embargo {0,1,2,3,5,10}: la accuracy **no es monótona**; pica en **embargo=1** (0.552) y el único
  p<0.05 vs B&H (block-perm 0.047) está ahí aislado (embargo 0 y 2 dan p≈0.12–0.13).
- **Decisión: embargo = 1, por PRINCIPIO** (no por el p-valor): para etiqueta de **horizonte 1** en
  walk-forward *rolling-origin* el embargo mínimo correcto es 1; el ≥5 es regla de **CPCV** (folds
  bidireccionales). Respaldo verificado: **López de Prado 2018 §7.4, Tashman 2000, Bergmeir/Hyndman/Koo 2018**
  (+ Burman 1994, Racine 2000). Documentado en `logic_esential.ipynb` §14b y
  `decisiones_respaldadas_literatura.md` §1.
- **Honestidad:** la significancia **no sobrevive** la corrección por multiplicidad del barrido (Bonferroni-5
  ≈ 0.28). Se reporta como **sensibilidad**; embargo=1 sube la accuracy **nominal**.

(Aclaración importante que pidió Raquel: la accuracy que se compara con B&H es la de **cobertura completa**
—M10 apuesta todos los días, como B&H—, **no** la de días activos; esa solo aplica a la abstención.)

### Fase 5 — ¿Por qué M5, M8 y M10 no se separan en SMCI?

Raquel detectó que M8 ≈ M5 y que M10 no bate al agente, e intuyó que "pasaba algo". El diagnóstico:
- **El agente (M5) está 95 % corto** en SMCI (2 % largo, 3 % neutral): bajista casi permanente.
- **STRATA interviene solo el 3 % de los días** (M8 ≠ M5): override-C dispara ante incoherencia
  agente↔régimen, pero el agente —ya corto— **coincide** con el régimen (alta vol → corto) → no hay nada que
  corregir → M8 ≈ M5.
- **M10 también es corto-sesgado** → discordantes de McNemar equilibrados (p≈0.48) → no se separa del agente.
- **Los tres son la misma apuesta corta.**

**Barrido del panel (`panel_intervention_scan.py`)** lo confirma: STRATA/M10 rescata al agente **solo donde el
agente discrepa de un régimen que acierta**. Ranking de discrepancia: ROKU, MARA, XLE, …, SPY, …, **SMCI al
fondo** (discrep 0.17, interv 3 %). **SPY es el ÚNICO con rescate significativo** (M10 vs M5 p=0.0041; leverage
effect fuerte en el índice). En SMCI no hay rescate, como se observó.

**Muro estructural 2×2** (por qué SMCI es el único que bate a todo): el agente es corto-sesgado en los 10
activos; en los que **caen** (B&H batible) ya acierta yendo corto → M10 no se separa; en los que **suben** M10
rescata pero B&H gana. La casilla "activo cae + agente equivocado" está **vacía**. SMCI es el caso umbral.

### Fase 6 — ¿Gana de forma consistente? (rolling-window)

`m10_smci_rolling.py` (ensemble, embargo=1): M10 bate a **B&H** en **71–82 %** de las ventanas (42/63/84 d) y
**al agente** en **67–76 %**. La significancia global es **borderline** (block-perm vs B&H p=0.047) pero **no
sobrevive** la multiplicidad → consistente pero no significativo. La curva rodante muestra que M10 lidera en
los tramos bajistas y sufre en el rally de mediados de 2025 (cuando el leverage effect falla en un valor
individual y M10, corto, pierde).

### Fase 7 — Elegir la mejor estrategia por selección en validación

`m10_smci_select.py`: validación = primeros días, test = últimos ~150 (intacto). Elegir (config, burn-in) por
accuracy en validación (legítimo, **no** p-hacking). Elegida: **ensemble / burn-in 180** → en test **acc
0.587, Sharpe 2.30, equity 2.71×**, batiendo a M5/M8 (0.533) y B&H (0.447). En **esa ventana** incluso bate a
B&H significativamente (McNemar p=0.026, block-perm 0.014) y a la moneda (sign 0.041). **Matiz honesto:** ese
split (burn-in 180 → validación de solo 70 días) era **desequilibrado** (validación alcista / test bajista),
así que el contraste estaba sesgado.

**Fase 7bis — Robustez a la partición (el respaldo correcto).** `m10_smci_valtest_robustez.py`: en vez de un
split raro, se prueban **3 ratios estándar** (60/40, 70/30, 80/20; burn-in 150 fijo, embargo=1). **En los
tres, M10 bate a M5, M8 y B&H tanto en validación como en test** (validación 0.52–0.535, test 0.60–0.62), con
regímenes equilibrados. → la conclusión "M10 gana a todo" **no depende del corte**. Es el respaldo de
**consistencia** del resultado principal (todo el OOS, 0.552). Honesto: al achicar el test la accuracy sube
(0.60→0.62) pero pierde potencia (sign p 0.057→0.119); por eso el número que se reporta es el de **todo el
OOS (0.552)**, y los splits son respaldo, **no** se elige el de mayor accuracy (eso sería *split-shopping*).

### Fase 8 — Ensemble (lícito) y abstención (literatura, descartada)

- **Ensemble:** 10 XGBoost que solo difieren en la semilla, promediando p1. Reduce la **varianza** del azar
  interno (subsample/colsample) → señal se conserva, ruido se cancela. **Lícito**: bagging (Breiman 1996),
  seed-averaging (Dietterich 2000); sin look-ahead, sin cherry-pick (se promedian las 10, no se elige la
  mejor). Reduce ruido, **no crea señal** → ganancia modesta, significancia no garantizada. (`logic_esential`
  §14d; `decisiones_respaldadas_literatura.md` §2.)
- **Abstención (probada y descartada):** la literatura (Chow 1970; Cortes, DeSalvo & Mohri 2016; El-Yaniv &
  Wiener 2010) propone abstenerse en días de baja confianza para subir la accuracy — **solo si** la confianza
  ordena la dificultad. En SMCI **no**: abst. por régimen baja a **0.489** (< 0.552 completa) y reduce la
  cobertura al 75 % (no comparable a B&H). Se descarta; el modelo opera a cobertura completa.
  (`decisiones_respaldadas_literatura.md` §11; notebook §C.3 con gráfica de cobertura vs accuracy.)

### Fase 9 — Consolidación y documentación

- Entregable: **`notebooks/m10_better_smci.ipynb`** (§A tuning, §B configs, §0bis embargo, §C avanzados, §E
  selección, §F muro estructural, §G rolling, §D conclusiones). Ejecutado sin errores.
- Decisiones: **`DECISIONES_ESENCIALES.md` #13–#16** (SMCI, M10-WF ensemble, embargo=1, límite STRATA).
- Literatura: **`decisiones_respaldadas_literatura.md`** (embargo, ensemble, batería de tests, detectores,
  abstención) con citas verificadas en `tesis/bibliography.bib`.
- Didáctico: **`logic_esential.ipynb`** §14b (embargo) y §14d (ensemble).

---

## ESTADO FINAL CONSOLIDADO

- **Activo del caso de estudio: SMCI** (B&H ≈ 0.484, benchmark justo; único que M10 bate a todo nominal).
- **Modelo final: M10 desplegable = walk-forward ensemble** (XGBoost 300×4, 10 semillas, 22 features STRATA,
  burn-in 150, reentreno 21 d, **embargo 1**, cobertura 100 %).
- **Cifras (todo el OOS, 250 d):** M10 **0.552** > M8 0.496 > M5 0.484 = B&H 0.484; Sharpe M10 **1.84** (B&H
  0.03); equity **3.24×** (B&H 0.71×); DSR 0.72 (<0.95).
- **Significancia:** **nominal**, no plena. block-perm vs B&H 0.047 (no sobrevive multiplicidad); no bate al
  agente ni a la moneda de forma significativa. Reportado honestamente.
- **Marco honesto:** en SMCI los tres modelos son la misma apuesta corta; STRATA rescata significativamente
  solo donde el agente va a contracorriente de un régimen que acierta (SPY). SMCI cumple el criterio del tutor
  (batir a todo nominal con B&H≈50 %); la significancia plena es **trabajo futuro** (límite de muestra: el
  agente solo existe en el OOS post-cutoff del LLM).

---

## FICHEROS Y EXPERIMENTOS

| Fichero | Qué hace |
|---|---|
| `experiments/m10_improve_smci.py` | Tuning en validación (fracasa) |
| `experiments/m10_smci_deep.py` | Configs fijas a priori (techo 0.552) |
| `experiments/m10_smci_advanced.py` | Triple-barrier, régimen, stacking, voting, abstención |
| `experiments/m10_smci_embargo.py` | Barrido de robustez al embargo {0,1,2,3,5,10} |
| `experiments/m10_smci_rolling.py` | Rolling-window (consistencia) |
| `experiments/m10_smci_select.py` | Selección de burn-in en validación |
| `experiments/panel_intervention_scan.py` | Discrepancia agente↔régimen e intervención (panel) |
| `notebooks/m10_better_smci.ipynb` | **Entregable**: todas las pruebas + gráficas + conclusiones |
| `notebooks/decision_activo.ipynb` | Registro de la elección de SMCI (Fase 2) |
| `decisiones_respaldadas_literatura.md` | Decisiones con respaldo bibliográfico verificado |

---

## CÓMO DEFENDERLO (resumen para el tribunal)

> *"Elegimos SMCI porque, en un panel de 10 activos, es el único donde un meta-learner desplegable
> (walk-forward, sin look-ahead) bate en accuracy direccional al agente, a la regla y a comprar-y-mantener, y
> donde B&H ≈ 50 % hace el benchmark justo. La mejor configuración es un ensemble de 10 XGBoost (bagging,
> Breiman 1996) con embargo=1 (correcto para horizonte 1 en rolling-origin, Tashman 2000): accuracy 0.552 vs
> 0.484 de B&H, con mejor Sharpe y equity. La ventaja es nominal, no significativa tras corrección por
> multiplicidad, y lo decimos honestamente: en SMCI el agente ya está alineado con el régimen (95 % corto), así
> que STRATA no tiene margen de rescate —ese margen aparece donde el agente va a contracorriente de un régimen
> que acierta, como en SPY (p=0.004)—. SMCI demuestra que el método es desplegable y bate al pasivo en un
> benchmark justo; la significancia plena requiere más muestra (trabajo futuro)."*
