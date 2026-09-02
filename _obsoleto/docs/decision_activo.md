# Decisión del activo del caso de estudio — registro de la conversación

Registro honesto del proceso para elegir el **activo del caso de estudio** del TFG y la **configuración del
modelo**, con los experimentos, los resultados y las decisiones. Complementa `notebooks/decision_activo.ipynb`
(que tiene las gráficas) y la BITACORA (pre-registros y hallazgos). Todas las cifras provienen de
`outputs/experiments/*.json` generados en esta sesión.

---

## 0. El encargo del tutor (la barra real)

Quedan ~10 días para entregar. El tutor pidió:
- Elegir **un activo donde B&H ("comprar y mantener", siempre largo) acierte ≈ 50 %**, para que el tribunal
  **no pueda tumbar el trabajo** con *"una estrategia trivial es mejor que tu modelo"*.
- Enseñar que el modelo (M10 o M8) **bate al agente y a B&H** en accuracy en ese activo.
- El resto (significancia, otros activos, etc.) → **trabajo futuro**.
- Que sea **desplegable** (utilizable día a día), no un número que mire el futuro.

Clave metodológica que el tutor ya asume: con ~400 días de OOS, **ninguna** estrategia diaria bate a un B&H
justo con *significancia estadística*. Por eso su criterio es "trivial no es mejor + trabajo futuro", **no**
significancia. (Esto evita la trampa de exigir una significancia imposible al tamaño de muestra.)

---

## 1. Las tres versiones de "M10" (y por qué importa)

| Versión | Cómo entrena | ¿Ve el futuro? | ¿Desplegable? |
|---|---|---|---|
| **M10-CPCV** | Validación cruzada combinatoria sobre todo el OOS; cada día se predice con modelos entrenados en otros bloques, **incluidos futuros** | **SÍ** | **NO** |
| **M10-WF (walk-forward)** | Ventana expandible: cada mes reentrena con **solo el pasado** (burn-in 150, embargo 5), predice el mes siguiente | NO | **SÍ** ✅ |
| **M10 val/test 60/40** | Entrena en el primer 60 % del OOS, predice el último 40 % intacto | NO | SÍ |

**El buen número del canónico (SPY 0.539) era CPCV** → look-ahead. El honesto es el **walk-forward**.

---

## 2. Búsqueda del activo (todo pre-registrado en BITACORA)

Se probaron, de forma honesta y pre-registrada, varias vías. Resumen:

1. **Recon causal (walk-forward vanilla) sobre los 10 del panel.** Candidatos donde el modelo desplegable
   bate a M5 **y** B&H: **SMCI** (M10-WF 0.522 > M5 0.482, M8 0.494, B&H 0.486), MSTR/MARA (vía M8). B&H débil.
2. **M10-v3 causal** (capacidad reducida + isotónica + abstención + P95, todo causal). **Negativo en los 10**
   (Brier > 0.25 → sin habilidad probabilística). Confirmó que los números buenos de `M10_V3_GUIA.md` eran
   **look-ahead** (isotónica/abstención/P95 globales + CPCV).
3. **M10 val/test 60/40.** Ningún activo sostenido. UNG nominal pero moneda. En SMCI el split único da **0.150**
   (anti-predictivo) → ver punto 6.
4. **Horizonte semanal (5 días).** Mayores márgenes nominales (UNG M8 0.595, ROKU M10 0.673, SMCI M10 0.579)
   pero **ninguno significativo** (~32 semanas; block-perm p ≈ 0.25–0.46).
5. **Bug del signo del régimen (hallazgo importante).** `strata/detectors.py:209` hardcodea
   Calma→long/Crisis→short, **invertido en 6/10 activos** (recalculado: MARA Calma −0.0019/Crisis +0.0056;
   NVDA Crisis +0.0017; SMCI Crisis +0.0016). Viola `CLAUDE.md §9`. Al corregirlo (signo data-driven por
   activo, congelado en calibración): M8 mejora en MARA/TSLA, M8-dd bate a M5+B&H **nominal** en TSLA/MSTR/MARA,
   pero **ninguno significativo** ni sostenido. *El arreglo es obligatorio por coherencia, pero no produce un
   B&H-beat significativo: el resto es eficiencia de mercado.*

**Conclusión de la búsqueda:** ningún activo da un B&H-beat **significativo** desplegable (esperable: mercado
eficiente, ~400 días). El candidato más limpio para "modelo desplegable bate a todo nominalmente en benchmark
justo" es **SMCI**.

---

## 3. DECISIÓN: SMCI, M10 walk-forward (vanilla)

- **Activo:** **SMCI** — B&H ≈ 0.48 (benchmark justo; el tribunal no puede tumbarlo con "lo trivial gana").
- **Modelo:** **M10 walk-forward desplegable**, XGBoost **300×4, 22 features, vanilla** (las mejoras v3 se
  descartan: empeoran en SMCI, ver §5).
- **Protocolo:** walk-forward expandible, **burn-in 150 días** (no se puntúa), reentreno mensual (21 d),
  embargo 5 d, solo pasado. CPCV descartado para despliegue.
- **Resultado (test desplegable, ~250 días):** accuracy **M10-WF 0.524** > M8 0.496 > M5 0.484 = B&H 0.484;
  **M10-CPCV 0.448 (peor)** → el buen resultado **no** viene de mirar el futuro. agent-only 0.476 → el
  **régimen aporta**.

---

## 4. ¿Cómo se dividió train/test? ¿Burn-in? (preguntas frecuentes)

- **No es un corte único: es walk-forward** (train/test **rodante**). Cada mes: train = todo el pasado,
  test = mes siguiente. Out-of-sample, sin look-ahead.
- **Burn-in = 150 días:** entrenamiento inicial, **NO se puntúa**. La accuracy se mide del día 151 al final.
- **No hay un set de "validación para tunear":** la config (300×4, 22 features) es el **default canónico**
  (§11), fijado a priori. La rejilla de configs (§5) es **chequeo de robustez**, no selección sobre el test.

---

## 5. Sensibilidad del protocolo y robustez por ventanas (`smci_protocol_study.py`)

- **Frecuencia de reentreno (no mejora con diario):** step 1 d = 0.516, 5 d = 0.528, 10 d = 0.528,
  21 d = 0.524. Todos ≈ 0.52 → **robusto al protocolo**; el diario incluso baja un pelo.
- **Burn-in:** N0=150 nominalmente mejor (0.573 en tramo tardío) pero la variación 0.51–0.57 (150 días) está
  **dentro del ruido**.
- **Configs (rejilla):** `all22`/`regime7` ≈ 0.52 > `agent15` 0.476 (→ el régimen aporta);
  **+isotónica+abstención (mejoras v3) = 0.42/0.39 → EMPEORAN** (el sub-split interno quita datos) → vanilla.
- **Rolling window (19 ventanas de 63 d):** **M10 > M5 en 68 %** (consistente → recupera al agente);
  **M10 > B&H solo en 53 %** (≈ moneda → margen sobre el pasivo **NO consistente**). Global no significativo
  (block-perm vs B&H **p = 0.25**, vs M5 p = 0.32, sign vs 0.5 **p = 0.49**).

**Lectura honesta:** el protocolo no se puede "exprimir" para subir accuracy, y el "bate a B&H" **no es
robusto entre sub-periodos** → margen pequeño de muestra → **trabajo futuro** (OOS más largo). Lo robusto es
que **M10 recupera al agente** (68 %, y desplegable).

---

## 6. Por qué el corte único 60/40 falla en SMCI (y el walk-forward es lo correcto)

| Protocolo | SMCI M10 (test) |
|---|---|
| **Walk-forward** (rodante, reentreno mensual) | **0.524** ✅ |
| **Corte único 60/40** (entrena 1 vez en el 60 %, predice 40 %) | **0.150** ❌ |

SMCI es **no estacionario**: un modelo entrenado una sola vez con el primer 60 % queda **anti-predictivo** en
el último 40 % (el régimen cambia). El **walk-forward lo arregla reentrenando cada mes** → se adapta. Y es lo
que harías en producción ("reentreno y opero adelante"). El 0.150 del corte único es **la prueba de que SMCI
cambia de régimen** → justifica el walk-forward. No es elegir el protocolo que mejor sale: el walk-forward es
el estándar desplegable; el corte único es un caso degenerado que falla bajo no-estacionariedad.

---

## 7. ¿Cómo "ganaba" M10 en SPY? (aclaración importante)

- El **0.539 de SPY era CPCV** (look-ahead), sobre el **mismo período** OOS 2024-10 → 2026-05 (~402 días).
  **NO se alargaron ventanas.** El desplegable (walk-forward) SPY = **0.534**.
- **M10 en SPY NUNCA batió a B&H** (B&H SPY ≈ **0.569**, toro fuerte → pasivo imbatible). "M10 gana en SPY"
  significó solo: **bate al agente (0.384) y a la regla (0.436)** — recupera la dirección — y eso **sí** era
  significativo y desplegable (McNemar M10-WF vs M5 **p < 0.001**). Nunca fue "bate al mercado".

| | SPY | SMCI |
|---|---|---|
| ¿M10 recupera al agente? | **Sí, significativo** (p<0.001) | Sí, consistente (68 % ventanas) |
| ¿M10 bate a B&H? | **NO** (B&H 0.569, toro) | **Sí, nominal** (0.524 vs 0.484), no significativo |
| B&H | 0.569 (toro) | 0.484 (justo) |

**Coherencia:** M10 recupera al agente en ambos; solo en un benchmark **justo** (SMCI) puede además rozar
batir al pasivo. Por eso SMCI es el activo del caso de estudio para la objeción anti-trivialidad.

---

## 8. Qué se puede probar de forma ROBUSTA (y qué no)

- **NO se puede** probar "M10 bate a B&H en SMCI de forma robusta/significativa" → los datos dicen que no
  (53 % ventanas, p=0.25). Afirmarlo sería p-hacking; el tribunal lo tumba con la rolling window.
- **SÍ se puede** defender, robustamente:
  1. **Sin look-ahead:** desplegable y **bate a su propia versión CPCV** → no es artefacto de ver el futuro.
  2. **Robusto al protocolo:** invariante a frecuencia de reentreno y burn-in.
  3. **Mecanismo:** el régimen aporta (agent-only baja a 0.476).
  4. **Frente al agente:** recupera al agente en el 68 % de las ventanas (la tesis: supervisar/rescatar, no
     batir al mercado).
- **Lo no robusto (se reporta honestamente → trabajo futuro):** el margen sobre B&H (53 %, no significativo).

Chequeos honestos adicionales propuestos (no fingen significancia, caracterizan "¿es suerte?"): estabilidad
por **semilla** del XGBoost e **IC bootstrap** del margen.

---

## 9. La afirmación defendible ante el tribunal

> *"STRATA supervisa a un agente LLM perdedor. En SMCI —donde comprar-y-mantener es esencialmente una moneda
> (B&H 0.48), así que una estrategia trivial no es superior— un meta-learner desplegable (M10 walk-forward, sin
> look-ahead, robusto al protocolo de reentreno) recupera la dirección: acierta 0.524, por encima del agente
> (0.484), la regla (0.496) y el pasivo (0.484). La ventaja sobre el agente es consistente entre sub-periodos
> (68 %); la ventaja sobre el pasivo es nominal y no significativa a este tamaño de muestra (~250 días), límite
> que dejamos como trabajo futuro. La aportación no es batir al mercado: es un protocolo de supervisión
> estadística interpretable, desplegable y honesto, que recupera accuracy direccional y delimita dónde funciona."*

---

## 10. Plan de los 10 días

1. **Días 1–3:** escribir el caso de estudio SMCI (activo, protocolo desplegable, tabla M5/M8/M10/B&H,
   gráficas de `decision_activo.ipynb`, SPY vs SMCI).
2. **Días 4–6:** límites y **trabajo futuro** (significancia, OOS más largo, bug del signo, horizonte semanal,
   panel).
3. **Días 7–8:** defensa — anticipar preguntas del tribunal (anti-trivialidad, look-ahead, ¿por qué SMCI?).
4. **Días 9–10:** repaso, coherencia de cifras, ensayo.

---

## 11. Ficheros producidos

- `experiments/m10_causal_panel_recon.py` → `outputs/experiments/m10_causal_panel_recon.json`
- `experiments/m10_v3_causal_panel.py` → `…/m10_v3_causal_panel.json`
- `experiments/m10_valtest_casestudy.py` → `…/m10_valtest_casestudy.json`
- `experiments/m10_weekly_horizon.py` → `…/m10_weekly_horizon.json`
- `experiments/m8_datadriven_sign.py` → `…/m8_datadriven_sign.json`
- `experiments/smci_config_study.py` → `…/smci_config_study.json`
- `experiments/smci_protocol_study.py` → `…/smci_protocol_study.json`
- `notebooks/_build_decision_activo.py` → `notebooks/decision_activo.ipynb` (con todas las gráficas)
- Pre-registros y hallazgos en `BITACORA.md` (entradas 2026-06-14/15/16).
