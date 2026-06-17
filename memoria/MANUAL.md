# MANUAL — lo esencial de la memoria (léeme en cada sesión)

> La estrella polar del TFG. Breve y claro: **qué demostramos, los objetivos, la estructura y las cifras
> canónicas**. Si una cifra o decisión no está aquí o en las fuentes que enlazo, **no entra en la memoria**.

---

## 1. Qué demostramos (hipótesis falsable)

Un **agente LLM** de trading (AI Hedge Fund, 5 personalidades) decide cada día una posición. Sin supervisar
**pierde y acierta la dirección menos del 50 %**. **STRATA** es una capa de **supervisión estadística** (no
predice; supervisa) con tres detectores clásicos: **RAM** (régimen, HMM), **PSA** (cambio estructural, BOCPD),
**GSO** (volatilidad, GARCH).

**Tesis:** la supervisión estadística clásica **rescata la dirección** del agente, y en un **benchmark justo**
un meta-learner desplegable **bate a lo trivial**. Falsable en tres niveles: (1) McNemar pareado M8 vs M5;
(2) atribución de P&L por detector (RAM domina); (3) universalidad (un XGBoost no debe batir a la regla, y SHAP
debe señalar las features STRATA).

## 2. Objetivos (tenerlos siempre presentes)

1. Mostrar que el agente solo (**M5**) es perdedor direccional (acc < 0,5, sign test p ≪ 0,05).
2. Mostrar que STRATA **rescata la dirección** del agente — **significativo** donde el agente discrepa de un
   régimen que acierta: **SPY** (M10 vs M5 p = 0,0041, leverage effect).
3. En un **benchmark justo** (SMCI, B&H ≈ 0,484), mostrar que un meta-learner **desplegable** (M10-WF) bate a
   agente, regla y pasivo en accuracy (**nominal**), **robusto a la partición** → responde a la objeción
   anti-trivial del tutor.
4. **Rigor:** cada cifra con su test; sin look-ahead (walk-forward, embargo=1); honestidad sobre la
   no-significancia (muestra corta → trabajo futuro). La economía (Sharpe, equity) es ilustración, no prueba.

## 3. El claim canónico (lo que defendemos, sin adornos)

- **Activo central del caso de estudio: SMCI** (B&H ≈ 0,484 → benchmark justo, el tribunal no lo tumba con
  "lo trivial gana").
- **Modelo final: M10-WF ensemble** — XGBoost (300×4, lr 0,05, subsample/colsample 0,8) **ensemble de 10
  semillas**, **22 features ALL22** (15 del agente + 7 STRATA/régimen), walk-forward expandible (burn-in 150,
  reentreno 21 d, **embargo 1**), posición = signo(p1−0,5), cobertura 100 %. **No** es CPCV; **no** lleva
  momentum/aug ni abstención.
- **Resultado (OOS 250 d):** **M10 0,552 > M8 0,496 > M5 0,484 = B&H 0,484**; Sharpe M10 1,84; equity 3,24×.
- **Honestidad:** ventaja **nominal, no significativa** (DSR 0,72; block-perm vs B&H 0,047 no sobrevive
  multiplicidad). **Robusto** a la partición (60/40, 70/30, 80/20, burn-in 150: M10 gana a todo) y al rolling
  (71–82 % de ventanas). **Significancia plena = trabajo futuro** (el agente solo existe en el OOS post-cutoff).
- **Por qué no es significativo en SMCI (y es coherente):** el agente está 95 % corto (alineado con el régimen)
  → M5/M8/M10 son la misma apuesta corta; STRATA solo rescata donde el agente va a contracorriente de un
  régimen que acierta (= SPY).
- **Aportación:** un **protocolo de supervisión estadística interpretable, desplegable y honesto** que recupera
  accuracy direccional y delimita dónde funciona. **No genera alfa.**

## 4. Estructura de la memoria

| Cap. | Contenido | Estado |
|---|---|---|
| 1 | Introducción (problema, pregunta, esquema) | esqueleto |
| 2 | Estado del arte | borrador |
| **3** | **Marco teórico** (4 bloques → ver `estructura_cap3.md`) | **rehaciendo** |
| 4 | Marco práctico (caso SMCI + SPY mecanismo) | esqueleto — **no tocar aún** |
| 5 | Conclusiones + trabajo futuro | esqueleto |

## 5. Tabla canónica mínima (SMCI, OOS 250 d, walk-forward, embargo=1)

| Estrategia | Accuracy | Sharpe | Equity |
|---|---:|---:|---:|
| **M10-WF ensemble** | **0,552** | **+1,84** | **3,24×** |
| M8 (STRATA override) | 0,496 | +0,33 | 1,02× |
| M5 (agente solo) | 0,484 | −0,24 | 0,98× |
| B&H (pasivo) | 0,484 | +0,03 | 0,71× |

*Fuente: `outputs/experiments/m10_smci_valtest_robustez.json` · `RESULTADOS_OBJETIVO.md §1bis`. DSR M10 = 0,72.*

## 6. Dónde está cada cosa (fuente única — no duplicar)

| Necesito… | Voy a… |
|---|---|
| Decisiones vivas (16, #13–16 = pivot SMCI) | `DECISIONES_ESENCIALES.md` |
| Cifras canónicas | `RESULTADOS_OBJETIVO.md` (§1 SPY método · **§1bis SMCI**) |
| Recorrido completo de la elección de SMCI | `docs/chats/decision_activo/smci.md` |
| SPY como caso-mecanismo | `docs/chats/decision_activo/spy_understandStrata.md` |
| Decisiones con cita académica | `decisiones_respaldadas_literatura.md` |
| Lo que probé y descarté (negativos) | `falsacion/INDICE.md` |
| Objeciones del tutor + respuestas | `docs/defensa_walkforward.md`, `docs/chats/questions_and_answers.md` |
| Estructura del cap. 3 | `memoria/estructura_cap3.md` |
| Reglas de estilo + anti-IA | `memoria/ESTILO_Y_ANTIIA.md` |
| Figuras clave del caso | `graficas_clave.md` |

## 7. Líneas rojas (de CLAUDE.md, no negociables)

- **Supervisión, no predicción.** STRATA supervisa; el agente y el XGBoost predicen.
- **Sin look-ahead:** posición de `t` × retorno de `t+1` (`signal_lag=1`); walk-forward, nunca KFold.
- **Cada cifra con su test** (McNemar, DM, sign, bootstrap, DSR) y desde JSON, nunca a mano.
- **Honestidad:** lo nominal se dice nominal; lo no significativo, también. La significancia plena = trabajo
  futuro, no se infla.
