# MANUAL — lo esencial de la memoria (léeme en cada sesión)

> La estrella polar del TFG. Breve y claro: **qué demostramos, los objetivos, la estructura y las cifras
> canónicas**. Si una cifra o decisión no está aquí o en las fuentes que enlazo, **no entra en la memoria**.

---

## 1. Qué demostramos

Un **agente LLM** de trading (AI Hedge Fund, 5 personalidades) decide cada día una posición. Sin supervisar
**pierde y acierta la dirección menos del 50 %**. **STRATA** es una capa de **supervisión estadística** que,
con tres detectores clásicos —**RAM** (régimen, HMM), **PSA** (cambio estructural, BOCPD), **GSO** (volatilidad,
GARCH)— produce **señales** sobre cada decisión del agente. **M10** es **mi modelo**: un meta-learner XGBoost
desplegable que **consume esas señales de STRATA** para decidir la dirección.

**Encuadre (lo que es esta tesis).** El objetivo de fondo es una **estrategia de supervisión estadística
desplegable en tiempo real**, concebida para supervisar a **cualquier agente** (LLM u otro) operando
**cualquier activo**. Esta memoria entrega un **caso de estudio** de esa estrategia sobre **un** activo, como
**prueba de concepto** que el tiempo del TFG permite validar con rigor. La **generalización —multi-agente,
multi-activo y despliegue en vivo— es el foco principal de la línea de investigación y se desarrolla como
trabajo futuro** (cap. 5). El caso de estudio de un activo lo exigió el tutor para que el tribunal no pudiera
tumbar el trabajo con "una estrategia trivial es mejor que tu modelo".

**Tesis.** En un **benchmark justo** (un activo donde comprar-y-mantener es casi una moneda), **M10 —el
meta-learner desplegable sobre las señales de STRATA— bate en accuracy direccional al agente (M5), a la regla a
mano (M8) y a las estrategias triviales** (comprar-y-mantener y "siempre la clase mayoritaria").

**Falsable.** Si el modelo desplegable (walk-forward, sin look-ahead) **no** supera a M5, a M8 y a lo trivial en
accuracy sobre un benchmark balanceado, la tesis cae. (Lo intenté en varios activos y solo SMCI lo cumple;
ver `falsacion/` para lo que no funcionó.)

## 2. Objetivos (tenerlos siempre presentes)

1. Mostrar que el agente solo (**M5**) es **perdedor direccional** (acc < 0,5; sign test).
2. Mostrar que **M10** (meta-learner desplegable sobre las señales de STRATA) **bate en accuracy a M5, a M8 y a
   lo trivial**, de forma **robusta a la partición** (60/40, 70/30, 80/20) y al **rolling** (71–82 % de ventanas),
   en el caso de estudio.
3. **Rigor:** cada cifra con su test; **sin look-ahead** (walk-forward, embargo=1); **honestidad** sobre la
   no-significancia (muestra corta → trabajo futuro). La economía (Sharpe, equity) es ilustración, no prueba.

## 3. El claim canónico (lo que defendemos, sin adornos)

- **Activo del caso de estudio: SMCI.** Sus clases están casi balanceadas (B&H ≈ 0,484), de modo que la
  **accuracy es una métrica informativa** y la comparación con las estrategias triviales es **justa** (no la
  decide la clase mayoritaria).
- **Mi modelo: M10-WF ensemble** — XGBoost (300×4, lr 0,05, subsample/colsample 0,8), **ensemble de 10
  semillas**, **22 features ALL22** (15 del agente + **7 señales STRATA/régimen**: ram, psa, gso, calm/stress/
  crisis prob, garch_sigma), walk-forward expandible (burn-in 150, reentreno 21 d, **embargo 1**), posición =
  signo(p1−0,5), cobertura 100 %. **No** es CPCV; **no** lleva momentum/aug ni abstención.
- **Resultado (OOS 250 d):** **M10 0,552 > "mayoritaria" 0,516 > M8 0,496 > M5 0,484 = B&H 0,484**; Sharpe M10
  1,84; equity 3,24×. **M10 es el único que supera a todos los baselines y a lo trivial.**
- **Honestidad:** la ventaja es **nominal, no significativa** (DSR 0,72; block-perm vs B&H 0,047 no sobrevive la
  multiplicidad). **Robusta** a la partición y al rolling. **Significancia plena = trabajo futuro** (muestra
  ≈250 días; el agente solo existe en el OOS posterior al cutoff del LLM).
- **Por qué la ventaja es pequeña (y honesto):** en SMCI el agente ya va casi siempre corto y el régimen también
  sesga a corto, así que M5/M8/M10 apuestan en la misma dirección la mayor parte del tiempo; M10 extrae algo más
  de señal que M8 y que el agente, pero el margen es modesto a este tamaño de muestra.
- **Aportación:** un **meta-learner desplegable e interpretable por sus features** (las señales de STRATA) que,
  en un benchmark justo, **recupera la dirección y supera al agente, a la regla a mano y a lo trivial**. No
  genera alfa: es disciplina estadística, no una máquina de batir al mercado.

## 4. Estructura de la memoria

| Cap. | Contenido | Estado |
|---|---|---|
| 1 | Introducción (problema, pregunta, esquema) | esqueleto |
| 2 | Estado del arte | borrador |
| **3** | **Marco teórico** (4 bloques → ver `estructura_cap3.md`) | **rehaciendo** |
| 4 | Marco práctico (caso de estudio SMCI: M5 → M8 → M10 → trivial) | esqueleto — **no tocar aún** |
| 5 | Conclusiones + **trabajo futuro = el foco real**: generalización a cualquier agente y activo, despliegue en tiempo real | esqueleto |

## 5. Tabla canónica mínima (SMCI, OOS 250 d, walk-forward, embargo=1)

| Estrategia | Accuracy | Sharpe | Equity |
|---|---:|---:|---:|
| **M10-WF ensemble (mi modelo)** | **0,552** | **+1,84** | **3,24×** |
| Trivial — "siempre mayoritaria" (corto) | 0,516 | — | — |
| M8 (regla a mano STRATA) | 0,496 | +0,33 | 1,02× |
| M5 (agente solo) | 0,484 | −0,24 | 0,98× |
| Trivial — B&H (comprar y mantener) | 0,484 | +0,03 | 0,71× |

*Fuente: `outputs/experiments/m10_smci_valtest_robustez.json` · `RESULTADOS_OBJETIVO.md §1bis`. DSR M10 = 0,72.*

## 6. Dónde está cada cosa (fuente única — no duplicar)

| Necesito… | Voy a… |
|---|---|
| Decisiones vivas (#13–16 = pivot SMCI) | `DECISIONES_ESENCIALES.md` |
| Cifras canónicas del caso | `RESULTADOS_OBJETIVO.md §1bis` |
| Recorrido completo de la elección de SMCI | `docs/chats/decision_activo/smci.md` |
| Decisiones con cita académica | `decisiones_respaldadas_literatura.md` |
| Lo que probé y descarté (negativos) | `falsacion/INDICE.md` |
| Objeciones del tutor + respuestas | `memoria/objeciones_tribunal.md` |
| Estructura del cap. 3 | `memoria/estructura_cap3.md` |
| Reglas de estilo + anti-IA | `memoria/ESTILO_Y_ANTIIA.md` |
| Figuras clave del caso | `graficas_clave.md` |

## 7. Líneas rojas (no negociables)

- **STRATA supervisa; M10 predice.** STRATA produce las señales (régimen/RAM/PSA/GSO); M10 es el meta-learner
  que predice la dirección a partir de ellas. No confundir las dos cosas.
- **Sin look-ahead:** posición de `t` × retorno de `t+1` (`signal_lag=1`); walk-forward, nunca KFold ni CPCV
  para el resultado desplegable.
- **Cada cifra con su test** (McNemar, sign, bootstrap por bloques, DSR) y desde JSON, nunca a mano.
- **Honestidad:** lo nominal se dice nominal; lo no significativo, también. La significancia plena = trabajo
  futuro, no se infla.
