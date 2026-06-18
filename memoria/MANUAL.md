# MANUAL — lo esencial de la memoria (léeme en cada sesión)

> La estrella polar del TFG. Breve y claro: **qué demostramos, los objetivos, la estructura y las cifras
> canónicas**. Si una cifra o decisión no está aquí o en las fuentes que enlazo, **no entra en la memoria**.

---

## 1. Qué demostramos

Un **agente LLM** de trading (AI Hedge Fund, 5 personalidades) decide cada día una posición. Sin supervisar
**pierde y acierta la dirección menos del 50 %**.

> **Glosario de alcance (importante).** En esta memoria, **STRATA** designa la **capa de supervisión estadística
> que produce señales** sobre cada decisión del agente mediante los detectores **RAM/PSA/GSO**. La implementación
> de la capa de intervención por reglas se denota **M8** y se reporta **como referencia, no como
> hipótesis principal**. La **hipótesis principal recae en M10**, el meta-learner que consume las señales de
> STRATA. El pivot respecto al `CLAUDE.md` original (donde STRATA = detectores + intervención) se documenta en
> `DECISIONES_ESENCIALES.md #13`.

**Encuadre (lo que es esta tesis).** El objetivo de fondo es una **estrategia de supervisión estadística
desplegable en tiempo real**, concebida para supervisar a **cualquier agente** (LLM u otro) operando
**cualquier activo**. Esta memoria entrega un **caso de estudio** de esa estrategia sobre **un** activo, como
**prueba de concepto** que el tiempo del TFG permite validar con rigor. La **generalización —multi-agente,
multi-activo y despliegue en vivo— es el foco principal de la línea de investigación y se desarrolla como
trabajo futuro** (cap. 5; coste cuantificado en §4).

**Tesis.** En un **benchmark justo** (un activo donde comprar-y-mantener es casi una moneda), **M10 —el
meta-learner desplegable sobre las señales de STRATA— bate en accuracy direccional al agente (M5), a la regla a
mano (M8) y a las estrategias triviales** (comprar-y-mantener y "siempre la clase mayoritaria").

**Falsable.** Si el modelo desplegable (walk-forward, sin look-ahead) **no** supera a M5, a M8 y a lo trivial en
accuracy sobre un benchmark balanceado, la tesis cae. (Se intentó en varios activos y solo SMCI lo cumple; ver
`falsacion/` para lo que no funcionó.)

## 2. Objetivos de la TESIS (van en el cap. 1)

*(Distintos del plan de ejecución —limpieza, montar `memoria/`, rehacer el cap. 3—, que es la lista de tareas.)*

**Objetivo general.** Diseñar y validar una **estrategia de supervisión estadística desplegable en tiempo real**
que, mediante detectores clásicos de series temporales (régimen, cambio estructural y volatilidad), supervise
las decisiones de un agente de trading y permita a un meta-learner **mejorar su acierto direccional sin fuga
temporal**, con vocación de generalizar a cualquier agente y cualquier activo.

**Objetivos específicos.**
1. **Establecer el problema:** que el agente solo (**M5**) es **perdedor direccional** (acc < 0,5; sign test).
2. **Formalizar** matemáticamente los tres detectores de STRATA (HMM gaussiano, GARCH(1,1)-t, BOCPD) y la capa
   de intervención, con demostraciones y citas (cap. 3).
3. **Construir y validar M10**, un meta-learner desplegable sobre las señales de STRATA, sin look-ahead
   (walk-forward, embargo=1, `signal_lag=1`), y demostrar que **bate en accuracy a M5, a M8 y a lo trivial** en
   el caso de estudio, robusto a la partición (60/40, 70/30, 80/20) y al rolling (cap. 4).
4. **Demostrar la interpretabilidad por features:** que **las señales de STRATA son las informativas** para M10
   —ablación: M10 con solo las 15 features del agente cae a 0,476 ≈ M5; SHAP: las 7 señales STRATA/régimen por
   delante de las 15 del agente—. Esto hace a **M10 inseparable de STRATA** y justifica el nombre del proyecto.
5. **Evaluar con rigor** (tests pareados, bootstrap por bloques, P(Sharpe>0) corregida) y **delimitar honestamente los límites**
   (no-significancia a muestra corta; generalización multi-agente/multi-activo y despliegue en vivo = trabajo
   futuro).

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
- **Interpretabilidad:** ablación en el notebook definitivo (walk-forward ensemble, embargo=1): **M10 solo-agente
  = 0,468 → 0,552 con las 22** (las 7 señales STRATA aportan ≈ +8 pp; McNemar 0,053, casi sig.) → el meta-aprendiz
  sí usa la señal de STRATA. SHAP (in-sample, modelo full-fit): las 7 features STRATA/régimen pesan **41,4 %** del
  total. Fuente única: `notebooks/STRATA_SMCI.ipynb` §7–§7b.
- **Honestidad:** la ventaja es **nominal, no significativa** (ver tests en §6). **Robusta** a la partición, al
  rolling y a la **ventana de calibración** (recortar la calibración degrada M10 hacia el nivel del agente → la
  ventana completa pre-registrada es la más robusta; `experiments/smci_calib_window.py`). **Significancia plena =
  trabajo futuro** (muestra ≈250 días; el agente solo existe en el OOS posterior al cutoff del LLM).
- **Entregable:** `notebooks/STRATA_SMCI.ipynb` (sustituye a `strata_canonical`); el Sharpe se reporta como
  **P(Sharpe>0)** (0,976 sin corregir / 0,72 corregida por multiplicidad), no como "DSR".
- **Por qué la ventaja es pequeña (y honesto):** en SMCI el agente ya va casi siempre corto y el régimen también
  sesga a corto, así que M5/M8/M10 apuestan en la misma dirección la mayor parte del tiempo; M10 extrae algo más
  de señal que M8 y que el agente, pero el margen es modesto a este tamaño de muestra.
- **Aportación:** un **meta-learner desplegable e interpretable por sus features** (las señales de STRATA) que,
  en un benchmark justo, **recupera la dirección y supera al agente, a la regla a mano y a lo trivial**. No
  genera alfa: es disciplina estadística, no una máquina de batir al mercado.

## 4. Límites declarados y trabajo futuro (cuantificado)

Por qué la generalización **no se hizo ahora** (respuesta lista para el tribunal):
- **Multi-agente:** reentrenar/recalibrar las cinco personalidades del agente sobre otro motor LLM exige
  ≈ cientos de inferencias por día y por activo → **varios meses de presupuesto cloud**.
- **Multi-activo:** requiere una **caché de decisiones del agente por activo**, posterior al cutoff del LLM
  (≈ 400 días por activo) → coste de inferencia y de tiempo por cada activo nuevo.
- **Despliegue en vivo:** integración en tiempo real (la tercera extensión).

Las tres **exceden el horizonte temporal y de presupuesto de un TFG** y se identifican como las extensiones
inmediatas (cap. 5). Así la aspiración deja de ser vaga: es una **limitación cuantificada y razonable**.

## 5. Estructura de la memoria

| Cap. | Contenido | Estado |
|---|---|---|
| 1 | Introducción (problema, objetivos, esquema) | esqueleto |
| 2 | Estado del arte | borrador |
| **3** | **Marco teórico** (4 bloques → ver `estructura_cap3.md`) | **rehaciendo** |
| 4 | Marco práctico (caso de estudio SMCI: M5 → M8 → M10 → trivial; ablación; SHAP) | esqueleto — **no tocar aún** |
| 5 | Conclusiones + **trabajo futuro = el foco real** (generalización; coste en §4) | esqueleto |

## 6. Tabla canónica mínima (SMCI, OOS 250 d, walk-forward, embargo=1)

| Estrategia | Accuracy | Sharpe | Equity |
|---|---:|---:|---:|
| **M10-WF ensemble (mi modelo)** | **0,552** | **+1,84** | **3,24×** |
| Trivial — "siempre mayoritaria" (corto) | 0,516 | — | — |
| M8 (regla a mano STRATA) | 0,496 | +0,33 | 1,02× |
| M5 (agente solo) | 0,484 | −0,24 | 0,98× |
| Trivial — B&H (comprar y mantener) | 0,484 | +0,03 | 0,71× |

**Tests (qué sobrevive y qué no) — coherente con la línea roja "cada cifra con su test":**
- **block-perm M10 vs B&H:** p = 0,047 → **NO sobrevive** la multiplicidad (Bonferroni-5 ≈ 0,28).
- **sign M10 vs 0,5:** p ≈ 0,06 (binomial, `…valtest_robustez.json`) / 0,11 (`RESULTADOS §1bis`) → **reconciliar
  en cap. 4**; no significativo en ninguno.
- **M10 vs M5 / M8:** M10 no bate al agente ni a la regla de forma significativa (margen nominal).
- **P(Sharpe>0) M10 = 0,976** (Sharpe positivo con alta prob.; penalizada por las configs exploradas ≈0,72 →
  el Sharpe se trata como ilustración económica, la prueba del TFG es la accuracy).
- **Ablación (interpretabilidad):** M10 solo-agente 0,468 → 0,552 con las 22 (las 7 señales STRATA aportan +8 pp,
  McNemar 0,053) → el meta-aprendiz sí usa la señal de STRATA.

**Lectura:** ningún test sobrevive la corrección por multiplicidad → la ventaja es **NOMINAL**. Es lo esperable a
≈250 días; la significancia plena es trabajo futuro. *Fuentes:* `outputs/experiments/m10_smci_valtest_robustez.json`,
`smci_config_study.json`, `RESULTADOS_OBJETIVO.md §1bis`.

## 7. Dónde está cada cosa (fuente única — no duplicar)

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
| Revisar/redactar con el consejero persistente | `memoria/CONSEJERO.md` (`/output-style consejero-tesis`) |

## 8. Líneas rojas (no negociables)

- **STRATA supervisa; M10 predice.** STRATA produce las señales (régimen/RAM/PSA/GSO); M10 es el meta-learner
  que predice la dirección a partir de ellas. No confundir las dos cosas.
- **Sin look-ahead:** posición de `t` × retorno de `t+1` (`signal_lag=1`); walk-forward, nunca KFold ni CPCV
  para el resultado desplegable.
- **Cada cifra con su test** (McNemar, sign, bootstrap por bloques, P(Sharpe>0) corregida) y desde JSON, nunca a mano.
- **Honestidad:** lo nominal se dice nominal; lo no significativo, también. La significancia plena = trabajo
  futuro, no se infla.
