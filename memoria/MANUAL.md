# MANUAL — lo esencial de la memoria (léeme en cada sesión)

> La estrella polar del TFG. Breve y claro: **qué demostramos, los objetivos, la estructura y las cifras
> canónicas**. Si una cifra o decisión no está aquí o en las fuentes que enlazo, **no entra en la memoria**.
>
> **⚠️ Fuente de verdad de la parte práctica (2026-06-24): [`MARCO_PRACTICO_CONTEXTO.md`](../MARCO_PRACTICO_CONTEXTO.md)** —
> caso central SPY + **panel de 10** (sin apéndice), pooled-10 (M8 +0,60/M10 +1,12/AutoML +1,08), ley leverage sobre 10 (p=0,093), TOST/DiD,
> alcance = rescate + riesgo (alfa = línea futura). Las cifras de SMCI/CPCV de abajo son históricas.
>
> **[Actualizado 2026-06-23 — cambio de enfoque.]** SMCI deja de ser el activo del caso de estudio. El caso
> central pasa a ser **SPY**, con un **panel de 15 activos** (10 beneficiados) y **clustering**. El brief
> completo del marco práctico (cap. 4) es **`memoria/MARCO_PRACTICO_SPEC.md`** (lo que ahí se dice manda sobre
> este resumen para el cap. 4). Lo viejo (M10 0,552 en SMCI como titular) queda como archivo histórico.

---

## 1. Qué demostramos

Un **agente LLM** de trading (AI Hedge Fund, 5 personalidades) decide cada día una posición. Sin supervisar
**pierde y acierta la dirección menos del 50 %**. Sobre esa decisión, **STRATA** —una capa de supervisión
estadística con tres detectores clásicos de series temporales— produce señales que permiten **rescatar al
agente** y, en los activos cuya naturaleza lo permite, **batir también a las estrategias triviales**.

> **Glosario de alcance (importante).** En esta memoria, **STRATA** designa la **capa de supervisión estadística
> que produce señales** sobre cada decisión del agente mediante los detectores **RAM/PSA/GSO**. De STRATA salen
> **tres estrategias derivadas**:
> - **M8** — la **regla a mano** (override-C con gate τ sobre el régimen). Referencia interpretable.
> - **M10** — el **meta-learner canónico** (XGBoost sobre las 22 features = 15 del agente + 7 de STRATA/régimen).
> - **AutoML-M10** — la **búsqueda automática** de modelo (H2O AutoML) sobre las mismas features.
>
> «STRATA supervisa; las derivadas predicen.» No confundir la capa de señales con sus tres explotaciones.

**Las tres tesis primarias** (fuente: `MARCO_PRACTICO_SPEC.md §0.A`):

> **T1.** Un agente LLM de trading sin supervisión es **direccionalmente perdedor** sobre el OOS.
>
> **T2.** STRATA **rescata al agente sistemáticamente** y aporta valor medible: tanto en **acierto direccional**
> como en **control de riesgo**. Las excepciones se documentan y se argumenta el mecanismo, no se esconden.
>
> **T3.** En los activos cuya naturaleza lo permite, las derivadas de STRATA (M8, M10 o AutoML) **superan también
> a las triviales** (Buy & Hold y «siempre la clase mayoritaria» / ZeroR). Ocurre menos veces que el rescate del
> agente, pero es un objetivo igualmente fuerte; donde no se cumple, se reporta sin maquillar con su mecanismo.

**Encuadre (lo que es esta tesis).** El objetivo de fondo es una **estrategia de supervisión estadística
desplegable en tiempo real**, concebida para supervisar a **cualquier agente** operando **cualquier activo**.
Esta memoria entrega un **caso central (SPY)** más un **panel de robustez (15 activos)** como prueba de concepto.
La **generalización —multi-agente, multi-activo y despliegue en vivo— es trabajo futuro** (cap. 5; coste en §4).

**Falsable.** Si el agente solo no es perdedor direccional (cae T1), o si las derivadas de STRATA no rescatan al
agente ni en accuracy ni en riesgo en la mayoría del panel (cae T2), la tesis cae. T3 es un objetivo fuerte pero
no bloqueante: su no-cumplimiento en parte del panel se reporta con mecanismo. Lo que **no** se hace es inflar
significancia (ver §8). Lo que no funcionó se documenta en `falsacion/`.

## 2. Objetivos de la TESIS (van en el cap. 1)

**Objetivo general.** Diseñar y validar una **estrategia de supervisión estadística desplegable en tiempo real**
que, mediante detectores clásicos de series temporales (régimen, cambio estructural y volatilidad), supervise las
decisiones de un agente de trading y **rescate su acierto direccional y su perfil de riesgo sin fuga temporal**,
con vocación de generalizar a cualquier agente y cualquier activo.

**Objetivos específicos** (mapean a los O1–O8 del SPEC):
1. **Establecer el problema (T1):** el agente solo (**M5**) es perdedor direccional (acc < 0,5; sign test).
2. **Formalizar** los tres detectores (HMM gaussiano, GARCH(1,1)-t, BOCPD) y la capa de intervención, con
   demostraciones y citas (cap. 3).
3. **Rescate en accuracy (T2):** una derivada de STRATA (M8/M10/AutoML) supera al agente en acierto direccional
   (McNemar pareado / bootstrap estacionario), sin look-ahead (walk-forward, embargo=1, `signal_lag=1`).
4. **Rescate en riesgo (T2):** STRATA mejora ΔSharpe / ΔMaxDD / ΔCalmar del agente (bootstrap pareado pooled sobre
   el panel).
5. **Interpretabilidad (T2):** la señal informativa de un meta-learner reside en las **features de STRATA**
   (cuota SHAP STRATA ≥ 50 %; ablación: quitar STRATA degrada hacia el nivel del agente).
6. **Triviales (T3):** las derivadas de STRATA baten a B&H y a la mayoritaria/ZeroR en una fracción razonable del
   panel; donde no, se argumenta el mecanismo.
7. **Explicabilidad del ML vs regla:** donde la regla determinista (M8) bate al ML (M10/AutoML), se explica el
   mecanismo. *(La vieja «universalidad» —que el ML no debe batir a la regla— queda desechada.)*
8. **Evaluar con rigor** (tests pareados, bootstrap por bloques, DSR/P(Sharpe>0) corregida) y **delimitar
   honestamente los límites** (no-significancia a muestra corta; generalización = trabajo futuro).

## 3. El claim canónico (lo que defendemos, sin adornos)

- **Caso central: SPY.** Activo amplio con *leverage effect* fuerte, donde el régimen del HMM tiene contenido
  direccional. Análisis extenso de las decisiones ex-ante, calibraciones, los tres detectores, y las tres
  derivadas en orden: **M8 → M10 + SHAP → AutoML** (`MARCO_PRACTICO_SPEC.md §1`).
- **Panel de 10 activos).** Robustez de la hipótesis de que STRATA aporta valor: pruebas por
  activo y **pooled bootstrap** (Sharpe, MaxDD, Calmar con IC95), robustez a ventanas/particiones/régimen y a la
  ventana de calibración. Config = **panel canónico mm25** (la de `decision_automl`). Criterio de inclusión de
  los 10 = mecanístico y ex-ante (ver `REVISAR_AL_VOLVER.md`; decisión pendiente §6.1 del SPEC).
- **Clustering.** Por qué ciertas estrategias funcionan en ciertos activos (KMeans/Ward/GMM/Spectral; silhouette,
  BIC, Rand). **Exploratorio**: hipótesis de reglas de aplicación por naturaleza del activo, no confirmatorio.
- **Métrica central: accuracy direccional** sobre `y_{t+1} = 1{r_log(t+1) > 0}`. Valor complementario reconocido
  en **Sharpe, MaxDD, Calmar, equity** (control de riesgo), expresados como resultado de respaldo, no como prueba
  principal.
- **Honestidad sobre el techo:** a este tamaño de muestra (OOS post-cutoff del LLM, n≈250–400) la dirección diaria
  de un activo individual es casi un paseo aleatorio; las ventajas en accuracy frente a B&H/ZeroR son **nominales**
  y no siempre sobreviven la corrección por multiplicidad. Eso se dice tal cual: la aportación es **el protocolo de
  supervisión interpretable** (rescate del agente + disciplina de riesgo), no una máquina de generar alfa.
- **Entregable:** `notebooks/STRATA_marco_practico.ipynb` (el que produce `MARCO_PRACTICO_SPEC.md`).
  `STRATA_SMCI.ipynb`, `strata_canonical.ipynb` y `decision_automl.ipynb` quedan como **inspiración/archivo**
  (ideas y config, NO datos; ver SPEC §9).

> **Cifras canónicas del cap. 4: pendientes del notebook nuevo.** Las cifras que vayan al cap. 4 salen del
> `STRATA_marco_practico.ipynb` y de sus JSON canónicos, no de los notebooks viejos. Hasta que el notebook exista,
> las cifras de SPY de referencia son las del método en `RESULTADOS_OBJETIVO.md §1` (M5 0,384 / M8 0,436 / M10
> 0,539 CPCV / B&H 0,569), marcadas como **provisionales** y a regenerar con la config mm25 / walk-forward.

## 4. Límites declarados y trabajo futuro (cuantificado)

Por qué la generalización **no se hizo ahora** (respuesta lista para el tribunal):
- **Multi-agente:** reentrenar/recalibrar las cinco personalidades sobre otro motor LLM exige ≈ cientos de
  inferencias por día y por activo → varios meses de presupuesto cloud.
- **Multi-activo (más allá del panel):** requiere caché de decisiones del agente por activo, posterior al cutoff
  del LLM (≈ 400 días por activo) → coste de inferencia y tiempo por cada activo nuevo.
- **Despliegue en vivo:** integración en tiempo real (la tercera extensión).
- **Significancia plena:** límite de potencia a muestra corta (no se fabrica con data augmentation; ver
  `docs/consideraciones.md`). Se consigue con más tiempo real o pool de activos, no inflando n.

Las extensiones exceden el horizonte de un TFG y se identifican como las líneas inmediatas (cap. 5).

## 5. Estructura de la memoria

| Cap. | Contenido | Estado |
|---|---|---|
| 1 | Introducción (problema, objetivos T1/T2/T3, esquema) | esqueleto |
| 2 | Estado del arte | borrador |
| **3** | **Marco teórico** (4 bloques → `estructura_cap3.md`) | **rehaciendo** |
| **4** | **Marco práctico** (4 secciones → `MARCO_PRACTICO_SPEC.md`: SPY · panel 10/15 · clustering · límites) | **se reescribe al aparecer el notebook (disparador automático)** |
| 5 | Conclusiones + **trabajo futuro = el foco real** (generalización; coste en §4) | esqueleto |

**Disparador del cap. 4:** un hook (`.claude/hooks/marco_practico_watch.py`) + watcher detectan el notebook
`STRATA_marco_practico.ipynb`; en cuanto aparece, se reescribe el cap. 4 según el SPEC, sin esperar aprobación.

## 6. Estructura del cap. 4 (resumen del SPEC — la fuente es `MARCO_PRACTICO_SPEC.md`)

1. **Caso de estudio SPY:** decisiones ex-ante justificadas; calibraciones (HMM/GARCH/BOCPD, umbrales, prior
   data-driven); qué hace cada detector y su efecto; tasa de intervención y de éxito por detector; las tres
   derivadas (M8 → M10+SHAP → AutoML); tests de significancia (accuracy central; Sharpe/MaxDD/equity de respaldo);
   mucha gráfica (intervención por detector, ablación, equity con todas las estrategias, regímenes HMM, scores con
   umbrales, McNemar).
2. **Panel 10/15:** pooled bootstrap (Sharpe/MaxDD/Calmar IC95); tabla y heatmap por activo y estrategia
   (M5/M8/M10/AutoML/ZeroR/B&H); robustez a ventanas/particiones/régimen; ablación de calibración; subsección a
   los activos especialmente significativos.
3. **Clustering:** KMeans/Ward/GMM/Spectral; PCA/t-SNE/UMAP; silhouette/BIC/Rand; lectura económica de cada
   cluster; hipótesis de reglas por naturaleza del activo (exploratorio).
4. **Límites y futuras líneas:** límites ilustrados con datos del propio notebook; soluciones ancladas en lo
   hallado.

**Detectores — hallazgo honesto a vigilar:** si en la intervención solo el de régimen (RAM) tiene efecto medible
y PSA/GSO quedan inertes en un activo, **aparece explícito** (la motivación eran tres ejes ortogonales; lo
empírico manda). Bueno y malo, siempre justificado por mecanismo.

## 7. Dónde está cada cosa (fuente única — no duplicar)

| Necesito… | Voy a… |
|---|---|
| **Brief del cap. 4 (manda)** | **`memoria/MARCO_PRACTICO_SPEC.md`** |
| Dudas/decisiones que dejé pendientes | `memoria/REVISAR_AL_VOLVER.md` |
| Decisiones vivas del proyecto | `DECISIONES_ESENCIALES.md` |
| Cifras de referencia (SPY = §1; SMCI §1bis = histórico) | `RESULTADOS_OBJETIVO.md` |
| Decisiones con cita académica | `decisiones_respaldadas_literatura.md` |
| Lo que probé y descarté (negativos) | `falsacion/INDICE.md` |
| Hallazgos nuevos del giro (riesgo, dial τ, dos canales, AutoML) | `docs/*_EXPLORATORIO.md` |
| Objeciones del tutor + respuestas | `memoria/objeciones_tribunal.md` |
| Estructura del cap. 3 | `memoria/estructura_cap3.md` |
| Reglas de estilo + anti-IA | `memoria/ESTILO_Y_ANTIIA.md` y `correcciones_aprendidas.md` |
| Figuras clave | `graficas_clave.md` |
| Revisar/redactar con el consejero | `memoria/CONSEJERO.md` |

## 8. Líneas rojas (no negociables)

- **STRATA supervisa; las derivadas (M8/M10/AutoML) predicen.** No confundir la capa de señales con su explotación.
- **Sin look-ahead:** posición de `t` × retorno de `t+1` (`signal_lag=1`); walk-forward, nunca KFold ni CPCV para
  el resultado desplegable (única excepción: M3, demostración del sesgo).
- **Cada cifra con su test** (McNemar, sign, bootstrap por bloques, DSR/P(Sharpe>0) corregida) y **desde JSON
  canónico**, nunca a mano ni de notebooks obsoletos.
- **Honestidad:** lo nominal se dice nominal; lo no significativo, también. No se nombra «p-hacking» en el
  notebook, pero tampoco se infla significancia ni se esconden resultados malos importantes.
- **Métrica central = accuracy**; Sharpe/MaxDD/Calmar/equity = respaldo (riesgo), no prueba principal.
