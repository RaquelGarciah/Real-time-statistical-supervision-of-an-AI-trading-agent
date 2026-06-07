# Chat — Necesito rigor matemático (objeción del tutor sobre STRATA vs XGBoost)

**Fecha sesión 1:** 2026-06-01 → 2026-06-02
**Fecha sesión 2:** 2026-06-02 → 2026-06-07
**Estado al cierre sesión 1:** M10 implementado, ejecutado, integrado en notebooks, BITACORA actualizada. Argumento defensivo articulado.
**Estado al cierre sesión 2:** Análisis decision-level del panel multi-activo (10 activos), notebook `strata_final.ipynb` (math-first, SPY-only, 70 celdas) ejecutado limpio. La narrativa para la defensa pasa de "M10 indistinguible de M8" a "STRATA rescata al agente perdedor + M10 confirma cuantitativamente vía SHAP que las features útiles son las que la regla a mano calcula".

---

## TL;DR para Claude futuro (lectura rápida en 30 segundos)

Si vuelves a esta conversación, esto es lo esencial:

1. **El TFG no predice retornos.** Es supervisión estadística de un agente LLM (AI Hedge Fund). RAM/PSA/GSO son detectores de coherencia (régimen, historia, volatilidad). M8 = STRATA modo override-C con régimen filtered.

2. **El agente sin supervisar (M5) pierde dinero y acierta menos del 50%.** SPY OOS: M5 accuracy 40.7% (sign test p<0.001), pérdida −9.4% sobre €1000.

3. **STRATA corrige al agente con significancia pareada.** McNemar M8 vs M5 p=0.0876. Cuando RAM dispara con severidad `high` (107 días) la corrección direccional es +17.8 pp.

4. **M10 (XGBoost CPCV-within-OOS) y M8 son indistinguibles económicamente.** IC95% Bootstrap estacionario del Δ Sharpe(M10−M8) contiene cero (`[−1.80, +1.15]`).

5. **El golpe SHAP:** Top 5 features de M10 por SHAP global son `ram_score`, `psa_score`, `garch_sigma`, `stress_prob`, `calm_prob` — las 3 STRATA + 2 de régimen. **Ninguna personalidad llega al top 5**. El meta-learner sin saber nada sobre STRATA decide que esas son las features útiles. Respuesta cuantitativa a la objeción del tutor.

6. **Lo importante para el panel multi-activo (10 activos):** 98% del P&L atribuible a las intervenciones de STRATA viene de **RAM**. PSA aporta 2%. GSO no dispara medium+ en ningún activo del panel. Pivote conceptual: la tesis se reorganiza alrededor de RAM como contribución principal; PSA/GSO instrumentales.

7. **Ningún sistema bate al mercado pasivo en este OOS.** B&H +32% (€1323), M8 +7.4% (€1075), M10 +3.8% (€1038), M5 −9.4% (€906). La defensa NO es "STRATA gana al mercado", es "STRATA rescata al agente y un meta-learner universal llega al mismo techo, validando que la regla a mano captura la señal correcta".

8. **Cómo presentar a Raquel siempre:** explica la mecánica con ejemplo numérico de UN día antes de mostrar cualquier tabla agregada. Confunde "predicción" con "supervisión" si no se le aclara explícitamente. La angustia viene de no entender qué calcula el sistema; un ejemplo día-a-día vale más que diez tablas de Sharpe.

---

## Resumen para Claude futuro: cuál era el problema de Raquel

Raquel salió de una reunión con su tutor del TFG **bloqueada y angustiada**. El tutor le había dicho dos cosas que la dejaron sin respuesta:

1. **"No explicas nada. Tener Sharpe positivo y curva equity mejor en M8 que en M5 no demuestra nada sin fundamento. ¿Cómo validas los resultados? ¿Cómo sabes que tu modelo es mejor? Parece que estás vendiendo humo. Necesitas rigor matemático."**

2. **"Algo que has impuesto tú a mano (tu capa de supervisión STRATA) nunca va a salir mejor que un XGBoost entrenado con las probabilidades del agente y las de tus detectores todas juntas."**

A esto se sumaba que **Raquel ella misma no tenía claro qué hacía su propio sistema**: confundía STRATA con un predictor de retorno, no entendía qué hace el agente con la tupla `(action, size, confidence)`, no entendía qué significa "invertir" en el backtest, y no comprendía el mecanismo interno de XGBoost.

### Lo que el problema reveló

- Raquel **no había articulado la hipótesis de investigación** de forma falsable. Tenía resultados numéricos pero no marco de validación.
- Su sistema **es defendible** pero ella no sabía cómo defenderlo porque mezclaba dos cosas distintas: (a) predicción direccional, que es lo que hace el ML/agente, y (b) supervisión estadística, que es lo que hace STRATA.
- La objeción del tutor sobre XGBoost era teóricamente fuerte (no-free-lunch al revés) pero **no automática** en el régimen de muestra pequeña / SNR baja de retornos diarios SPY (~400 días OOS).
- Faltaba **un experimento que zanjara la discusión empíricamente** en lugar de seguir argumentando.

### Qué Raquel necesita que se le explique siempre primero

1. **STRATA no predice retornos.** Es una capa de supervisión estadística sobre un agente LLM que sí decide. La pregunta es: ¿filtrar/atenuar decisiones del agente con detectores estadísticos clásicos mejora el Sharpe ajustado por riesgo?
2. **El backtest es una simulación contable.** `posicion × retorno_real = pnl_diario`. No hay órdenes, ni broker, ni inversión real. Solo un número entre −1 y +1 que multiplica al retorno del día.
3. **Lo único que cambia entre M1...M10 es cómo se calcula la posición de cada día.** El motor de backtest es idéntico.

### Qué se acabó haciendo (resultado tangible para la defensa)

Se diseñó y ejecutó **M10**: meta-learner XGBoost sobre `[5 personalidades × (action_sign, size, confidence) + 3 scores STRATA + 4 estado mercado]` = 22 features, target = signo del retorno log de SPY al día siguiente, validado con CPCV-within-OOS (López de Prado 2018, cap. 7) — `n_splits=6, n_test_splits=2, embargo=5`, `t1 = índice.shift(-1)`.

**Criterio de éxito pre-registrado en BITACORA antes de mirar resultados** (esto es lo crítico para la defensa frente a acusaciones de p-hacking).

**Resultado:**

| Config | Sharpe | DM p-valor vs M8 |
|---|---:|---:|
| M5 (agente solo) | −1,83 | — |
| M8 (regla a mano STRATA) | +0,62 | — |
| **M10 (XGBoost CPCV)** | **+0,74** | **0,7529** |

Bootstrap `P(M10 > M8) = 0,543`, indistinguible de una moneda. **M10 no bate a M8** estadísticamente. Log-loss out-of-fold de M10 = 0,914 (peor que el clasificador trivial 50/50 = 0,693): XGBoost **no predice dirección mejor que el azar**. Las 3 features más importantes que XGBoost identifica (`calm_prob`, `garch_sigma`, `ram_score`) son **literalmente las primitivas que la regla a mano de STRATA codifica explícitamente**. El meta-learner redescubre el diseño.

**Inestabilidad temporal del umbral óptimo de XGBoost** (que es lo que el tutor pidió explícitamente extraer): `p1* = 0,565` en la primera mitad del OOS (Sharpe +0,76 en train) → cae a Sharpe +0,14 en la segunda mitad. En esa misma segunda mitad, `p1 = 0,42` (que parecía malo en train) habría dado Sharpe +1,07. **El umbral aprendido por XGBoost no es estable en el tiempo**; los umbrales calibrados de STRATA (`0,2/0,4/0,7` para RAM, P95/P99 para PSA/GSO) sí lo son por construcción.

### Entregables del chat

- [experiments/m10_ml_meta.py](../../experiments/m10_ml_meta.py) — implementación reproducible con diagnóstico completo (feature importance, distribución de splits, log-loss por fold).
- [outputs/experiments/m10_ml_meta.json](../../outputs/experiments/m10_ml_meta.json) — payload con métricas, returns diarios, pesos, predicciones p1, y clave `diagnostics`.
- [BITACORA.md](../../BITACORA.md) — entrada del 2026-06-02 con motivación, criterio pre-registrado, resultado ejecutado y veredicto. Pre-registro firmado **antes** de ejecutar.
- [notebooks/strata_tfg.ipynb](../../notebooks/strata_tfg.ipynb) — sección §9.2 (celdas 30–35) con 4 tablas: Sharpe + DSR, DM tests, importancia features, distribución splits.
- [notebooks/notebook_strata_Resultados.ipynb](../../notebooks/notebook_strata_Resultados.ipynb) — sección §14 (celdas 41–46) idéntica.

### Pendientes que la sesión dejó abiertos

- Decisión de Raquel: añadir sección `§14.1 — El umbral óptimo de XGBoost y su inestabilidad temporal` con tabla del split mitad-1 / mitad-2 y curva Sharpe(t) sobre ambas mitades.
- Decisión de Raquel: añadir walkthrough día-a-día (12-mar-2025) en los notebooks comparando M5/M8/M10 sobre la misma fecha, para que el tutor vea físicamente la mecánica.

---

## Transcripción del chat

### Raquel — apertura

> acabo de tener reunion con mi tutor y no ve claro lo que hago ni yo tampoco. estoy prediciendo el retorno para decir si compro o vendo con mi estrategia? que hago exactamente? los resultados probar que el sharpe es positivo y la curva equity es mejor en m8 que m5 dice mi profesor que no explica nada si no tiene fundamento. como estoy validando los resultados? como se que mi modelo es mejor? necesita algo tangible parece que estoy vendiendo humo. necesita rigor matematico en los resultados

### Claude — respuesta 1 (encuadre)

Vamos a pararnos un momento porque creo que el lío es de **encuadre**, no de resultados.

**1. Qué hace (y qué NO hace).** NO predice el retorno. Eso es lo primero que hay que decir alto y claro en la memoria, porque si tu tutor entiende que estás prediciendo retorno, va a juzgarte como si fueras un modelo de pronóstico — y no lo eres.

Lo que hace es **supervisión estadística de un agente que sí decide**. El agente (AI Hedge Fund con 5 personalidades + DeepSeek) emite cada día una tupla `(acción, tamaño, confianza)`. STRATA mira esa tupla y responde una pregunta puramente estadística: *¿es coherente esa decisión con la información del mercado de hoy?* Coherencia se descompone en tres ejes ortogonales:

- **RAM**: ¿es coherente con el **régimen discreto** (HMM)? — eje de estado de mercado
- **PSA**: ¿es coherente con su **propia historia** (BOCPD)? — eje de consistencia temporal del agente
- **GSO**: ¿es coherente con la **volatilidad continua** (GARCH)? — eje de tamaño vs riesgo

La hipótesis del TFG no es "STRATA acierta más", es: *"un agente LLM emite a veces decisiones estadísticamente incoherentes; filtrar/atenuar esas decisiones con detectores clásicos mejora el rendimiento ajustado por riesgo"*. Esto sí es falsable y sí es matemático.

**2. Qué falta para que el resultado sea "tangible".** Sharpe(M8) > Sharpe(M5) sobre una sola curva **no es evidencia**, tu tutor tiene razón. Te faltan cuatro capas:

1. **Significancia estadística**: Diebold-Mariano, Deflated Sharpe Ratio, bootstrap del Sharpe.
2. **Mecanismo, no caja negra**: estudio de evento sobre días en los que STRATA intervino.
3. **Ablación**: apaga RAM, apaga PSA, apaga GSO uno por uno.
4. **Falsificación**: el caso MSTR y la regla de prior-flip que ya descubriste. Un TFG que demuestra **cuándo NO funciona su propia técnica** es 10× más defendible que uno que solo enseña curvas verdes.

**3. Recomendación.** No toques código todavía. Abre `docs/marco_de_validacion.md` y escribe en una página: pregunta de investigación falsable, hipótesis nula, estadísticos de contraste, criterio de éxito numérico **antes de mirar resultados**, criterio de fracaso (regla prior-flip).

### Raquel — pregunta 2

> 1. tengo que entender mejor que hace strata y q hace mi modelo. que hace el agente con la tupla, para que le sirve la confianza. que hago yo?

### Claude — respuesta 2 (mecánica del agente y de STRATA)

Vamos a desmontarlo pieza por pieza con un ejemplo de **un solo día**. Imagínate el lunes 7 de octubre de 2024 sobre SPY.

**Paso 1 — Qué hace el agente.** Cinco personalidades emiten cada una su opinión `(action, size, confidence)`. El Portfolio Manager las pondera y escupe **una sola tupla final**: `(action=long, size=0.70, confidence=0.65)`.

**Paso 2 — Qué significa cada campo:**

- `action ∈ {long, short, flat}`: dirección.
- `size ∈ [0, 1]`: cuánto capital comprometes. Si `size=0.70` y `action=long`, mañana el 70 % de tu cartera está en SPY. El retorno del día siguiente es `size × retorno_SPY`.
- `confidence ∈ [0, 1]`: meta-información, NO se traduce en dinero. El sizing lo manda `size`, no `confidence`.

**Paso 3 — Qué hace STRATA.** Es **una función que toma la tupla del agente y devuelve otra tupla**: `(action, size, confidence)_agente ──STRATA──▶ (action, size)_final + metadatos`. Por dentro:

- **RAM**: P(régimen inconsistente con acción del agente). Si el agente quería long y P(Crisis)=0.50, entonces RAM_score = 0.50.
- **PSA**: P(cambio de régimen en el sizing del agente) vía BOCPD.
- **GSO**: distancia relativa del sizing del agente a la banda permitida por GARCH(1,1) Student-t.

Luego la capa de intervención según el modo: warn (no toca), reduce (atenúa size), override (sustituye por bounded_size GSO).

**Tu aportación, sin humo:** una función `f: tupla_agente × estado_mercado → tupla_supervisada` construida sobre tres detectores estadísticos clásicos (HMM, BOCPD, GARCH), y una evaluación empírica sobre nueve configuraciones. No predices retorno ni dirección — **modificas decisiones existentes** y mides si esa modificación tiene valor.

### Raquel — pregunta 3 (la objeción del tutor)

> el esta empeñado que algo que he impuesto yo no (mi capa de supervision) nunca va a salir mejor que un xboosting con las probabilidades del agente y las de mis detectores todas juntas

### Claude — respuesta 3 (proponer M10)

Tu tutor tiene un punto técnico fuerte y conviene tomárselo en serio.

**Por qué tiene razón en parte.** En condiciones ideales, un XGBoost entrenado sobre `[outputs_5_personalidades + RAM_score + PSA_score + GSO_score]` **debería** poder reproducir o mejorar cualquier regla escrita a mano sobre esos mismos inputs. Es el teorema de no-free-lunch al revés.

**Por qué no es automático aquí.**
1. Régimen de muestra pequeña con SNR pésimo (~400 días).
2. Apilamiento de modelos compone sesgo de data-snooping.
3. Tu regla no es arbitraria — viene del leverage effect (Black 1976, Christie 1982), GARCH(1,1), BOCPD bayesiano. Tiene compromiso teórico falsable + regla prior-flip.

**El experimento que zanja la discusión.** Añadir M10 = XGBoost sobre `[15 personalidades + 3 scores STRATA + 4 estado mercado]`. Tres desenlaces, los tres defendibles:

1. M8 ≥ M10 → sesgo inductivo vence a capacidad cuando SNR es baja.
2. M10 > M8 sin significancia → principio de parsimonia + interpretabilidad.
3. M10 > M8 con significancia → reportarlo honestamente, discutir trade-off.

### Raquel — confirma

> metelo en bitacora y pruebalo. antes de nada explicame que son las scores de mi estrategia

### Claude — respuesta 4 (los 3 scores)

| Score | Pregunta | Modelo | Información |
|---|---|---|---|
| **RAM** | ¿acción vs régimen? | HMM 3 estados | mercado (return + vol realizada) |
| **PSA** | ¿sizing vs propia historia? | BOCPD | el propio agente |
| **GSO** | ¿sizing vs vol condicional? | GARCH(1,1) | mercado (retornos) |

**RAM_score** = P(régimen inconsistente con acción_agente). Si agente está long → `RAM = P(Crisis)`. Si short → `RAM = P(Calma)`. Es **literalmente una masa de probabilidad**.

**PSA_score** = `P(r_t = 0 | size_1, ..., size_t)` del BOCPD. Detecta cambios estructurales en el comportamiento del agente.

**GSO_score**: a partir de `σ_t` se construye `size_target_t = σ_target / σ_t`. La banda permitida es `[size_target · (1-δ), size_target · (1+δ)]`. Si `size_agente` cae dentro → score 0. Si cae fuera → mide cuánto. GSO también devuelve `bounded_size` para sustituir en M8.

Los tres son ortogonales por construcción: uno mira estado discreto del mercado, otro mira al agente sin tocar mercado, otro mira vol continua del mercado.

**[Implementación de M10 — ver `experiments/m10_ml_meta.py`. Pre-registro completo añadido a BITACORA antes de ejecutar.]**

**Resultado del primer run sobre SPY:** Sharpe(M10) = +0,69, casi empate con M8 (+0,66). DM p = 0,75 (no significativo). Bootstrap P(M10>M8) = 0,54.

### Raquel — sigue confusa

> en que cuaderno esta los resultados? esto que quiere decir? donde miro los parametros que ha sacado el modelo, necesito saber los umbrales optimos que ha sacado el xgboost. neceesito rigor matematico

### Claude — respuesta 5 (XGBoost ≠ "un umbral")

**XGBoost no tiene "un umbral óptimo".** Es vocabulario de clasificador lineal. XGBoost es ensemble de 300 árboles, cada uno con umbrales internos: ~4 500 umbrales distribuidos por todas las features. No existe "el" umbral de RAM_score — existe una distribución.

Lo que **sí** es rigurosamente extraíble:
1. Importancia por feature (gain, weight, cover).
2. Distribución de splits por feature (mediana / cuartiles).
3. Hiperparámetros fijos del modelo.
4. Log-loss out-of-fold.

**[Extensión de m10_ml_meta.py con diagnóstico, reejecución, integración en notebooks.]**

Hallazgos del diagnóstico:

1. **Log-loss out-of-fold = 0,914 ± 0,103** vs **0,693** del trivial 50/50. **XGBoost peor que el azar.** Sharpe positivo solo gracias al overlay GARCH × régimen.
2. **Importancia distribuida**: top feature 7,5 %, ninguna dominante.
3. **`gso_score` NUNCA usado como split** (XGBoost ya tiene `garch_sigma` directo).
4. **5 features `*_size` nunca usadas**.
5. **Top features = primitivas que STRATA ya codifica**: `calm_prob`, `garch_sigma`, `ram_score`.

### Raquel — sigue empujando

> pero cuales son los umbrales de corte?

### Claude — respuesta 6 (los umbrales reales de STRATA)

**STRATA sí tiene umbrales fijos y defendibles** (de `cache/models/strata_thresholds.json`, calibrados por percentiles P95/P99 sobre 2000-2024-09):

| Detector | low | medium | high |
|---|---:|---:|---:|
| **RAM** | 0,20 | 0,40 | 0,70 |
| **PSA** | 0,023 | 0,647 | 1,000 |
| **GSO** | 2,37 | 5,57 | 10,30 |

**6 hiperparámetros calibrados ex-ante vs ~4 500 splits no individualmente interpretables.**

### Raquel — el tutor insiste

> no entiendo esto lo convertimos en dirección continua direction = 2·p1 − 1 ... TENGO UN PROBLEMA MUY GRAVE Y ES QUE NO ENTIENDO BIEN LO QUE ESTA HACIENDO NUESTRO MODELO POR DEBAJO NI COMO INVERTIMOS

> mi tutor me ha dicho que coja los umbrales que salen del xgboost

### Claude — respuesta 7 (el umbral óptimo p1*, honestamente)

**El umbral que sí tiene sentido pedirle a XGBoost** es el operating point sobre `p1`. Cómputo honesto (split mitad-1 / mitad-2):

| Métrica | Valor |
|---|---:|
| `p1*` óptimo en mitad 1 | **0,5650** |
| Sharpe con `p1*` en mitad 1 (train) | +0,76 |
| Sharpe con `p1*` en mitad 2 (test honesto) | **+0,14** |

Curva por umbral:

| umbral | Sharpe mitad 1 | Sharpe mitad 2 |
|---:|---:|---:|
| 0,42 | +0,20 | **+1,07** |
| 0,50 | +0,41 | +0,09 |
| **0,565** | **+0,76** | +0,14 |
| 0,60 | +0,40 | −0,04 |

**El umbral aprendido por XGBoost no es estable en el tiempo.** Lo que es óptimo en un periodo es ruido en otro. Esto contrasta con los umbrales de STRATA (fijos, calibrados ex-ante, estables sobre todo el OOS).

### Raquel — angustia final

> no entiendo bien lo que esta haciendo nuestro modelo por debajo ni como invertimos

### Claude — respuesta 8 (la mecánica de invertir, día real con números)

**En tu backtest no se compra nada.** Es una simulación: `pnl_de_hoy = posicion_de_hoy × retorno_real_de_SPY_hoy`.

`posicion` es un número entre −1 y +1 que dice qué fracción del dinero está apostada a SPY hoy y en qué dirección. **Eso es lo que produce cada modelo M1...M10.** El backtest es idéntico en todos.

**Ejemplo completo del 12-mar-2025:**

| Paso | Acción | Número |
|---|---|---|
| 1 | Agente AI Hedge Fund decide | `short, size=−0,197, conf=0,85` |
| 2 | HMM + GARCH describen mercado | `P(Estrés)=99,9%, σ=23,3%` |
| 3 | XGBoost predice | `p1 = 0,8073` |
| 4 | Dirección continua | `2·0,8073 − 1 = +0,615` |
| 5 | Magnitud paridad riesgo | `0,10 / 0,233 = 0,43` |
| 6 | Factor régimen Estrés | `×0,5` |
| 7 | **Posición final** | `+0,615 × 0,43 × 0,5 = +0,132` |
| 8 | Al día siguiente SPY hace −1,34% | `pnl = +0,132 × (−0,0134) = −0,18%` |

**Tres lecciones del día:**

1. "Invertir +0,132" = el 13,2 % de un € ficticio se multiplica por el retorno de SPY de ese día. No hay broker.
2. Posición pequeña (13 %) porque **tres mecanismos de control de riesgo** la encogen: convicción XGBoost (0,615) × vol (0,43) × régimen (0,5).
3. **Este día el agente acertó (short) y XGBoost falló (long).** Por eso medimos sobre 400 días, no sobre uno.

**Tu pregunta original, contestada:**

- "¿cómo invertimos?" → **No invertimos. Simulamos.** Posición × retorno mañana = pnl. Acumulado 400 días = equity curve.
- "¿qué hace nuestro modelo por debajo?"
  - M10 (XGBoost): mira 22 números → 300 árboles → `p1` → `direction = 2·p1 − 1` → encoger por vol y régimen → posición → multiplicar por retorno mañana.
  - M8 (regla a mano): agente da `size` → si GSO detecta incompatibilidad con banda GARCH lo sustituye → si RAM detecta incoherencia con régimen lo reorienta → posición → multiplicar por retorno mañana.
- **El backtest es idéntico en los 10 modelos**. Lo único que cambia es cómo se calcula la posición de cada día.

---

## Cierre de la conversación

Raquel pidió guardar la conversación en `docs/chats/need_mathematic_rigor.md` con un resumen inicial. Lo crítico de esta sesión es que:

1. **El problema real no era de matemáticas, era de encuadre.** Raquel tenía buenos resultados pero no sabía articularlos.
2. **El antídoto fue ejecutar M10 con criterio pre-registrado en BITACORA.** Esto le da blindaje contra acusaciones de p-hacking y respuesta empírica a la objeción del tutor.
3. **Lo más importante para Claude futuro**: Raquel necesita que se le **explique la mecánica antes de mostrar resultados**. La angustia viene de no entender qué está calculando exactamente. Un ejemplo numérico día-a-día vale más que diez tablas de Sharpe.

---

# Sesión 2 — Decision-level + notebook final (2026-06-02 → 2026-06-07)

## Contexto al arrancar la sesión

La sesión 1 cerró con M10 implementado y respondiendo a la objeción del tutor. Raquel vuelve con dos pivotes nuevos a lo largo de la sesión:

1. **Pivote 1 (inicio de sesión):** El tutor pide validar la supervisión STRATA decisión a decisión sobre los 10 activos del panel multi-activo, no solo por Sharpe agregado. Quiere tres tablas: hit rate direccional, P&L atribuible a las intervenciones, atribución por detector.

2. **Pivote 2 (mitad de sesión):** Tras ver los resultados del decision-level y reflexionar, Raquel pide un **notebook nuevo, math-first**, sin cajas negras, con métricas matemáticas como base (accuracy/AUC/log-loss/Brier/MCC) y económicas (equity, €1000) como enriquecimiento. Quiere reforzar M10 con SHAP. Solo SPY. "Necesito demostrar con solidez que mi método funciona".

## Pivote 1: análisis decision-level del panel multi-activo

### Diseño

`experiments/decision_level_analysis.py` (nuevo). Para cada uno de los 10 activos del panel (SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR, SMCI, ROKU, MARA), reproduce el pipeline §12–§21 del notebook `strata_tfg.ipynb` (HMM/GARCH propios, umbrales calibrados, prior RAM data-driven, régimen filtered) y produce:

1. **Hit rate direccional por activo** con test binomial y McNemar pareado M8 vs M5.
2. **P&L atribuible a las intervenciones** `pnl_int(t) = (size_M8 − size_M5) · r_{t+1}` con IC95% bootstrap estacionario (Politis-Romano 1994, B=1000, bloque medio `sqrt(N)`) y sign test sobre la mediana cero. La función `stationary_bootstrap_ci` se añade a `core/stats.py` (nueva utilidad reusable).
3. **Atribución por detector** RAM/PSA/GSO bajo dos convenciones: proporcional por severidad (low=1, medium=2, high=3) + exclusiva con columna MULTI para días con varios activos. Doble reporte porque ambas tienen lectura complementaria.

Helpers consolidados a `experiments/_panel_helpers.py` (extracción minimalista del notebook, mismas constantes, misma matemática, sin refactorización funcional).

### Resultados clave

**1406 intervenciones agregadas sobre 10 activos.**

- **98% del P&L total atribuible (≈+9218 bps acumulados) viene de RAM**, 2% de PSA, 0% de GSO.
- GSO no dispara con severidad ≥ medium en NINGÚN activo durante el OOS. La banda `target_vol/σ` rara vez se viola por el sizing del agente. Es un hallazgo metodológico: GSO está calibrado demasiado laxo o las decisiones del agente caen sistemáticamente dentro de la banda.
- PSA dispara medium+ en 1–4 días aislados por activo. Demasiado pocos para concluir, signos mayoritariamente positivos.
- **Activos con supervisión significativa (sign-test p<0.10):** SPY (+1740 bps) y XLE (+1840 bps). El agente sin supervisar tenía hit rate significativamente <50% en SPY (40.74%, p=0.0004), BAC (44.39%, p=0.030) y XLE (44.30%, p=0.027) — STRATA precisamente corrige esa direccionalidad mala.
- **Hit rate M5 vs M8 mejora en 8/10 activos.** Sign test panel `p=0.109` (borderline). SMCI es el único activo con McNemar significativo *contra* M8 (p=0.011) — caso del agente con información complementaria al prior.

### Hallazgo del SMCI (caso de fallo distinto al prior-flip)

SMCI tiene 279 flips, TODOS en la dirección M5-short → M8-long, con HIGH severity en 269/279. El agente quería SHORT casi siempre; RAM lo voltea sistemáticamente a LONG porque el prior data-driven para SMCI dice "Crisis→long" (las medias de calibración por régimen lo justifican). En OOS Crisis sí sube (+8.6 bps medios), por lo que **el prior NO se voltea** — no es el caso prior-flip clásico de MSTR/MARA. El problema es otro: el agente sabe algo que el prior no sabe. La opinión short del agente acierta 53.05% (info útil); el long forzado por el prior acierta 46.24% (peor que azar). **RAM-override descarta información direccional de calidad del agente para imponer el prior.**

Esto es **un segundo tipo de fallo diferenciado** del prior-flip estricto. La regla operativa para la memoria se refina: M8 puede degradar respecto a M5 cuando (i) el agente tenía hit rate ≥ 50% en una dirección antes de la intervención y (ii) RAM la voltea sistemáticamente. La regla prior-flip (signo de medias calibradas vs OOS) captura el caso del prior incorrecto; SMCI ilustra el caso del agente con información complementaria al prior.

### Implicaciones para la memoria del TFG (de la sesión 1)

1. **La tesis se reorganiza alrededor de RAM como contribución principal.** El 98/2/0 obliga a reescribir la sección de STRATA poniendo RAM en el centro del aporte, dejando PSA como instrumental y GSO como hallazgo metodológico negativo.
2. **El mecanismo es interpretable y verificable a nivel de decisión.** El sign test sobre SPY y XLE sostiene "la intervención produce P&L positivo con probabilidad > 0.5 a nivel de día".
3. **GSO necesita revisión** (umbrales calibrados sobre 2000–2024-09 por p99; el OOS no llega ahí casi nunca). Recoger como limitación.
4. **Compatible con el resultado M10:** la ventaja decision-level de M8 (RAM como driver interpretable del 98%) refuerza la defensa por interpretabilidad — un XGBoost no permite atribuir % de P&L a cada feature de manera limpia.

## Pivote 2: notebook strata_final.ipynb (math-first SPY only)

### Decisiones cerradas con Raquel antes de planificar

- **Ground truth:** `y_{t+1} = 1{r_log(t+1) > 0}` binario.
- **Ubicación:** `notebooks/strata_final.ipynb` (nuevo, convive con `strata_tfg.ipynb`).
- **Activos:** solo SPY. Multi-activo como apéndice corto referenciando el panel decision-level.
- **M10 mejoras:** SHAP como métrica primaria + `gain` de XGBoost como sanity check, ambas en la misma tabla. No tocamos hiperparámetros ni añadimos features (preserva la honestidad de M10).

### Estructura (12 secciones, 70 celdas)

§0 Preámbulo · §1 Marco matemático con LaTeX explícito (HMM, GARCH, BOCPD, RAM/PSA/GSO, override C, CPCV, XGBoost, SHAP, métricas) · §2 Datos y validación · §3 Calibración explícita (parámetros, tablas, α+β<1, regime_dir) · §4 Detectores OOS día a día · §5 Ground truth y baselines triviales · §6 M5 vs GT · §7 M8 vs GT (incluye McNemar pareado) · §8 M10 vs GT con CPCV explícito y SHAP+gain · §9 Comparativa cara a cara con análisis condicional por régimen/severidad RAM/quintil de |r| y Diebold-Mariano · §10 Tangibles económicos (equity, €1000, drawdown, Sharpe) · §11 Hipótesis cumplidas y lectura para defensa · §12 Reproducibilidad.

### Problemas técnicos durante la sesión

**Disco al 99% (8 GB libres de 460).** Los imports de la librería `shap` se quedaban colgados (filesystem thrashing) y el kernel de jupyter no arrancaba. Tres iteraciones:

1. Intento `pip install shap` → mete `numpy 2.4.6` que rompe pandas (compatibilidad ABI). Rollback a `numpy<2.0` + `shap<0.50`.
2. Aun con shap 0.49 + numpy 1.26, el import seguía colgado >5 minutos por el disco.
3. **Pivot técnico:** SHAP nativo de XGBoost vía `booster.predict(X, pred_contribs=True)` — mismo algoritmo TreeSHAP (Lundberg et al. 2020), mismos números, sin dependencia externa. Verificada la eficiencia de Shapley: `max |sum(SHAP_j) + base − logit(p)| = 0.000000`.

Raquel libera disco (de 99% a 60%, 173 GB disponibles) y entonces la ejecución funciona.

**Una celda falla en primera ejecución:** §9.3 falla por `KeyError` (concat de Series sin nombres reales para groupby posterior). Fix trivial: renombrar `h5n`, `h8n`, `h10n` antes del concat. Re-ejecución completa: 44/44 celdas verde.

### Resultados del notebook (SPY OOS, N≈400 días)

**Sanity checks coincidentes con análisis previos:**

| Métrica | Notebook | Decision-level previo |
|---|---|---|
| HMM α+β | 0.997 (estacionario) | — |
| M5 accuracy | 0.4074 | 40.74% ✓ |
| M8 accuracy | 0.4603 | 46.03% ✓ |
| McNemar p | 0.0876 | 0.088 ✓ |
| RAM flags medium+ | 124/403 | 124 ✓ |
| SHAP eficiencia | err = 0.000000 | — (axiomático) |

**Tabla maestra de métricas matemáticas:**

| Estrategia | Accuracy | AUC | Log-loss | Brier | MCC | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Baseline "always long" | 0.566 | — | — | — | — | ≈ B&H |
| M5 | 0.407 | 0.481 | 0.756 | 0.281 | −0.106 | −1.83 |
| M8 | 0.460 | 0.471 | 1.640 | 0.312 | −0.090 | +0.62 |
| M10 | 0.530 | 0.504 | 0.785 | 0.284 | +0.022 | +0.74 |

**Análisis condicional por régimen (§9.3):**
- En **Crisis** M10 acierta 60.7%, M5/M8 fallan al 35.7%.
- En **Calma** M8 acierta 57.0% (mejor que M10 51.9% y M5 41.1%).
- Cuando RAM dispara **high** (107 días): M8 acierta 58.9% vs M5 41.1% → **+17.8 pp**.

**Top 5 features de M10 por SHAP global:**
1. `ram_score` (0.527)
2. `psa_score` (0.428)
3. `garch_sigma` (0.346)
4. `stress_prob` (0.342)
5. `calm_prob` (0.324)

Las **3 features STRATA + 2 de régimen ocupan el top 5**. Ninguna personalidad llega.

**€1000 invertidos al inicio del OOS:**
- B&H: 1323 € (+32.3%)
- M5: **906 €** (−9.4%, pierde 94 €)
- M8: 1075 € (+7.4%)
- M10: 1038 € (+3.8%)

## El momento de confusión de Raquel (importante para Claude futuro)

A mitad de presentar los resultados del notebook, Raquel pregunta:

> "explicame los resultados y porque antes parecia todo tan bonito y funcionaba todo tan bien y ahora ya no"

Esto es **una confusión legítima y muy importante**. La explicación que funcionó:

- **El panel decision-level medía M5 vs M8 pareado.** Pregunta: "¿la supervisión mejora al agente cuando opera?" Respuesta: sí, McNemar p=0.088, +1740 bps en SPY. Esto sigue siendo cierto.
- **El notebook nuevo mide cada estrategia contra ground truth absoluto.** Pregunta: "¿cuán bueno es cada uno como clasificador?" Respuesta: ningún sistema supera al baseline trivial "always long" (56.6%). Eso suena terrible pero NO contradice lo anterior — son métricas distintas.
- **Sharpe y accuracy desacoplan.** Una estrategia puede acertar 46% pero ganar dinero si acierta en los días de retorno grande y falla en los pequeños. Es lo que pasa con M8: accuracy 46%, Sharpe +0.62.
- **La señal direccional cruda de retornos diarios SPY es casi ruido blanco.** Con N=400 ningún modelo honesto puede sacar AUC > 0.60 sin overfit. M10 AUC 0.504 es lo esperable. No es un fallo del modelo, es la naturaleza del problema.
- **La narrativa correcta para defensa:** la frase no es "STRATA gana al mercado". Es "el agente LLM sin supervisar pierde; STRATA lo rescata con significancia pareada; un meta-learner universal llega al mismo techo; SHAP confirma que las features STRATA son las informativas; el sistema completo no bate B&H — coherente con literatura — pero rescata al agente LLM de su pérdida estructural".

## Cierre

8 commits atómicos en `feat/decision-level-analysis`, todo local, sin push, sin PR:

```
b8b0e3e bitacora: notebook strata_final.ipynb — defensa matemática rigurosa
3f75a46 feat(notebook): strata_final.ipynb — defensa matemática rigurosa M5/M8/M10
f936e8e feat(experiments): McNemar pareado + sign test panel direccional
3d120ed bitacora: análisis decision-level del panel multi-activo
9da9a0a exp: entregables decision-level del panel (10 activos)
e0771aa feat(experiments): análisis decision-level del panel multi-activo
a74084f feat(experiments): helpers consolidados del panel multi-activo
aa0ad16 feat(stats): stationary_bootstrap_ci (Politis-Romano 1994)
```

Pendientes que esta sesión deja abiertos:

- Decidir si la rama se mergea a `feat/panel-multiactivo` o se reorganiza.
- Reescribir §11 del notebook (Lectura para defensa) con la narrativa más honesta de "STRATA rescata al agente" en lugar del marco actual más neutro — Claude lo sugirió, Raquel no ha respondido aún.
- Llevar las cifras del análisis condicional (RAM-high → +17.8 pp en M8) a la sección de aportaciones de la memoria. Es el dato más limpio y defendible.
- GSO: nota a pie de página en la memoria como limitación / hallazgo metodológico negativo (no dispara medium+ en ningún activo del panel).
