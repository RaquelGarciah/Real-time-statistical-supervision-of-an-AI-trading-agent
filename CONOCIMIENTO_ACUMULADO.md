# Conocimiento acumulado — síntesis ejecutiva (10 min)

Lectura obligatoria antes de cualquier experimento. Para profundidad, consulta `_archivo_proyecto_anterior/`.

---

## El problema en una frase

Un **agente LLM** (AI Hedge Fund con 5 personalidades: Buffett, Wood, Druckenmiller, Burry, Ackman) decide cada día sobre SPY. Sin supervisar pierde dinero y acierta direccionalmente **menos del 50%**. **STRATA** es una capa de supervisión estadística que filtra/atenúa esas decisiones con tres detectores clásicos. La pregunta del TFG: *¿esta supervisión rescata al agente con significancia estadística?*

---

## Los 3 detectores STRATA

| Detector | Eje | Pregunta | Modelo subyacente |
|---|---|---|---|
| **RAM** | Régimen de mercado (discreto) | ¿Es la acción del agente coherente con el régimen? | HMM gaussiano de 3 estados (Calma/Estrés/Crisis) sobre `[log_return, realized_vol_21d]` |
| **PSA** | Coherencia temporal del agente | ¿Está cambiando de opinión estructuralmente? | BOCPD (Adams & MacKay 2007) sobre el sizing del agente |
| **GSO** | Volatilidad continua del mercado | ¿Es el tamaño compatible con la volatilidad? | GARCH(1,1) Student-t sobre retornos del activo |

**Política RAM simétrica con leverage effect** (SPY): Calma ⇒ permitido long; Crisis ⇒ permitido short; Estrés ⇒ ambos. El `RAM_score` = masa de probabilidad sobre regímenes donde la acción es inconsistente.

**Umbrales STRATA fijos calibrados ex-ante** (de `cache/models/strata_thresholds.json`):

- RAM: low 0.25 / **medium (τ) 0.50** / high 0.70. El gate operativo (el que dispara override en M8 y reduce en M7) es `medium = τ = 0.5`. `low` y `high` solo re-etiquetan severidad sin efecto sobre P&L. Se reporta también con el default conservador 0.40 como blindaje anti-p-hacking.
- PSA: P95 / P99 sobre el periodo de calibración
- GSO: P95 / P99 sobre el periodo de calibración

---

## Las 3 estrategias canónicas a comparar

| ID | Qué es | Rol en la defensa |
|---|---|---|
| **M5** | El agente solo, sin supervisar | **La víctima.** Demuestra que el agente LLM por sí solo no es viable |
| **M8** | M5 + STRATA en modo `override C` con régimen `filtered` | **El rescate.** La regla a mano estadística que corrige al agente |
| **M10** | XGBoost meta-learner con CPCV-within-OOS sobre [5 personalidades × (action_sign, size, conf) + 3 scores STRATA + 4 features de régimen] = 22 features | **Co-protagonista de M8 por accuracy (0.539 > 0.436).** Demuestra que la señal de STRATA es real: sin features de régimen/RAM/PSA/GSO, M10 cae a Sharpe +0.21 (ablación §11). En P&L es equivalente a M8 (DM p=0.67): ninguno bate al otro; ambos consumen la misma señal. Responde también a la objeción del tutor ("un XGBoost debería batir tu regla"): la regla a mano captura exactamente la señal que el XGBoost redescubre. |

---

## El número clave de la defensa

> McNemar pareado M8 vs M5: **p=0.069 (τ=0.5, canónico) / 0.088 (τ=0.40, blindaje anti-p-hacking)**. STRATA mejora la accuracy direccional con significancia pareada. **En el plano Sharpe el rescate es condicional al alza** (§13 walk-forward: ΔSharpe se invierte en el tramo bajista para M8 −3.92 y M10 −1.06; falsificación pre-registrada disparada para ambos). **En el plano accuracy, M10 rescata al agente en AMBOS regímenes** (bajista Holm p_adj=0.075, block-perm p=0.061). El resultado honesto es: "la dirección se recupera cross-régimen para M10; la rentabilidad económica es condicional al alza para ambos".

> **Accuracy escalera (métrica primaria):** M5 0.384 → M8 0.436 → **M10 0.539** → B&H 0.569. M10 es el mejor decodificador; ambos quedan por debajo de B&H (STRATA reduce el daño, no genera alfa).

> Diebold-Mariano M10 vs M8: **p ≈ 0.67**. Equivalentes en P&L; M10 gana en accuracy. Ambos consumen la misma señal de STRATA.

> SHAP global de M10 — top 5 features: `ram_score`, `psa_score`, `garch_sigma`, `stress_prob`, `calm_prob`. **Las 3 STRATA + 2 de régimen. Ninguna personalidad del agente llega al top 5.** El meta-learner sin conocer STRATA redescubre el diseño.

La señal de STRATA es real: ablación M10 sin features de régimen/RAM/PSA/GSO → Sharpe +0.21 (desde +0.64). M8 y M10 son dos consumidores de la misma señal: M8 interpretable (white box), M10 el de mayor accuracy.

---

## Lo que el tutor exige (de la transcripción 2026-06-07)

El tutor pidió explícitamente:

1. **Accuracy contra ground truth binario** (`y_{t+1} = 1{r_log(t+1) > 0}`), no solo Sharpe.
2. **AUC, log-loss, Brier, MCC** como métricas matemáticas de clasificación.
3. **Comparación contra baseline trivial "always long"** (≈0.566 acc en SPY OOS).
4. **Sign test direccional** sobre el agente solo (M5) contra 0.5.
5. **Umbral operativo p1\*** que produce XGBoost, con **análisis de estabilidad temporal** (mitad-1 vs mitad-2 del OOS).
6. **Mecanismo, no caja negra.** Ejemplo de UN día concreto donde STRATA intervino, con tabla paso a paso.
7. **No vender humo.** Cada cifra con su test estadístico y su intervalo de confianza.

Cita textual del tutor (de la transcripción): *"Necesitas rigor matemático. Tener Sharpe positivo y curva equity mejor en M8 que en M5 no demuestra nada sin fundamento."*

Lo que también dejó claro: *"Algo que has impuesto tú a mano (tu capa de supervisión STRATA) nunca va a salir mejor que un XGBoost entrenado con las probabilidades del agente y las de tus detectores todas juntas."* — La respuesta: M10 confirma esto **empíricamente** (empata, no bate) y SHAP confirma **por qué** (las features útiles ya estaban en STRATA).

---

## Los 5 hallazgos no triviales del proyecto anterior

1. **RAM domina la supervisión.** Sobre el panel multi-activo (10 tickers, 1406 intervenciones), el **98% del P&L atribuible** a las intervenciones viene de RAM. PSA = 2%. GSO = 0%. **La tesis se reorganiza alrededor de RAM como contribución principal.**

2. **M10 co-protagonista de M8.** El meta-learner XGBoost (validado con CPCV-within-OOS, n_splits=6, embargo=5) es equivalente a M8 en P&L (DM p=0.67) pero **supera a M8 en accuracy** (0.539 vs 0.436). Ambos consumen la misma señal: la ablación sin features STRATA cae M10 a Sharpe +0.21, y **SHAP** confirma que las features informativas son las 3 de STRATA + las 2 de régimen. **Ninguna personalidad** del agente llega al top-5 SHAP. El Sharpe de M8 es frágil (Deflated Sharpe ≈0.10); la métrica robusta es la accuracy.

3. **Ningún sistema bate B&H.** Buy & Hold +32.3% (€1323 sobre €1000). M8 +6.4% (€1064). M10 +3.5% (€1035). M5 −9.7% (€903). La defensa NO es "STRATA gana al mercado" — es "STRATA rescata al agente".

4. **Prior RAM debe ser data-driven y re-signado por activo.** En SPY/BAC/XLE: Crisis ⇒ short (leverage effect estándar). En NVDA: Crisis ⇒ long (las medias de retorno calibradas en Crisis son positivas para NVDA). Hardcodear "Crisis ⇒ short" rompe STRATA en activos con leverage no estándar.

5. **Estabilidad temporal de umbrales.** Los umbrales fijos de STRATA (RAM 0.25/τ=0.50/0.70, PSA/GSO P95/P99) son estables sobre todo el OOS por construcción. El **umbral óptimo `p1*` de XGBoost no lo es:** `p1* = 0.565` óptimo en mitad-1 (Sharpe +0.76 train) → Sharpe +0.14 en mitad-2 (test honesto). En esa misma mitad-2, `p1 = 0.42` habría dado Sharpe +1.07. **El umbral aprendido por XGBoost no es estable; los calibrados de STRATA sí.**

---

## Hallazgos del panel multi-activo (decision-level)

- **Activos con supervisión significativa** (sign test p<0.10 sobre P&L de intervención): SPY (+1740 bps), XLE (+1840 bps).
- **Hit rate M5 vs M8 mejora en 8/10 activos.** Sign test panel p=0.109 (borderline).
- **SMCI = contraejemplo McNemar contra M8** (p=0.011). Caso del *"agente con información direccional complementaria al prior"*: el agente quería short y acertaba 53%; RAM lo volteaba a long que acertaba 46%. Diferente del clásico `prior-flip` de MSTR.
  > **[Actualizado 2026-06-17]** Esta lectura es del proyecto anterior (signo de RAM hardcodeado, invertido en SMCI — corregido). En el proyecto actual **SMCI es el ACTIVO DEL CASO DE ESTUDIO del tutor** (B&H≈0.48, benchmark justo): el M10 desplegable (WF ensemble, embargo=1) bate a M5/M8/B&H **nominalmente** (acc 0.552), pero **no significativamente** porque el agente está 95% corto → M5/M8/M10 son la misma apuesta corta y STRATA no tiene margen de rescate. STRATA rescata donde el agente discrepa de un régimen que acierta = **SPY** (M10 vs M5 p=0.0041). Recorrido completo: `docs/chats/decision_activo/smci.md`; decisiones #13–#16 en DECISIONES_ESENCIALES.md.
- **GSO no dispara medium+ en NINGÚN activo del panel.** Limitación metodológica: la banda `target_vol/σ` rara vez se viola por el sizing del agente. GSO está calibrado demasiado laxo o el sizing del agente es naturalmente conservador. **Reportable como hallazgo metodológico negativo.**

---

## Cómo se decide la posición de cada día (mecánica)

**El backtest es una simulación contable, no una compra real.** `pnl_de_hoy = posicion_de_hoy × retorno_real_mañana`. `posicion` es un número en `[−1, +1]`. **Lo único que cambia entre M1...M10 es cómo se calcula la posición.** El motor de backtest es idéntico.

Ejemplo del 12-mar-2025 (M10):

| Paso | Cálculo | Resultado |
|---|---|---|
| 1. Agente decide | `short, size=−0.197, conf=0.85` | tupla del agente |
| 2. HMM + GARCH | `P(Estrés)=99.9%, σ=23.3%` | estado de mercado |
| 3. XGBoost predice | `p1 = 0.8073` | prob. retorno > 0 |
| 4. Dirección continua | `2·0.8073 − 1` | `+0.615` |
| 5. Magnitud por paridad de riesgo | `0.10 / 0.233` | `0.43` |
| 6. Factor de régimen | `×0.5` (Estrés) | `0.215` |
| 7. **Posición final** | `0.615 × 0.43 × 0.5` | **`+0.132`** |
| 8. SPY ese día | `−1.34%` | retorno real |
| 9. P&L | `0.132 × (−0.0134)` | **`−0.18%`** |

Tres lecciones de ese día:

- Posición pequeña (13%) porque tres mecanismos de control de riesgo la encogen.
- Ese día el agente acertó (short) y XGBoost falló (long). **Por eso se mide sobre 400 días, no sobre uno.**
- "Invertir +0.132" = el 13.2% del € ficticio se multiplica por el retorno SPY de ese día. No hay broker.

---

## Punteros al archivo (para profundidad)

Cuando necesites más detalle, ve al archivo del proyecto anterior:

| Tema | Dónde mirar |
|---|---|
| Pivot de N configuraciones → 3 canónicas (M5/M8/M10) | `_archivo_proyecto_anterior/BITACORA.md` (entradas 2026-05-19 → 2026-06-07) |
| Diseño de M10 y respuesta a objeción del tutor | `_archivo_proyecto_anterior/docs/chats/need_mathematic_rigor.md` |
| Decision-level del panel multi-activo | `_archivo_proyecto_anterior/docs/chats/expand_STRATA_strategy.md` + `outputs_canonicos/decision_level/` |
| Qué exige el tutor textualmente | `_archivo_proyecto_anterior/docs/tutor_transcripts/` |
| Métodos: leverage effect, HMM, BOCPD, validación honesta | `_archivo_proyecto_anterior/docs/marco_teorico.md` |
| Hallazgos numéricos por activo, prior-flip, NVDA per-asset HMM | `_archivo_proyecto_anterior/docs/hallazgos_strata.md` |
| Cifras canónicas brutas de M5/M8/M10 | `_archivo_proyecto_anterior/outputs_canonicos/m{5,8,10}*.json` |
| Tabla maestra ya consolidada | `RESULTADOS_OBJETIVO.md` |

---

## La narrativa de defensa (frase a memorizar)

> *"Un agente LLM perdedor direccional (38.4%, < azar, sign test p<0.001) es rescatado por supervisión estadística clásica: la accuracy sube 0.384 → 0.436 (regla M8) → 0.539 (XGBoost M10 sobre features STRATA), y regla a mano y caja negra son equivalentes en P&L (DM p=0.67). La señal informativa es la de STRATA: sin features de régimen/RAM/PSA/GSO, M10 cae a Sharpe +0.21 (ablación). Ningún sistema bate B&H pasivo (0.569 accuracy, +32% equity) — STRATA reduce el daño, no genera alfa. En el plano accuracy M10 rescata al agente en ambos regímenes (bajista Holm p_adj=0.075, block-perm p=0.061); en el plano Sharpe el rescate es condicional al alza para ambos modelos (walk-forward §13: ΔSharpe se invierte en bajista, M8=−3.92 / M10=−1.06; falsificación pre-registrada disparada). El componente de modelo K=3 sí generaliza inter-época (15/16 orígenes). La aportación es un **protocolo de supervisión estadística interpretable** que recupera accuracy direccional de un agente perdedor —robusto en ambos regímenes para M10— y delimita honestamente dónde funciona y dónde no."*

Memorízala. Es la respuesta a "qué has hecho exactamente".
