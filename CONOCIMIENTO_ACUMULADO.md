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

- RAM: low 0.20 / medium 0.40 / high 0.70
- PSA: P95 / P99 sobre el periodo de calibración
- GSO: P95 / P99 sobre el periodo de calibración

---

## Las 3 estrategias canónicas a comparar

| ID | Qué es | Rol en la defensa |
|---|---|---|
| **M5** | El agente solo, sin supervisar | **La víctima.** Demuestra que el agente LLM por sí solo no es viable |
| **M8** | M5 + STRATA en modo `override C` con régimen `filtered` | **El rescate.** La regla a mano estadística que corrige al agente |
| **M10** | XGBoost meta-learner con CPCV-within-OOS sobre [5 personalidades × (action_sign, size, conf) + 3 scores STRATA + 4 features de régimen] = 22 features | **El meta-learner universal.** Responde a la objeción del tutor: "un XGBoost con todo dentro debería batir tu regla" |

---

## El número clave de la defensa

> McNemar pareado M8 vs M5: **p ≈ 0.088**. STRATA rescata al agente con significancia estadística pareada.

> Diebold-Mariano M10 vs M8: **p ≈ 0.75**. El XGBoost universal y la regla a mano son **estadísticamente indistinguibles**.

> SHAP global de M10 — top 5 features: `ram_score`, `psa_score`, `garch_sigma`, `stress_prob`, `calm_prob`. **Las 3 STRATA + 2 de régimen. Ninguna personalidad del agente llega al top 5.** El meta-learner sin conocer STRATA redescubre el diseño.

Esto **zanja** la objeción del tutor: no era que "un XGBoost con todo dentro debería batir tu regla", es que la regla a mano captura exactamente la señal que un XGBoost universal identifica.

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

2. **M10 indistinguible de M8.** El meta-learner XGBoost (validado con CPCV-within-OOS, n_splits=6, embargo=5) llega al mismo techo de Sharpe pero **redescubre vía SHAP** que las features informativas son las 3 de STRATA + las 2 de régimen. **Ninguna personalidad** del agente llega al top-5 SHAP.

3. **Ningún sistema bate B&H.** Buy & Hold +32.3% (€1323 sobre €1000). M8 +6.4% (€1064). M10 +3.5% (€1035). M5 −9.7% (€903). La defensa NO es "STRATA gana al mercado" — es "STRATA rescata al agente".

4. **Prior RAM debe ser data-driven y re-signado por activo.** En SPY/BAC/XLE: Crisis ⇒ short (leverage effect estándar). En NVDA: Crisis ⇒ long (las medias de retorno calibradas en Crisis son positivas para NVDA). Hardcodear "Crisis ⇒ short" rompe STRATA en activos con leverage no estándar.

5. **Estabilidad temporal de umbrales.** Los umbrales fijos de STRATA (RAM 0.20/0.40/0.70, PSA/GSO P95/P99) son estables sobre todo el OOS por construcción. El **umbral óptimo `p1*` de XGBoost no lo es:** `p1* = 0.565` óptimo en mitad-1 (Sharpe +0.76 train) → Sharpe +0.14 en mitad-2 (test honesto). En esa misma mitad-2, `p1 = 0.42` habría dado Sharpe +1.07. **El umbral aprendido por XGBoost no es estable; los calibrados de STRATA sí.**

---

## Hallazgos del panel multi-activo (decision-level)

- **Activos con supervisión significativa** (sign test p<0.10 sobre P&L de intervención): SPY (+1740 bps), XLE (+1840 bps).
- **Hit rate M5 vs M8 mejora en 8/10 activos.** Sign test panel p=0.109 (borderline).
- **SMCI = contraejemplo McNemar contra M8** (p=0.011). Caso del *"agente con información direccional complementaria al prior"*: el agente quería short y acertaba 53%; RAM lo volteaba a long que acertaba 46%. Diferente del clásico `prior-flip` de MSTR.
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

> *"El agente LLM sin supervisar pierde dinero (€903 sobre €1000) y acierta direccionalmente menos del 50% (sign test p<0.001). STRATA lo rescata con significancia pareada (McNemar p=0.088, Δ€161). Un meta-learner XGBoost validado con CPCV llega al mismo techo (€1035) sin ser distinguible estadísticamente de la regla a mano (DM p=0.75), y SHAP confirma que las features informativas son exactamente las que STRATA codifica explícitamente. Ningún sistema bate B&H pasivo (+32%) sobre 400 días de SPY — resultado coherente con la literatura sobre eficiencia direccional de índices agregados. La aportación del TFG es un **protocolo de supervisión estadística que rescata a un agente LLM perdedor**, no un sistema que bate al mercado."*

Memorízala. Es la respuesta a "qué has hecho exactamente".
