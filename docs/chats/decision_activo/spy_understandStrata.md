# SPY como activo central, y qué configuración de M10 funciona mejor — conversación y decisiones

*Periodo: 2026-06-14 → 2026-06-17. Resume la investigación sobre si SPY debe ser el caso central, qué
configuración de M10 rinde mejor, y por qué el momentum no entra al modelo. Todas las cifras provienen de
scripts reales en `experiments/` y sus JSON en `outputs/experiments/`. Documento defendible ante el tribunal.*

---

## 0. El replanteamiento (de "no bato a todo" a "supervisión medida")

El punto de partida fue una crisis de enfoque: *"el proyecto se cae porque no tengo una estrategia que bata
a todo"*. Es un **marco equivocado** y peligroso para un TFG de matemáticas:

- STRATA **no predice retornos**: es una función `f:(decisión_agente, estado_mercado) → posición w_t`. No
  es una estrategia que deba batir al mercado.
- **M8 vs M10 no es una carrera.** Es la pregunta de **universalidad** (CLAUDE.md §2, nivel 3): el resultado
  *esperado y pre-registrado* es que un XGBoost (M10) **NO** bata significativamente a la regla determinista
  (M8) — `DM p>0.10` — y que SHAP señale las features de STRATA. Empate = tesis, no derrota.
- El valor de STRATA se mide con **tests pareados**: corrige al agente cuando pierde (McNemar M8/M10 vs M5),
  no con el P&L absoluto. "Batir a B&H en el activo X" es *backtest overfitting* (Bailey & López de Prado 2014).

**Frase de defensa:** *"No afirmo que STRATA gane dinero. Afirmo, con un test pareado, que corrige al agente
cuando se equivoca; y que un optimizador no mejora esa corrección de forma significativa."*

---

## 1. Qué pidió el tutor (reunión 2026-06-16)

Transcripción: `docs/tutor_transcripts/Reunion_Dani_2026-06-16.md` (audio → faster-whisper).

- **NO pidió "un activo donde M10 gane".** No aparece en ninguna fuente.
- **Pidió clases balanceadas (~50/50)** para que el baseline trivial "clase mayoritaria / siempre largo" no
  gane gratis: *"[21:51] a ti te interesa encontrar un activo que esté 50-50"*. En SPY el OOS es alcista
  (~57% días al alza), así que "siempre largo" acierta ~0.57 y los modelos (M8 0.44, M10 0.53) quedan
  por debajo en *accuracy* — flanco ante el tribunal.
- **Pidió una gráfica de sensibilidad del umbral PSA/GSO** con curva de validación y de test, eligiendo el
  umbral SOLO en validación (último año de calibración).
- La objeción histórica del XGBoost (reunión anterior) era *teórica* ("la regla a mano no debería batir a un
  XGBoost con las mismas señales"), ya **falsada**: DM p=0.753, M10 ≈ M8.

---

## 2. Balanceo de clases en el OOS (`class_balance_diagnostic.py`)

Fracción de días al alza en OOS (clase = signo de `r_{t+1}`, causal):

| activo | % sube | baseline mayoritario | desbalanceo |
|---|---|---|---|
| TSLA | 50.1% | 50.1% | 0.001 |
| UNG | 49.9% | 50.1% | 0.001 |
| SMCI | 48.4% | 51.6% | — |
| **SPY** | **56.9%** | **56.9%** | 0.069 |

TSLA y UNG son 50/50 casi perfecto; SPY está desbalanceado (alcista). **Pero** TSLA/UNG son activos sin
*leverage effect* claro → donde el mecanismo RAM pierde base direccional (ver §3).

---

## 3. Dominio de validez: signo data-driven y prior-flip (lo que decide SPY)

`m8_datadriven_sign.json` y `regime_direction_table.py`. El HMM da regímenes de **volatilidad**; la dirección
solo aparece vía *leverage effect* (Black 1976; Christie 1982), que es **contemporáneo**, y solo en índices.

Retorno medio por régimen filtrado, calibración → OOS:

| | SPY (calib → oos, día siguiente) | SMCI (calib → oos, día siguiente) |
|---|---|---|
| Calma | +0.00032 → +0.00012 | −0.00006 → **+0.02546** |
| Estrés | +0.00033 → +0.00082 | +0.00117 → **−0.00384** |
| Crisis | +0.00015 → +0.00329 | +0.00248 → **−0.00003** |

- **SPY: el signo transfiere** (prior-flip = False). El leverage effect se sostiene → RAM tiene base.
- **SMCI / TSLA / UNG: el signo FLIPEA** (prior-flip = True). El régimen no predice dirección OOS.

**Conclusión:** el **dominio de validez de STRATA son los índices con leverage effect (SPY)**. En activos
individuales balanceados el método no transfiere, y la regla prior-flip (pre-registrada) lo detecta sola.
Esto convierte la objeción del tutor en un resultado: *"hice el caso balanceado; ahí el método no transfiere
y mi regla de falsación lo detecta; mi caso de estudio es SPY"*.

---

## 4. ¿Hay un activo donde M10 gane? Barrido Holm-30 (`m10_pivot_scan.py`)

Walk-forward desplegable, 3 configs fijas (base / ens / aug-con-momentum), todo el OOS = test, Holm-30 sobre
M10-vs-B&H, cohorte ex-ante B&H-débil.

**Caso fuerte = NINGUNO.** Bajo Holm-30, M10 no bate a B&H significativamente en ningún activo.

| activo | mejor M10 | acc | vs B&H (McNemar) | sign vs azar | Sharpe |
|---|---|---|---|---|---|
| SPY/aug | con momentum | 0.590 | 0.65 | **0.005** | 2.56 |
| UNG/aug | con momentum | 0.551 | 0.149 | 0.127 | 1.39 |
| SMCI/ens | sin momentum | 0.524 | 0.39 | 0.49 | 1.23 |
| MSTR/ens | sin momentum | 0.530 | 0.093 | 0.38 | −0.03 |

SMCI-ensemble (el que se barajaba como pivote) es **solo nominal**, sin significancia. El único significativo
vs azar es SPY/aug, pero **por el momentum** y **sin batir a B&H** (ver §5–§7).

---

## 5. SPY con momentum: el espejismo del Sharpe 2.56 (`spy_momentum_ablation.py`)

¿De dónde sale el Sharpe 2.56 de SPY/aug? **Del momentum, no de STRATA.** Ablación (ensemble 10 semillas):

| features | acc | Sharpe | sign vs azar |
|---|---|---|---|
| momentum_solo (5) | 0.522 | 2.14 | 0.53 (no sig) |
| strata_regime7 (sin agente, sin mom) | 0.538 | 1.74 | 0.26 |
| all22 (STRATA completo, sin mom) | 0.526 | 0.47 | 0.45 |
| **strata7+mom** (sin agente) | 0.586 | **3.09** | **0.008** |
| all22+mom (= SPY/aug) | 0.590 | 2.56 | 0.005 |

- El **momentum solo** da Sharpe alto pero **accuracy de moneda** (0.522, no sig): Sharpe frágil.
- **STRATA+régimen añade los puntos significativos**: strata7+mom 0.586 (sign p=0.008). STRATA aporta
  **+6.8 puntos sobre el momentum puro** (McNemar p=0.072).
- Es causal (rolling sin shift negativo, `signal_lag=1`): no es look-ahead. Pero el motor es el momentum.

### Robustez (5 bloques de semillas disjuntos, `spy_ablation_robustness.py`)

- **C1 (STRATA aporta sobre momentum): robusto en efecto** (Δacc positivo 5/5, media +0.061), **marginal en
  significancia** (McNemar < 0.10 en 3/5; los otros 0.17). Limitado por n=251.
- **C2 ("quitar el agente mejora"): NO robusto.** Solo se cumplía en 1 de 5 bloques → era artefacto de una
  semilla. **Retirado.**

---

## 6. ¿Se puede decidir el momentum a priori (sin look-ahead)? — NO, demostrado

Varias vías, todas honestas, **todas fallan**:

1. **Diagnóstico ex-ante por activo** (`momentum_exante_rule.py`, `_battery.py`): cruzar trendiness en
   calibración vs Δacc OOS, n=10 activos. Ninguna técnica es candidata; la mejor (momentum reciente, 1 año)
   da ρ=+0.45 pero **p=0.19**, y con 9 técnicas hay 61% de falso positivo. El promedio de 24 años incluso
   apunta al revés (ρ=−0.55).
2. **Regla condicional descubierta en calibración** (`momentum_conditional_calib_oos.py`): en calibración la
   accuracy del momentum crece con la tendencia (0.502 → 0.519, monótona); pero en OOS el edge es **+1 punto
   (0.510 vs 0.500) y no significativo** (p=0.52). EXITO = False.
3. **TS-momentum mensual à la Moskowitz** (`momentum_tsmom_monthly.py`): lookback 12 meses, rebalanceo
   mensual no solapado. **Real en calibración** (acc 0.551) pero **no transfiere al OOS** (0.538, p=0.30) y
   B&H gana (0.570). El OOS es corto y alcista.
4. **Regla de decisión "mete momentum si funcionó el último año"** (`momentum_decision_rule.py`): acierta
   **7/10** y clasifica bien SPY (meter) y SMCI (no). **PERO no es robusta** (`momentum_rule_robustness.py`):
   - El supuesto (persistencia del rendimiento del momentum) **es falso**: Spearman señal↔resultado = **0.033**
     sobre **708 puntos** de 24 años. No hay persistencia, ni siquiera en SPY (ρ=−0.08).
   - El 7/10 es un filo de cuchillo: varía **2–8 / 10** según parámetros (media 5.6 ≈ azar).

**Conclusión:** no existe regla ex-ante robusta para incluir momentum, **porque su beneficio no persiste**.
El subidón de SPY/aug fue suerte del OOS. **El momentum NO entra al modelo desplegable.**

---

## 7. ¿Puede el HMM predecir dirección en vez de volatilidad?

Sí técnicamente (features direccionales / HMM multivariante / medias por régimen tipo Hamilton 1989 — que es
lo que ya hace el signo data-driven). **Pero choca con el mismo muro:** la volatilidad es predecible
(clustering → GARCH), la dirección a un día es casi una martingala (eficiencia). Un HMM direccional
sobreajusta el pasado y no transfiere (mismo fenómeno que el prior-flip y que el momentum). Hacerlo
**perdería el relato del leverage effect** (la aportación teórica) sin ganar señal fiable. El diseño actual
—predecir lo predecible (vol) y cosechar dirección donde la economía la regala (leverage)— es el inteligente.
Además, solo el régimen **filtrado** (causal) es legal; el **suavizado** (Viterbi sobre toda la serie) es
look-ahead. Documentado en `logic_esential` §14.

---

## 8. M10 canónico sobre SPY: el cuadro completo (`spy_m10_full_report.py`, embargo=1)

ALL22 (sin momentum), ensemble 10 semillas, OOS 2025-05→2026-05, n=251:

| modelo | accuracy | Sharpe | equity | maxDD | AUC | log-loss | Brier |
|---|---|---|---|---|---|---|---|
| M5 (agente) | 0.367 | −2.73 | 0.932 | −0.069 | — | — | — |
| M8 (STRATA) | 0.442 | **+1.60** | **1.097** | −0.060 | — | — | — |
| M10 (ALL22) | 0.494 | −0.60 | 0.920 | −0.161 | 0.531 | 0.856 | 0.308 |
| B&H | 0.566 | +2.20 | 1.302 | −0.098 | — | — | — |

Tests M10: vs M5 **McNemar p=0.007** · vs M8 p=0.29 · vs B&H p=0.13 · **sign vs 0.5 p=0.90** · IC95 exceso
accuracy [−0.058, +0.042].

- El **M10 desplegable (sin momentum) es una moneda que pierde dinero** (acc 0.494, AUC 0.53, log-loss
  > 0.693, equity < 1). Direccionalmente indistinguible del azar.
- Su único valor robusto —igual que el de M8— es **corregir al agente** (p=0.007).
- **M8 (regla determinista) gana al ML** económicamente (Sharpe +1.60 vs −0.60, equity 1.097 vs 0.920),
  con accuracy menor: acierta el signo y el tamaño en los días grandes. Refuerza la **universalidad**.
- Nadie bate a B&H (OOS alcista) → coherente con eficiencia de mercado.

---

## 9. El embargo: por qué 1, no 5 (`embargo_sweep.py`)

El embargo es **control de fuga** (López de Prado 2018 §7.4), **no un hiperparámetro de rendimiento**.
Con etiqueta de horizonte 1 (`y_t = 1[r_{t+1}>0]`) y walk-forward de origen móvil (test siempre futuro),
**embargo = 1** elimina el único solape. Elegirlo por accuracy OOS sería look-ahead.

El barrido (1,2,3,5,10,21) lo confirma como **ruido**:
- Rango de accuracy entre embargos: 0.032 (SPY), 0.040 (SMCI) ≈ **1 desviación binomial** (±0.0316, n≈250).
- **Direcciones opuestas**: el "mejor" es embargo 5 en SPY, embargo 1 en SMCI. Se contradicen.
- Quitar 4 días de entrenamiento desplaza `p1` ~0.06 y **voltea el 10% de las posiciones**. Que un cambio tan
  pequeño mueva tanto es **prueba de que M10 no tiene señal direccional** (un modelo con señal sería estable).
- Cambiar la semilla mueve la accuracy lo mismo que cambiar el embargo.

**Regla:** embargo = 1 por principio (horizonte de etiqueta), justificado a priori, no por p-valor.
Documentado en `logic_esential` §14b.

---

## 10. Decisión final y configuración

- **Activo central: SPY.** Es donde el mecanismo (RAM/leverage effect) está justificado (prior-flip = False)
  y donde STRATA corrige al agente de forma significativa. La salida que el propio tutor ofreció:
  *"es un caso de estudio, el SP500, y así gano"*.
- **Activos balanceados (SMCI / TSLA / UNG): contraste honesto.** Responden a la petición del tutor (clases
  50/50) y enseñan el **dominio de validez**: ahí el método solo gana nominalmente y el prior-flip se dispara.
- **Configuración de M10: ALL22 canónico, ensemble 10 semillas, embargo = 1. SIN momentum.** El momentum no
  es desplegable (no se decide a priori, no persiste). El M10 desplegable no tiene alfa direccional; su valor
  es la corrección del agente.
- **El protagonista económico es M8** (regla interpretable), que no es batido por el ML (universalidad).

---

## 11. Frases listas para la defensa

1. *"STRATA no busca alfa: corrige al agente. Lo pruebo con McNemar pareado (M10/M8 vs M5, p=0.007), no con el P&L."*
2. *"Un XGBoost con las mismas señales no bate a mi regla determinista (DM p=0.75): universalidad. De hecho la regla M8 le gana económicamente."*
3. *"El dominio de validez es los índices con leverage effect: en SPY el signo por régimen transfiere (prior-flip False); en activos balanceados sin leverage no transfiere, y mi regla de falsación lo detecta sola."*
4. *"Probé incluir momentum por cinco vías y demuestro, sin look-ahead y con 708 puntos, que su beneficio no persiste (ρ=0.03). Por eso no entra al modelo: lo reporto como observación ex-post, no como componente."*
5. *"El embargo lo fijo a priori por el horizonte de la etiqueta (=1); el barrido muestra que su efecto es ruido (1 SD, direcciones opuestas entre activos). Elegirlo por rendimiento sería look-ahead."*

---

## Apéndice — scripts y outputs

| tema | script | output |
|---|---|---|
| balanceo de clases | `experiments/class_balance_diagnostic.py` | `class_balance_diagnostic.json` |
| signo data-driven / prior-flip | `experiments/m8_datadriven_sign.py` | `m8_datadriven_sign.json` |
| régimen → dirección (calib vs oos) | `experiments/regime_direction_table.py` | `regime_direction_table.json` |
| barrido de activos (Holm-30) | `experiments/m10_pivot_scan.py` | `m10_pivot_scan.json` |
| ablación momentum vs STRATA (SPY) | `experiments/spy_momentum_ablation.py` | `spy_momentum_ablation.json` |
| robustez de la ablación (5 semillas) | `experiments/spy_ablation_robustness.py` | `spy_ablation_robustness.json` |
| diagnóstico ex-ante (n=10) | `experiments/momentum_exante_rule.py`, `_battery.py` | `momentum_exante_rule.json`, `_battery.json` |
| regla condicional calib→oos | `experiments/momentum_conditional_calib_oos.py` | `momentum_conditional_calib_oos.json` |
| TS-momentum mensual (Moskowitz) | `experiments/momentum_tsmom_monthly.py` | `momentum_tsmom_monthly.json` |
| regla de decisión + robustez | `experiments/momentum_decision_rule.py`, `momentum_rule_robustness.py` | idem `.json` |
| informe completo M10 SPY | `experiments/spy_m10_full_report.py` | `spy_m10_full_report.json` |
| barrido de embargo | `experiments/embargo_sweep.py` | `embargo_sweep.json` |

Notebook didáctico: `notebooks/logic_esential.ipynb` §14 (HMM=vol), §14b (embargo), §14c (momentum/M10 SPY).
