# AutoML para M10 — registro de la investigación

> **Propósito.** Dejar claro y ordenado todo lo hecho con AutoML (H2O) como alternativa/complemento a M10,
> porque la sesión se enredó con configuraciones, reproducibilidad y memoria. Resumen arriba, decisiones en
> orden, estado actual de procesos/ficheros, resultados y pendientes.

---

## RESUMEN EN UNA FRASE

Dejar que **H2O AutoML** busque, por activo, el mejor modelo (6 familias) en el **mismo pipeline causal que M10**
(ALL22 features, target signo r_{t+1}, walk-forward con embargo, Purged K-Fold). Resultado: **AutoML no bate a
ZeroR de forma significativa en ningún activo**; aporta donde el leverage-effect (SPY, QQQ): rescata al agente y
bate al azar. Confirma la **universalidad** (§2 nivel 3): la búsqueda automática redescubre, no bate.

---

## DECISIONES EN ORDEN (lo que hay que recordar)

1. **AutoML = mismo pipeline que M10.** Mismas ALL22, target signo(r_{t+1}), walk-forward expandible, Purged
   K-Fold interno. Solo cambia el estimador (XGBoost → búsqueda H2O). Script: `experiments/automl_m10.py`,
   wrapper `core/h2o_automl.py`.
2. **HMM y GARCH calibrados POR ACTIVO** (`build_states_onthefly`): cada ticker su propio HMM(K=3) + GARCH(1,1)-t
   sobre su histórico. Verificado: σ GARCH OOS SPY=0.154, NVDA=0.455, BAC=0.257 (distintas → específicas).
3. **`max_models`, NO `max_runtime_secs`.** Con presupuesto de tiempo AutoML NO es reproducible (NVDA dio 3
   resultados distintos con la misma semilla 42, incluso entre chats). `max_models` fijo + excluir DeepLearning
   = determinista. Ver memoria `[[automl-h2o-reproducibilidad]]`.
4. **Semilla = 42 (config.SEED) SIEMPRE.** Nunca se eligió semilla por resultado (sería p-hacking). Los números
   distintos de SPY vinieron de la config (max_runtime / embargo / multithread), no de la semilla.
5. **embargo = 1** (no 5): es el walk-forward DESPLEGABLE canónico del TFG (horizonte=1, decisión
   `[[embargo-1-walkforward]]`). Aplica a M10 y AutoML por igual porque es el protocolo, no el modelo.
   Que embargo=5 diera 0.574 vs 0.562 de embargo=1 es **ruido** (~3 días/251), no preferir uno.
6. **M10 canónico = ensemble de 10 XGBoost** (semillas), no XGBoost único. El ensemble es SOLO de M10; AutoML
   NO lleva ensemble de semillas (de momento es búsqueda H2O de 1 semilla).
7. **Ventana: ~250 (desplegable) para meta-learners; OOS-completo solo para no-learners.** M10/AutoML necesitan
   burn-in (entrenan sobre features del agente, que solo existen en el OOS) → su única evaluación honesta es
   N0=150 → ~250 días. M5/M8/ZeroR/B&H/Régimen no necesitan burn-in → pueden ir en OOS completo (~400, más
   potencia). Forzar AutoML al OOS-completo bajando N0 a 40 lo hunde por artefacto de burn-in corto (verificado:
   tramo early mal entrenado acc 0.45). NO mezclar ventanas entre meta-learners y triviales como comparación directa.
8. **Reproducibilidad bit-exacta (nthreads=1) ≠ viable en paralelo aquí.** 16GB de RAM no aguantan varios clústers
   H2O a la vez (OOM repetido). Decisión: **serial + multi-thread (1 clúster, 4G, 8 núcleos)** = rápido y fiable;
   el residuo de ±1 día por multithreading se documenta. nthreads=1 solo para fijar UN activo headline si hace falta.
9. **Un fichero por configuración** (no pisar `automl_panel.json`). Convención: `outputs/experiments/automl_runs/
   panel_<embargo>_<maxmodels>_<thread>.json`. Cada corrida su nombre → comparables, sin clobbering.

---

## ESTADO ACTUAL (al escribir esto)

**Procesos corriendo:**
- Panel **mm25** (15 activos, embargo=1, multithread) → `outputs/experiments/automl_runs/panel_emb1_mm25_multithread.json`.
  Va por SPY-cerrado (1/15). ETA ~60-90 min.
- **SPY mm10** (1 activo, comparar con mm25) → `outputs/experiments/automl_runs/spy_emb1_mm10.json`. En curso.

**Convención de ficheros:** `outputs/experiments/automl_runs/` con la config en el nombre. `automl_panel.json`
queda para copiar SOLO la corrida elegida como definitiva.

---

## RESULTADOS HASTA AHORA

### SPY (la config va en el título — NO comparar entre configs distintas a la ligera)

| corrida | embargo | thread | AutoML acc | AutoML Sharpe | nota |
|---|---|---|---:|---:|---|
| inicial | 5 | multi (max_runtime) | 0.546 | 1.27 | NO reproducible |
| panel viejo | 5 | multi (max_runtime) | 0.51 | −0.02 | NO reproducible |
| test mm25 | 5 | multi | 0.578 | 2.57 | wobble multithread |
| panel mm25 actual | **1** | multi | **0.5618** | **2.16** | maxDD −8.5%, Calmar 3.50, fam GBM |

SPY mm25 actual (embargo=1, n=251): M5 0.367 · M8 0.442 · M10 0.494 · **AutoML 0.562** · ZeroR/B&H 0.566.
McNemar AutoML: vs ZeroR p=1.0 (igual), **vs M5 p=0.0006 (rescata)**, vs M8 p=0.013, vs M10 p=0.115; sign p=0.058.

### Panel 15 (corrida previa multithread embargo=1, M10-ensemble — orientativa, se está rehaciendo limpia)

- **Nadie bate a ZeroR significativamente** en ningún activo (medias: ZeroR acc 0.544 / Sharpe 0.93 / Calmar 1.39
  domina; entre no-triviales AutoML acc 0.505 y M8 0.501 empatan arriba; AutoML > M10-XGB en riesgo).
- **AutoML bate al azar (sign p<0.10): SPY, QQQ, XLK.** **Rescata al agente (vs M5 sig.): SPY, QQQ.**
- **AutoML supera a ZeroR en punto** solo en SPY/UNG (marginal); nunca significativo.
- Nicho de AutoML = índices de leverage fuerte (SPY, QQQ). Stocks que cayeron (SMCI, MARA, DIA): no aporta.

---

## HALLAZGO METODOLÓGICO (vale para el TFG)

H2O AutoML con `max_runtime_secs` **no es reproducible** ni con semilla fija (lo demostramos: 3 NVDA distintos).
**Regla:** cualquier AutoML que entre a la memoria usa `max_models` + excluir DeepLearning. Pista de diagnóstico
de que un Sharpe es ruido: **sube el Sharpe mientras baja la accuracy** (pasó en SPY y NVDA con max_runtime).

---

## CONCLUSIÓN PROVISIONAL (defendible)

> Ni una búsqueda automática sobre 6 familias, calibrada por activo, bate al naïve **ZeroR** en accuracy en ningún
> activo. El techo lo pone el OOS (alcista, sin estructura direccional), no el modelo. AutoML aporta en **riesgo**
> y en **rescate del agente** en los índices de leverage (SPY, QQQ), y **redescubre** lo que M8 (supervisor
> interpretable) ya captura → **confirma la universalidad**, no la refuta.

---

## PENDIENTES

- [ ] Terminar panel **mm25 reproducible** (corriendo) y **SPY mm10** para comparar mm25 vs mm10.
- [ ] Decidir config definitiva (embargo=1 fijo; mm25 vs mm10) y copiar esa corrida a `automl_panel.json`.
- [ ] (Opcional) AutoML = ensemble de N semillas para igualar trato con M10 y matar el wobble sin nthreads=1.
- [ ] Commitear: `experiments/automl_m10.py` + `core/h2o_automl.py` + JSON elegido + `docs/automl_m10_EXPLORATORIO.md`.
- [ ] NO hacer: elegir semilla/ventana/embargo por resultado (p-hacking); mezclar ventanas meta-learner vs trivial.

---

## PANEL FINAL — mm25 + include_algos (config elegida por Raquel)

Config: `max_models=25`, `include_algos=GBM,XGBoost,StackedEnsemble`, `sort_metric=AUC`, embargo=1, N0=150,
STEP=21, Purged K-fold, seed=42, multi-thread. Fichero:
`outputs/experiments/automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json`.

| act | M5 | M8 | M10 | AutoML | ZeroR | Ganadora | sig (ganadora-tuya vs ZeroR) |
|---|--:|--:|--:|--:|--:|:--|:--|
| SPY | 0.367 | 0.442 | 0.494 | 0.574 | 0.566 | **AutoML** | p=0.90 no sig |
| QQQ | 0.418 | 0.486 | 0.522 | 0.534 | 0.590 | ZeroR | — |
| DIA | 0.44 | 0.484 | 0.468 | 0.520 | 0.552 | ZeroR | — |
| IWM | 0.450 | 0.470 | 0.458 | 0.482 | 0.554 | ZeroR | — |
| XLE | 0.448 | 0.528 | 0.508 | 0.532 | 0.565 | ZeroR | — |
| XLF | 0.429 | 0.502 | 0.526 | 0.510 | 0.538 | ZeroR | — |
| XLK | 0.51 | 0.470 | 0.542 | 0.590 | 0.641 | ZeroR | — |
| NVDA | 0.467 | 0.521 | 0.483 | 0.517 | 0.552 | ZeroR | — |
| BAC | 0.451 | 0.516 | 0.419 | 0.488 | 0.561 | ZeroR | — |
| TSLA | 0.474 | 0.478 | 0.522 | 0.454 | 0.458 | **M10** | p=0.18 no sig |
| MSTR | 0.554 | 0.558 | 0.534 | 0.498 | 0.530 | **M8** | p=0.40 no sig |
| SMCI | 0.484 | 0.496 | 0.552 | 0.472 | 0.516 | **M10** | p=0.49 no sig |
| ROKU | 0.444 | 0.528 | 0.508 | 0.544 | 0.548 | ZeroR | — |
| MARA | 0.528 | 0.528 | 0.532 | 0.544 | 0.532 | **AutoML** | p=0.83 no sig |
| UNG | 0.510 | 0.502 | 0.518 | 0.482 | 0.449 | **M10** | p=0.16 no sig |

**Medias acc:** M5 0.465 · M8 0.501 · M10 0.506 · AutoML **0.516** · ZeroR **0.543**.
**Medias Sharpe:** M5 −0.83 · M8 0.24 · M10 −0.10 · AutoML **0.40** · ZeroR 0.93.

- **Ganadora por activo:** ZeroR 9/15 · M10 3 (TSLA, SMCI, UNG) · AutoML 2 (SPY, MARA) · M8 1 (MSTR) · M5 0.
- **Donde gana una estrategia tuya (6 activos), la ventaja vs ZeroR es NO significativa en los 6** (McNemar p 0.16–0.90). 0/6 significativos.
- mm25+include_algos es la **mejor config de AutoML** (mejor media acc y Sharpe entre no-triviales; rescata M5 sig. en SPY/QQQ/DIA/XLK/ROKU) pero **confirma el techo ZeroR**, no lo rompe.

---

## NOTA METODOLÓGICA — ¿data augmentation para ganar potencia/significancia? NO

A n≈250 no hay potencia para separar las estrategias de ZeroR. Pregunta natural: ¿generar datos sintéticos
para tener más muestra? **Respuesta: no es válido para ganar significancia de edge real, y el tribunal lo
tumbaría.**

**Por qué (límite fundamental).** Todo dato sintético se genera de un modelo ajustado a los 250 días reales.
Por el **principio de procesamiento de datos**, una transformación no contiene más información que el original.
Si el generador no tiene estructura direccional (porque la dirección real es ~Bernoulli), el sintético tampoco
→ no batirás a ZeroR. Si el generador SÍ mete estructura, testas tu estrategia **contra tus propias
suposiciones, no contra el mercado** → circular. Es **pseudoreplicación** (mismo error que contar ventanas
solapadas como independientes: N efectivo ≪ N).

**Qué usan los quants de verdad — y para qué (NUNCA para fabricar significancia de edge):**
- **Block bootstrap** → IC del estimador (no potencia).
- **CPCV (López de Prado)** → distribución de backtests + **PBO** (prob. de overfitting); **deflacta**, no infla.
- **Deflated Sharpe Ratio** → descuenta el nº de configs probadas; hace el claim **más conservador**.
- **GANs / TimeGAN / Monte Carlo** → **stress-test** de riesgo y robustez del modelo; circular para edge.
- **Permutación/null** → p-valor bajo H0; da p igual o **más conservador**.

Patrón: todas las herramientas serias van a **deflactar/ser más conservadoras**, no a fabricar significancia.

**Vías legítimas de más potencia (información nueva real):** (1) **pool de activos reales** — ya hecho, ZeroR
sigue imbatido; (2) **más tiempo real** (extender OOS); (3) un OOS histórico con crisis — pero reintroduce el
look-ahead del LLM (el OOS empieza post-cutoff justo para evitarlo).

**Encuadre de defensa:** la no-significancia **es el hallazgo**, no un bug a parchear. Se reporta como límite de
potencia ("nominal, no significativo a n≈250; significancia = trabajo futuro con muestra mayor") — eso blinda;
fabricar significancia con augmentation sería el fraude que el proyecto pre-registra para no cometer.

**Tratamiento quant honesto pendiente (opcional):** aplicar **Deflated Sharpe Ratio + PBO** sobre lo que hay,
descontando todas las configs de AutoML probadas → blinda la conclusión (más conservadora, no batirá ZeroR).

---

## DECISIONES FIJADAS (canónico)

### Panel AutoML CANÓNICO
- **Script:** `experiments/automl_m10.py`
- **JSON:** `outputs/experiments/automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json`
- **Config fija:** `max_models=25`, `include_algos=GBM,XGBoost,StackedEnsemble`, `sort_metric=AUC`,
  `embargo=1`, `N0=150`, `step=21`, Purged K-Fold interno, `seed=42`, multi-thread.
- **Regenerar:** `python experiments/automl_m10.py --max-models 25 --n0 150 --embargo 1 --include-algos "GBM,XGBoost,StackedEnsemble" --sort-metric AUC --out <ese JSON>`
- (mm20 daba algo mejor Sharpe medio, pero el **canónico es mm25** por decisión de Raquel.)

### Entregable de "qué aporta STRATA": notebook `decision_automl`
- **Builder:** `notebooks/_build_decision_automl.py` → **`notebooks/decision_automl.ipynb`** (22 celdas, 0 errores, auto-test OK).
- **Datos que consume:** el panel canónico (arriba) + 3 JSON nuevos:
  - `experiments/decision_automl_prep.py` → `outputs/experiments/decision_automl_prep.json`
    (M10 canónico determinista: accuracy/Sharpe/maxDD/Calmar de M5/M8/M10/ZeroR/B&H, **ablación** agente15/strata7/all22,
    **TreeSHAP** por bloque, **bootstrap pareado** ΔSharpe y ΔmaxDD por activo y **pooled**).
  - `experiments/automl_importance.py` → `outputs/experiments/automl_importance.json`
    (H2O, SPY/MARA/UNG: **SHAP del mejor árbol** del leaderboard + **permutation importance** del ensemble).
  - `experiments/strategy_clustering.py` → `outputs/experiments/strategy_clustering15.json`
    (clustering **multi-algoritmo**: KMeans/Ward/GMM/Spectral).

### Decisiones metodológicas fijadas
1. **Importancia/§2.3:** SHAP **sobre árbol** (XGBoost si existe; en macOS H2O no entrena XGBoost → **GBM**),
   NO sobre el StackedEnsemble (no admite atribución exacta). El **ensemble** se reporta con **permutation
   importance** (model-agnostic). Ambos coinciden: cuota STRATA ~0.52–0.84 → el ML se apoya en STRATA.
2. **Accuracy vs ZeroR/B&H:** ganadores **nominales**, NO significativos (n≈250, ventana corta) → línea futura.
   El "rescate de B&H" es **condicional al activo** (real donde el activo cae; en alcistas B&H gana).
3. **Riesgo (Sharpe/maxDD): SÍ significativo** por bootstrap pareado. Resultado clave **pooled** (15 activos):
   **M8 rescata al agente** ΔSharpe **+0.66 IC95[0.23,1.16]** y ΔmaxDD **+0.24 IC95[0.017,0.44]** (ambos excluyen 0).
4. **Ablación** agente15→ALL22: Δacc medio ≈ 0 (mixto) → el efecto STRATA NO es "más accuracy del meta-learner",
   es **rescate del agente** (riesgo) + **SHAP** (el modelo se apoya en STRATA). Reportado honesto.
5. **Clustering:** KMeans/Ward/GMM coinciden (Rand=1.0, k=3); Spectral difiere (0.40). 3 grupos → la elección
   del algoritmo final la decide Raquel. Patrón: leverage-fuerte (índices)→Régimen/M8; volátiles
   (NVDA/TSLA/ROKU/MARA)→AutoML/M10; alta-vol/inverso (MSTR/SMCI/UNG)→M10/M8.
6. **M10 = canónico de la memoria** (ensemble 10 XGBoost, embargo=1, ALL22; SMCI M10≈0.552 verificado).
