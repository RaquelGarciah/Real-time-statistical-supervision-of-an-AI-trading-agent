# Conclusiones del notebook central — STRATA, marco práctico (Cap. 4)

> **Qué es este documento.** Extracción **verificada y validada** del notebook canónico
> `notebooks/STRATA_marco_practico.ipynb` (89 celdas, 0 errores, AUTO-TEST verde): los **objetivos**, las
> **decisiones**, los **resultados** y las **conclusiones** que el proyecto ha sido capaz de sostener, con su
> cifra, su test y su límite. Todas las cifras están cruzadas contra los `outputs/experiments/*.json` (auto-test
> + auditoría independiente). Solo contiene lo que **está en el notebook** y dimos por bueno; lo descartado
> (costes, métrica de producción, config mm20, data-augmentation) **no entra**. Sirve para que la memoria escriba
> el Cap. 4 con propiedad. Complementa al `docs/marco_practico_dossier.md` (desarrollo por secciones); aquí va lo
> esencial: objetivos + conclusiones validadas.
>
> **Estado de validación (auditoría 2026-06-23).** Cifras SPY coherentes entre §1/§2/§3/ablación; reconciliación
> SHAP (0.565 SPY tree vs 0.715 otro árbol) declarada; universo 10 vs 15 sin contaminación; honestidad sólida.
> Se corrigió la única incoherencia grave (el TOST no estaba propagado al notebook → ya lo está) y dos menores
> (un "n=15" del clustering→10; una línea duplicada en §9).

---

## La tesis en una frase

Supervisar estadísticamente a un agente LLM de trading **aporta valor diferencial medible**, y ese valor se ordena
en una **jerarquía honesta, contrastada peldaño a peldaño**:

> **El aprendiz RESCATA al agente (significativo) > BATE modestamente a la regla en accuracy (TOST: superior, no
> equivalente, por flexibilidad no lineal) > EMPATA con la regla en riesgo (Sharpe indistinguible) > NO BATE a lo
> trivial (ZeroR, accuracy nominal).**

El objetivo inicial era **vencer en accuracy**; el resultado honesto es que, en este OOS corto, **nada bate a las
triviales en accuracy de forma significativa**. La contribución real —y más científica— es el **mapa falsable de
cuándo y cómo la supervisión rescata al agente**: qué canal según la naturaleza del activo, en qué régimen, con
qué modelo y con qué mecanismo. **STRATA rescata, ordena el valor con rigor y acota; no genera alfa.**

---

## Objetivos del proyecto (con su validación)

| # | Objetivo | Veredicto | Evidencia (test + cifra) |
|---|---|---|---|
| **O1** | El agente solo (M5) pierde y acierta < 50 % | ✅ | SPY M5 acc 0.366, Sharpe −3.07, equity 0.699; sign test vs 0.5 p<0.001 |
| **O2** | STRATA **rescata** al agente | ✅ | accuracy: McNemar M10/AutoML vs M5 p=0.0074/0.0002 (SPY); riesgo: pooled-15 M8 vs M5 ΔSharpe +0.66 IC95[0.225,1.157] |
| **O3** | El ML **redescubre** STRATA y la **supera modestamente** | ✅ (refinado) | cuota SHAP 0.66 (usa STRATA) + **TOST**: bate a M8 en accuracy (pooled), indistinguible en riesgo |
| **O4** | **Dos capas** complementarias (regla=riesgo, aprendiz=accuracy) | ✅ | pooled ΔSharpe M8 sig + McNemar aprendiz sig; complementariedad por régimen (DiD p=0.008) |
| **O5** | **Patrón** naturaleza→resultado | ✅ (1 ley) | ley leverage→rescate-ML: Pearson r=−0.55 p=0.034, robusta a LOO (p_max 0.095); clustering PC1≈leverage r=0.83 |
| **O6** | **Honestidad** y límite | ✅ | no bate a ZeroR (nominal, p=0.90); apéndice de 5 excluidos; leverage débil declarado |
| **O7** | **Rigor** | ✅ | signal_lag=1, embargo=1, ex-ante, tests con cita, auto-test cruza cada cifra con su JSON |

---

## Decisiones fijadas (las que no se rediscuten)

1. **Caso de estudio = SPY**; **cuerpo = 10 activos** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG),
   **apéndice = 5** (MSTR, NVDA, BAC, TSLA, IWM). Selección **ex-ante por naturaleza**, no por significancia
   per-activo (con n≈250 casi nada es significativo; la significancia vive en el pooled).
2. **Notebook único canónico** = `STRATA_marco_practico.ipynb` (decisión #18).
3. **Calibración 2000→2024-09-30** (una vez, ex-ante); **OOS 2024-10-01→cierre** (posterior al cutoff del LLM);
   **desplegable** ≈250 días tras burn-in para los meta-learners. **Dos ventanas, nunca mezcladas.**
4. **`signal_lag=1`** (P&L = w_t·r_{t+1}); **embargo=1** en walk-forward (horizonte=1, rolling-origin).
5. **HMM K=3** por verosimilitud held-out (−1.30 > −1.69 de K=2) + interpretabilidad (Calma/Estrés/Crisis); no
   por el OOS. Calibrado **por activo**.
6. **AutoML canónico** = H2O, **max_models=25** (NUNCA `max_runtime_secs`, que no es reproducible),
   GBM/XGBoost/StackedEnsemble, AUC, Purged K-fold, **seed=42** → determinista. **M10** = ensemble de 10 XGBoost
   (semillas 42–51), params fijos, ALL22 features.
7. **Pooled-15** (`decision_automl_prep.json`, n=3751) = cifra **canónica de riesgo**; pooled-10 (n=2493) =
   sensibilidad. **Clustering = sobre los 10** (la versión de 15 se conserva como respaldo).
8. **prior RAM data-driven por activo** (signo de la media del régimen), nunca "Crisis⇒short" hardcoded.

---

## Conclusiones validadas (cada una con su test)

### C1 — El agente pierde; STRATA lo rescata en riesgo (significativo) · §3, §4
- SPY: M5 acc **0.366**, Sharpe **−3.07**, equity **0.699** (pierde dinero, acierta < 50 %).
- **Rescate de riesgo (resultado duro):** pooled-15 **M8 vs M5 ΔSharpe +0.66 IC95[0.225, 1.157]** (excluye 0);
  ΔmaxDD +0.24 IC95[0.017, 0.445]. Pooled-10 consistente (M8 +0.64, M10 +0.93, AutoML +0.97, todos sig).
- Confirmatorio (cota **Bonferroni**, m=3): **M10 y AutoML pasan** (SPY +0.02/+1.91; pooled +0.26/+0.26);
  **M8 sola NO** (SPY −0.66; pooled −0.05) → **falsación honesta de la regla pura** en el plano riesgo.
- **DSR (deflactado, n_trials=6): AutoML-SPY = 0.924**; M5/M8/M10 reprueban (M10-SPY 0.048 → pasa Bonferroni por
  *rescate* vs un M5 pésimo, no por skill absoluta). *Puente honesto:* que M8 no pase Bonferroni (pooled-10, ±1,
  corrección por familia) no contradice el titular de riesgo (pooled-15, retorno neto, IC95) — es el mismo efecto
  bajo un test más exigente y otro universo; quien **sostiene** el rescate en Sharpe bajo corrección es el aprendiz.

### C2 — El aprendiz redescubre STRATA Y la bate modestamente (TOST) · §4
- **SHAP:** cuota de las features de STRATA en el aprendiz **>0.5 en 10/10** (media **0.66**; SPY canónica 0.565
  tree / 0.564 permutation) → el ML **usa** las señales de STRATA. *(La columna 0.715 de la tabla SPY es otro
  árbol del ensemble; reconciliación declarada — ambas >0.5.)*
- **Test de equivalencia (TOST, Schuirmann 1987):** **NO hay equivalencia; el aprendiz BATE a la regla M8 en
  accuracy** — pooled M10 Δacc **+0.021** IC90[+0.001,+0.039], AutoML **+0.034** IC90[+0.010,+0.056] (modesto,
  2–3 pp); SPY AutoML con fuerza (+0.132). En **Sharpe es no concluyente** → regla y aprendiz **indistinguibles
  en riesgo**.
- **Lectura:** batir a la regla **no es "otra señal"**, es **la misma combinada con más flexibilidad** — el
  aprendiz modela interacciones no lineales que la regla determinista no puede (los activos de leverage invertido
  donde M8 falla, ver C4). Refuta el viejo "redescubre y no bate" sin romper la universalidad (el SHAP aguanta).

### C3 — El rescate no es de un solo régimen de mercado · §7
- McNemar pooled (sup vs M5) **significativo en alcista Y bajista** (6/6 contrastes; M10/AutoML p<0.02 en ambos).
- A nivel SPY-solo el rescate de Sharpe se concentra en alcista y la **regla M8 se invierte en bajista**
  (ΔSharpe −1.49, n=50) → **falsación pre-registrada** cumplida al nivel de un activo; **la agregación la resuelve**.

### C4 — [PATRÓN] Complementariedad por régimen de los aprendices (significativa, cross-asset) · §7
- En el pooled-10, los dos aprendices se reparten los regímenes **en espejo**: **M10 rescata más en alcista**
  (ΔSharpe +1.37 vs +0.72), **AutoML más en bajista** (+1.52 vs +0.81); la **regla M8 es simétrica** (+0.63/+0.55).
- **Test diferencia-en-diferencias (pre-registrado):** **significativo en el pooled** (DiD +1.37, IC95[+0.20,+2.60]
  excluye 0, p=0.008), **no en SPY-solo** (IC cruza 0; AutoML domina ambos regímenes ahí) → es un **fenómeno de
  PANEL/cross-asset**, no de un activo.
- **Implicación:** AutoML —que modela la interacción condicional— protege mejor en el régimen **peligroso**
  (bajista). **Línea futura pre-registrable:** ensemble **enrutado por régimen** (M10 alcista / AutoML bajista)
  vía la señal de RAM (no validado; enrutar post-hoc sería p-hacking).

### C5 — [PATRÓN] La naturaleza del activo gobierna el canal — la única ley que sobrevive un test · §5, §6
- **Ley leverage→rescate-ML:** el rescate del aprendiz en accuracy **escala con el leverage effect** — Pearson
  **r=−0.55 (p=0.034)**, Spearman ρ=−0.54 (p=0.038), **robusta a leave-one-out** (peor caso drop-MSTR r=−0.46
  p=0.095 < 0.10) → ningún activo es punto influyente.
- **Honesto:** **ninguna** variable de naturaleza predice el valor de la **regla** M8 (crisis_mean, leverage,
  sesgo corto: todas p>0.14) → el discriminante `crisis_mean<0⇒régimen` es **descriptivo, no una ley**.
- **Clustering (10, consenso unánime KMeans/Ward/GMM/Spectral, Rand=1.0, silhouette 0.55):** 3 grupos por
  naturaleza — **C0 índices leverage fuerte** (SPY/QQQ/XLF/DIA/XLK/XLE) → AutoML; **C1 leverage invertido**
  (SMCI/UNG) → M10; **C2 volátiles** (ROKU/MARA) → AutoML. **PC1≈leverage (r=0.83)** → cadena cerrada:
  naturaleza (leverage) → eje del clustering → rescate del aprendiz.
- **Mecanismo (dos casos):** XLE (leverage presente) → la regla del régimen corrige (M8 acierta 0.57 al
  intervenir); MARA (leverage invertido) → la regla mete ruido (acierta <0.5) y **solo el aprendiz** rescata
  volteando el sesgo corto del agente.

### C6 — [MECANISMO] Las dos capas hacen trabajos distintos y visibles · §2, §4, §6
- **M8 = capa de riesgo:** por grupo, levanta el Sharpe del agente (ΔSharpe ≈ +1.3 en índices/volátiles; ≈0 en
  leverage invertido, donde gana M10). **Casi nunca es el máximo de accuracy** por activo (0/10), pero rescata el
  riesgo.
- **Gate RAM:** cuando RAM dispara, seguir el **régimen** bate a seguir al **agente** en **6/10**; y la
  **intervención de M8 crece con la discrepancia agente↔régimen** (Pearson **r=0.93, p<0.001**) → STRATA actúa
  donde el agente se aparta del régimen, como se diseñó.
- **Anatomía de la intervención (SPY):** 121 intervenciones, M8 acierta **58.7 % vs 41.3 %** del agente
  (**71 aciertan, 50 fallan**), P&L rescate +0.312. Dos días reales, mismo mecanismo, desenlace opuesto
  (2024-11-05 +2.46 % acierta / 2024-10-30 −1.98 % falla) → favorable en el agregado, **no infalible**.

### C7 — [MECANISMO] Que STRATA ayude como feature depende del modelo; el ganador la usa · §2
- **AutoML** alcanza su máximo con las 22 features (acc 0.574) y **degrada al quitar PSA+GSO** (0.550) → **sí**
  extrae su señal. **M10-XGBoost (params fijos) se sobreajusta** con 22 (0.494 < agente-15 0.542).
- **PSA/GSO** apenas disparan como reglas (RAM domina; P&L de rescate 100 % RAM), pero sus **scores continuos**
  informan a un aprendiz capaz. Se conservan por doble motivo: el mejor modelo los usa **aquí**, y son ejes de
  seguridad para regímenes que este OOS calmado no contiene (en calibración con 2008/2020 **sí** disparan).
- La dirección **no es univariante**: la mejor variable sola (crisis_prob) separa al 0.594; el valor está en
  **combinar las 22** + el plano riesgo.

### C8 — Honestidad y límite (lo que NO se afirma) · §3, §7, §8
- **No se bate a ZeroR/B&H en accuracy de forma significativa** (SPY AutoML vs ZeroR p=0.90) → la superioridad en
  accuracy es **nominal** (ventana corta, n≈250 → línea futura). El valor está en el **rescate** y el **riesgo**.
- **Robustez:** rodante (mejor STRATA > agente en >50 % de ventanas en 9/10), val/test (8/10 en las 3
  particiones), y ventana de calibración (acortar a 2010 no daña — incluso mejora en índices; se mantiene la
  completa pre-registrada, sin p-hacking).
- **Apéndice (5):** MSTR (el agente ya gana → STRATA defiere), BAC/NVDA/TSLA (rescate no significativo o
  redundante), IWM (discriminante ambiguo). Delimitar dónde **no** aporta refuerza la tesis.
- **Punto más blando (declarado):** UNG en el cuerpo (el agente no pierde ahí; encuadrado como caso ML).

---

## Síntesis del valor (lo que la memoria debe liderar)

El trabajo aporta cuatro cosas, en orden de fuerza:
1. **Resultado duro contrastado** — el rescate de riesgo del agente (pooled bootstrap, Bonferroni, DSR). [C1]
2. **Mapa de patrones falsables** — naturaleza→canal (ley del leverage), complementariedad por régimen. El
   corazón científico. [C4, C5]
3. **Mecanismo interpretable** — dos capas, gate RAM, anatomía de la intervención. Hace la técnica entendible y
   fiable. [C6, C7]
4. **Jerarquía de valor honesta** — rescata > bate modestamente a la regla > empata en riesgo > no bate a lo
   trivial, cada peldaño con su test. La **frase-tesis** del capítulo. [C2, C3, C8]

La accuracy frente a lo trivial es **nominal** y se declara; el valor está en **entender el fenómeno de la
supervisión estadística de agentes LLM** — cuánto aporta, frente a qué y por qué—, no en un punto de acierto.

---

## Mapa de trazabilidad (resultado → experimento → JSON)

| Resultado | Experimento | JSON |
|---|---|---|
| Tabla 6 estrategias + McNemar (SPY/panel) | automl_m10 (panel canónico) | `automl_runs/panel_mm25_*.json` |
| Rescate riesgo pooled-15 (canónico) + SHAP + ablación | decision_automl_prep | `decision_automl_prep.json` |
| Series netas AutoML (reconstruidas) | automl_net_returns | `automl_net_returns.json` |
| Ley leverage→rescate + LOO; casos XLE/MARA | mechanism_panel / detector_analysis_* | `mechanism_panel.json`, `detector_analysis_*.json` |
| Clustering-10 + 4 gráficas por grupo | cluster_panel10 | `cluster_panel10.json` |
| Activación detectores + ablación M10/AutoML | detector_ablation_panel / automl_ablation_detectors | `detector_ablation_panel.json`, `automl_ablation_detectors.json` |
| Matrices de confusión | confusion_panel | `confusion_panel.json` |
| Anatomía de la intervención | spy_intervention_anatomy | `spy_intervention_anatomy.json` |
| Gate RAM + descriptivo SPY | spy_panel_gate_descriptive | `spy_panel_gate_descriptive.json` |
| PARTE B confirmatoria (Bonferroni+DSR) + por régimen | bullbear_confirmatory | `bullbear_confirmatory.json` |
| Complementariedad por régimen (DiD) | regime_did_learners | `regime_did_learners.json` |
| **¿Redescubre o bate? (TOST)** | equivalence_tost | `equivalence_tost.json` |
| Rodante / val-test / bull-bear panel | panel_robustness | `panel_robustness.json` |
| Robustez calibración | calib_window_panel | `calib_window_panel.json` |

---

## Caveats globales declarados (para que el tribunal no los "encuentre")

1. **Accuracy nominal:** no se bate a ZeroR/B&H con significancia (ventana corta n≈250 → línea futura).
2. **pooled:** apila días-activo como independientes; la correlación cruzada entre activos hace la **n efectiva <
   nominal** (el IC es algo optimista). El block bootstrap corrige autocorrelación dentro de un activo, no entre
   activos. El resultado canónico (ΔSharpe +0.66) excluye 0 con holgura.
3. **Dos universos por diseño:** riesgo = pooled-15 (retorno neto causal); accuracy/complementariedad/clustering =
   10 (los aprendices necesitan burn-in). Declarado, no mezclado.
4. **Complementariedad por régimen:** significativa en pooled, **no** en SPY-solo (fenómeno cross-asset).
5. **Clustering n=10:** exploratorio; qué modelo concreto gana por activo **no** es predecible por el cluster.
6. **STRATA no genera alfa:** los Sharpe absolutos siguen mayormente negativos; se mide el rescate **relativo** al
   agente, que es la tesis.
