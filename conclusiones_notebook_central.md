# Contexto de la tesis — marco práctico STRATA (Cap. 4) y qué falta del Cap. 3

> **Para la otra sesión (redacción).** Este es el **documento de contexto único** del enfoque actual, extraído y
> **verificado** del notebook canónico `notebooks/STRATA_marco_practico.ipynb` (87 celdas, 0 errores, AUTO-TEST
> verde) y de sus JSON en `outputs/experiments/`. Sirve para: (a) **redactar el Cap. 4** con propiedad; (b) saber
> **qué desarrollar del Cap. 3**; (c) **actualizar** los docs raíz (mapeo al final).
>
> **Enfoque actual = SPY (caso de estudio) + PANEL DE 10 + PATRONES.** Ya **no** es el caso de uso SMCI. **No hay
> apéndice**: los 5 activos que no están en el panel (MSTR, NVDA, BAC, TSLA, IWM) **quedan fuera del TFG**. El
> panel es: **SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG** (SMCI sí está, es uno de los 10).
>
> **Distribuir** en `DECISIONES_ESENCIALES.md` / `RESULTADOS_OBJETIVO.md` / `CONOCIMIENTO_ACUMULADO.md` (mapeo §XI).

---

## I. Tesis y encuadre honesto

**Frase-tesis (jerarquía de valor, contrastada peldaño a peldaño):**
> El aprendiz **RESCATA al agente** (significativo) > **BATE modestamente a la regla en accuracy** (TOST: superior,
> no equivalente, por flexibilidad no lineal) > **EMPATA con la regla en riesgo** (Sharpe indistinguible) > **NO
> BATE a lo trivial** (ZeroR, accuracy nominal).

**Alcance y honestidad (línea roja).** El listón de STRATA es **el agente, no el mercado**: es una capa de
supervisión y control de riesgo; generar alfa absoluta queda **fuera de su alcance por diseño**. La superioridad en accuracy frente a las triviales
(ZeroR/B&H) es **nominal** (McNemar AutoML vs ZeroR p≈0.90, n≈250 sin potencia). Lo que **sí** sobrevive a un test:
(a) **rescate del agente** en accuracy y en riesgo; (b) **universalidad** (el ML usa STRATA, SHAP); (c) **patrón**
naturaleza→canal. El objetivo del trabajo **no es vencer en accuracy** sino **medir y entender el valor de la
supervisión** con patrones falsables.

---

## II. Objetivos del proyecto (con su validación)

| # | Objetivo | Validación |
|---|---|---|
| **O1** | El agente solo (M5) pierde y acierta < 0.5 | SPY M5 acc 0.366, Sharpe −3.07, equity 0.699; sign test vs 0.5 |
| **O2 (central)** | STRATA **rescata** al agente | accuracy: McNemar vs M5 sig; riesgo: pooled-10 ΔSharpe sig |
| **O3** | Mecanismo de **dos capas** (regla=riesgo, aprendiz=accuracy) | pooled ΔSharpe (M8) + McNemar (aprendiz), cada uno con su test |
| **O4** | **Universalidad:** el ML redescubre STRATA y la supera modestamente | cuota SHAP 0.66 + **TOST** (superior en accuracy, equivalente en riesgo) |
| **O5 (patrón)** | La **naturaleza (leverage)** explica qué supervisión funciona | ley leverage→rescate (sobre 10) + clustering + complementariedad por régimen |
| **O6** | **Alcance: supervisión, no alfa** | listón = el agente; accuracy nominal vs trivial; alfa = línea futura (§XIV) |
| **O7** | **Rigor** | signal_lag=1, embargo=1, ex-ante, tests con cita, reproducibilidad determinista |

---

## III. Conclusiones (C1–C9), CLARAS y con igual peso — cada una con su test

**C1. El agente pierde; STRATA lo rescata en RIESGO (significativo).** SPY M5 acc 0.366, Sharpe −3.07, equity
0.699. Rescate de riesgo (resultado duro): **pooled-10 M8 vs M5 ΔSharpe +0.64 IC95[0.10, 1.29]** (excluye 0); M10
**+0.93**, AutoML **+0.97**. Confirmatorio (cota Bonferroni, m=3): M10/AutoML pasan, **M8 sola no** (falsación
honesta de la regla en el plano riesgo). **DSR (deflactado, n_trials=6): AutoML-SPY = 0.924.**

**C2. STRATA rescata en ACCURACY (significativo).** McNemar vs M5 → **AutoML p=0.0002 · M10 p=0.0074 · M8
p=0.051** (SPY). En el panel, la mejor STRATA mejora al agente en accuracy en **10/10** (media +0.086).

**C3. El ML redescubre STRATA y la BATE modestamente (TOST).** SHAP: cuota de las features de STRATA en el
aprendiz **>0.5 en 10/10** (media **0.66**; SPY canónica 0.565 tree / 0.564 permutation). **Test de equivalencia
(TOST, Schuirmann 1987):** NO hay equivalencia; el aprendiz **supera a la regla M8 en accuracy** (pooled M10 Δacc
**+0.021** IC90[+0.001,+0.039]; AutoML **+0.034** IC90[+0.010,+0.056]; SPY AutoML fuerte). En **Sharpe es no
concluyente** → regla y aprendiz **indistinguibles en riesgo**. Lo explica el mecanismo (C5): el aprendiz modela
interacciones no lineales que la regla fija no puede.

**C4. Dos capas complementarias (mecanismo).** **M8 = capa de riesgo** (pooled ΔSharpe sig, interpretable: 100%
del P&L de rescate es del canal régimen/RAM; casi nunca es el máximo de accuracy por activo, 0/10). **Aprendiz
M10/AutoML = capa de accuracy** (McNemar vs M5 sig). **Gate RAM:** cuando RAM dispara, seguir el régimen bate a
seguir al agente en **6/10**; la intervención crece con la discrepancia agente↔régimen (**Pearson r=0.93,
p<0.001**). **Anatomía:** de 121 intervenciones en SPY, M8 acierta 58.7% vs 41.3% del agente (71/50; P&L +0.312).

**C5. PATRÓN naturaleza→canal: la ley del leverage (IMPORTANTE).** El rescate del aprendiz en accuracy **escala
con el leverage effect**: **Pearson r=−0.56 (p=0.093), Spearman ρ=−0.59 (p=0.074) sobre los 10** → **tendencia
significativa al α=0.10** del proyecto, **visible en el panel**. Honesto: con n=10 **no** se afirma p<0.05 ni
robustez a leave-one-out. La tabla por activo (abajo) la hace tangible: 9/10 la siguen (solo ROKU va a
contracorriente); índices de leverage fuerte rescatan **0.097** de media vs **0.059** los de leverage débil.
Ninguna variable predice el valor de la **regla** (crisis_mean/leverage/sesgo corto, todas p>0.14) → el
discriminante `crisis_mean` es **descriptivo, no una ley**. **Clustering (10, consenso unánime de
KMeans/Ward/GMM/Spectral, Rand=1.0, silhouette 0.55):** C0 índices leverage fuerte (SPY/QQQ/XLF/DIA/XLK/XLE) →
AutoML; C1 leverage invertido (SMCI/UNG) → M10; C2 volátiles (ROKU/MARA) → AutoML. **PC1≈leverage (r=0.83)** →
cadena cerrada naturaleza→eje del clustering→rescate. Casos mecánicos: XLE (régimen corrige, M8 acierta 0.57 al
intervenir) vs MARA (leverage invertido, M8 acierta <0.5; solo el aprendiz rescata).

### Tabla leverage↔rescate por activo (núcleo del discurso de patrones — verificada)

| activo | leverage_corr | M5 | M10 | AutoML | rescate (max−M5) | ¿cumple la ley? |
|---|---|---|---|---|---|---|
| DIA  | −0.112 | 0.440 | 0.468 | 0.520 | 0.080 | ✓ |
| SPY  | −0.110 | 0.366 | 0.494 | 0.574 | **0.207** | ✓✓ |
| QQQ  | −0.092 | 0.418 | 0.522 | 0.534 | 0.116 | ✓ |
| XLF  | −0.091 | 0.429 | 0.526 | 0.510 | 0.097 | ✓ |
| XLE  | −0.089 | 0.448 | 0.508 | 0.532 | 0.085 | ✓ |
| XLK  | −0.086 | 0.510 | 0.542 | 0.590 | 0.080 | ✓ |
| MARA | −0.059 | 0.528 | 0.532 | 0.544 | 0.016 | ✓ |
| SMCI | −0.004 | 0.484 | 0.552 | 0.472 | 0.068 | ≈ |
| ROKU | +0.003 | 0.444 | 0.508 | 0.544 | 0.100 | ✗ (excepción) |
| UNG  | +0.041 | 0.510 | 0.518 | 0.482 | 0.008 | ✓✓ |

Medias: leverage fuerte (corr<−0.05) → **0.097**; leverage débil (≥−0.05) → **0.059**. Pearson(10) r=−0.559,
p=0.093. *Por qué p=0.09 y no p<0.05: la correlación es igual de fuerte que sobre el universo mayor; con menos
puntos el test es menos potente. La ley SE CUMPLE en los 10 — no es que "con 10 no pase".*

**C6. Complementariedad por régimen — SIGNIFICATIVA (DiD).** M10 rescata más en alcista (ΔSharpe +1.37 vs +0.72),
AutoML más en bajista (+1.52 vs +0.81); M8 simétrica. **Test diferencia-en-diferencias pre-registrado: pooled DiD
+1.37, IC95[+0.20,+2.60] (excluye 0), p=0.008.** Fenómeno de **panel** (no aparece en SPY-solo). Implicación:
AutoML protege mejor en el régimen **peligroso** (bajista). Línea futura: ensemble enrutado por régimen.

**C7. El rescate no es de un solo régimen.** McNemar pooled (sup vs M5) sig en **alcista Y bajista** (6/6). A nivel
SPY-solo el rescate de Sharpe se concentra en alcista y la regla se invierte en bajista (n=50, falsación
pre-registrada); la agregación lo resuelve.

**C8. La ablación depende del modelo; el ganador usa STRATA.** AutoML alcanza su máximo con las 22 features (acc
0.574) y degrada al quitar PSA+GSO (0.550) → usa los detectores. M10-XGBoost (params fijos) se sobreajusta con 22.
PSA/GSO apenas disparan como reglas (RAM domina) pero sus scores continuos informan al aprendiz.

**C9. Alcance: supervisión, no alfa.** El listón de STRATA es **el agente, no el mercado** — rescata al perdedor y
acota su riesgo. No se bate a ZeroR/B&H en accuracy con significancia (nominal, ventana corta n≈250); generar alfa
absoluta queda **fuera de su alcance por diseño**. *(La lectura alfa-vs-beta que sugiere valor direccional en
SMCI/MARA/UNG es descriptiva y va a **líneas futuras** §XIV — NO es una conclusión del trabajo.)*

---

## IV. Decisiones esenciales (las que no se rediscuten)

1. **Caso central = SPY; panel = 10 activos** (SPY,QQQ,XLF,DIA,XLK,XLE,ROKU,SMCI,MARA,UNG), **sin apéndice**.
   Selección ex-ante por naturaleza (clases con distinto leverage), no por significancia per-activo.
2. **Notebook único canónico** = `STRATA_marco_practico.ipynb`.
3. **Calibración 2000→2024-09-30** (una vez, ex-ante); **OOS 2024-10-01→cierre**; **dos ventanas nunca mezcladas**:
   OOS completo (n≈401, M5/M8/ZeroR/B&H) y desplegable (n≈250 tras burn-in, M10/AutoML; SPY n=251).
4. **`signal_lag=1`** (P&L = w_t·r_{t+1}); **embargo=1** en walk-forward (horizonte=1, rolling-origin).
5. **HMM K=3** por verosimilitud held-out (−1.30 > −1.69 de K=2) + interpretabilidad; calibrado por activo.
6. **AutoML canónico** = H2O, **max_models=25** (NUNCA runtime, no reproducible), GBM/XGBoost/StackedEnsemble, AUC,
   Purged K-fold, **seed=42** → determinista. **M10** = ensemble 10 XGBoost (42–51), params fijos, ALL22.
7. **Riesgo agregado = pooled-10** (apilar los días de los 10) — la significancia vive en el agregado, no per-activo.
8. **prior RAM data-driven por activo** (signo de la media del régimen), nunca "Crisis⇒short" hardcoded.

---

## V. Resultados esenciales (headline, con su test)

- **SPY, 6 estrategias (desplegable, n=251):** M5 0.366 (Sharpe −3.07) · M8 0.442 · M10 0.494 · **AutoML 0.574** ·
  ZeroR/B&H 0.566. McNemar AutoML vs ZeroR **p=0.90** (nominal); McNemar AutoML/M10/M8 vs M5 = **0.0002/0.0074/0.051**.
- **Pooled-10 riesgo:** M8 vs M5 ΔSharpe **+0.64 IC95[0.10,1.29]**; M10 +0.93; AutoML +0.97 (todos sig).
- **Confirmatorio Bonferroni + DSR:** M10/AutoML pasan; M8 no; **DSR AutoML-SPY 0.924**.
- **TOST aprendiz vs regla:** accuracy superior (pooled M10 +0.021, AutoML +0.034); Sharpe no concluyente.
- **DiD complementariedad:** +1.37, IC95[0.20,2.60], p=0.008 (pooled; no en SPY-solo).
- **SHAP cuota:** media 0.66, >0.5 en 10/10. **Clustering:** Rand=1.0, silhouette 0.55. **Ley leverage:** r=−0.56,
  p=0.093 (α=0.10).

---

## VI. Cap. 4 (marco práctico) — qué debe aparecer

Estructura del notebook (a replicar en prosa): **§1** datos/panel-10/protocolo → **§2** mecánica ex-ante (HMM K=3,
GARCH-t, BOCPD; leverage honesto; intervención/atribución por detector; anatomía de la intervención; ablación
M10/AutoML; régimen×dirección) → **§3** caso SPY (tabla 6 estrategias + McNemar; equity; override vs abstención;
M10 vs M5 por régimen + SHAP dependency; descriptivo por variable) → **§4** panel de 10 (activación detectores;
gate RAM; naturaleza por activo; mejor-STRATA vs agente; ablación + SHAP; **TOST**; heatmap; **pooled-10 riesgo**)
→ **§5** mecanismo dos capas + **ley del leverage (tabla por activo)** + casos XLE/MARA → **§6** clustering
(naturaleza→canal, Rand=1.0) + 4 gráficas por grupo → **§7** robustez (rodante, val/test, **PARTE B confirmatoria
Bonferroni+DSR**, **rescate por régimen + DiD**, calibración) → **§8** conclusiones. **Sin §-apéndice.** Liderar
con **rescate + patrón**, no con la accuracy.

---

## VII. Cap. 3 (marco teórico) — qué QUEDA por desarrollar y con qué enfoque

**La parte matemática de los MODELOS se deja TAL CUAL** (completa y rigurosa, auditada): preliminares, HMM
(forward/filtrado/Baum-Welch/Viterbi/K), GARCH(1,1)-t, BOCPD, los 3 detectores, capa M8, aprendizaje supervisado
(árboles/boosting/XGBoost/M10/stacking/AutoML). **No se reescribe.** Solo falta la **caja de validación**:

- **Vacío pero se usa → escribir:** §métricas (Sharpe, maxDD, Calmar); §validación (purga y embargo,
  CPCV/por-qué-no-KFold); §tests (sign test, McNemar, bootstrap estacionario, DSR).
- **Se usa y NO tiene sección → añadir:** **AUC**; **accuracy direccional + matriz de confusión** (precision/recall,
  la métrica titular); **block-permutation**; **Holm/Bonferroni** (multiplicidad); **Pearson/Spearman** (la ley);
  **TOST** (equivalencia/superioridad aprendiz vs regla); **SHAP/TreeSHAP + permutation importance**
  (universalidad); **clustering** (KMeans/Ward/GMM/Spectral, PCA, silhouette, BIC, índice de Rand); definición de
  **pooled** (panel) + su caveat.
- **Quitar (no se usan):** **Sortino**, **Diebold-Mariano**.
- **Reorganizar la sección de tests por la pregunta que responde:** dirección (sign/McNemar/block-perm) · riesgo
  (bootstrap/DSR) · multiplicidad (Holm/Bonferroni) · equivalencia (TOST) · asociación (Pearson/Spearman). Orden
  sugerido de redacción: métricas → SHAP → clustering → tests.

---

## VIII. Defensa: objeción del tribunal → respuesta

- *"Solo funciona en alcista"* → rescate sig en **alcista Y bajista** (pooled, 6/6 McNemar).
- *"El pooled infla la significancia"* → caveat n efectiva < nominal (correlación cruzada); IC excluye 0 con
  holgura; **block bootstrap** (respeta dependencia).
- *"M8 no pasa la cota Bonferroni en Sharpe"* → test **más exigente** y **otra serie** (±1 vs net-causal); M8
  rescata en el titular pooled-10; bajo corrección por familia quien sostiene el rescate es el meta-learner.
- *"¿Redescubre o bate a la regla?"* → **TOST**: bate **modestamente** en accuracy, **empata** en riesgo.
- *"¿AutoML es reproducible?"* → **max_models + seed** (no runtime) → determinista.
- *"n≈250 es poco"* → por eso la significancia vive en el **pooled**; accuracy declarada **nominal**.
- *"La ley del leverage es p=0.09"* → **se cumple en los 10** (tabla por activo), sig al **α=0.10**; no se afirma
  p<0.05 ni robustez LOO.
- *"Habéis elegido los 10 a mano"* → selección **ex-ante por naturaleza**, no por significancia; el valor (rescate)
  vive en el pooled, no per-activo.

---

## IX. Líneas rojas (lo que NUNCA se afirma)

No "bate al mercado"; no "bate a ZeroR significativamente"; no "genera alfa"; no presentar la accuracy nominal como
significativa; no afirmar la ley a p<0.05 (sobre 10); no afirmar robustez LOO (sobre 10); no mezclar ventanas
(OOS-completo n≈401 vs desplegable n≈250).

---

## X. Figuras/tablas clave (criterio: una figura = evidencia de un objetivo, fuente trazable)

**Capítulo: ~6–8 curadas, NO las ~30 del notebook** (una galería diluye los resultados de carga). Tiers:
- **TIER 1 — carga (4):** (1) forest plot **pooled-10 ΔSharpe** (O2 riesgo); (2) tabla **6 estrategias SPY +
  McNemar** (O2 accuracy); (3) **tabla leverage↔rescate + scatter** (O5 ley); (4) **cuadro 2×2 del TOST** (O4).
- **TIER 2 — refuerzo (2–3):** equity SPY (O1); clustering PCA Rand=1.0 (O5); gate RAM o complementariedad DiD.
- **TIER 3 — apoyo (al notebook, no al cuerpo):** confusión, SHAP cuota, régimen×dirección, rodante/val-test,
  calibración, descriptivo, activación de detectores.

Mapeo figura→prueba→fuente: ver tabla del plan (`marco_practico_dossier.md` lo mantiene en detalle).

**Arco narrativo del Cap. 4:** **funciona** (rescate sig accuracy + riesgo) → **por qué** (dos capas +
naturaleza→leverage) → **límites** (nominal vs trivial, no alfa).

---

## XI. Mapa de distribución a los docs raíz (para la otra sesión)

- `DECISIONES_ESENCIALES.md`: decisión #18 → **"panel de 10, SIN apéndice"** + jerarquía de valor; los 5 fuera del
  TFG; pooled de riesgo = **pooled-10**. Reformular #13–#17 (SMCI-era) a su lugar histórico; SMCI ya no es el caso
  central (es uno de los 10).
- `RESULTADOS_OBJETIVO.md`: §1ter → **pooled-10 +0.64**, **TOST**, **DiD**, ley leverage **sobre 10 (p=0.093)**.
  Retirar §1bis-como-apéndice.
- `CONOCIMIENTO_ACUMULADO.md`: banner → caso central **SPY + patrones** (no SMCI); el valor es rescate + dos
  capas + naturaleza→leverage.
- `LECCIONES_APRENDIDAS.md`: notebook canónico = `STRATA_marco_practico.ipynb`.

---

## XII. Caveats globales declarados

1. **Accuracy nominal** vs lo trivial (ventana corta n≈250 → línea futura).
2. **pooled:** apila días-activo como independientes; correlación cruzada → **n efectiva < nominal** (IC algo
   optimista); el resultado (+0.64) excluye 0 con holgura. Block bootstrap.
3. **Ley del leverage:** sobre 10, **tendencia sig al α=0.10** (p=0.093); no p<0.05 ni LOO-robusta. Se cumple en
   el panel (tabla por activo).
4. **Clustering n=10:** exploratorio; qué modelo concreto gana por activo NO es predecible por el cluster.
5. **Complementariedad por régimen:** sig en pooled, **no** en SPY-solo (fenómeno cross-asset).
6. **Alcance = supervisión, no alfa:** el listón es el agente, no el mercado; generar alfa absoluta queda fuera de
   alcance por diseño. (El indicio de valor direccional en SMCI/MARA/UNG va a líneas futuras §XIV, no a conclusiones.)

---

## XIV. Líneas de investigación futura (NO son conclusiones del trabajo)

1. **Ensemble enrutado por régimen** (M10 en alcista / AutoML en bajista) usando la señal de RAM — motivado por la
   complementariedad significativa (DiD p=0.008). Pre-registrable; enrutar post-hoc sería p-hacking.
2. **¿Puede STRATA generar alfa direccional robusta en leverage débil/invertido?** La **lectura alfa-vs-beta
   (F4.9, descriptiva, modelo de mercado, SIN test)** lo sugiere como *indicio nominal*: en índices alcistas
   (SPY/QQQ/XLE) el Sharpe positivo es **beta** (β≈0.5–0.7, va largo); en **SMCI/MARA/UNG** (leverage débil/invertido,
   B&H Sharpe 0.04/−0.28/−0.83) la mejor STRATA saca Sharpe positivo (1.91/1.28/0.67) yendo corta/defensiva (β
   bajo/negativo) → valor **direccional**, no exposición. **Nominal, a posteriori, sin contraste** → no se reporta
   como resultado; queda como hipótesis a probar con muestra mayor y pre-registro. Fuente: `alfa_beta_lectura.json`.
3. **Ventana OOS mayor** para llevar la accuracy de nominal a significativa (el techo ZeroR actual es por n≈250).

---

## XIII. Trazabilidad (resultado → experimento → JSON)

| Resultado | JSON |
|---|---|
| Tabla 6 estrategias + McNemar | `automl_runs/panel_mm25_*.json` |
| Pooled-10 riesgo + SHAP cuota + ablación | `decision_automl_prep.json` (recomputado a 10 en el notebook) |
| Series netas AutoML | `automl_net_returns.json` |
| Ley leverage + casos XLE/MARA | `mechanism_panel.json`, `detector_analysis_*.json` |
| Clustering 10 + gráficas por grupo | `cluster_panel10.json` |
| Activación/ablación detectores | `detector_ablation_panel.json`, `automl_ablation_detectors.json` |
| Matrices de confusión | `confusion_panel.json` |
| Anatomía intervención · gate RAM | `spy_intervention_anatomy.json`, `spy_panel_gate_descriptive.json` |
| PARTE B confirmatoria + por régimen | `bullbear_confirmatory.json` |
| DiD complementariedad | `regime_did_learners.json` |
| TOST aprendiz vs regla | `equivalence_tost.json` |
| Lectura alfa-vs-beta (F4.9) | `alfa_beta_lectura.json` |
| Rodante/val-test/bull-bear · calibración | `panel_robustness.json`, `calib_window_panel.json` |
