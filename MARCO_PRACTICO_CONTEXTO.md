# STRATA — Documento de referencia del marco práctico (fuente de verdad de la redacción)

> **Qué es esto.** El documento de contexto **blindado y completo** del enfoque actual del TFG, extraído y
> verificado del notebook canónico `notebooks/STRATA_marco_practico.ipynb` y de sus JSON en
> `outputs/experiments/`. Representa el trabajo **entero**: objetivos, decisiones, conclusiones (todas, con su
> peso), resultados, figuras, encuadre honesto y trazabilidad. **De aquí beben** los docs raíz
> (`DECISIONES_ESENCIALES`, `RESULTADOS_OBJETIVO`, `CONOCIMIENTO_ACUMULADO`, `LECCIONES_APRENDIDAS`) y la
> redacción de los capítulos 4 y 5. Sustituye, ampliándolo, al `conclusiones_notebook_central.md` (que era
> demasiado al grano). Toda cifra está verificada contra su JSON (mapa en §IX).
>
> **Enfoque actual.** Caso de estudio **SPY** + **panel de 10 activos** + **patrones**. El TFG es **sobre 10
> activos** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG): es el universo del estudio, punto. Los otros
> activos **no existen en el trabajo** — ni panel, ni apéndice, ni "elegimos 10 de N". SMCI es **uno de los 10**.

---

## I. Tesis y jerarquía de valor (honesta, peldaño a peldaño)

**Tesis.** Un agente de trading basado en un modelo de lenguaje (HedgeFoundAI: cinco personalidades sobre un
mismo LLM) puede perder dinero y acertar la dirección menos del 50 %. La pregunta no es *"¿la IA acierta?"*,
sino *"¿una capa de supervisión estadística clásica (STRATA: régimen, cambio de opinión, volatilidad)
**rescata** a ese agente, y puede **probarse**?"*.

**Jerarquía de valor (cada peldaño con su test):**
1. El aprendiz **RESCATA al agente** — significativo, en accuracy (McNemar) y en riesgo (pooled-10).
2. **BATE modestamente a la regla** en accuracy — TOST: superioridad, no equivalencia (flexibilidad no lineal).
3. **EMPATA con la regla** en riesgo — Sharpe indistinguible (TOST no concluyente).
4. **BATE a lo trivial donde la naturaleza lo permite** — en **punto**: en SPY (AutoML gana a ZeroR/B&H en
   accuracy, Sharpe, máxima caída y equity) y en SMCI/MARA/UNG (una derivada de STRATA supera a las dos
   triviales en accuracy). Es parte del valor y se enseña; el matiz honesto es que la **significancia** pooled
   en accuracy frente a la trivial **no se alcanza** (nominal, n≈250).
5. **NO genera alfa** — los Sharpe absolutos son mayormente negativos; el valor es **relativo al agente**, no
   absoluto frente al mercado.

**Línea roja.** STRATA no genera alfa y no bate a lo trivial *con significancia* en accuracy. Lo que sí
sobrevive a un test: (a) rescate del agente en accuracy y riesgo; (b) que el aprendiz se apoya en STRATA y la
mejora (SHAP + TOST); (c) el patrón naturaleza→canal (ley del leverage); (d) la complementariedad por régimen.

---

## II. Objetivos (O1–O7) con su validación

| # | Objetivo | Validación |
|---|---|---|
| **O1** | El agente solo (M5) pierde y acierta < 0,5 | SPY M5 acc 0,366, Sharpe −3,07, equity 0,70; sign test vs 0,5 ($p<0,001$); transversal en el panel |
| **O2 (central)** | STRATA **rescata** al agente, y se prueba | accuracy: McNemar vs M5 sig; riesgo: pooled-10 ΔSharpe sig (IC excluye 0) |
| **O3** | Mecanismo de **dos capas** (regla = riesgo, aprendiz = accuracy) | pooled ΔSharpe (M8) + McNemar (aprendiz); gate RAM |
| **O4** | El aprendiz **redescubre STRATA y la mejora** | cuota SHAP 0,66 (>0,5 en 10/10) + **TOST** (superior en accuracy, equivalente en riesgo) |
| **O5 (patrón)** | La **naturaleza (leverage)** explica qué supervisión funciona | ley leverage→rescate (sobre 10) + clustering (Rand=1,0) + complementariedad DiD |
| **O6** | STRATA **gana a la trivial donde puede**, y se dice el límite | victorias en punto (SPY, SMCI, MARA, UNG); significancia vs trivial nominal; no genera alfa |
| **O7** | **Rigor** | `signal_lag=1`, embargo=1, ex-ante, tests con cita, reproducibilidad determinista |

---

## III. Decisiones esenciales (con su porqué)

1. **Caso central = SPY; universo = panel de 10** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG).
   Selección **ex ante por naturaleza** (clases con distinto leverage), no por significancia per-activo (con
   n≈250 no la habría). **No hay apéndice ni otros activos en el TFG.**
2. **Notebook único canónico** = `STRATA_marco_practico.ipynb`.
3. **Calibración 2000→2024-09-30** (una vez, ex-ante); **OOS 2024-10-01→cierre**, posterior al corte del LLM
   (sin look-ahead). **Dos ventanas, nunca mezcladas:** OOS completo ($n\approx401$, para M5/M8/ZeroR/B&H, que
   no entrenan) y desplegable ($n\approx250$ tras burn-in 150, para M10/AutoML; SPY $n=251$).
4. **`signal_lag=1`** (P&L $= w_t\cdot r_{t+1}$); **embargo=1** en walk-forward (horizonte de etiqueta = 1,
   origen rodante; protocolo desplegable estándar, Tashman/López de Prado).
5. **HMM K=3** por verosimilitud held-out ($-1{,}30 > -1{,}69$ de K=2) + interpretabilidad (Calma/Estrés/Crisis);
   calibrado por activo. **GARCH(1,1)-t** (Bollerslev); **BOCPD** (Adams-MacKay).
6. **Intervención = override-C** (reorienta al signo del régimen): voltear acierta 0,478 frente a 0,401 de
   abstenerse y 0,397 de reducir. Corregir activamente gana a evitar.
7. **prior RAM data-driven por activo** (signo de la media del régimen), nunca "Crisis ⇒ short" hardcoded.
8. **Umbrales ex-ante:** RAM $\tau=0{,}50$; PSA $P_{95}=0{,}023$; GSO $P_{95}=2{,}371$ (cuantiles de calibración).
9. **`leverage_corr` = $\operatorname{corr}(r_t,\ \mathrm{RV}^{21}_{t+1}-\mathrm{RV}^{21}_t)$** (correlación de
   Black 1976): el retorno de hoy frente al cambio de la volatilidad realizada a 21 días.
10. **M8** = override-C; **M10** = ensemble de 10 XGBoost (semillas 42–51), params fijos, ALL22 features;
    **AutoML** = H2O, **max_models=25** (determinista; nunca por tiempo), GBM/XGBoost/StackedEnsemble, AUC,
    Purged K-fold, seed=42. **Riesgo agregado = pooled-10** (apilar los días de los 10).

---

## IV. Conclusiones (C1–C9), todas con su test y su matiz

**C1. El agente pierde; STRATA lo rescata en RIESGO (significativo).** SPY M5 acc 0,366, Sharpe −3,07, equity
0,70. **Pooled-10 ΔSharpe vs M5: M8 +0,64 IC95[0,10, 1,29]; M10 +0,93 [0,19, 1,65]; AutoML +0,97 [0,27, 1,70]**
(los tres excluyen 0). Bajo **cota de Bonferroni (m=3)**: M10 y AutoML pasan; **M8 no** (cota inferior −0,047) —
falsación honesta de la regla en riesgo bajo corrección. **DSR (deflactado) AutoML-SPY = 0,92.**

**C2. STRATA rescata en ACCURACY (significativo).** McNemar vs M5: **AutoML p=0,0002 · M10 p=0,0074 · M8
p=0,051** (SPY). En el panel, la mejor derivada mejora al agente en **10/10** (media +0,086). Por régimen (Holm):
M10 y AutoML significativos en alcista Y bajista; M8 sig en bajista, al borde en alcista (0,099).

**C3. El aprendiz redescubre STRATA y la BATE modestamente (TOST).** SHAP: cuota de las features de STRATA en el
aprendiz **>0,5 en 10/10** (media **0,66**; SPY: 0,71 en el ensemble M10, 0,565/0,564 en el leader AutoML).
**TOST (Schuirmann 1987):** no hay equivalencia, el aprendiz **supera a M8 en accuracy** (pooled M10 +0,021
IC90[+0,001,+0,039]; AutoML +0,034 IC90[+0,010,+0,056]); en **Sharpe es no concluyente** → indistinguibles en
riesgo. Lo explica el mecanismo: el aprendiz modela interacciones no lineales que la regla fija no alcanza.

**C4. Dos capas complementarias.** **M8 = capa de riesgo** (pooled ΔSharpe sig; 100 % del P&L de rescate viene
del canal régimen/RAM; nunca lidera en accuracy, 0/10). **Aprendiz = capa de accuracy** (McNemar sig). **Gate
RAM:** cuando RAM dispara, seguir el régimen bate a seguir al agente en **6/10**; la intervención crece con la
discrepancia agente↔régimen (**Pearson r=0,93, p<0,001**). **Anatomía SPY:** de 121 intervenciones, M8 acierta
58,7 % vs 41,3 % del agente (71/50; P&L +0,312).

**C5. PATRÓN naturaleza→canal: la ley del leverage.** El rescate del aprendiz en accuracy **escala con el
leverage effect**: **Pearson r=−0,56 (p=0,093), Spearman ρ=−0,59 (p=0,074) sobre los 10** → tendencia
significativa al **α=0,10** del proyecto, **visible en el panel** (9/10 la siguen; ROKU excepción). Con n=10 **no**
se afirma p<0,05 ni robustez leave-one-out: la correlación es tan fuerte como sobre un universo mayor, pero el
test es menos potente. Medias: leverage fuerte rescata **0,097**, leverage débil **0,059**. La naturaleza **no**
predice el valor de la **regla** (crisis_mean/leverage/sesgo, todas p>0,14) → `crisis_mean` es **descriptivo, no
ley**. **Clustering (10, consenso unánime KMeans/Ward/GMM/Spectral, Rand=1,0, silhouette 0,55):** C0 índices
leverage fuerte (SPY/QQQ/XLF/DIA/XLK/XLE)→AutoML; C1 leverage invertido (SMCI/UNG)→M10; C2 volátiles
(ROKU/MARA)→AutoML. **PC1 ≈ leverage (r=0,84)** → cadena cerrada naturaleza→eje del clustering→rescate.

**C6. Complementariedad por régimen — SIGNIFICATIVA (DiD).** M10 rescata más en alcista (ΔSharpe +1,37 vs +0,72),
AutoML más en bajista (+1,52 vs +0,81). **Test diferencia-en-diferencias pre-registrado: pooled +1,37,
IC95[0,20, 2,60], p=0,008** (excluye 0). Es fenómeno de **panel**, no aparece en SPY solo. Implicación: AutoML
protege mejor en el régimen peligroso (bajista). Línea futura: ensemble enrutado por régimen.

**C7. El rescate no es de un solo régimen.** McNemar pooled (sup vs M5) sig en alcista Y bajista. A nivel SPY-solo
el rescate de Sharpe se concentra en alcista y la regla M8 se invierte en bajista (n=50, ΔSharpe −1,49,
falsación pre-registrada); la agregación del panel lo resuelve.

**C8. STRATA bate a la trivial donde la naturaleza lo permite (en punto).** En **SPY**, AutoML supera a ZeroR/B&H
en accuracy (0,574 vs 0,566), Sharpe (+2,68 vs +2,21), máxima caída (−5,5 % vs −9,8 %) y equity (1,38× vs 1,30×)
— la gráfica de equity lo hace visible. En **SMCI** (M10 0,552 > 0,516/0,484), **MARA** (AutoML 0,544 >
0,532/0,468) y **UNG** (M10 0,518 > 0,449/0,486) una derivada de STRATA bate a las dos triviales en accuracy.
Honesto: estas victorias son **en punto**; la significancia pooled vs trivial es nominal (McNemar AutoML vs
ZeroR SPY p=0,90, n≈250). El valor está, parte, en estas victorias por activo; el límite es que no se prueban
con un test a esta muestra.

**C9. La ablación depende del modelo; el ganador usa STRATA.** AutoML alcanza su máximo con las 22 features
(0,574) y **degrada al quitar PSA+GSO** (0,550) → usa los detectores. M10 con params fijos se sobreajusta con 22.
PSA/GSO apenas disparan como reglas (RAM domina) pero sus scores continuos informan al aprendiz.

---

## V. Resultados headline (con su test)

- **SPY, 6 estrategias (desplegable, n=251):** M5 0,366 (Sharpe −3,07, eq 0,70) · M8 0,442 (−0,46) · M10 0,494
  (−0,60) · **AutoML 0,574 (+2,68, maxDD −5,5 %, eq 1,38×)** · ZeroR/B&H 0,566 (+2,21, −9,8 %, 1,30×). McNemar
  AutoML vs ZeroR **p=0,90** (nominal); vs M5 = **0,0002/0,0074/0,051** (AutoML/M10/M8).
- **Pooled-10 riesgo:** M8 +0,64 [0,10,1,29] · M10 +0,93 [0,19,1,65] · AutoML +0,97 [0,27,1,70]. Bonferroni:
  M10/AutoML sí, M8 no. **DSR AutoML-SPY 0,92.**
- **TOST aprendiz vs regla:** accuracy superior (M10 +0,021, AutoML +0,034); Sharpe no concluyente.
- **DiD complementariedad:** +1,37 [0,20,2,60], p=0,008 (pooled; no en SPY solo).
- **SHAP cuota:** media 0,66, >0,5 en 10/10. **Clustering:** Rand 1,0, silhouette 0,55. **Ley leverage:** r=−0,56,
  p=0,093 (α=0,10). **Gate RAM:** seguir régimen bate agente 6/10; intervención↔discrepancia r=0,93.
- **Costes:** M8 rota 61 (menos que el agente, 105); Sharpe agregado del aprendiz positivo hasta ~10 pb.

---

## VI. Figuras clave (una figura = evidencia de un objetivo; fuente trazable)

Curar **6–8 para el capítulo** (una galería de 30 diluye). Tiers:
- **TIER 1 (carga):** (1) **equity de SPY** con las 6 estrategias, AutoML batiendo a la trivial en Sharpe/maxDD
  (O1+O6, la gráfica que importa); (2) tabla **6 estrategias SPY + McNemar** (O2 accuracy); (3) **forest plot
  pooled-10 ΔSharpe + Bonferroni** (O2 riesgo); (4) **tabla+scatter leverage↔rescate** (O5 ley); (5) **2×2 del
  TOST** (O4 redescubre/bate).
- **TIER 2 (refuerzo):** clustering PCA (Rand=1,0, PC1≈leverage); complementariedad DiD; gate RAM; heatmap
  accuracy activo×estrategia (centrado en 0,5: dónde STRATA bate a la trivial).
- **TIER 3 (apoyo del notebook, no del cuerpo):** atribución por detector, SHAP cuota/dependency, régimen×
  dirección, rodante/val-test, variantes de intervención, sensibilidad a umbrales, descriptivo por variable.

**Arco del capítulo:** funciona (rescate sig en accuracy + riesgo; bate a la trivial donde puede) → por qué (dos
capas + naturaleza→leverage + complementariedad) → límites (nominal vs trivial en significancia, no alfa).

---

## VII. Líneas rojas (lo que NUNCA se afirma ni se escribe)

- No "bate al mercado"; no "genera alfa"; no "bate a ZeroR/B&H **con significancia** en accuracy".
- No presentar la accuracy nominal como significativa; no afirmar la ley a p<0,05 (sobre 10); no afirmar
  robustez leave-one-out (sobre 10); no mezclar las dos ventanas (OOS completo n≈401 vs desplegable n≈250).
- **No escribir que el trabajo se hizo sobre más de 10 activos ni que "elegimos 10".** El universo es 10.

---

## VIII. Caveats declarados

1. **Accuracy nominal** vs lo trivial (ventana corta n≈250 → significancia plena, línea futura).
2. **Pooled:** apila días-activo como independientes; la correlación cruzada (SPY-QQQ >0,9) hace el $n$ efectivo
   menor que el nominal y el IC algo optimista; aun así +0,64 excluye 0 con holgura. Bloque medio $\sqrt N$.
3. **Ley del leverage:** sobre 10, tendencia sig al α=0,10 (p=0,093); no p<0,05 ni LOO. Se cumple en el panel.
4. **Clustering n=10:** exploratorio; qué modelo concreto gana por activo no es predecible por el cluster.
5. **Complementariedad por régimen:** sig en pooled, no en SPY solo (fenómeno cross-asset).
6. **Desplegabilidad:** M8 opera desde el día 1 (regla ex-ante, rota poco); M10/AutoML necesitan ~150 días de
   arranque y reentreno, y AutoML cambia de leader en cada reentreno (más difícil de validar en producción).

---

## IX. Trazabilidad (resultado → JSON)

| Resultado | JSON |
|---|---|
| Tabla 6 estrategias SPY + McNemar + panel-10 accuracy | `automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_*.json` |
| Pooled-10 riesgo (net-causal) + ablación + SHAP cuota | `decision_automl_prep.json` |
| Series netas de AutoML | `automl_net_returns.json` |
| Bonferroni confirmatorio + DSR + McNemar bull/bear | `bullbear_confirmatory.json` |
| TOST aprendiz vs regla | `equivalence_tost.json` |
| DiD complementariedad | `regime_did_learners.json` |
| Ley leverage (r, p, Spearman) + PC1↔leverage | `leverage_law_panel.json`, `mechanism_panel.json` |
| Clustering 10 (Rand, silhouette, perfiles) | `cluster_panel10.json` |
| Atribución/anatomía detector SPY + gate RAM | `spy_intervention_anatomy.json`, `spy_panel_gate_descriptive.json` |
| Variantes de intervención + sensibilidad umbrales | `spy_intervention_variants.json` |
| Ablación de detectores (M10 y AutoML) | `detector_ablation_panel.json`, `automl_ablation_detectors.json` |
| Umbrales calibrados (RAM/PSA/GSO) | `cache/models/strata_thresholds.json` |
| K=3 held-out | `k_selection.json` |
| Costes/rotación · rodante/val-test/calibración | `net_of_cost_panel.json`, `panel_robustness.json`, `calib_window_panel.json` |

---

## X. Mapa de distribución a los docs raíz

- `DECISIONES_ESENCIALES.md`: universo = **10, sin apéndice** (los demás fuera del TFG); jerarquía de valor (§I);
  pooled = **pooled-10**; SMCI es uno de los 10, no caso central.
- `RESULTADOS_OBJETIVO.md`: headline canónico = §V (pooled-10 +0,64; TOST; DiD; ley leverage sobre 10, p=0,093;
  victorias en punto vs trivial en SPY/SMCI/MARA/UNG).
- `CONOCIMIENTO_ACUMULADO.md`: banner caso central **SPY + panel de 10 + patrones**; el valor es rescate + dos
  capas + naturaleza→leverage + batir-a-la-trivial-donde-se-puede; no genera alfa.
- `LECCIONES_APRENDIDAS.md`: notebook canónico = `STRATA_marco_practico.ipynb`.

---

## XI. Mandato de calidad (Q1–Q6) — condición permanente de toda cifra, gráfica y celda

Estos no se auditan como objetivos cumplidos/no cumplidos: son **condición permanente** de toda la redacción.
Aquí no hay flexibilidad. Es lo que convierte el capítulo en un research note de mesa, no en un resumen.

- **Q1 — Trazabilidad.** Cada cifra con su **test, su IC y su cita** (a JSON o a literatura). Sin trazabilidad,
  la cifra no entra. Las cifras vienen de los JSON canónicos (§IX), nunca a mano, nunca de notebooks obsoletos.
- **Q2 — Causalidad estricta.** `signal_lag=1`; embargo=1; sin KFold (única excepción documentada: M3, para
  denunciar el sesgo). Las dos ventanas (OOS completo n≈401, desplegable n≈250) nunca se mezclan.
- **Q3 — Honestidad.** Lo nominal se etiqueta nominal; las limitaciones se declaran; no se ocultan resultados
  malos. Si un detector no aporta, se dice (PSA/GSO inertes). Si algo no llega a significancia, se reporta y se
  argumenta, con la lectura que lo pone en contexto.
- **Q4 — Reproducibilidad.** Semillas fijas; AutoML por `max_models` (determinista), nunca por tiempo; auto-test
  verde que cruza cada cifra de cabecera con su JSON.
- **Q5 — Estética profesional.** Paleta única coherente; escala común en gráficas comparables; legibilidad para
  un lector experto.
- **Q6 — Presentación sin humo.** Cada cifra **contextualizada con su significado** (económico o estadístico), no
  flotando suelta. Cada gráfica **autocontenida**: título, ejes con unidades, leyenda, nota al pie con su fuente
  JSON. Cada sección **cierra con una lectura razonada** que conecta los números con la jerarquía de valor (§I).
  Si una cifra parece favorable pero no se entiende su mecanismo, no entra; si una gráfica decora pero no
  informa, no entra.

**Principio rector de la redacción:** *cada afirmación lleva su evidencia (tabla/figura), su test, su
interpretación y su límite. Nada queda en una frase suelta.* El lector puede auditar cada paso sin fiarse de la
palabra de la autora. **Análisis extenso, no resumen.**

---

## XII. Estructura del cap. 4 con figuras y tablas imprescindibles por sección (canónico de 10)

Estructura de 4 secciones (SPY → panel → mecanismo+clustering → límites). El arco: **funciona** (rescate sig en
accuracy + riesgo; bate a la trivial donde puede) → **por qué** (dos capas + naturaleza→leverage +
complementariedad) → **límites** (nominal vs trivial en significancia, no alfa). La estructura final la fija el
`coordinador-redaccion` y la valida Raquel; aquí queda el contenido obligatorio.

**§1 — Caso de estudio SPY (el mecanismo, transparente).**
- *Tablas:* parámetros de calibración (con criterio + cita); 6 estrategias (acc/Sharpe/maxDD/Calmar/equity);
  McNemar pareado; variantes de intervención (override/abstención/reduce).
- *Figuras imprescindibles:* **equity de las 6 estrategias con AutoML batiendo a la trivial** (la que importa);
  régimen HMM sobre el precio; tasa de intervención y de éxito por detector + atribución de P&L; distribución de
  scores con umbrales; matriz de confusión; ablación STRATA; sensibilidad a umbrales (meseta, anti-tuning).

**§2 — Panel de 10 (universalidad y riesgo).**
- *Tablas:* accuracy por activo × estrategia (10×6); pooled-10 ΔSharpe + Bonferroni; ablación + cuota SHAP por
  activo; TOST (aprendiz vs regla); robustez por régimen (bull/bear, Holm); val/test y ventana de calibración.
- *Figuras imprescindibles:* **heatmap accuracy activo×estrategia centrado en 0,5** (dónde se bate a la trivial);
  forest plot pooled-10 ΔSharpe; equity por activo (ganadora destacada); 2×2 del TOST; complementariedad DiD;
  gate RAM; activación de detectores en el panel.

**§3 — Mecanismo y clustering (por qué).**
- *Tablas:* leverage↔rescate por activo (10); silueta/Rand por método; perfiles de cluster (naturaleza media +
  mejor canal).
- *Figuras imprescindibles:* scatter leverage↔rescate; PCA 2D de la naturaleza (PC1≈leverage); cuatro vistas por
  grupo; dos casos trabajados (XLE régimen / MARA leverage invertido).

**§4 — Límites y futuras líneas.**
- *Contenido:* qué no sobrevive a un test (nominal vs trivial; ley α=0,10; M8 no pasa Bonferroni; SPY-solo
  bajista); desplegabilidad (M8 día 1; M10/AutoML burn-in + leader cambiante); líneas futuras (muestra mayor,
  ensemble enrutado por régimen, otros agentes, despliegue).

---

## XIII. Reglas de honestidad operativas y tono

- **Tono:** estilo de **cliente experto** (el lector sabe del tema, no se le engaña); claro, riguroso, sin
  filler; research note de mesa cuant, no colección de outputs. Voz de Raquel (frase corta, dos puntos, primera
  persona del plural, objeto concreto; sin "no es X sino Y", sin meta-comentarios, sin repetir la causalidad).
- **Intros que captan la atención** (prioridad alta): cada capítulo abre directo al problema y a la jerarquía de
  valor, sin andamiaje.
- **Frase canónica del universo (la que se usa):** *"el estudio se realiza sobre un panel de diez activos
  (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG), elegidos ex ante por su naturaleza de mercado"*. **Nunca**
  "de entre 15", "elegimos 10", ni "apéndice".
- Lo nominal se dice nominal; lo que pasa un test se dice con su test; no se nombra "p-hacking".

---

## XIV. Prohibiciones (qué NO hacer)

1. No mezclar configuraciones en una misma tabla sin etiquetar las diferencias.
2. No coger cifras de notebooks obsoletos (`STRATA_SMCI`, `decision_automl`); solo de los JSON canónicos (§IX).
3. No nombrar "p-hacking" ni dejar indicios de él.
4. No inventar datos ni rellenar huecos a ojo.
5. No ocultar resultados malos importantes (M8 no pasa Bonferroni; SPY-solo bajista; nominal vs trivial).
6. No usar KFold (salvo M3, demostración del sesgo).
7. No presentar lo nominal como significativo.
8. No usar `signal_lag` distinto de 1.
9. No justificar una decisión con "para que los resultados saliesen mejor": mecanismo, literatura o criterio ex-ante.
10. Ninguna afirmación sin atar matemáticamente o por literatura.
11. No paleta incoherente; no gráficas que decoren sin informar.
12. **No escribir que el trabajo se hizo sobre más de 10 activos ni que "elegimos 10".**

---

## XV. Contexto crudo del notebook (detalle conceptual por sección, para entrar al detalle)

Resumen denso del notebook canónico (`STRATA_marco_practico.ipynb`, §0–§9), para saber qué hay y qué importa.

- **§0 Portada/tesis/objetivos/notación.** Se presenta como "venta ante un comité quant": se vende con SPY, se
  prueba con lo que sobrevive a un test, cada decisión con su material. Dos supervisores complementarios (M8=riesgo,
  aprendiz=accuracy) como eje. Honestidad cableada (no alfa).
- **§1 Datos, panel de 10, protocolo.** Universo de 10 ex-ante por naturaleza; calibración 2000–2024-09; OOS
  2024-10; dos ventanas; barrera temporal sin fuga (asserts anti-fuga). Tabla del panel con clase/canal por activo.
- **§2 Mecánica ex-ante.** Los tres detectores (RAM/PSA/GSO) con su definición y umbral; leverage honesto
  (contemporáneo, no predictivo: frac sube día sig ≈0,5 incluso en Crisis); intervención + atribución de P&L
  (RAM 100%, PSA/GSO inertes); anatomía (121 interv., 71/50, acc 0,587 vs 0,413); ablación M10 vs AutoML (AutoML
  usa los detectores, degrada al ablar PSA+GSO); matriz régimen×dirección.
- **§3 Caso SPY.** Tabla de 6 estrategias; matrices de confusión; **equity (AutoML bate a la trivial)**; rescate
  de riesgo SPY (IC cruza 0 a nivel SPY → la significancia vive en el pooled); SHAP + permutation; override vs
  abstención vs reduce; sensibilidad a umbrales (meseta, robustez no tuning); M10 vs M5 por régimen; SHAP
  dependency; descriptivo de cada variable vs signo del retorno.
- **§4 Panel de 10 (universalidad y riesgo).** Activación de detectores; gate RAM (seguir régimen bate al agente
  6/10; intervención↔discrepancia r=0,93); naturaleza de los 10 (4 gráficas); mejor-STRATA vs agente por activo;
  ablación + cuota SHAP por activo; **TOST** (aprendiz bate la regla en accuracy, empata en riesgo); **heatmap
  accuracy** (dónde se bate a la trivial); matrices de confusión por activo; **pooled-10 de riesgo (el resultado
  duro)** + Bonferroni; cuota SHAP rodante (estable).
- **§5 Mecanismo: dos supervisores + la única ley.** Dos tests, dos funciones (O4: M8 lidera riesgo nunca
  accuracy; aprendiz al revés); **la ley del leverage** (r=−0,56, p=0,093 sobre 10; tabla por activo, 9/10);
  correlaciones que NO predicen la regla (honestidad); dos casos trabajados (XLE/MARA); timeline diario M8 vs M10;
  rescate estratificado por tipo de activo (estratos pre-registrados).
- **§6 Clustering: naturaleza→resultado.** Calidad (silueta/BIC/Rand=1,0 unánime); PCA 2D (PC1≈leverage r=0,84);
  perfil económico de cada cluster; comportamiento por grupo (4 vistas). Cadena cerrada naturaleza→eje→rescate.
- **§7 Robustez y honestidad.** Equity por activo (10); robustez de M8−M5 por sub-ventana y partición; STRATA
  sobre momentum; **límite (la naturaleza débil-leverage estrecha el rescate)**; confirmatorio Bonferroni+DSR;
  rescate por régimen + **DiD complementariedad** (+1,37, p=0,008); robustez a la ventana de calibración.
- **§8–§9 Conclusiones + auto-test.** Las C1–C9 (§IV) con su test; auto-test que cruza cada cifra de cabecera con
  su JSON (patrón de mesa cuant).
