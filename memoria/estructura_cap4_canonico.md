# Estructura validada del Capítulo 4 (Marco práctico — enfoque canónico de 10 activos)

> **Fuente de verdad de contenido:** [`MARCO_PRACTICO_CONTEXTO.md`](../MARCO_PRACTICO_CONTEXTO.md).
> **Estado:** VALIDADA por Raquel (2026-06-24). Este es el outline que `redactor-tesis` sigue sección a
> sección. Sustituye a [`estructura_cap4.md`](estructura_cap4.md) (era SMCI, OBSOLETO).

## Reglas duras (heredadas de CONTEXTO)
- Universo = **panel de 10 activos** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG), ex ante por
  naturaleza. **SIN apéndice; NUNCA "10 de N"**. SMCI es uno de los 10.
- Arco: **funciona → por qué → límites**. Liderar con **rescate + patrón**, nunca con la accuracy frente a la
  trivial.
- Jerarquía de valor (titular en positivo): rescata al agente (sig) > bate modestamente a la regla (TOST) >
  empata en riesgo > bate a la trivial en punto donde la naturaleza lo permite. **El alfa NO va en la
  narrativa**: límite de alcance (caveat §4.4) + línea futura.
- **Sin nada de costes** (decisión Raquel 2026-06-24): ni break-even, ni rotación, ni Sharpe-neto-de-coste. Se
  eliminan T4.15 y F4.16 del outline original.
- **Nomenclatura de la intervención** (decisión Raquel 2026-06-24): al **comparar variantes** (vs abstención,
  vs reduce) se llama **override**; el **mecanismo en sí**, cuando va solo, se llama **intervención**. Nunca
  "intervención fuerte" ni "override-C".
- Mandato Q1–Q6: cada cifra con test+IC+cita y su fuente JSON; cada gráfica autocontenida; cada sección cierra
  con lectura razonada. Análisis extenso, no resumen.

## Títulos (validados por Raquel)
- **§4.1 — SPY: cómo opera STRATA, paso a paso**
- **§4.2 — Panel multi-activo: rescate pareado, agregado y por régimen**
- **§4.3 — Patrones del rescate. Clustering**
- **§4.4 — Límites y futuras líneas de investigación**

---

## INTRO DEL CAPÍTULO (~½ pág, sin numerar)
Abre directo al problema (agente LLM pierde y acierta <50%) y a la jerarquía de valor. Anuncia el arco de 4
secciones y el contrato de lectura (cada cifra con test, IC, cita, fuente JSON). Titular en positivo. Sin
tabla/figura. Sin citas nuevas.

---

## §4.1 — SPY: cómo opera STRATA, paso a paso (7–9 pág)
Sirve a O1, O2, O3, O6, C1, C2, C4, C8, C9. Mostrar la mecánica entera sobre un activo auditable; ya etiquetar
qué sobrevive a un test.

### §4.1.1 Protocolo y calibración ex-ante
- **Contamos:** dos ventanas (OOS completo n≈401 vs desplegable n=251, burn-in 150), `signal_lag=1`, embargo=1,
  calibración 2000→2024-09 una sola vez, OOS posterior al corte del LLM. → **O7; blinda Q2.**
- **T4.1** Parámetros de calibración: HMM K=3 (held-out −1,30 > −1,69), GARCH(1,1)-t, BOCPD, umbrales RAM
  τ=0,50 / PSA P95=0,023 / GSO P95=2,371. Fuente: `k_selection.json`, `cache/models/strata_thresholds.json`.
- **F4.1** Regímenes HMM sobre el precio SPY (Calma/Estrés/Crisis), proxy direccional vía leverage. Fuente:
  notebook §3 / `regime_direction_table.json`.
- **Citas:** Rabiner (HMM, selección K), Bollerslev (GARCH-t), Adams & MacKay (BOCPD), Tashman / López de Prado
  (walk-forward, embargo), Black 1976 (proxy direccional en SPY).

### §4.1.2 Mecánica de los tres detectores y la intervención
- **Contamos:** definición operativa de RAM/PSA/GSO con su umbral; la **intervención** reorienta al signo del
  régimen; prior RAM data-driven por activo. Leverage honesto: contemporáneo, no predictivo. → **O3, O7.**
- **T4.2** Tasa de intervención/acierto por detector + atribución de P&L (RAM = 100% del rescate; PSA/GSO
  inertes como reglas). Fuente: `spy_intervention_anatomy.json`.
- **F4.2** Distribución de scores RAM/PSA/GSO con sus umbrales (cuantiles de calibración). Fuente:
  `spy_intervention_variants.json` / notebook §2.
- **Citas:** Adams & MacKay (score PSA), Bollerslev (score GSO), Black/Christie (lectura del régimen).

### §4.1.3 Anatomía de la intervención y elección de modo
- **Contamos:** anatomía SPY (121 intervenciones, M8 0,587 vs 0,413 agente, 71/50, P&L +0,312); **override**
  gana a abstención y a reduce (0,478 vs 0,401 vs 0,397). La intervención crece con la discrepancia
  agente↔régimen. → **O3, C4.**
- **T4.3** Variantes (**override** / abstención / reduce), acc por variante. Fuente:
  `spy_intervention_variants.json`.
- **F4.3** Matriz de confusión SPY (agente vs intervención), el 71/50 visible. Fuente: notebook §3.
- **Citas:** ninguna nueva obligatoria.

### §4.1.4 Las seis estrategias sobre SPY: accuracy, riesgo y la trivial
- **Contamos:** 6 estrategias (acc/Sharpe/maxDD/Calmar/equity) + McNemar pareado vs M5; el rescate de riesgo a
  nivel SPY no es sig (IC cruza 0 → se anuncia el pooled de §4.2); AutoML bate a la trivial en este punto
  (matiz: SPY ≈ beta de mercado, OOS alcista); McNemar AutoML vs ZeroR p=0,90 (nominal, se dice ya). → **O1
  (sign test vs 0,5 p<0,001), O2, O6/C8, C2.**
- **T4.4 (TIER 1)** Seis estrategias SPY + McNemar: M5 0,366 (Sh −3,07, eq 0,70) · M8 0,442 (−0,46) · M10 0,494
  (−0,60) · **AutoML 0,574 (+2,68, maxDD −5,5%, eq 1,38×)** · ZeroR/B&H 0,566 (+2,21, −9,8%, 1,30×). McNemar
  vs M5: AutoML 0,0002 / M10 0,0074 / M8 0,051; vs ZeroR 0,90. Fuente:
  `automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_*.json`.
- **F4.4 (TIER 1, LA QUE IMPORTA)** Equity de las 6 estrategias SPY: AutoML batiendo a la trivial en
  Sharpe/maxDD/equity. Autocontenida (€/×, leyenda, nota JSON). Fuente: `automl_net_returns.json`.
- **Citas:** McNemar, López de Prado (Deflated Sharpe), Sharpe (ratio, ref. Cap. 3).

### §4.1.5 Robustez del caso SPY: ablación y anti-tuning
- **Contamos:** ablación (AutoML 0,574→0,550 al quitar PSA+GSO → usa los detectores); sensibilidad a umbrales
  (meseta → robustez, no tuning); SHAP cuota SPY (0,71 ensemble M10; 0,565/0,564 leader AutoML). → **C9, O4
  (anticipo), Q3.**
- **T4.5** Ablación de detectores SPY (ALL22 vs sin-PSA+GSO). Fuente: `automl_ablation_detectors.json`,
  `detector_ablation_panel.json`.
- **F4.5** Sensibilidad a umbrales (meseta). Fuente: `psa_gso_threshold_sensitivity.json`.
- **Citas:** Lundberg & Lee (SHAP).

> **Cierre §4.1:** sobre SPY el mecanismo es transparente y el agente, solo, pierde; la intervención por régimen
> lo corrige y la derivada aprendida supera a la trivial en este punto. Pero a nivel de un activo la corrección
> de riesgo no alcanza significancia y la victoria es nominal: la prueba dura exige agregar el panel (§4.2).

---

## §4.2 — Panel multi-activo: rescate pareado, agregado y por régimen (8–10 pág)
Sirve a O2, O4, O6, C1, C2, C3, C6, C7, C8. El resultado duro (pooled-10 de riesgo) + rescate universal.

### §4.2.1 El agente a través del panel y la activación de detectores
- **Contamos:** transversalidad de O1 (agente <0,5 en el panel); activación de detectores por activo; gate RAM
  (seguir el régimen bate al agente 6/10; intervención↔discrepancia Pearson r=0,93, p<0,001). → **O1
  transversal, C4.**
- **T4.6** Activación de detectores y gate RAM por activo. Fuente: `spy_panel_gate_descriptive.json`,
  `mechanism_panel.json`.
- **F4.6 (TIER 2)** Gate RAM: discrepancia agente↔régimen vs tasa de intervención (r=0,93). Fuente:
  `spy_panel_gate_descriptive.json`.
- **Citas:** ninguna nueva obligatoria.

### §4.2.2 Accuracy por activo × estrategia: dónde se bate a la trivial
- **Contamos:** la mejor derivada mejora al agente en 10/10 (media +0,086); victorias en punto vs las dos
  triviales (SPY AutoML 0,574>0,566; SMCI M10 0,552>0,516/0,484; MARA AutoML 0,544>0,532/0,468; UNG M10
  0,518>0,449/0,486); honestidad: nominal (n≈250, a posteriori), ganadora cambia por activo, ninguna bate a
  ZeroR con significancia. → **C2, O6/C8, Q3.**
- **T4.7** Accuracy por activo × estrategia (10×6). Fuente: `automl_runs/...emb1_*.json`,
  `decision_automl_prep.json`.
- **F4.7 (TIER 2, IMPRESCINDIBLE)** Heatmap accuracy activo×estrategia centrado en 0,5 (color = distancia a la
  trivial). Autocontenida con nota JSON. Fuente: `automl_runs/...emb1_*.json`.
- **Citas:** McNemar (vs ZeroR), sign test vs 0,5 (ref. Cap. 3).

### §4.2.3 Pooled-10: el rescate en riesgo (el resultado duro)
- **Contamos:** pooled-10 ΔSharpe vs M5 (M8 +0,64 [0,10,1,29]; M10 +0,93 [0,19,1,65]; AutoML +0,97 [0,27,1,70],
  los tres excluyen 0); Bonferroni m=3 → M10/AutoML pasan, **M8 no** (cota inferior −0,047, falsación honesta
  de la regla en riesgo); DSR AutoML-SPY 0,92; caveat pooled (corr cruzada SPY-QQQ>0,9, n efectivo menor,
  bloque medio √N). → **O2 (el central), C1, C8, Q1/Q3.**
- **T4.8** Pooled-10 ΔSharpe + Bonferroni + DSR. Fuente: `decision_automl_prep.json`,
  `bullbear_confirmatory.json`.
- **F4.8 (TIER 1)** Forest plot pooled-10 ΔSharpe con cota de Bonferroni (M8 tocando el cero). Fuente:
  `decision_automl_prep.json`, `bullbear_confirmatory.json`.
- **F4.9** Equity por activo (ganadora destacada): las 4 victorias en Sharpe (SPY/SMCI/MARA/UNG), con el matiz
  alfa-vs-beta como **lectura razonada** (no afirmación con test). Fuente: `automl_net_returns.json`.
- **Citas:** Politis & Romano (bootstrap estacionario / bloque √N), López de Prado (Deflated Sharpe, Bonferroni).

### §4.2.4 El aprendiz redescubre STRATA y bate a la regla (SHAP + TOST)
- **Contamos:** cuota SHAP de features STRATA >0,5 en 10/10 (media 0,66); TOST: no equivalencia, el aprendiz
  **supera** a M8 en accuracy (pooled M10 +0,021 [+0,001,+0,039]; AutoML +0,034 [+0,010,+0,056]) y en Sharpe es
  no concluyente. Mecanismo: el aprendiz modela interacciones no lineales que la regla fija no alcanza. → **O4
  (central), C3.**
- **T4.9** TOST aprendiz vs regla (accuracy y Sharpe) + cuota SHAP por activo. Fuente: `equivalence_tost.json`,
  `decision_automl_prep.json`.
- **F4.10 (TIER 1)** Diagrama 2×2 del TOST (ejes accuracy/Sharpe, regiones superioridad/equivalencia). Fuente:
  `equivalence_tost.json`.
- **Citas:** Schuirmann (TOST), Lundberg & Lee (SHAP).

### §4.2.5 Robustez por régimen: bull/bear y complementariedad
- **Contamos:** McNemar pooled (sup vs M5) sig en alcista Y bajista (Holm: M10/AutoML sig en ambos; M8 sig en
  bajista, borde 0,099 en alcista) → rescate no de un solo régimen (C7); DiD complementariedad: M10 rescata más
  en alcista, AutoML en bajista; DiD pre-registrado +1,37 [0,20,2,60], p=0,008 (pooled, NO en SPY solo); a
  nivel SPY-solo la regla M8 se invierte en bajista (n=50, ΔSharpe −1,49, falsación pre-registrada resuelta por
  la agregación). → **C6, C7, O5 (anticipo), Q3.**
- **T4.10** Rescate por régimen (bull/bear, Holm) + DiD. Fuente: `bullbear_confirmatory.json`,
  `regime_did_learners.json`.
- **F4.11 (TIER 2)** Complementariedad DiD: ΔSharpe M10 vs AutoML por régimen. Fuente: `regime_did_learners.json`.
- **Citas:** Holm (corrección múltiple), DiD (ref. Cap. 3).

> **Cierre §4.2:** agregado el panel, el rescate deja de ser anécdota: en riesgo es significativo (pooled-10, IC
> excluye 0, dos de tres modelos sobreviven a Bonferroni) y en accuracy el aprendiz rescata al agente y bate
> modestamente a la regla, apoyándose en STRATA. La regla protege el riesgo, el aprendiz afina la dirección, y
> se complementan por régimen. Falta el porqué del reparto por activo: §4.3.

---

## §4.3 — Patrones del rescate. Clustering (6–8 pág)
Sirve a O3, O5, C4, C5. Cierra la cadena naturaleza→eje→rescate. **Enmarcar el clustering como exploratorio
(n=10).**

### §4.3.1 Dos supervisores, dos funciones
- **Contamos:** M8 = capa de riesgo (pooled ΔSharpe sig; 100% del P&L de rescate del canal RAM/régimen; nunca
  lidera en accuracy, 0/10). Aprendiz = capa de accuracy (McNemar sig, nunca lidera en riesgo en exclusiva).
  Timeline diario M8 vs M10. → **O3, C4.**
- **T4.11** Atribución por capa (riesgo vs accuracy, quién lidera). Fuente: `decision_automl_prep.json`,
  `spy_intervention_anatomy.json`.
- **F4.12** Atribución de P&L por detector / timeline M8 vs M10. Fuente: notebook §5.
- **Citas:** ninguna nueva obligatoria.

### §4.3.2 La ley del leverage (la única ley)
- **Contamos:** el rescate del aprendiz en accuracy escala con el leverage effect: Pearson r=−0,56 (p=0,093),
  Spearman ρ=−0,59 (p=0,074) sobre los 10 → tendencia sig al **α=0,10** del proyecto, 9/10 la siguen (ROKU
  excepción). Medias: leverage fuerte 0,097, débil 0,059. Honestidad: con n=10 **no** se afirma p<0,05 ni LOO;
  la naturaleza NO predice el valor de la regla (crisis_mean/leverage/sesgo p>0,14). `leverage_corr` = corr
  (Black 1976). → **O5 (central), C5, Q3.**
- **T4.12 (TIER 1)** Leverage↔rescate por activo (10), con la marca 9/10. Fuente: `leverage_law_panel.json`.
- **F4.13 (TIER 1)** Scatter leverage↔rescate (recta, r=−0,56, p=0,093, ROKU señalado). Autocontenida. Fuente:
  `leverage_law_panel.json`.
- **Citas:** Black (1976), Christie (1982) (leverage effect = `leverage_corr`); α=0,10 pre-registrado.

### §4.3.3 Clustering: la naturaleza dibuja los grupos
- **Contamos:** consenso unánime KMeans/Ward/GMM/Spectral (Rand=1,0, silhouette 0,55); C0 índices leverage
  fuerte (SPY/QQQ/XLF/DIA/XLK/XLE)→AutoML; C1 leverage invertido (SMCI/UNG)→M10; C2 volátiles
  (ROKU/MARA)→AutoML; PC1≈leverage (r=0,84) → cadena naturaleza→eje→rescate cerrada. **Caveat: exploratorio
  (n=10)**, qué modelo concreto gana por activo no es predecible por el cluster. → **O5, C5.**
- **T4.13** Calidad del clustering (silhouette/Rand por método) + perfiles de cluster. Fuente: `cluster_panel10.json`.
- **F4.14 (TIER 2)** PCA 2D de la naturaleza: PC1≈leverage (r=0,84), 3 clusters coloreados, mejor canal por
  grupo. Fuente: `cluster_panel10.json`, `mechanism_panel.json`.
- **F4.15** Dos casos trabajados (XLE régimen / MARA leverage invertido). Fuente: notebook §5–§6.
- **Citas:** métodos de clustering (KMeans, Ward, GMM, spectral), índice de Rand; Black/Christie (PC1≈leverage).

> **Cierre §4.3:** el reparto de funciones no es casual: la regla protege el riesgo vía régimen y el aprendiz
> gana accuracy modelando interacciones que la regla no alcanza; qué canal rescata cada activo lo dicta su
> naturaleza. El leverage ordena el rescate (tendencia sig al α=0,10), el clustering recupera los mismos grupos
> sin etiquetas (Rand=1,0) y su primer eje es el leverage. Cadena cerrada, con la cautela de que sobre diez
> activos es un patrón fuerte, no un teorema.

---

## §4.4 — Límites y futuras líneas de investigación (3–4 pág)
Sirve a O7, C8, Q3. Recoge los caveats §VIII. Aquí —y solo aquí— vive el ángulo del alfa.

### §4.4.1 Qué no sobrevive a un test
- **Contamos:** (a) accuracy nominal vs la trivial (n≈250, a posteriori; ninguna bate a ZeroR con
  significancia, AutoML vs ZeroR SPY p=0,90); (b) la ley del leverage es tendencia al α=0,10 (p=0,093), no
  p<0,05 ni LOO; (c) M8 no pasa Bonferroni en riesgo (cota inferior −0,047); (d) SPY-solo en bajista la regla
  M8 se invierte (ΔSharpe −1,49, falsación pre-registrada, resuelta por el panel); (e) clustering exploratorio
  (n=10). → **Q3, O7; líneas rojas §VII.**
- **T4.14** Resumen de límites (claim · test · veredicto), una fila por límite con su cifra y fuente JSON.
  Fuente: cruzada (`bullbear_confirmatory.json`, `leverage_law_panel.json`, `decision_automl_prep.json`,
  `automl_runs/...`).
- Sin figura nueva.
- **Citas:** López de Prado (Bonferroni/DSR, multiple testing), Politis & Romano (límite del IC pooled).

### §4.4.2 Desplegabilidad
- **Contamos:** M8 opera desde el día 1 (regla ex-ante); M10/AutoML necesitan ~150 días de burn-in y reentreno,
  y AutoML cambia de leader en cada reentreno (más difícil de validar en producción). **SIN cifras de coste**
  (decisión Raquel). → **O7, caveat 6.**
- **Sin tabla/figura de costes** (T4.15 y F4.16 ELIMINADAS).
- **Citas:** ref. interna a López de Prado (validación en producción / reentreno).

### §4.4.3 Alcance (alfa) y líneas futuras
- **Contamos:** el resultado probado es **relativo al agente** (el rescate); STRATA no supera al benchmark
  pasivo de forma significativa (n≈250). En punto, 4/10 una derivada supera al pasivo en Sharpe —notablemente
  en leverage débil/invertido (SMCI/MARA/UNG)—: valor direccional, nominal y a posteriori. Líneas futuras:
  muestra mayor (significancia plena), ensemble enrutado por régimen (de la complementariedad DiD), otros
  agentes/LLMs, despliegue real, y **desarrollar una estrategia que genere alfa robusta** (el ángulo α que
  queda fuera de la narrativa del cuerpo). → **C8 (matiz honesto), caveat 7.**
- Sin tabla/figura nuevas (ref. a F4.9).
- **Citas:** ninguna nueva obligatoria.

> **Cierre §4.4 / cierre del capítulo:** STRATA hace lo que prometió y se prueba: rescata al agente en accuracy
> y en riesgo con significancia, el aprendiz redescubre y mejora modestamente la regla, y la naturaleza del
> activo explica qué canal rescata. Lo que no se prueba se dice sin maquillaje: frente a la trivial el avance es
> nominal, la ley del leverage es tendencia al α=0,10 y la regla no sobrevive a Bonferroni en riesgo. El alcance
> probado es la supervisión —rescate y control de riesgo—, no la rentabilidad absoluta; generar alfa robusto
> queda como línea futura.

---

## Citas a verificar antes de redactar (encargo a `revisor-bibliografico`)
Black 1976 · Christie 1982 · Bollerslev 1986 · Adams & MacKay 2007 · Rabiner 1989 · Tashman 2000 ·
López de Prado 2018 (CPCV/DSR/Bonferroni) · Bailey & López de Prado (Deflated Sharpe) · Lundberg & Lee 2017
(SHAP/TreeSHAP) · Politis & Romano 1994 (stationary bootstrap) · Schuirmann 1987 (TOST) · McNemar 1947 ·
Holm 1979 · Sharpe (ratio) · métodos de clustering (KMeans/Ward/GMM/spectral) + índice de Rand (Hubert &
Arabie 1985) · referencia de diferencia-en-diferencias (DiD).

## Dependencias de notación (deben venir cerradas del Cap. 3)
w_t, r_{t+1}, signal_lag=1, σ_t (GARCH), μ_k / regímenes Calma-Estrés-Crisis (HMM K=3), RAM/PSA/GSO con sus
umbrales, M5/M8/M10/AutoML/ZeroR/B&H, pooled-10, Sharpe, Deflated Sharpe, McNemar, TOST, DiD, índice de Rand,
leverage_corr.
