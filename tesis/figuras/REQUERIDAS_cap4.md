# Figuras requeridas por el Capítulo 4 (marco práctico, canónico-10)

> Handoff para la sesión del notebook (`STRATA_marco_practico.ipynb`). Los captions del cap. 4 ya están
> escritos y referencian estos nombres exactos. Exportar a `tesis/figuras/<nombre>.pdf` (vectorial, sin título
> embebido: el caption va en LaTeX). Las cifras del caption deben cuadrar con el JSON indicado.
> Estado: las figuras `cap4_*` actuales en `tesis/figuras/` son de la era SMCI y NO sirven.

| # | Fichero (sin ext.) | Sección | Qué muestra | Fuente |
|---|---|---|---|---|
| F4.1 | `cap4_regimenes_spy` | §4.1.1 | Regímenes HMM (Calma/Estrés/Crisis) sombreados sobre el precio de SPY | `regime_direction_table.json` / notebook §3 |
| F4.2 | `cap4_scores_detectores` | §4.1.2 | Distribución de scores RAM/PSA/GSO con su umbral (RAM τ=0,50; PSA P95=0,023; GSO P95=2,371) | `spy_intervention_variants.json` / notebook §2 |
| F4.3 | `cap4_confusion_spy` | §4.1.3 | Matriz de confusión agente (M5) vs intervención (M8) sobre las 121 intervenciones (71 aciertos / 50 fallos) | `spy_intervention_anatomy.json` |
| F4.4 | `cap4_equity_spy` | §4.1.4 | **Equity de las 6 estrategias SPY** (n=251); AutoML bate a la trivial (Sh +2,68 vs +2,21; eq 1,38× vs 1,30×) | `automl_net_returns.json` |
| F4.5 | `cap4_sensibilidad_umbrales` | §4.1.5 | Sensibilidad de accuracy al umbral (meseta = robustez): M8 vs τ RAM y M10 vs umbral de decisión | `psa_gso_threshold_sensitivity.json` |
| F4.6 | `cap4_gate_ram` | §4.2.1 | Gate RAM: tasa de intervención vs discrepancia agente↔régimen, 1 punto/activo (Pearson r=0,93) | `spy_panel_gate_descriptive.json` |
| F4.7 | `cap4_heatmap_accuracy` | §4.2.2 | **Heatmap accuracy activo×estrategia** centrado en 0,5 (color = distancia a la trivial) | `automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json` |
| F4.8 | `cap4_forest_pooled` | §4.2.3 | **Forest plot ΔSharpe pooled-10** + cota Bonferroni: M8 +0,60 [0,05,1,22] (no pasa), M10 +1,12 [0,39,1,84], AutoML +1,08 [0,40,1,85] | `bullbear_confirmatory.json` → `confirmatorio.POOLED10` |
| F4.9 | `cap4_equity_panel` | §4.2.3 | Equity por activo con la ganadora destacada; las 4 victorias en Sharpe vs pasivo (SPY, SMCI, MARA, UNG) | `automl_net_returns.json` |
| F4.10 | `cap4_tost_2x2` | §4.2.4 | Diagrama 2×2 del TOST (ejes accuracy/Sharpe; superioridad/equivalencia); sitúa M10 y AutoML | `equivalence_tost.json` → `POOLED10` |
| F4.11 | `cap4_did_regimen` | §4.2.5 | Complementariedad DiD: ΔSharpe de M10 y AutoML por régimen (alcista/bajista); ΔΔSharpe +1,37 | `regime_did_learners.json` |
| F4.12 | `cap4_atribucion_capas` | §4.3.1 | Izq.: atribución P&L por detector en SPY (RAM 100%, PSA/GSO 0). Der.: timeline diario rescate riesgo (M8) vs accuracy (aprendiz) | `spy_intervention_anatomy.json` + **notebook (timeline)** |
| F4.13 | `cap4_scatter_leverage` | §4.3.2 | **Scatter leverage↔rescate** (10 activos), recta r=−0,56 p=0,093, ROKU señalado como excepción | `leverage_law_panel10.json` |
| F4.14 | `cap4_pca_clusters` | §4.3.3 | PCA 2D de la naturaleza (10 activos), 3 clusters coloreados, PC1≈leverage (r=0,84), mejor canal/grupo | `cluster_panel10.json` + `mechanism_panel.json` |
| F4.15 | `cap4_casos` | §4.3.3 | Dos casos trabajados: XLE rescatado por régimen / MARA por leverage invertido (dirección agente vs supervisor vs régimen) | **notebook §5–§6** |

## Notas
- **JSON nuevo en este capítulo:** `outputs/experiments/leverage_law_panel10.json` (ley del leverage recomputada
  sobre los 10; el `leverage_law_panel.json` anterior lleva metadatos de 15). F4.13 debe leer del `*10`.
- **Pooled-10:** todas las cifras de riesgo salen de `bullbear_confirmatory.json` bloque `POOLED10`
  (n=2493), NO de `decision_automl_prep.json` (que es pooled-15, +0,66). F4.8 y F4.9 deben cuadrar con eso.
- **Necesitan datos a nivel notebook** (no hay JSON directo): el panel derecho de F4.12 (timeline diario) y
  F4.15 (casos XLE/MARA). El resto sale de su JSON.
- Estilo sugerido: vectorial PDF, fuente legible a tamaño de página, sin título embebido, leyenda y ejes en
  español con coma decimal.
