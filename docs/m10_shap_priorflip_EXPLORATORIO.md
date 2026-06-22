# ¿De qué features depende M10 según leverage-effect vs prior-flip? (EXPLORATORIO)

> **Estado: EXPLORATORIO.** No canónico. n≈250/activo y prior-flip n=2 → descriptivo, no confirmatorio
> (garden of forking paths). Ver `[[trabajo-exploratorio-aislado]]`.

## Pregunta

Hipótesis tentadora: en prior-flip (donde el signo del régimen se invierte), ¿M10 sobrevive porque
**se apoya en otras features** (las del agente) en vez del régimen? Se mide con SHAP por grupo de activos.

## Método

Por activo: 22 features (run_master, override-C) + etiqueta direccional → XGBoost (PARAMS canónicos) →
SHAP (TreeExplainer), |SHAP| medio normalizado a 1. Agregado por grupos: leverage-effect (SPY/QQQ/XLK/
BAC/DIA/XLF), prior-flip (MSTR/SMCI), inverso-estable (NVDA/TSLA/MARA/ROKU). Importancia in-sample
(describe en qué se apoya el modelo, no rendimiento). `experiments/m10_shap_priorflip.py`.

## Resultado — [verificado] (exploratorio)

Peso por bloque (|SHAP| medio):

| grupo | agente | régimen | volatilidad | psa |
|---|---:|---:|---:|---:|
| leverage-effect | 0,34 | 0,38 | 0,13 | 0,14 |
| prior-flip | 0,34 | 0,37 | 0,19 | 0,11 |
| inverso-estable | 0,31 | 0,39 | 0,16 | 0,14 |

**No hay separación limpia entre grupos.** El bloque régimen es el mayor en los tres (~0,38), incluido
prior-flip. Las features individuales dominantes son **las de STRATA** (garch_sigma, psa_score, ram_score,
probas de régimen) en todos los grupos; las personalidades del agente son diminutas una a una.

## Conclusión

1. **La hipótesis NO se sostiene:** M10 **no** se apoya más en el agente en prior-flip. Usa el régimen
   como **contexto** en todos los activos por igual. Su pequeña supervivencia en SMCI/MSTR **no** se
   explica por un cambio de features → coherente con que es **ruido**, no compensación aprendida (encaja
   con `c2_decompose`: las victorias de M10 en ese grupo no eran significativas).
2. **Refuerza el control de universalidad** (CLAUDE.md §3): SHAP identifica **las features de STRATA**
   como las informativas en todos los activos → M10 **redescubre la señal de STRATA**, no una alternativa.
   La única diferencia mínima (prior-flip pesa algo más la volatilidad, 0,19 vs 0,13) está dentro del
   ruido de n=2.
