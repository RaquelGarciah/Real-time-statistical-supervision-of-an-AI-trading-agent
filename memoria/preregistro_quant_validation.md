# BORRADOR de pre-registro para BITACORA — Validación de producción de M10 (panel)

> Raquel: este es un **borrador** para que lo revises y lo pegues tú en `BITACORA.md` (no lo escribo yo
> porque la BITACORA es tu cuaderno). `@rigor-matematico` y `@experto-inferencia` exigen que el pre-registro
> exista **antes** de que estas cifras entren a la memoria del TFG (CLAUDE.md §7). El experimento ya se
> ejecutó, así que en rigor esto es un pre-registro *a posteriori*: lo honesto es marcarlo como tal o, mejor,
> re-ejecutar tras pegarlo si quieres que conste como pre-registro estricto.

---

## [2026-06-20] [Pre-registro] - Validación de producción 'real quant' de M10 (SMCI + panel de 10)

**Contexto.** Aplicar a M10 la batería de *due-diligence* de un comité de inversión (separar habilidad de
suerte con control de multiplicidad + mérito económico neto de costes), reutilizando `core.stats`/`core.metrics`
y un módulo nuevo `core.validation`. Caso SMCI + panel de 10 (SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR, SMCI,
ROKU, MARA). No es la defensa del TFG: es el contraste honesto de aptitud para producción.

**Hipótesis (falsable).** M10 exhibe habilidad direccional desplegable que **sobrevive al control de
multiplicidad** del panel (no es un ganador por *data-snooping* entre 10 activos).

**H0.** Ningún activo del panel tiene accuracy direccional > 0,5 una vez controlada la tasa de falsos
descubrimientos (FDR) sobre los 10 contrastes.

**Estadístico.** (a) Por activo: sign test de **una cola** (accuracy > 0,5), HAC t (Newey-West), Sharpe e IC
de Lo (2002), PSR/DSR, McNemar vs M5/M8/B&H, Diebold-Mariano de P&L, permutación por bloques vs B&H.
(b) Panel/multiplicidad: FDR Benjamini-Hochberg y Benjamini-Yekutieli sobre los 10 p de habilidad; haircut de
Sharpe Harvey-Liu-Zhu (best-of-10); PBO/CSCV (Bailey et al. 2017); MinBTL; White Reality Check y Hansen SPA
vs cash y vs B&H propio. (c) Economía: P&L neto con escenarios de borrow {0,100,300,500} pb; Sharpe/Sortino/
Calmar/Information Ratio; VaR/CVaR; turnover; capacidad ADV; atribución factorial Fama-French con errores HAC.

**Criterio de éxito.** ≥1 activo sobrevive al FDR-BH a α=0,10 **y** el haircut de Sharpe del mejor activo se
mantiene > 0 **y** la PBO < 0,5 **y** el alpha factorial sigue significativo tras Bonferroni del panel.

**Criterio de fracaso (pre-registrado).** Si 0/10 activos sobreviven al FDR, o el haircut lleva el Sharpe a 0,
o la PBO ≥ 0,5, o el alpha pierde significancia tras multiplicidad → **NO-GO**: la ventaja es nominal pero no
apta para producción con esta muestra. (Análogo a la regla `prior-flip`: documenta cuándo NO funciona.)

**Datos.** OOS desde 2024-10-01 (el agente LLM solo existe post-cutoff); calibración por activo hasta
2024-09-30, congelada. M10 walk-forward, embargo=1, burn-in=150, 10 semillas, 22 features, signal_lag=1.
Multiplicidad axes: n_trials_dsr=6 (configs metodológicas por activo) y n_trials_haircut=10 (selección
best-of-panel). Factores Fama-French diarios (5 factores + momentum) descargados de Kenneth French.

**Output esperado.** `outputs/experiments/quant_validation_panel.json` con claves `meta`, `por_activo.<TK>`
({headline, skill_vs_luck, risk, econ, factor_attribution}), `multiplicity_panel` ({fdr_bh, fdr_by,
haircut_sharpe_mejor_activo, alpha_mejor_activo, pbo_cscv, min_btl_years, white_reality_check_vs_cash/_vs_bh,
hansen_spa_vs_cash/_vs_bh}) y `verdict`.

**Resultado (ya ejecutado, para la entrada [Hallazgo] que acompañe).** 10/10 activos evaluados. FDR-BH y
FDR-BY: **0/10** rechazos. SMCI (mejor): accuracy 0,552, Sharpe 1,84, sign test 1-cola p=0,057, HAC t=2,09
(p=0,037), block-perm vs B&H p=0,047 — *borderline nominal*. Bajo multiplicidad: haircut de Sharpe → 0
(recorte ~100 %), PBO=0,38, alpha factorial t=2,47 (p=0,0135 → 0,135 tras Bonferroni del panel), DSR 0,72 /
0,57 / 0,43 para 6/12/24 pruebas, MinBTL≈2,5 años (vs ~1 de OOS). White RC vs cash p=0,024 (rechaza, benchmark
laxo) pero vs B&H propio p=0,069 (no rechaza). Borrow despreciable (Sharpe 1,84→1,81 a 500 pb): M10 no está
permanentemente corto. **Veredicto: NO-GO (condicional)** — se cumple el criterio de fracaso pre-registrado.

**Implicaciones para el TFG.** Refuerza la honestidad del relato: la ventaja de M10 es nominal y no sobrevive
al control de multiplicidad a n≈250; la aportación es el protocolo de supervisión, no alfa. Material para el
capítulo de límites / trabajo futuro (forward test, muestra más larga, borrow y capacidad reales).

**Referencias.** `core/validation.py`, `experiments/quant_validation_panel.py`,
`notebooks/STRATA_quant_validation.ipynb`, `outputs/experiments/quant_validation_panel.json`. Newey-West (1987),
Lo (2002), Harvey-Liu-Zhu (2016), Bailey-Borwein-López de Prado-Zhu (2017), White (2000), Hansen (2005),
Benjamini-Hochberg (1995), Benjamini-Yekutieli (2001), López de Prado (2014).
