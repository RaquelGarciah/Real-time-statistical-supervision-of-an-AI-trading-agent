# Fuentes bibliográficas de la memoria (solo académicas)

Lista consolidada de las referencias que se citarán en el TFG. Todas deben ser **académicas**
(artículo, libro o tesis), nunca blogs. El agente `experto-citas` verifica cada una contra la
web (Crossref / Semantic Scholar / DOI) antes de darla por válida; las marcadas como "por buscar"
las localiza `revisor-bibliografico` para el estado del arte.

Estado: ⬜ por verificar · ✅ verificada (DOI/URL confirmado) · 🔎 por buscar (no está aún)

Los PDFs de los papers (cuando se consigan) van en esta misma carpeta para verificación de
similitud por `detector-plagio`.

## Modelos de mercado y régimen (cap. 3)

- ⬜ Black, F. (1976). *Studies of stock price volatility changes.* Proc. ASA. — leverage effect.
- ⬜ Christie, A. A. (1982). *The stochastic behavior of common stock variances.* J. Financial Economics 10(4). — leverage effect.
- ⬜ Engle, R. F. (1982). *ARCH with estimates of UK inflation variance.* Econometrica 50(4).
- ⬜ Bollerslev, T. (1986). *Generalized ARCH.* J. Econometrics 31(3).
- ⬜ Bollerslev, T. (1987). *A conditionally heteroskedastic time series model…* Rev. Econ. Stat. 69(3). — GARCH-t.
- ⬜ Hamilton, J. D. (1989). *A new approach to nonstationary time series and the business cycle.* Econometrica 57(2). — regime switching.

## HMM (cap. 3)

- ⬜ Baum, L. et al. (1970). *A maximization technique… Markov chains.* Ann. Math. Stat. 41(1). — Baum-Welch/EM.
- ⬜ Rabiner, L. R. (1989). *A tutorial on hidden Markov models…* Proc. IEEE 77(2).
- ⬜ Viterbi, A. J. (1967). *Error bounds for convolutional codes…* IEEE Trans. Inf. Theory 13(2).

## Detección de cambios / anomalías (cap. 3 y estado del arte)

- ⬜ Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian online changepoint detection.* arXiv:0710.3742. — PSA.
- ⬜ Fearnhead, P. (2006). *Exact and efficient Bayesian inference for multiple changepoint problems.* Stat. Comput. 16(2).

## Validación sin fuga / ML financiero (cap. 3 y 4)

- ⬜ López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. — CPCV, causalidad.
- ⬜ Bailey, D. H. & López de Prado, M. (2014). *The deflated Sharpe ratio.* J. Portfolio Management 40(5). — DSR.
- ⬜ Harvey, C. R., Liu, Y. & Zhu, H. (2016). *… and the cross-section of expected returns.* Rev. Financial Studies 29(1). — multiplicidad.

## Contraste de hipótesis (cap. 3)

- ⬜ McNemar, Q. (1947). *Note on the sampling error… correlated proportions.* Psychometrika 12(2).
- ⬜ Edwards, A. L. (1948). *Note on the "correction for continuity"…* Psychometrika 13(3).
- ⬜ Diebold, F. X. & Mariano, R. S. (1995). *Comparing predictive accuracy.* J. Business & Econ. Stat. 13(3).
- ⬜ Politis, D. N. & Romano, J. P. (1994). *The stationary bootstrap.* JASA 89(428).
- ⬜ Schuirmann, D. J. (1987). *Two one-sided tests procedure…* J. Pharmacokinet. Biopharm. 15(6). — TOST.
- ⬜ Conover, W. J. (1999). *Practical Nonparametric Statistics* (3ª ed.). Wiley. — sign test.

## Riesgo y métricas económicas (cap. 3)

- ⬜ Sharpe, W. F. (1994). *The Sharpe ratio.* J. Portfolio Management 21(1).
- ⬜ Moreira, A. & Muir, T. (2017). *Volatility-managed portfolios.* J. Finance 72(4). — volatility targeting.

## Estado del arte — POR BUSCAR (revisor-bibliografico)

- 🔎 Agentes LLM aplicados a decisiones de inversión / trading (papers recientes, 2023–2025).
- 🔎 Supervisión / monitorización en tiempo de ejecución de sistemas de ML (runtime monitoring).
- 🔎 Detección de regímenes de mercado como capa de control de riesgo (revisión reciente).
- 🔎 Manual/serie temporal en español de referencia (p. ej. Pérez López, C. (2011). *Series Temporales*. Garceta) — confirmar edición exacta.

> Nota: las entradas BibTeX correspondientes están en `tesis/bibliography.bib` (semilla).
