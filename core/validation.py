"""Batería de validación 'quant' para decidir si una estrategia merece capital.

Complementa a ``core.stats`` (primitivas habilidad-vs-suerte) y ``core.metrics``
(rentabilidad/riesgo) con las herramientas que el sector usa para separar
habilidad de suerte controlando la multiplicidad de pruebas y para medir el
mérito económico neto de costes. No duplica nada de ``core.stats``: lo reutiliza.

Bloques:

- Inferencia robusta a autocorrelación: ``hac_tstat`` (Newey-West 1987), error
  estándar e IC del Sharpe de Lo (2002).
- Riesgo de cola: ``var_historical``/``cvar_historical`` y sus variantes
  Cornish-Fisher (colas no normales); ``information_ratio``.
- Multiplicidad y *overfitting*: ``fdr_bh``/``fdr_by`` (Benjamini-Hochberg 1995;
  Benjamini-Yekutieli 2001), ``haircut_sharpe`` (Harvey-Liu-Zhu 2016),
  ``pbo_cscv`` (Bailey et al. 2017), ``min_btl`` (López de Prado 2014),
  ``reality_check`` (White 2000) y ``hansen_spa`` (Hansen 2005).
- Atribución factorial: ``load_ff_factors`` (Kenneth French) y
  ``factor_attribution`` (OLS con errores HAC).
- Coste de préstamo en corto: ``apply_borrow_cost``.

Referencias:

- Newey & West (1987), "A simple, positive semi-definite, heteroskedasticity and
  autocorrelation consistent covariance matrix", Econometrica.
- Lo (2002), "The statistics of Sharpe ratios", Financial Analysts Journal.
- Rockafellar & Uryasev (2000), "Optimization of conditional value-at-risk", J. Risk.
- Cornish & Fisher (1938), "Moments and cumulants in the specification of distributions".
- Grinold & Kahn (1999), *Active Portfolio Management*, 2ª ed. (Information Ratio).
- Benjamini & Hochberg (1995), J. R. Statist. Soc. B; Benjamini & Yekutieli (2001),
  Ann. Statist. (FDR bajo dependencia).
- Harvey, Liu & Zhu (2016), "...and the cross-section of expected returns", RFS;
  Harvey & Liu (2015), "Backtesting", J. Portfolio Mgmt. (haircut Sharpe).
- Bailey, Borwein, López de Prado & Zhu (2017), "The probability of backtest
  overfitting", J. Computational Finance (PBO vía CSCV).
- López de Prado (2014), "Pseudo-mathematics and financial charlatanism", Notices AMS (MinBTL).
- White (2000), "A reality check for data snooping", Econometrica.
- Hansen (2005), "A test for superior predictive ability", J. Bus. Econ. Stat.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import COST_BPS, DATA_DIR, SEED

ANN = 252


# ============================================================================
# 1 · Inferencia robusta a autocorrelación
# ============================================================================

def _nw_bandwidth(n: int) -> int:
    """Ancho de banda automático de Newey-West (1994): ``floor(4*(n/100)^(2/9))``."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def hac_tstat(x, lags: int | None = None) -> tuple[float, float, float]:
    """t de ``H0: E[x]=0`` con varianza Newey-West (kernel de Bartlett).

    La varianza de largo plazo de la media es ``S/n`` con
    ``S = gamma_0 + 2*sum_{k=1..L}(1 - k/(L+1)) gamma_k`` (Newey-West 1987), que
    corrige la autocorrelación que infla el error tipo I en retornos diarios.
    ``lags`` por defecto = regla automática de Newey-West (1994).

    Devuelve ``(media, t, p_dos_colas)`` (p asintótica normal).
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n < 3 or a.std() == 0:
        return (float(a.mean()) if n else float("nan")), float("nan"), float("nan")
    L = _nw_bandwidth(n) if lags is None else int(lags)
    d = a - a.mean()
    s = float(np.mean(d * d))  # gamma_0
    for k in range(1, L + 1):
        gamma_k = float(np.mean(d[k:] * d[:-k]))
        s += 2.0 * (1.0 - k / (L + 1.0)) * gamma_k
    if s <= 0:
        return float(a.mean()), float("nan"), float("nan")
    t = a.mean() / np.sqrt(s / n)
    p = float(2 * (1 - stats.norm.cdf(abs(t))))
    return float(a.mean()), float(t), p


def sharpe_se_lo(returns, freq: int = ANN, autocorr: bool = True,
                 q: int | None = None, alpha: float = 0.05) -> dict:
    """Error estándar e IC del Sharpe anualizado (Lo 2002).

    Bajo iid (Lo 2002, ec. 9) el SE del Sharpe por periodo es
    ``sqrt((1 + 0.5*SR_p^2)/n)``. La anualización ingenua multiplica por
    ``sqrt(freq)``; bajo autocorrelación Lo (2002, ec. 19) sustituye ese factor
    por ``eta_q = q / sqrt(q + 2*sum_{k=1..q-1}(q-k) rho_k)``, que se reduce a
    ``sqrt(q)`` si los retornos son iid. ``eta`` devuelto es el cociente
    ``eta_q / sqrt(freq)`` (penalización por autocorrelación; 1.0 = sin penalización).

    Devuelve ``{sharpe, se, ci_low, ci_high, eta, sharpe_period, se_period}``.

    Nota: es una APROXIMACIÓN de la ec. 19 de Lo, no su forma literal: las
    autocorrelaciones se estiman solo hasta la banda Newey-West sobre retornos
    diarios (las de orden alto se asumen ~0); con n~cientos sumar las q-1
    muestrales solo añadiría ruido. Supone además normalidad asintótica de la
    media, optimista con colas gordas a n pequeño (usar block-permutation/sign
    test como contraste libre de distribución).
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = r.size
    sd = r.std(ddof=1) if n > 1 else 0.0
    if n < 3 or sd == 0:
        return {"sharpe": float("nan"), "se": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "eta": float("nan"),
                "sharpe_period": float("nan"), "se_period": float("nan")}
    sr_p = r.mean() / sd
    se_p = np.sqrt((1 + 0.5 * sr_p ** 2) / n)
    qh = freq if q is None else int(q)
    if autocorr and qh > 1:
        d = r - r.mean()
        var = float(np.mean(d * d))
        # Solo se estiman las autocorrelaciones de orden bajo (ancho de banda
        # Newey-West); las de orden alto se asumen ~0, como en los retornos
        # reales. Sumar las q-1 muestrales sobre n~cientos solo añadiría ruido.
        L = min(qh - 1, max(1, _nw_bandwidth(n)))
        acc = 0.0
        if var > 0:
            for k in range(1, L + 1):
                acc += (qh - k) * float(np.mean(d[k:] * d[:-k]) / var)
        denom = qh + 2.0 * acc
        eta_q = qh / np.sqrt(denom) if denom > 0 else np.sqrt(qh)
    else:
        eta_q = np.sqrt(freq)
    sr_ann = sr_p * eta_q
    se_ann = se_p * eta_q
    z = stats.norm.ppf(1 - alpha / 2)
    return {"sharpe": float(sr_ann), "se": float(se_ann),
            "ci_low": float(sr_ann - z * se_ann), "ci_high": float(sr_ann + z * se_ann),
            "eta": float(eta_q / np.sqrt(freq)),
            "sharpe_period": float(sr_p), "se_period": float(se_p)}


# ============================================================================
# 2 · Riesgo de cola
# ============================================================================

def var_historical(returns, level: float = 0.95) -> float:
    """VaR histórico al nivel ``level`` (número positivo = magnitud de pérdida)."""
    r = pd.Series(returns).dropna().to_numpy()
    if r.size == 0:
        return float("nan")
    return float(-np.quantile(r, 1 - level))


def cvar_historical(returns, level: float = 0.95) -> float:
    """CVaR / *expected shortfall* histórico: pérdida media más allá del VaR (Rockafellar-Uryasev 2000)."""
    r = pd.Series(returns).dropna().to_numpy()
    if r.size == 0:
        return float("nan")
    thr = np.quantile(r, 1 - level)
    tail = r[r <= thr]
    return float(-tail.mean()) if tail.size else float(-thr)


def _cornish_fisher_z(z: float, s: float, k_exc: float) -> float:
    """Cuantil ajustado de Cornish-Fisher (1938) a partir del cuantil normal ``z``."""
    return (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k_exc / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)


def var_cornish_fisher(returns, level: float = 0.95) -> float:
    """VaR paramétrico con expansión de Cornish-Fisher (ajusta el cuantil normal por asimetría y curtosis)."""
    r = pd.Series(returns).dropna().to_numpy()
    if r.size < 4:
        return float("nan")
    mu, sigma = r.mean(), r.std(ddof=1)
    s = float(stats.skew(r))
    k_exc = float(stats.kurtosis(r))  # exceso (Fisher) por defecto
    z_cf = _cornish_fisher_z(stats.norm.ppf(1 - level), s, k_exc)
    return float(-(mu + z_cf * sigma))


def cvar_cornish_fisher(returns, level: float = 0.95, n_grid: int = 200) -> float:
    """CVaR bajo la distribución ajustada por Cornish-Fisher.

    ``ES = (1/alpha) * integral_0^alpha VaR(u) du`` con ``alpha = 1-level``,
    integrada numéricamente sobre la cola con cuantiles de Cornish-Fisher.
    """
    r = pd.Series(returns).dropna().to_numpy()
    if r.size < 4:
        return float("nan")
    mu, sigma = r.mean(), r.std(ddof=1)
    s = float(stats.skew(r))
    k_exc = float(stats.kurtosis(r))
    alpha = 1 - level
    us = np.linspace(alpha / n_grid, alpha, n_grid)
    q = np.array([mu + _cornish_fisher_z(stats.norm.ppf(u), s, k_exc) * sigma for u in us])
    return float(-q.mean())


def information_ratio(returns, benchmark, freq: int = ANN) -> dict:
    """Information Ratio frente a un *benchmark* (Grinold-Kahn 1999).

    ``IR = media(activo)/desv(activo) * sqrt(freq)`` con
    ``activo = retorno - benchmark`` alineados por fecha. Devuelve
    ``{ir, tracking_error, active_mean_ann}``.
    """
    a = pd.Series(returns).dropna()
    b = pd.Series(benchmark).dropna()
    common = a.index.intersection(b.index)
    act = (a.loc[common] - b.loc[common]).to_numpy()
    if act.size < 2 or act.std(ddof=1) == 0:
        return {"ir": float("nan"), "tracking_error": float("nan"), "active_mean_ann": float("nan")}
    te = act.std(ddof=1) * np.sqrt(freq)
    return {"ir": float(act.mean() / act.std(ddof=1) * np.sqrt(freq)),
            "tracking_error": float(te), "active_mean_ann": float(act.mean() * freq)}


# ============================================================================
# 3 · Multiplicidad y overfitting
# ============================================================================

def fdr_bh(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg (1995): control de FDR bajo independencia/dependencia positiva."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    kmax = np.nonzero(below)[0].max() + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    p_adj = np.empty(m)
    p_adj[order] = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1].clip(0, 1)
    return {"reject": reject.tolist(), "pvals_adj": p_adj.tolist(),
            "threshold": float(thresh[kmax - 1]) if kmax > 0 else 0.0,
            "n_rejected": int(reject.sum())}


def fdr_by(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Yekutieli (2001): FDR bajo dependencia arbitraria (factor ``c(m)=sum 1/i``)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    cm = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / (m * cm))
    below = ranked <= thresh
    kmax = np.nonzero(below)[0].max() + 1 if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    p_adj = np.empty(m)
    p_adj[order] = np.minimum.accumulate((ranked * m * cm / np.arange(1, m + 1))[::-1])[::-1].clip(0, 1)
    return {"reject": reject.tolist(), "pvals_adj": p_adj.tolist(),
            "threshold": float(thresh[kmax - 1]) if kmax > 0 else 0.0,
            "n_rejected": int(reject.sum()), "c_m": cm}


def _sr_to_p(sr_ann: float, n_obs: int, freq: int = ANN) -> float:
    """p-valor de una cola de un Sharpe anualizado (t = SR_periodo * sqrt(n)).

    Usa la aproximación normal del estadístico t (Harvey-Liu-Zhu emplean la t de
    Student); con n~cientos la diferencia es despreciable.
    """
    t = (sr_ann / np.sqrt(freq)) * np.sqrt(n_obs)
    return float(1 - stats.norm.cdf(t))


def _p_to_sr(p: float, n_obs: int, freq: int = ANN) -> float:
    """Sharpe anualizado implícito por un p-valor de una cola."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    t = stats.norm.ppf(1 - p)
    return float(t / np.sqrt(n_obs) * np.sqrt(freq))


def haircut_sharpe(sharpe_obs: float, n_trials: int, n_obs: int,
                   freq: int = ANN) -> dict:
    """Recorte de Sharpe por multiplicidad (Harvey-Liu-Zhu 2016, Harvey-Liu 2015).

    Convierte el Sharpe observado en un p-valor de una cola, lo penaliza por las
    ``n_trials`` configuraciones probadas (Bonferroni; Benjamini-Yekutieli con el
    factor ``c(m)``) y reconvierte cada p ajustado en un Sharpe 'descontado'.

    ``sr_haircut_holm`` se devuelve por completitud pero COINCIDE con Bonferroni:
    para el contraste de rango 1 (el mejor Sharpe, único que recibe la función) el
    primer paso de Holm usa el factor ``m``, idéntico a Bonferroni; no es un método
    independiente (el step-down real exigiría el vector completo de p de las n_trials).

    Devuelve los Sharpe recortados y el haircut porcentual (BY = el más severo).
    """
    p = _sr_to_p(sharpe_obs, n_obs, freq)
    cm = float(np.sum(1.0 / np.arange(1, n_trials + 1)))
    p_bonf = min(1.0, p * n_trials)
    p_holm = min(1.0, p * n_trials)  # rango 1 entre n_trials → idéntico a Bonferroni
    p_bhy = min(1.0, p * n_trials * cm)
    # El Sharpe descontado se acota en 0: si el p ajustado satura, la estrategia
    # es indistinguible del azar (haircut del 100%), no un Sharpe negativo.
    sr_bonf = max(0.0, _p_to_sr(p_bonf, n_obs, freq))
    sr_holm = max(0.0, _p_to_sr(p_holm, n_obs, freq))
    sr_bhy = max(0.0, _p_to_sr(p_bhy, n_obs, freq))
    haircut = 1 - sr_bhy / sharpe_obs if sharpe_obs > 0 else float("nan")
    return {"sr_obs": float(sharpe_obs), "p_single": float(p),
            "sr_haircut_bonferroni": sr_bonf, "sr_haircut_holm": sr_holm,
            "sr_haircut_bhy": sr_bhy, "haircut_pct": float(haircut)}


def min_btl(n_trials: int, target_sharpe: float = 1.0) -> float:
    """Minimum Backtest Length en años (López de Prado 2014).

    Años mínimos para que el Sharpe anual máximo esperado bajo no-habilidad de
    ``n_trials`` pruebas no supere ``target_sharpe``:
    ``MinBTL = (E[max_N]/SR*)^2`` con
    ``E[max_N] = (1-gamma)*Z^{-1}(1-1/N) + gamma*Z^{-1}(1-1/(N e))``.
    """
    g = np.euler_gamma
    e_max = (1 - g) * stats.norm.ppf(1 - 1 / n_trials) + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float((e_max / target_sharpe) ** 2)


def _sharpe_blocks(sel: np.ndarray, block_sum, block_sumsq, block_cnt):
    """Sharpe (anualizado) por config sobre los bloques seleccionados, vía sumas precomputadas."""
    cnt = block_cnt[sel].sum()
    s = block_sum[sel].sum(axis=0)
    ss = block_sumsq[sel].sum(axis=0)
    mean = s / cnt
    var = ss / cnt - mean ** 2
    var = np.where(var > 0, var, np.nan)
    return mean / np.sqrt(var) * np.sqrt(ANN)


def pbo_cscv(returns_matrix, n_splits: int = 16) -> dict:
    """Probability of Backtest Overfitting vía CSCV (Bailey, Borwein, López de Prado & Zhu 2017).

    ``returns_matrix``: DataFrame ``(T x N)`` con la serie de retornos de cada
    configuración. Se parte el tiempo en ``S`` bloques; en cada combinación de
    ``S/2`` bloques como *in-sample* se elige la config con mejor Sharpe IS y se
    mira su rango OOS en el complementario. PBO = fracción de combinaciones en
    que la mejor IS cae por debajo de la mediana OOS (logit ≤ 0).

    Devuelve ``{pbo, n_combinations, n_configs, slope_degradation, prob_oos_loss}``.
    """
    from itertools import combinations
    M = pd.DataFrame(returns_matrix).dropna().to_numpy(dtype=float)
    T, N = M.shape
    S = n_splits - (n_splits % 2)
    if T < S or N < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "n_configs": int(N),
                "slope_degradation": float("nan"), "prob_oos_loss": float("nan"),
                "nota": "muestra/configs insuficientes para CSCV"}
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]
    b_cnt = np.array([len(b) for b in blocks], dtype=float)
    b_sum = np.array([M[b].sum(axis=0) for b in blocks])
    b_sumsq = np.array([(M[b] ** 2).sum(axis=0) for b in blocks])
    logits, is_perf, oos_perf = [], [], []
    n_loss = 0
    for combo in combinations(range(S), S // 2):
        sel = np.zeros(S, dtype=bool); sel[list(combo)] = True
        sr_is = _sharpe_blocks(sel, b_sum, b_sumsq, b_cnt)
        sr_oos = _sharpe_blocks(~sel, b_sum, b_sumsq, b_cnt)
        if np.all(np.isnan(sr_is)):
            continue
        nstar = int(np.nanargmax(sr_is))
        oos_star = sr_oos[nstar]
        valid = ~np.isnan(sr_oos)
        rank = (np.sum(sr_oos[valid] < oos_star) + 1) / (valid.sum() + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
        is_perf.append(sr_is[nstar]); oos_perf.append(oos_star)
        n_loss += int(oos_star < 0)
    logits = np.array(logits)
    n_comb = len(logits)
    if n_comb == 0:
        return {"pbo": float("nan"), "n_combinations": 0, "n_configs": int(N),
                "slope_degradation": float("nan"), "prob_oos_loss": float("nan")}
    slope = float(np.polyfit(is_perf, oos_perf, 1)[0]) if n_comb > 2 else float("nan")
    return {"pbo": float(np.mean(logits <= 0)), "n_combinations": int(n_comb),
            "n_configs": int(N), "slope_degradation": slope,
            "prob_oos_loss": float(n_loss / n_comb)}


def _stationary_boot_indices(n: int, n_boot: int, mean_block_len: float, seed: int):
    """Genera ``n_boot`` vectores de índices del bootstrap estacionario (Politis-Romano 1994)."""
    p = 1.0 / mean_block_len
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        u = rng.random(n - 1)
        jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        yield idx


def reality_check(strategy_returns_matrix, benchmark, n_boot: int = 2000,
                  mean_block_len: float | None = None, seed: int = SEED) -> dict:
    """White (2000) Reality Check vía bootstrap estacionario.

    ``f_k = retorno_estrategia_k - benchmark`` (positivo = mejor que el
    *benchmark*). Estadístico ``V = max_k sqrt(n) * mean(f_k)``; el bootstrap
    recentrado en la media muestral da la distribución nula. Controla el FWER
    sobre el universo completo de estrategias frente al data-snooping.

    Devuelve ``{best_strategy, V, p_value, n_strategies}``.
    """
    F = pd.DataFrame(strategy_returns_matrix).dropna()
    b = pd.Series(benchmark).reindex(F.index)
    common = F.index[b.notna()]
    F = F.loc[common]; b = b.loc[common]
    f = F.sub(b, axis=0).to_numpy()  # (n x K)
    n, K = f.shape
    if n < 5 or K < 1:
        return {"best_strategy": None, "V": float("nan"), "p_value": float("nan"), "n_strategies": int(K)}
    if mean_block_len is None:
        mean_block_len = max(2.0, float(round(np.sqrt(n))))
    fbar = f.mean(axis=0)
    V = np.sqrt(n) * fbar.max()
    best = int(np.argmax(fbar))
    count = 0
    for idx in _stationary_boot_indices(n, n_boot, mean_block_len, seed):
        fbar_b = f[idx].mean(axis=0)
        V_b = np.sqrt(n) * (fbar_b - fbar).max()
        count += int(V_b >= V)
    return {"best_strategy": F.columns[best] if hasattr(F, "columns") else best,
            "V": float(V), "p_value": float(count / n_boot), "n_strategies": int(K)}


def hansen_spa(strategy_returns_matrix, benchmark, n_boot: int = 2000,
               mean_block_len: float | None = None, seed: int = SEED) -> dict:
    """Hansen (2005) SPA: versión estudentizada y recentrada del Reality Check.

    ``T_SPA = max_k sqrt(n)*mean(f_k)/omega_k`` con ``omega_k`` la desv. típica
    bootstrap. El p-valor consistente recentra solo las estrategias no demasiado
    malas (umbral ``A_n``), siendo menos conservador que White. Devuelve los tres
    p-valores de Hansen (lower/consistent/upper).
    """
    F = pd.DataFrame(strategy_returns_matrix).dropna()
    b = pd.Series(benchmark).reindex(F.index)
    common = F.index[b.notna()]
    F = F.loc[common]; b = b.loc[common]
    f = F.sub(b, axis=0).to_numpy()
    n, K = f.shape
    if n < 5 or K < 1:
        return {"t_spa": float("nan"), "p_lower": float("nan"),
                "p_consistent": float("nan"), "p_upper": float("nan"), "n_strategies": int(K)}
    if mean_block_len is None:
        mean_block_len = max(2.0, float(round(np.sqrt(n))))
    fbar = f.mean(axis=0)
    boot_means = np.empty((n_boot, K))
    for i, idx in enumerate(_stationary_boot_indices(n, n_boot, mean_block_len, seed)):
        boot_means[i] = f[idx].mean(axis=0)
    omega = boot_means.std(axis=0, ddof=1) * np.sqrt(n)
    omega = np.where(omega > 0, omega, np.nan)
    t_k = np.sqrt(n) * fbar / omega
    t_spa = float(np.nanmax(t_k))
    # Umbral de recentrado A_n (Hansen 2005): -sqrt(var_k/n)*sqrt(2 log log n).
    thr = -(omega / np.sqrt(n)) * np.sqrt(2 * np.log(np.log(n))) if n > np.e else -np.inf
    def _p(mask):
        g = np.where(mask, fbar, 0.0)  # media recentrada según política
        stat = np.sqrt(n) * (boot_means - g) / omega
        Tb = np.nanmax(stat, axis=1)
        return float(np.mean(Tb >= t_spa))
    p_lower = _p(np.ones(K, dtype=bool))                 # recentra todas (liberal)
    p_consistent = _p(fbar >= thr)                       # recentra las no muy malas
    p_upper = _p(fbar >= 0)                              # recentra solo las positivas (conservador)
    return {"t_spa": t_spa, "p_lower": p_lower, "p_consistent": p_consistent,
            "p_upper": p_upper, "n_strategies": int(K)}


# ============================================================================
# 4 · Atribución factorial (Fama-French)
# ============================================================================

_FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
_MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Momentum_Factor_daily_CSV.zip")


def _read_ff_zip(url: str) -> pd.DataFrame:
    """Descarga un ZIP de la librería de Kenneth French y parsea su CSV diario."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        raw = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = zf.namelist()[0]
    text = zf.read(name).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append(parts)
    header_cols = None
    for line in text.splitlines():
        if "Mkt-RF" in line or "Mom" in line:
            header_cols = [c.strip() for c in line.split(",")]
            break
    df = pd.DataFrame(rows).set_index(0)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    if header_cols and len(header_cols) - 1 == df.shape[1]:
        df.columns = header_cols[1:]
    return df


def load_ff_factors(start: str, end: str, cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Carga los 5 factores Fama-French + momentum diarios (decimal), con caché en parquet.

    Descarga de la web de Kenneth French (gratis) si no hay caché. Columnas:
    ``Mkt-RF, SMB, HML, RMW, CMA, MOM, RF``. Lanza excepción si no hay red ni caché.
    """
    cache = Path(cache_dir) / "ff_factors_daily.parquet"
    if cache.exists():
        ff = pd.read_parquet(cache)
    else:
        ff5 = _read_ff_zip(_FF5_URL)
        mom = _read_ff_zip(_MOM_URL)
        mom.columns = ["MOM"]
        ff = ff5.join(mom, how="inner")
        cache.parent.mkdir(parents=True, exist_ok=True)
        ff.to_parquet(cache)
    return ff.loc[(ff.index >= pd.Timestamp(start)) & (ff.index <= pd.Timestamp(end))]


def _ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS con errores estándar Newey-West HAC. ``X`` incluye la constante."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    L = _nw_bandwidth(n) if lags is None else int(lags)
    S = (X * resid[:, None]).T @ (X * resid[:, None])  # gamma_0
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        Xe_t = (X[j:] * resid[j:, None])
        Xe_s = (X[:-j] * resid[:-j, None])
        Gamma = Xe_t.T @ Xe_s
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "se": se, "t": tval, "r2": r2, "n": n}


def factor_attribution(returns, factors: pd.DataFrame, freq: int = ANN,
                       hac_lags: int | None = None, returns_are_log: bool = True) -> dict:
    """Atribución factorial: OLS del exceso de retorno sobre los factores FF con errores HAC.

    Regresa ``r_estrategia - RF`` sobre ``[Mkt-RF, SMB, HML, RMW, CMA, MOM]`` con
    errores Newey-West. Si tras quitar las betas conocidas el ``alpha`` sigue
    siendo significativo, hay valor propio; si no, solo se cobra una prima de
    riesgo conocida. Devuelve ``{alpha_ann, t_alpha, betas, t_betas, r2, n_obs}``.
    """
    r = pd.Series(returns).dropna()
    if returns_are_log:  # los factores FF son retornos simples; convertir explícitamente
        r = np.expm1(r)
    common = r.index.intersection(factors.index)
    facs = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"] if c in factors.columns]
    if len(common) < 20:
        return {"alpha_ann": float("nan"), "t_alpha": float("nan"), "betas": {},
                "t_betas": {}, "r2": float("nan"), "n_obs": int(len(common)),
                "nota": "solapamiento de fechas insuficiente con los factores FF"}
    y = (r.loc[common] - factors.loc[common, "RF"]).to_numpy()
    Xf = factors.loc[common, facs].to_numpy()
    X = np.column_stack([np.ones(len(common)), Xf])
    fit = _ols_hac(y, X, lags=hac_lags)
    betas = {f: float(fit["beta"][i + 1]) for i, f in enumerate(facs)}
    t_betas = {f: float(fit["t"][i + 1]) for i, f in enumerate(facs)}
    return {"alpha_ann": float(fit["beta"][0] * freq), "t_alpha": float(fit["t"][0]),
            "betas": betas, "t_betas": t_betas, "r2": float(fit["r2"]), "n_obs": int(fit["n"])}


# ============================================================================
# 5 · Coste de préstamo en corto
# ============================================================================

def apply_borrow_cost(returns, weights, borrow_bps_yr: float = 0.0,
                      cost_bps: float = COST_BPS, signal_lag: int = 1,
                      ann: int = ANN) -> pd.DataFrame:
    """``run_backtest`` + comisión diaria de préstamo sobre el notional en corto.

    El fee usa el **mismo peso retardado** que aplica ``run_backtest``
    (``w.shift(signal_lag)``), de modo que la posición que gana ``r_{t+1}`` es la
    que paga el borrow de ese día. ``borrow_bps_yr`` es la tasa anual de préstamo
    (pb); se reparte ``/ann`` por día sobre ``max(-w, 0)``. Añade columnas
    ``borrow``, ``net_return_borrow`` y ``equity_borrow``.
    """
    from core.backtest import run_backtest
    bt = run_backtest(returns, weights, cost_bps=cost_bps, signal_lag=signal_lag)
    w = pd.Series(weights).reindex(bt.index).astype(float).fillna(0.0)
    if signal_lag:
        w = w.shift(signal_lag).fillna(0.0)
    short_notional = (-w).clip(lower=0.0)
    borrow = (borrow_bps_yr / 10_000) / ann * short_notional
    bt = bt.copy()
    bt["borrow"] = borrow
    bt["net_return_borrow"] = bt["net_return"] - borrow
    bt["equity_borrow"] = (1 + bt["net_return_borrow"]).cumprod()
    return bt
