"""Tests estadísticos avanzados para evaluación de estrategias.

Incluye:

- ``deflated_sharpe`` según López de Prado (2014) para ajustar el Sharpe
  observado por el sesgo de selección cuando se prueban múltiples estrategias.
- ``bootstrap_ci`` con remuestreo no paramétrico (Efron, 1979).
- ``stationary_bootstrap_ci`` de Politis-Romano (1994) para series con
  dependencia temporal.
- ``diebold_mariano`` (Diebold & Mariano, 1995) para comparar pérdidas
  de dos predicciones.

Referencias:

- Efron (1979), "Bootstrap methods: another look at the jackknife", Ann. Statist.
- Politis & Romano (1994), "The stationary bootstrap", J. Amer. Statist. Assoc.
- Diebold & Mariano (1995), "Comparing predictive accuracy", J. Bus. Econ. Stat.
- López de Prado (2014), "The Deflated Sharpe Ratio: correcting for selection
  bias, backtest overfitting, and non-normality", J. Portfolio Mgmt.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from config import SEED


def deflated_sharpe(
    sr_observed: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR), probabilidad de que el verdadero Sharpe > 0.

    ``DSR = Phi((SR - E[SR_max]) * sqrt((n_obs - 1) / (1 - skew*SR + (kurt-1)/4 * SR^2)))``

    donde ``E[SR_max]`` es la esperanza del máximo de ``n_trials`` Sharpes
    independientes bajo H0:SR_true=0 (Bailey & López de Prado, 2014).
    """
    gamma = np.euler_gamma
    # E[max] de n_trials Z's estándar (Bailey & López de Prado 2014, eq 7).
    e_max_z = (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials) + gamma * stats.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    # Bajo H0:SR_true=0, sigma(SR_obs) ≈ 1/sqrt(T-1); la esperanza del máximo
    # del SR muestral es e_max_z / sqrt(T-1).
    expected_max_sr = e_max_z / np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr_observed + (kurt - 1) / 4 * sr_observed**2)
    z = (sr_observed - expected_max_sr) * np.sqrt(n_obs - 1) / den
    return float(stats.norm.cdf(z))


def bootstrap_ci(
    data: np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float],
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Intervalo de confianza percentil bootstrap a nivel ``(1-alpha)``.

    Devuelve ``(low, high)`` con los percentiles ``alpha/2`` y ``1-alpha/2`` de
    la distribución bootstrap de ``statistic``.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data debe ser 1D.")
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    n_obs = len(arr)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        samples[i] = statistic(arr[idx])
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return low, high


def stationary_bootstrap_ci(
    data: np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n: int = 1_000,
    mean_block_len: float | None = None,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """IC bootstrap estacionario (Politis-Romano, 1994) a nivel ``(1-alpha)``.

    Los bloques tienen longitud geométrica con media ``mean_block_len``; este
    esquema preserva la estacionariedad de la serie remuestreada y captura la
    dependencia serial de las series financieras (autocorrelación de retornos,
    *clustering* de volatilidad). Devuelve ``(low, high, point_estimate)``,
    donde ``point_estimate = statistic(data)``.

    Si ``mean_block_len`` es ``None`` se usa ``max(2, round(sqrt(N)))``,
    convención estándar para retornos diarios.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError("data debe ser 1D.")
    n_obs = len(arr)
    if n_obs < 2:
        return float("nan"), float("nan"), float(statistic(arr)) if n_obs else float("nan")
    if mean_block_len is None:
        mean_block_len = max(2.0, float(round(np.sqrt(n_obs))))
    p = 1.0 / mean_block_len  # parámetro de la geométrica de longitud de bloque
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    for i in range(n):
        idx = np.empty(n_obs, dtype=np.int64)
        # Política de "wrap-around" del bootstrap estacionario: índice módulo N.
        idx[0] = rng.integers(0, n_obs)
        u = rng.random(n_obs - 1)
        jumps = rng.integers(0, n_obs, n_obs - 1)
        for t in range(1, n_obs):
            if u[t - 1] < p:
                idx[t] = jumps[t - 1]
            else:
                idx[t] = (idx[t - 1] + 1) % n_obs
        samples[i] = statistic(arr[idx])
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return low, high, float(statistic(arr))


def diebold_mariano(loss1: np.ndarray, loss2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Test de Diebold-Mariano para igualdad de pérdidas predictivas.

    ``H0: E[L_1 - L_2] = 0`` (los dos modelos predicen igual de bien).

    ``loss1, loss2`` son las pérdidas (e.g., error cuadrático) por observación
    de los dos modelos. ``h`` es el horizonte de predicción (afecta a la
    autocorrelación). Devuelve ``(estadístico, p-valor)`` a dos colas.
    """
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    n = len(d)
    mean_d = d.mean()

    # Varianza de largo plazo con autocorrelaciones hasta lag h-1.
    var_d = d.var(ddof=0)
    for lag in range(1, h):
        cov = ((d[:-lag] - mean_d) * (d[lag:] - mean_d)).mean()
        var_d += 2 * cov

    if var_d <= 0:
        return float("nan"), float("nan")
    stat = mean_d / np.sqrt(var_d / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)
