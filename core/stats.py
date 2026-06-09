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
- McNemar (1947), "Note on the sampling error of the difference between
  correlated proportions or percentages", Psychometrika.
- Edwards (1948), corrección de continuidad para proporciones correlacionadas.
- Conover (1999), *Practical Nonparametric Statistics*, 3ª ed., §3.4 (sign test).
- Schuirmann (1987), "A comparison of the two one-sided tests procedure...",
  J. Pharmacokinet. Biopharm. (TOST de equivalencia).
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


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    exact_threshold: int = 25,
) -> tuple[float, float, int, int]:
    """Test de McNemar pareado sobre aciertos correlacionados (McNemar 1947).

    ``H0: P(b) = P(c)``, donde sobre los mismos ``N`` días pareados ``b`` cuenta
    los días en que A acierta y B falla, y ``c`` los días en que B acierta y A
    falla. Las concordancias (ambos aciertan / ambos fallan) no informan y se
    ignoran por diseño.

    Usa el **binomial exacto** (``b ~ Binom(b+c, 1/2)``) cuando ``b + c <
    exact_threshold``, donde la aproximación chi-cuadrado es poco fiable; en
    otro caso el chi-cuadrado con corrección de continuidad de Edwards (1948):
    ``chi2 = (|b - c| - 1)^2 / (b + c)``.

    ``correct_a``, ``correct_b`` son arrays booleanos de aciertos **alineados
    día a día** (mismo índice). Devuelve ``(estadístico, p_valor_2colas, b, c)``;
    el estadístico es el chi-cuadrado, o ``nan`` en el caso exacto.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    if a.shape != b_arr.shape:
        raise ValueError("correct_a y correct_b deben tener la misma longitud (pares alineados).")
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    if b + c == 0:
        return float("nan"), 1.0, b, c
    if b + c < exact_threshold:
        p = float(stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
        return float("nan"), p, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), p, b, c


def sign_test(
    correct: np.ndarray,
    p0: float = 0.5,
) -> tuple[int, int, float, tuple[float, float]]:
    """Sign test binomial exacto contra ``p0`` (Conover 1999, §3.4).

    ``H0: P(acierto) = p0``. Contraste a dos colas mediante el binomial exacto,
    sin aproximación normal. Devuelve ``(k_aciertos, n, p_valor_2colas,
    ic95_proporción)`` con intervalo de Clopper-Pearson (exacto).

    Supuesto: independencia de las observaciones. Con días consecutivos hay
    autocorrelación que infla el error tipo I; úsese como contraste de margen
    amplio, no para diferencias finas.
    """
    x = np.asarray(correct, dtype=bool)
    n = int(x.size)
    k = int(x.sum())
    if n == 0:
        return 0, 0, float("nan"), (float("nan"), float("nan"))
    res = stats.binomtest(k, n, p0, alternative="two-sided")
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    return k, n, float(res.pvalue), (float(ci.low), float(ci.high))


def block_permutation_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    block_len: int | None = None,
    n: int = 10_000,
    seed: int = SEED,
) -> tuple[float, float]:
    """Test de permutación por bloques para Δ-aciertos entre dos modelos pareados.

    Blinda a McNemar contra la autocorrelación serial: permuta las etiquetas
    A/B por bloques contiguos de longitud ``block_len`` (por defecto
    ``round(sqrt(N))``), preservando la dependencia temporal, y construye la
    distribución nula de ``mean(correct_a) - mean(correct_b)``.

    Devuelve ``(estadístico_observado, p_valor_2colas)``.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("correct_a y correct_b deben tener la misma longitud.")
    n_obs = a.size
    if n_obs == 0:
        return float("nan"), float("nan")
    if block_len is None:
        block_len = max(1, int(round(np.sqrt(n_obs))))
    obs = float(a.mean() - b.mean())
    diff = a - b  # d_t; bajo H0 el signo de cada bloque es intercambiable
    starts = np.arange(0, n_obs, block_len)
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        signs = rng.choice((-1.0, 1.0), size=len(starts))
        flipped = diff.copy()
        for s_idx, start in enumerate(starts):
            flipped[start : start + block_len] *= signs[s_idx]
        null[i] = flipped.mean()
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


def tost(
    diff: np.ndarray | pd.Series,
    margin: float,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    """Two One-Sided Tests de equivalencia (Schuirmann 1987).

    Contrasta ``H0: |E[diff]| >= margin`` frente a equivalencia. Permite
    AFIRMAR que dos series son equivalentes dentro de ``±margin`` (lo que un
    contraste de superioridad como Diebold-Mariano no permite: ese solo da
    "no hay evidencia de diferencia"). ``diff`` es la serie pareada de
    diferencias (p. ej. pérdida diaria de A menos la de B).

    Devuelve ``(p_valor_TOST, equivalente)``, con ``p_valor_TOST = max`` de los
    dos p-valores unilaterales y ``equivalente = p_valor_TOST < alpha``.
    """
    d = np.asarray(diff, dtype=float)
    n = d.size
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        equiv = abs(mean_d) < margin
        return (0.0 if equiv else 1.0), bool(equiv)
    df = n - 1
    t_lower = (mean_d - (-margin)) / se  # H0: E[diff] <= -margin
    t_upper = (mean_d - margin) / se     # H0: E[diff] >= +margin
    p_lower = float(stats.t.sf(t_lower, df))    # cola derecha
    p_upper = float(stats.t.cdf(t_upper, df))   # cola izquierda
    p_tost = max(p_lower, p_upper)
    return p_tost, bool(p_tost < alpha)
