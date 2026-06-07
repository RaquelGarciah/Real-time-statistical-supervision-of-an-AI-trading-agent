"""Bayesian Online Changepoint Detection según Adams & MacKay (2007).

Implementación recursiva con modelo predictivo gaussiano de varianza desconocida
y prior conjugado Normal-Gamma. La recursión se mantiene en espacio logarítmico
sobre la distribución conjunta ``P(r_t, x_{1:t})`` para evitar underflow.

Nota sobre la salida: bajo *hazard* constante ``h`` se tiene
``P(r_t = 0 | x_{1:t}) = h`` por una identidad trivial del filtro (la información
de los datos se cancela al marginalizar). El signo de detección útil no es
``R[t, 0]`` sino:

- ``map_run_length[t] = argmax_r P(r_t = r | x_{1:t})``, que cae bruscamente
  cuando el filtro detecta un cambio.
- ``cp_prob[t] = P(r_t <= short_window | x_{1:t})``, masa acumulada en runs
  recientes; tiende a 1 cuando se sospecha un cambio.

Referencia:

- Adams, R.P. & MacKay, D.J.C. (2007), "Bayesian Online Changepoint
  Detection", arXiv:0710.3742.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import logsumexp


@dataclass
class BOCPDResult:
    """Resultado del filtro BOCPD.

    Atributos:
        run_length_probs: matriz ``(T+1, T+1)`` con ``P(r_t = r | x_{1:t})``.
        map_run_length: longitud de run con mayor probabilidad por paso, ``(T,)``.
        cp_prob: ``P(r_t <= short_window | x_{1:t})``, signo de detección, ``(T,)``.
    """

    run_length_probs: np.ndarray
    map_run_length: np.ndarray
    cp_prob: np.ndarray


def _log_student_t_predictive(
    x: float,
    mu: np.ndarray,
    kappa: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """log-densidad predictiva Student-t bajo prior Normal-Gamma conjugado.

    Predictiva marginal: ``t_{2*alpha}(loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa)))``.
    Ver Adams & MacKay (2007), eq. (3).
    """
    df = 2 * alpha
    scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
    return stats.t.logpdf(x, df=df, loc=mu, scale=scale)


def bocpd(
    observations: np.ndarray,
    hazard: float = 1 / 250,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
    short_window: int = 5,
) -> BOCPDResult:
    """Filtro BOCPD con hazard constante y prior Normal-Gamma.

    Args:
        observations: serie 1D ``x_t``.
        hazard: probabilidad constante de cambio por paso.
        mu0, kappa0, alpha0, beta0: hiperparámetros del prior Normal-Gamma.
        short_window: umbral de longitud de run usado en ``cp_prob``.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1:
        raise ValueError("observations debe ser 1D.")
    T = len(x)

    log_h = np.log(hazard)
    log_1mh = np.log1p(-hazard)

    log_R = np.full((T + 1, T + 1), -np.inf)
    log_R[0, 0] = 0.0

    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])

    for t in range(T):
        log_pred = _log_student_t_predictive(x[t], mu, kappa, alpha, beta)

        log_growth = log_R[t, : t + 1] + log_pred + log_1mh
        log_cp = logsumexp(log_R[t, : t + 1] + log_pred + log_h)

        log_R[t + 1, 1 : t + 2] = log_growth
        log_R[t + 1, 0] = log_cp

        mu_new = (kappa * mu + x[t]) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new = beta + (kappa * (x[t] - mu) ** 2) / (2 * (kappa + 1))

        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])

    log_evidence = logsumexp(log_R, axis=1, keepdims=True)
    run_length_probs = np.exp(log_R - log_evidence)

    map_run_length = run_length_probs[1:].argmax(axis=1).astype(int)
    cp_prob = run_length_probs[1:, : short_window + 1].sum(axis=1)
    return BOCPDResult(
        run_length_probs=run_length_probs,
        map_run_length=map_run_length,
        cp_prob=cp_prob,
    )
