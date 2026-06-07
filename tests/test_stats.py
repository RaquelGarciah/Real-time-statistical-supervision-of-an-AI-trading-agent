"""Tests de ``core.stats``."""

from __future__ import annotations

import numpy as np
import pytest

from core.stats import bootstrap_ci, deflated_sharpe, diebold_mariano


def test_deflated_sharpe_bajo_dsr_cuando_n_trials_grande():
    """Probar muchas estrategias debería penalizar más el SR observado."""
    dsr_pocos = deflated_sharpe(sr_observed=0.05, n_trials=10, n_obs=252)
    dsr_muchos = deflated_sharpe(sr_observed=0.05, n_trials=1000, n_obs=252)
    assert dsr_muchos < dsr_pocos


def test_deflated_sharpe_alto_dsr_si_sr_grande():
    """Con SR muy alto, DSR -> 1 incluso con muchos trials."""
    dsr = deflated_sharpe(sr_observed=0.30, n_trials=1000, n_obs=2520)
    assert dsr > 0.95


def test_bootstrap_ci_contiene_estadistico_real():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.5, scale=1.0, size=500)
    low, high = bootstrap_ci(data, statistic=np.mean, n=2000, seed=1)
    assert low < 0.5 < high


def test_bootstrap_ci_input_1d():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros((10, 2)), statistic=np.mean)


def test_diebold_mariano_iguales_p_alto():
    """Si los dos modelos tienen las mismas pérdidas, el p-valor es alto."""
    rng = np.random.default_rng(0)
    loss1 = rng.normal(1.0, 0.1, 500)
    loss2 = loss1.copy()
    # Las pérdidas idénticas dan diferencia 0, var 0 -> NaN. Añadimos ruido mínimo.
    loss2 = loss1 + rng.normal(0, 0.001, 500)
    _, p = diebold_mariano(loss1, loss2)
    assert p > 0.05


def test_diebold_mariano_diferentes_p_bajo():
    """Si un modelo tiene sistemáticamente menos pérdida, p-valor < 0.05."""
    rng = np.random.default_rng(0)
    loss1 = rng.normal(1.0, 0.1, 500)
    loss2 = rng.normal(0.5, 0.1, 500)
    _, p = diebold_mariano(loss1, loss2)
    assert p < 0.05
