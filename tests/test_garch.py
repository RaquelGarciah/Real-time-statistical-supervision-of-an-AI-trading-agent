"""Tests del módulo ``core.garch``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.garch import GARCHModel


def _sim_garch(seed: int = 0, n: int = 2000) -> pd.Series:
    """Simula retornos GARCH(1,1) Student-t con parámetros conocidos."""
    rng = np.random.default_rng(seed)
    omega = 1e-6
    alpha = 0.08
    beta = 0.90
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    nu = 6.0
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(eps, index=idx, name="r")


def test_fit_recupera_estacionariedad():
    r = _sim_garch()
    m = GARCHModel().fit(r)
    assert m.params is not None
    assert m.params.is_stationary()
    # alpha y beta razonables (en torno a los verdaderos)
    assert 0.02 < m.params.alpha < 0.15
    assert 0.80 < m.params.beta < 0.97


def test_forecast_path_positivo_y_anualizado():
    r = _sim_garch()
    m = GARCHModel().fit(r.iloc[:1500])
    sigma = m.forecast_path(r.iloc[1500:])
    assert (sigma > 0).all()
    # Volatilidad anualizada de un GARCH realista debe quedar entre 1% y 200%
    assert sigma.between(0.01, 2.0).all()


def test_forecast_sin_fit_levanta():
    with pytest.raises(RuntimeError):
        GARCHModel().forecast_path(pd.Series([0.001, 0.002, 0.003]))


def test_forecast_path_causal_sin_lookahead():
    """σ_t de forecast_path NO puede depender de r_t: es la previsión a un paso
    hecha al cierre de t-1, la información disponible al decidir el día t. Por eso
    en GSO se alimenta sigma.iloc[t] directamente (no shift). Perturbar r_k solo
    puede mover σ a partir de k+1, nunca σ_k."""
    r = _sim_garch(seed=3)
    m = GARCHModel().fit(r.iloc[:1500])
    oos = r.iloc[1500:]
    sigma = m.forecast_path(oos)
    oos_pert = oos.copy()
    k = 20
    oos_pert.iloc[k] *= 5.0
    sigma_pert = m.forecast_path(oos_pert)
    # σ en k (y antes) intacta; σ en k+1 cambia.
    np.testing.assert_allclose(sigma.iloc[: k + 1].to_numpy(), sigma_pert.iloc[: k + 1].to_numpy())
    assert sigma.iloc[k + 1] != sigma_pert.iloc[k + 1]


def test_determinismo_misma_serie():
    r = _sim_garch(seed=7, n=1500)
    m1 = GARCHModel().fit(r)
    m2 = GARCHModel().fit(r)
    assert m1.params.alpha == pytest.approx(m2.params.alpha, rel=1e-8)
    assert m1.params.beta == pytest.approx(m2.params.beta, rel=1e-8)
