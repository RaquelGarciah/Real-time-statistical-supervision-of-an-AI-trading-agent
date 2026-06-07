"""Tests de ``core.backtest``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest import run_backtest


def test_sin_lag_recupera_serie():
    """Con ``signal_lag=0``, peso constante 1.0 y coste 0: neto = bruto."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    ret = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    w = pd.Series(1.0, index=idx)
    res = run_backtest(ret, w, cost_bps=0.0, signal_lag=0)
    np.testing.assert_allclose(res["net_return"].values, ret.values)


def test_lag_causal_desplaza_la_senal():
    """``signal_lag=1`` aplica la decisión de ``t`` al retorno de ``t+1``."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    w = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=idx)  # posición solo el día 0
    res = run_backtest(ret, w, cost_bps=0.0, signal_lag=1)
    # El peso del día 0 gana el retorno del día 1; el día 0 está plano.
    assert res["gross_return"].iloc[0] == 0.0
    assert res["gross_return"].iloc[1] == 0.02
    assert res["gross_return"].iloc[2:].sum() == 0.0


def test_coste_reduce_retorno_neto():
    """Con coste >0 y rotación, el neto < bruto."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    ret = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    w = pd.Series(np.tile([1.0, -1.0], len(idx) // 2), index=idx)
    res = run_backtest(ret, w, cost_bps=10.0, signal_lag=0)
    assert (res["cost"] > 0).any()
    assert res["net_return"].sum() < (w * ret).sum()


def test_equity_curve_creciente_si_neto_positivo():
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    ret = pd.Series(0.001, index=idx)
    w = pd.Series(1.0, index=idx)
    res = run_backtest(ret, w, cost_bps=0.0, signal_lag=0)
    assert (res["equity"].diff().dropna() > 0).all()


def test_indices_se_intersecan():
    idx_r = pd.date_range("2020-01-01", periods=100, freq="B")
    idx_w = idx_r[20:80]
    ret = pd.Series(0.001, index=idx_r)
    w = pd.Series(0.5, index=idx_w)
    res = run_backtest(ret, w)
    assert len(res) == 60
    assert res.index[0] == idx_w[0]
