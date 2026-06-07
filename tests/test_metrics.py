"""Tests de ``core.metrics``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import metrics as M


@pytest.fixture()
def retornos() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.012, len(idx)), index=idx)


def test_equity_curve_creciente_si_media_positiva(retornos):
    eq = M.equity_curve(retornos)
    # Aunque hay oscilaciones, el final debe estar por encima del inicio.
    assert eq.iloc[-1] > eq.iloc[0]


def test_sharpe_signo(retornos):
    r_pos = retornos.copy() + 0.001
    r_neg = retornos.copy() - 0.001
    assert M.sharpe(r_pos) > 0
    assert M.sharpe(r_neg) < 0


def test_max_drawdown_no_positivo(retornos):
    eq = M.equity_curve(retornos)
    assert M.max_drawdown(eq) <= 0


def test_profit_factor_solo_ganancias():
    r = pd.Series([0.01, 0.02, 0.03])
    pf = M.profit_factor(r)
    assert pf == float("inf")


def test_hit_rate_rango(retornos):
    h = M.hit_rate(retornos)
    assert 0.0 <= h <= 1.0


def test_turnover_no_negativo():
    w = pd.Series([0.0, 0.5, 0.5, -0.5, 0.5])
    assert M.turnover(w) >= 0


def test_summary_devuelve_diccionario_completo(retornos):
    s = M.summary(retornos)
    esperadas = {"sharpe", "sortino", "max_drawdown", "calmar", "profit_factor", "hit_rate"}
    assert esperadas.issubset(s.keys())
