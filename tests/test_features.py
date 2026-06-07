"""Tests del módulo ``core.features``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import features as F


@pytest.fixture()
def precios() -> pd.Series:
    """Serie de precios sintética con un suave drift al alza."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


def test_log_returns_identidad_geometrica(precios):
    r = F.log_returns(precios).dropna()
    reconstruido = precios.iloc[0] * np.exp(r.cumsum())
    np.testing.assert_allclose(reconstruido.values, precios.iloc[1:].values, rtol=1e-12)


def test_realized_vol_no_negativa(precios):
    r = F.log_returns(precios)
    rv = F.realized_vol(r, window=5).dropna()
    assert (rv >= 0).all()


def test_rsi_acotado(precios):
    valores = F.rsi(precios, window=14).dropna()
    assert valores.between(0, 100).all()


def test_sma_constante_para_serie_constante():
    s = pd.Series([10.0] * 50)
    media = F.sma(s, window=10).dropna()
    assert (media == 10.0).all()


def test_momentum_signo_correcto():
    s = pd.Series(np.linspace(100, 200, 60))
    m = F.momentum(s, window=22).dropna()
    assert (m > 0).all()


def test_build_feature_matrix_columnas_esperadas(precios):
    df = pd.DataFrame({"close_spx": precios, "close_vix": precios * 0.2})
    feats = F.build_feature_matrix(df)
    esperadas = {
        "ret_log",
        "log_vix",
        "rv_5",
        "rsi_14",
        "sma_50",
        "sma_200",
        "mom_22",
        "ret_lag_1",
        "ret_lag_5",
    }
    assert esperadas.issubset(feats.columns)
    assert len(feats) == len(df)


def test_features_deterministas(precios):
    df = pd.DataFrame({"close_spx": precios, "close_vix": precios * 0.2})
    a = F.build_feature_matrix(df)
    b = F.build_feature_matrix(df)
    pd.testing.assert_frame_equal(a, b)
