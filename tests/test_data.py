"""Tests del módulo ``core.data``.

Estos tests no descargan de red: usan un parquet sintético ya cacheado en
``data/`` que los propios tests crean y borran. Así CI no depende de yfinance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DATA_DIR
from core import data as data_module


@pytest.fixture()
def fake_cache(tmp_path, monkeypatch):
    """Redirige DATA_DIR a un directorio temporal y crea parquets dummy."""
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)

    idx = pd.date_range("2020-01-01", periods=10, freq="B", name="date")
    spx = pd.DataFrame({"Close": np.linspace(3000, 3100, 10)}, index=idx)
    vix = pd.DataFrame({"Close": np.linspace(15, 20, 10)}, index=idx)

    spx.to_parquet(tmp_path / "GSPC_2020-01-01_2020-01-15.parquet")
    vix.to_parquet(tmp_path / "VIX_2020-01-01_2020-01-15.parquet")
    return tmp_path


def test_load_market_data_lee_de_cache(fake_cache):
    df = data_module.load_market_data("^GSPC", "2020-01-01", "2020-01-15")
    assert len(df) == 10
    assert "Close" in df.columns


def test_load_sp500_and_vix_alinea_por_fecha(fake_cache):
    df = data_module.load_sp500_and_vix("2020-01-01", "2020-01-15")
    assert list(df.columns) == ["close_spx", "close_vix"]
    assert len(df) == 10
    assert df["close_spx"].iloc[0] == pytest.approx(3000)
    assert df["close_vix"].iloc[-1] == pytest.approx(20)


def test_cache_path_es_determinista():
    p = data_module._cache_path("^GSPC", "2000-01-01", "2021-12-31")
    assert p.name == "GSPC_2000-01-01_2021-12-31.parquet"
    assert p.parent == DATA_DIR
