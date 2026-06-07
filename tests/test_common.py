"""Tests de utilidades compartidas de experimentos (``experiments/_common.py``).

Cubren la convención de caché por activo introducida para que STRATA sea
multi-activo: ``cache/agent/<TICKER>/<TICKER>_<date>.json``.
"""

from __future__ import annotations

import pytest

from config import CACHE_AGENT_DIR
from experiments._common import agent_cache_path, load_agent_decision_cache


def test_agent_cache_path_layout_por_activo():
    """La ruta vive en ``cache/agent/<TICKER>/<TICKER>_<date>.json``."""
    p = agent_cache_path("2024-10-01", "SPY")
    assert p.parts[-2:] == ("SPY", "SPY_2024-10-01.json")
    assert p.parent == CACHE_AGENT_DIR / "SPY"


def test_agent_cache_path_normaliza_ticker_a_mayusculas():
    """Un ticker en minúsculas resuelve a la misma carpeta canónica en mayúsculas."""
    assert agent_cache_path("2024-10-01", "spy") == agent_cache_path("2024-10-01", "SPY")


@pytest.mark.parametrize("ticker", ["SPY", "NVDA"])
def test_carga_decision_cacheada_desde_subcarpeta(ticker):
    """Las decisiones cacheadas de SPY y NVDA se cargan desde su subcarpeta por activo."""
    d = load_agent_decision_cache("2024-10-01", ticker)
    assert d is not None, f"falta la decisión cacheada de {ticker} 2024-10-01"
    assert d["ticker"] == ticker
    assert {"action", "size"} <= d.keys()
