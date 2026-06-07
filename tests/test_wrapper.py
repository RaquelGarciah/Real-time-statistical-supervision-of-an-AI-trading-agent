"""Tests unitarios de ``agent.wrapper`` (sin tocar la red).

El test del flujo completo (con OpenRouter) vive en ``live/`` como smoke test.
Aquí solo verificamos las utilidades puras.
"""

from __future__ import annotations

from agent.wrapper import _action_to_strata, _quantity_to_size


def test_action_to_strata_long():
    assert _action_to_strata("buy") == "long"
    assert _action_to_strata("cover") == "long"


def test_action_to_strata_short():
    assert _action_to_strata("short") == "short"
    assert _action_to_strata("sell") == "short"


def test_action_to_strata_hold():
    assert _action_to_strata("hold") == "hold"
    assert _action_to_strata("desconocido") == "hold"


def test_quantity_to_size_long():
    # 100 acciones × 50€ = 5000€ sobre 100k → 0.05
    s = _quantity_to_size(quantity=100, action="buy", cash=100_000, price=50)
    assert s == 0.05


def test_quantity_to_size_short_signo_negativo():
    s = _quantity_to_size(quantity=200, action="short", cash=100_000, price=50)
    assert s < 0
    assert abs(s) == 0.10


def test_quantity_to_size_clip():
    """Si el tamaño excede 1.0, se clipea a 1.0."""
    s = _quantity_to_size(quantity=10_000, action="buy", cash=100_000, price=50)
    assert s == 1.0


def test_quantity_to_size_hold_cero():
    assert _quantity_to_size(100, "hold", 100_000, 50) == 0.0
    assert _quantity_to_size(0, "buy", 100_000, 50) == 0.0
