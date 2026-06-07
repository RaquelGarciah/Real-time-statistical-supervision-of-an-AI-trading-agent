"""Sustituye el proveedor de precios de AI Hedge Fund por yfinance.

AI Hedge Fund usa por defecto la API de pago ``financialdatasets.ai`` para
obtener precios diarios. ``CLAUDE.md`` prohíbe APIs de pago (sección 13), así
que aquí inyectamos un equivalente que llama a yfinance y construye los
objetos ``Price`` que el submódulo espera.

El patch se aplica sin tocar nada dentro de ``agent/ai_hedge_fund/``
(regla CLAUDE.md sección 12). Se reemplaza el símbolo ``get_prices`` en
todos los módulos del submódulo que lo hayan importado con ``from … import``.

Se patchea una sola vez por proceso vía ``apply_price_patch()``; el wrapper
``agent.wrapper.run_agent`` la llama al inicio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yfinance as yf

# El submódulo debe estar accesible en sys.path para importar ``src.*``.
_AGENT_ROOT = Path(__file__).resolve().parent / "ai_hedge_fund"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

_PATCH_APPLIED = False


def _yfinance_get_prices(ticker: str, start_date: str, end_date: str, api_key: str | None = None):
    """Imitación de ``src.tools.api.get_prices`` con datos de yfinance."""
    from src.data.models import Price  # type: ignore

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "time"})
    prices: list = []
    for _, row in df.iterrows():
        prices.append(
            Price(
                open=float(row["Open"]),
                close=float(row["Close"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"]) if not _is_nan(row["Volume"]) else 0,
                time=row["time"].strftime("%Y-%m-%dT%H:%M:%S") if hasattr(row["time"], "strftime") else str(row["time"]),
            )
        )
    return prices


def _is_nan(x) -> bool:
    try:
        return x != x  # noqa: PLR0124  (NaN != NaN trick)
    except Exception:
        return False


def apply_price_patch() -> None:
    """Reemplaza ``get_prices`` en todos los módulos del submódulo que lo importen.

    Idempotente: aplicada más de una vez no tiene efecto adicional.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import importlib

    # Asegurar que el módulo de origen está cargado antes de patchear consumidores.
    api_mod = importlib.import_module("src.tools.api")
    api_mod.get_prices = _yfinance_get_prices  # type: ignore[attr-defined]

    # Cualquier módulo del submódulo que ya haya hecho ``from src.tools.api
    # import get_prices`` mantiene una referencia local; lo reemplazamos en
    # los que sabemos que la importan.
    consumers = [
        "src.agents.risk_manager",
        "src.agents.warren_buffett",
        "src.agents.cathie_wood",
        "src.agents.stanley_druckenmiller",
        "src.agents.michael_burry",
        "src.agents.bill_ackman",
        "src.agents.technicals",
        "src.agents.valuation",
        "src.agents.sentiment",
        "src.agents.fundamentals",
    ]
    for name in consumers:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "get_prices"):
            mod.get_prices = _yfinance_get_prices

    _PATCH_APPLIED = True
