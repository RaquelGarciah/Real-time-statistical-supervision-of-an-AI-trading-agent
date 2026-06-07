"""Descarga y caché de datos de mercado del S&P 500 y el VIX.

La caché vive en ``data/`` (regenerable, no versionada en Git). Cuando un fichero
parquet existe se carga sin red; en caso contrario se descarga con ``yfinance``,
se persiste y se devuelve. La granularidad es diaria de cierre ajustado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_DIR, TICKER_INDEX, TICKER_VIX


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """Ruta del parquet de caché para una serie."""
    safe = ticker.replace("^", "").replace("/", "_")
    return DATA_DIR / f"{safe}_{start}_{end}.parquet"


def load_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Devuelve OHLCV diario ajustado de un ticker entre ``start`` y ``end``.

    Lee de ``data/{ticker}_{start}_{end}.parquet`` si existe; si no, descarga
    de yfinance y guarda. Las fechas son strings ``YYYY-MM-DD`` inclusivas.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)
    if path.exists():
        return pd.read_parquet(path)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance no devolvió datos para {ticker} en {start}-{end}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_parquet(path)
    return df


def load_sp500_and_vix(start: str, end: str) -> pd.DataFrame:
    """DataFrame con cierres ajustados de S&P 500 y VIX alineados por fecha.

    Columnas devueltas: ``close_spx``, ``close_vix``. Las fechas son la
    intersección de ambos calendarios (días con cierre en los dos).
    """
    spx = load_market_data(TICKER_INDEX, start, end)
    vix = load_market_data(TICKER_VIX, start, end)
    df = pd.DataFrame(
        {
            "close_spx": spx["Close"],
            "close_vix": vix["Close"],
        }
    ).dropna()
    return df
