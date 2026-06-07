"""Features macro y de sentimiento para alimentar a las personalidades del agente.

SPY es un ETF agregado y carece de fundamentales empresariales en sentido
estricto. El wrapper por defecto del submódulo AI Hedge Fund consulta
Financial Datasets API en busca de FCF, márgenes, deuda, insider trading,
etc., y para SPY devuelve sistemáticamente *"insufficient data on fundamentals"*
(BITACORA 2026-05-16). El pivot del 2026-05-19 sustituye esa fuente por un
**snapshot macro y de sentimiento** que las personalidades consumen para
razonar sobre el mercado agregado en lugar de pedir fundamentales empresariales.

Fuentes:

- ``^VIX`` y ``^TNX`` (yfinance): volatilidad implícita y tipo del bono 10Y.
- ETFs sectoriales SPDR (XLF, XLK, XLE, XLI, XLY, XLP, XLV, XLU, XLB,
  XLRE, XLC): retornos relativos a SPY que indican qué sectores lideran.
- SPY mismo: momentum 1M y YTD para situar el régimen general.

La función central es ``build_macro_snapshot(date, ticker)``: devuelve un
``dict`` serializable con los indicadores para esa fecha. Lo consume
``agent/_macro_patch.py`` para inyectar el contexto en el prompt de las cinco
personalidades antes de cada llamada al LLM.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from core.data import load_market_data

# Lista canónica de ETFs sectoriales SPDR. Cubren el 100 % del S&P 500 por
# sectores GICS. Se reusa también en otros experimentos del proyecto.
SECTOR_ETFS: tuple[str, ...] = (
    "XLF",   # Financieros
    "XLK",   # Tecnología
    "XLE",   # Energía
    "XLI",   # Industriales
    "XLY",   # Consumo discrecional
    "XLP",   # Consumo básico
    "XLV",   # Salud
    "XLU",   # Utilities
    "XLB",   # Materiales
    "XLRE",  # Real estate
    "XLC",   # Comunicaciones
)

# Ventanas de momentum (días bursátiles) usadas en el snapshot.
_M1_DAYS = 21       # ≈ un mes
_M3_DAYS = 63       # ≈ tres meses
_LOOKBACK_BUFFER = 365  # días de calendario para descargar suficientes datos.


def _last_close_on_or_before(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    """Devuelve el último cierre con índice ≤ ``target`` y su fecha."""
    up_to = df.loc[:target]
    if up_to.empty:
        raise ValueError(f"Sin datos previos a {target.date()} para la serie.")
    last_idx = up_to.index[-1]
    return last_idx, float(up_to["Close"].iloc[-1])


def _return_n_days_back(df: pd.DataFrame, target: pd.Timestamp, n: int) -> float | None:
    """Retorno simple a ``n`` cierres bursátiles vista desde ``target`` (incluido)."""
    up_to = df.loc[:target]
    if len(up_to) <= n:
        return None
    p_now = float(up_to["Close"].iloc[-1])
    p_then = float(up_to["Close"].iloc[-1 - n])
    return (p_now / p_then) - 1.0 if p_then > 0 else None


def _ytd_return(df: pd.DataFrame, target: pd.Timestamp) -> float | None:
    """Retorno desde el primer cierre del año calendario de ``target``."""
    year_start = pd.Timestamp(year=target.year, month=1, day=1)
    in_year = df.loc[year_start:target]
    if in_year.empty:
        return None
    p_first = float(in_year["Close"].iloc[0])
    p_now = float(in_year["Close"].iloc[-1])
    return (p_now / p_first) - 1.0 if p_first > 0 else None


def _vix_percentile(vix: pd.DataFrame, target: pd.Timestamp, lookback_days: int = 252) -> float | None:
    """Percentil del VIX vs el último año de cierres."""
    window = vix.loc[:target].tail(lookback_days)
    if window.empty:
        return None
    last = float(window["Close"].iloc[-1])
    return float((window["Close"] <= last).mean())


def build_macro_snapshot(date: str, ticker: str = "SPY") -> dict:
    """Construye el snapshot macro/sentimiento usado para ``(date, ticker)``.

    Args:
        date: fecha de la decisión en formato ``YYYY-MM-DD``.
        ticker: subyacente principal (por defecto SPY). Si es distinto de SPY,
            la lectura de momentum y la rotación sectorial se mantienen sobre
            SPY (es la referencia de mercado), y solo el bloque ``ticker`` se
            recalcula para el activo concreto.

    Returns:
        Diccionario serializable con las claves ``as_of``, ``index_state``,
        ``risk_state``, ``sector_rotation``, ``ticker`` y ``ticker_state``.
        Todos los valores numéricos son ``float`` salvo cuando faltan datos
        suficientes, en cuyo caso son ``None``.
    """
    target = pd.Timestamp(date)
    start = (target - timedelta(days=_LOOKBACK_BUFFER)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance excluye end.

    # Bloque 1: estado del índice (SPY) y métricas de riesgo (VIX, TNX).
    spy = load_market_data("SPY", start, end)
    vix = load_market_data("^VIX", start, end)
    tnx = load_market_data("^TNX", start, end)

    spy_date, spy_close = _last_close_on_or_before(spy, target)
    vix_date, vix_close = _last_close_on_or_before(vix, target)
    tnx_date, tnx_close = _last_close_on_or_before(tnx, target)

    index_state = {
        "close": spy_close,
        "ret_1m": _return_n_days_back(spy, target, _M1_DAYS),
        "ret_3m": _return_n_days_back(spy, target, _M3_DAYS),
        "ret_ytd": _ytd_return(spy, target),
    }
    risk_state = {
        "vix": vix_close,
        "vix_change_1d": _return_n_days_back(vix, target, 1),
        "vix_percentile_1y": _vix_percentile(vix, target),
        "tnx_10y": tnx_close,
        "tnx_change_1d": _return_n_days_back(tnx, target, 1),
    }

    # Bloque 2: rotación sectorial relativa a SPY (1M).
    spy_1m = index_state["ret_1m"]
    rotation: dict[str, float | None] = {}
    for etf in SECTOR_ETFS:
        try:
            df = load_market_data(etf, start, end)
        except Exception:
            rotation[etf] = None
            continue
        r = _return_n_days_back(df, target, _M1_DAYS)
        if r is None or spy_1m is None:
            rotation[etf] = None
        else:
            rotation[etf] = r - spy_1m

    # Bloque 3: estado específico del ticker (idéntico a SPY salvo ETF distinto).
    if ticker == "SPY":
        ticker_state = dict(index_state)
    else:
        try:
            df = load_market_data(ticker, start, end)
            _, close_t = _last_close_on_or_before(df, target)
            ticker_state = {
                "close": close_t,
                "ret_1m": _return_n_days_back(df, target, _M1_DAYS),
                "ret_3m": _return_n_days_back(df, target, _M3_DAYS),
                "ret_ytd": _ytd_return(df, target),
            }
        except Exception:
            ticker_state = {"close": None, "ret_1m": None, "ret_3m": None, "ret_ytd": None}

    return {
        "as_of": target.strftime("%Y-%m-%d"),
        "last_close_dates": {
            "spy": spy_date.strftime("%Y-%m-%d"),
            "vix": vix_date.strftime("%Y-%m-%d"),
            "tnx": tnx_date.strftime("%Y-%m-%d"),
        },
        "index_state": index_state,
        "risk_state": risk_state,
        "sector_rotation_1m_vs_spy": rotation,
        "ticker": ticker,
        "ticker_state": ticker_state,
    }


def format_macro_text(snapshot: dict) -> str:
    """Convierte el snapshot en texto legible para inyectarlo en un prompt.

    El formato es bloques de líneas ``clave: valor`` agrupados, suficientemente
    explícitos para que el LLM razone sobre régimen de mercado, riesgo y
    liderazgo sectorial sin necesidad de fundamentales empresariales.
    """
    def _pct(x: float | None) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
        return f"{x * 100:+.2f}%"

    def _num(x: float | None, fmt: str = ".2f") -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "n/a"
        return format(x, fmt)

    idx = snapshot["index_state"]
    risk = snapshot["risk_state"]
    rot = snapshot["sector_rotation_1m_vs_spy"]
    tkr = snapshot["ticker_state"]
    tkr_name = snapshot["ticker"]

    rotation_lines = []
    sorted_rot = sorted(
        ((etf, r) for etf, r in rot.items() if r is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for etf, r in sorted_rot:
        rotation_lines.append(f"  {etf}: {_pct(r)}")

    sections = [
        f"Contexto macro y de mercado a fecha {snapshot['as_of']}",
        "(SPY como proxy del S&P 500, datos cerrados al último día bursátil disponible)",
        "",
        "Estado del índice (SPY):",
        f"  Cierre: {_num(idx['close'])}",
        f"  Retorno 1 mes: {_pct(idx['ret_1m'])}",
        f"  Retorno 3 meses: {_pct(idx['ret_3m'])}",
        f"  Retorno YTD: {_pct(idx['ret_ytd'])}",
        "",
        "Métricas de riesgo:",
        f"  VIX: {_num(risk['vix'])} (cambio diario {_pct(risk['vix_change_1d'])}; "
        f"percentil 1Y {_num(risk['vix_percentile_1y'], '.2f')})",
        f"  TNX (10Y): {_num(risk['tnx_10y'])} (cambio diario {_pct(risk['tnx_change_1d'])})",
        "",
        "Rotación sectorial — exceso de retorno 1M vs SPY (positivo = lidera):",
    ]
    sections.extend(rotation_lines or ["  (sin datos)"])
    if tkr_name != "SPY":
        sections.extend([
            "",
            f"Estado del subyacente {tkr_name}:",
            f"  Cierre: {_num(tkr['close'])}",
            f"  Retorno 1 mes: {_pct(tkr['ret_1m'])}",
            f"  Retorno 3 meses: {_pct(tkr['ret_3m'])}",
            f"  Retorno YTD: {_pct(tkr['ret_ytd'])}",
        ])
    return "\n".join(sections)
