"""Features cuantitativas: retornos log, volatilidad realizada e indicadores técnicos.

Todas las funciones devuelven ``pd.Series`` alineadas con el índice de entrada.
Los NaN del calentamiento (warm-up) de cada ventana móvil se preservan: la
limpieza con ``dropna`` queda en mano del consumidor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Retornos logarítmicos ``r_t = log(p_t / p_{t-1})``."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series, window: int = 5) -> pd.Series:
    """Volatilidad realizada en ventana móvil, sin anualizar.

    ``sigma_t = sqrt( (1/(w-1)) * sum_{i=t-w+1}^{t} r_i^2 )`` aplicada con
    ``rolling.std`` (estimador insesgado por defecto en pandas).
    """
    return returns.rolling(window).std()


def realized_vol_annualized(returns: pd.Series, window: int = 21) -> pd.Series:
    """Volatilidad realizada anualizada en ventana móvil ``window`` (≈ mes).

    Multiplica la ``rolling.std`` por ``√252`` para anualizar. Es la feature
    estándar del HMM de regímenes descrita en ``replicar_regimen_mercado.md``:
    backward-looking, derivada solo de retornos pasados (sin VIX).
    """
    return returns.rolling(window).std() * np.sqrt(252)


def ewma_vol_annualized(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad EWMA anualizada estilo RiskMetrics (J.P. Morgan 1996).

    ``sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_t^2``, anualizada con ``√252``.
    Reacciona más rápido que la ventana móvil rectangular ``realized_vol_annualized``
    (pondera exponencialmente el pasado reciente, sin ventana dura). ``adjust=False``
    la hace estrictamente backward-looking: cada punto usa solo ``r_{1:t}`` → causal,
    apta para el filtrado del HMM sin look-ahead. ``lam=0.94`` es el valor diario
    canónico de RiskMetrics.
    """
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var) * np.sqrt(252)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index de Wilder (1978) con suavizado exponencial.

    ``RSI = 100 - 100 / (1 + RS)`` donde ``RS`` es la razón entre la media
    exponencial de las ganancias y la de las pérdidas en la ventana ``window``.
    """
    diff = prices.diff()
    gains = diff.clip(lower=0)
    losses = (-diff).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Media móvil simple sobre ``window`` periodos."""
    return prices.rolling(window).mean()


def momentum(prices: pd.Series, window: int = 22) -> pd.Series:
    """Momentum a ``window`` periodos: ``p_t / p_{t-w} - 1``."""
    return prices / prices.shift(window) - 1


def build_feature_matrix(df_market: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de features completa para HMM y XGBoost.

    Recibe un DataFrame con columnas ``close_spx`` y ``close_vix`` (salida de
    ``core.data.load_sp500_and_vix``). Devuelve un DataFrame con:

    - ``ret_log``: retorno log diario del S&P 500.
    - ``log_vix``: ``log(VIX)`` (auxiliar; el HMM ya no la usa).
    - ``rv_5``: volatilidad realizada a 5 días.
    - ``rv_21_ann``: volatilidad realizada a 21 días anualizada (feature 2 del HMM).
    - ``rsi_14``: RSI de 14 días.
    - ``sma_50``, ``sma_200``: medias móviles del S&P.
    - ``mom_22``: momentum a 22 días.
    - ``ret_lag_1`` .. ``ret_lag_5``: retornos retardados (features XGBoost).

    Las filas con NaN derivados del calentamiento se conservan; el consumidor
    decide cuándo aplicar ``dropna``.
    """
    spx = df_market["close_spx"]
    vix = df_market["close_vix"]

    feats = pd.DataFrame(index=df_market.index)
    feats["ret_log"] = log_returns(spx)
    feats["log_vix"] = np.log(vix)
    feats["rv_5"] = realized_vol(feats["ret_log"], window=5)
    feats["rv_21_ann"] = realized_vol_annualized(feats["ret_log"], window=21)
    feats["rsi_14"] = rsi(spx, window=14)
    feats["sma_50"] = sma(spx, 50)
    feats["sma_200"] = sma(spx, 200)
    feats["mom_22"] = momentum(spx, 22)
    for lag in range(1, 6):
        feats[f"ret_lag_{lag}"] = feats["ret_log"].shift(lag)
    return feats
