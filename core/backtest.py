"""Motor de backtest vectorizado con costes lineales de transacción.

Convención:

- ``returns`` son los retornos diarios netos del activo subyacente (escala
  decimal, NO porcentaje), indexados por fecha de cierre.
- ``weights`` es la **decisión** de peso tomada con datos hasta el cierre del
  día ``t``. Por defecto ``run_backtest`` la desfasa ``signal_lag=1`` día
  (causalidad): la posición decidida el día ``t`` gana el retorno de ``t+1``.
  Esto evita el look-ahead de aplicar ``peso_t × retorno_t`` (la decisión del
  día ``t`` usa el cierre de ``t``, así que no es ejecutable sobre el retorno de
  ``t``). Para medir sin desfase (p. ej. comparar alineamientos) usar
  ``signal_lag=0``.
- Los costes son ``cost_bps`` puntos básicos sobre el notional rotado, esto
  es: ``cost_t = cost_bps/10000 * |w_t - w_{t-1}|`` (sobre los pesos ya desfasados).

La función ``run_backtest`` devuelve los retornos netos y la equity curve
en formato ``pd.DataFrame``.
"""

from __future__ import annotations

import pandas as pd

from config import COST_BPS


def run_backtest(
    returns: pd.Series,
    weights: pd.Series,
    cost_bps: float = COST_BPS,
    signal_lag: int = 1,
) -> pd.DataFrame:
    """Aplica los pesos a los retornos del subyacente con coste lineal.

    Args:
        returns: serie de retornos del subyacente, indexada por fecha.
        weights: serie de pesos en ``[-1, 1]`` (o más amplio si admite
            apalancamiento) con el mismo índice que ``returns`` o subconjunto.
        cost_bps: coste en puntos básicos por unidad de notional rotada.
        signal_lag: días de desfase de la señal antes de aplicarla al retorno.
            ``1`` (por defecto) = causal (decisión en ``t`` → retorno ``t+1``);
            ``0`` = mismo día (sin desfase).

    Returns:
        DataFrame con columnas ``gross_return``, ``cost``, ``net_return`` y
        ``equity`` (curva acumulada partiendo de 1.0).
    """
    common = returns.index.intersection(weights.index)
    r = returns.loc[common].astype(float)
    w = weights.loc[common].astype(float).fillna(0.0)
    if signal_lag:
        # La posición decidida en t se mantiene durante t+signal_lag (causal).
        w = w.shift(signal_lag).fillna(0.0)

    gross = w * r
    cost = (cost_bps / 10_000) * w.diff().abs().fillna(abs(w.iloc[0]))
    net = gross - cost
    equity = (1 + net).cumprod()

    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "equity": equity,
        }
    )
