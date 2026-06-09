"""Métricas de rendimiento para evaluación de estrategias de trading.

Todas las funciones operan sobre ``pd.Series`` de retornos netos diarios (escala
decimal) o sobre la equity curve correspondiente. Las que anualizan asumen 252
días bursátiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

ANN = 252


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Equity acumulada multiplicativa a partir de retornos netos."""
    return initial * (1 + returns.fillna(0)).cumprod()


def sharpe(returns: pd.Series, freq: int = ANN, rf: float = 0.0) -> float:
    """Sharpe ratio anualizado, riesgo libre constante por defecto ``0``.

    ``SR = (mean(r) - rf/freq) / std(r) * sqrt(freq)`` con std muestral.
    """
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return float("nan")
    return float((r.mean() - rf / freq) / r.std() * np.sqrt(freq))


def sortino(returns: pd.Series, freq: int = ANN, mar: float = 0.0) -> float:
    """Sortino ratio anualizado con MAR (mínimo aceptable) ``mar``.

    Usa la desviación a la baja: ``downside_std = sqrt(mean(min(r-mar, 0)^2))``.
    """
    r = returns.dropna()
    downside = np.minimum(r - mar / freq, 0.0)
    dd_std = np.sqrt((downside**2).mean())
    if dd_std == 0:
        return float("nan")
    return float((r.mean() - mar / freq) / dd_std * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    """Máximo drawdown como número negativo en escala decimal.

    Ej: ``-0.32`` indica una caída máxima del 32% respecto a un pico previo.
    """
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    return float(dd.min())


def calmar(returns: pd.Series, freq: int = ANN) -> float:
    """Calmar ratio: rentabilidad anualizada / |MaxDD|."""
    r = returns.dropna()
    eq = equity_curve(r)
    mdd = max_drawdown(eq)
    if mdd == 0:
        return float("nan")
    ann_return = (eq.iloc[-1]) ** (freq / len(r)) - 1
    return float(ann_return / abs(mdd))


def profit_factor(returns: pd.Series) -> float:
    """Profit factor: suma de ganancias / suma de pérdidas en valor absoluto."""
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def hit_rate(returns: pd.Series) -> float:
    """Proporción de días con retorno estrictamente positivo."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def turnover(weights: pd.Series) -> float:
    """Turnover medio diario: ``mean(|w_t - w_{t-1}|)``."""
    w = weights.fillna(0).astype(float)
    return float(w.diff().abs().mean())


def summary(returns: pd.Series, weights: pd.Series | None = None) -> dict[str, float]:
    """Compendio de métricas en un único diccionario."""
    eq = equity_curve(returns)
    out = {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar(returns),
        "profit_factor": profit_factor(returns),
        "hit_rate": hit_rate(returns),
    }
    if weights is not None:
        out["turnover"] = turnover(weights)
    return out


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Métricas de clasificación direccional para la tabla maestra.

    Reúne en un dict ``accuracy``, ``auc``, ``log_loss``, ``brier`` y ``mcc``
    para que la tabla maestra muestre acierto y rentabilidad lado a lado y el
    desacople Sharpe↔accuracy no parezca contradictorio.

    ``y_true`` es la etiqueta binaria (1 = subida); ``y_score`` la probabilidad
    estimada de subida; ``y_pred`` la predicción binaria (si ``None``, se umbral
    en 0.5). Si solo hay una clase presente, ``auc`` y ``log_loss`` quedan en
    ``nan`` (no están definidos).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    single_class = len(np.unique(y_true)) < 2
    eps = 1e-15
    y_score_clip = np.clip(y_score, eps, 1 - eps)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "log_loss": (
            float("nan")
            if single_class
            else float(log_loss(y_true, y_score_clip, labels=[0, 1]))
        ),
        "brier": float(brier_score_loss(y_true, y_score_clip)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
