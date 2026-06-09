"""Calibración robusta del gate direccional τ de RAM.

El detector RAM dispara el override cuando la confianza del régimen en su dirección
dominante supera un umbral τ. τ se calibra **ex-ante** (solo con la calibración, sin OOS
ni P&L) como el punto donde el régimen pasa de ruido (acierto direccional <0.5) a
informativo (≥0.5).

El estimador ingenuo "primer punto donde la curva isotónica cruza 0.5" es **frágil**: es un
funcional discontinuo de la curva estimada (saltos ante perturbaciones pequeñas) y degenera
a τ=0 cuando la curva ya arranca ≥0.5 (no cruza por abajo). Aquí se usa un estimador robusto:

1. **Curva de fiabilidad por regresión logística monótona** ``P(acierto | c) = σ(a + b·c)``
   (suave; el cruce de 0.5 es ``τ = -a/b``, continuo y diferenciable).
2. **Identificabilidad explícita**: si ``b ≤ 0`` (el régimen no se vuelve más fiable con la
   confianza) o el cruce cae fuera de ``(0,1)``, el gate **no está identificado** → la capa de
   intervención debe **abstenerse** (no τ=0 forzado). Conecta con la regla ``prior-flip``.
3. **Bootstrap estacionario por días** (Politis-Romano 1994) → se reporta la **mediana** de τ
   (robusta al estimador puntual) y su IC95, además de la fracción de réplicas identificadas.

Referencias: Politis & Romano (1994); el cruce logístico es el estándar para localizar el
punto de indiferencia de una curva de calibración monótona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from config import SEED


@dataclass
class GateResult:
    """Resultado de la calibración del gate τ.

    - ``tau``: estimador robusto (mediana bootstrap si está identificado, si no ``nan``).
    - ``tau_point``: cruce logístico sobre la muestra completa.
    - ``ci``: IC95 ``(low, high)`` de τ por bootstrap (``nan`` si no identificado).
    - ``identified``: ``True`` si el régimen es direccionalmente informativo (``b>0`` y cruce en (0,1)).
    - ``frac_identified``: fracción de réplicas bootstrap con gate identificado.
    - ``slope``: pendiente logística ``b`` (signo positivo = más confianza ⇒ más acierto).
    """

    tau: float
    tau_point: float
    ci: tuple[float, float]
    identified: bool
    frac_identified: float
    slope: float


def directional_reliability(calm: np.ndarray, crisis: np.ndarray, r_next: np.ndarray):
    """Confianza del régimen en su dirección dominante y si acierta el signo de ``r_next``.

    Dirección dominante = long si ``P(Calma) ≥ P(Crisis)``, short si no. Confianza
    ``c = máx(P(Calma), P(Crisis))``. Es la apuesta que hace el override-C al disparar.
    """
    conf = np.maximum(calm, crisis)
    correct = np.where(calm >= crisis, r_next > 0, r_next < 0).astype(float)
    return conf, correct


def _logistic_cross(conf: np.ndarray, correct: np.ndarray) -> tuple[float, float, bool]:
    """Cruce de 0.5 de la logística ``P(correct|conf)``. Devuelve ``(tau, slope, identified)``."""
    if len(np.unique(correct)) < 2:
        return float("nan"), 0.0, False
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(conf.reshape(-1, 1), correct)
    b = float(lr.coef_[0, 0])
    a = float(lr.intercept_[0])
    if b <= 0:  # el régimen no se vuelve más fiable con la confianza → no identificable
        return float("nan"), b, False
    tau = -a / b
    if not (0.0 < tau < 1.0):  # el cruce cae fuera del rango de confianza observable
        return (tau, b, False)
    return tau, b, True


def calibrate_gate(
    calm: np.ndarray,
    crisis: np.ndarray,
    r_next: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> GateResult:
    """Calibra τ de forma robusta sobre datos de calibración (ver docstring del módulo)."""
    calm = np.asarray(calm, float); crisis = np.asarray(crisis, float); r_next = np.asarray(r_next, float)
    conf, correct = directional_reliability(calm, crisis, r_next)
    tau_point, slope, identified = _logistic_cross(conf, correct)

    n = len(conf)
    bl = max(2, int(round(np.sqrt(n)))); pr = 1.0 / bl
    rng = np.random.default_rng(seed)
    taus = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jmp = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jmp[t - 1] if u[t - 1] < pr else (idx[t - 1] + 1) % n
        taus[i], _, ident = _logistic_cross(conf[idx], correct[idx])
        if not ident:
            taus[i] = np.nan
    frac_ident = float(np.isfinite(taus).mean())
    if identified and frac_ident > 0:
        tau = float(np.nanmedian(taus))
        lo, hi = (float(x) for x in np.nanpercentile(taus, [2.5, 97.5]))
    else:
        tau, lo, hi = float("nan"), float("nan"), float("nan")
    return GateResult(tau=tau, tau_point=tau_point, ci=(lo, hi),
                      identified=identified, frac_identified=frac_ident, slope=slope)
