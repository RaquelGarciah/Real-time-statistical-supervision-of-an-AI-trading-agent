"""Modelo GARCH(1,1) con innovaciones Student-t.

La especificación es la habitual de Bollerslev (1986) con distribución condicional
``t`` de Bollerslev (1987):

    r_t = mu + eps_t,    eps_t = sigma_t * z_t,    z_t ~ t_nu

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

Se ajustan ``mu, omega, alpha, beta, nu`` por máxima verosimilitud sobre el
periodo de calibración (delegado a la librería ``arch``). Para uso OOS, los
parámetros quedan congelados y la volatilidad se propaga por recursión.

Referencias:

- Engle (1982), "Autoregressive conditional heteroscedasticity...", *Econometrica 50*.
- Bollerslev (1986), "Generalized autoregressive conditional heteroscedasticity",
  *Journal of Econometrics 31*.
- Bollerslev (1987), "A conditionally heteroskedastic time series model for
  speculative prices and rates of return", *Review of Economics and Statistics 69*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model

ANN_FACTOR = np.sqrt(252)


@dataclass
class GARCHParams:
    """Parámetros ajustados del GARCH(1,1) Student-t."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float
    # Última varianza condicional y residuo del periodo de calibración,
    # necesarios para arrancar la recursión OOS.
    sigma2_last: float
    eps_last: float

    def is_stationary(self) -> bool:
        """Estacionariedad débil del proceso: ``alpha + beta < 1``."""
        return self.alpha + self.beta < 1.0


@dataclass
class GARCHModel:
    """GARCH(1,1) Student-t calibrado y propagado por recursión.

    Internamente la librería ``arch`` espera retornos en porcentaje
    (escala ×100) para mejorar la condicionalidad numérica; el wrapper
    se encarga de hacer la conversión transparentemente.
    """

    params: GARCHParams | None = None
    _scale: float = field(default=100.0, init=False, repr=False)

    def fit(self, returns: pd.Series) -> GARCHModel:
        """Ajusta el modelo por máxima verosimilitud sobre ``returns``.

        ``returns`` son retornos diarios en escala decimal (≈ 0.01, no 1.0).
        """
        r = returns.dropna() * self._scale
        am = arch_model(r, mean="constant", vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        p = res.params
        cond_vol = res.conditional_volatility
        eps_last = float(r.iloc[-1] - p["mu"])
        sigma2_last = float(cond_vol.iloc[-1] ** 2)
        self.params = GARCHParams(
            mu=float(p["mu"]),
            omega=float(p["omega"]),
            alpha=float(p["alpha[1]"]),
            beta=float(p["beta[1]"]),
            nu=float(p["nu"]),
            sigma2_last=sigma2_last,
            eps_last=eps_last,
        )
        return self

    def forecast_path(self, returns_oos: pd.Series) -> pd.Series:
        """Propaga ``sigma_t`` con parámetros congelados sobre ``returns_oos``.

        Devuelve la **volatilidad anualizada** ``sigma_t * sqrt(252)`` en escala
        decimal (para alinear con la convención del resto del proyecto).
        """
        if self.params is None:
            raise RuntimeError("Llama a fit() antes de propagar.")
        p = self.params
        r = (returns_oos * self._scale).to_numpy()

        sigma2 = np.empty(len(r))
        eps_prev = p.eps_last
        sigma2_prev = p.sigma2_last
        for t, r_t in enumerate(r):
            sigma2[t] = p.omega + p.alpha * eps_prev**2 + p.beta * sigma2_prev
            eps_prev = r_t - p.mu
            sigma2_prev = sigma2[t]

        sigma_decimal = np.sqrt(sigma2) / self._scale
        return pd.Series(sigma_decimal * ANN_FACTOR, index=returns_oos.index, name="sigma_garch")
