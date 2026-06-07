"""Orquestador de STRATA: combina los tres detectores y la capa de intervención.

``StrataSupervisor.supervise(agent, market_state, sizing_history)`` recoge en
un solo paso:

1. Ejecutar RAM con el régimen HMM observado.
2. Ejecutar PSA con el historial de sizing del propio agente.
3. Ejecutar GSO con la σ_t GARCH observada.
4. Aplicar la capa de intervención (``warn`` / ``reduce`` / ``override``).

El estado de mercado se pasa como diccionario simple para mantener desacoplado
el orquestador de los detalles internos del HMM/GARCH.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from strata.detectors import gso_detector, psa_detector, ram_detector
from strata.intervention import supervise
from strata.types import AgentOutput, InterventionMode, SupervisedDecision


@dataclass
class StrataSupervisor:
    """Supervisor estadístico configurado con un modo de intervención y opciones.

    ``enabled`` permite activar/desactivar detectores para el experimento de
    ablación (E5). Por defecto los tres están activos.
    """

    mode: InterventionMode = "warn"
    enabled: dict[str, bool] | None = None
    override_variant: str = "A"
    psa_signal: str = "cp_prob"
    psa_hazard: float = 1 / 250
    gso_mode: str = "absolute"
    reduce_mode: str = "bucket"

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = {"ram": True, "psa": True, "gso": True}

    def supervise(
        self,
        agent: AgentOutput,
        market_state: dict,
        sizing_history: Sequence[float],
    ) -> SupervisedDecision:
        """Ejecuta detectores activos + intervención sobre ``agent``.

        Args:
            agent: salida del agente para hoy.
            market_state: diccionario con ``regime`` (``{calm_prob, stress_prob,
                crisis_prob, viterbi_state}``) y ``garch_vol_annualized``.
            sizing_history: serie con el sizing histórico del agente, incluyendo
                el del día actual al final (para que PSA detecte cambios al
                instante).
        """
        regime = market_state.get("regime", {})
        regime_probs = {
            "Calma": float(regime.get("calm_prob", 0.0)),
            "Estrés": float(regime.get("stress_prob", 0.0)),
            "Crisis": float(regime.get("crisis_prob", 0.0)),
        }
        sigma = float(market_state.get("garch_vol_annualized", 0.0))

        detectors = {}
        if self.enabled["ram"]:
            detectors["ram"] = ram_detector(agent.size, regime_probs)
        if self.enabled["psa"]:
            detectors["psa"] = psa_detector(
                sizing_history, hazard=self.psa_hazard, signal=self.psa_signal
            )
        if self.enabled["gso"]:
            detectors["gso"] = gso_detector(agent.size, sigma, gso_mode=self.gso_mode)

        # GSO debe estar disponible en modo override aunque esté "desactivado".
        if self.mode == "override" and "gso" not in detectors:
            detectors["gso"] = gso_detector(agent.size, sigma, gso_mode=self.gso_mode)

        return supervise(
            agent, detectors, mode=self.mode,
            override_variant=self.override_variant, reduce_mode=self.reduce_mode,
        )
