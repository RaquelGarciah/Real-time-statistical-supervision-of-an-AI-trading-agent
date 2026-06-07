"""Tests del orquestador ``strata.strata``."""

from __future__ import annotations

from strata.strata import StrataSupervisor
from strata.types import AgentOutput


def _agent(size: float = 0.7) -> AgentOutput:
    return AgentOutput(
        date="2024-10-15", ticker="SPY", action="long", size=size, confidence=0.8
    )


def _market(state: str = "Calma", sigma: float = 0.10) -> dict:
    probs = {"Calma": 0.0, "Estrés": 0.0, "Crisis": 0.0}
    probs[state] = 1.0
    return {
        "regime": {
            "calm_prob": probs["Calma"],
            "stress_prob": probs["Estrés"],
            "crisis_prob": probs["Crisis"],
            "viterbi_state": state,
        },
        "garch_vol_annualized": sigma,
    }


def test_warn_pasa_agente_intacto_si_consistente():
    sup = StrataSupervisor(mode="warn")
    s = sup.supervise(_agent(0.5), _market("Calma", 0.10), sizing_history=[0.5] * 20)
    assert s.final_size == 0.5
    assert s.final_action == "long"


def test_override_en_crisis_anula_long_alto():
    """Crisis activa RAM=high, GSO restringe; override sustituye size."""
    sup = StrataSupervisor(mode="override")
    s = sup.supervise(
        _agent(size=0.8),
        _market("Crisis", sigma=0.40),
        sizing_history=[0.8] * 20,
    )
    assert s.was_intervened is True
    assert abs(s.final_size) < 0.5  # GSO bound = 0.10/0.40 = 0.25


def test_reduce_atenuacion_proporcional_severidad():
    sup = StrataSupervisor(mode="reduce")
    s = sup.supervise(
        _agent(size=0.5),
        _market("Crisis", sigma=0.10),  # GSO sí limita pero score depende del exceso
        sizing_history=[0.5] * 20,
    )
    # En Crisis, RAM dispara alto → reduce a 0.
    assert s.was_intervened
    assert abs(s.final_size) <= 0.5


def test_ablacion_desactiva_detector():
    sup = StrataSupervisor(mode="warn", enabled={"ram": False, "psa": True, "gso": True})
    s = sup.supervise(
        _agent(size=0.5),
        _market("Crisis", sigma=0.10),
        sizing_history=[0.5] * 20,
    )
    assert "ram" not in s.detectors
    assert "psa" in s.detectors
    assert "gso" in s.detectors


def test_historial_corto_psa_emite_none():
    sup = StrataSupervisor(mode="warn")
    s = sup.supervise(_agent(0.5), _market("Calma"), sizing_history=[0.5])
    assert s.detectors["psa"].severity == "none"
