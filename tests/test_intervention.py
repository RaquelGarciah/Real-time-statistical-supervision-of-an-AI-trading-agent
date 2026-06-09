"""Tests de la capa de intervención."""

from __future__ import annotations

import pytest

from strata.intervention import supervise
from strata.types import AgentOutput, DetectorResult


def _agent(size: float = 0.7) -> AgentOutput:
    return AgentOutput(
        date="2024-10-15", ticker="SPY", action="long", size=size, confidence=0.8
    )


def _detectors(ram="none", psa="none", gso_sev="none", bounded=0.0) -> dict:
    return {
        "ram": DetectorResult(name="ram", score=0.0, flag=False, severity=ram),
        "psa": DetectorResult(name="psa", score=0.0, flag=False, severity=psa),
        "gso": DetectorResult(
            name="gso",
            score=0.0,
            flag=False,
            severity=gso_sev,
            extra={"bounded_size": bounded, "bound": abs(bounded)},
        ),
    }


def test_warn_no_modifica_size():
    a = _agent(0.7)
    s = supervise(a, _detectors(), "warn")
    assert s.final_size == 0.7
    assert s.final_action == "long"
    assert s.was_intervened is False


def test_reduce_severidad_baja_atenua_25pct():
    a = _agent(0.8)
    s = supervise(a, _detectors(ram="low"), "reduce")
    assert s.final_size == pytest.approx(0.8 * 0.75)
    assert s.was_intervened is True


def test_reduce_severidad_alta_anula_size():
    a = _agent(0.8)
    s = supervise(a, _detectors(gso_sev="high"), "reduce")
    assert s.final_size == pytest.approx(0.0)
    assert s.final_action == "hold"


def test_reduce_continuo_atenua_proporcional_al_score():
    """reduce_mode='continuous' atenúa por 1 - max(score), no por buckets."""
    a = _agent(0.8)
    dets = {
        "ram": DetectorResult(name="ram", score=0.3, flag=False, severity="low"),
        "psa": DetectorResult(name="psa", score=0.0, flag=False, severity="none"),
        "gso": DetectorResult(name="gso", score=0.0, flag=False, severity="none",
                              extra={"bounded_size": 0.0, "bound": 1.0}),
    }
    s = supervise(a, dets, "reduce", reduce_mode="continuous")
    # factor = 1 - 0.3 = 0.7 → 0.8 * 0.7 = 0.56 (frente a 0.8*0.75=0.6 del bucket low).
    assert s.final_size == pytest.approx(0.56)
    assert s.was_intervened is True


def test_reduce_ram_continuous_gated_atenua_solo_si_ram_dispara():
    """reduce_mode='ram_continuous' (M7) atenúa por 1-RAM solo si RAM es medium/high."""
    a = _agent(0.8)
    # RAM en 'low' (score 0.30 < τ): NO interviene, deja el size del agente intacto.
    dets_low = {
        "ram": DetectorResult(name="ram", score=0.30, flag=False, severity="low"),
        "psa": DetectorResult(name="psa", score=0.0, flag=False, severity="none"),
        "gso": DetectorResult(name="gso", score=0.0, flag=False, severity="none",
                              extra={"bounded_size": 0.0, "bound": 1.0}),
    }
    s_low = supervise(a, dets_low, "reduce", reduce_mode="ram_continuous")
    assert s_low.final_size == pytest.approx(0.8)
    assert s_low.was_intervened is False
    # RAM en 'medium' (score 0.55 ≥ τ): atenúa por 1-0.55=0.45 → 0.8*0.45=0.36.
    dets_med = dict(dets_low)
    dets_med["ram"] = DetectorResult(name="ram", score=0.55, flag=True, severity="medium")
    s_med = supervise(a, dets_med, "reduce", reduce_mode="ram_continuous")
    assert s_med.final_size == pytest.approx(0.8 * 0.45)
    assert s_med.was_intervened is True


def test_override_no_actua_si_solo_severidad_low():
    a = _agent(0.9)
    s = supervise(a, _detectors(ram="low", bounded=0.2), "override")
    assert s.final_size == 0.9
    assert s.was_intervened is False


def test_override_sustituye_size_por_bounded_si_medium():
    a = _agent(0.9)
    s = supervise(a, _detectors(gso_sev="medium", bounded=0.3), "override")
    assert s.final_size == 0.3
    assert s.was_intervened is True


def test_override_ram_high_anula_position():
    """Con RAM high, el override anula la posición aunque GSO sugiera un bound."""
    a = _agent(0.9)
    s = supervise(a, _detectors(ram="high", gso_sev="high", bounded=0.15), "override")
    assert s.final_size == 0.0
    assert s.was_intervened is True


def test_override_gso_solo_usa_bounded():
    """Si solo GSO marca medium/high (RAM/PSA limpias), override = bounded_size."""
    a = _agent(0.9)
    s = supervise(a, _detectors(gso_sev="high", bounded=0.2), "override")
    assert s.final_size == 0.2
    assert s.was_intervened is True


def _detectors_regime(ram="high", regime_sign=1.0, p_dom=0.8, bound=0.6) -> dict:
    """RAM con dirección de régimen en el extra (para variantes B/C/D)."""
    return {
        "ram": DetectorResult(
            name="ram", score=0.9, flag=ram in ("medium", "high"), severity=ram,
            extra={"regime_sign": regime_sign, "p_dominant": p_dom},
        ),
        "psa": DetectorResult(name="psa", score=0.0, flag=False, severity="none"),
        "gso": DetectorResult(
            name="gso", score=0.0, flag=False, severity="none",
            extra={"bound": bound, "bounded_size": 0.0},
        ),
    }


def test_override_variante_c_invierte_a_sizing_garch():
    # Agente short en régimen alcista; C = regime_sign(+1) * bound(0.6).
    a = AgentOutput(date="2025-01-02", ticker="SPY", action="short", size=-0.2, confidence=0.7)
    s = supervise(a, _detectors_regime(regime_sign=1.0, bound=0.6), "override", override_variant="C")
    assert s.final_size == pytest.approx(0.6)
    assert s.final_action == "long"


def test_override_variante_d_conserva_magnitud():
    # D = regime_sign(+1) * |size|(0.2): corrige dirección, mantiene escala.
    a = AgentOutput(date="2025-01-02", ticker="SPY", action="short", size=-0.2, confidence=0.7)
    s = supervise(a, _detectors_regime(regime_sign=1.0), "override", override_variant="D")
    assert s.final_size == pytest.approx(0.2)


def test_override_variante_b_inversion_parcial():
    # B = 0.5 * regime_sign(+1) * bound(0.6) * p_dom(0.8) = 0.24.
    a = AgentOutput(date="2025-01-02", ticker="SPY", action="short", size=-0.2, confidence=0.7)
    s = supervise(a, _detectors_regime(bound=0.6, p_dom=0.8), "override", override_variant="B")
    assert s.final_size == pytest.approx(0.24)


def test_modo_invalido_levanta():
    with pytest.raises(ValueError):
        supervise(_agent(), _detectors(), "doesnotexist")  # type: ignore[arg-type]
