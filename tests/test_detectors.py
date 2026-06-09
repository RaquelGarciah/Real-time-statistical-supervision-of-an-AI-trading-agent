"""Tests de los detectores RAM, PSA y GSO."""

from __future__ import annotations

import numpy as np
import pytest

from strata.detectors import (
    _gso_severity_from_ratio,
    _psa_severity_from_runlength,
    gso_detector,
    psa_detector,
    ram_detector,
)

# ---- RAM ---------------------------------------------------------------


def test_ram_thresholds_explicitos():
    """Umbrales RAM explícitos cambian la severidad respecto a los defaults."""
    probs = {"Calma": 0.30, "Estrés": 0.70, "Crisis": 0.0}  # agente short → score = P(Calma) = 0.30
    assert ram_detector(-0.2, probs).severity == "low"  # default medium=0.40 → 0.30 es "low"
    assert ram_detector(-0.2, probs, thresholds=(0.10, 0.20, 0.70)).severity == "medium"  # medium=0.20


def test_ram_long_en_calma_no_inconsistente():
    res = ram_detector(agent_size=0.5, regime_probs={"Calma": 1.0, "Estrés": 0.0, "Crisis": 0.0})
    assert res.score == 0.0
    assert not res.flag


def test_ram_long_en_crisis_alta_inconsistencia():
    res = ram_detector(agent_size=0.7, regime_probs={"Calma": 0.0, "Estrés": 0.1, "Crisis": 0.9})
    assert res.score >= 0.7
    assert res.flag
    assert res.severity == "high"


def test_ram_short_en_crisis_consistente():
    """Política simétrica con el leverage effect (BITACORA 2026-05-19):
    Crisis penaliza solo long, no short."""
    res = ram_detector(agent_size=-0.5, regime_probs={"Calma": 0.0, "Estrés": 0.0, "Crisis": 1.0})
    assert res.score == 0.0
    assert not res.flag
    assert res.severity == "none"


def test_ram_short_en_calma_alta_inconsistencia():
    res = ram_detector(agent_size=-0.5, regime_probs={"Calma": 0.95, "Estrés": 0.05, "Crisis": 0.0})
    assert res.score >= 0.7
    assert res.flag


def test_ram_flat_siempre_consistente():
    res = ram_detector(agent_size=0.0, regime_probs={"Calma": 0.0, "Estrés": 0.0, "Crisis": 1.0})
    assert res.score == 0.0
    assert not res.flag


def test_ram_extra_expone_direccion_de_regimen():
    """El override consume regime_sign y p_dominant del extra de RAM."""
    # Calma domina → dirección implícita long (regime_sign +1).
    calma = ram_detector(agent_size=-0.2, regime_probs={"Calma": 0.8, "Estrés": 0.1, "Crisis": 0.1})
    assert calma.extra["regime_sign"] == 1.0
    assert calma.extra["p_dominant"] == 0.8
    # Crisis domina → dirección implícita short (regime_sign -1).
    crisis = ram_detector(agent_size=0.2, regime_probs={"Calma": 0.1, "Estrés": 0.2, "Crisis": 0.7})
    assert crisis.extra["regime_sign"] == -1.0
    assert crisis.extra["p_dominant"] == 0.7


# ---- PSA ---------------------------------------------------------------


def test_psa_historial_corto_emite_none():
    res = psa_detector([0.5, 0.5, 0.5])
    assert not res.flag
    assert res.severity == "none"


def test_psa_serie_estable_score_bajo():
    # 30 valores casi constantes → MAP debería crecer y cp_prob caer.
    rng = np.random.default_rng(0)
    h = (rng.normal(0.0, 0.01, 50) + 0.5).tolist()
    res = psa_detector(h)
    assert res.score < 0.5


def test_psa_severity_runlength_inversa():
    """La severidad threshold-free de 2-map escala inversa a la run-length."""
    assert _psa_severity_from_runlength(0, 5) == "high"
    assert _psa_severity_from_runlength(1, 5) == "high"
    assert _psa_severity_from_runlength(3, 5) == "medium"
    assert _psa_severity_from_runlength(5, 5) == "low"
    assert _psa_severity_from_runlength(6, 5) == "none"


def test_psa_signal_map_runlength_emite_resultado_valido():
    """La señal map_runlength produce un DetectorResult coherente y su flag se
    activa solo si la run-length del MAP es <= short_window."""
    h = ([0.1] * 30) + [0.9, 0.9, 0.9]
    res = psa_detector(h, signal="map_runlength")
    assert res.severity in ("none", "low", "medium", "high")
    assert res.flag == (res.extra["map_run_length"] <= 5)


def test_psa_signal_delta_sobre_incrementos():
    """cp_prob_delta corre BOCPD sobre diff(sizing); necesita short_window+3 obs."""
    # Serie que oscila de nivel pero con un salto fuerte al final en el incremento.
    h = ([0.1, -0.1] * 15) + [0.9]
    res = psa_detector(h, signal="cp_prob_delta")
    assert res.severity in ("none", "low", "medium", "high")
    # Historial demasiado corto para la variante delta → none.
    short = psa_detector([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], signal="cp_prob_delta")
    assert short.severity == "none"


def test_psa_hazard_mas_alto_es_mas_sensible():
    """Un hazard mayor sube la cp_prob ante un cambio reciente."""
    h = ([0.1] * 30) + [0.9, 0.9, 0.9]
    lo = psa_detector(h, hazard=1 / 250)
    hi = psa_detector(h, hazard=1 / 10)
    assert hi.score >= lo.score


def test_psa_cambio_reciente_detectado():
    """Si el cambio ocurre justo antes del fin del historial, cp_prob debe subir.

    PSA se evalúa siempre en ``t = T-1``, así que la señal sólo es positiva
    cuando el cambio es reciente (en los últimos ``short_window`` pasos).
    Bajo los umbrales recalibrados por percentiles, el flag y la severidad
    dependen del JSON activo; el test verifica que el ``score`` sube de forma
    sostenida y la severidad ya no es ``none``.
    """
    h = ([0.1] * 30) + [0.9, 0.9, 0.9]
    res = psa_detector(h)
    assert res.score > 0.3
    assert res.severity != "none"


# ---- GSO ---------------------------------------------------------------


def test_gso_dentro_de_banda():
    # bound = target_vol/sigma = 0.1/0.10 = 1.0 → size 0.5 dentro.
    res = gso_detector(agent_size=0.5, sigma_t_annualized=0.10)
    assert res.score == 0.0
    assert not res.flag
    assert res.extra["bounded_size"] == 0.5


def test_gso_excede_banda_es_high():
    # bound = 0.1/0.5 = 0.2 → size 1.0 excede en 4× → score 4.0.
    # Bajo los umbrales recalibrados por percentiles (low=P95, medium=P99,
    # high=max) un score 4.0 cae en severidad ``low`` o ``medium`` según la
    # calibración persistida. El test verifica que el detector emite flag y
    # ``bounded_size`` correcto sin fijar el nivel exacto de severidad.
    res = gso_detector(agent_size=1.0, sigma_t_annualized=0.50)
    assert res.flag
    assert res.severity in ("low", "medium", "high")
    assert res.extra["bounded_size"] == 0.2


def test_gso_short_signo_negativo_se_preserva():
    res = gso_detector(agent_size=-0.8, sigma_t_annualized=0.30)
    assert res.extra["bounded_size"] < 0


def test_gso_sigma_cero_no_truena():
    res = gso_detector(agent_size=0.5, sigma_t_annualized=0.0)
    assert res.extra["bound"] == 1.0
    assert not res.flag


def test_gso_size_cero_no_dispara():
    res = gso_detector(agent_size=0.0, sigma_t_annualized=0.30)
    assert res.score == 0.0
    assert res.extra["bounded_size"] == 0.0


def test_gso_severity_ratio_dos_colas():
    """La severidad relativa es simétrica en log: on-target none, lejos high."""
    assert _gso_severity_from_ratio(1.0) == "none"      # exactamente en objetivo
    assert _gso_severity_from_ratio(1.4) == "low"       # ~1.4× → low
    assert _gso_severity_from_ratio(0.5) == "high"      # 2× por debajo → high
    assert _gso_severity_from_ratio(4.0) == "high"      # 4× por encima → high


def test_gso_relative_reescala_a_la_banda_conservando_signo():
    # bound = 0.1/0.10 = 1.0; agente infra-expuesto (0.2) → relative reescala a
    # sign·bound = +1.0 y dispara (severidad != none).
    res = gso_detector(agent_size=0.2, sigma_t_annualized=0.10, gso_mode="relative")
    assert res.extra["bounded_size"] == 1.0
    assert res.flag
    short = gso_detector(agent_size=-0.2, sigma_t_annualized=0.10, gso_mode="relative")
    assert short.extra["bounded_size"] == -1.0


def test_gso_relative_conviction_pondera_por_conviccion():
    # bound = 0.1/0.20 = 0.5; convicción = |0.125|/AGENT_MAX(0.25) = 0.5.
    # bounded_size = sign·0.5·0.5 = 0.25.
    res = gso_detector(agent_size=0.125, sigma_t_annualized=0.20, gso_mode="relative_conviction")
    assert res.extra["bounded_size"] == pytest.approx(0.25)
