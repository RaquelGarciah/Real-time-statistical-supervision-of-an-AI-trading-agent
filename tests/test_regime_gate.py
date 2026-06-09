"""Tests del gate direccional robusto τ (core.regime_gate)."""

from __future__ import annotations

import numpy as np

from core.regime_gate import calibrate_gate, directional_reliability


def _make(p_func, n=4000, seed=0):
    # Genera (calm, crisis, r_next) con confianza c=máx(calm,crisis) en (0.05,0.95), dirección
    # dominante = argmax(calm,crisis), y P(acierto direccional | c) = p_func(c). El estado no
    # dominante queda por debajo (el resto de masa sería "estrés"), como en el HMM de 3 estados.
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.05, 0.95, n)
    long_call = rng.random(n) < 0.5
    calm = np.where(long_call, conf, conf * 0.3)
    crisis = np.where(long_call, conf * 0.3, conf)
    correct = rng.random(n) < p_func(conf)
    sign = np.where(long_call, 1.0, -1.0) * np.where(correct, 1.0, -1.0)
    r_next = sign * 0.01
    return calm, crisis, r_next


def test_directional_reliability_basico():
    calm = np.array([0.8, 0.2]); crisis = np.array([0.1, 0.7]); rn = np.array([0.01, -0.01])
    conf, correct = directional_reliability(calm, crisis, rn)
    assert np.allclose(conf, [0.8, 0.7])
    assert np.allclose(correct, [1.0, 1.0])  # día1 long acierta subida; día2 short acierta bajada


def test_gate_identificado_cruza_en_el_punto_esperado():
    # P(acierto) = 0.30 + 0.5*conf cruza 0.5 en conf=0.40.
    calm, crisis, rn = _make(lambda c: 0.30 + 0.5 * c, seed=1)
    res = calibrate_gate(calm, crisis, rn, n_boot=200, seed=1)
    assert res.identified
    assert res.slope > 0
    assert 0.30 < res.tau < 0.50           # cerca de 0.40
    assert res.ci[0] < res.tau < res.ci[1]


def test_gate_no_identificable_si_regimen_es_ruido():
    # Acierto independiente de la confianza (≈0.5): pendiente ~0 → no identificable.
    calm, crisis, rn = _make(lambda c: np.full_like(c, 0.5), seed=2)
    res = calibrate_gate(calm, crisis, rn, n_boot=200, seed=2)
    assert not res.identified
    assert np.isnan(res.tau)               # abstención, NO τ=0


def test_gate_no_identificable_si_curva_arranca_sobre_05():
    # Acierto ≥0.5 en todo el rango (0.62 + 0.3*conf): el cruce cae <0 → no identificable.
    calm, crisis, rn = _make(lambda c: 0.62 + 0.3 * c, seed=3)
    res = calibrate_gate(calm, crisis, rn, n_boot=200, seed=3)
    assert not res.identified
    assert np.isnan(res.tau)               # NO degenera a τ=0
