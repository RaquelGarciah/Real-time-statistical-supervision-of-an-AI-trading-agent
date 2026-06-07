"""Tests del módulo ``core.hmm``."""

from __future__ import annotations

import numpy as np
import pytest

from core.hmm import RegimeHMM


def _sim_3regimenes(seed: int = 0) -> np.ndarray:
    """Genera observaciones (ret_log, log_vix) bajo 3 regímenes diferenciados."""
    rng = np.random.default_rng(seed)
    n_per = 400
    calma = np.column_stack(
        [
            rng.normal(0.0005, 0.005, n_per),
            rng.normal(np.log(13), 0.05, n_per),
        ]
    )
    estres = np.column_stack(
        [
            rng.normal(0.0001, 0.015, n_per),
            rng.normal(np.log(20), 0.10, n_per),
        ]
    )
    crisis = np.column_stack(
        [
            rng.normal(-0.002, 0.035, n_per),
            rng.normal(np.log(40), 0.15, n_per),
        ]
    )
    return np.vstack([calma, estres, crisis, calma, crisis])


def test_fit_y_etiquetas_deterministas():
    X = _sim_3regimenes()
    a = RegimeHMM().fit(X)
    b = RegimeHMM().fit(X)
    # Dos ajustes sobre la misma entrada deben coincidir salvo ruido numérico de
    # coma flotante (~1e-15 en entradas casi nulas, dependiente del orden BLAS): el
    # determinismo relevante es el etiquetado y la matriz hasta tolerancia, no el bit.
    np.testing.assert_allclose(a.transition_matrix, b.transition_matrix, atol=1e-9)
    assert a.state_labels == {0: "Calma", 1: "Estrés", 2: "Crisis"}


def test_filas_transmat_suman_uno():
    X = _sim_3regimenes()
    h = RegimeHMM().fit(X)
    np.testing.assert_allclose(h.transition_matrix.sum(axis=1), 1.0)


def test_estados_predichos_reconocen_regimen():
    """En el bloque de calma, el estado mayoritario debe ser 0 (Calma)."""
    X = _sim_3regimenes()
    h = RegimeHMM().fit(X)
    estados = h.predict_states(X)
    bloque_calma = estados[:400]
    # tolerancia: al menos 70% del bloque debe estar en estado 0
    proporcion = (bloque_calma == 0).mean()
    assert proporcion > 0.7


def test_predict_proba_suma_uno():
    X = _sim_3regimenes()
    h = RegimeHMM().fit(X)
    p = h.predict_proba(X)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_fit_rechaza_input_1d():
    with pytest.raises(ValueError):
        RegimeHMM().fit(np.array([0.1, 0.2, 0.3]))


def test_predict_sin_fit_levanta():
    with pytest.raises(RuntimeError):
        RegimeHMM().predict_states(np.zeros((10, 2)))
