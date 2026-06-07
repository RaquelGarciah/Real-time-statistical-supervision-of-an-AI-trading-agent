"""Tests del módulo ``core.bocpd``."""

from __future__ import annotations

import numpy as np
import pytest

from core.bocpd import bocpd


def test_bocpd_detecta_cambio_de_varianza():
    """Un cambio claro en la varianza colapsa el MAP de longitud de run a 0.

    Pre-cambio: sigma=0.1 durante 200 pasos. Post-cambio: sigma=1.0 (×10).
    Tras el cambio, el filtro debe asignar baja longitud de run (el régimen
    es nuevo).
    """
    rng = np.random.default_rng(0)
    pre = rng.normal(0.0, 0.1, 200)
    post = rng.normal(0.0, 1.0, 200)
    x = np.concatenate([pre, post])
    res = bocpd(x, hazard=1 / 100)
    # Pre-cambio: el MAP debe crecer monótonamente hasta cerca de 199.
    assert res.map_run_length[199] >= 150
    # Post-cambio (al cabo de ~30 pasos): el MAP debe haber caído a valores bajos.
    assert res.map_run_length[230] < 50
    # cp_prob (masa en runs <= short_window) sube tras el cambio.
    assert res.cp_prob[230] > res.cp_prob[150]


def test_bocpd_serie_estable_map_crece_monotono():
    """En serie estable, el MAP de longitud de run debe acercarse a ``t``."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.1, 500)
    res = bocpd(x, hazard=1 / 250)
    # Tras el calentamiento, el MAP debe ser alto (cercano a la longitud total).
    assert res.map_run_length[-1] >= 400
    # Y la masa en runs cortos debe colapsar a casi cero pasado el warmup.
    assert res.cp_prob[100:].mean() < 0.1


def test_run_length_probs_suman_uno():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 100)
    res = bocpd(x)
    sumas = res.run_length_probs[1:].sum(axis=1)
    np.testing.assert_allclose(sumas, 1.0, atol=1e-9)


def test_bocpd_input_1d():
    with pytest.raises(ValueError):
        bocpd(np.zeros((10, 2)))


def test_bocpd_determinista():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    a = bocpd(x)
    b = bocpd(x)
    np.testing.assert_array_equal(a.cp_prob, b.cp_prob)
