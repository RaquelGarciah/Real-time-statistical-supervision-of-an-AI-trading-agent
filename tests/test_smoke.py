"""Smoke test mínimo: comprueba que ``config`` se importa y que las semillas se fijan.

Sirve para que el workflow de CI tenga al menos un test verde durante la fase de
andamiaje. Se elimina cuando existan tests reales para cada módulo.
"""

import numpy as np

import config


def test_set_seeds_es_determinista() -> None:
    config.set_seeds(123)
    a = np.random.rand(4)
    config.set_seeds(123)
    b = np.random.rand(4)
    assert np.allclose(a, b)


def test_constantes_basicas() -> None:
    assert config.HMM_N_STATES == 3
    # La calibración termina antes del OOS unificado (sin solape temporal).
    assert config.CALIBRATION_END < config.STRATA_OOS_START
    assert config.TICKER_PRIMARY == "SPY"
