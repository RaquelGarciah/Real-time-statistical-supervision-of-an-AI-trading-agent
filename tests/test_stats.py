"""Tests de ``core.stats``."""

from __future__ import annotations

import numpy as np
import pytest

from core.stats import (
    block_permutation_test,
    bootstrap_ci,
    deflated_sharpe,
    diebold_mariano,
    mcnemar_test,
    sign_test,
    tost,
)


def test_deflated_sharpe_bajo_dsr_cuando_n_trials_grande():
    """Probar muchas estrategias debería penalizar más el SR observado."""
    dsr_pocos = deflated_sharpe(sr_observed=0.05, n_trials=10, n_obs=252)
    dsr_muchos = deflated_sharpe(sr_observed=0.05, n_trials=1000, n_obs=252)
    assert dsr_muchos < dsr_pocos


def test_deflated_sharpe_alto_dsr_si_sr_grande():
    """Con SR muy alto, DSR -> 1 incluso con muchos trials."""
    dsr = deflated_sharpe(sr_observed=0.30, n_trials=1000, n_obs=2520)
    assert dsr > 0.95


def test_bootstrap_ci_contiene_estadistico_real():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.5, scale=1.0, size=500)
    low, high = bootstrap_ci(data, statistic=np.mean, n=2000, seed=1)
    assert low < 0.5 < high


def test_bootstrap_ci_input_1d():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros((10, 2)), statistic=np.mean)


def test_diebold_mariano_iguales_p_alto():
    """Si los dos modelos tienen las mismas pérdidas, el p-valor es alto."""
    rng = np.random.default_rng(0)
    loss1 = rng.normal(1.0, 0.1, 500)
    loss2 = loss1.copy()
    # Las pérdidas idénticas dan diferencia 0, var 0 -> NaN. Añadimos ruido mínimo.
    loss2 = loss1 + rng.normal(0, 0.001, 500)
    _, p = diebold_mariano(loss1, loss2)
    assert p > 0.05


def test_diebold_mariano_diferentes_p_bajo():
    """Si un modelo tiene sistemáticamente menos pérdida, p-valor < 0.05."""
    rng = np.random.default_rng(0)
    loss1 = rng.normal(1.0, 0.1, 500)
    loss2 = rng.normal(0.5, 0.1, 500)
    _, p = diebold_mariano(loss1, loss2)
    assert p < 0.05


def test_mcnemar_cuenta_discordancias():
    # A acierta donde B falla en 3 pares; B acierta donde A falla en 0.
    a = np.array([1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    b = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=bool)
    _, p, nb, nc = mcnemar_test(a, b)
    assert (nb, nc) == (3, 0)
    assert 0.0 <= p <= 1.0


def test_mcnemar_exacto_si_pocas_discordancias():
    """Con b+c pequeño usa el binomial exacto (estadístico chi2 = nan)."""
    a = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    b = np.array([0, 0, 0, 0, 0, 0], dtype=bool)
    stat, p, nb, nc = mcnemar_test(a, b)
    assert (nb, nc) == (2, 0)
    assert np.isnan(stat)  # rama exacta
    assert p == pytest.approx(0.5, abs=1e-9)  # binomial 2 de 2 a dos colas


def test_mcnemar_sin_discordancias_p_uno():
    a = np.array([1, 0, 1, 0], dtype=bool)
    _, p, nb, nc = mcnemar_test(a, a)
    assert (nb, nc) == (0, 0)
    assert p == 1.0


def test_sign_test_peor_que_azar():
    correct = np.array([1] * 163 + [0] * 237)  # 163/400 ≈ 0.41 < 0.5
    k, n, p, (lo, hi) = sign_test(correct)
    assert (k, n) == (163, 400)
    assert p < 0.001
    assert lo < 163 / 400 < hi


def test_sign_test_centrado_no_significativo():
    correct = np.array([1] * 200 + [0] * 200)
    _, _, p, _ = sign_test(correct)
    assert p > 0.5


def test_block_permutation_detecta_diferencia():
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.60).astype(float)  # 60% aciertos
    b = (rng.random(400) < 0.40).astype(float)  # 40% aciertos
    obs, p = block_permutation_test(a, b, seed=1)
    assert obs > 0
    assert p < 0.05


def test_tost_equivalencia():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.0, 0.01, 500)  # diferencia centrada y pequeña
    p, equiv = tost(diff, margin=0.02)
    assert equiv
    # Una diferencia grande no es equivalente dentro del mismo margen.
    diff_big = rng.normal(0.05, 0.01, 500)
    _, equiv_big = tost(diff_big, margin=0.02)
    assert not equiv_big
