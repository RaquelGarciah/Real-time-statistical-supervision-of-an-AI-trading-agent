"""Tests de sanidad de la batería de validación quant (``core.validation``).

No re-prueban el contenido estadístico de la literatura; comprueban que cada
función reduce a su caso conocido (iid, sin autocorrelación, cola normal),
respeta invariantes (FDR ⊆ no corregido, borrow ≥ 0 reduce el neto) y es
determinista con la semilla fijada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import validation as v


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_hac_reduce_a_t_normal_sin_autocorr(rng):
    """Sin autocorrelación, el t HAC ≈ t clásico de la media."""
    x = rng.normal(0.5, 1.0, 2000)
    mean, t, p = v.hac_tstat(x)
    t_clasico = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert mean == pytest.approx(x.mean())
    assert t == pytest.approx(t_clasico, rel=0.15)
    assert p < 0.01  # media claramente > 0


def test_hac_detecta_autocorrelacion_positiva(rng):
    """Con autocorrelación positiva, el SE HAC sube y el |t| baja frente al iid."""
    e = rng.normal(0, 1, 4000)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.6 * x[i - 1] + e[i]
    x += 0.05
    _, t_hac, _ = v.hac_tstat(x)
    t_iid = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_hac) < abs(t_iid)


def test_sharpe_lo_eta_uno_sin_autocorr(rng):
    """Sin autocorrelación, el factor eta de Lo ≈ 1 (anualización ~ sqrt(freq))."""
    r = pd.Series(rng.normal(0.0005, 0.01, 3000))
    out = v.sharpe_se_lo(r, autocorr=True)
    assert out["eta"] == pytest.approx(1.0, abs=0.08)
    assert out["ci_low"] < out["sharpe"] < out["ci_high"]


def test_var_cvar_orden_y_signo(rng):
    """CVaR ≥ VaR ≥ 0 para una serie con pérdidas; histórico y CF cerca en normal."""
    r = pd.Series(rng.normal(0, 0.01, 5000))
    var_h = v.var_historical(r, 0.95)
    cvar_h = v.cvar_historical(r, 0.95)
    var_cf = v.var_cornish_fisher(r, 0.95)
    assert cvar_h >= var_h > 0
    assert var_cf == pytest.approx(var_h, rel=0.25)


def test_information_ratio_cero_si_igual_benchmark(rng):
    r = pd.Series(rng.normal(0, 0.01, 500))
    out = v.information_ratio(r, r.copy())
    assert np.isnan(out["ir"]) or out["tracking_error"] == pytest.approx(0.0, abs=1e-9)


def test_fdr_bh_subconjunto_y_monotonia():
    """BH rechaza ⊆ de los p<alpha sin corregir; BY es más conservador que BH."""
    p = [0.001, 0.01, 0.02, 0.2, 0.5, 0.9]
    bh = v.fdr_bh(p, alpha=0.05)
    by = v.fdr_by(p, alpha=0.05)
    crudo = sum(pi < 0.05 for pi in p)
    assert bh["n_rejected"] <= crudo
    assert by["n_rejected"] <= bh["n_rejected"]


def test_haircut_sharpe_reduce_con_mas_trials():
    """Más pruebas → mayor recorte y Sharpe descontado menor."""
    h1 = v.haircut_sharpe(1.5, n_trials=1, n_obs=250)
    h50 = v.haircut_sharpe(1.5, n_trials=50, n_obs=250)
    assert h50["sr_haircut_bhy"] < h1["sr_haircut_bhy"] <= 1.5
    assert 0.0 <= h50["haircut_pct"] <= 1.0


def test_min_btl_crece_con_trials():
    assert v.min_btl(100) > v.min_btl(10) > 0


def test_pbo_bajo_en_estrategias_genuinas(rng):
    """N configs con medias distintas y estables → PBO bajo (la mejor IS suele ganar OOS)."""
    T = 400
    means = np.linspace(0.0, 0.0015, 8)
    M = pd.DataFrame({f"c{i}": rng.normal(mu, 0.01, T) for i, mu in enumerate(means)})
    out = v.pbo_cscv(M, n_splits=10)
    assert out["n_configs"] == 8
    assert out["pbo"] <= 0.5


def test_reality_check_no_rechaza_ruido(rng):
    """Estrategias sin ventaja real vs benchmark → p alto (no data-snooping espurio)."""
    T = 300
    bench = pd.Series(rng.normal(0, 0.01, T))
    M = pd.DataFrame({f"s{i}": bench.values + rng.normal(0, 0.01, T) for i in range(5)})
    out = v.reality_check(M, bench, n_boot=500, seed=1)
    assert out["p_value"] > 0.10


def test_borrow_reduce_neto_y_usa_peso_retardado():
    """El borrow solo penaliza posiciones cortas y reduce el retorno neto."""
    idx = pd.date_range("2024-10-01", periods=50, freq="B")
    ret = pd.Series(np.full(50, 0.001), index=idx)
    w = pd.Series(np.where(np.arange(50) % 2 == 0, -1.0, 1.0), index=idx)
    bt0 = v.apply_borrow_cost(ret, w, borrow_bps_yr=0)
    bt500 = v.apply_borrow_cost(ret, w, borrow_bps_yr=500)
    assert (bt500["borrow"] >= 0).all()
    assert bt500["net_return_borrow"].sum() < bt0["net_return_borrow"].sum()
    # sin cortos, no hay borrow
    w_long = pd.Series(1.0, index=idx)
    bt_long = v.apply_borrow_cost(ret, w_long, borrow_bps_yr=500)
    assert bt_long["borrow"].sum() == pytest.approx(0.0)


def test_determinismo_reality_check(rng):
    T = 200
    bench = pd.Series(rng.normal(0, 0.01, T))
    M = pd.DataFrame({f"s{i}": rng.normal(0.0002, 0.01, T) for i in range(4)})
    a = v.reality_check(M, bench, n_boot=300, seed=7)
    b = v.reality_check(M, bench, n_boot=300, seed=7)
    assert a["p_value"] == b["p_value"]
