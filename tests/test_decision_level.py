"""Tests del análisis decision-level (M5 vs M8) y del bootstrap estacionario.

Cubre §11 de CLAUDE.md con tests de caso feliz, propiedad matemática,
determinismo y disjunción de buckets para la convención de atribución
exclusiva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.stats import stationary_bootstrap_ci
from experiments.decision_level_analysis import (
    DETECTORS,
    SEV_WEIGHT,
    _intervention_days,
    _intervention_pnl_series,
    attribution_exclusive_table,
    attribution_proportional_table,
    hit_rate_table,
)


# ---------------------------------------------------------------------------
# core.stats.stationary_bootstrap_ci
# ---------------------------------------------------------------------------

def test_stationary_bootstrap_determinism():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 400)
    first = stationary_bootstrap_ci(x, n=500, seed=42)
    second = stationary_bootstrap_ci(x, n=500, seed=42)
    assert first == second


def test_stationary_bootstrap_iid_coverage_above_nominal_minus_slack():
    """Cobertura empírica del IC 95% no debe caer muy por debajo de 0.90.

    Réplica reducida de la prueba de Politis-Romano: 100 muestras iid normales,
    el IC del 95% del estimador de la media cubre la media verdadera en ≥ 88 de
    cada 100 (el undercoverage del estacionario sobre iid es esperable; 88 es
    cota conservadora). El test es rápido (n=500, B=500).
    """
    mu = 0.3
    cover = 0
    trials = 100
    for s in range(trials):
        x = np.random.default_rng(s).normal(mu, 1.0, 500)
        lo, hi, _ = stationary_bootstrap_ci(x, n=500, seed=s)
        cover += int(lo <= mu <= hi)
    assert cover >= 88, f"Cobertura {cover}/{trials} < 88"


def test_stationary_bootstrap_default_block_length_matches_sqrt_n():
    """Si no se pasa ``mean_block_len``, debe equivaler a ``round(sqrt(N))``."""
    x = np.arange(100, dtype=float)
    a = stationary_bootstrap_ci(x, n=200, mean_block_len=10.0, seed=1)
    b = stationary_bootstrap_ci(x, n=200, seed=1)  # default = round(sqrt(100)) = 10
    assert a == b


# ---------------------------------------------------------------------------
# Hit rate: excluye días con size=0
# ---------------------------------------------------------------------------

def _toy_panel():
    dates = pd.date_range("2025-01-02", periods=6, freq="B")
    return {
        "AAA": {
            "size_M5": pd.Series([0.5, 0.0, -0.5, 0.0, 0.5, -0.5], index=dates),
            "size_M8": pd.Series([0.5, 0.5, -0.5, 0.5, -0.5, 0.5], index=dates),
            "r_next":  pd.Series([0.01, -0.02, -0.01, 0.02, 0.01, 0.01], index=dates),
            "flags": pd.DataFrame({
                "ram_severity": ["high", "none", "none", "medium", "none", "high"],
                "ram_flag":     [True, False, False, True, False, True],
                "psa_severity": ["none", "medium", "none", "none", "none", "low"],
                "psa_flag":     [False, True, False, False, False, False],
                "gso_severity": ["none", "none", "none", "medium", "none", "none"],
                "gso_flag":     [False, False, False, True, False, False],
                "gso_bound": [1.0]*6, "gso_bounded_size": [0.0]*6,
            }, index=dates),
        }
    }


def test_hit_rate_excludes_zero_size_days():
    """Días con size==0 no entran en el hit rate; M5 cuenta 4 días no nulos."""
    df = hit_rate_table(_toy_panel())
    # M5: 4 días no nulos (0.5,-0.5,0.5,-0.5) frente a r_next (0.01,-0.01,0.01,0.01).
    # signos coincidentes: +×+=ok, -×-=ok, +×+=ok, -×+=fail → hit=3/4.
    assert int(df.loc["AAA", "N_M5"]) == 4
    assert df.loc["AAA", "hit_rate_M5"] == 0.75
    # M8: 6 días no nulos.
    assert int(df.loc["AAA", "N_M8"]) == 6


# ---------------------------------------------------------------------------
# Atribución proporcional: la suma por detector iguala la suma de pnl_int
# en días con al menos un detector activo.
# ---------------------------------------------------------------------------

def test_attribution_proportional_sums_match_pnl_with_active_detector():
    panel = _toy_panel()
    df_int = _intervention_days(panel["AAA"])
    # Días con al menos un detector activo (severidad >= medium).
    mask_any = np.any([df_int[f"{d}_severity"].map(SEV_WEIGHT) > 0 for d in DETECTORS], axis=0)
    expected_total = float(df_int.loc[mask_any, "pnl_bps"].sum())
    prop = attribution_proportional_table(panel)
    got_total = float(prop.loc["AAA", "total_pnl_bps"].sum())
    assert np.isclose(got_total, expected_total, atol=1e-9)


def test_attribution_exclusive_is_disjoint_partition_of_active_days():
    """RAM/PSA/GSO/MULTI son disjuntos y particionan los días con algún flag."""
    panel = _toy_panel()
    df_int = _intervention_days(panel["AAA"])
    flags = pd.DataFrame({d: df_int[f"{d}_flag"] for d in DETECTORS})
    expected_days = int((flags.sum(axis=1) >= 1).sum())
    excl = attribution_exclusive_table(panel)
    got_days = int(excl.loc["AAA", "N_act"].sum())
    assert got_days == expected_days


# ---------------------------------------------------------------------------
# Intervención: pnl_int restringido a |delta| > 0
# ---------------------------------------------------------------------------

def test_intervention_pnl_excludes_zero_delta():
    """``_intervention_pnl_series`` filtra los días con delta=0 (no intervención)."""
    panel = _toy_panel()
    pnl = _intervention_pnl_series(panel["AAA"])
    delta = panel["AAA"]["size_M8"] - panel["AAA"]["size_M5"]
    expected_n = int((delta.abs() > 0).sum())
    assert len(pnl) == expected_n
    assert (pnl.index.isin(delta[delta.abs() > 0].index)).all()
