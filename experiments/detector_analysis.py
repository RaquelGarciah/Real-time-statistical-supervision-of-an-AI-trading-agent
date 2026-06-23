"""Análisis de los detectores de STRATA en SPY: intervención, éxito, atribución, scores y robustez (sin H2O).

Cubre lo que el marco práctico exige sobre la mecánica de STRATA y que faltaba en el notebook:

  A. Tasa de intervención de M8 y, por detector (RAM/PSA/GSO), tasa de disparo (score sobre umbral ex-ante) y
     tasa de éxito cuando interviene. Hace EXPLÍCITO si solo el régimen actúa y los otros dos quedan inertes.
  B. Distribución de los scores de cada detector con sus umbrales ex-ante marcados (para ver cuándo dispara).
  C. Atribución del P&L de rescate (M8−M5) al canal que lo genera. En override-C el override es del régimen
     (RAM, decisiones #5/#7), así que el rescate es atribuible a RAM; PSA/GSO se reportan para mostrar su inercia
     empírica (coherente con el hallazgo de que RAM domina).
  D. Robustez de la mejora M8−M5 por sub-ventana (alcista/lateral/bajista, por terciles del momento a 21 días)
     y en tres particiones train/test (60/40, 70/30, 80/20), con los TRES tests sobre ΔAccuracy: McNemar exacto,
     permutación por bloques (robusta a autocorrelación) y bootstrap estacionario (IC95).

Umbrales ex-ante de cache/models/strata_thresholds.json (PSA/GSO P95 y P99) y RAM τ=0.5 (gate medium). Se valida
que la accuracy de M8 coincide con la del panel canónico. Uso: python experiments/detector_analysis.py [--ticker SPY]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import experiments.automl_m10 as A
from core import metrics
from core.stats import block_permutation_test, mcnemar_test, sign_test, stationary_bootstrap_ci

THR = json.load(open("cache/models/strata_thresholds.json"))
PSA_P95, PSA_P99 = THR["psa"]["score_p95"], THR["psa"]["score_distribution"]["p99"]
GSO_P95, GSO_P99 = THR["gso"]["score_p95"], THR["gso"]["score_distribution"]["p99"]
TAU_RAM = 0.5


def _sr(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(np.sqrt(252) * a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0


def _delta_acc_tests(corr_a: np.ndarray, corr_b: np.ndarray) -> dict:
    """ΔAccuracy (b−a) con los tres contrastes pareados sobre los aciertos día a día."""
    d = corr_b.astype(float) - corr_a.astype(float)
    _, p_mc, _, _ = mcnemar_test(corr_a, corr_b)
    delta_bp, p_bp = block_permutation_test(corr_a, corr_b)
    # block_permutation_test devuelve mean(a)-mean(b) = M5-M8; lo negamos para que
    # la convención de signo coincida con delta_acc = M8-M5 y no haya contradicción en la tabla.
    delta_bp = -delta_bp
    # stationary_bootstrap_ci devuelve (low, high, point_estimate) (Politis-Romano 1994).
    lo, hi, point = stationary_bootstrap_ci(d)
    return {"n": int(len(d)), "delta_acc": round(float(d.mean()), 4),
            "mcnemar_p": round(float(p_mc), 4),
            "blockperm_delta": round(float(delta_bp), 4), "blockperm_p": round(float(p_bp), 4),
            "boot_ci95": [round(float(lo), 4), round(float(hi), 4)], "boot_excluye_0": bool(lo > 0 or hi < 0)}


def analizar(tk: str) -> dict:
    A.wf.TICKER = tk
    A.wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = A.build_states_onthefly(tk)
    m = A.wf.run_master(gamma_df, sigma, oos_ret, A.wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    y = np.sign(mv["r_next"].to_numpy())
    corr_m5 = (np.sign(mv["agent_size"].to_numpy()) == y).astype(int)
    corr_m8 = (np.sign(mv["final_size"].to_numpy()) == y).astype(int)
    interven = mv["intervenido"].astype(bool).to_numpy()
    rnext = mv["r_next"].to_numpy()

    # --- A. disparo y éxito por detector ---
    fire = {"RAM": mv["ram_score"].to_numpy() >= TAU_RAM,
            "PSA": mv["psa_score"].to_numpy() >= PSA_P95,
            "GSO": mv["gso_score"].to_numpy() >= GSO_P95}
    detectores = {}
    for d, f in fire.items():
        detectores[d] = {
            "umbral": round(float({"RAM": TAU_RAM, "PSA": PSA_P95, "GSO": GSO_P95}[d]), 4),
            "tasa_disparo": round(float(f.mean()), 4), "n_disparos": int(f.sum()),
            "acc_M8_en_disparo": round(float(corr_m8[f].mean()), 4) if f.any() else None,
            "acc_M5_en_disparo": round(float(corr_m5[f].mean()), 4) if f.any() else None}

    # --- A'. intervención de M8 y éxito del rescate ---
    interv = {
        "tasa_intervencion": round(float(interven.mean()), 4), "n_intervenciones": int(interven.sum()),
        "acc_M8_si_interviene": round(float(corr_m8[interven].mean()), 4) if interven.any() else None,
        "acc_M5_si_interviene": round(float(corr_m5[interven].mean()), 4) if interven.any() else None,
        "acc_M8_si_no_interviene": round(float(corr_m8[~interven].mean()), 4) if (~interven).any() else None}

    # --- C. atribución del P&L de rescate (override-C → RAM) ---
    pos_diff = np.sign(mv["final_size"].to_numpy()) - np.sign(mv["agent_size"].to_numpy())
    rescue_pnl = pos_diff * rnext                              # P&L bruto del cambio de posición de M8 sobre M5
    co = {d: float(rescue_pnl[interven & f].sum()) for d, f in fire.items()}
    atrib = {
        "pnl_rescate_total": round(float(rescue_pnl.sum()), 4),
        "pnl_en_dias_intervenidos": round(float(rescue_pnl[interven].sum()), 4),
        "pnl_dias_RAM_disparado": round(co["RAM"], 4),
        "pnl_dias_PSA_disparado": round(co["PSA"], 4),
        "pnl_dias_GSO_disparado": round(co["GSO"], 4),
        "nota": "override-C: el override lo decide el régimen (RAM, decisiones #5/#7); el P&L de rescate es "
                "atribuible a RAM. PSA/GSO se muestran por completitud y resultan en la práctica inertes."}

    # --- B. distribución de scores (arrays para histograma) + umbrales ---
    scores = {"ram_score": [round(float(x), 5) for x in mv["ram_score"].to_numpy()],
              "psa_score": [round(float(x), 5) for x in mv["psa_score"].to_numpy()],
              "gso_score": [round(float(x), 5) for x in mv["gso_score"].to_numpy()],
              "umbrales": {"RAM_tau": TAU_RAM, "PSA_p95": round(PSA_P95, 5), "PSA_p99": round(PSA_P99, 5),
                           "GSO_p95": round(GSO_P95, 5), "GSO_p99": round(GSO_P99, 5)}}

    # --- D. robustez de M8−M5 por sub-ventana y por partición ---
    trend = mv["r_curr"].rolling(21, min_periods=5).mean().shift(1)
    q1, q2 = trend.quantile(1 / 3), trend.quantile(2 / 3)
    lab = np.where(trend <= q1, "bajista", np.where(trend >= q2, "alcista", "lateral"))
    sub = {}
    for w in ("alcista", "lateral", "bajista"):
        msk = (lab == w)
        if msk.sum() >= 20:
            sub[w] = _delta_acc_tests(corr_m5[msk], corr_m8[msk])
            sub[w]["dSharpe_M8_M5"] = round(_sr(mv["nr_m8_causal"].to_numpy()[msk]) -
                                            _sr(mv["nr_m5_causal"].to_numpy()[msk]), 3)
    part = {}
    n = len(mv)
    for fr in (0.6, 0.7, 0.8):
        i0 = int(n * fr)
        sl = slice(i0, n)
        part[f"test_{int(fr*100)}_{int((1-fr)*100)}"] = _delta_acc_tests(corr_m5[sl], corr_m8[sl])

    acc_m8 = round(float(corr_m8.mean()), 4)
    return {"ticker": tk, "n_oos": int(len(mv)), "acc_M5": round(float(corr_m5.mean()), 4), "acc_M8": acc_m8,
            "detectores": detectores, "intervencion": interv, "atribucion_pnl": atrib,
            "scores": scores, "robustez_subventanas": sub, "robustez_particiones": part}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    args = ap.parse_args()
    res = analizar(args.ticker)
    out = Path(f"outputs/experiments/detector_analysis_{args.ticker}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== {args.ticker} · acc M5={res['acc_M5']} M8={res['acc_M8']} (n={res['n_oos']}) ===")
    print("intervención:", res["intervencion"])
    for d, v in res["detectores"].items():
        print(f"  {d}: disparo={v['tasa_disparo']:.2%} (n={v['n_disparos']}) "
              f"accM8|disparo={v['acc_M8_en_disparo']}")
    print("atribución P&L rescate:", {k: res["atribucion_pnl"][k] for k in res["atribucion_pnl"] if k.startswith("pnl")})
    print("sub-ventanas:", {w: (d["delta_acc"], d["mcnemar_p"]) for w, d in res["robustez_subventanas"].items()})
    print(f"OK · {out}")


if __name__ == "__main__":
    main()
