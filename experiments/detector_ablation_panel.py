"""Activación de detectores en el panel + ablación del meta-learner quitando los detectores (sin H2O).

Para cerrar el gap "¿importan los detectores?" con evidencia, no con prosa:
  A. ACTIVACIÓN por activo (panel 10): tasa de disparo de RAM/PSA/GSO (score ≥ umbral ex-ante) y tasa de
     intervención de M8 → gráfica de barras para §4. Muestra que RAM es el que actúa y PSA/GSO casi no.
  B. ABLACIÓN del meta-learner M10 en SPY (misma config: ensemble 10 XGBoost, WF emb=1) con distintos conjuntos
     de features: ALL22 · sin-PSA/GSO · sin-GSO · sin-PSA · solo-agente15 · solo-STRATA7. Mide accuracy/Sharpe →
     ¿cuánto cambia el modelo si NO usa los detectores? Compara con M5 (agente) y M8 (= el detector de régimen
     como regla; sin detectores M8 colapsa al agente M5).

Determinista (seed 42, sin H2O). Uso: python experiments/detector_ablation_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core.backtest import run_backtest
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22, AGENT15, STRATA7

THR = json.load(open("cache/models/strata_thresholds.json"))
PSA_P95, GSO_P95, TAU = THR["psa"]["score_p95"], THR["gso"]["score_p95"], 0.5
PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/detector_ablation_panel.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def main() -> None:
    config.set_seeds(config.SEED)
    # --- A. activación por activo (panel) ---
    activacion = {}
    for tk in PANEL10:
        wf.TICKER = tk; wf.reset_thresholds_cache()
        g, sig, oos = build_states(tk)
        m = wf.run_master(g, sig, oos, wf.load_agent(tk))
        mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)]
        activacion[tk] = {
            "RAM": round(float((mv["ram_score"] >= TAU).mean()), 4),
            "PSA": round(float((mv["psa_score"] >= PSA_P95).mean()), 4),
            "GSO": round(float((mv["gso_score"] >= GSO_P95).mean()), 4),
            "intervencion_M8": round(float(mv["intervenido"].astype(bool).mean()), 4)}
        print(f"{tk:5s} RAM={activacion[tk]['RAM']:.0%} PSA={activacion[tk]['PSA']:.1%} GSO={activacion[tk]['GSO']:.1%} "
              f"interv={activacion[tk]['intervencion_M8']:.0%}", flush=True)

    # --- B. ablación del meta-learner en SPY ---
    wf.TICKER = "SPY"; wf.reset_thresholds_cache()
    g, sig, oos = build_states("SPY")
    m = wf.run_master(g, sig, oos, wf.load_agent("SPY"))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    sets = {"ALL22 (canónico)": ALL22,
            "sin PSA+GSO": [c for c in ALL22 if c not in ("psa_score", "gso_score")],
            "sin GSO": [c for c in ALL22 if c != "gso_score"],
            "sin PSA": [c for c in ALL22 if c != "psa_score"],
            "solo agente (15)": AGENT15,
            "solo STRATA (7)": STRATA7}
    abl = {}
    for nm, cols in sets.items():
        p1 = wf_p1(mv[cols], y); sub = mv.index[p1.notna().to_numpy()]
        truth = np.sign(mv.loc[sub, "r_next"].to_numpy()); pos = np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0)
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
        nr = run_backtest(oos, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
        abl[nm] = {"n_features": len(cols), "accuracy": round(float((pos == truth).mean()), 4), "sharpe": round(_sr(nr), 3)}
        print(f"  M10[{nm:18s}] ({len(cols):2d} feat): acc={abl[nm]['accuracy']} Sharpe={abl[nm]['sharpe']:+.2f}", flush=True)
    # referencia M5/M8 sobre el mismo sub
    sub = mv.index[wf_p1(mv[ALL22], y).notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    ref = {}
    for nm, col in (("M5_agente", "agent_size"), ("M8_regla_régimen", "final_size")):
        pos = np.sign(mv.loc[sub, col].to_numpy())
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
        nr = run_backtest(oos, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
        ref[nm] = {"accuracy": round(float((pos == truth).mean()), 4), "sharpe": round(_sr(nr), 3)}

    res = {"meta": {"panel": PANEL10, "umbrales": {"RAM_tau": TAU, "PSA_p95": round(PSA_P95, 5), "GSO_p95": round(GSO_P95, 5)},
                    "ablacion_ticker": "SPY", "nota": "M10 = ensemble 10 XGBoost WF emb=1 (determinista); M8 sin "
                    "detector de régimen = el agente M5 (la regla ES el detector)."},
           "activacion_panel": activacion, "ablacion_m10_spy": abl, "referencia_spy": ref}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nReferencia SPY: M5 acc={ref['M5_agente']['accuracy']} | M8 acc={ref['M8_regla_régimen']['accuracy']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
