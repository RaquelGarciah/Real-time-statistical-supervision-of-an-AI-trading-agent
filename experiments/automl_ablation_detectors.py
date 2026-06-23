"""Ablación de detectores con AutoML-H2O en SPY (determinista: seed=42 + max_models fijo).

A petición de Raquel: en lugar de solo el M10-XGBoost, repetir la ablación con el BUSCADOR AutoML (H2O), en la
MISMA configuración canónica del panel (max_models=25, GBM/XGBoost/StackedEnsemble, AUC, Purged K-Fold,
walk-forward emb=1, seed=42 → determinista por max_models+semilla), variando el conjunto de features:
  ALL22 (canónico) · sin PSA+GSO · solo agente (15) · solo STRATA (7).
Mide accuracy/Sharpe sobre la ventana desplegable → ¿cuánto cambia el AutoML si NO usa los detectores?
Reutiliza automl_wf_p1 de automl_m10 (idéntico pipeline). Uso: python experiments/automl_ablation_detectors.py
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
import experiments.automl_m10 as A
from experiments.quant_validation_panel import AGENT15, STRATA7

TICKER = "SPY"
MAX_MODELS = 25
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/automl_ablation_detectors.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def main() -> None:
    config.set_seeds(config.SEED)
    A.INCLUDE_ALGOS = ["GBM", "XGBoost", "StackedEnsemble"]  # config canónica del panel mm25
    A.SORT_METRIC = "AUC"; A.HOLDOUT_FRAC = None
    A.wf.TICKER = TICKER; A.wf.reset_thresholds_cache()
    gamma, sigma, oos = A.build_states_onthefly(TICKER)
    m = A.wf.run_master(gamma, sigma, oos, A.wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    ALL22 = A.FULL_COLS
    sets = {"ALL22 (canónico)": ALL22,
            "sin PSA+GSO": [c for c in ALL22 if c not in ("psa_score", "gso_score")],
            "solo agente (15)": AGENT15,
            "solo STRATA (7)": STRATA7}
    res = {"meta": {"ticker": TICKER, "max_models": MAX_MODELS, "seed": config.SEED,
                    "config": "AutoML-H2O canónico: GBM/XGBoost/StackedEnsemble, AUC, Purged K-Fold, WF emb=1; "
                              "determinista por max_models+seed", "ventana": "desplegable (tras burn-in 150)"},
           "ablacion_automl_spy": {}}
    try:
        for nm, cols in sets.items():
            print(f"\n##### AutoML[{nm}] ({len(cols)} feat) #####", flush=True)
            p1, leaders = A.automl_wf_p1(mv[cols], y, MAX_MODELS)
            sub = mv.index[p1.notna().to_numpy()]
            truth = np.sign(mv.loc[sub, "r_next"].to_numpy()); pos = np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0)
            w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
            nr = run_backtest(oos, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
            fams = sorted({l["family"] for l in leaders})
            res["ablacion_automl_spy"][nm] = {"n_features": len(cols), "n": int(len(sub)),
                                              "accuracy": round(float((pos == truth).mean()), 4),
                                              "sharpe": round(_sr(nr), 3), "familias_leader": fams}
            print(f"  acc={res['ablacion_automl_spy'][nm]['accuracy']} Sharpe={res['ablacion_automl_spy'][nm]['sharpe']:+.2f} familias={fams}", flush=True)
            OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    finally:
        A.ha.shutdown_h2o()
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
