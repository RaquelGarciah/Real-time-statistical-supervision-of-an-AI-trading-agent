"""SPY: material extra para mecanismo, calibración e interpretabilidad (sin H2O).

Genera lo que necesitan varias gráficas nuevas del notebook definitivo, todo sobre SPY y la ventana desplegable:
  - daily: serie diaria (régimen, posición M5/M8/M10, p1 continuo del meta-learner, verdad) → confusión M10 vs M5
    por régimen (#2), timeline de acuerdo/desacuerdo M8↔M10 (#3) y curva de calibración de M10 (#6).
  - shap_dependency: para 3 features STRATA (crisis_prob, garch_sigma, psa_score), pares (valor, |SHAP|/SHAP) con
    el régimen del día → SHAP dependency plots (#8).
  - shap_rolling: cuota STRATA en |SHAP| recalculada en cada reentreno del walk-forward → ¿deriva la importancia
    de STRATA? (#9).

M10 = XGBoost canónico (ensemble de semillas para p1; un fit único para SHAP, igual que m10_shap_priorflip).
Uso: python experiments/spy_mechanism_extras.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22, AGENT15, STRATA7, PARAMS

TICKER = "SPY"
N0, STEP, EMBARGO, N_SEEDS = 150, 21, 1, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
OUT = Path("outputs/experiments/spy_mechanism_extras.json")
REG = {0: "Calma", 1: "Estrés", 2: "Crisis"}


def main() -> None:
    config.set_seeds(config.SEED)
    wf.TICKER = TICKER; wf.reset_thresholds_cache()
    gamma, sigma, oos = build_states(TICKER)
    m = wf.run_master(gamma, sigma, oos, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p1 = wf_p1(mv[ALL22], y); sub = mv.index[p1.notna().to_numpy()]
    mvs = mv.loc[sub]
    truth = np.sign(mvs["r_next"].to_numpy())
    daily = {"dates": [str(d.date()) for d in sub],
             "regime": [int(x) for x in mvs["regime_dom"].to_numpy()],
             "m5_pos": [int(x) for x in np.sign(mvs["agent_size"].to_numpy())],
             "m8_pos": [int(x) for x in np.sign(mvs["final_size"].to_numpy())],
             "m10_pos": [int(x) for x in np.sign(p1.dropna().to_numpy() - 0.5)],
             "m10_p1": [round(float(x), 5) for x in p1.dropna().to_numpy()],
             "truth": [int(x) for x in truth]}

    # --- SHAP del XGBoost canónico (fit único sobre la ventana válida; describe en qué se apoya) ---
    clf = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(mvs[ALL22], y.loc[sub])
    import shap
    sv = shap.TreeExplainer(clf).shap_values(mvs[ALL22])
    sv = np.asarray(sv)
    feats_dep = ["crisis_prob", "garch_sigma", "psa_score"]
    shap_dep = {}
    for f in feats_dep:
        j = ALL22.index(f)
        shap_dep[f] = {"x": [round(float(v), 5) for v in mvs[f].to_numpy()],
                       "shap": [round(float(s), 5) for s in sv[:, j]],
                       "regime": daily["regime"]}

    # --- Cuota STRATA en |SHAP| recalculada por reentreno (walk-forward) → ¿deriva la importancia? ---
    Xall = mv[ALL22]; n = len(Xall); roll = {"fecha_fin": [], "cuota_strata": []}
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 50:
            continue
        end = min(start + STEP, n)
        c = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(Xall.iloc[:tr], y.iloc[:tr])
        block = Xall.iloc[start:end]
        s = np.abs(np.asarray(shap.TreeExplainer(c).shap_values(block))).mean(0)
        tot = max(s.sum(), 1e-12)
        cuota = float(sum(s[ALL22.index(f)] for f in STRATA7) / tot)
        roll["fecha_fin"].append(str(Xall.index[end - 1].date())); roll["cuota_strata"].append(round(cuota, 4))

    out = {"meta": {"ticker": TICKER, "n_sub": int(len(sub)), "ventana": "desplegable",
                    "nota": "SHAP del XGBoost canónico (fit único, in-sample, describe apoyo del modelo); "
                            "cuota rodante = |SHAP| de STRATA7 por reentreno walk-forward"},
           "daily": daily, "shap_dependency": shap_dep, "shap_rolling": roll}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # resumen
    reg = np.array(daily["regime"]); cm5 = (np.array(daily["m5_pos"]) == truth); cm10 = (np.array(daily["m10_pos"]) == truth)
    print(f"SPY n={len(sub)} · acc M5={cm5.mean():.3f} M10={cm10.mean():.3f}")
    for k, nm in REG.items():
        msk = reg == k
        if msk.any(): print(f"  {nm}: n={msk.sum()} accM5={cm5[msk].mean():.3f} accM10={cm10[msk].mean():.3f}")
    print(f"cuota STRATA rodante: min={min(roll['cuota_strata']):.2f} max={max(roll['cuota_strata']):.2f} "
          f"media={np.mean(roll['cuota_strata']):.2f} ({len(roll['cuota_strata'])} reentrenos)")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
