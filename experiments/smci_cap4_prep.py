"""Preparación de cifras para el Capítulo 4 (SMCI): contrastes M8 vs M10 y ablación/SHAP.

Reproduce en vivo el M10 definitivo (walk-forward ensemble, embargo=1, 22 features) y calcula:
- McNemar M10 vs M5 y M10 vs M8 sobre el OOS COMPLETO (confirma el 0.16 vs M5).
- Diebold-Mariano sobre P&L (loss = -retorno) M10 vs M8 y M10 vs M5 → hipótesis nivel 3 (CLAUDE.md §2:
  un ML no debe batir significativamente a la regla).
- Ablación: accuracy de M10 con agente-15 / STRATA-7 / 22 (mismo WF ensemble) + McNemar 22 vs agente-15.
- SHAP (TreeExplainer sobre el modelo full-fit): importancia por familia (STRATA/régimen vs agente).

Escribe outputs/experiments/m10_smci_cap4_prep.json. Uso: python experiments/smci_cap4_prep.py
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
from config import CALIBRATION_END, STRATA_OOS_START
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import diebold_mariano, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf

TICKER = "SMCI"
STEP, EMBARGO, N0, N_SEEDS = 21, 1, 150, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA7
OUT = Path("outputs/experiments/m10_smci_cap4_prep.json")


def build_states():
    feat_df, ret = wf.load_features(TICKER)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    return gamma, sigma, oos_ret


def wf_p1(X, y):
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 50:
            continue
        end = min(start + STEP, n)
        p.iloc[start:end] = np.mean(
            [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(X.iloc[:tr], y.iloc[:tr])
             .predict_proba(X.iloc[start:end])[:, 1] for sd in SEEDS], axis=0)
    return p


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states()
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)

    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    pos10 = np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    pos5 = np.sign(mv.loc[sub, "agent_size"].to_numpy())
    pos8 = np.sign(mv.loc[sub, "final_size"].to_numpy())
    c10 = (pos10 == truth).astype(int)
    c5 = (pos5 == truth).astype(int)
    c8 = (pos8 == truth).astype(int)

    # McNemar sobre OOS completo
    _, p_mc_m5, b5, cc5 = mcnemar_test(c5, c10)
    _, p_mc_m8, b8, cc8 = mcnemar_test(c8, c10)

    # P&L causales y Diebold-Mariano (loss = -retorno; H0: igual P&L medio)
    w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos10
    nr10 = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
    nr8 = mv["nr_m8_causal"].reindex(sub).to_numpy()
    nr5 = mv["nr_m5_causal"].reindex(sub).to_numpy()
    ok = ~(np.isnan(nr10) | np.isnan(nr8) | np.isnan(nr5))
    dm_m8 = diebold_mariano(-nr10[ok], -nr8[ok])
    dm_m5 = diebold_mariano(-nr10[ok], -nr5[ok])

    # Ablación: accuracy por feature set (mismo WF ensemble)
    abl = {}
    corr_sets = {}
    for nm, cols in (("agente15", AGENT15), ("strata7", STRATA7), ("all22", ALL22)):
        pa = p1 if nm == "all22" else wf_p1(mv[cols], y)
        sa = mv.index[pa.notna().to_numpy()]
        ta = np.sign(mv.loc[sa, "r_next"].to_numpy())
        posa = np.where(pa.dropna().to_numpy() >= 0.5, 1.0, -1.0)
        abl[nm] = round(float((posa == ta).mean()), 4)
        corr_sets[nm] = pd.Series((posa == ta).astype(int), index=sa)
    ci = corr_sets["all22"].index.intersection(corr_sets["agente15"].index)
    _, p_abl, b_abl, c_abl = mcnemar_test(corr_sets["agente15"].loc[ci].to_numpy(),
                                          corr_sets["all22"].loc[ci].to_numpy())

    # SHAP (TreeExplainer sobre modelo full-fit) → cuota por familia
    clf = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(mv[ALL22], y)
    try:
        import shap
        sv = shap.TreeExplainer(clf).shap_values(mv[ALL22])
        imp = pd.Series(np.abs(sv).mean(axis=0), index=ALL22)
        shap_metodo = "media |TreeSHAP|"
    except Exception as e:  # noqa: BLE001
        imp = pd.Series(clf.feature_importances_, index=ALL22)
        shap_metodo = f"XGB gain (shap no disponible: {type(e).__name__})"
    share_strata = float(imp[STRATA7].sum() / imp.sum())
    top = imp.sort_values(ascending=False).head(10)

    k_s, n_s, p_s, ci_s = sign_test(c10)
    res = {
        "meta": {"ticker": TICKER, "n": int(len(sub)), "embargo": EMBARGO, "n_seeds": N_SEEDS,
                 "nota": "prep cap.4: contrastes M8 vs M10 (hipótesis nivel 3) + ablación + SHAP; OOS completo",
                 "pre_registro": "BITACORA 2026-06-18 (cap.4 prep)"},
        "accuracy": {"m10": round(float(c10.mean()), 4), "m5": round(float(c5.mean()), 4),
                     "m8": round(float(c8.mean()), 4)},
        "mcnemar_oos_completo": {
            "m10_vs_m5": {"p": round(float(p_mc_m5), 4), "b": int(b5), "c": int(cc5)},
            "m10_vs_m8": {"p": round(float(p_mc_m8), 4), "b": int(b8), "c": int(cc8)}},
        "diebold_mariano_pnl": {
            "m10_vs_m8": {"stat": round(float(dm_m8[0]), 4), "p": round(float(dm_m8[1]), 4)},
            "m10_vs_m5": {"stat": round(float(dm_m5[0]), 4), "p": round(float(dm_m5[1]), 4)}},
        "sign_vs_0.5": {"k": int(k_s), "n": int(n_s), "p_2cola": round(float(p_s), 4),
                        "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]},
        "ablacion": {"accuracy": abl, "mcnemar_22_vs_agente15": {"p": round(float(p_abl), 4),
                                                                 "b": int(b_abl), "c": int(c_abl)}},
        "shap": {"metodo": shap_metodo, "cuota_strata7": round(share_strata, 4),
                 "top10": {k: round(float(v), 5) for k, v in top.items()}},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"acc M10={res['accuracy']['m10']} M8={res['accuracy']['m8']} M5={res['accuracy']['m5']}")
    print(f"McNemar M10 vs M5 (OOS)={res['mcnemar_oos_completo']['m10_vs_m5']['p']} "
          f"| vs M8={res['mcnemar_oos_completo']['m10_vs_m8']['p']}")
    print(f"DM P&L M10 vs M8 p={res['diebold_mariano_pnl']['m10_vs_m8']['p']} "
          f"| vs M5 p={res['diebold_mariano_pnl']['m10_vs_m5']['p']}")
    print(f"ablación: {abl} · McNemar 22 vs agente15 p={res['ablacion']['mcnemar_22_vs_agente15']['p']}")
    print(f"SHAP cuota STRATA7={res['shap']['cuota_strata7']} ({shap_metodo})")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
