"""Robustez a la ventana de calibración (SMCI), sugerida por el tutor.

Recalibra HMM(K=3)+GARCH con distintos inicios de ventana (fin fijo en CALIBRATION_END,
anterior al OOS → sin fuga), recomputa las features de régimen/vol y el walk-forward
ensemble de M10, y mide accuracy/Sharpe/equity sobre el OOS FIJO. También reporta la media
de retorno por régimen en cada ventana (para ver si Crisis se vuelve direccional).

Pre-registro: robustez, NO selección. El OOS no se toca; la ventana completa (2007→2024-09)
es la pre-registrada (CLAUDE.md §3). Reportamos TODAS las ventanas; elegir la que maximiza
OOS sería p-hacking. Hipótesis del tutor a contrastar: "acortar la calibración (el pasado
lejano aporta poco) mejora". Criterio de lectura: ¿mejora M10 al acortar? ¿Crisis<0?

Uso: python experiments/smci_calib_window.py
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
from core.metrics import equity_curve
import experiments.walkforward_robustez as wf

TICKER = "SMCI"
STEP, EMBARGO, N0, N_SEEDS = 21, 1, 150, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
ANN = np.sqrt(252)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA7
STARTS = ["2007-01-01", "2010-01-01", "2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01", "2022-01-01"]
OUT = Path("outputs/experiments/smci_calib_window.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _wf_p1(X, y):
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


def run_window(feat_df, ret, start: str) -> dict:
    calib = feat_df.loc[(feat_df.index >= pd.Timestamp(start)) & (feat_df.index <= pd.Timestamp(CALIBRATION_END))]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(CALIBRATION_END))])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    st = pd.Series(hmm.predict_states(calib.to_numpy()), index=calib.index)
    rc = ret.reindex(calib.index)
    means = {lbl: round(float(rc[st == k].mean()), 6) for k, lbl in enumerate(["Calma", "Estrés", "Crisis"])}

    wf.reset_thresholds_cache()
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p = _wf_p1(mv[ALL22], y)
    sub = mv.index[p.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    pos = np.where(p.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
    return {"start": start, "n_cal": int(len(calib)), "medias_regimen": means,
            "m10_acc": round(float((pos == truth).mean()), 4), "m10_sharpe": round(_sr(nr), 3),
            "m10_equity": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4),
            "m5_acc": round(float((np.sign(mv.loc[sub, "agent_size"].to_numpy()) == truth).mean()), 4),
            "m8_acc": round(float((np.sign(mv.loc[sub, "final_size"].to_numpy()) == truth).mean()), 4),
            "n": int(len(sub))}


def main() -> None:
    feat_df, ret = wf.load_features(TICKER)
    res = {"meta": {"ticker": TICKER, "calibration_end": CALIBRATION_END, "oos_start": STRATA_OOS_START,
                    "n_seeds": N_SEEDS, "embargo": EMBARGO, "burn_in": N0,
                    "nota": "robustez a la ventana de calibración (sugerencia del tutor); OOS fijo, sin fuga; "
                            "ventana completa = pre-registrada, NO se elige por el número (no p-hacking)",
                    "hipotesis_tutor": "acortar la calibración mejora / vuelve direccional al régimen",
                    "pre_registro": "BITACORA 2026-06-18"},
           "por_ventana": []}
    for s in STARTS:
        r = run_window(feat_df, ret, s)
        res["por_ventana"].append(r)
        print(f"{s} n_cal={r['n_cal']:>5} Crisis={r['medias_regimen']['Crisis']:+.5f} "
              f"M10={r['m10_acc']} SR={r['m10_sharpe']:+.2f} eq={r['m10_equity']} | M5={r['m5_acc']} M8={r['m8_acc']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
