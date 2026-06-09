"""M10 (XGBoost-CPCV) con la lógica K=2 + τ=0.5 para SPY — prueba aislada antes de reescribir el canónico.

Replica la mecánica de M10 del notebook canónico (22 features, CPCV n=6/2 embargo=5, w=sign(p1-0.5),
SHAP TreeSHAP nativo, ablación) pero con las features de RÉGIMEN derivadas de un HMM K=2 (sin Estrés)
y el override M8 con K=2 / τ=0.5. Compara M10 vs M8(K=2).

Uso: ``python experiments/m10_k2.py``
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as _logloss

import config
from config import CACHE_AGENT_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.cpcv import CombinatorialPurgedKFold
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import diebold_mariano, tost
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

TICKER = "SPY"
K = 2
TAU = 0.5
PERS = list(config.ACTIVE_PERSONALITIES)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
              eval_metric="logloss", random_state=config.SEED, tree_method="hist")


def load_agent_full(tk):
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))):
        d = json.load(open(fp))
        pers = {k: PersonalityOutput(name=k, action=v["action"], size=v["size"],
                                     confidence=v["confidence"], reasoning=v.get("reasoning", ""))
                for k, v in d.get("personalities", {}).items()}
        out[pd.Timestamp(d["date"])] = AgentOutput(date=d["date"], ticker=d["ticker"], action=d["action"],
                                                   size=d["size"], confidence=d["confidence"], reasoning="",
                                                   personalities=pers)
    return out


def cpcv_oof(Xm, ym, collect_shap=False):
    t1 = pd.Series(Xm.index, index=Xm.index).shift(-1).ffill()
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo=5)
    oof_sum = np.zeros(len(Xm)); oof_cnt = np.zeros(len(Xm)); frows = []
    sabs = np.zeros(Xm.shape[1]); sn = 0
    for fid, (tr, te) in enumerate(cv.split(Xm, t1=t1)):
        clf = xgb.XGBClassifier(**PARAMS); clf.fit(Xm.iloc[tr], ym.iloc[tr])
        pte = clf.predict_proba(Xm.iloc[te])[:, 1]
        oof_sum[te] += pte; oof_cnt[te] += 1
        frows.append({"fold": fid, "logloss": _logloss(ym.iloc[te], pte, labels=[0, 1])})
        if collect_shap:
            ct = clf.get_booster().predict(xgb.DMatrix(Xm.iloc[te], feature_names=list(Xm.columns)),
                                           pred_contribs=True)
            sabs += np.abs(ct[:, :-1]).sum(0); sn += len(te)
    p1 = pd.Series(oof_sum / np.maximum(oof_cnt, 1), index=Xm.index)
    return p1, pd.DataFrame(frows), (sabs / max(sn, 1) if collect_shap else None)


def main():
    end = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"]); rv = features.realized_vol_annualized(ret, 21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]; sigma = garch.forecast_path(oos_ret)
    hmm = RegimeHMM(n_states=K, seed=config.SEED).fit(calib.to_numpy())
    g = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index, columns=["s0", "s1"])
    agents = load_agent_full(TICKER)

    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD, ram_thresholds=(TAU / 2, TAU, 0.70))
    rows = []; sh = []
    for t in sorted(agents):
        if t not in g.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); gg = g.loc[t]
        ms = {"regime": {"calm_prob": float(gg["s0"]), "stress_prob": 0.0, "crisis_prob": float(gg["s1"]),
                         "viterbi_state": int(gg["s1"] > 0.5)}, "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, sh)
        row = {"date": t, "agent_size": a.size, "final_size": dec.final_size,
               "ram_score": dec.detectors["ram"].score, "psa_score": dec.detectors["psa"].score,
               "gso_score": dec.detectors["gso"].score, "calm_prob": float(gg["s0"]),
               "stress_prob": 0.0, "crisis_prob": float(gg["s1"]), "garch_sigma": float(sigma.loc[t])}
        for nm in PERS:
            po = a.personalities.get(nm)
            row[f"{nm}_sign"] = 0.0 if po is None else (1.0 if po.action == "long" else -1.0 if po.action == "short" else 0.0)
            row[f"{nm}_size"] = 0.0 if po is None else float(po.size)
            row[f"{nm}_conf"] = 0.0 if po is None else float(po.confidence)
        rows.append(row)
    master = pd.DataFrame(rows).set_index("date")
    master["r_next"] = oos_ret.shift(-1).reindex(master.index)
    yv = np.sign(master["r_next"]); valid = yv.notna() & (yv != 0)
    yb = (master.loc[valid, "r_next"] > 0).astype(int)

    feat_cols = [f"{nm}_{k}" for nm in PERS for k in ("sign", "size", "conf")] + \
                ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
    X = master.loc[valid, feat_cols]
    print(f"M10 K=2 — features {X.shape[1]} × {X.shape[0]} días (stress_prob=0 constante con K=2)")

    p1, folds, shap_mean = cpcv_oof(X, yb, collect_shap=True)
    w10 = np.sign(p1 - 0.5).reindex(master.index)
    nr10 = run_backtest(oos_ret, w10, signal_lag=1)["net_return"]
    nr8 = run_backtest(oos_ret, master["final_size"], signal_lag=1)["net_return"]
    nr5 = run_backtest(oos_ret, master["agent_size"], signal_lag=1)["net_return"]
    acc10 = float((np.sign(w10).reindex(master.index)[valid] == yv[valid]).mean())
    print(f"\nM10 K=2: Sharpe={metrics.sharpe(nr10):+.3f}  acc={acc10:.3f}  logloss OOF mediana={folds['logloss'].median():.3f} media={folds['logloss'].mean():.3f}")
    print(f"M8 K=2:  Sharpe={metrics.sharpe(nr8):+.3f}   |  M5 agente: Sharpe={metrics.sharpe(nr5):+.3f}")

    common = nr10.index.intersection(nr8.index)
    dm_s, dm_p = diebold_mariano((-nr10.loc[common]).to_numpy(), (-nr8.loc[common]).to_numpy())
    dsh = metrics.sharpe(nr10.loc[common]) - metrics.sharpe(nr8.loc[common])
    d = (nr10.loc[common] - nr8.loc[common]); margin = 0.5 / np.sqrt(252) * d.std(ddof=1)
    p_tost, equiv = tost(d, margin=margin)
    print(f"\nM10 vs M8 (K=2): DM stat={dm_s:+.3f} p={dm_p:.3f}  ΔSharpe={dsh:+.3f}  TOST p={p_tost:.3f} equiv={equiv}")

    def familia(f):
        if f in ("ram_score", "psa_score", "gso_score"): return "STRATA"
        if f in ("calm_prob", "stress_prob", "crisis_prob", "garch_sigma"): return "régimen"
        return "personalidad"
    shp = pd.DataFrame({"feature": X.columns, "familia": [familia(f) for f in X.columns], "shap": shap_mean})
    shp = shp.sort_values("shap", ascending=False).reset_index(drop=True)
    print("\nTop-8 SHAP (|SHAP| medio pooled OOF):")
    print(shp.head(8).to_string(index=False))

    # Ablación: solo features del agente (sin STRATA ni régimen).
    ag_cols = [c for c in X.columns if c.startswith(tuple(PERS))]
    p1a, _, _ = cpcv_oof(X[ag_cols], yb)
    sh_ag = metrics.sharpe(run_backtest(oos_ret, np.sign(p1a - 0.5).reindex(master.index), signal_lag=1)["net_return"])
    print(f"\nAblación M10 solo-agente (15 feats): Sharpe={sh_ag:+.3f}  vs completo {metrics.sharpe(nr10):+.3f}")
    print("(si completo >> solo-agente → las features STRATA/régimen aportan; si ≈ → no)")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
