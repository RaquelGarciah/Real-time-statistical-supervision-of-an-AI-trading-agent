"""¿M8 bate a M10 SOLO cuando su apuesta al drift paga? Test falsable sobre activos alcistas vs bajistas.

Hipótesis: M8 (regla mecánica que va 'largo en baja-vol', K=2) bate a M10 (predictor) en activos con
drift OOS positivo (cabalga la tendencia), pero M10 ≥ M8 en activos con drift OOS negativo (donde la
apuesta ciega al drift falla). Mismo K=2 y τ=0.5 para todos, para aislar el efecto del drift.

Uso: ``python experiments/m10_vs_m8_drift.py``
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
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

TICKERS = ["NVDA", "SPY", "UNG", "MARA", "SMCI"]   # mezcla de drift OOS + y −
PERS = list(config.ACTIVE_PERSONALITIES)
TAU = 0.5
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
              eval_metric="logloss", random_state=config.SEED, tree_method="hist")


def load_agent_full(tk):
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))):
        d = json.load(open(fp))
        pers = {k: PersonalityOutput(name=k, action=v["action"], size=v["size"],
                                     confidence=v["confidence"], reasoning="") for k, v in d.get("personalities", {}).items()}
        out[pd.Timestamp(d["date"])] = AgentOutput(date=d["date"], ticker=d["ticker"], action=d["action"],
                                                   size=d["size"], confidence=d["confidence"], reasoning="", personalities=pers)
    return out


def cpcv_oof(Xm, ym):
    t1 = pd.Series(Xm.index, index=Xm.index).shift(-1).ffill()
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo=5)
    oof_sum = np.zeros(len(Xm)); oof_cnt = np.zeros(len(Xm))
    for tr, te in cv.split(Xm, t1=t1):
        clf = xgb.XGBClassifier(**PARAMS); clf.fit(Xm.iloc[tr], ym.iloc[tr])
        oof_sum[te] += clf.predict_proba(Xm.iloc[te])[:, 1]; oof_cnt[te] += 1
    return pd.Series(oof_sum / np.maximum(oof_cnt, 1), index=Xm.index)


def run_ticker(tk):
    end = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(tk, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"]); rv = features.realized_vol_annualized(ret, 21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]; sigma = garch.forecast_path(oos_ret)
    hmm = RegimeHMM(n_states=2, seed=config.SEED).fit(calib.to_numpy())
    g = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index, columns=["s0", "s1"])
    agents = load_agent_full(tk)
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD, ram_thresholds=(TAU / 2, TAU, 0.70))
    rows, sh = [], []
    for t in sorted(agents):
        if t not in g.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); gg = g.loc[t]
        ms = {"regime": {"calm_prob": float(gg["s0"]), "stress_prob": 0.0, "crisis_prob": float(gg["s1"]),
                         "viterbi_state": int(gg["s1"] > 0.5)}, "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, sh)
        row = {"date": t, "final_size": dec.final_size, "ram_score": dec.detectors["ram"].score,
               "psa_score": dec.detectors["psa"].score, "gso_score": dec.detectors["gso"].score,
               "calm_prob": float(gg["s0"]), "stress_prob": 0.0, "crisis_prob": float(gg["s1"]),
               "garch_sigma": float(sigma.loc[t])}
        for nm in PERS:
            po = a.personalities.get(nm)
            row[f"{nm}_sign"] = 0.0 if po is None else (1.0 if po.action == "long" else -1.0 if po.action == "short" else 0.0)
            row[f"{nm}_size"] = 0.0 if po is None else float(po.size)
            row[f"{nm}_conf"] = 0.0 if po is None else float(po.confidence)
        rows.append(row)
    m = pd.DataFrame(rows).set_index("date")
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    yb = (m.loc[valid, "r_next"] > 0).astype(int)
    cols = [f"{nm}_{k}" for nm in PERS for k in ("sign", "size", "conf")] + \
           ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
    p1 = cpcv_oof(m.loc[valid, cols], yb)
    w10 = np.sign(p1 - 0.5).reindex(m.index)
    sh8 = metrics.sharpe(run_backtest(oos_ret, m["final_size"], signal_lag=1)["net_return"])
    sh10 = metrics.sharpe(run_backtest(oos_ret, w10, signal_lag=1)["net_return"])
    drift = float(oos_ret.mean() * 252)
    return {"ticker": tk, "drift_oos": round(drift, 2), "M8_sharpe": round(sh8, 3),
            "M10_sharpe": round(sh10, 3), "M10_minus_M8": round(sh10 - sh8, 3)}


def main():
    rows = []
    for tk in TICKERS:
        try:
            r = run_ticker(tk); rows.append(r)
            print(f"{tk:6} drift_oos={r['drift_oos']:+.2f}  M8={r['M8_sharpe']:+.2f}  M10={r['M10_sharpe']:+.2f}  "
                  f"M10−M8={r['M10_minus_M8']:+.2f}  {'M10 GANA' if r['M10_minus_M8']>0 else 'M8 gana'}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk}: ERROR {e!r}")
    ok = [r for r in rows if "ticker" in r]
    from scipy.stats import spearmanr
    dr = np.array([r["drift_oos"] for r in ok]); dd = np.array([r["M10_minus_M8"] for r in ok])
    rho, p = spearmanr(dr, dd)
    print(f"\nCorrelación drift_oos vs (M10−M8): ρ={rho:+.2f} (p={p:.3f})")
    print("Hipótesis: ρ<0 ⇒ M10 gana a M8 cuanto MÁS negativo el drift (M8 solo gana cabalgando el alza).")
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    Path("outputs/experiments/m10_vs_m8_drift.json").write_text(json.dumps({"per_asset": ok, "rho": float(rho), "p": float(p)}, indent=2))


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
