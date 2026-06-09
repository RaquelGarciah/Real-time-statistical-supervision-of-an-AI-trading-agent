"""¿Cabalga M8 el drift? Test falsable para K=2 Y K=3, sobre los 10 activos (corrige el autogol).

El test previo (m10_vs_m8_drift.py) solo corrió K=2 y n=5 — no podía decir nada sobre K=3. Aquí
comparamos M10 (meta-learner adaptativo) vs M8 (override-C) para CADA K∈{2,3} en los 10 activos,
y medimos ρ(drift_OOS, M10−M8). Hipótesis del canónico: K=2 cabalga el drift (ρ<0, M8 gana solo
en alcistas) PERO K=3 NO (ρ_K3≈0, M8-K3 gana con independencia del drift = supervisa). Si ρ_K3
también es fuerte y negativo, K=3 es "drift-riding moderado" y la afirmación de §12 es falsa.

Uso: ``python experiments/drift_test_k2k3.py`` → outputs/experiments/drift_test_k2k3.json
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

import config
from config import CACHE_AGENT_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.cpcv import CombinatorialPurgedKFold
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
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
    s = np.zeros(len(Xm)); c = np.zeros(len(Xm))
    for tr, te in cv.split(Xm, t1=t1):
        clf = xgb.XGBClassifier(**PARAMS); clf.fit(Xm.iloc[tr], ym.iloc[tr])
        s[te] += clf.predict_proba(Xm.iloc[te])[:, 1]; c[te] += 1
    return pd.Series(s / np.maximum(c, 1), index=Xm.index)


def run(tk, K):
    end = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(tk, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"]); rv = features.realized_vol_annualized(ret, 21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]; sigma = garch.forecast_path(oos_ret)
    hmm = RegimeHMM(n_states=K, seed=config.SEED).fit(calib.to_numpy())
    g = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index, columns=[f"s{j}" for j in range(K)])
    agents = load_agent_full(tk)
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD, ram_thresholds=(TAU / 2, TAU, 0.70))
    rows, sh = [], []; lo, hi = "s0", f"s{K-1}"; mids = [f"s{j}" for j in range(1, K - 1)]
    for t in sorted(agents):
        if t not in g.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); gg = g.loc[t]
        rp = {"calm_prob": float(gg[lo]), "crisis_prob": float(gg[hi]),
              "stress_prob": float(gg[mids].sum()) if mids else 0.0, "viterbi_state": int(np.argmax(gg.values))}
        dec = sup.supervise(a, {"regime": rp, "garch_vol_annualized": float(sigma.loc[t])}, sh)
        row = {"date": t, "size": dec.final_size, "ram_score": dec.detectors["ram"].score,
               "psa_score": dec.detectors["psa"].score, "gso_score": dec.detectors["gso"].score,
               "calm_prob": float(gg[lo]), "stress_prob": float(gg[mids].sum()) if mids else 0.0,
               "crisis_prob": float(gg[hi]), "garch_sigma": float(sigma.loc[t])}
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
    sh8 = metrics.sharpe(run_backtest(oos_ret, m["size"], signal_lag=1)["net_return"])
    sh10 = metrics.sharpe(run_backtest(oos_ret, np.sign(p1 - 0.5).reindex(m.index), signal_lag=1)["net_return"])
    return {"M8": round(sh8, 3), "M10": round(sh10, 3), "M10_minus_M8": round(sh10 - sh8, 3),
            "drift_oos": round(float(oos_ret.mean() * 252), 2)}


def main():
    from scipy.stats import spearmanr
    res = {}
    for K in (2, 3):
        rows = []
        for tk in PANEL:
            try:
                r = run(tk, K); r["ticker"] = tk; rows.append(r)
                print(f"K={K} {tk:6} drift={r['drift_oos']:+.2f} M8={r['M8']:+.2f} M10={r['M10']:+.2f} M10−M8={r['M10_minus_M8']:+.2f}")
            except Exception as e:  # noqa: BLE001
                print(f"K={K} {tk}: ERROR {e!r}")
        dr = np.array([r["drift_oos"] for r in rows]); dd = np.array([r["M10_minus_M8"] for r in rows])
        rho, p = spearmanr(dr, dd)
        res[K] = {"per_asset": rows, "rho_drift_vs_M10minusM8": round(float(rho), 3), "p": round(float(p), 3),
                  "n_M8_beats_M10": int((dd < 0).sum())}
        print(f"  → K={K}: ρ(drift, M10−M8)={rho:+.2f} (p={p:.2f}); M8 bate a M10 en {int((dd<0).sum())}/{len(rows)} activos\n")
    print("INTERPRETACIÓN: ρ_K2 fuerte y negativo + ρ_K3≈0 ⇒ K=3 supervisa (no condicional al drift).")
    print("Si ρ_K3 también fuerte y negativo ⇒ K=3 también cabalga el drift (la afirmación de §12 sería falsa).")
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    Path("outputs/experiments/drift_test_k2k3.json").write_text(json.dumps(res, indent=2, default=str))
    print("Escrito: outputs/experiments/drift_test_k2k3.json")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
