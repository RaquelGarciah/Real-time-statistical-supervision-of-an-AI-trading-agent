"""Estabilidad por semilla del HMM en SPY (auditoría harvard-professor 2026-06-08).

¿El resultado canónico de M8 (Sharpe 0.92, McNemar p=0.037) es frágil al óptimo concreto de
Baum-Welch? Re-ajustamos el HMM K=3 sobre SPY con varias semillas base (cada una con el
procedimiento canónico n_seeds=10, best-likelihood), re-calibramos τ y corremos M8 override-C.
Reportamos la dispersión de (τ, Sharpe, McNemar p, intervenciones) y comparamos con el HMM del
pickle congelado (el canónico). Pregunta clave: ¿baila el Sharpe pero el McNemar (acierto)
aguanta significativo?

Uso: ``python experiments/hmm_seed_stability_spy.py`` → outputs/experiments/hmm_seed_stability_spy.json
"""

from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config
from config import CACHE_AGENT_DIR, CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import mcnemar_test
from strata.strata import StrataSupervisor
from strata.types import AgentOutput

TICKER = "SPY"
GRID = np.linspace(0, 1, 501)
BASE_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]


def calibrate_tau(Ca, Cr, rn, nbins=10):
    conf = np.maximum(Ca, Cr)
    correct = np.where(Ca >= Cr, rn > 0, rn < 0).astype(float)
    edges = np.linspace(0, 1, nbins + 1); mids = 0.5 * (edges[:-1] + edges[1:])
    acc = np.array([correct[(conf >= lo) & (conf < hi)].mean()
                    if ((conf >= lo) & (conf < hi)).sum() else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])
    cnt = np.array([int(((conf >= lo) & (conf < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])])
    keep = ~np.isnan(acc)
    if keep.sum() < 2:
        return float("nan")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(mids[keep], acc[keep], sample_weight=cnt[keep])
    cross = GRID[iso.predict(GRID) >= 0.5]
    return float(cross.min()) if len(cross) else float("nan")


def load_agent(ticker):
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        d = json.load(open(fp))
        out[pd.Timestamp(d["date"])] = AgentOutput(date=d["date"], ticker=d["ticker"], action=d["action"],
                                                   size=d["size"], confidence=d["confidence"], reasoning="",
                                                   personalities={})
    return out


def _downstream(hmm, feat, sigma, agents, oos_ret, ret):
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                         columns=["Calma", "Estrés", "Crisis"])
    gcal = gamma.loc[gamma.index <= pd.Timestamp(CALIBRATION_END)]
    rnc = ret.shift(-1).reindex(gcal.index).to_numpy()
    ok = ~np.isnan(rnc)
    tau = calibrate_tau(gcal["Calma"].to_numpy()[ok], gcal["Crisis"].to_numpy()[ok], rnc[ok])
    tau = 0.5 if not np.isfinite(tau) else tau
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=(tau / 2, tau, 0.70))
    rows, sh = [], []
    for t in sorted(agents):
        if t not in gamma.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); g = gamma.loc[t]
        rp = {"calm_prob": float(g["Calma"]), "stress_prob": float(g["Estrés"]),
              "crisis_prob": float(g["Crisis"]), "viterbi_state": int(np.argmax(g.values))}
        dec = sup.supervise(a, {"regime": rp, "garch_vol_annualized": float(sigma.loc[t])}, sh)
        rows.append({"date": t, "size": dec.final_size, "agent": a.size, "intv": dec.was_intervened})
    m = pd.DataFrame(rows).set_index("date")
    y = np.sign(oos_ret.shift(-1).reindex(m.index))
    v = (y.notna() & (y != 0)).to_numpy()
    nr = run_backtest(oos_ret, m["size"], signal_lag=1)["net_return"]
    corr5 = (np.sign(m["agent"]) == y)[v].to_numpy()
    corr8 = (np.sign(m["size"]) == y)[v].to_numpy()
    _, p_mc, b, c = mcnemar_test(corr5, corr8)
    return {"tau": round(tau, 3), "sharpe_m8": round(metrics.sharpe(nr), 3),
            "acc_m8": round(float(corr8.mean()), 4), "interv": int(m["intv"].sum()),
            "mcnemar_p": round(float(p_mc), 4), "mcnemar_b": int(b), "mcnemar_c": int(c),
            "loglik": round(float(hmm.best_score), 2) if hmm.best_score else None}


def main():
    end = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    agents = load_agent(TICKER)

    # Referencia: HMM del pickle congelado (el canónico).
    hmm_pickle = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
    ref = _downstream(hmm_pickle, feat, sigma, agents, oos_ret, ret)
    ref["source"] = "pickle (canónico)"
    print("PICKLE (canónico):", ref)

    rows = []
    for s in BASE_SEEDS:
        hmm = RegimeHMM(n_states=3, seed=s).fit(calib.to_numpy())  # n_seeds=10 interno (procedimiento canónico)
        r = _downstream(hmm, feat, sigma, agents, oos_ret, ret)
        r["seed"] = s
        rows.append(r)
        print(f"seed {s:3}: τ={r['tau']:.3f}  Sharpe={r['sharpe_m8']:+.3f}  acc={r['acc_m8']:.3f}  "
              f"int={r['interv']:3}  McNemar p={r['mcnemar_p']:.4f}  loglik={r['loglik']}")

    tau_a = np.array([r["tau"] for r in rows])
    sh_a = np.array([r["sharpe_m8"] for r in rows])
    p_a = np.array([r["mcnemar_p"] for r in rows])
    agg = {
        "n_seeds": len(rows),
        "tau": {"min": float(tau_a.min()), "median": float(np.median(tau_a)), "max": float(tau_a.max())},
        "sharpe_m8": {"min": float(sh_a.min()), "median": float(np.median(sh_a)), "max": float(sh_a.max())},
        "mcnemar_p": {"min": float(p_a.min()), "median": float(np.median(p_a)), "max": float(p_a.max())},
        "frac_mcnemar_signif_p10": float((p_a < 0.10).mean()),
        "frac_mcnemar_signif_p05": float((p_a < 0.05).mean()),
        "frac_sharpe_positive": float((sh_a > 0).mean()),
    }
    print("\nAGREGADO sobre semillas:", json.dumps(agg, indent=0))
    print("PICKLE para comparar:", {k: ref[k] for k in ("tau", "sharpe_m8", "acc_m8", "interv", "mcnemar_p")})

    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "hmm_seed_stability_spy.json").write_text(json.dumps(
        {"ticker": TICKER, "pickle_ref": ref, "per_seed": rows, "aggregate": agg}, indent=2, default=str))
    print(f"\nEscrito: {dst / 'hmm_seed_stability_spy.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
