"""K por activo con criterio DIRECCIONAL ex-ante + validación fuera de muestra.

Decide K por activo con el criterio alineado al propósito (dirección, no densidad), calibrado
SIN futuro: la estrategia régimen-dirección fuera de muestra dentro de la calibración
(k_selection_directional.json, ya calculado). Luego VALIDA en el OOS (2024-10+), que solo se
usa para comprobar, no para elegir:
  1. ¿El K elegido en calibración coincide con el K mejor-OOS? (concordancia)
  2. ¿La cartera "K-por-activo" bate a K=3 fijo y a K=2 fijo en el OOS?
Override-C con τ=0.5 fijo (gate paramétrico-libre) en todos los casos.

Uso: ``python experiments/k_per_asset_directional.py`` → outputs/experiments/k_per_asset_directional.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR
from core import data, features, metrics
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import diebold_mariano, mcnemar_test
from strata.strata import StrataSupervisor
from strata.types import AgentOutput

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
TAU = 0.5  # gate paramétrico-libre (regla de mayoría), fijo para todos


def load_agent(tk):
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))):
        d = json.load(open(fp))
        out[pd.Timestamp(d["date"])] = AgentOutput(date=d["date"], ticker=d["ticker"], action=d["action"],
                                                   size=d["size"], confidence=d["confidence"], reasoning="",
                                                   personalities={})
    return out


def oos_m8(tk, K):
    end = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(tk, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"]); rv = features.realized_vol_annualized(ret, 21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(config.STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    hmm = RegimeHMM(n_states=K, seed=config.SEED).fit(calib.to_numpy())
    g = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                     columns=[f"s{j}" for j in range(K)])
    agents = load_agent(tk)
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=(TAU / 2, TAU, 0.70))
    rows, sh = [], []
    lo, hi = f"s0", f"s{K-1}"; mids = [f"s{j}" for j in range(1, K - 1)]
    for t in sorted(agents):
        if t not in g.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); gg = g.loc[t]
        rp = {"calm_prob": float(gg[lo]), "crisis_prob": float(gg[hi]),
              "stress_prob": float(gg[mids].sum()) if mids else 0.0,
              "viterbi_state": int(np.argmax(gg.values))}
        dec = sup.supervise(a, {"regime": rp, "garch_vol_annualized": float(sigma.loc[t])}, sh)
        rows.append({"date": t, "size": dec.final_size, "agent": a.size, "intv": dec.was_intervened})
    m = pd.DataFrame(rows).set_index("date")
    y = np.sign(oos_ret.shift(-1).reindex(m.index)); v = (y.notna() & (y != 0)).to_numpy()
    nr = run_backtest(oos_ret, m["size"], signal_lag=1)["net_return"]
    corr5 = (np.sign(m["agent"]) == y)[v].to_numpy()
    corr8 = (np.sign(m["size"]) == y)[v].to_numpy()
    _, pmc, _, _ = mcnemar_test(corr5, corr8)
    return {"sharpe": round(metrics.sharpe(nr), 3), "acc": round(float(corr8.mean()), 4),
            "interv": int(m["intv"].sum()), "mcnemar_p": round(float(pmc), 4),
            "nr": nr, "agent_nr": run_backtest(oos_ret, m["agent"], signal_lag=1)["net_return"]}


def main():
    dir_sel = {r["ticker"]: r for r in json.load(open("outputs/experiments/k_selection_directional.json"))["per_asset"]
               if "error" not in r}
    rows = []
    for tk in PANEL:
        calib_k = dir_sel[tk]["k_mejor_dir_sharpe"]          # K elegido en CALIBRACIÓN (sin futuro)
        o2, o3 = oos_m8(tk, 2), oos_m8(tk, 3)
        oos_best = 3 if o3["sharpe"] > o2["sharpe"] else 2
        chosen = o3 if calib_k == 3 else o2
        rows.append({"ticker": tk, "calib_K": calib_k, "oos_best_K": oos_best, "match": calib_k == oos_best,
                     "dir_acc_K2": dir_sel[tk]["dir_acc_K2"], "dir_acc_K3": dir_sel[tk]["dir_acc_K3"],
                     "dir_sharpe_K2": dir_sel[tk]["dir_sharpe_K2"], "dir_sharpe_K3": dir_sel[tk]["dir_sharpe_K3"],
                     "oos_sharpe_K2": o2["sharpe"], "oos_sharpe_K3": o3["sharpe"],
                     "oos_sharpe_chosen": chosen["sharpe"], "oos_acc_chosen": chosen["acc"],
                     "oos_mcnemar_chosen": chosen["mcnemar_p"],
                     "_nr_chosen": chosen["nr"], "_nr_k3": o3["nr"]})
        r = rows[-1]
        print(f"{tk:6} calibK={r['calib_K']} oosBestK={r['oos_best_K']} match={r['match']} | "
              f"OOS Sharpe: chosen={r['oos_sharpe_chosen']:+.2f} K3={r['oos_sharpe_K3']:+.2f} K2={r['oos_sharpe_K2']:+.2f}")

    match = sum(r["match"] for r in rows)
    sh_chosen = np.array([r["oos_sharpe_chosen"] for r in rows])
    sh_k3 = np.array([r["oos_sharpe_K3"] for r in rows])
    sh_k2 = np.array([r["oos_sharpe_K2"] for r in rows])
    # DM apilando pérdidas diarias OOS: cartera K-por-activo vs K=3 fijo.
    loss_ch = np.concatenate([(-r["_nr_chosen"].dropna()).to_numpy() for r in rows])
    loss_k3 = np.concatenate([(-r["_nr_k3"].dropna()).to_numpy() for r in rows])
    n = min(len(loss_ch), len(loss_k3))
    dm, dmp = diebold_mariano(loss_ch[:n], loss_k3[:n])
    agg = {"n_assets": len(rows), "n_match_calibK_vs_oosbestK": int(match),
           "mean_oos_sharpe_perasset": round(float(sh_chosen.mean()), 3),
           "mean_oos_sharpe_fixedK3": round(float(sh_k3.mean()), 3),
           "mean_oos_sharpe_fixedK2": round(float(sh_k2.mean()), 3),
           "median_oos_sharpe_perasset": round(float(np.median(sh_chosen)), 3),
           "median_oos_sharpe_fixedK3": round(float(np.median(sh_k3)), 3),
           "dm_perasset_vs_k3_p": round(float(dmp), 4)}
    print(f"\nConcordancia calibK vs oosBestK: {match}/{len(rows)}")
    print(f"OOS Sharpe medio — K-por-activo: {agg['mean_oos_sharpe_perasset']:+.3f}  "
          f"K3 fijo: {agg['mean_oos_sharpe_fixedK3']:+.3f}  K2 fijo: {agg['mean_oos_sharpe_fixedK2']:+.3f}")
    print(f"DM (pérdida diaria) K-por-activo vs K3 fijo: p={agg['dm_perasset_vs_k3_p']:.3f}")

    clean = [{k: v for k, v in r.items() if not k.startswith("_nr")} for r in rows]
    out = {"tau": TAU, "per_asset": clean, "aggregate": agg,
           "criterio_K": "direccional held-out en calibracion (k_mejor_dir_sharpe), validado en OOS"}
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    Path("outputs/experiments/k_per_asset_directional.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nEscrito: outputs/experiments/k_per_asset_directional.json")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
