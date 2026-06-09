"""Experimento K-por-activo: selección ex-ante del nº de regímenes (pre-registro BITACORA 2026-06-08).

Por cada activo del panel y cada K en {2,3,4} se calibra un HMM y un GARCH sobre el propio
histórico (2000→2024-09), se calibra el gate τ_K con la isotónica de §4, y se mide la
informatividad direccional del régimen EN CALIBRACIÓN (acc_at_gate). El K* se elige ex-ante
por ``argmax acc_at_gate`` (con guardia de soporte y desempate por parsimonia), SIN mirar el
OOS ni el P&L. Como diagnóstico —no como selección— se compara K* con el K mejor-OOS por activo.

Uso: ``python experiments/k_per_asset.py`` → outputs/experiments/k_per_asset.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config
from config import CACHE_AGENT_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import diebold_mariano
from strata.strata import StrataSupervisor
from strata.types import AgentOutput

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
KS = [2, 3, 4]
GRID = np.linspace(0, 1, 501)
MIN_FIRE_FRAC = 0.05  # soporte mínimo de días disparados en calibración para fiarse del K


def _dominant(gamma: pd.DataFrame):
    # Confianza c=máx de los DOS estados direccionales extremos (menor y mayor vol) y su dirección.
    lo, hi = gamma.columns[0], gamma.columns[-1]
    Ca, Cr = gamma[lo].to_numpy(), gamma[hi].to_numpy()
    return np.maximum(Ca, Cr), (Ca >= Cr)


def calibrate_tau(conf: np.ndarray, long_call: np.ndarray, rn: np.ndarray, nbins: int = 10) -> float:
    correct = np.where(long_call, rn > 0, rn < 0).astype(float)
    edges = np.linspace(0, 1, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    acc = np.array([correct[(conf >= lo) & (conf < hi)].mean()
                    if ((conf >= lo) & (conf < hi)).sum() else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])
    cnt = np.array([int(((conf >= lo) & (conf < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])])
    keep = ~np.isnan(acc)
    if keep.sum() < 2:
        return float("nan")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        mids[keep], acc[keep], sample_weight=cnt[keep])
    cross = GRID[iso.predict(GRID) >= 0.5]
    return float(cross.min()) if len(cross) else float("nan")


def load_agent(ticker: str) -> dict:
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        d = json.load(open(fp))
        out[pd.Timestamp(d["date"])] = AgentOutput(
            date=d["date"], ticker=d["ticker"], action=d["action"], size=d["size"],
            confidence=d["confidence"], reasoning="", personalities={})
    return out


def _latest_end(ticker: str) -> str:
    ps = sorted(glob.glob(str(DATA_DIR / f"{ticker}_{CALIBRATION_START}_*.parquet")))
    return ps[-1].rsplit("_", 1)[1].replace(".parquet", "")


def _run_override(agents, gamma, sigma, tau):
    # M8 override-C con un HMM de K estados: calm_prob = estado de MENOR vol, crisis_prob = MAYOR vol,
    # el resto (estados intermedios) van a stress_prob (la zona de abstención de RAM).
    cols = list(gamma.columns)
    lo, hi, mids = cols[0], cols[-1], cols[1:-1]
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=(tau / 2, tau, 0.70))
    rows, sh = [], []
    for t in sorted(agents):
        if t not in gamma.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); g = gamma.loc[t]
        rp = {"calm_prob": float(g[lo]), "crisis_prob": float(g[hi]),
              "stress_prob": float(g[mids].sum()) if mids else 0.0,
              "viterbi_state": int(np.argmax(g.values))}
        dec = sup.supervise(a, {"regime": rp, "garch_vol_annualized": float(sigma.loc[t])}, sh)
        rows.append({"date": t, "size": dec.final_size, "intv": dec.was_intervened})
    return pd.DataFrame(rows).set_index("date")


def run_ticker(ticker: str) -> dict:
    prices = data.load_market_data(ticker, CALIBRATION_START, _latest_end(ticker))
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    agents = load_agent(ticker)
    rn_full = ret.shift(-1)

    per_k = {}
    for K in KS:
        hmm = RegimeHMM(n_states=K, seed=config.SEED).fit(calib.to_numpy())
        gamma = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                             columns=[f"s{j}" for j in range(K)])
        # τ_K y métricas ex-ante (calibración) de informatividad direccional.
        gcal = gamma.loc[gamma.index <= pd.Timestamp(CALIBRATION_END)]
        conf_c, long_c = _dominant(gcal)
        rnc = rn_full.reindex(gcal.index).to_numpy()
        ok = ~np.isnan(rnc)
        conf_c, long_c, rnc = conf_c[ok], long_c[ok], rnc[ok]
        tau = calibrate_tau(conf_c, long_c, rnc)
        tau = 0.5 if not np.isfinite(tau) else tau
        fired = conf_c >= tau
        correct = np.where(long_c, rnc > 0, rnc < 0).astype(float)
        acc_at_gate = float(correct[fired].mean()) if fired.sum() else float("nan")
        fire_frac = float(fired.mean())
        # OOS downstream.
        m = _run_override(agents, gamma, sigma, tau)
        idx = m.index
        y = np.sign(oos_ret.shift(-1).reindex(idx))
        v = (y.notna() & (y != 0)).to_numpy()
        nr = run_backtest(oos_ret, m["size"], signal_lag=1)["net_return"]
        acc_oos = float((np.sign(m["size"]).reindex(idx) == y)[v].to_numpy().mean())
        per_k[K] = {
            "tau": round(tau, 3), "acc_at_gate_calib": round(acc_at_gate, 4),
            "fire_frac_calib": round(fire_frac, 3),
            "sharpe_oos": round(metrics.sharpe(nr), 3), "acc_oos": round(acc_oos, 4),
            "int_oos": int(m["intv"].sum()), "n_oos": int(v.sum()),
            "_nr": nr,  # serie para DM (se elimina antes de serializar)
        }

    # K* ex-ante: argmax acc_at_gate con guardia de soporte y desempate por parsimonia.
    elegibles = [K for K in KS if per_k[K]["fire_frac_calib"] >= MIN_FIRE_FRAC
                 and np.isfinite(per_k[K]["acc_at_gate_calib"])]
    if not elegibles:
        elegibles = KS
    best_acc = max(per_k[K]["acc_at_gate_calib"] for K in elegibles)
    k_star = min(K for K in elegibles if per_k[K]["acc_at_gate_calib"] >= best_acc - 1e-9)
    # K mejor-OOS (diagnóstico, NO selección).
    k_best_oos = max(KS, key=lambda K: per_k[K]["sharpe_oos"])

    out = {"ticker": ticker, "k_star_exante": int(k_star), "k_best_oos": int(k_best_oos),
           "match": bool(k_star == k_best_oos),
           "per_k": {K: {kk: vv for kk, vv in per_k[K].items() if kk != "_nr"} for K in KS}}
    out["_nr_kstar"] = per_k[k_star]["_nr"]
    out["_nr_k3"] = per_k[3]["_nr"]
    return out


def main() -> None:
    rows = []
    for tk in PANEL:
        try:
            rows.append(run_ticker(tk))
            r = rows[-1]
            print(f"{tk:6} K*(ex-ante)={r['k_star_exante']}  K(mejor-OOS)={r['k_best_oos']}  "
                  f"match={r['match']}  | "
                  + "  ".join(f"K{K}:Sh={r['per_k'][K]['sharpe_oos']:+.2f},"
                              f"accCal={r['per_k'][K]['acc_at_gate_calib']:.3f}" for K in KS))
        except Exception as e:  # noqa: BLE001
            rows.append({"ticker": tk, "error": repr(e)})
            print(f"{tk}: ERROR {e!r}")

    ok = [r for r in rows if "error" not in r]
    matches = sum(r["match"] for r in ok)
    # Cartera K-por-activo vs K=3 fijo (Sharpe medio OOS por activo) + DM agregado de pérdidas.
    sh_kstar = np.array([r["per_k"][r["k_star_exante"]]["sharpe_oos"] for r in ok])
    sh_k3 = np.array([r["per_k"][3]["sharpe_oos"] for r in ok])
    # DM apilando las pérdidas diarias de todos los activos (kstar vs K=3).
    loss_kstar = np.concatenate([(-r["_nr_kstar"].dropna()).to_numpy() for r in ok])
    loss_k3 = np.concatenate([(-r["_nr_k3"].dropna()).to_numpy() for r in ok])
    n = min(len(loss_kstar), len(loss_k3))
    dm, dmp = diebold_mariano(loss_kstar[:n], loss_k3[:n])
    agg = {
        "n_assets": len(ok),
        "n_match_kstar_vs_bestoos": int(matches),
        "k_star_counts": {str(K): int(sum(r["k_star_exante"] == K for r in ok)) for K in KS},
        "mean_sharpe_kstar": round(float(sh_kstar.mean()), 3),
        "mean_sharpe_k3_fijo": round(float(sh_k3.mean()), 3),
        "median_sharpe_kstar": round(float(np.median(sh_kstar)), 3),
        "median_sharpe_k3_fijo": round(float(np.median(sh_k3)), 3),
        "dm_kstar_vs_k3_p": round(float(dmp), 4),
    }
    print("\nAGREGADO:", json.dumps(agg, indent=0))

    clean = [{k: v for k, v in r.items() if not k.startswith("_nr")} for r in rows]
    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "k_per_asset.json").write_text(json.dumps(
        {"pre_registro": "BITACORA 2026-06-08 K-por-activo", "ks": KS,
         "min_fire_frac": MIN_FIRE_FRAC, "per_asset": clean, "aggregate": agg,
         "oos_start": STRATA_OOS_START, "seed": config.SEED}, indent=2, default=str))
    print(f"\nEscrito: {dst / 'k_per_asset.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
