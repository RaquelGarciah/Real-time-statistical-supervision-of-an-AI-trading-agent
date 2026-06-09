"""Estudio panel K=2 vs K=3 del HMM de régimen (pre-registro BITACORA 2026-06-08).

¿La ventaja nominal del HMM binario sobre el de tres estados en SPY generaliza al panel,
o es un artefacto del OOS alcista de SPY? Para cada activo de ``cache/agent/`` se calibra
un HMM K=2 y K=3 y un GARCH sobre su propio histórico (2000→2024-09), se re-calibra el
gate τ por (activo, K) con la misma isotónica que el notebook canónico, y se ejecuta M8
(override-C) en el OOS con cada HMM. El único grado de libertad es K∈{2,3}.

Uso: ``python experiments/k_ablation_panel.py``  → escribe outputs/experiments/k_ablation_panel.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raíz del repo en el path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config
from config import CACHE_AGENT_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import diebold_mariano, mcnemar_test
from strata.strata import StrataSupervisor
from strata.types import AgentOutput

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
GRID = np.linspace(0, 1, 501)


def calibrate_tau(Ca: np.ndarray, Cr: np.ndarray, rn: np.ndarray, nbins: int = 10) -> float:
    """τ data-driven: isotónica creciente sobre la curva de fiabilidad direccional por bins.

    Misma metodología que §4 del notebook canónico: dirección dominante del régimen
    (long si Calma≥Crisis) con confianza c=máx(Calma,Crisis); primer cruce de 0.5.
    """
    conf = np.maximum(Ca, Cr)
    correct = np.where(Ca >= Cr, rn > 0, rn < 0).astype(float)
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
            confidence=d["confidence"], reasoning=d.get("reasoning", ""), personalities={})
    return out


def _latest_end(ticker: str) -> str:
    ps = sorted(glob.glob(str(DATA_DIR / f"{ticker}_{CALIBRATION_START}_*.parquet")))
    return ps[-1].rsplit("_", 1)[1].replace(".parquet", "")


def _supervise_panel(agents, greg, sigma, tau, k2):
    # Devuelve un DataFrame indexado por fecha con la posición supervisada (override-C).
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=(tau / 2, tau, 0.70))
    rows, sh = [], []
    for t in sorted(agents):
        if t not in greg.index or t not in sigma.index:
            continue
        a = agents[t]; sh.append(a.size); gg = greg.loc[t]
        if k2:
            rp = {"calm_prob": float(gg["S0"]), "stress_prob": 0.0,
                  "crisis_prob": float(gg["S1"]), "viterbi_state": int(gg["S1"] > 0.5)}
        else:
            rp = {"calm_prob": float(gg["Calma"]), "stress_prob": float(gg["Estrés"]),
                  "crisis_prob": float(gg["Crisis"]), "viterbi_state": int(np.argmax(gg.values))}
        dec = sup.supervise(a, {"regime": rp, "garch_vol_annualized": float(sigma.loc[t])}, sh)
        rows.append({"date": t, "size": dec.final_size, "intv": dec.was_intervened})
    return pd.DataFrame(rows).set_index("date")


def run_ticker(ticker: str) -> dict:
    prices = data.load_market_data(ticker, CALIBRATION_START, _latest_end(ticker))
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]

    h3 = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    h2 = RegimeHMM(n_states=2, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])

    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)

    g3 = pd.DataFrame(h3.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                      columns=["Calma", "Estrés", "Crisis"])
    g2 = pd.DataFrame(h2.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                      columns=["S0", "S1"])

    rn = ret.shift(-1)
    c3 = g3.loc[g3.index <= pd.Timestamp(CALIBRATION_END)]
    r3 = rn.reindex(c3.index); m3 = r3.notna().to_numpy()
    tau3 = calibrate_tau(c3["Calma"].to_numpy()[m3], c3["Crisis"].to_numpy()[m3], r3.to_numpy()[m3])
    c2 = g2.loc[g2.index <= pd.Timestamp(CALIBRATION_END)]
    r2 = rn.reindex(c2.index); mm2 = r2.notna().to_numpy()
    tau2 = calibrate_tau(c2["S0"].to_numpy()[mm2], c2["S1"].to_numpy()[mm2], r2.to_numpy()[mm2])
    # Si la fiabilidad nunca cruza 0.5 (régimen no informativo), gate conservador 0.5.
    tau3 = 0.5 if not np.isfinite(tau3) else tau3
    tau2 = 0.5 if not np.isfinite(tau2) else tau2

    agents = load_agent(ticker)
    m_k3 = _supervise_panel(agents, g3, sigma, tau3, k2=False)
    m_k2 = _supervise_panel(agents, g2, sigma, tau2, k2=True)
    idx = m_k3.index
    y = np.sign(oos_ret.shift(-1).reindex(idx))
    v = (y.notna() & (y != 0)).to_numpy()

    def met(df):
        nr = run_backtest(oos_ret, df["size"], signal_lag=1)["net_return"]
        corr = (np.sign(df["size"]).reindex(idx) == y)[v].to_numpy()
        return nr, metrics.sharpe(nr), float(corr.mean()), int(df["intv"].sum()), corr

    nr3, sh3, acc3, int3, corr3 = met(m_k3)
    nr2, sh2v, acc2, int2, corr2 = met(m_k2)
    _, pmc, b, c = mcnemar_test(corr3, corr2)  # b: K3✓K2✗; c: K3✗K2✓
    common = nr3.index.intersection(nr2.index)
    dm, dmp = diebold_mariano((-nr3.loc[common]).to_numpy(), (-nr2.loc[common]).to_numpy())

    return {
        "ticker": ticker, "n": int(v.sum()),
        "tau_k3": round(tau3, 3), "tau_k2": round(tau2, 3),
        "sharpe_k3": round(sh3, 3), "sharpe_k2": round(sh2v, 3),
        "d_sharpe_k3_minus_k2": round(sh3 - sh2v, 3),
        "acc_k3": round(acc3, 3), "acc_k2": round(acc2, 3),
        "int_k3": int3, "int_k2": int2,
        "mcnemar_p": round(float(pmc), 4), "mcnemar_b_k3win": int(b), "mcnemar_c_k2win": int(c),
        "dm_p": round(float(dmp), 4),
        "k2_better_sharpe": bool(sh2v > sh3),
    }


def main() -> None:
    rows = []
    for tk in PANEL:
        try:
            r = run_ticker(tk)
        except Exception as e:  # noqa: BLE001 — un activo no debe tumbar el panel
            r = {"ticker": tk, "error": repr(e)}
        rows.append(r)
        print(r)

    ok = [r for r in rows if "error" not in r]
    d_sh = np.array([r["d_sharpe_k3_minus_k2"] for r in ok])
    agg = {
        "n_assets": len(ok),
        "n_k2_better_sharpe": int(sum(r["k2_better_sharpe"] for r in ok)),
        "median_d_sharpe_k3_minus_k2": round(float(np.median(d_sh)), 3),
        "mean_d_sharpe_k3_minus_k2": round(float(np.mean(d_sh)), 3),
        "n_dm_signif_p10": int(sum(r["dm_p"] < 0.10 for r in ok)),
        "n_mcnemar_signif_p10": int(sum(r["mcnemar_p"] < 0.10 for r in ok)),
    }
    print("\nAGREGADO:", agg)

    out = {"pre_registro": "BITACORA 2026-06-08 estudio panel K=2 vs K=3", "per_asset": rows,
           "aggregate": agg, "oos_start": STRATA_OOS_START, "signal_lag": 1, "seed": config.SEED}
    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "k_ablation_panel.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nEscrito: {dst / 'k_ablation_panel.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
