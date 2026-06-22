"""Comparación JUSTA: todas las estrategias con el MISMO sizing (vol-target). Zanja STRATA-U vs M8.

Problema detectado: en tablas previas STRATA-U iba vol-targeted (posiciones pequeñas) y el resto a
±1 unidad → su menor drawdown era artefacto del sizing, no de la estrategia. Aquí se aplica a TODAS
la MISMA capa de riesgo (w = dirección · target_vol/σ), de modo que accuracy, Sharpe y maxDD son
comparables y aíslan la calidad de la DIRECCIÓN. Ventana ~250 (mv.index[150:]).

Direcciones: M5=sign(agent), M8=sign(final_size override-C), M10=sign(p1-0.5),
STRATA-U=régimen-led (regime_flip), Régimen=signo data-driven del régimen, ZeroR=clase mayoritaria,
B&H=+1, S&H=-1.  Uso: python experiments/fair_sizing_compare.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22
from experiments.strata_u import (_regime_drift, TARGET_VOL, CAP, TAU_CONF, REG_REL_MIN,
                                   AGENT_REL_MIN, AGENT_MIN_OBS)

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]
N0 = 150
STRATS = ["M5", "M8", "M10", "STRATA-U", "Régimen", "B&H", "S&H", "ZeroR"]
OUT = Path("outputs/experiments/fair_sizing_compare.json")


def _complete(tk):
    import glob
    return len(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))) >= 400


def _row(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    R = reg.reindex(mv.index)
    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy()
    drift = np.where(R["drift"].to_numpy() != 0, R["drift"].to_numpy(), 1.0); gmax = R["gmax"].to_numpy()
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom != 0)
    agent_sign_all = np.sign(mv["agent_size"].to_numpy())
    ag_gate = pd.Series((agent_sign_all == np.sign(mv["r_next"].to_numpy())).astype(float),
                        index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    agent_ok = (np.arange(len(mv)) >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN) & (agent_sign_all != 0)
    su_dir_all = np.where(reg_on, s_dom, drift); su_dir_all = np.where(agent_ok, agent_sign_all, su_dir_all)
    su_dir_all = np.where(su_dir_all == 0, 1.0, su_dir_all)

    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    # direcciones en sub
    sel = np.isin(mv.index, sub)
    dirs = {
        "M5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
        "M8": np.sign(mv.loc[sub, "final_size"].to_numpy()),
        "M10": np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0),
        "STRATA-U": su_dir_all[sel], "Régimen": s_dom[sel],
        "B&H": np.ones_like(truth), "S&H": -np.ones_like(truth), "ZeroR": np.full_like(truth, maj)}
    # MISMO sizing vol-target para todas
    vol_scale = np.where(mv["garch_sigma"].to_numpy() > 0,
                         np.minimum(CAP, TARGET_VOL / mv["garch_sigma"].to_numpy()), CAP)[sel]
    out = {"n": int(sel.sum()), "frac_up": round(frac_up, 4), "estrategias": {}}
    for nm, d in dirs.items():
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = d * vol_scale
        nrx = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        out["estrategias"][nm] = {"acc": round(float((d == truth).mean()), 4),
                                  "sharpe": round(float(sharpe(nrx)), 4),
                                  "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                                  "calmar": round(float(calmar(nrx)), 4)}
    return out


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            print(f"{tk:5s} (sin caché, omitido)"); continue
        try:
            rows[tk] = _row(tk); e = rows[tk]["estrategias"]
            print(f"{tk:5s} | acc " + " ".join(f"{s}={e[s]['acc']:.3f}" for s in ["M8","M10","STRATA-U","Régimen","ZeroR"]) +
                  f" | Sh M8={e['M8']['sharpe']:+.2f} U={e['STRATA-U']['sharpe']:+.2f} Rég={e['Régimen']['sharpe']:+.2f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    A = list(rows)
    avg = lambda s, m: round(float(np.mean([rows[t]["estrategias"][s][m] for t in A])), 4)
    resumen = {s: {m: avg(s, m) for m in ("acc", "sharpe", "maxdd", "calmar")} for s in STRATS}
    res = {"meta": {"activos": A, "ventana": "~250 (mv[150:])", "sizing": "MISMO vol-target para todas (target_vol/σ)",
                    "seed": config.SEED, "nota": "comparación justa: aísla la DIRECCIÓN; Sharpe/maxDD comparables. Exploratorio."},
           "por_activo": rows, "resumen_medias": resumen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== MEDIAS (mismo sizing vol-target para TODAS) ===")
    print(f"  {'estrategia':10s}{'acc':>8s}{'Sharpe':>9s}{'maxDD':>9s}{'Calmar':>9s}")
    for s in STRATS:
        r = resumen[s]
        print(f"  {s:10s}{r['acc']:>8.3f}{r['sharpe']:>9.2f}{r['maxdd']:>8.1%}{r['calmar']:>9.2f}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
