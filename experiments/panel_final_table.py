"""Tabla consolidada del set canónico de Raquel sobre la ventana de 250 días.

Estrategias (todas sobre el MISMO tramo, ~250 d tras burn-in 150 del walk-forward de M10):
  M10 (XGBoost 22 features, embargo=1), M8 (STRATA override-C), STRATA-U (régimen+vol gateado,
  variante regime_flip), Régimen (signo data-driven del régimen dominante), y triviales
  B&H / S&H / ZeroR (clase mayoritaria). M5 (agente) se incluye como referencia.

Para cada activo: accuracy direccional, Sharpe, equity, máx. drawdown. Escribe
outputs/experiments/panel_final_table.json. Uso: python experiments/panel_final_table.py
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
from config import CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22
from experiments.strata_u import (_regime_drift, TARGET_VOL, CAP, TAU_CONF, REG_REL_MIN,
                                   AGENT_REL_MIN, AGENT_MIN_OBS, DERISK)

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]   # se saltan los que no tengan caché completa
STRATS = ["M10", "M8", "STRATA-U", "Régimen", "B&H", "S&H", "ZeroR", "M5"]
OUT = Path("outputs/experiments/panel_final_table.json")


def _complete(tk: str) -> bool:
    from config import CACHE_AGENT_DIR
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
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0

    # STRATA-U (regime_flip) sobre todo mv, luego se restringe a sub
    R = reg
    agent_sign_all = np.sign(mv["agent_size"].to_numpy())
    ag_gate = pd.Series((agent_sign_all == np.sign(mv["r_next"].to_numpy())).astype(float),
                        index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    ag_obs = np.arange(len(mv))
    s_dom_all = R["s_dom"].reindex(mv.index).to_numpy(); reg_gate = R["reg_gate"].reindex(mv.index).to_numpy()
    drift = R["drift"].reindex(mv.index).to_numpy(); gmax = R["gmax"].reindex(mv.index).to_numpy()
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom_all != 0)
    agent_ok = (ag_obs >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN) & (agent_sign_all != 0)
    drift_dir = np.where(drift != 0, drift, 1.0)
    base = np.where(reg_on, s_dom_all, drift_dir); base = np.where(agent_ok, agent_sign_all, base)
    base = np.where(base == 0, 1.0, base)
    fac = np.where((~reg_on) & (~agent_ok) & (gmax < TAU_CONF), DERISK, 1.0)
    vol_scale = np.where(mv["garch_sigma"].to_numpy() > 0, np.minimum(CAP, TARGET_VOL / mv["garch_sigma"].to_numpy()), CAP)
    w_su = pd.Series(base * vol_scale * fac, index=mv.index)

    # posiciones (en sub) de cada estrategia
    pos = {
        "M10": np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0),
        "M8": np.sign(mv.loc[sub, "final_size"].to_numpy()),
        "M5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
        "Régimen": s_dom_all[np.isin(mv.index, sub)] if len(sub) == len(mv) else R["s_dom"].reindex(sub).to_numpy(),
        "B&H": np.ones_like(truth), "S&H": -np.ones_like(truth), "ZeroR": np.full_like(truth, maj)}
    out = {"n": int(len(sub)), "frac_up": round(frac_up, 4), "estrategias": {}}

    def metr(nrx, accpos):
        return {"acc": round(float((accpos == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
                "equity": round(float((1 + nrx.fillna(0)).prod()), 4),
                "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                "calmar": round(float(calmar(nrx)), 4)}

    for nm, p in pos.items():
        ws = pd.Series(0.0, index=mv.index); ws.loc[sub] = p
        nrx = run_backtest(oos_ret, ws, signal_lag=1)["net_return"].reindex(sub)
        out["estrategias"][nm] = metr(nrx, p)
    # STRATA-U
    nr_su = run_backtest(oos_ret, w_su, signal_lag=1)["net_return"].reindex(sub)
    out["estrategias"]["STRATA-U"] = metr(nr_su, np.sign(w_su.reindex(sub).to_numpy()))
    return out


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            print(f"{tk:5s} (sin caché completa, se omite)"); continue
        try:
            rows[tk] = _row(tk)
            e = rows[tk]["estrategias"]
            print(f"{tk:5s} n={rows[tk]['n']} up={rows[tk]['frac_up']:.2f} | " +
                  " ".join(f"{s}={e[s]['acc']:.3f}" for s in STRATS), flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    res = {"meta": {"activos": list(rows), "ventana": "~250 d post burn-in (walk-forward M10)",
                    "estrategias": STRATS, "strata_u_variante": "regime_flip",
                    "seed": config.SEED, "signal_lag": 1, "nota": "set canónico de Raquel; exploratorio (docs/)"},
           "por_activo": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # tablas resumen accuracy y Sharpe
    for metric in ("acc", "sharpe"):
        print(f"\n=== {metric.upper()} ===")
        print("activo " + "".join(f"{s:>9s}" for s in STRATS))
        for tk in rows:
            e = rows[tk]["estrategias"]
            print(f"{tk:6s}" + "".join(f"{e[s][metric]:>9.3f}" for s in STRATS))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
