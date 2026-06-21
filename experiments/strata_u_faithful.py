"""STRATA-U FIEL: M8 y STRATA-U como dos puntos de UN eje (tasa de intervención).

Demuestra que M8 y STRATA-U son el mismo marco de supervisión con la palanca de intervención en
posiciones distintas, NO dos estrategias. Define STRATA-U FIEL (mantiene al agente como núcleo,
fiel al título "supervisión del agente"):
  - dirección por DEFECTO = la del agente (signo de agent_size);
  - la capa estadística OVERRIDE la dirección hacia el signo del régimen SOLO donde el régimen está
    identificado-fiable (gate causal expansible + confianza) Y el agente lo contradice (o se abstiene);
  - GSO siempre activo: tamaño = vol_target/σ (capa de riesgo permanente).
Misma lógica de intervención que M8 (RAM corrige al agente, GSO acota), pero (1) interviene allí donde
el régimen es fiable —no solo en severidad alta—, (2) gateada por fiabilidad, (3) GSO siempre on.

Mide la TASA DE INTERVENCIÓN (fracción de días en que la dirección difiere de la del agente) de:
M8 (la real, de run_master), STRATA-U fiel, y STRATA-U agresivo (régimen al mando) → el eje.

Ventana: ~250 d post burn-in (mv.index[150:]) para alinear con la tabla de M10. Sin M10 (rápido).
Uso: python experiments/strata_u_faithful.py
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
from experiments.strata_u import (_regime_drift, TARGET_VOL, CAP, TAU_CONF, REG_REL_MIN,
                                   AGENT_REL_MIN, AGENT_MIN_OBS, DERISK)

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]
N0 = 150
OUT = Path("outputs/experiments/strata_u_faithful.json")


def _complete(tk):
    import glob
    return len(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))) >= 400


def _row(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    R = reg.reindex(mv.index)
    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy()
    drift = np.where(R["drift"].to_numpy() != 0, R["drift"].to_numpy(), 1.0)
    gmax = R["gmax"].to_numpy()
    agent_dir = np.sign(mv["agent_size"].to_numpy())
    m8_dir = np.sign(mv["final_size"].to_numpy())
    interv_m8_all = mv["intervenido"].to_numpy().astype(bool)
    vol = mv["garch_sigma"].to_numpy()
    truth_all = np.sign(mv["r_next"].to_numpy())
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom != 0)

    # agente fiable (expansible) para el tilt en la variante agresiva
    ag_gate = pd.Series((agent_dir == truth_all).astype(float), index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    agent_ok = (np.arange(len(mv)) >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN) & (agent_dir != 0)

    # --- STRATA-U FIEL: default = agente; override a régimen donde fiable y contradice/abstiene ---
    override = reg_on & ((agent_dir == 0) | (s_dom != agent_dir))
    dir_faithful = np.where(override, s_dom, agent_dir)
    dir_faithful = np.where(dir_faithful == 0, 0.0, dir_faithful)   # si agente hold y régimen off → flat

    # --- STRATA-U AGRESIVO (regime_flip): régimen manda; agente solo tilt fiable ---
    base_aggr = np.where(reg_on, s_dom, drift)
    base_aggr = np.where(agent_ok, agent_dir, base_aggr)
    dir_aggr = np.where(base_aggr == 0, 1.0, base_aggr)

    vol_scale = np.where(vol > 0, np.minimum(CAP, TARGET_VOL / vol), CAP)

    sub = mv.index[N0:]
    msk = np.isin(mv.index, sub)
    truth = truth_all[msk]

    def metr(dir_arr, sized=True):
        d = dir_arr[msk]
        w = pd.Series(0.0, index=mv.index)
        w.loc[sub] = (dir_arr * vol_scale)[msk] if sized else d
        nrx = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        return {"acc": round(float((d == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
                "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                "calmar": round(float(calmar(nrx)), 4)}

    # tasas de intervención (dirección ≠ agente) en la ventana de evaluación
    def interv_rate(dir_arr): return round(float((dir_arr[msk] != agent_dir[msk]).mean()), 4)

    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    return {"n": int(msk.sum()), "frac_up": round(frac_up, 4),
            "estrategias": {
                "M5": metr(agent_dir, sized=False), "M8": metr(m8_dir, sized=False),
                "STRATA-U-fiel": metr(dir_faithful, sized=True),
                "STRATA-U-agresivo": metr(dir_aggr, sized=True),
                "Régimen": metr(s_dom, sized=False),
                "ZeroR": metr(np.full(len(mv), maj), sized=False)},
            "intervencion": {"M8": round(float(interv_m8_all[msk].mean()), 4),
                             "STRATA-U-fiel": interv_rate(dir_faithful),
                             "STRATA-U-agresivo": interv_rate(dir_aggr)}}


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            print(f"{tk:5s} (sin caché, omitido)"); continue
        try:
            rows[tk] = _row(tk); r = rows[tk]; iv = r["intervencion"]; e = r["estrategias"]
            print(f"{tk:5s} n={r['n']} | INTERVENCIÓN M8={iv['M8']:.0%} fiel={iv['STRATA-U-fiel']:.0%} "
                  f"agresivo={iv['STRATA-U-agresivo']:.0%} | acc M8={e['M8']['acc']:.3f} "
                  f"fiel={e['STRATA-U-fiel']['acc']:.3f} agr={e['STRATA-U-agresivo']['acc']:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    A = list(rows)
    avg = lambda strat, met: round(float(np.mean([rows[t]["estrategias"][strat][met] for t in A])), 4)
    avg_iv = lambda strat: round(float(np.mean([rows[t]["intervencion"][strat] for t in A])), 4)
    resumen = {"tasa_intervencion_media": {s: avg_iv(s) for s in ("M8", "STRATA-U-fiel", "STRATA-U-agresivo")},
               "acc_media": {s: avg(s, "acc") for s in ("M5", "M8", "STRATA-U-fiel", "STRATA-U-agresivo", "Régimen", "ZeroR")},
               "sharpe_media": {s: avg(s, "sharpe") for s in ("M5", "M8", "STRATA-U-fiel", "STRATA-U-agresivo", "Régimen", "ZeroR")},
               "maxdd_media": {s: avg(s, "maxdd") for s in ("M5", "M8", "STRATA-U-fiel", "STRATA-U-agresivo", "Régimen", "ZeroR")}}
    res = {"meta": {"activos": A, "ventana": "~250 d (mv.index[150:])", "seed": config.SEED,
                    "nota": "M8 y STRATA-U como dos puntos del eje 'tasa de intervención'; mismo marco. "
                            "STRATA-U-fiel = agente-default + override gateado por fiabilidad + GSO siempre on. Exploratorio."},
           "por_activo": rows, "resumen": resumen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== EJE: TASA DE INTERVENCIÓN MEDIA (fracción de días que la capa cambia la dirección del agente) ===")
    for s, v in resumen["tasa_intervencion_media"].items():
        print(f"  {s:20s} {v:.1%}")
    print("\n=== MEDIAS por estrategia ===")
    print(f"  {'estrategia':20s}{'acc':>8s}{'Sharpe':>9s}{'maxDD':>9s}")
    for s in ("M5", "M8", "STRATA-U-fiel", "STRATA-U-agresivo", "Régimen", "ZeroR"):
        print(f"  {s:20s}{resumen['acc_media'][s]:>8.3f}{resumen['sharpe_media'][s]:>9.2f}{resumen['maxdd_media'][s]:>8.1%}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
