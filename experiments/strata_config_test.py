"""¿Se alcanza STRATA-U cambiando SOLO la config de los detectores de M8? Verificación.

Corre el MISMO StrataSupervisor (override-C) con dos configuraciones, sin código nuevo:
  - M8        : gso_mode="absolute"  (solo capa sobre-exposición) · ram_thresholds=(0.25,0.5,0.70)
  - STRATA-U? : gso_mode="relative"  (reescala SIEMPRE al objetivo de vol) · ram_thresholds iguales
  - STRATA-U+ : gso_mode="relative"  + ram_thresholds=(0.125,0.25,0.50)  (RAM más asertivo)

Hipótesis: solo cambiar gso_mode absolute→relative (capa de riesgo siempre activa) reproduce el
perfil de STRATA-U (drawdown bajo), porque en relative el GSO también dispara ante infra-exposición
y reescala el size del agente (~0.10 plano) al objetivo de volatilidad cada día.

Métricas sobre la ventana ~250 (mv.index[150:]): accuracy, Sharpe, maxDD + tasa de intervención.
Uso: python experiments/strata_config_test.py
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
from experiments.quant_validation_panel import build_states
from strata.strata import StrataSupervisor

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]
N0 = 150
CONFIGS = {
    "M8":        dict(gso_mode="absolute", ram_thresholds=(0.25, 0.5, 0.70)),
    "STRATA-U?": dict(gso_mode="relative", ram_thresholds=(0.25, 0.5, 0.70)),
    "STRATA-U+": dict(gso_mode="relative", ram_thresholds=(0.125, 0.25, 0.50)),
}
OUT = Path("outputs/experiments/strata_config_test.json")


def _complete(tk):
    import glob
    return len(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))) >= 400


def _run_cfg(agents, gamma, sigma, gso_mode, ram_thresholds) -> pd.DataFrame:
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode=gso_mode,
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=ram_thresholds)
    rows, hist = [], []
    for t in sorted(agents):
        if t not in gamma.index or t not in sigma.index:
            continue
        a = agents[t]; hist.append(a.size); g = gamma.loc[t]
        ms = {"regime": {"calm_prob": float(g["Calma"]), "stress_prob": float(g["Estrés"]),
                         "crisis_prob": float(g["Crisis"]), "viterbi_state": int(np.argmax(g.values))},
              "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, hist)
        rows.append({"date": t, "final_size": dec.final_size, "interv": dec.was_intervened})
    return pd.DataFrame(rows).set_index("date")


def _row(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    gamma, sigma, oos_ret = build_states(tk)
    agents = wf.load_agent(tk)
    r_next = oos_ret.shift(-1)
    out = {}
    for name, cfg in CONFIGS.items():
        df = _run_cfg(agents, gamma, sigma, **cfg)
        df["r_next"] = r_next.reindex(df.index)
        mv = df[df["r_next"].notna() & (np.sign(df["r_next"]) != 0)]
        sub = mv.index[N0:]
        truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
        w = pd.Series(0.0, index=df.index); w.loc[sub] = mv.loc[sub, "final_size"].to_numpy()
        nrx = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        out[name] = {"acc": round(float((np.sign(mv.loc[sub, "final_size"].to_numpy()) == truth).mean()), 4),
                     "sharpe": round(float(sharpe(nrx)), 4), "calmar": round(float(calmar(nrx)), 4),
                     "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                     "interv_rate": round(float(mv.loc[sub, "interv"].mean()), 4),
                     "n": int(len(sub))}
    return out


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            print(f"{tk:5s} (sin caché, omitido)"); continue
        try:
            rows[tk] = _row(tk); r = rows[tk]
            print(f"{tk:5s} | M8: acc={r['M8']['acc']:.3f} Sh={r['M8']['sharpe']:+.2f} dd={r['M8']['maxdd']:.0%} iv={r['M8']['interv_rate']:.0%}"
                  f"  || STRATA-U?(relative): acc={r['STRATA-U?']['acc']:.3f} Sh={r['STRATA-U?']['sharpe']:+.2f} "
                  f"dd={r['STRATA-U?']['maxdd']:.0%} iv={r['STRATA-U?']['interv_rate']:.0%}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    A = list(rows)
    avg = lambda c, m: round(float(np.mean([rows[t][c][m] for t in A])), 4)
    resumen = {c: {m: avg(c, m) for m in ("acc", "sharpe", "maxdd", "calmar", "interv_rate")} for c in CONFIGS}
    res = {"meta": {"activos": A, "ventana": "~250 (mv[150:])", "seed": config.SEED,
                    "configs": {c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()} for c, cfg in CONFIGS.items()},
                    "nota": "MISMO StrataSupervisor (override-C); solo cambia gso_mode y ram_thresholds. "
                            "Verifica si absolute→relative (GSO siempre on) reproduce STRATA-U. Exploratorio."},
           "por_activo": rows, "resumen_medias": resumen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== MEDIAS por config (solo cambiando umbrales/modo de TUS detectores) ===")
    print(f"  {'config':12s}{'acc':>8s}{'Sharpe':>9s}{'maxDD':>9s}{'Calmar':>9s}{'interv':>9s}")
    for c in CONFIGS:
        s = resumen[c]
        print(f"  {c:12s}{s['acc']:>8.3f}{s['sharpe']:>9.2f}{s['maxdd']:>8.1%}{s['calmar']:>9.2f}{s['interv_rate']:>8.1%}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
