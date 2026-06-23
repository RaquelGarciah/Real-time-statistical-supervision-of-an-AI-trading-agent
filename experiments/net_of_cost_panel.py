"""Net-of-cost y turnover del panel-10 — hueco de despliegue #2 (pre-registro BITACORA 2026-06-23).

El notebook no reportaba coste ni rotación. Aquí se reconstruyen las posiciones EXACTAS de las 6
estrategias desde el panel canónico (`correct_by_arm`), sin reentrenar ningún modelo:

    pos_t = signo(r_{t+1}) · (2·acierto_t − 1)         (recupera la posición ±1, idéntica a la canónica)

y se mide, sobre la ventana desplegable [150:] (~250 días, signal_lag=1):
  - turnover anualizado por estrategia = mean(|Δw|)·252;
  - Sharpe y equity neto a coste {0,1,2,5,10,20} bp (coste lineal, core.backtest);
  - coste de break-even del RESCATE de riesgo: bp donde ΔSharpe pooled (arm vs M5) cruza 0.

Tesis a contrastar: la capa de riesgo (M8) rota por ESTADO (poco) y su rescate sobrevive al coste; el
aprendiz diario (M10/AutoML) rota mucho más. Uso: python experiments/net_of_cost_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import experiments.automl_m10 as A
from core.backtest import run_backtest

PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
ARMS = ["m5", "m8", "m10_xgb", "automl", "zeror", "bh"]
LABEL = {"m5": "M5", "m8": "M8", "m10_xgb": "M10", "automl": "AutoML", "zeror": "ZeroR", "bh": "B&H"}
COSTS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
PANEL_JSON = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
OUT = Path("outputs/experiments/net_of_cost_panel.json")


def _sr(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    return float(np.sqrt(252) * r.mean() / sd) if sd > 0 else 0.0


def positions(tk: str, entry: dict):
    """Posiciones ±1 reconstruidas de las 6 estrategias + oos_ret y el índice desplegable td."""
    A.wf.TICKER = tk
    A.wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = A.build_states_onthefly(tk)
    m = A.wf.run_master(gamma_df, sigma, oos_ret, A.wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)]
    td = mv.index[A.N0:]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    pos = {}
    for arm in ARMS:
        corr = np.array(entry["correct_by_arm"][arm], float)
        if len(corr) != len(td):
            raise ValueError(f"{tk}/{arm}: {len(corr)} ≠ {len(td)}")
        pos[arm] = truth * (2 * corr - 1)
    return pos, oos_ret, m.index, td, truth


def _net(oos_ret, full_idx, td, pos, cost) -> np.ndarray:
    w = pd.Series(0.0, index=full_idx); w.loc[td] = pos
    return run_backtest(oos_ret, w, cost_bps=cost, signal_lag=1)["net_return"].reindex(td).to_numpy()


def run_ticker(tk: str, entry: dict) -> dict:
    pos, oos_ret, full_idx, td, truth = positions(tk, entry)
    out = {"n": int(len(td)), "turnover_ann": {}, "net_by_cost": {}, "_nr0": {}}
    for arm in ARMS:
        turn = float(pd.Series(pos[arm]).diff().abs().mean()) * 252
        out["turnover_ann"][LABEL[arm]] = round(turn, 2)
        out["net_by_cost"][LABEL[arm]] = {}
        for c in COSTS:
            nr = _net(oos_ret, full_idx, td, pos[arm], c)
            out["net_by_cost"][LABEL[arm]][f"{c:g}bp"] = {
                "sharpe": round(_sr(nr), 3), "equity": round(float(np.nanprod(1 + np.nan_to_num(nr))), 4)}
        out["_nr0"][arm] = pos[arm]  # posiciones para el pooled (se quitan antes de serializar)
    # validación de identidad: accuracy reconstruida == canónica
    for arm in ARMS:
        acc = float((pos[arm] == truth).mean())
        assert abs(acc - entry["table"][arm]["accuracy"]) < 1e-3, f"{tk}/{arm}: {acc} vs canon"
    out["_pos"] = pos
    out["_oos"] = oos_ret
    out["_full"] = full_idx
    out["_td"] = td
    return out


def main() -> None:
    P = json.load(open(PANEL_JSON))["por_activo"]
    res, blobs = {}, {}
    for tk in PANEL10:
        r = run_ticker(tk, P[tk])
        blobs[tk] = r
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        res[tk] = clean
        t = r["turnover_ann"]
        print(f"{tk:5s} turnover/yr  M5={t['M5']:.1f} M8={t['M8']:.1f} M10={t['M10']:.1f} "
              f"AutoML={t['AutoML']:.1f} ZeroR={t['ZeroR']:.1f} | "
              f"Sharpe@10bp M8={r['net_by_cost']['M8']['10bp']['sharpe']:+.2f} "
              f"M10={r['net_by_cost']['M10']['10bp']['sharpe']:+.2f} "
              f"AutoML={r['net_by_cost']['AutoML']['10bp']['sharpe']:+.2f}", flush=True)

    # Pooled: concatena posiciones y retornos del panel, recalcula a cada coste.
    def pooled_net(arm_key, cost):
        nrs = []
        for tk in PANEL10:
            b = blobs[tk]
            nrs.append(_net(b["_oos"], b["_full"], b["_td"], b["_pos"][arm_key], cost))
        return np.concatenate(nrs)

    pooled = {"sharpe_vs_cost": {}, "dSharpe_vs_m5_vs_cost": {}, "turnover_ann_medio": {}}
    for c in COSTS:
        nr_m5 = pooled_net("m5", c)
        pooled["sharpe_vs_cost"][f"{c:g}bp"] = {LABEL[a]: round(_sr(pooled_net(a, c)), 3) for a in ARMS}
        pooled["dSharpe_vs_m5_vs_cost"][f"{c:g}bp"] = {
            LABEL[a]: round(_sr(pooled_net(a, c)) - _sr(nr_m5), 3) for a in ARMS if a != "m5"}
    pooled["turnover_ann_medio"] = {
        LABEL[a]: round(float(np.mean([blobs[tk]["turnover_ann"][LABEL[a]] for tk in PANEL10])), 2)
        for a in ARMS}

    # Break-even del rescate: coste (bp) donde ΔSharpe pooled (arm vs M5) cruza 0.
    def breakeven(arm_key) -> float:
        f = lambda c: _sr(pooled_net(arm_key, c)) - _sr(pooled_net("m5", c))
        if f(0.0) <= 0:
            return 0.0
        lo, hi = 0.0, 500.0
        if f(hi) > 0:
            return float("inf")
        for _ in range(40):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if f(mid) > 0 else (lo, mid)
        return round(lo, 1)

    pooled["breakeven_rescate_bps"] = {LABEL[a]: breakeven(a) for a in ARMS if a not in ("m5",)}

    out = {"meta": {"panel": PANEL10, "ventana": "desplegable [150:] (~250d)", "costs_bps": COSTS,
                    "metodo": "pos = signo(r_{t+1})·(2·acierto−1) desde panel canónico; sin reentrenar; "
                              "coste lineal core.backtest; signal_lag=1",
                    "pre_registro": "BITACORA 2026-06-23"},
           "por_activo": res, "pooled": pooled}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== POOLED panel-10 ===")
    print("  turnover/yr medio:", pooled["turnover_ann_medio"])
    for c in ("0bp", "1bp", "10bp"):
        print(f"  ΔSharpe vs M5 @{c}: M8={pooled['dSharpe_vs_m5_vs_cost'][c]['M8']:+.2f} "
              f"M10={pooled['dSharpe_vs_m5_vs_cost'][c]['M10']:+.2f} "
              f"AutoML={pooled['dSharpe_vs_m5_vs_cost'][c]['AutoML']:+.2f}")
    print("  break-even del rescate (bp):", pooled["breakeven_rescate_bps"])
    print(f"OK · {OUT}")


if __name__ == "__main__":
    A.config.set_seeds(A.config.SEED)
    main()
