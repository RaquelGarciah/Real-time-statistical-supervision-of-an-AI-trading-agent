"""Canal régimen (solo precio) en un universo HELD-OUT — generalización de la ley leverage→canal + coste.

Pre-registro: BITACORA 2026-06-23. El régimen (HMM) y la vol (GARCH) solo necesitan precio, así que
esto corre para 12 activos con precio pero SIN decisiones del agente, nunca usados para construir la ley
naturaleza→canal. Dos propósitos:

  1. Generalización: ¿leverage_corr (calibración congelada) sigue correlacionando con la ventaja del
     régimen fuera del panel-15? (criterio de éxito/fracaso prior-flip en el pre-registro).
  2. Coste/turnover del régimen-solo: turnover anualizado, equity neta a 0/1/5/10 bp y coste de
     break-even (bp donde el net-return del régimen cruza 0). El régimen rota POR ESTADO, no a diario.

Estrategia 'régimen crudo': posición = signo de la media de calibración del régimen dominante del día
(prior data-driven congelado ≤2024-09; causal, signal_lag=1). Baselines: B&H y ZeroR (clase mayoritaria).

Uso: python experiments/regime_channel_heldout.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

import config
from config import CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import equity_curve, max_drawdown, sharpe
from experiments.leverage_screen import screen_ticker
from experiments.quant_validation_panel import build_states

# Universo held-out: precio sí, decisiones del agente no, y nunca tocado para construir el canal.
HELDOUT = ["META", "NFLX", "INTC", "AMD", "COIN", "GME", "RIOT", "PLTR", "SNOW", "SHOP", "PYPL", "ARKK"]
CALIB_CORTA = {"COIN", "PLTR", "SNOW"}  # inicio 2020-21: menos transiciones de régimen (caveat)
REGNAMES = ["Calma", "Estrés", "Crisis"]
COST_GRID = [0.0, 1.0, 5.0, 10.0]
OUT = Path("outputs/experiments/regime_channel_heldout.json")


def _breakeven_bps(pos: np.ndarray, oos_ret: pd.Series, idx) -> float:
    """Coste (bp) donde el net-return acumulado del régimen cruza 0. inf si nunca es positivo a 0bp."""
    w = pd.Series(0.0, index=oos_ret.index); w.loc[idx] = pos
    gross = run_backtest(oos_ret, w, cost_bps=0.0, signal_lag=1)["net_return"].reindex(idx).sum()
    if gross <= 0:
        return 0.0  # ni a coste cero gana: edge inexistente
    lo, hi = 0.0, 200.0
    for _ in range(40):  # bisección sobre el coste que anula el net acumulado
        mid = (lo + hi) / 2
        net = run_backtest(oos_ret, w, cost_bps=mid, signal_lag=1)["net_return"].reindex(idx).sum()
        if net > 0:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def _metrics(pos: np.ndarray, truth: np.ndarray, oos_ret: pd.Series, idx, cost: float = 1.0) -> dict:
    acc = float((pos == truth).mean())
    w = pd.Series(0.0, index=oos_ret.index); w.loc[idx] = pos
    nr = run_backtest(oos_ret, w, cost_bps=cost, signal_lag=1)["net_return"].reindex(idx)
    turn_ann = float(pd.Series(pos, index=idx).diff().abs().mean()) * 252  # rotación anualizada
    return {"acc": round(acc, 4), "sharpe": round(float(sharpe(nr)), 4),
            "equity": round(float((1 + nr.fillna(0)).prod()), 4),
            "maxDD": round(float(max_drawdown(equity_curve(nr))), 4),
            "turnover_ann": round(turn_ann, 3)}


def run_ticker(ticker: str) -> dict:
    data.load_market_data(ticker, CALIBRATION_START, datetime.date.today().isoformat())
    sc = screen_ticker(ticker)  # calibración congelada: media_regimen (prior de signo) + leverage_corr
    gamma, sigma, oos_ret = build_states(ticker)
    r_next = oos_ret.shift(-1)
    idx = r_next.dropna().index
    idx = idx[np.sign(r_next.loc[idx]) != 0]
    truth = np.sign(r_next.loc[idx].to_numpy())
    dom = gamma.reindex(idx).to_numpy().argmax(axis=1)
    sign_prior = {k: float(np.sign(sc["media_regimen"][nm])) for k, nm in enumerate(REGNAMES)}
    pos_reg = np.array([sign_prior[d] for d in dom])
    frac_up = float((truth > 0).mean())
    maj = 1.0 if frac_up >= 0.5 else -1.0
    reg = _metrics(pos_reg, truth, oos_ret, idx)
    return {
        "n": int(len(idx)), "frac_up": round(frac_up, 4),
        "leverage_corr": sc["leverage_corr"], "crisis_mean": sc["crisis_mean"], "clase": sc["clase"],
        "calib_corta": ticker in CALIB_CORTA,
        "regimen": reg, "bh": _metrics(np.ones_like(truth), truth, oos_ret, idx),
        "zeror": _metrics(np.full_like(truth, maj), truth, oos_ret, idx),
        "breakeven_bps": _breakeven_bps(pos_reg, oos_ret, idx),
        "net_acum_by_cost": {f"{c:g}bp": round(float(
            run_backtest(oos_ret, pd.Series(pos_reg, index=idx).reindex(oos_ret.index).fillna(0.0),
                         cost_bps=c, signal_lag=1)["net_return"].reindex(idx).sum()), 4) for c in COST_GRID},
    }


def _corr(xs, ys) -> dict:
    if len(xs) < 4:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None, "n": len(xs)}
    pr, pp = stats.pearsonr(xs, ys)
    sr, sp = stats.spearmanr(xs, ys)
    return {"pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 4), "n": len(xs)}


def main() -> None:
    res, ok = {}, []
    for tk in HELDOUT:
        try:
            r = run_ticker(tk); res[tk] = r; ok.append((tk, r))
            rg, zr = r["regimen"], r["zeror"]
            print(f"{tk:5s} lev={r['leverage_corr']:+.3f} up={r['frac_up']:.2f} n={r['n']:3d} "
                  f"{'[calib corta]' if r['calib_corta'] else '':13s}| RÉGIMEN acc={rg['acc']:.3f} "
                  f"Sh={rg['sharpe']:+.2f} DD={rg['maxDD']:+.2f} turn={rg['turnover_ann']:.1f} "
                  f"BE={r['breakeven_bps']:.1f}bp | ZeroR acc={zr['acc']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            res[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)

    # Ventaja del régimen vs leverage, en el held-out.
    lev = [r["leverage_corr"] for _, r in ok]
    edge_acc = [r["regimen"]["acc"] - r["zeror"]["acc"] for _, r in ok]
    sh_reg = [r["regimen"]["sharpe"] for _, r in ok]
    fuerte = [(t, r) for t, r in ok if r["leverage_corr"] < -0.05]
    debil = [(t, r) for t, r in ok if r["leverage_corr"] >= -0.05]

    # Pooled con el panel-15 (regime_baselines_priceonly), si existe.
    pooled = None
    p15 = Path("outputs/experiments/regime_baselines_priceonly.json")
    if p15.exists():
        d15 = json.load(open(p15))["por_activo"]
        lev_all = lev + [r["leverage_corr"] for r in d15.values() if "regimen" in r]
        sh_all = sh_reg + [r["regimen"]["sharpe"] for r in d15.values() if "regimen" in r]
        ea_all = edge_acc + [r["regimen"]["acc"] - r["zeror"]["acc"] for r in d15.values() if "regimen" in r]
        pooled = {"n": len(lev_all), "corr_lev_sharpe": _corr(lev_all, sh_all),
                  "corr_lev_edge_acc": _corr(lev_all, ea_all)}

    def gmean(grp, sel):
        vals = [sel(r) for _, r in grp]
        return round(float(np.mean(vals)), 4) if vals else None

    resumen = {
        "held_out_n": len(ok),
        "corr_lev_sharpe_heldout": _corr(lev, sh_reg),
        "corr_lev_edge_acc_heldout": _corr(lev, edge_acc),
        "grupo_fuerte": {"n": len(fuerte), "sharpe_medio": gmean(fuerte, lambda r: r["regimen"]["sharpe"]),
                         "acc_medio": gmean(fuerte, lambda r: r["regimen"]["acc"]),
                         "turn_medio": gmean(fuerte, lambda r: r["regimen"]["turnover_ann"]),
                         "BE_medio": gmean(fuerte, lambda r: r["breakeven_bps"])},
        "grupo_debil": {"n": len(debil), "sharpe_medio": gmean(debil, lambda r: r["regimen"]["sharpe"]),
                        "acc_medio": gmean(debil, lambda r: r["regimen"]["acc"]),
                        "turn_medio": gmean(debil, lambda r: r["regimen"]["turnover_ann"]),
                        "BE_medio": gmean(debil, lambda r: r["breakeven_bps"])},
        "pooled_con_15": pooled,
    }
    out = {"meta": {"universo": HELDOUT, "oos_start": config.STRATA_OOS_START,
                    "nota": "canal régimen price-only en universo HELD-OUT (sin agente, sin uso previo); "
                            "prior de signo congelado en calibración; signal_lag=1; coste 1bp salvo barrido",
                    "pre_registro": "BITACORA 2026-06-23"},
           "por_activo": res, "resumen": resumen}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== resumen held-out ({len(ok)} activos) ===")
    print(f"  corr leverage↔Sharpe régimen: {resumen['corr_lev_sharpe_heldout']}")
    print(f"  corr leverage↔ventaja-acc:    {resumen['corr_lev_edge_acc_heldout']}")
    print(f"  FUERTE (n={len(fuerte)}) Sharpe {resumen['grupo_fuerte']['sharpe_medio']} "
          f"turn {resumen['grupo_fuerte']['turn_medio']} BE {resumen['grupo_fuerte']['BE_medio']}bp | "
          f"DÉBIL (n={len(debil)}) Sharpe {resumen['grupo_debil']['sharpe_medio']} "
          f"turn {resumen['grupo_debil']['turn_medio']} BE {resumen['grupo_debil']['BE_medio']}bp")
    if pooled:
        print(f"  POOLED con 15 (n={pooled['n']}): lev↔Sharpe {pooled['corr_lev_sharpe']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
