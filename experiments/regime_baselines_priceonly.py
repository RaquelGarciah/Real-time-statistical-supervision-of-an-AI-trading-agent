"""Estrategia de régimen (solo precio) vs B&H y ZeroR, para los 15 activos — SIN agente.

El régimen (HMM) y la vol (GARCH) solo necesitan la serie de precios, así que esto corre
para TODOS los activos, incluidos los que aún no tienen decisiones del agente (IWM/XLF/XLK).
Las estrategias dependientes del agente (M5/M8/M10) NO se computan aquí.

Estrategia 'régimen crudo': posición = signo de la media de calibración del régimen dominante
del día (prior data-driven congelado ≤2024-09; causal, signal_lag=1). Baselines price-only:
B&H (siempre largo) y ZeroR/NIR (signo de la clase mayoritaria). Evaluación sobre el OOS
COMPLETO (el régimen no necesita burn-in; se indica n por activo).

Uso: python experiments/regime_baselines_priceonly.py
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
from core.metrics import equity_curve, max_drawdown, sharpe
from experiments.quant_validation_panel import build_states

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA",
         "QQQ", "DIA", "IWM", "XLF", "XLK"]
REGNAMES = ["Calma", "Estrés", "Crisis"]
OUT = Path("outputs/experiments/regime_baselines_priceonly.json")


def _metrics(pos: np.ndarray, truth: np.ndarray, oos_ret: pd.Series, idx) -> dict:
    acc = float((pos == truth).mean())
    w = pd.Series(0.0, index=oos_ret.index); w.loc[idx] = pos
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx)
    return {"acc": round(acc, 4), "sharpe": round(float(sharpe(nr)), 4),
            "equity": round(float((1 + nr.fillna(0)).prod()), 4),
            "maxDD": round(float(max_drawdown(equity_curve(nr))), 4)}


def run_ticker(ticker: str, lev: dict) -> dict:
    data.load_market_data(ticker, CALIBRATION_START, datetime.date.today().isoformat())  # asegura OOS
    gamma, sigma, oos_ret = build_states(ticker)
    r_next = oos_ret.shift(-1)
    idx = r_next.dropna().index
    idx = idx[np.sign(r_next.loc[idx]) != 0]
    truth = np.sign(r_next.loc[idx].to_numpy())
    dom = gamma.reindex(idx).to_numpy().argmax(axis=1)  # 0=Calma,1=Estrés,2=Crisis (orden por vol)
    sign_prior = {k: float(np.sign(lev["media_regimen"][nm])) for k, nm in enumerate(REGNAMES)}
    frac_up = float((truth > 0).mean())
    maj = 1.0 if frac_up >= 0.5 else -1.0
    return {
        "n": int(len(idx)), "frac_up": round(frac_up, 4), "leverage_corr": lev["leverage_corr"],
        "regimen": _metrics(np.array([sign_prior[d] for d in dom]), truth, oos_ret, idx),
        "bh": _metrics(np.ones_like(truth), truth, oos_ret, idx),
        "zeror": _metrics(np.full_like(truth, maj), truth, oos_ret, idx),
    }


def main() -> None:
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    res, rows = {}, []
    for tk in PANEL:
        try:
            r = run_ticker(tk, lev[tk]); res[tk] = r
            rows.append((tk, r))
            rg, bh, zr = r["regimen"], r["bh"], r["zeror"]
            print(f"{tk:5s} lev={r['leverage_corr']:+.3f} up={r['frac_up']:.2f} n={r['n']:3d} | "
                  f"RÉGIMEN acc={rg['acc']:.3f} Sh={rg['sharpe']:+.2f} | B&H acc={bh['acc']:.3f} "
                  f"Sh={bh['sharpe']:+.2f} | ZeroR acc={zr['acc']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            res[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)

    ok = [(t, r) for t, r in rows]
    rb = sum(r["regimen"]["acc"] > r["bh"]["acc"] for _, r in ok)
    rz = sum(r["regimen"]["acc"] > r["zeror"]["acc"] for _, r in ok)
    rb_sh = sum(r["regimen"]["sharpe"] > r["bh"]["sharpe"] for _, r in ok)
    fuerte = [(t, r) for t, r in ok if r["leverage_corr"] < -0.05]
    debil = [(t, r) for t, r in ok if r["leverage_corr"] >= -0.05]
    out = {"meta": {"panel": PANEL, "oos_start": config.STRATA_OOS_START,
                    "nota": "régimen y baselines price-only (sin agente); OOS completo, signal_lag=1; "
                            "prior de signo del régimen congelado en calibración (sin look-ahead)",
                    "evaluacion": "OOS completo por activo (el régimen no necesita burn-in)"},
           "por_activo": res,
           "resumen": {"regimen_bate_bh_acc": f"{rb}/{len(ok)}", "regimen_bate_zeror_acc": f"{rz}/{len(ok)}",
                       "regimen_bate_bh_sharpe": f"{rb_sh}/{len(ok)}",
                       "regimen_acc_medio_fuerte": round(float(np.mean([r['regimen']['acc'] for _, r in fuerte])), 4),
                       "regimen_acc_medio_debil": round(float(np.mean([r['regimen']['acc'] for _, r in debil])), 4),
                       "regimen_sharpe_medio_fuerte": round(float(np.mean([r['regimen']['sharpe'] for _, r in fuerte])), 4),
                       "regimen_sharpe_medio_debil": round(float(np.mean([r['regimen']['sharpe'] for _, r in debil])), 4)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== resumen (price-only, {len(ok)} activos) ===")
    print(f"  régimen bate B&H en accuracy: {rb}/{len(ok)} · en Sharpe: {rb_sh}/{len(ok)} · "
          f"bate ZeroR: {rz}/{len(ok)}")
    print(f"  régimen acc medio — FUERTE: {out['resumen']['regimen_acc_medio_fuerte']} "
          f"(Sharpe {out['resumen']['regimen_sharpe_medio_fuerte']}) | "
          f"DÉBIL: {out['resumen']['regimen_acc_medio_debil']} (Sharpe {out['resumen']['regimen_sharpe_medio_debil']})")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
