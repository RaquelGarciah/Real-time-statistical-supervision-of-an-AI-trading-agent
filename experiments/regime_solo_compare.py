"""¿La estrategia 'régimen crudo' bate sola a M8/M10 en los demás activos? (comparación consistente).

Para cada activo computa la accuracy direccional (convención canónica: hold=fallo) y el Sharpe
de la estrategia 'régimen crudo' (seguir el signo de calibración del régimen dominante) sobre el
MISMO tramo de evaluación que el panel (sub = mv.index[150:], post burn-in), y la enfrenta a
M5/M8/M10 (leídos del panel auditado) y a B&H. No re-entrena M10 (rápido).

Uso: python experiments/regime_solo_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core.backtest import run_backtest
from core.metrics import sharpe
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
N0 = 150
REGNAMES = ["Calma", "Estrés", "Crisis"]


def regime_solo(ticker: str, lev: dict) -> dict:
    gamma, sigma, oos_ret = build_states(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    sub = mv.index[N0:]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    sign_prior = {k: float(np.sign(lev["media_regimen"][nm])) for k, nm in enumerate(REGNAMES)}
    dom = mv.loc[sub, "regime_dom"].to_numpy().astype(int)
    pos_reg = np.array([sign_prior[d] for d in dom])
    acc = float((pos_reg == truth).mean())                       # hold (signo 0) = fallo
    w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos_reg
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
    return {"regime_acc": round(acc, 4), "regime_sharpe": round(float(sharpe(nr)), 4),
            "bh_acc": round(float((truth > 0).mean()), 4), "n": int(len(sub))}


def main() -> None:
    wf.reset_thresholds_cache()
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    pan = json.load(open("outputs/experiments/quant_validation_panel.json"))["por_activo"]
    rows = []
    for tk in PANEL:
        rs = regime_solo(tk, lev[tk])
        h = pan[tk]["headline"]
        rows.append({"ticker": tk, "lev": lev[tk]["leverage_corr"],
                     "regime": rs["regime_acc"], "M5": h["accuracy_m5"], "M8": h["accuracy_m8"],
                     "M10": h["accuracy_m10"], "B&H": rs["bh_acc"], "reg_Sharpe": rs["regime_sharpe"]})
        r = rows[-1]
        print(f"{tk:5s} lev={r['lev']:+.3f} | régimen={r['regime']:.3f} M5={r['M5']:.3f} "
              f"M8={r['M8']:.3f} M10={r['M10']:.3f} B&H={r['B&H']:.3f} | regSharpe={r['reg_Sharpe']:+.2f}")
    df = pd.DataFrame(rows).set_index("ticker")
    df["regimen_bate_M8"] = df["regime"] > df["M8"]
    df["regimen_bate_M10"] = df["regime"] > df["M10"]
    df["regimen_bate_BH"] = df["regime"] > df["B&H"]
    out = Path("outputs/experiments/regime_solo_compare.json")
    out.write_text(df.reset_index().to_json(orient="records", indent=2, force_ascii=False))
    print("\n=== ¿el régimen crudo bate a...? (de 10) ===")
    print(f"  > M8:  {int(df['regimen_bate_M8'].sum())}/10")
    print(f"  > M10: {int(df['regimen_bate_M10'].sum())}/10")
    print(f"  > B&H: {int(df['regimen_bate_BH'].sum())}/10")
    fuerte = df[df["lev"] < -0.05]; debil = df[df["lev"] >= -0.05]
    print(f"\n  régimen acc medio — leverage FUERTE {list(fuerte.index)}: {fuerte['regime'].mean():.3f}")
    print(f"  régimen acc medio — leverage DÉBIL {list(debil.index)}: {debil['regime'].mean():.3f}")
    print(f"OK · {out}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
