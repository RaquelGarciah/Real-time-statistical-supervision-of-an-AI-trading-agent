"""Screen ex-ante del leverage effect por activo (criterio de alcance de STRATA).

La hipótesis de alcance del TFG: RAM (el régimen del HMM) es un proxy DIRECCIONAL solo
cuando el activo tiene leverage effect fuerte (Black 1976; Christie 1982): la alta
volatilidad coincide con caídas, de modo que el estado Crisis del HMM tiene media de
retorno negativa. En activos con leverage débil o inverso (acciones individuales,
commodities, cripto-proxies) el régimen no separa por dirección y la estrategia no debería
funcionar.

Este screen mide ese criterio SOLO sobre la ventana de calibración (≤2024-09-30), sin tocar
el OOS: es pre-registrable y no incurre en data-snooping. Dos métricas complementarias:

  1. media de retorno del régimen Crisis (estado de mayor vol del HMM) en calibración.
     Negativa ⇒ leverage fuerte (régimen direccional, STRATA debería funcionar).
  2. correlación leverage = corr(r_t, Δ rv_{t+1}) en calibración (Black): negativa ⇒
     las caídas anticipan más volatilidad (leverage effect).

No usa el agente LLM ni el OOS. Uso: python experiments/leverage_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, CALIBRATION_START
from core import data, features
from core.hmm import RegimeHMM

# Panel actual (10) + candidatos de leverage fuerte (índices/ETFs amplios).
ACTUAL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
CANDIDATOS = ["QQQ", "DIA", "IWM", "XLF", "XLK"]
PANEL = ACTUAL + CANDIDATOS
OUT = Path("outputs/experiments/leverage_screen.json")


def calib_features(ticker: str) -> pd.DataFrame:
    """Retorno + RV21 en la calibración (descarga de yfinance si falta el parquet)."""
    prices = data.load_market_data(ticker, CALIBRATION_START, CALIBRATION_END)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    return pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()


def screen_ticker(ticker: str) -> dict:
    feat = calib_features(ticker)
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(feat.to_numpy())
    states = hmm.predict_states(feat.to_numpy())  # ordenados por vol ascendente: 0=Calma..2=Crisis
    r = feat["r"].to_numpy()
    medias = {nm: float(r[states == k].mean()) if (states == k).any() else float("nan")
              for k, nm in enumerate(["Calma", "Estrés", "Crisis"])}
    fracs = {nm: float((states == k).mean()) for k, nm in enumerate(["Calma", "Estrés", "Crisis"])}
    # Correlación leverage de Black: r_t vs cambio de vol del día siguiente.
    drv = feat["rv"].shift(-1) - feat["rv"]
    lev_corr = float(pd.Series(r, index=feat.index).corr(drv))
    crisis_mean = medias["Crisis"]
    # Clasificación ex-ante por la media del régimen Crisis (criterio principal).
    if crisis_mean < -0.0003:
        clase = "leverage fuerte (régimen direccional)"
    elif crisis_mean > 0.0003:
        clase = "leverage inverso/ausente (régimen NO direccional)"
    else:
        clase = "leverage débil/neutro"
    return {"n_calib": int(len(feat)), "media_regimen": {k: round(v, 6) for k, v in medias.items()},
            "frac_regimen": {k: round(v, 4) for k, v in fracs.items()},
            "crisis_mean": round(crisis_mean, 6), "leverage_corr": round(lev_corr, 4),
            "clase": clase, "calib_ini": str(feat.index[0].date()), "calib_fin": str(feat.index[-1].date())}


def main() -> None:
    res = {}
    for tk in PANEL:
        try:
            res[tk] = screen_ticker(tk)
            r = res[tk]
            print(f"{tk:5s} Crisis_mean={r['crisis_mean']:+.5f} lev_corr={r['leverage_corr']:+.3f} "
                  f"n={r['n_calib']:5d}  {r['clase']}")
        except Exception as e:  # noqa: BLE001
            res[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}")

    fuerte = [t for t in PANEL if "error" not in res[t] and res[t]["crisis_mean"] < -0.0003]
    inverso = [t for t in PANEL if "error" not in res[t] and res[t]["crisis_mean"] > 0.0003]
    out = {
        "meta": {"panel_actual": ACTUAL, "candidatos": CANDIDATOS,
                 "calibration_window": [CALIBRATION_START, CALIBRATION_END], "seed": config.SEED,
                 "criterio": "media de retorno del régimen Crisis (HMM K=3) en calibración; <0 = leverage fuerte",
                 "nota": "pre-registrable: medido SOLO en calibración, sin tocar el OOS (sin data-snooping)",
                 "hipotesis_alcance": "STRATA/RAM funciona donde el leverage es fuerte (Crisis_mean<0); "
                                      "no en leverage débil/inverso"},
        "por_activo": res,
        "grupos": {"leverage_fuerte": fuerte, "leverage_inverso_o_ausente": inverso},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nLeverage FUERTE (Crisis_mean<0): {fuerte}")
    print(f"Leverage inverso/ausente (Crisis_mean>0): {inverso}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
