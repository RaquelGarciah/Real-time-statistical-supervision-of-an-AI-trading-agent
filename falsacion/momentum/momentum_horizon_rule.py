"""Time-series momentum a horizonte semanal/mensual (Moskowitz, Ooi & Pedersen 2012), sin look-ahead.

El momentum casi no vive en la dirección DIARIA (near-martingala); MOP 2012 lo mide a horizonte MENSUAL
y con retornos NO SOLAPADOS (imprescindible: bloques solapados inflan n y la significancia). Aquí:
  - Bloques NO solapados de h días (h=5 semanal, 10 quincenal, 21 mensual).
  - Señal TS-momentum: signo del retorno del bloque ANTERIOR (continuación). Predice signo del bloque actual.
  - Causal: la señal usa solo el bloque pasado. Baseline "siempre largo" para separar habilidad de deriva.
  - Refinamiento condicional: θ del efficiency ratio (63d, causal) fijado SOLO en calibración; se mira si en
    bloques de tendencia alta el acierto sube. Descubrir en calibración, testear en OOS.

PRE-REGISTRO (antes de ver el OOS):
  H1: a horizonte ≥ semanal, el acierto direccional del momentum en OOS es >0.5 con sign test p<0.10,
      y supera al baseline "siempre largo"; el efecto es mayor que a horizonte diario (0.510, no sig).
  H0: acierto ≤0.5 o no significativo, o no supera a "siempre largo".
  Bloques no solapados (independencia aproximada → sign test válido). θ solo de calibración.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

import experiments.walkforward_robustez as wf
from config import CALIBRATION_END, STRATA_OOS_START

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
HORIZONS = [5, 10, 21]
EFF_W = 63
OUT = Path("outputs/experiments/momentum_horizon_rule.json")


def sign_p(corr: np.ndarray) -> float:
    k, n = int(np.sum(corr)), len(corr)
    if n == 0:
        return 1.0
    z = (k - n / 2) / np.sqrt(n / 4)
    return float(2 * min(norm.cdf(z), 1 - norm.cdf(z)))


def eff_ratio_series(r: pd.Series, w: int) -> pd.Series:
    num = r.rolling(w).apply(lambda x: abs(x.sum()), raw=True)
    den = r.rolling(w).apply(lambda x: np.abs(x).sum(), raw=True)
    return num / den


def blocks(r: pd.Series, h: int, eff: pd.Series) -> pd.DataFrame:
    """Bloques NO solapados de h días: retorno acumulado, fecha fin, y eff ratio al INICIO del bloque (causal)."""
    vals, idx = r.to_numpy(), r.index
    nb = len(vals) // h
    rows = []
    for i in range(nb):
        seg = vals[i * h:(i + 1) * h]
        start_date, end_date = idx[i * h], idx[(i + 1) * h - 1]
        rows.append((end_date, float(seg.sum()), float(eff.reindex([start_date]).iloc[0])))
    return pd.DataFrame(rows, columns=["end", "cum", "eff_ini"]).set_index("end")


def build_signals(tk: str, h: int) -> pd.DataFrame:
    _, ret = wf.load_features(tk)
    eff = eff_ratio_series(ret, EFF_W)
    b = blocks(ret, h, eff)
    b["signal"] = np.sign(b["cum"].shift(1))           # TS-momentum: signo del bloque anterior (causal)
    b["truth"] = np.sign(b["cum"])
    b = b.dropna()
    b = b[(b["signal"] != 0) & (b["truth"] != 0)]
    b["hit_mom"] = (b["signal"] == b["truth"]).astype(int)
    b["hit_long"] = (b["truth"] > 0).astype(int)        # baseline "siempre largo"
    b["ticker"] = tk
    return b


def split(b: pd.DataFrame):
    return (b[b.index <= pd.Timestamp(CALIBRATION_END)], b[b.index >= pd.Timestamp(STRATA_OOS_START)])


def main() -> None:
    res = {}
    for h in HORIZONS:
        allb = pd.concat([build_signals(tk, h) for tk in PANEL])
        calib, oos = split(allb)
        theta = float(calib["eff_ini"].quantile(2 / 3))         # θ tercil superior, SOLO calibración
        mom_oos = oos["hit_mom"].to_numpy()
        long_oos = oos["hit_long"].to_numpy()
        trend = oos[oos["eff_ini"] >= theta]["hit_mom"].to_numpy()
        res[f"h{h}"] = {
            "n_oos_bloques": int(len(oos)),
            "calib_mom_acc": round(float(calib["hit_mom"].mean()), 4),
            "oos_mom_acc": round(float(mom_oos.mean()), 4),
            "oos_mom_sign_p": round(sign_p(mom_oos), 4),
            "oos_siempre_largo_acc": round(float(long_oos.mean()), 4),
            "oos_mom_menos_largo": round(float(mom_oos.mean() - long_oos.mean()), 4),
            "theta_eff": round(theta, 4),
            "oos_mom_acc_tendencia": round(float(trend.mean()), 4) if len(trend) else None,
            "oos_n_tendencia": int(len(trend)),
            "oos_mom_sign_p_tendencia": round(sign_p(trend), 4) if len(trend) else None,
        }

    exito = any(r["oos_mom_acc"] > 0.5 and r["oos_mom_sign_p"] < 0.10
                and r["oos_mom_acc"] > r["oos_siempre_largo_acc"] for r in res.values())
    OUT.write_text(json.dumps({"meta": {"horizons": HORIZONS, "eff_w": EFF_W, "no_solapados": True,
                                        "calib_end": CALIBRATION_END, "oos_start": STRATA_OOS_START},
                               "por_horizonte": res, "EXITO_PREREGISTRADO": exito}, indent=2, ensure_ascii=False))

    print(f"{'horiz':<7}{'n_oos':>7}{'calib':>9}{'OOS_mom':>9}{'sign_p':>9}{'siempre_L':>11}{'mom-L':>9}{'OOS_tend':>10}{'p_tend':>9}")
    for h in HORIZONS:
        r = res[f"h{h}"]
        print(f"{h}d{'':<4}{r['n_oos_bloques']:>7}{r['calib_mom_acc']:>9.4f}{r['oos_mom_acc']:>9.4f}"
              f"{r['oos_mom_sign_p']:>9.4f}{r['oos_siempre_largo_acc']:>11.4f}{r['oos_mom_menos_largo']:>+9.4f}"
              f"{(r['oos_mom_acc_tendencia'] or 0):>10.4f}{(r['oos_mom_sign_p_tendencia'] or 1):>9.4f}")
    print(f"\n>>> EXITO PRE-REGISTRADO (algún horizonte mom>0.5 sig y > siempre-largo): {exito}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
