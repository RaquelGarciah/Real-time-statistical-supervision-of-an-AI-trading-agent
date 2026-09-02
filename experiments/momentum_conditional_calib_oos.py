"""Regla CONDICIONAL del momentum, descubierta SOLO en calibración y testeada en OOS (sin look-ahead).

Idea: el momentum no es bueno/malo por activo, sino por RÉGIMEN DE TENDENCIA, y eso se mide causalmente.
Para no caer en look-ahead: el umbral θ y la propia relación se aprenden en CALIBRACIÓN (2000→2024-09,
miles de días, sin agente), y se aplican INTACTOS al OOS. La variable de estado (efficiency ratio de
Kaufman a 63 días) es causal (solo pasado). Momentum puro = regla pos=signo(suma 21d) → sin ML, sin
sobreajuste. Se mide accuracy direccional contra signo(r_{t+1}).

PRE-REGISTRO (criterio fijado ANTES de ver el OOS):
  H1: la accuracy del momentum crece con la fuerza de tendencia. El umbral θ = corte del tercil SUPERIOR
      del efficiency ratio EN CALIBRACIÓN identifica, EN OOS, días donde el momentum acierta >0.5
      (sign test p<0.10) Y con accuracy mayor que en los días no-tendencia.
  H0: en calibración no hay relación monótona accuracy↔tendencia, o θ no separa el OOS
      (trending ≤ no-trending, o accuracy trending no >0.5).
  θ se fija SOLO con calibración. Ningún dato del OOS interviene en θ ni en la regla.
  Ventanas fijadas a priori: efficiency ratio 63d, momentum lookback 21d.
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
EFF_W, L_MOM = 63, 21
OUT = Path("outputs/experiments/momentum_conditional_calib_oos.json")


def eff_ratio_series(r: pd.Series, w: int) -> pd.Series:
    """Kaufman efficiency ratio causal: |Σr|/Σ|r| en ventana de w días terminando en t."""
    num = r.rolling(w).apply(lambda x: abs(x.sum()), raw=True)
    den = r.rolling(w).apply(lambda x: np.abs(x).sum(), raw=True)
    return num / den


def momentum_hits(r: pd.Series):
    """pos=signo(suma L_MOM) prediciendo signo(r_{t+1}); devuelve Series de aciertos (1/0) y eff ratio."""
    pos = np.sign(r.rolling(L_MOM).sum())
    truth = np.sign(r.shift(-1))
    eff = eff_ratio_series(r, EFF_W)
    df = pd.DataFrame({"pos": pos, "truth": truth, "eff": eff}).dropna()
    df = df[(df["pos"] != 0) & (df["truth"] != 0)]
    df["hit"] = (df["pos"] == df["truth"]).astype(int)
    return df


def sign_p(corr: np.ndarray) -> float:
    k, n = int(corr.sum()), len(corr)
    if n == 0:
        return 1.0
    z = (k - n / 2) / np.sqrt(n / 4)
    return float(2 * min(norm.cdf(z), 1 - norm.cdf(z)))


def main() -> None:
    calib_rows, oos_rows = [], []
    for tk in PANEL:
        _, ret = wf.load_features(tk)
        df = momentum_hits(ret)
        df["ticker"] = tk
        calib_rows.append(df[df.index <= pd.Timestamp(CALIBRATION_END)])
        oos_rows.append(df[df.index >= pd.Timestamp(STRATA_OOS_START)])
    calib = pd.concat(calib_rows)
    oos = pd.concat(oos_rows)

    # --- DESCUBRIMIENTO en calibración: terciles de eff ratio y θ = corte del tercil superior ---
    q1, q2 = calib["eff"].quantile([1 / 3, 2 / 3])
    theta = float(q2)                      # umbral de despliegue, fijado SOLO con calibración
    def acc_by_tercile(d):
        lo = d[d["eff"] <= q1]["hit"]; mid = d[(d["eff"] > q1) & (d["eff"] <= q2)]["hit"]; hi = d[d["eff"] > q2]["hit"]
        return {"bajo": round(float(lo.mean()), 4), "medio": round(float(mid.mean()), 4),
                "alto": round(float(hi.mean()), 4), "n_alto": int(len(hi))}
    calib_terciles = acc_by_tercile(calib)
    monotona_calib = bool(calib_terciles["alto"] > calib_terciles["medio"] > calib_terciles["bajo"])

    # --- TEST en OOS aplicando θ de calibración (intacto) ---
    trend = oos[oos["eff"] >= theta]
    flat = oos[oos["eff"] < theta]
    acc_trend = float(trend["hit"].mean()); acc_flat = float(flat["hit"].mean())
    p_trend = sign_p(trend["hit"].to_numpy()); p_flat = sign_p(flat["hit"].to_numpy())

    # por activo en OOS (cuántos confirman el patrón trend>flat)
    por_activo = {}
    for tk in PANEL:
        o = oos[oos["ticker"] == tk]
        t, f = o[o["eff"] >= theta]["hit"], o[o["eff"] < theta]["hit"]
        por_activo[tk] = {"acc_trend": round(float(t.mean()), 4) if len(t) else None, "n_trend": int(len(t)),
                          "acc_flat": round(float(f.mean()), 4) if len(f) else None, "n_flat": int(len(f)),
                          "confirma": bool(len(t) and len(f) and t.mean() > f.mean())}
    n_confirma = sum(v["confirma"] for v in por_activo.values())

    exito = bool(monotona_calib and acc_trend > 0.5 and p_trend < 0.10 and acc_trend > acc_flat)
    resumen = {
        "theta_eff_ratio": round(theta, 4), "q1_calib": round(float(q1), 4),
        "calib_acc_por_tercil": calib_terciles, "monotona_en_calib": monotona_calib,
        "oos_acc_trending": round(acc_trend, 4), "oos_n_trending": int(len(trend)), "oos_sign_p_trending": round(p_trend, 4),
        "oos_acc_flat": round(acc_flat, 4), "oos_n_flat": int(len(flat)), "oos_sign_p_flat": round(p_flat, 4),
        "activos_que_confirman": f"{n_confirma}/{len(PANEL)}",
        "EXITO_PREREGISTRADO": exito,
    }
    OUT.write_text(json.dumps({"meta": {"eff_w": EFF_W, "l_mom": L_MOM, "calib_end": CALIBRATION_END,
                                        "oos_start": STRATA_OOS_START, "panel": PANEL},
                               "resumen": resumen, "por_activo_oos": por_activo}, indent=2, ensure_ascii=False))

    print("DESCUBRIMIENTO (solo calibración):")
    print(f"  accuracy momentum por tercil de tendencia: bajo={calib_terciles['bajo']} "
          f"medio={calib_terciles['medio']} alto={calib_terciles['alto']}  (monótona={monotona_calib})")
    print(f"  θ (corte tercil superior eff ratio) = {theta:.4f}  [fijado SOLO con calibración]\n")
    print("TEST (OOS, aplicando θ intacto):")
    print(f"  días TENDENCIA (eff≥θ): acc={acc_trend:.4f} (n={len(trend)}, sign p={p_trend:.4f})")
    print(f"  días LATERAL  (eff<θ):  acc={acc_flat:.4f} (n={len(flat)}, sign p={p_flat:.4f})")
    print(f"  activos que confirman trend>flat: {n_confirma}/{len(PANEL)}")
    print(f"\n>>> EXITO PRE-REGISTRADO: {exito}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
