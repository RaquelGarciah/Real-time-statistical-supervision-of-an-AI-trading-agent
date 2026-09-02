"""TS-momentum à la Moskowitz, Ooi & Pedersen (2012): lookback LARGO, rebalanceo MENSUAL, rodando.

Corrección del test anterior (que usaba lookback=holding=1 bloque, señal débil). La especificación canónica
de MOP 2012: en cada rebalanceo mensual, señal = signo del retorno de los ÚLTIMOS K meses (K=12 cabecera),
se mantiene la posición el mes siguiente. Se evalúa RODANDO mes a mes. Holding de 21 días NO solapado
(truths independientes). Causal: la señal usa solo el pasado. Se reporta acierto direccional Y la métrica
real de MOP: retorno medio y Sharpe de la estrategia (que va corto en tendencias bajistas), vs "siempre largo".

PRE-REGISTRO (antes de ver el OOS):
  Primario: K=12 meses (252d), la cabecera de MOP 2012. Secundarios K=3,6 (63,126d) por robustez.
  H1: en OOS, la estrategia TS-momentum mensual tiene acierto >0.5 con sign test p<0.10, O Sharpe causal
      superior a "siempre largo". (MOP encuentran momentum mensual significativo.)
  H0: acierto ≤0.5/no sig Y Sharpe ≤ siempre-largo.
  Rebalanceo cada 21 días no solapado. Sin look-ahead.
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
LOOKBACKS = {"K3": 63, "K6": 126, "K12": 252}      # 3/6/12 meses
HOLD = 21
ANN = np.sqrt(12)                                   # rebalanceo mensual
OUT = Path("outputs/experiments/momentum_tsmom_monthly.json")


def sign_p(corr: np.ndarray) -> float:
    k, n = int(np.sum(corr)), len(corr)
    if n == 0:
        return 1.0
    z = (k - n / 2) / np.sqrt(n / 4)
    return float(2 * min(norm.cdf(z), 1 - norm.cdf(z)))


def sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / s * ANN) if s > 0 else 0.0


def tsmom_rows(tk: str, lb: int) -> pd.DataFrame:
    """Rebalanceo cada HOLD días: señal=signo(retorno trailing lb), retorno fwd no solapado del mes siguiente."""
    _, ret = wf.load_features(tk)
    vals, idx = ret.to_numpy(), ret.index
    rows = []
    t = lb
    while t + HOLD <= len(vals):
        sig = np.sign(vals[t - lb:t].sum())              # trailing lb (causal)
        fwd = float(vals[t:t + HOLD].sum())              # mes siguiente (no solapado)
        date = idx[t]                                    # fecha de decisión
        rows.append((date, sig, fwd))
        t += HOLD
    b = pd.DataFrame(rows, columns=["date", "sig", "fwd"]).set_index("date")
    b = b[(b["sig"] != 0) & (b["fwd"] != 0)]
    b["truth"] = np.sign(b["fwd"])
    b["hit"] = (b["sig"] == b["truth"]).astype(int)
    b["strat_ret"] = b["sig"] * b["fwd"]                 # TS-momentum (largo/corto)
    b["long_ret"] = b["fwd"]                             # siempre largo
    b["ticker"] = tk
    return b


def main() -> None:
    res = {}
    for name, lb in LOOKBACKS.items():
        allb = pd.concat([tsmom_rows(tk, lb) for tk in PANEL])
        calib = allb[allb.index <= pd.Timestamp(CALIBRATION_END)]
        oos = allb[allb.index >= pd.Timestamp(STRATA_OOS_START)]
        hit = oos["hit"].to_numpy()
        res[name] = {
            "lookback_d": lb, "n_oos_meses": int(len(oos)),
            "calib_acc": round(float(calib["hit"].mean()), 4),
            "oos_acc": round(float(hit.mean()), 4), "oos_sign_p": round(sign_p(hit), 4),
            "oos_siempre_largo_acc": round(float(oos["long_ret"].gt(0).mean()), 4),
            "oos_sharpe_tsmom": round(sharpe(oos["strat_ret"].to_numpy()), 3),
            "oos_sharpe_siempre_largo": round(sharpe(oos["long_ret"].to_numpy()), 3),
            "calib_sharpe_tsmom": round(sharpe(calib["strat_ret"].to_numpy()), 3),
        }

    exito = any(((r["oos_acc"] > 0.5 and r["oos_sign_p"] < 0.10)
                 or r["oos_sharpe_tsmom"] > r["oos_sharpe_siempre_largo"]) for r in res.values())
    OUT.write_text(json.dumps({"meta": {"lookbacks": LOOKBACKS, "hold": HOLD, "rebalanceo": "mensual no solapado",
                                        "calib_end": CALIBRATION_END, "oos_start": STRATA_OOS_START},
                               "por_lookback": res, "EXITO_PREREGISTRADO": exito}, indent=2, ensure_ascii=False))

    print(f"{'lookback':<10}{'n_oos':>7}{'calib_acc':>11}{'OOS_acc':>9}{'sign_p':>9}{'siempreL_acc':>14}"
          f"{'SR_tsmom':>10}{'SR_largo':>10}{'SR_calib':>10}")
    for name, lb in LOOKBACKS.items():
        r = res[name]
        print(f"{name}({lb}d){'':<2}{r['n_oos_meses']:>7}{r['calib_acc']:>11.4f}{r['oos_acc']:>9.4f}"
              f"{r['oos_sign_p']:>9.4f}{r['oos_siempre_largo_acc']:>14.4f}{r['oos_sharpe_tsmom']:>10.3f}"
              f"{r['oos_sharpe_siempre_largo']:>10.3f}{r['calib_sharpe_tsmom']:>10.3f}")
    print(f"\n>>> EXITO PRE-REGISTRADO (acierto>0.5 sig O Sharpe>siempre-largo en algún K): {exito}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
