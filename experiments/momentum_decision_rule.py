"""LA REGLA: ¿meto momentum a M10 para este activo, sí o no? Decidida ex-ante (sin look-ahead).

Señal ex-ante (causal, solo calibración): accuracy de la regla momentum (pos=signo suma 21d → signo r_{t+1})
en el ÚLTIMO AÑO de calibración (252 días previos al OOS). Intuición: si el momentum ha venido funcionando
para ese activo justo antes del OOS, mételo; si no, no. Umbral natural θ=0.50 (sin parámetro libre).

Se contrasta contra la verdad OOS ya calculada (m10_pivot_scan): Δacc = acc(M10 aug) − acc(M10 base) > 0
⇒ el momentum AYUDÓ de verdad. La regla acierta si su decisión (meter/no) coincide con Δacc>0 / ≤0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import experiments.walkforward_robustez as wf
from config import CALIBRATION_END

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
L_MOM, RECENT = 21, 252
THETA = 0.50
SCAN = Path("outputs/experiments/m10_pivot_scan.json")
OUT = Path("outputs/experiments/momentum_decision_rule.json")


def recent_mom_acc(tk: str) -> float:
    """Accuracy del momentum en los últimos RECENT días de calibración (causal, ex-ante)."""
    _, ret = wf.load_features(tk)
    rc = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)].iloc[-(RECENT + L_MOM):]
    pos = np.sign(rc.rolling(L_MOM).sum())
    truth = np.sign(rc.shift(-1))
    m = pos.notna() & truth.notna() & (pos != 0) & (truth != 0)
    return float((pos[m] == truth[m]).mean())


def main() -> None:
    scan = json.load(open(SCAN))["por_activo"]
    rows = []
    for tk in PANEL:
        sig = recent_mom_acc(tk)
        c = scan[tk]["configs"]
        d_oos = round(c["aug"]["accuracy"] - c["base"]["accuracy"], 4)
        regla = sig > THETA                        # decisión ex-ante: meter momentum
        ayuda = d_oos > 0                           # verdad OOS
        rows.append({"ticker": tk, "senal_mom_reciente": round(sig, 4),
                     "regla_dice_meter": bool(regla), "delta_acc_oos": d_oos,
                     "momentum_ayuda_real": bool(ayuda), "acierto": bool(regla == ayuda)})

    aciertos = sum(r["acierto"] for r in rows)
    spy = next(r for r in rows if r["ticker"] == "SPY")
    smci = next(r for r in rows if r["ticker"] == "SMCI")
    resumen = {"theta": THETA, "aciertos": f"{aciertos}/{len(PANEL)}",
               "spy_ok": spy["acierto"], "smci_ok": smci["acierto"],
               "falsos_positivos": [r["ticker"] for r in rows if r["regla_dice_meter"] and not r["momentum_ayuda_real"]],
               "falsos_negativos": [r["ticker"] for r in rows if not r["regla_dice_meter"] and r["momentum_ayuda_real"]]}
    OUT.write_text(json.dumps({"meta": {"l_mom": L_MOM, "recent": RECENT, "theta": THETA},
                               "por_activo": rows, "resumen": resumen}, indent=2, ensure_ascii=False))

    print(f"REGLA: meter momentum si accuracy del momentum en el último año de calibración > {THETA}\n")
    print(f"{'activo':<7}{'señal(últ.año)':>16}{'regla':>9}{'Δacc OOS':>11}{'ayuda real':>13}{'acierto':>10}")
    for r in sorted(rows, key=lambda x: -x["senal_mom_reciente"]):
        print(f"{r['ticker']:<7}{r['senal_mom_reciente']:>16.4f}{('METER' if r['regla_dice_meter'] else 'no'):>9}"
              f"{r['delta_acc_oos']:>+11.4f}{('sí' if r['momentum_ayuda_real'] else 'no'):>13}"
              f"{('✓' if r['acierto'] else '✗'):>10}")
    print(f"\nAciertos: {aciertos}/{len(PANEL)}  |  SPY ok={spy['acierto']}  SMCI ok={smci['acierto']}")
    print(f"Falsos positivos (mete y no ayuda): {resumen['falsos_positivos']}")
    print(f"Falsos negativos (no mete y sí ayuda): {resumen['falsos_negativos']}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
