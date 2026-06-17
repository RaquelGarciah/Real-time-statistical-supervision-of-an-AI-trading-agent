"""Robustez de la regla de inclusión de momentum, sin overfitting a 10 activos.

Parte A (la fuerte) — persistencia temporal DENTRO de calibración: el supuesto de la regla es que el
rendimiento del momentum persiste (si funcionó el último año, funciona el próximo trimestre). Se testea
en toda la historia (2000→2024-09), cada 63 días no solapados, por activo: señal = accuracy del momentum
en el último año (252d, causal) ; resultado = accuracy del momentum en el TRIMESTRE siguiente (63d).
¿La señal predice el resultado? Cientos de puntos → potencia real, sin look-ahead, sin agente.

Parte B — sensibilidad: clasificación de los 10 activos (vs Δacc OOS de m10_pivot_scan) variando ventana
reciente, lookback del momentum y umbral. Si 7/10 se mantiene en una banda, no es un filo de cuchillo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

import experiments.walkforward_robustez as wf
from config import CALIBRATION_END

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
SCAN = Path("outputs/experiments/m10_pivot_scan.json")
OUT = Path("outputs/experiments/momentum_rule_robustness.json")


def mom_hits(ret: pd.Series, lb: int) -> pd.Series:
    pos = np.sign(ret.rolling(lb).sum())
    truth = np.sign(ret.shift(-1))
    m = pos.notna() & truth.notna() & (pos != 0) & (truth != 0)
    return (pos[m] == truth[m]).astype(int)


def sign_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    z = (k - n / 2) / np.sqrt(n / 4)
    return float(2 * min(norm.cdf(z), 1 - norm.cdf(z)))


def parte_A() -> dict:
    """Persistencia: señal (acc momentum últ. 252d) vs resultado (acc momentum prox. 63d), toda la calib."""
    STEP, RECENT, LB = 63, 252, 21
    sig_all, out_all, per_asset_rho = [], [], {}
    for tk in PANEL:
        _, ret = wf.load_features(tk)
        rc = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
        hit = mom_hits(rc, LB)                       # serie de aciertos diarios (causal)
        h = hit.to_numpy()
        sig_tk, out_tk = [], []
        for t in range(RECENT, len(h) - STEP, STEP):
            sig_tk.append(h[t - RECENT:t].mean())    # último año (causal)
            out_tk.append(h[t:t + STEP].mean())      # trimestre siguiente (no solapado)
        if len(sig_tk) >= 5:
            per_asset_rho[tk] = round(float(spearmanr(sig_tk, out_tk)[0]), 3)
        sig_all += sig_tk
        out_all += out_tk
    sig_all, out_all = np.array(sig_all), np.array(out_all)
    rho, p = spearmanr(sig_all, out_all)
    # regla binaria a lo largo de la historia: señal>0.5 ⇒ predice "momentum ayuda prox. trimestre" (out>0.5)
    rule = sig_all > 0.5
    helps = out_all > 0.5
    acc_regla = float((rule == helps).mean())
    out_si = float(out_all[rule].mean()); out_no = float(out_all[~rule].mean())
    n_pos = int((per_asset_rho and sum(v > 0 for v in per_asset_rho.values())) or 0)
    return {"n_puntos": int(len(sig_all)),
            "spearman_senal_vs_resultado": {"rho": round(float(rho), 3), "p": round(float(p), 4)},
            "accuracy_regla_binaria_historica": round(acc_regla, 4),
            "out_medio_si_meter": round(out_si, 4), "out_medio_si_no": round(out_no, 4),
            "rho_por_activo": per_asset_rho, "activos_persistencia_positiva": f"{n_pos}/{len(PANEL)}"}


def recent_acc(ret: pd.Series, recent: int, lb: int) -> float:
    rc = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)].iloc[-(recent + lb):]
    return float(mom_hits(rc, lb).mean())


def parte_B(scan: dict) -> dict:
    """Sensibilidad del 7/10 a ventana, lookback y umbral (se reporta toda la malla, sin elegir la mejor)."""
    target = {tk: (scan[tk]["configs"]["aug"]["accuracy"] - scan[tk]["configs"]["base"]["accuracy"]) > 0
              for tk in PANEL}
    rets = {tk: wf.load_features(tk)[1] for tk in PANEL}
    grid = []
    for recent in (126, 189, 252):
        for lb in (10, 21, 63):
            for theta in (0.49, 0.50, 0.51):
                acc = sum((recent_acc(rets[tk], recent, lb) > theta) == target[tk] for tk in PANEL)
                grid.append({"recent": recent, "lb": lb, "theta": theta, "aciertos": acc})
    accs = [g["aciertos"] for g in grid]
    return {"malla": grid, "aciertos_min": min(accs), "aciertos_max": max(accs),
            "aciertos_medio": round(float(np.mean(accs)), 2), "n_configs": len(grid)}


def main() -> None:
    scan = json.load(open(SCAN))["por_activo"]
    A, B = parte_A(), parte_B(scan)
    OUT.write_text(json.dumps({"parte_A_persistencia": A, "parte_B_sensibilidad": B}, indent=2, ensure_ascii=False))

    print("=== PARTE A: persistencia del momentum en la historia (sin look-ahead, sin agente) ===")
    print(f"  n={A['n_puntos']} puntos (10 activos × trimestres de 24 años)")
    print(f"  Spearman señal(últ.año) vs resultado(prox.trimestre): rho={A['spearman_senal_vs_resultado']['rho']} "
          f"(p={A['spearman_senal_vs_resultado']['p']})")
    print(f"  accuracy de la regla binaria a lo largo de la historia: {A['accuracy_regla_binaria_historica']}")
    print(f"  acc momentum prox. trimestre: si la regla dice METER={A['out_medio_si_meter']} vs NO={A['out_medio_si_no']}")
    print(f"  activos con persistencia positiva: {A['activos_persistencia_positiva']}  (rho/activo: {A['rho_por_activo']})")
    print("\n=== PARTE B: sensibilidad del 7/10 (27 configs) ===")
    print(f"  aciertos sobre 10 — min={B['aciertos_min']}, medio={B['aciertos_medio']}, max={B['aciertos_max']}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
