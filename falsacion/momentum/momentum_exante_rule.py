"""Regla EX-ANTE (sin look-ahead) para decidir si añadir momentum a M10, validada en el panel.

Pregunta: ¿se puede justificar, mirando SOLO la calibración (2000→2024-09), si meterle momentum a M10
va a ayudar en el OOS? No se puede garantizar al 100% (sería look-ahead / violaría eficiencia de mercado),
pero sí se puede PRE-REGISTRAR una regla basada en estructura de tendencia medida en el pasado y comprobar
que clasifica bien en el panel.

Diagnóstico de tendencia por activo, calculado SOLO en calibración:
  1. Variance Ratio test (Lo & MacKinlay 1988), z heterocedástico-robusto, horizontes q=5,10,21.
     VR>1 ⇒ tendencia/momentum; VR≈1 ⇒ paseo aleatorio; VR<1 ⇒ reversión. Sin look-ahead (ret≤CALIB_END).
  2. Momentum interno: regla pos_t=signo(suma retornos t-L+1..t) prediciendo signo(r_{t+1}), accuracy
     dentro de calibración (causal, sin modelo → sin sobreajuste), sign test vs 0.5. L=21.

Objetivo (ya calculado, m10_pivot_scan): Δacc_OOS = acc(aug) − acc(base) por activo (momentum ayuda si >0).

PRE-REGISTRO (criterio fijado ANTES de ver el cruce):
  H1: el diagnóstico de calibración predice el beneficio OOS del momentum.
  H0: no hay relación (Spearman ρ=0) entre diagnóstico de calibración y Δacc_OOS.
  Estadístico: Spearman ρ (VR_z@21 vs Δacc_OOS) y (mom_acc_calib vs Δacc_OOS), n=10 activos.
  Éxito: ρ>0 con p<0.10 EN AL MENOS UNO de los dos diagnósticos, Y la regla binaria
         (VR@21>1 con z>1.28  ⇒ "añadir momentum") clasifica correctamente ≥7/10 activos,
         con SPY (clase: ayuda) y SMCI (clase: no ayuda) ambos bien clasificados.
  Fracaso: ρ≤0 o p≥0.10 en ambos, o SPY/SMCI mal clasificados. ⇒ no hay regla ex-ante defendible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import experiments.walkforward_robustez as wf
from config import CALIBRATION_END

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
QS = [5, 10, 21]
L_MOM = 21
SCAN = Path("outputs/experiments/m10_pivot_scan.json")
OUT = Path("outputs/experiments/momentum_exante_rule.json")


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """VR(q) y z heterocedástico-robusto (Lo & MacKinlay 1988, ec. M2). VR>1 ⇒ tendencia."""
    r = r[~np.isnan(r)]
    n = len(r)
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    # varianza del retorno agregado q-periodos (solapado, insesgado)
    rq = np.convolve(r, np.ones(q), mode="valid")            # sumas de q retornos consecutivos
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    # varianza heterocedástico-robusta de VR(q)
    dev2 = (r - mu) ** 2
    denom = np.sum(dev2) ** 2
    theta = 0.0
    for k in range(1, q):
        delta_k = np.sum(dev2[k:] * dev2[:-k]) / denom        # Lo-MacKinlay 1988 ec. M2 (sin factor n)
        theta += ((2 * (q - k) / q) ** 2) * delta_k
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else 0.0
    return float(vr), float(z)


def momentum_internal_acc(r: pd.Series, L: int) -> tuple[float, float, int]:
    """Accuracy de la regla momentum pos=signo(suma L) prediciendo signo(r_{t+1}), DENTRO de calibración."""
    mom = r.rolling(L).sum()
    pos = np.sign(mom)
    truth = np.sign(r.shift(-1))
    valid = pos.notna() & truth.notna() & (pos != 0) & (truth != 0)
    corr = (pos[valid] == truth[valid]).astype(int).to_numpy()
    acc = float(corr.mean())
    # sign test vs 0.5 (binomial de dos colas)
    k, n = int(corr.sum()), len(corr)
    p = float(2 * min(norm.cdf((k - n / 2) / np.sqrt(n / 4)), 1 - norm.cdf((k - n / 2) / np.sqrt(n / 4))))
    return acc, p, n


def main() -> None:
    scan = json.load(open(SCAN))["por_activo"]
    rows = []
    for tk in PANEL:
        feat_df, ret = wf.load_features(tk)
        rc = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
        vr = {q: variance_ratio(rc.to_numpy(), q) for q in QS}
        mom_acc, mom_p, mom_n = momentum_internal_acc(rc, L_MOM)
        c = scan[tk]["configs"]
        d_oos = round(c["aug"]["accuracy"] - c["base"]["accuracy"], 4)
        rows.append({
            "ticker": tk,
            "vr5": round(vr[5][0], 3), "z5": round(vr[5][1], 2),
            "vr10": round(vr[10][0], 3), "z10": round(vr[10][1], 2),
            "vr21": round(vr[21][0], 3), "z21": round(vr[21][1], 2),
            "mom_acc_calib": round(mom_acc, 4), "mom_p_calib": round(mom_p, 4), "mom_n": mom_n,
            "delta_acc_oos": d_oos, "momentum_ayuda_oos": bool(d_oos > 0),
            "regla_dice_anadir": bool(vr[21][0] > 1 and vr[21][1] > 1.28),  # VR@21>1 con z>1.28 (p<0.10 una cola)
        })

    df = pd.DataFrame(rows).set_index("ticker")
    rho_vr, p_vr = spearmanr(df["z21"], df["delta_acc_oos"])
    rho_mom, p_mom = spearmanr(df["mom_acc_calib"], df["delta_acc_oos"])
    aciertos = int((df["regla_dice_anadir"] == df["momentum_ayuda_oos"]).sum())
    spy_ok = bool(df.loc["SPY", "regla_dice_anadir"] == df.loc["SPY", "momentum_ayuda_oos"])
    smci_ok = bool(df.loc["SMCI", "regla_dice_anadir"] == df.loc["SMCI", "momentum_ayuda_oos"])
    exito = bool(((p_vr < 0.10 and rho_vr > 0) or (p_mom < 0.10 and rho_mom > 0))
                 and aciertos >= 7 and spy_ok and smci_ok)

    resumen = {
        "spearman_z21_vs_deltaOOS": {"rho": round(float(rho_vr), 3), "p": round(float(p_vr), 4)},
        "spearman_momacc_vs_deltaOOS": {"rho": round(float(rho_mom), 3), "p": round(float(p_mom), 4)},
        "regla_binaria_aciertos": f"{aciertos}/10", "spy_bien_clasificado": spy_ok,
        "smci_bien_clasificado": smci_ok, "EXITO_PREREGISTRADO": exito,
    }
    OUT.write_text(json.dumps({"meta": {"calib_end": CALIBRATION_END, "L_mom": L_MOM, "qs": QS,
                                        "pre_registro": "docstring de este script"},
                               "por_activo": rows, "resumen": resumen}, indent=2, ensure_ascii=False))

    cols = ["vr21", "z21", "mom_acc_calib", "mom_p_calib", "delta_acc_oos", "momentum_ayuda_oos", "regla_dice_anadir"]
    print(df[cols].to_string())
    print(f"\nSpearman z21 vs Δacc_OOS:      ρ={rho_vr:+.3f} (p={p_vr:.4f})")
    print(f"Spearman mom_acc vs Δacc_OOS:  ρ={rho_mom:+.3f} (p={p_mom:.4f})")
    print(f"Regla binaria (VR@21>1, z>1.28): {aciertos}/10 aciertos | SPY ok={spy_ok} SMCI ok={smci_ok}")
    print(f">>> EXITO PRE-REGISTRADO: {exito}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
