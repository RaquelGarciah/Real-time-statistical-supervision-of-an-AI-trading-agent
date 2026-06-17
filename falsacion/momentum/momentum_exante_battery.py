"""Batería de técnicas EX-ANTE para predecir si el momentum ayudará a M10, con correctivo de multiplicidad.

Extiende momentum_exante_rule: además del momentum interno full y el VR full (que fallaron, ρ<0), prueba
diagnósticos de tendencia en VENTANAS RECIENTES (la trendiness reciente persiste algo → puede anticipar
la del OOS próximo). Todas se calculan SOLO con datos ≤ CALIBRATION_END. Se reportan TODAS (sin cherry-pick).

Diagnósticos (cada uno: valor por activo → Spearman vs Δacc_OOS del barrido + orden SPY vs SMCI):
  - mom_acc_{full,504,252,126}: accuracy de la regla momentum L=21 dentro de la ventana.
  - vr21_z_{full,252}: z del variance ratio (Lo-MacKinlay), VR>1 ⇒ tendencia.
  - effratio_{252,126}: Kaufman efficiency ratio |Σr|/Σ|r| ∈ [0,1], alto ⇒ trending.
  - signac1_{252}: autocorrelación lag-1 del signo del retorno.

PRE-REGISTRO (criterio antes de ver los cruces):
  H1: existe ≥1 diagnóstico ex-ante con Spearman ρ>0 vs Δacc_OOS, p<0.10, que ordene SPY (ayuda) por
      encima de SMCI (no ayuda).
  H0: ningún diagnóstico cumple (todo ρ≤0 o p≥0.10 o SPY≤SMCI).
  Correctivo de multiplicidad: con K diagnósticos, P(≥1 falso positivo a 0.10) ≈ 1−0.9^K. Un único acierto
      sobre K~9 NO se declara hallazgo: se reporta como HIPÓTESIS que exige confirmación OOS (que no tenemos).
  Holm sobre los K p-valores de Spearman para el veredicto formal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import experiments.walkforward_robustez as wf
from config import CALIBRATION_END
from experiments.momentum_exante_rule import variance_ratio, momentum_internal_acc

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
SCAN = Path("outputs/experiments/m10_pivot_scan.json")
OUT = Path("outputs/experiments/momentum_exante_battery.json")


def eff_ratio(r: np.ndarray) -> float:
    """Kaufman efficiency ratio: |movimiento neto| / longitud del camino. Alto ⇒ tendencia limpia."""
    path = np.sum(np.abs(r))
    return float(abs(np.sum(r)) / path) if path > 0 else 0.0


def sign_ac1(r: np.ndarray) -> float:
    s = np.sign(r)
    s = s[s != 0]
    if len(s) < 3:
        return 0.0
    return float(np.corrcoef(s[1:], s[:-1])[0, 1])


def diagnostics(ret_calib: pd.Series) -> dict:
    r = ret_calib
    out = {}
    for w, tag in [(None, "full"), (504, "504"), (252, "252"), (126, "126")]:
        rw = r if w is None else r.iloc[-w:]
        out[f"mom_acc_{tag}"] = round(momentum_internal_acc(rw, 21)[0], 4)
    out["vr21_z_full"] = round(variance_ratio(r.to_numpy(), 21)[1], 3)
    out["vr21_z_252"] = round(variance_ratio(r.iloc[-252:].to_numpy(), 21)[1], 3)
    out["effratio_252"] = round(eff_ratio(r.iloc[-252:].to_numpy()), 4)
    out["effratio_126"] = round(eff_ratio(r.iloc[-126:].to_numpy()), 4)
    out["signac1_252"] = round(sign_ac1(r.iloc[-252:].to_numpy()), 4)
    return out


def main() -> None:
    scan = json.load(open(SCAN))["por_activo"]
    rows = {}
    for tk in PANEL:
        _, ret = wf.load_features(tk)
        rc = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
        d = diagnostics(rc)
        c = scan[tk]["configs"]
        d["delta_acc_oos"] = round(c["aug"]["accuracy"] - c["base"]["accuracy"], 4)
        rows[tk] = d

    df = pd.DataFrame(rows).T
    diag_cols = [c for c in df.columns if c != "delta_acc_oos"]
    y = df["delta_acc_oos"].astype(float)

    resultados = {}
    for col in diag_cols:
        rho, p = spearmanr(df[col].astype(float), y)
        spy_gt_smci = bool(df.loc["SPY", col] > df.loc["SMCI", col])    # SPY debe rankear más trending que SMCI
        resultados[col] = {"rho": round(float(rho), 3), "p": round(float(p), 4),
                           "spy_sobre_smci": spy_gt_smci,
                           "candidato": bool(rho > 0 and p < 0.10 and spy_gt_smci)}

    # Holm sobre los p de Spearman (solo informativo: dirección la fija rho>0)
    pares = sorted(resultados.items(), key=lambda kv: kv[1]["p"])
    K = len(pares)
    holm_rechaza = None
    for i, (col, r) in enumerate(pares):
        umbral = 0.10 / (K - i)
        if r["p"] <= umbral and r["rho"] > 0:
            holm_rechaza = col
            break

    candidatos = [c for c, r in resultados.items() if r["candidato"]]
    veredicto = {
        "n_diagnosticos": K,
        "prob_1_falso_positivo_0.10": round(1 - 0.9 ** K, 3),
        "candidatos_nominales": candidatos,
        "holm_rechaza": holm_rechaza,
        "hay_regla_exante_robusta": bool(holm_rechaza is not None),
    }
    OUT.write_text(json.dumps({"meta": {"calib_end": CALIBRATION_END, "panel": PANEL},
                               "por_activo": rows, "spearman_por_diagnostico": resultados,
                               "veredicto": veredicto}, indent=2, ensure_ascii=False))

    print(df[diag_cols + ["delta_acc_oos"]].to_string())
    print(f"\n{'diagnóstico':<16}{'rho':>8}{'p':>9}{'SPY>SMCI':>10}{'candidato':>11}")
    for col, r in sorted(resultados.items(), key=lambda kv: kv[1]["p"]):
        print(f"{col:<16}{r['rho']:>8.3f}{r['p']:>9.4f}{str(r['spy_sobre_smci']):>10}{str(r['candidato']):>11}")
    print(f"\nK={K} diagnósticos · P(≥1 falso positivo a 0.10)≈{veredicto['prob_1_falso_positivo_0.10']}")
    print(f"Candidatos nominales (ρ>0, p<0.10, SPY>SMCI): {candidatos or 'NINGUNO'}")
    print(f"Sobrevive Holm: {holm_rechaza or 'NINGUNO'}  ⇒  ¿regla ex-ante robusta? {veredicto['hay_regla_exante_robusta']}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
