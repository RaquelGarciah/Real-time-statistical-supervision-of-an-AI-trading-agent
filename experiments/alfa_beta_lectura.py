"""Lectura alfa-vs-beta (F4.9) — DESCRIPTIVA, sin contraste de hipótesis.

¿El Sharpe positivo de una estrategia STRATA viene de exposición pasiva al activo (BETA) o de valor direccional
propio (ALFA)? Lo leemos con el modelo de mercado clásico (Sharpe 1964; Jensen 1968), pero **como descomposición
descriptiva, NO como test**: regresamos el retorno diario de la mejor STRATA sobre el del benchmark pasivo (B&H,
que es estar siempre largo → su retorno diario es r_{t+1}):

    r_strat,t = α + β · r_mercado,t + ε_t

  - **β** (pendiente) = cuánto co-mueve la estrategia con el activo. β≈1 → es esencialmente "estar largo" = BETA.
  - **α** (intercepto, anualizado ×252) = parte del rendimiento NO explicada por la exposición pasiva.
  - Lectura: BETA si β alto y B&H ya gana (Sharpe>0); ALFA DIRECCIONAL si β bajo/negativo y α>0 mientras el
    pasivo pierde (B&H Sharpe≤0) — la estrategia gana yendo corto/defensiva, no por la subida del activo.

Esperado: SPY = BETA (AutoML va largo en un OOS alcista); SMCI/MARA/UNG = ALFA DIRECCIONAL (el pasivo pierde y la
STRATA acierta el lado corto). Posiciones ±1 reconstruidas del acierto canónico. **Nominal/descriptivo, sin p.**
Uso: python experiments/alfa_beta_lectura.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import experiments.walkforward_robustez as wf
from config import STRATA_OOS_START

PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
ANR = json.load(open("outputs/experiments/automl_net_returns.json"))["por_activo"]
PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
SUP = {"m8": "M8", "m10_xgb": "M10", "automl": "AutoML"}
ANN = np.sqrt(252.0)
OUT = Path("outputs/experiments/alfa_beta_lectura.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def main() -> None:
    pan = json.load(open(PANEL_FILE))["por_activo"]
    res = {"meta": {"panel": PANEL10, "modelo": "market-model r_strat = α + β·r_BH (descriptivo, sin test); "
                    "α anualizada ×252; β = exposición al activo. F4.9 lectura alfa-vs-beta.",
                    "nota": "LECTURA RAZONADA, no afirmación con contraste. nominal."},
           "por_activo": {}}
    for tk in PANEL10:
        t = pan[tk]["table"]; cba = pan[tk]["correct_by_arm"]
        dates = ANR[tk]["dates"]
        _, ret = wf.load_features(tk)
        oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
        rnext = oos.shift(-1).reindex(pd.to_datetime(dates)).to_numpy(); absr = np.abs(rnext)
        valid = ~np.isnan(rnext) & (np.sign(rnext) != 0)
        mkt = rnext[valid]                                  # B&H diario = r_{t+1} (siempre largo)
        best = max(SUP, key=lambda a: t[a]["sharpe"])       # mejor STRATA por Sharpe
        strat = ((2 * np.asarray(cba[best], float) - 1) * absr)[valid]
        beta, alpha = np.polyfit(mkt, strat, 1)             # OLS descriptivo: slope=β, intercept=α diaria
        corr = float(np.corrcoef(strat, mkt)[0, 1])
        bh_sr = _sr(mkt); strat_sr = _sr(strat); alpha_ann = float(alpha) * 252
        # lectura descriptiva: BETA = gana porque el activo sube y va largo; ALFA DIRECCIONAL = gana donde el
        # pasivo NO (B&H Sharpe≤0), vía posicionamiento corto/defensivo (β bajo/negativo o α positiva)
        if beta > 0.5 and bh_sr > 0.5:
            lectura = "BETA (gana por estar largo en un activo que sube)"
        elif strat_sr > 0.5 and bh_sr <= 0.5:
            lectura = "ALFA DIRECCIONAL (gana donde el pasivo pierde, vía posicionamiento corto/defensivo)"
        else:
            lectura = "mixto"
        res["por_activo"][tk] = {"mejor_strata": SUP[best], "strat_sharpe": round(strat_sr, 2),
                                 "bh_sharpe": round(bh_sr, 2), "beta": round(float(beta), 3),
                                 "alfa_anual": round(alpha_ann, 4), "corr_con_mercado": round(corr, 3),
                                 "lectura": lectura}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"{'activo':6}{'mejor':7}{'Sharpe':>8}{'B&H_Sh':>8}{'beta':>7}{'alfa_an':>9}{'corr':>7}  lectura")
    for tk in PANEL10:
        d = res["por_activo"][tk]
        print(f"{tk:6}{d['mejor_strata']:7}{d['strat_sharpe']:>8.2f}{d['bh_sharpe']:>8.2f}{d['beta']:>7.2f}"
              f"{d['alfa_anual']:>9.3f}{d['corr_con_mercado']:>7.2f}  {d['lectura']}")
    print(f"\nLectura (descriptiva, sin test): el Sharpe positivo en activos ALCISTAS (SPY/índices) es sobre todo "
          "BETA (β≈1, va largo); en activos de leverage débil/invertido que CAEN (SMCI/MARA/UNG) la STRATA saca "
          "valor DIRECCIONAL (β bajo, alfa>0) yendo corta/defensiva. No se afirma con contraste: es nominal.")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
