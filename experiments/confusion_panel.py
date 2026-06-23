"""Matrices de confusión direccionales — SPY (cada estrategia) y panel (mejor STRATA por activo). Sin recomputar.

Una estrategia direccional toma posición ±1; su matriz de confusión 2×2 cruza la dirección PREDICHA (posición)
con la REAL (signo de r_{t+1}). Como la posición es ±1, se reconstruye EXACTA desde el acierto día a día del
panel canónico: pos_t = sign(r_{t+1})·(2·acierto_t − 1). La verdad sale de los retornos del activo
(`wf.load_features`), alineada a la ventana evaluada. Así las matrices son 100% consistentes con la accuracy del
panel (TP+TN)/n = mean(acierto), sin re-entrenar nada.

  - SPY: las 6 estrategias (M5/M8/M10/AutoML/ZeroR/B&H).
  - Panel (10): la MEJOR estrategia derivada de STRATA por activo (argmax accuracy de M8/M10/AutoML).

Uso: python experiments/confusion_panel.py
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
ARMS = ["m5", "m8", "m10_xgb", "automl", "zeror", "bh"]
NAME = {"m5": "M5", "m8": "M8", "m10_xgb": "M10", "automl": "AutoML", "zeror": "ZeroR", "bh": "B&H"}
SUP = ["m8", "m10_xgb", "automl"]
OUT = Path("outputs/experiments/confusion_panel.json")


def _truth(tk: str, dates: list) -> np.ndarray:
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    rnext = oos.shift(-1).reindex(pd.to_datetime(dates))
    return np.sign(rnext.to_numpy())


def _confusion(pos: np.ndarray, truth: np.ndarray) -> dict:
    m = ~np.isnan(truth) & (truth != 0)
    p, t = pos[m], truth[m]
    TP = int(np.sum((p > 0) & (t > 0))); FP = int(np.sum((p > 0) & (t < 0)))
    FN = int(np.sum((p < 0) & (t > 0))); TN = int(np.sum((p < 0) & (t < 0)))
    n = TP + FP + FN + TN
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": n,
            "accuracy": round((TP + TN) / n, 4) if n else None,
            "precision_long": round(TP / (TP + FP), 4) if (TP + FP) else None,
            "recall_long": round(TP / (TP + FN), 4) if (TP + FN) else None,
            "frac_pred_long": round((TP + FP) / n, 3) if n else None}


def main() -> None:
    pan = json.load(open(PANEL_FILE))["por_activo"]
    res = {"meta": {"nota": "confusión direccional: predicho (posición ±1) vs real (signo r_{t+1}); pos "
                    "reconstruida desde correct_by_arm canónico → consistente con la accuracy del panel."},
           "spy_por_estrategia": {}, "panel_mejor_strata": {}}

    # SPY: las 6 estrategias
    dates = ANR["SPY"]["dates"]; truth = _truth("SPY", dates)
    cba = pan["SPY"]["correct_by_arm"]
    for a in ARMS:
        c = np.array(cba[a]); pos = truth * (2 * c - 1)
        res["spy_por_estrategia"][NAME[a]] = _confusion(pos, truth)
    print("=== SPY · confusión por estrategia ===")
    for nm, cm in res["spy_por_estrategia"].items():
        print(f"  {nm:7s} TP={cm['TP']:3d} FP={cm['FP']:3d} FN={cm['FN']:3d} TN={cm['TN']:3d} | acc={cm['accuracy']} prec_long={cm['precision_long']}")

    # Panel: mejor STRATA-derivada por activo
    print("\n=== Panel · mejor STRATA por activo ===")
    for tk in PANEL10:
        t = pan[tk]["table"]; best = max(SUP, key=lambda s: t[s]["accuracy"])
        dates = ANR[tk]["dates"]; truth = _truth(tk, dates)
        c = np.array(pan[tk]["correct_by_arm"][best]); pos = truth * (2 * c - 1)
        cm = _confusion(pos, truth); cm["estrategia"] = NAME[best]
        res["panel_mejor_strata"][tk] = cm
        print(f"  {tk:5s} mejor={NAME[best]:6s} TP={cm['TP']:3d} FP={cm['FP']:3d} FN={cm['FN']:3d} TN={cm['TN']:3d} | acc={cm['accuracy']}")
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
