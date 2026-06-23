"""Robustez del panel de 10 (rodante, val/test, alcista/bajista) — SIN re-entrenar, desde el panel canónico.

Las tres pruebas que faltaban a nivel de PANEL (no solo SMCI), para mostrar que el rescate no es suerte:
  1. ACCURACY RODANTE: media móvil de aciertos por estrategia → fracción de ventanas donde la mejor STRATA
     bate al agente y al baseline trivial. Si gana en la mayoría de ventanas, no es un golpe de suerte puntual.
  2. VAL/TEST en distintas particiones (60/40, 70/30, 80/20): accuracy de cada estrategia en val y test →
     ¿gana la mejor STRATA en ambos tramos y en las tres particiones?
  3. RESCATE EN ALCISTA vs BAJISTA: McNemar pareado (M8/M10/AutoML vs M5) en sub-periodos definidos por la
     tendencia a 21 días, por activo y POOLED sobre los 10 → ¿el rescate sobrevive a un test en cada régimen?

Todo se calcula desde `correct_by_arm` del panel canónico mm25 (acierto día a día de las 6 estrategias, incl.
AutoML) — es la MISMA configuración, sin recomputar. La tendencia (alcista/bajista) sale de los retornos del
activo (`wf.load_features`), sin re-fitear HMM/GARCH. Uso: python experiments/panel_robustness.py
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
from core.stats import mcnemar_test

PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
ANR = json.load(open("outputs/experiments/automl_net_returns.json"))["por_activo"]
PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
ARMS = ["m5", "m8", "m10_xgb", "automl", "zeror", "bh"]
SUP = ["m8", "m10_xgb", "automl"]
OUT = Path("outputs/experiments/panel_robustness.json")


def _trend_labels(tk: str, dates: list) -> np.ndarray:
    """Etiqueta cada día como tendencia 21d (causal, shift-1) positiva/negativa, alineada a `dates`."""
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    trend = oos.rolling(21, min_periods=5).mean().shift(1)
    idx = pd.to_datetime(dates)
    return trend.reindex(idx).to_numpy()


def run_ticker(tk: str, pan: dict) -> dict:
    cba = {a: np.array(pan[tk]["correct_by_arm"][a], dtype=int) for a in ARMS}
    dates = ANR[tk]["dates"]; n = len(dates)
    acc = {a: float(cba[a].mean()) for a in ARMS}
    best = max(SUP, key=lambda a: acc[a])                       # mejor STRATA por accuracy en este activo
    triv_arm = "zeror" if acc["zeror"] >= acc["bh"] else "bh"

    # 1) RODANTE (ventana 63 días): serie de accuracy por estrategia + fracción de ventanas que gana la mejor STRATA
    W = 63
    roll = {a: pd.Series(cba[a]).rolling(W).mean().to_numpy() for a in ("m5", best, triv_arm)}
    valid = ~np.isnan(roll["m5"])
    frac_gt_m5 = float((roll[best][valid] > roll["m5"][valid]).mean())
    frac_gt_triv = float((roll[best][valid] > roll[triv_arm][valid]).mean())
    rolling = {"ventana": W, "mejor_strata": best, "trivial": triv_arm,
               "frac_ventanas_mejorSTRATA_gt_M5": round(frac_gt_m5, 3),
               "frac_ventanas_mejorSTRATA_gt_trivial": round(frac_gt_triv, 3),
               "serie": {a: [None if np.isnan(x) else round(float(x), 4) for x in roll[a]] for a in roll},
               "dates": dates}

    # 2) VAL/TEST en tres particiones
    valtest = {}
    for fr in (0.6, 0.7, 0.8):
        i0 = int(n * fr)
        v = {a: round(float(cba[a][:i0].mean()), 4) for a in ARMS}
        t = {a: round(float(cba[a][i0:].mean()), 4) for a in ARMS}
        bs = max(SUP, key=lambda a: t[a])
        valtest[f"{int(round(fr*100))}_{int(round((1-fr)*100))}"] = {
            "val": v, "test": t, "mejor_strata_test": bs,
            "mejorSTRATA_gt_M5_en_val_y_test": bool(v[bs] > v["m5"] and t[bs] > t["m5"]),
            "mejorSTRATA_gt_trivial_en_test": bool(t[bs] > max(t["zeror"], t["bh"]))}

    # 3) ALCISTA vs BAJISTA: McNemar (sup vs M5) por sub-periodo
    trend = _trend_labels(tk, dates)
    bull = trend > 0; bear = trend < 0
    bb = {"n_alcista": int(np.nansum(bull)), "n_bajista": int(np.nansum(bear))}
    for sup in SUP:
        for reg, msk in (("alcista", bull), ("bajista", bear)):
            m = msk & ~np.isnan(trend)
            if m.sum() >= 20:
                _, p, b, c = mcnemar_test(cba["m5"][m], cba[sup][m])
                bb[f"{sup}_vs_m5_{reg}"] = {"n": int(m.sum()), "dAcc": round(float(cba[sup][m].mean() - cba["m5"][m].mean()), 4),
                                            "mcnemar_p": round(float(p), 4), "b": int(b), "c": int(c)}
    return {"n": n, "acc": {a: round(acc[a], 4) for a in ARMS}, "rolling": rolling,
            "valtest": valtest, "bullbear": bb, "_cba": cba, "_trend": trend}


def main() -> None:
    pan = json.load(open(PANEL_FILE))["por_activo"]
    res = {"meta": {"panel": PANEL10, "fuente": "correct_by_arm del panel canónico mm25 (sin recomputar); "
                    "tendencia 21d de wf.load_features", "ventana_rodante": 63,
                    "particiones": ["60/40", "70/30", "80/20"]},
           "por_activo": {}}
    pooled_cba = {a: [] for a in ARMS}; pooled_bull = []; pooled_bear = []
    for tk in PANEL10:
        r = run_ticker(tk, pan)
        cba = r.pop("_cba"); trend = r.pop("_trend")
        res["por_activo"][tk] = r
        for a in ARMS:
            pooled_cba[a].append(cba[a])
        pooled_bull.append(trend > 0); pooled_bear.append(trend < 0)
        bb = r["bullbear"]
        print(f"{tk:5s} rodante mejorSTRATA>M5 {r['rolling']['frac_ventanas_mejorSTRATA_gt_M5']:.0%} | "
              f"alcista {bb.get(r['rolling']['mejor_strata']+'_vs_m5_alcista',{}).get('mcnemar_p','—')} "
              f"bajista {bb.get(r['rolling']['mejor_strata']+'_vs_m5_bajista',{}).get('mcnemar_p','—')}", flush=True)

    # POOLED alcista/bajista (concatenar los 10): McNemar de cada sup vs M5 en cada régimen
    PC = {a: np.concatenate(pooled_cba[a]) for a in ARMS}
    PB = np.concatenate(pooled_bull); PR = np.concatenate(pooled_bear)
    pooled = {"n_total": int(len(PC["m5"])), "n_alcista": int(PB.sum()), "n_bajista": int(PR.sum()), "tests": {}}
    for sup in SUP:
        for reg, msk in (("alcista", PB), ("bajista", PR)):
            _, p, b, c = mcnemar_test(PC["m5"][msk], PC[sup][msk])
            pooled["tests"][f"{sup}_vs_m5_{reg}"] = {"n": int(msk.sum()),
                "dAcc": round(float(PC[sup][msk].mean() - PC["m5"][msk].mean()), 4),
                "mcnemar_p": round(float(p), 4), "b": int(b), "c": int(c),
                "sig_0.10": bool(p < 0.10)}
    res["pooled_bullbear"] = pooled
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== POOLED alcista/bajista (rescate del agente, McNemar sup vs M5) ===")
    for k, v in pooled["tests"].items():
        print(f"  {k:22s} n={v['n']:4d} ΔAcc={v['dAcc']:+.3f} p={v['mcnemar_p']:.4f} {'SIG' if v['sig_0.10'] else '—'}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
