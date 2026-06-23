"""¿Es SIGNIFICATIVA la complementariedad por régimen de los dos aprendices? (difference-in-differences).

El patrón descriptivo (bullbear_confirmatory): M10 rescata más en alcista y AutoML más en bajista. Aquí lo
convertimos en contraste: la "diferencia-en-diferencias" de Sharpe

    DiD = [SR_M10(alcista) − SR_AutoML(alcista)] − [SR_M10(bajista) − SR_AutoML(bajista)]

mide si la VENTAJA RELATIVA M10 vs AutoML CAMBIA entre regímenes. La M5 se cancela (ambos vs el mismo agente), así
que DiD>0 ⇔ M10 es relativamente mejor en alcista y AutoML en bajista (complementariedad en espejo). H0: DiD=0
(los dos aprendices no se especializan por régimen). Bootstrap estacionario pareado (Politis-Romano 1994) sobre
los retornos ±1 reconstruidos del acierto canónico; el régimen viaja con cada día remuestreado. Misma convención
que bullbear_confirmatory (block=√n, B=2000, seed=42). SPY y POOLED-10.

Uso: python experiments/regime_did_learners.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
import experiments.walkforward_robustez as wf
from config import STRATA_OOS_START

PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
ANR = json.load(open("outputs/experiments/automl_net_returns.json"))["por_activo"]
PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
B_REPS = 2000
ANN = np.sqrt(252.0)
OUT = Path("outputs/experiments/regime_did_learners.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _series(tk: str):
    """Retornos diarios ±1 de M10 y AutoML + signo del drift 21d, en días válidos."""
    pan = json.load(open(PANEL_FILE))["por_activo"][tk]
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    dates = ANR[tk]["dates"]
    rnext = oos.shift(-1).reindex(pd.to_datetime(dates)).to_numpy(); absr = np.abs(rnext)
    dr = np.sign(oos.rolling(21, min_periods=5).mean().shift(1).reindex(pd.to_datetime(dates)).to_numpy())
    valid = ~np.isnan(rnext) & (np.sign(rnext) != 0) & ~np.isnan(dr) & (dr != 0)
    r_m10 = (2 * np.asarray(pan["correct_by_arm"]["m10_xgb"], float) - 1) * absr
    r_aml = (2 * np.asarray(pan["correct_by_arm"]["automl"], float) - 1) * absr
    return r_m10[valid], r_aml[valid], dr[valid]


def _did(r_m10, r_aml, dr) -> float:
    bull, bear = dr > 0, dr < 0
    return (_sr(r_m10[bull]) - _sr(r_aml[bull])) - (_sr(r_m10[bear]) - _sr(r_aml[bear]))


def _boot(r_m10, r_aml, dr) -> dict:
    n = len(r_m10); block = max(2, int(round(np.sqrt(n)))); p = 1.0 / block
    rng = np.random.default_rng(config.SEED); dd = np.empty(B_REPS)
    for i in range(B_REPS):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        dd[i] = _did(r_m10[idx], r_aml[idx], dr[idx])
    point = _did(r_m10, r_aml, dr)
    p_one = float(np.mean(dd <= 0))  # H0: DiD<=0 vs H1: DiD>0 (complementariedad)
    bull, bear = dr > 0, dr < 0
    return {"did_point": round(point, 4), "ci95_low": round(float(np.quantile(dd, 0.025)), 4),
            "ci95_high": round(float(np.quantile(dd, 0.975)), 4),
            "p_one_sided_did_gt_0": round(p_one, 4), "median_boot": round(float(np.median(dd)), 4),
            "n_obs": int(n), "n_alcista": int(bull.sum()), "n_bajista": int(bear.sum()),
            "m10_minus_aml_alcista": round(_sr(r_m10[bull]) - _sr(r_aml[bull]), 4),
            "m10_minus_aml_bajista": round(_sr(r_m10[bear]) - _sr(r_aml[bear]), 4),
            "sr_m10_alcista": round(_sr(r_m10[bull]), 3), "sr_aml_alcista": round(_sr(r_aml[bull]), 3),
            "sr_m10_bajista": round(_sr(r_m10[bear]), 3), "sr_aml_bajista": round(_sr(r_aml[bear]), 3),
            "block_len": int(block), "n_reps": B_REPS}


def main() -> None:
    config.set_seeds(config.SEED)
    per = {tk: _series(tk) for tk in PANEL10}
    spy = _boot(*per["SPY"])
    pm10 = np.concatenate([per[t][0] for t in PANEL10]); pam = np.concatenate([per[t][1] for t in PANEL10])
    pdr = np.concatenate([per[t][2] for t in PANEL10])
    pooled = _boot(pm10, pam, pdr)

    res = {"meta": {"panel": PANEL10, "b_reps": B_REPS, "seed": config.SEED,
                    "estadistico": "DiD = [SR_M10(alc)−SR_AutoML(alc)] − [SR_M10(baj)−SR_AutoML(baj)]; H0: DiD=0; "
                                   "H1 (complementariedad): DiD>0 (M10 mejor en alcista, AutoML en bajista).",
                    "nota": "bootstrap estacionario pareado; el régimen (drift 21d) viaja con cada día remuestreado. "
                            "Veredicto pre-registrado: IC95 excluye 0 (o p one-sided<0.10) → complementariedad "
                            "CONFIRMADA; si cruza 0 → descriptiva (línea futura)."},
           "SPY": spy, "POOLED10": pooled}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    for scope, d in (("SPY", spy), ("POOLED10", pooled)):
        ok = "CONFIRMADA" if d["ci95_low"] > 0 else "no (IC cruza 0)"
        print(f"=== {scope} (n={d['n_obs']}: alc={d['n_alcista']}, baj={d['n_bajista']}) ===")
        print(f"  M10−AutoML alcista={d['m10_minus_aml_alcista']:+.2f}  bajista={d['m10_minus_aml_bajista']:+.2f}")
        print(f"  DiD={d['did_point']:+.2f}  IC95=[{d['ci95_low']:+.2f},{d['ci95_high']:+.2f}]  "
              f"p(DiD>0)={d['p_one_sided_did_gt_0']:.4f}  → complementariedad {ok}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
