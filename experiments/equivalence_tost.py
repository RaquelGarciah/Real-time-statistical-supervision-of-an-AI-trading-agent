"""¿El aprendiz REDESCUBRE la regla o la BATE? Test de equivalencia (TOST) M10/AutoML vs M8.

La hipótesis de universalidad ("el ML redescubre STRATA, no inventa otra señal ni la bate") NO se sostiene con un
test de diferencia no significativo (ausencia de evidencia ≠ evidencia de ausencia). Se sostiene con un
**contraste de equivalencia**: TOST (two one-sided tests, Schuirmann 1987). Reportamos el cuadro 2×2 honesto
—equivalencia y/o superioridad— para cada par aprendiz-vs-regla:

  - Δ = métrica(aprendiz) − métrica(M8), pareado, sobre los mismos días.
  - EQUIVALENCIA (TOST a α=0.05): se concluye si el IC al 90% de Δ (bootstrap por bloques) ⊂ (−δ, +δ),
    con δ el margen de irrelevancia pre-registrado. Equivale a los dos contrastes unilaterales de Schuirmann.
  - SUPERIORIDAD (unilateral): p = P_boot(Δ ≤ 0); pequeño ⇒ el aprendiz bate a la regla.

Métricas: accuracy direccional (titular) y Sharpe. Margen δ pre-registrado + sensibilidad (no cherry-picking).
Posiciones/retornos ±1 reconstruidos del acierto canónico. Determinista. Uso: python experiments/equivalence_tost.py
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
PAIRS = [("m10_xgb", "m8"), ("automl", "m8")]   # aprendiz vs regla
NAME = {"m10_xgb": "M10", "automl": "AutoML", "m8": "M8"}
DELTA_ACC = 0.03          # margen pre-registrado de accuracy: ≈ 1 SE de un activo (√(0.25/250)); económicamente nimio
DELTA_SR = 0.50           # margen pre-registrado de Sharpe anualizado (diferencia de riesgo irrelevante entre estrategias)
SENS_ACC = [0.01, 0.02, 0.03, 0.05]
SENS_SR = [0.25, 0.50, 0.75, 1.00]
B_REPS = 2000
ANN = np.sqrt(252.0)
OUT = Path("outputs/experiments/equivalence_tost.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _series(tk: str):
    pan = json.load(open(PANEL_FILE))["por_activo"][tk]
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    dates = ANR[tk]["dates"]; rnext = oos.shift(-1).reindex(pd.to_datetime(dates)).to_numpy()
    absr = np.abs(rnext); valid = ~np.isnan(rnext) & (np.sign(rnext) != 0)
    cba = pan["correct_by_arm"]
    out = {}
    for a in ("m10_xgb", "automl", "m8"):
        c = np.asarray(cba[a], float)
        out[a] = {"correct": c[valid], "ret": ((2 * c - 1) * absr)[valid]}
    return out


def _block_boot(diff_fn, arrs, n):
    """Bootstrap por bloques (estacionario) del estadístico diff_fn sobre índices compartidos."""
    block = max(2, int(round(np.sqrt(n)))); p = 1.0 / block
    rng = np.random.default_rng(config.SEED); out = np.empty(B_REPS)
    for i in range(B_REPS):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        out[i] = diff_fn(idx)
    return out


def _tost(boot, point, deltas, sens):
    """IC90 (TOST α=0.05) + veredicto de equivalencia por margen + superioridad unilateral."""
    lo, hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    res = {"point": round(point, 4), "ci90_low": round(lo, 4), "ci90_high": round(hi, 4),
           "p_superioridad_aprendiz": round(float(np.mean(boot <= 0)), 4),
           "equivalente_delta_preReg": bool(-deltas < lo and hi < deltas), "delta_preReg": deltas,
           "sensibilidad": {str(d): bool(-d < lo and hi < d) for d in sens}}
    return res


def _analyze(s, scope):
    res = {}
    for a, b in PAIRS:
        n = len(s[a]["correct"])
        ca, cb = s[a]["correct"], s[b]["correct"]; ra, rb = s[a]["ret"], s[b]["ret"]
        d_acc = _block_boot(lambda idx: ca[idx].mean() - cb[idx].mean(), None, n)
        d_sr = _block_boot(lambda idx: _sr(ra[idx]) - _sr(rb[idx]), None, n)
        res[f"{NAME[a]}_vs_{NAME[b]}"] = {
            "n": n,
            "accuracy": _tost(d_acc, float(ca.mean() - cb.mean()), DELTA_ACC, SENS_ACC),
            "sharpe": _tost(d_sr, _sr(ra) - _sr(rb), DELTA_SR, SENS_SR),
            "acc_aprendiz": round(float(ca.mean()), 4), "acc_regla": round(float(cb.mean()), 4),
            "sharpe_aprendiz": round(_sr(ra), 3), "sharpe_regla": round(_sr(rb), 3)}
    return res


def _verdict(r):
    """Cuadro 2×2: equivalente / superior / no concluyente."""
    eq = r["equivalente_delta_preReg"]; sup = r["p_superioridad_aprendiz"] < 0.05
    if sup and not eq: return "BATE (superior, no equivalente)"
    if eq and not sup: return "REDESCUBRE (equivalente, no bate)"
    if eq and sup: return "equivalente y borde-superior (margen amplio)"
    return "no concluyente (ni equivalencia ni superioridad)"


def main() -> None:
    config.set_seeds(config.SEED)
    per = {tk: _series(tk) for tk in PANEL10}
    spy = _analyze(per["SPY"], "SPY")
    pool = {a: {k: np.concatenate([per[t][a][k] for t in PANEL10]) for k in ("correct", "ret")}
            for a in ("m10_xgb", "automl", "m8")}
    pooled = _analyze(pool, "POOLED10")

    res = {"meta": {"panel": PANEL10, "pares": [f"{NAME[a]}_vs_{NAME[b]}" for a, b in PAIRS],
                    "delta_acc_preReg": DELTA_ACC, "delta_sharpe_preReg": DELTA_SR, "b_reps": B_REPS,
                    "metodo": "TOST (Schuirmann 1987) vía IC90 bootstrap-bloque ⊂ (−δ,δ); superioridad = "
                              "unilateral P_boot(Δ≤0). Δ = aprendiz − M8, pareado.",
                    "margen_just": "δ_acc=0.03 ≈ 1 SE de accuracy de un activo (√(0.25/250)); δ_Sharpe=0.50 "
                                   "anualizado = diferencia de riesgo económicamente irrelevante. Se reporta sensibilidad."},
           "SPY": spy, "POOLED10": pooled}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    for scope, d in (("SPY", spy), ("POOLED10", pooled)):
        print(f"\n===== {scope} =====")
        for par, r in d.items():
            print(f"  {par}: acc {r['acc_aprendiz']} vs {r['acc_regla']} | Sharpe {r['sharpe_aprendiz']:+.2f} vs {r['sharpe_regla']:+.2f}")
            for m in ("accuracy", "sharpe"):
                x = r[m]
                print(f"    [{m:8s}] Δ={x['point']:+.3f} IC90=[{x['ci90_low']:+.3f},{x['ci90_high']:+.3f}] "
                      f"δ={x['delta_preReg']} → {_verdict(x)}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
