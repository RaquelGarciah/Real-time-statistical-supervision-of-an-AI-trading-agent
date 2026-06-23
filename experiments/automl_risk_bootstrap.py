"""Bootstrap de riesgo de AutoML vs M5/B&H/ZeroR (fix 2 del gate del marco práctico).

Liviano: NO recomputa M10 ni SHAP. Reusa las series netas ya en disco —AutoML de
automl_net_returns.json y M5/B&H/ZeroR de decision_automl_prep.json— y aplica el MISMO
bootstrap estacionario pareado (Politis-Romano 1994, bloque sqrt(n), B=2000, semilla fija)
que decision_automl_prep.py, para que automl_vs_* sea comparable con m8_vs_*/m10_vs_*.

Escribe a fichero propio (no pisa el canónico):
  outputs/experiments/automl_risk_bootstrap.json  con por_activo[SPY].boot y pooled.boot.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import config

ANN = np.sqrt(252.0)
B_BOOT = 2000
PREP = Path("outputs/experiments/decision_automl_prep.json")
ANR = Path("outputs/experiments/automl_net_returns.json")
OUT = Path("outputs/experiments/automl_risk_bootstrap.json")
BASE = ("m5", "bh", "zeror")


def _sr(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / s * ANN) if s > 0 else 0.0


def _maxdd(r: np.ndarray) -> float:
    r = np.nan_to_num(r)
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _boot_paired(r_a: np.ndarray, r_b: np.ndarray, stat, seed: int) -> dict:
    """Bootstrap estacionario PAREADO de la mediana de stat(a)-stat(b) (Politis-Romano 1994)."""
    n = len(r_a)
    block = max(2, int(round(np.sqrt(n))))
    p = 1.0 / block
    rng = np.random.default_rng(seed)
    d = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        u = rng.random(n - 1)
        jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        d[i] = stat(r_a[idx]) - stat(r_b[idx])
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return {"median": round(float(np.median(d)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "point": round(stat(r_a) - stat(r_b), 4), "sig": bool(lo > 0 or hi < 0)}


def main() -> None:
    prep = json.load(open(PREP))["por_activo"]
    anr = json.load(open(ANR))["por_activo"]
    ok = [t for t in prep if "net_returns" in prep[t] and t in anr and "automl" in anr[t]]

    res = {"meta": {"seed": config.SEED, "B": B_BOOT, "block": "sqrt(n)",
                    "metodo": "bootstrap estacionario pareado Politis-Romano 1994 (idéntico a decision_automl_prep)",
                    "fuentes": {"automl": str(ANR), "m5/bh/zeror": str(PREP)},
                    "nota": "fix 2 del gate raquel-quant: AutoML faltaba en el bootstrap de riesgo"},
           "por_activo": {}}

    pooled = {k: [] for k in ("automl",) + BASE}
    for t in ok:
        nr_t = {k: np.array(prep[t]["net_returns"][k], dtype=float) for k in BASE}
        aut = np.nan_to_num(np.array(anr[t]["automl"], dtype=float))
        # Alinear longitudes (mismo tramo desplegable); ambos vienen del mismo WF.
        n = min(len(aut), min(len(v) for v in nr_t.values()))
        aut = aut[-n:]; nr_t = {k: v[-n:] for k, v in nr_t.items()}
        boot = {f"automl_vs_{b}": {"dSharpe": _boot_paired(aut, nr_t[b], _sr, config.SEED),
                                   "dMaxDD": _boot_paired(aut, nr_t[b], _maxdd, config.SEED)}
                for b in BASE}
        res["por_activo"][t] = {"n": n, "boot": boot}
        pooled["automl"].append(aut)
        for b in BASE:
            pooled[b].append(nr_t[b])

    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    res["pooled"] = {"n_total": int(len(pooled["automl"])), "n_activos": len(ok),
                     "boot": {f"automl_vs_{b}": {"dSharpe": _boot_paired(pooled["automl"], pooled[b], _sr, config.SEED),
                                                 "dMaxDD": _boot_paired(pooled["automl"], pooled[b], _maxdd, config.SEED)}
                              for b in BASE}}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("=== AutoML vs M5 (rescate de riesgo) ===")
    sp = res["por_activo"]["SPY"]["boot"]["automl_vs_m5"]
    print(f"SPY    ΔSharpe {sp['dSharpe']['point']:+.3f} IC{sp['dSharpe']['ci95']} {'SIG' if sp['dSharpe']['sig'] else '—'}"
          f" · ΔmaxDD {sp['dMaxDD']['point']:+.3f} IC{sp['dMaxDD']['ci95']} {'SIG' if sp['dMaxDD']['sig'] else '—'}")
    pl = res["pooled"]["boot"]["automl_vs_m5"]
    print(f"POOLED ΔSharpe {pl['dSharpe']['point']:+.3f} IC{pl['dSharpe']['ci95']} {'SIG' if pl['dSharpe']['sig'] else '—'}"
          f" · ΔmaxDD {pl['dMaxDD']['point']:+.3f} IC{pl['dMaxDD']['ci95']} {'SIG' if pl['dMaxDD']['sig'] else '—'}")
    print(f"(pooled n={res['pooled']['n_total']}, {res['pooled']['n_activos']} activos) -> {OUT}")


if __name__ == "__main__":
    main()
