"""Estabilidad del ganador-por-activo bajo bootstrap por bloques.

Pre-registro (BITACORA, antes de mirar resultados):
- Pregunta: ¿es estable la estrategia de mayor accuracy por activo? Si el ganador se
  voltea con frecuencia bajo remuestreo, el patrón "cada activo prefiere una estrategia"
  es ruido muestral y NO se puede clusterizar sobre él.
- H0: el ganador puntual es indistinguible del azar; ningún brazo domina bajo bootstrap.
- Metodo: bootstrap estacionario por bloques (Politis-Romano 1994), bloque medio sqrt(n),
  B=2000, semilla fija. Por replica se recalcula accuracy de cada brazo sobre los dias
  remuestreados y se toma el argmax. Se mide sobre los vectores de acierto diario
  (correct_by_arm) de la ventana comun td del panel canonico.
- Estadistico: winner-stability = frecuencia con que el ganador observado sigue ganando;
  distribucion completa de ganadores; margen al 2do con IC bootstrap.
- Criterio: ganador estable si gana en >=60% de las replicas. Si <50% de los 15 activos
  tienen ganador estable, clusterizar por estrategia-ganadora no procede.

No reentrena nada: reusa correct_by_arm del JSON canonico (determinista).
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

import config

PANEL_JSON = Path("outputs/experiments/automl_runs/"
                  "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
OUT = Path("outputs/experiments/automl_runs/winner_stability_bootstrap.json")
B = 2000
NONTRIVIAL = ("m5", "m8", "m10_xgb", "automl")
ALL_ARMS = ("m5", "m8", "m10_xgb", "automl", "zeror", "bh")


def stationary_boot_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices de una replica del bootstrap estacionario (Politis-Romano 1994)."""
    p = 1.0 / block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(0, n)
    u = rng.random(n - 1)
    jumps = rng.integers(0, n, n - 1)
    for t in range(1, n):
        idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
    return idx


def winner_stability(correct: dict[str, np.ndarray], arms: tuple[str, ...],
                     rng: np.random.Generator) -> dict:
    n = len(correct[arms[0]])
    block = max(2, int(round(np.sqrt(n))))
    C = np.vstack([correct[a] for a in arms])           # (n_arms, n)
    acc_point = C.mean(axis=1)
    win_point = int(np.argmax(acc_point))
    gap_point = float(np.sort(acc_point)[-1] - np.sort(acc_point)[-2])

    wins = np.zeros(len(arms), dtype=int)
    gaps = np.empty(B)
    for b in range(B):
        idx = stationary_boot_idx(n, block, rng)
        acc = C[:, idx].mean(axis=1)
        wins[int(np.argmax(acc))] += 1
        s = np.sort(acc)
        gaps[b] = s[-1] - s[-2]
    dist = {a: round(float(wins[i] / B), 4) for i, a in enumerate(arms)}
    return {
        "n": n, "block": block,
        "acc_point": {a: round(float(acc_point[i]), 4) for i, a in enumerate(arms)},
        "winner_point": arms[win_point],
        "winner_stability": round(float(wins[win_point] / B), 4),
        "winner_dist": dist,
        "gap_to_2nd_point": round(gap_point, 4),
        "gap_ci95": [round(float(np.quantile(gaps, 0.025)), 4),
                     round(float(np.quantile(gaps, 0.975)), 4)],
    }


def main() -> None:
    d = json.load(open(PANEL_JSON))
    pa = d["por_activo"]
    res = {"meta": {"B": B, "seed": config.SEED, "source": str(PANEL_JSON),
                    "block": "sqrt(n)", "stable_threshold": 0.60},
           "por_activo": {}}
    for a, blk in pa.items():
        cba = {k: np.array(v, dtype=float) for k, v in blk["correct_by_arm"].items()}
        rng = np.random.default_rng(config.SEED)
        res["por_activo"][a] = {
            "nontrivial": winner_stability(cba, NONTRIVIAL, rng),
            "all6": winner_stability(cba, ALL_ARMS, np.random.default_rng(config.SEED + 1)),
        }

    # Resumen agregado: cuantos activos tienen ganador estable (>=0.60).
    thr = 0.60
    for scope in ("nontrivial", "all6"):
        stable = [a for a in pa if res["por_activo"][a][scope]["winner_stability"] >= thr]
        winners = Counter(res["por_activo"][a][scope]["winner_point"] for a in pa)
        res.setdefault("resumen", {})[scope] = {
            "n_activos": len(pa), "n_estable": len(stable),
            "frac_estable": round(len(stable) / len(pa), 4),
            "activos_estables": stable,
            "ganadores_puntuales": dict(winners),
            "stability_media": round(float(np.mean(
                [res["por_activo"][a][scope]["winner_stability"] for a in pa])), 4),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)

    print(f"=== Estabilidad del ganador-por-activo (B={B}) ===")
    for scope, lab in (("nontrivial", "M5/M8/M10/AutoML"), ("all6", "+ZeroR/B&H")):
        r = res["resumen"][scope]
        print(f"\n[{lab}] estables(>=.60): {r['n_estable']}/{r['n_activos']} "
              f"(frac={r['frac_estable']}) · stability media={r['stability_media']}")
        print(f"  ganadores puntuales: {r['ganadores_puntuales']}")
        for a in pa:
            w = res["por_activo"][a][scope]
            print(f"  {a:5} gana={w['winner_point']:8} stab={w['winner_stability']:.2f} "
                  f"gap={w['gap_to_2nd_point']:+.3f} ci{w['gap_ci95']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
