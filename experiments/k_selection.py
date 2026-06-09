"""Selección rigurosa del número de regímenes K del HMM, sobre la CALIBRACIÓN (sin OOS ni trading).

Decide K con criterios de modelo, no de P&L:
  1. Verosimilitud FUERA DE MUESTRA (validación temporal expanding-window dentro de 2000–2024-09):
     ¿cada estado extra mejora la descripción de datos no vistos, o sobreajusta? Criterio honesto,
     a diferencia de BIC/AIC que premian más estados por la mala especificación gaussiana.
  2. Ocupación y duración esperada por estado: ¿son todos regímenes reales o hay redundantes/transitorios?
  3. Interpretabilidad: media y dispersión del retorno por estado (¿mapean a Calma/Estrés/Crisis?).

Uso: ``python experiments/k_selection.py`` → outputs/experiments/k_selection.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, CALIBRATION_START, DATA_DIR
from core import data, features
from core.hmm import RegimeHMM

TICKER = "SPY"
KS = [2, 3]  # solo 2 vs 3: K≥4 no es interpretable (no mapea a regímenes económicos)
FOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]  # fracciones de corte: fit en [0,s), score held-out en [s, s+0.1)


def heldout_loglik(calib: np.ndarray, K: int) -> float:
    # Verosimilitud media por observación sobre bloques futuros no vistos (expanding window).
    n = len(calib)
    lls = []
    for i, s in enumerate(FOLDS):
        a = int(s * n)
        b = int((s + 0.1) * n) if i < len(FOLDS) - 1 else n
        train, test = calib[:a], calib[a:b]
        if len(test) < 30:
            continue
        try:
            h = RegimeHMM(n_states=K, seed=config.SEED).fit(train)
            ll = float(h.model.score(h._standardize(test))) / len(test)
            lls.append(ll)
        except Exception:
            continue
    return float(np.mean(lls)) if lls else float("nan")


def bic_aic(calib: np.ndarray, K: int) -> dict:
    h = RegimeHMM(n_states=K, seed=config.SEED).fit(calib)
    ll = float(h.model.score(h._standardize(calib)))
    d, T = calib.shape[1], len(calib)
    k = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
    states = h.predict_states(calib)
    A = h.transition_matrix
    r = calib[:, 0]
    occ = [float((states == s).mean()) for s in range(K)]
    dur = [float(1.0 / (1.0 - A[s, s])) if A[s, s] < 1 else float("inf") for s in range(K)]
    mu = [float(r[states == s].mean()) if (states == s).any() else float("nan") for s in range(K)]
    sd = [float(r[states == s].std()) if (states == s).any() else float("nan") for s in range(K)]
    return {"logL_insample": ll, "n_params": k, "AIC": -2 * ll + 2 * k, "BIC": -2 * ll + k * np.log(T),
            "occupancy": [round(x, 3) for x in occ], "duration_days": [round(x, 1) for x in dur],
            "mean_ret": [round(x, 5) for x in mu], "std_ret": [round(x, 4) for x in sd],
            "min_occupancy": round(min(occ), 3)}


def main() -> None:
    end = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)].to_numpy()
    print(f"Calibración {CALIBRATION_START}→{CALIBRATION_END}: {len(calib)} obs. Selección de K (sin OOS).\n")

    rows = {}
    for K in KS:
        ic = bic_aic(calib, K)
        ho = heldout_loglik(calib, K)
        ic["heldout_loglik_perobs"] = round(ho, 4)
        rows[K] = ic
        print(f"K={K}: held-out LL/obs={ho:+.4f}  BIC={ic['BIC']:.0f}  AIC={ic['AIC']:.0f}  "
              f"min_occ={ic['min_occupancy']:.3f}  dur={ic['duration_days']}")
        print(f"      mean_ret={ic['mean_ret']}  occ={ic['occupancy']}")

    ho_by_k = {K: rows[K]["heldout_loglik_perobs"] for K in KS}
    k_best_ho = max(KS, key=lambda K: ho_by_k[K])
    # ¿A partir de qué K deja de mejorar la held-out LL de forma material (>0.001/obs)?
    plateau = KS[0]
    for K in KS[1:]:
        if ho_by_k[K] - ho_by_k[plateau] > 0.001:
            plateau = K
    print(f"\nHeld-out LL/obs por K: {ho_by_k}")
    print(f"K que maximiza held-out LL: {k_best_ho}")
    print(f"K donde la held-out LL deja de mejorar materialmente (>0.001/obs): {plateau}")
    print(f"BIC mínimo en K={min(KS, key=lambda K: rows[K]['BIC'])}; AIC mínimo en K={min(KS, key=lambda K: rows[K]['AIC'])}")

    out = {"ticker": TICKER, "n_obs": len(calib), "ks": KS, "folds": FOLDS,
           "per_k": rows, "k_best_heldout": int(k_best_ho), "k_plateau_heldout": int(plateau),
           "criterio": "held-out log-likelihood (CV temporal) + ocupacion + interpretabilidad, sin OOS ni trading"}
    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "k_selection.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nEscrito: {dst / 'k_selection.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
