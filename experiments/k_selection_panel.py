"""¿Se puede elegir K por activo SIN mirar el futuro? Selección por verosimilitud fuera de muestra.

La autora observó que en el P&L OOS K=2 parecía mejor en SPY (alcista) y K=3 en los volátiles, y
quiere un criterio ex-ante (sin OOS) que capture eso. Aquí elegimos K por activo con la
verosimilitud fuera de muestra de SU PROPIA calibración (CV temporal, 2000→2024-09, nunca toca el
OOS 2024-10+) y comprobamos si el K elegido correlaciona con la volatilidad del activo.

Uso: ``python experiments/k_selection_panel.py`` → outputs/experiments/k_selection_panel.json
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

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
FOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def heldout_loglik(calib, K):
    n = len(calib); lls = []
    for i, s in enumerate(FOLDS):
        a = int(s * n); b = int((s + 0.1) * n) if i < len(FOLDS) - 1 else n
        train, test = calib[:a], calib[a:b]
        if len(test) < 30:
            continue
        try:
            h = RegimeHMM(n_states=K, seed=config.SEED).fit(train)
            lls.append(float(h.model.score(h._standardize(test))) / len(test))
        except Exception:
            continue
    return float(np.mean(lls)) if lls else float("nan")


def run_ticker(tk):
    end = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(tk, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    X = calib.to_numpy()
    ho2, ho3 = heldout_loglik(X, 2), heldout_loglik(X, 3)
    vol_ann = float(calib["r"].std() * np.sqrt(252))      # volatilidad anualizada de calibración
    ann_ret = float(calib["r"].mean() * 252)              # tendencia (drift) de calibración
    return {"ticker": tk, "heldout_LL_K2": round(ho2, 4), "heldout_LL_K3": round(ho3, 4),
            "delta_LL_K3_minus_K2": round(ho3 - ho2, 4), "k_elegido": 3 if ho3 > ho2 else 2,
            "vol_anual_calib": round(vol_ann, 3), "drift_anual_calib": round(ann_ret, 3),
            "n_calib": len(calib)}


def main():
    rows = []
    for tk in PANEL:
        try:
            r = run_ticker(tk); rows.append(r)
            print(f"{tk:6} K_elegido={r['k_elegido']}  ΔLL(K3-K2)={r['delta_LL_K3_minus_K2']:+.4f}  "
                  f"vol_calib={r['vol_anual_calib']:.2f}  drift_calib={r['drift_anual_calib']:+.2f}")
        except Exception as e:  # noqa: BLE001
            rows.append({"ticker": tk, "error": repr(e)}); print(f"{tk}: ERROR {e!r}")

    ok = [r for r in rows if "error" not in r]
    n2 = sum(r["k_elegido"] == 2 for r in ok); n3 = sum(r["k_elegido"] == 3 for r in ok)
    # ¿Correlaciona el K elegido (o ΔLL) con la volatilidad? (hipótesis: más vol → K=3)
    vol = np.array([r["vol_anual_calib"] for r in ok])
    dll = np.array([r["delta_LL_K3_minus_K2"] for r in ok])
    from scipy.stats import spearmanr
    rho_vol, p_vol = spearmanr(vol, dll)
    print(f"\nK=2 elegido en {n2}/{len(ok)} activos; K=3 en {n3}/{len(ok)}.")
    print(f"Correlación Spearman (vol calibración vs ventaja de K=3): ρ={rho_vol:+.2f} (p={p_vol:.3f})")
    print("  ρ>0 ⇒ tu hipótesis: más volátil → K=3 gana más. ρ≈0 ⇒ la vol no decide K.")

    out = {"per_asset": rows, "n_k2": n2, "n_k3": n3,
           "spearman_vol_vs_deltaLL": {"rho": round(float(rho_vol), 3), "p": round(float(p_vol), 3)},
           "criterio": "held-out log-likelihood por activo (CV temporal en calibracion, sin OOS)"}
    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "k_selection_panel.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nEscrito: {dst / 'k_selection_panel.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
