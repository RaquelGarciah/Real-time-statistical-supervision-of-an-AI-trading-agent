"""Selección de K por rendimiento DIRECCIONAL fuera de muestra sobre la calibración (sin OOS).

Crítica de Raquel: la held-out likelihood mide DENSIDAD (y por eso pide 3 siempre), pero STRATA
usa el régimen para DIRECCIÓN. Aquí juzgamos K por el criterio alineado con el propósito: el
rendimiento direccional del régimen fuera de muestra, promediado sobre 24 años de calibración
(validación temporal expanding-window). Nunca toca el OOS 2024-10+.

Por fold: ajusta HMM(K) en [0,a), calcula el posterior FILTRADO (causal) en el bloque [a,b),
toma la dirección del régimen dominante (Calma→+1, Crisis→−1, Estrés→0 en K=3; baja-vol→+1,
alta-vol→−1 en K=2) y mide retorno direccional r_{t+1}·posición y acierto. Agrega entre folds.

Uso: ``python experiments/k_selection_directional.py`` → outputs/experiments/k_selection_directional.json
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
ANN = np.sqrt(252)


def _positions(h, feat_block_full, n_train, K):
    # Posterior filtrado causal sobre train+bloque; usa solo las filas del bloque.
    g = h.predict_proba_filtered(feat_block_full)[n_train:]  # (n_block, K), columnas ordenadas por vol asc
    map_state = g.argmax(1)
    pos = np.zeros(len(map_state))
    pos[map_state == 0] = 1.0           # estado de menor vol → long (Calma / baja-vol)
    pos[map_state == K - 1] = -1.0      # estado de mayor vol → short (Crisis / alta-vol)
    # estados intermedios (Estrés en K=3) → 0 (abstención)
    return pos


def directional_oos(feat, ret, K):
    X = feat.to_numpy(); n = len(X)
    rets, hits = [], []
    for i, s in enumerate(FOLDS):
        a = int(s * n); b = int((s + 0.1) * n) if i < len(FOLDS) - 1 else n
        if b - a < 30:
            continue
        try:
            h = RegimeHMM(n_states=K, seed=config.SEED).fit(X[:a])
        except Exception:
            continue
        pos = _positions(h, X[:b], a, K)                       # posiciones causales en el bloque
        idx_block = feat.index[a:b]
        r_next = ret.shift(-1).reindex(idx_block).to_numpy()
        ok = ~np.isnan(r_next)
        pr = pos[ok] * r_next[ok]
        rets.append(pr)
        nz = pos[ok] != 0
        if nz.any():
            hits.append((np.sign(pos[ok][nz]) == np.sign(r_next[ok][nz])).astype(float))
    if not rets:
        return float("nan"), float("nan"), 0
    allr = np.concatenate(rets)
    acc = float(np.concatenate(hits).mean()) if hits else float("nan")
    sharpe = float(allr.mean() / allr.std() * ANN) if allr.std() > 0 else float("nan")
    n_act = int(sum(len(h) for h in hits))
    return sharpe, acc, n_act


def run_ticker(tk):
    end = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(tk, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    cret = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
    sh2, acc2, n2 = directional_oos(calib, cret, 2)
    sh3, acc3, n3 = directional_oos(calib, cret, 3)
    return {"ticker": tk, "dir_sharpe_K2": round(sh2, 3), "dir_sharpe_K3": round(sh3, 3),
            "dir_acc_K2": round(acc2, 4), "dir_acc_K3": round(acc3, 4),
            "k_mejor_dir_sharpe": 3 if sh3 > sh2 else 2, "k_mejor_dir_acc": 3 if acc3 > acc2 else 2,
            "vol_anual_calib": round(float(calib["r"].std() * ANN), 3)}


def main():
    rows = []
    for tk in PANEL:
        try:
            r = run_ticker(tk); rows.append(r)
            print(f"{tk:6} dirSharpe K2={r['dir_sharpe_K2']:+.2f} K3={r['dir_sharpe_K3']:+.2f} "
                  f"| dirAcc K2={r['dir_acc_K2']:.3f} K3={r['dir_acc_K3']:.3f} "
                  f"| mejor(Sharpe)=K{r['k_mejor_dir_sharpe']} | vol={r['vol_anual_calib']:.2f}")
        except Exception as e:  # noqa: BLE001
            rows.append({"ticker": tk, "error": repr(e)}); print(f"{tk}: ERROR {e!r}")

    ok = [r for r in rows if "error" not in r]
    n3s = sum(r["k_mejor_dir_sharpe"] == 3 for r in ok)
    n3a = sum(r["k_mejor_dir_acc"] == 3 for r in ok)
    from scipy.stats import spearmanr
    vol = np.array([r["vol_anual_calib"] for r in ok])
    dsh = np.array([r["dir_sharpe_K3"] - r["dir_sharpe_K2"] for r in ok])
    rho, p = spearmanr(vol, dsh)
    print(f"\nPor dir-Sharpe OOS-calibración: K=3 mejor en {n3s}/{len(ok)}; por dir-accuracy: {n3a}/{len(ok)}.")
    print(f"Correlación vol vs ventaja direccional de K=3 (Sharpe): ρ={rho:+.2f} (p={p:.3f})")
    print("  ρ>0 ⇒ tu hipótesis (más volátil → K=3 mejor direccionalmente). ρ≈0 ⇒ la vol no decide K.")

    out = {"per_asset": rows, "n_k3_best_sharpe": n3s, "n_k3_best_acc": n3a,
           "spearman_vol_vs_dir_advantage": {"rho": round(float(rho), 3), "p": round(float(p), 3)},
           "criterio": "rendimiento direccional fuera de muestra (CV temporal en calibracion, posterior filtrado, sin OOS)"}
    dst = Path("outputs/experiments"); dst.mkdir(parents=True, exist_ok=True)
    (dst / "k_selection_directional.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nEscrito: {dst / 'k_selection_directional.json'}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
