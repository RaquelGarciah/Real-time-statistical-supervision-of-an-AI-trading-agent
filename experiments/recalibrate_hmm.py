"""Regenera cache/models/hmm.pkl desde el pipeline documentado (reproducibilidad).

El pickle heredado se entrenó con 6025 obs (preprocesado antiguo) y NO lo reproduce el código
actual, que produce 6204 obs con la feature documentada (realized_vol_21d, no VIX). Esto rompe
"Restart & Run All reproduce todo". Este script re-ajusta el HMM K=3 con el pipeline actual
(seed 42, n_seeds=10), hace backup del legacy, verifica el GARCH y actualiza calibration_summary.

Uso: ``python experiments/recalibrate_hmm.py``
"""

from __future__ import annotations

import glob
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR
from core import data, features
from core.garch import GARCHModel
from core.hmm import RegimeHMM

TICKER = "SPY"


def main() -> None:
    end = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    calib_ret = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
    print("Pipeline documentado: feature = (log_return, realized_vol_21d).")
    print(f"Calibración {CALIBRATION_START} → {CALIBRATION_END}: n_obs = {len(calib)}")

    # --- HMM legacy (para comparar) ---
    legacy = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
    print(f"\nLegacy hmm.pkl: best_seed={legacy.best_seed}, logL={legacy.best_score:.2f}, "
          f"transmat diag={np.round(np.diag(legacy.transition_matrix), 4)}")

    # --- HMM nuevo, reproducible ---
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    print(f"Nuevo   hmm.pkl: best_seed={hmm.best_seed}, logL={hmm.best_score:.2f}, "
          f"transmat diag={np.round(np.diag(hmm.transition_matrix), 4)}")
    print(f"state_labels: {hmm.state_labels}")

    # Backup + escritura.
    shutil.copy(CACHE_MODELS_DIR / "hmm.pkl", CACHE_MODELS_DIR / "hmm_legacy_6025.pkl")
    pickle.dump(hmm, open(CACHE_MODELS_DIR / "hmm.pkl", "wb"))
    print("\nGuardado: cache/models/hmm.pkl (legacy → hmm_legacy_6025.pkl)")

    # --- GARCH: verificar reproducibilidad (no depende del HMM) ---
    g_new = GARCHModel().fit(calib_ret)
    g_old = pickle.load(open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "rb"))
    p_new, p_old = g_new.params, g_old.params
    same = all(abs(getattr(p_new, k) - getattr(p_old, k)) < 1e-6
               for k in ("omega", "alpha", "beta", "nu"))
    print(f"\nGARCH α+β: nuevo={p_new.alpha + p_new.beta:.4f}  legacy={p_old.alpha + p_old.beta:.4f}  "
          f"{'(coincide, no se toca)' if same else '(DIFIERE — se regenera)'}")
    if not same:
        shutil.copy(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", CACHE_MODELS_DIR / f"garch_{TICKER}_legacy.pkl")
        pickle.dump(g_new, open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "wb"))

    # --- Actualizar calibration_summary.json (bloque HMM + n_obs) ---
    cs_path = CACHE_MODELS_DIR / "calibration_summary.json"
    cs = json.load(open(cs_path))
    cs["n_obs"] = int(len(calib))
    cs["hmm"]["transition_matrix"] = hmm.transition_matrix.tolist()
    cs["hmm"]["state_labels"] = {str(k): v for k, v in hmm.state_labels.items()}
    cs["hmm"]["best_seed"] = int(hmm.best_seed)
    cs["hmm"]["logL"] = float(hmm.best_score)
    cs["hmm"]["n_obs"] = int(len(calib))
    cs["hmm"]["feature"] = "(log_return, realized_vol_21d_annualized)"
    cs["_regenerado"] = "2026-06-08: HMM re-ajustado desde pipeline documentado (6204 obs), reproducible."
    json.dump(cs, open(cs_path, "w"), indent=1, ensure_ascii=False)
    print(f"calibration_summary.json actualizado (n_obs {cs['n_obs']}).")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
