"""Recalibración de los modelos de STRATA sobre NVDA (réplica fiel del pipeline de SPY).

Réplica multi-activo: ajusta el detector de régimen (HMM K=3) sobre la PROPIA historia de
NVDA 2000-01-01 → 2024-09-30, en lugar de reutilizar el HMM de SPY. Esto convierte a NVDA en
una prueba honesta de universalidad: SPY funciona porque el *leverage effect* (Black 1976;
Christie 1982) hace que el régimen de alta volatilidad coincida con caídas, así que el régimen
sirve de proxy direccional. En un activo individual esa correspondencia puede romperse — y la
regla pre-registrada **prior-flip** está para cazarlo.

Qué se recalibra sobre NVDA (todo reproducible, semilla 42):
- HMM K=3 sobre (log_return, realized_vol_21d) → ``cache/models/hmm_nvda.pkl``.
- El **prior RAM** (régimen→signo) NO es un fichero: se deriva del orden por volatilidad de los
  estados (Calma→long, Crisis→short). Aquí se mide su validez sobre NVDA con la media de retorno
  por estado en calibración (μ_estado) y se anota si el *leverage effect* se sostiene.
- GARCH(1,1)-t: ya existe ``garch_NVDA.pkl``; se verifica reproducibilidad (no se sobrescribe).

Lo que NO se recalibra (decisión documentada, auditada por @rigor-matematico):
- Compuerta RAM τ=0.5: criterio parameter-free (histograma bimodal), no depende del activo.
- Umbrales de severidad PSA/GSO: se reutiliza ``strata_thresholds.json`` global.
  Justificación de RIGOR (suficiente): en override-C el SIGNO de la posición lo fija RAM
  (strata/intervention.py:158, ``final_size = regime_sign · bound``); PSA y GSO solo ESCALAN la
  magnitud (GSO acota con la banda de garch_NVDA; PSA frena ×0.5). Cambiar sus percentiles de
  severidad no puede voltear la dirección del rescate, que es la dimensión que mide la hipótesis
  del TFG. Además la banda GSO ya es propia de NVDA (sale de garch_NVDA).
  LIMITACIÓN declarada (no es una justificación de rigor): el generador ex-ante de esos percentiles
  no está versionado (strata_thresholds.json tiene n_obs legacy 6025), así que regenerarlos no
  sería reproducible. Por eso se heredan los de SPY y toda cifra de magnitud/Sharpe de M8 se reporta
  con esa salvedad.

NUNCA toca artefactos de SPY: hmm.pkl, garch_SPY.pkl, strata_thresholds.json, calibration_summary.json.

Pre-registro: BITACORA.md [2026-06-14] [Pre-registro] Réplica NVDA.
Uso: ``python experiments/recalibrate_nvda.py``
"""

from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR
from core import data, features
from core.garch import GARCHModel
from core.hmm import RegimeHMM

TICKER = "NVDA"
HELDOUT_CUT = "2020-01-01"          # held-out LL: ajusta antes, evalúa después (K=2 vs K=3)
HMM_OUT = CACHE_MODELS_DIR / "hmm_nvda.pkl"
SUMMARY_OUT = CACHE_MODELS_DIR / "calibration_summary_nvda.json"

# Artefactos de SPY que este script tiene PROHIBIDO escribir (red de seguridad).
_SPY_LOCKED = {"hmm.pkl", "garch_SPY.pkl", "strata_thresholds.json", "calibration_summary.json"}


def _heldout_ll(X: np.ndarray, idx: pd.DatetimeIndex, cut: pd.Timestamp, k: int) -> tuple[float, int]:
    """LL/obs fuera de muestra de un HMM K-estados: ajusta en idx<cut, puntúa en idx>=cut."""
    n_fit = int((idx < cut).sum())
    eval_pos = np.where(idx >= cut)[0]
    h = RegimeHMM(n_states=k, seed=config.SEED).fit(X[:n_fit])
    ll = float(h.model.score(h._standardize(X[eval_pos])) / len(eval_pos))
    return ll, int(len(eval_pos))


def main() -> None:
    assert HMM_OUT.name not in _SPY_LOCKED and SUMMARY_OUT.name not in _SPY_LOCKED

    # --- Datos de NVDA: calibración 2000 → 2024-09 (ANTERIOR a todo OOS → sin look-ahead) ---
    end = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, end)
    ret = features.log_returns(prices["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    calib = feat.loc[feat.index <= pd.Timestamp(CALIBRATION_END)]
    X, idx = calib.to_numpy(), calib.index
    print(f"NVDA · feature = (log_return, realized_vol_21d) · calibración {CALIBRATION_START} → "
          f"{CALIBRATION_END}: n_obs = {len(calib)}")

    # --- HMM K=3 sobre NVDA (réplica del de SPY, pero con datos propios) ---
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(X)
    print(f"HMM-NVDA: best_seed={hmm.best_seed}, logL={hmm.best_score:.2f}, "
          f"transmat diag={np.round(np.diag(hmm.transition_matrix), 4)}, labels={hmm.state_labels}")

    # --- Prior RAM (régimen→signo): media de retorno por estado en calibración ---
    # El código de RAM mapea Calma(0)→long y Crisis(2)→short. Aquí medimos si NVDA respeta ese
    # mapeo: leverage effect ⇒ μ_Crisis < μ_Calma (alta vol coincide con caídas). Si no, prior-flip.
    states = hmm.predict_states(X)
    rfit = calib["r"].to_numpy()
    mu_state = {int(s): (float(rfit[states == s].mean()) if (states == s).any() else float("nan"))
                for s in range(3)}
    leverage_holds = bool(mu_state[2] < mu_state[0])          # Crisis con menos drift que Calma
    prior_calma_long = bool(mu_state[0] > 0)                  # Calma con drift positivo → long coherente
    prior_crisis_short = bool(mu_state[2] < 0)               # Crisis con drift negativo → short coherente
    print(f"μ_estado (Calma/Estrés/Crisis) = "
          f"{[round(mu_state[s], 6) for s in range(3)]} · leverage_holds={leverage_holds}")

    # --- Selección de K: held-out LL K=2 vs K=3 sobre NVDA ---
    cut = pd.Timestamp(HELDOUT_CUT)
    ll_k2, n_eval = _heldout_ll(X, idx, cut, 2)
    ll_k3, _ = _heldout_ll(X, idx, cut, 3)
    print(f"Held-out LL/obs (corte {HELDOUT_CUT}, n_eval={n_eval}): K2={ll_k2:.4f}  K3={ll_k3:.4f}  "
          f"ΔK3-K2={ll_k3 - ll_k2:+.4f} → K elegido = {3 if ll_k3 > ll_k2 else 2}")

    # --- GARCH-NVDA: verificar reproducibilidad del pickle existente (NO se sobrescribe) ---
    calib_ret = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
    g_new = GARCHModel().fit(calib_ret)
    g_old = pickle.load(open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "rb"))
    p_new, p_old = g_new.params, g_old.params
    same = all(abs(getattr(p_new, k) - getattr(p_old, k)) < 1e-4 for k in ("omega", "alpha", "beta", "nu"))
    print(f"GARCH-NVDA α+β={p_old.alpha + p_old.beta:.4f}, ν={p_old.nu:.2f}, estacionario={p_old.is_stationary()} "
          f"· reproducible={same} (no se sobrescribe el pickle cacheado)")

    # --- Escritura: solo artefactos NVDA ---
    pickle.dump(hmm, open(HMM_OUT, "wb"))
    summary = {
        "ticker": TICKER,
        "calibration_window": [CALIBRATION_START, CALIBRATION_END],
        "n_obs": int(len(calib)),
        "seed": config.SEED,
        "hmm": {
            "n_states": 3,
            "best_seed": int(hmm.best_seed),
            "logL": float(hmm.best_score),
            "transition_matrix": hmm.transition_matrix.tolist(),
            "state_labels": {str(k): v for k, v in hmm.state_labels.items()},
            "feature": "(log_return, realized_vol_21d_annualized)",
        },
        "prior_ram": {
            "mu_state": {str(s): mu_state[s] for s in range(3)},
            "leverage_effect_holds": leverage_holds,
            "prior_calma_long": prior_calma_long,
            "prior_crisis_short": prior_crisis_short,
            "nota": ("Régimen→signo: Calma→long, Crisis→short. leverage_effect_holds = μ_Crisis<μ_Calma. "
                     "Validez OOS del prior se contrasta con la regla prior-flip en walkforward_robustez."),
        },
        "k_selection": {
            "metodo": "held-out LL/obs, ajuste en idx<cut, evaluación en idx>=cut",
            "cut": HELDOUT_CUT, "n_eval": n_eval,
            "heldout_LL_K2": round(ll_k2, 4), "heldout_LL_K3": round(ll_k3, 4),
            "delta_LL_K3_menos_K2": round(ll_k3 - ll_k2, 4),
            "k_elegido": 3 if ll_k3 > ll_k2 else 2,
        },
        "garch": {
            "omega": p_old.omega, "alpha": p_old.alpha, "beta": p_old.beta, "nu": p_old.nu,
            "alpha_mas_beta": p_old.alpha + p_old.beta, "estacionario": p_old.is_stationary(),
            "reproducible_desde_pipeline": same,
        },
        "psa_gso_thresholds": "reutiliza strata_thresholds.json global (solo modulan magnitud, E4); banda GSO propia vía garch_NVDA",
        "_nota": "Recalibración NVDA réplica de SPY. NO sobrescribe artefactos de SPY.",
    }
    json.dump(summary, open(SUMMARY_OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nGuardado: {HMM_OUT.name}, {SUMMARY_OUT.name} (artefactos de SPY intactos).")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
