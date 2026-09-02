"""M10 walk-forward CAUSAL: ¿es desplegable a diario, o CPCV lo halagaba viendo el futuro?

CPCV (López de Prado 2018; Decisión #10) es el estimador insesgado canónico, pero entrena con bloques
cronológicamente POSTERIORES al test → no simula producción. Aquí, como VALIDACIÓN ADICIONAL (no sustituye
CPCV), se reentrena M10 en **ventana expandible anclada con reentreno mensual**, SOLO con el pasado, y se
compara con M10-CPCV y M5 en el MISMO tramo de test. Responde a la exigencia del tutor ("lánzalo en distintos
periodos de inicio; es el target de mañana y la restricción de hoy").

Pre-registro: BITACORA.md [2026-06-15]. Uso: python experiments/walkforward_m10_causal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import config
from core.backtest import run_backtest
from core.stats import mcnemar_test, sign_test
import experiments.walkforward_robustez as wf

TICKERS = ["SPY", "NVDA"]
N0 = 150            # ventana inicial (coherente con B_WINDOW=150 ya pre-registrada)
STEP = 21           # reentreno mensual
EMBARGO = 5         # idéntico a CPCV
ANN = np.sqrt(252)
B_BOOT = 2000
OUT = Path("outputs/experiments/walkforward_m10_causal.json")

FULL_COLS = ([f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
             + ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"])


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def expanding_wf_p1(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """p1 causal: en cada paso mensual entrena con [0 : start-EMBARGO] y predice [start : start+STEP]."""
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        clf = xgb.XGBClassifier(**wf.PARAMS)
        clf.fit(X.iloc[:tr_end], y.iloc[:tr_end])
        end = min(start + STEP, n)
        p1.iloc[start:end] = clf.predict_proba(X.iloc[start:end])[:, 1]
    return p1


def _paired_boot_dsharpe(r_a: np.ndarray, r_b: np.ndarray) -> dict:
    """Bootstrap estacionario PAREADO de la mediana ΔSharpe(a−b) (mismo método que part_b_confirmatory)."""
    n = len(r_a); block = max(2, int(round(np.sqrt(n)))); p = 1.0 / block
    rng = np.random.default_rng(config.SEED); deltas = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        deltas[i] = _sr(r_a[idx]) - _sr(r_b[idx])
    return {"median": round(float(np.median(deltas)), 4),
            "ci95": [round(float(np.quantile(deltas, 0.025)), 4), round(float(np.quantile(deltas, 0.975)), 4)],
            "point": round(_sr(r_a) - _sr(r_b), 4), "n_obs": int(n)}


def run_ticker(ticker: str) -> dict:
    wf.TICKER = ticker
    wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = wf.build_market_states_oos()
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    p1_wf = expanding_wf_p1(X, y)
    p1_cpcv = wf.cpcv_oof(X, y)                      # CPCV-OOF sobre el MISMO X (referencia)

    test_mask = p1_wf.notna()                        # tramo [N0:fin] que el WF causal puede predecir
    td = X.index[test_mask]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())

    # Posiciones por brazo en el tramo de test.
    pos = {
        "m5": np.sign(mv.loc[td, "agent_size"].to_numpy()),
        "m8": np.sign(mv.loc[td, "final_size"].to_numpy()),
        "m10_cpcv": np.sign(p1_cpcv.loc[td].to_numpy() - 0.5),
        "m10_wf": np.sign(p1_wf.loc[td].to_numpy() - 0.5),
        "bh": np.ones(len(td)),
    }
    correct = {k: (v == truth).astype(int) for k, v in pos.items()}

    def sharpe_arm(w_td: np.ndarray, lag: int = 1) -> float:
        w = pd.Series(0.0, index=m.index); w.loc[td] = w_td
        nr = run_backtest(oos_ret, w, signal_lag=lag)["net_return"].reindex(td).to_numpy()
        return _sr(nr)

    def net_arm(w_td: np.ndarray, lag: int = 1) -> np.ndarray:
        w = pd.Series(0.0, index=m.index); w.loc[td] = w_td
        return run_backtest(oos_ret, w, signal_lag=lag)["net_return"].reindex(td).to_numpy()

    from sklearn.metrics import matthews_corrcoef

    def _mcc(pred: np.ndarray):
        nz = pred != 0                                   # solo días con apuesta direccional
        if nz.sum() < 2 or len(np.unique(truth[nz])) < 2 or len(np.unique(pred[nz])) < 2:
            return None                                  # MCC indefinido (clase única) → B&H, etc.
        return round(float(matthews_corrcoef(truth[nz], pred[nz])), 4)

    yt = y.loc[td]
    metrics = {}
    for k, v in pos.items():
        auc = None                                       # guarda: AUC indefinido si una sola clase en [N0:fin]
        if k in ("m10_wf", "m10_cpcv") and yt.nunique() >= 2:
            p1k = (p1_wf if k == "m10_wf" else p1_cpcv).loc[td]
            auc = round(float(roc_auc_score(yt, p1k)), 4)
        metrics[k] = {"accuracy": round(float(correct[k].mean()), 4), "auc": auc, "mcc": _mcc(v),
                      "sharpe_causal": round(sharpe_arm(v, 1), 3),
                      "sharpe_sameday": round(sharpe_arm(v, 0), 3)}

    # McNemar pareado: mcnemar_test(a, b) → b = a✓&b✗, c = a✗&b✓.
    _, p_wf_m5, b1, c1 = mcnemar_test(correct["m5"], correct["m10_wf"])      # c1 = WF✓ & M5✗ (rescate)
    _, p_wf_cpcv, b2, c2 = mcnemar_test(correct["m10_cpcv"], correct["m10_wf"])
    k_s, n_s, p_s, ci_s = sign_test(correct["m10_wf"])                        # WF vs azar

    dS = _paired_boot_dsharpe(net_arm(pos["m10_wf"]), net_arm(pos["m5"]))

    # --- Veredicto pre-registrado ---
    acc_wf, acc_cpcv, acc_m5 = metrics["m10_wf"]["accuracy"], metrics["m10_cpcv"]["accuracy"], metrics["m5"]["accuracy"]
    rescata_por = [t for t, p in (("mcnemar_vs_m5", p_wf_m5), ("sign_test_vs_azar", p_s)) if p < 0.10]
    rescata_causal = bool(acc_wf > acc_m5 and len(rescata_por) > 0 and c1 > b1)
    # No-rechazo de McNemar(WF vs CPCV) NO es equivalencia (logic_esential §8.1); con n_test~250 puede ser baja potencia.
    cpcv_no_halaga = bool(p_wf_cpcv >= 0.10)
    desplegable = bool(rescata_causal and cpcv_no_halaga)
    cpcv_halagaba = bool(acc_cpcv > 0.5 and (acc_wf <= 0.5 or acc_wf <= acc_m5))
    sanity_ok = bool(np.sign(metrics["m10_wf"]["sharpe_causal"]) == np.sign(metrics["m10_wf"]["sharpe_sameday"])
                     or abs(metrics["m10_wf"]["sharpe_causal"]) < 0.1)

    return {
        "config": {"N0": N0, "step": STEP, "embargo": EMBARGO, "n_test": int(test_mask.sum()),
                   "n_retrains": int(len(range(N0, len(X), STEP))),
                   "test_span": [str(td.min().date()), str(td.max().date())],
                   "hmm_file": wf._hmm_path(ticker).name},
        "metrics_test_span": metrics,
        "mcnemar_wf_vs_m5": {"p": float(p_wf_m5), "b_m5_solo": int(b1), "c_wf_solo": int(c1)},
        "mcnemar_wf_vs_cpcv": {"p": float(p_wf_cpcv), "b_cpcv_solo": int(b2), "c_wf_solo": int(c2)},
        "sign_test_wf": {"k": int(k_s), "n": int(n_s), "p": float(p_s), "ci95": [float(ci_s[0]), float(ci_s[1])]},
        "delta_sharpe_wf_vs_m5": dS,
        "verdict": {"rescata_causal": rescata_causal, "rescata_causal_por": rescata_por,
                    "cpcv_no_halaga_no_equiv": cpcv_no_halaga, "desplegable_diario": desplegable,
                    "cpcv_halagaba": cpcv_halagaba, "sanity_ok": sanity_ok},
    }


def main() -> None:
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "scheme": "expanding_anchored_monthly_retrain",
                       "nota": "Validación causal ADICIONAL de M10; no sustituye CPCV (Decisión #10).",
                       "no_fwer_cross_ticker": "SPY = caso central; NVDA = chequeo de consistencia → sin Holm entre activos (declarado).",
                       "m8_nvda_umbrales": "M8-NVDA usa umbrales PSA/GSO/RAM de SPY (no recalibrados); su Sharpe es ilustrativo.",
                       "potencia": "n_test~250, ~12 reentrenos; p≥0.10 en WF-vs-CPCV es no-rechazo, NO equivalencia.",
                       "pre_registro": "BITACORA 2026-06-15"},
              "por_activo": {}}
    for tk in TICKERS:
        print(f"\n=== {tk} ===")
        r = run_ticker(tk)
        result["por_activo"][tk] = r
        mt = r["metrics_test_span"]
        print(f"test_span={r['config']['test_span']} n_test={r['config']['n_test']} retrains={r['config']['n_retrains']}")
        print(f"  acc: M5={mt['m5']['accuracy']}  M8={mt['m8']['accuracy']}  "
              f"M10-CPCV={mt['m10_cpcv']['accuracy']}  M10-WF={mt['m10_wf']['accuracy']}  B&H={mt['bh']['accuracy']}")
        print(f"  McNemar WF vs M5 p={r['mcnemar_wf_vs_m5']['p']:.3f} (c_WF={r['mcnemar_wf_vs_m5']['c_wf_solo']} "
              f"b_M5={r['mcnemar_wf_vs_m5']['b_m5_solo']}) · WF vs CPCV p={r['mcnemar_wf_vs_cpcv']['p']:.3f}")
        print(f"  ΔSharpe(WF−M5) mediana={r['delta_sharpe_wf_vs_m5']['median']} IC95={r['delta_sharpe_wf_vs_m5']['ci95']}")
        print(f"  veredicto: {r['verdict']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    # Validación de claves prometidas en el pre-registro.
    loaded = json.loads(OUT.read_text())
    for tk in TICKERS:
        a = loaded["por_activo"][tk]
        for key in ("config", "metrics_test_span", "mcnemar_wf_vs_m5", "mcnemar_wf_vs_cpcv",
                    "delta_sharpe_wf_vs_m5", "verdict"):
            assert key in a, f"Falta {tk}.{key}"
        for arm in ("m5", "m8", "m10_cpcv", "m10_wf", "bh"):
            assert "accuracy" in a["metrics_test_span"][arm]
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
