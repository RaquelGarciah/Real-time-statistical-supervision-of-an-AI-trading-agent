"""AutoML como M10: ¿una búsqueda automática de modelos bate a M10-XGBoost (y a ZeroR)?

M10 fija un único estimador (XGBoost) por pre-registro. Aquí se relaja esa restricción y se deja que
H2O AutoML busque sobre familias (GLM, GBM, RF, DeepLearning, XGBoost, StackedEnsembles) en EXACTAMENTE
el mismo pipeline causal que M10: mismas ALL22 features, mismo target signo(r_{t+1}), walk-forward
expandible con embargo, validación interna por Purged K-Fold (López de Prado 2018, sec. 7.4).

Encaje en la hipótesis del TFG (§2 nivel 3, universalidad): se ESPERA que AutoML NO bata significativamente
a M10/M8 y que NO supere a ZeroR causal. Si lo hiciera, hay que descontar la búsqueda (multiple testing):
AutoML prueba decenas de modelos, así que un p crudo favorable es sospechoso por construcción.

H0 (pre-registro). acc(AutoML) <= acc(ZeroR) y McNemar(AutoML vs M5/M8/M10) con p >= 0.10.
Criterio de éxito (refutaría la universalidad). acc(AutoML) > acc(ZeroR) Y McNemar vs M5 p<0.10 con rescate.
Criterio de fracaso (confirma universalidad). AutoML no supera a ZeroR ni a M10 → AutoML redescubre, no bate.

Uso: python experiments/automl_m10.py [--ticker SPY] [--budget 45]
"""

from __future__ import annotations

import argparse
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
from core import h2o_automl as ha
import experiments.walkforward_robustez as wf

N0 = 150            # ventana inicial (idéntica a walkforward_m10_causal)
STEP = 21           # reentreno mensual
EMBARGO = 5         # idéntico a CPCV / M10-WF
N_SPLITS = 5        # Purged K-Fold interno de AutoML
ANN = np.sqrt(252)
B_BOOT = 2000
OUT = Path("outputs/experiments/automl_m10.json")

FULL_COLS = ([f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
             + ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"])


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def automl_wf_p1(X: pd.DataFrame, y: pd.Series, budget: int) -> tuple[pd.Series, list[dict]]:
    """p1 causal con H2O AutoML: en cada paso entrena con [0:start-EMBARGO] y predice [start:start+STEP].

    Cada reentreno relanza la búsqueda AutoML con folds Purged K-Fold internos sobre el tramo de pasado.
    Devuelve (p1, leaders) donde leaders registra el modelo ganador de cada reentreno (para auditar
    qué familia gana y descontar multiple testing).
    """
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    leaders: list[dict] = []
    n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        leader, res = ha.train_h2o(Xtr, ytr, use_fold_column=True,
                                   max_runtime_secs=budget, n_splits=N_SPLITS,
                                   embargo=EMBARGO, seed=config.SEED)
        end = min(start + STEP, n)
        p1.iloc[start:end] = ha.predict_class1_proba(leader, X.iloc[start:end])
        leaders.append({"retrain_start": str(X.index[start].date()),
                        "leader_id": res.leader_id, "metric": res.leader_metric,
                        "family": res.leader_id.split("_")[0]})
    return p1, leaders


def xgb_wf_p1(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """p1 causal de M10-XGBoost (mismo esquema WF), para comparar en el MISMO tramo de test."""
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
    """Bootstrap estacionario PAREADO de la mediana ΔSharpe(a−b) (Politis-Romano 1994)."""
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


def run_ticker(ticker: str, budget: int) -> dict:
    wf.TICKER = ticker
    wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = wf.build_market_states_oos()
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    p1_automl, leaders = automl_wf_p1(X, y, budget)
    p1_xgb = xgb_wf_p1(X, y)

    test_mask = p1_automl.notna()
    td = X.index[test_mask]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())

    # ZeroR causal: en cada fecha apuesta la clase mayoritaria del PASADO disponible (sin look-ahead).
    yb_full = (mv["r_next"] > 0).astype(int)
    zr_pred = np.empty(len(td))
    for i, t in enumerate(td):
        past = yb_full.loc[:t].iloc[:-1]              # estrictamente anterior a t
        maj = 1.0 if past.mean() >= 0.5 else -1.0 if len(past) else 1.0
        zr_pred[i] = maj

    pos = {
        "m5": np.sign(mv.loc[td, "agent_size"].to_numpy()),
        "m8": np.sign(mv.loc[td, "final_size"].to_numpy()),
        "m10_xgb": np.sign(p1_xgb.loc[td].to_numpy() - 0.5),
        "automl": np.sign(p1_automl.loc[td].to_numpy() - 0.5),
        "zeror": zr_pred,
        "bh": np.ones(len(td)),
    }
    correct = {k: (v == truth).astype(int) for k, v in pos.items()}

    def net_arm(w_td: np.ndarray, lag: int = 1) -> np.ndarray:
        w = pd.Series(0.0, index=m.index); w.loc[td] = w_td
        return run_backtest(oos_ret, w, signal_lag=lag)["net_return"].reindex(td).to_numpy()

    yt = y.loc[td]
    metrics = {}
    for k, v in pos.items():
        auc = None
        if k in ("automl", "m10_xgb") and yt.nunique() >= 2:
            p1k = (p1_automl if k == "automl" else p1_xgb).loc[td]
            auc = round(float(roc_auc_score(yt, p1k)), 4)
        metrics[k] = {"accuracy": round(float(correct[k].mean()), 4), "auc": auc,
                      "sharpe_causal": round(_sr(net_arm(v, 1)), 3),
                      "sharpe_sameday": round(_sr(net_arm(v, 0)), 3)}

    # McNemar pareado: mcnemar_test(a, b) → b=a✓&b✗, c=a✗&b✓. Aquí a=otro, b=automl → c=automl rescata.
    tests = {}
    for ref in ("m5", "m8", "m10_xgb", "zeror"):
        _, p, b, c = mcnemar_test(correct[ref], correct["automl"])
        tests[f"automl_vs_{ref}"] = {"p": round(float(p), 4), "b_ref_solo": int(b), "c_automl_solo": int(c)}
    k_s, n_s, p_s, ci_s = sign_test(correct["automl"])
    tests["automl_sign_vs_0.5"] = {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4),
                                   "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}
    dS = _paired_boot_dsharpe(net_arm(pos["automl"]), net_arm(pos["m5"]))

    # --- Veredicto pre-registrado ---
    acc = {k: metrics[k]["accuracy"] for k in metrics}
    bate_zeror = bool(acc["automl"] > acc["zeror"])
    bate_m10 = bool(acc["automl"] > acc["m10_xgb"] and tests["automl_vs_m10_xgb"]["p"] < 0.10)
    rescata_m5 = bool(acc["automl"] > acc["m5"] and tests["automl_vs_m5"]["p"] < 0.10
                      and tests["automl_vs_m5"]["c_automl_solo"] > tests["automl_vs_m5"]["b_ref_solo"])
    refuta_universalidad = bool(bate_zeror and (bate_m10 or rescata_m5))
    fams = sorted({l["family"] for l in leaders})

    return {
        "config": {"N0": N0, "step": STEP, "embargo": EMBARGO, "n_splits_interno": N_SPLITS,
                   "budget_secs_por_reentreno": budget, "n_test": int(test_mask.sum()),
                   "n_retrains": len(leaders),
                   "test_span": [str(td.min().date()), str(td.max().date())],
                   "hmm_file": wf._hmm_path(ticker).name},
        "metrics_test_span": metrics,
        "tests": tests,
        "delta_sharpe_automl_vs_m5": dS,
        "leaders_por_reentreno": leaders,
        "familias_ganadoras": fams,
        "verdict": {"automl_bate_zeror": bate_zeror, "automl_bate_m10": bate_m10,
                    "automl_rescata_m5": rescata_m5, "refuta_universalidad": refuta_universalidad,
                    "nota_multiple_testing": "AutoML prueba decenas de modelos; un p<0.10 crudo debe descontarse."},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--budget", type=int, default=45, help="segundos de búsqueda AutoML por reentreno")
    args = ap.parse_args()

    config.set_seeds(config.SEED)
    print(f"=== AutoML-M10 · {args.ticker} · budget={args.budget}s/reentreno ===")
    try:
        r = run_ticker(args.ticker, args.budget)
    finally:
        ha.shutdown_h2o()

    mt = r["metrics_test_span"]
    print(f"test_span={r['config']['test_span']} n_test={r['config']['n_test']} retrains={r['config']['n_retrains']}")
    print(f"  acc: M5={mt['m5']['accuracy']} M8={mt['m8']['accuracy']} M10-XGB={mt['m10_xgb']['accuracy']} "
          f"AutoML={mt['automl']['accuracy']} ZeroR={mt['zeror']['accuracy']} B&H={mt['bh']['accuracy']}")
    print(f"  AutoML vs ZeroR: {'BATE' if r['verdict']['automl_bate_zeror'] else 'no bate'} | "
          f"vs M10 p={r['tests']['automl_vs_m10_xgb']['p']} | vs M5 p={r['tests']['automl_vs_m5']['p']}")
    print(f"  familias ganadoras: {r['familias_ganadoras']}")
    print(f"  veredicto: refuta_universalidad={r['verdict']['refuta_universalidad']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "ticker": args.ticker,
                       "scheme": "expanding_anchored_monthly_retrain + H2O AutoML (Purged K-Fold interno)",
                       "encaje": "Universalidad §2 nivel 3: se espera que AutoML NO bata a M10/ZeroR (redescubre).",
                       "pre_registro": "docstring de este fichero; H0 y criterios explícitos."},
              "por_activo": {args.ticker: r}}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    loaded = json.loads(OUT.read_text())
    a = loaded["por_activo"][args.ticker]
    for key in ("config", "metrics_test_span", "tests", "delta_sharpe_automl_vs_m5", "verdict"):
        assert key in a, f"Falta {key}"
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
