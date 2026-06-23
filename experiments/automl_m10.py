"""AutoML por activo: cada activo encuentra su modelo óptimo. ¿Rescata al agente mejor que M10?

M10 fija UN estimador (XGBoost). Aquí se deja que H2O AutoML busque, por activo, sobre 6 familias
(GLM, GBM, Random Forest, Deep Learning, XGBoost, StackedEnsembles) en el MISMO pipeline causal que M10:
ALL22 features, target signo(r_{t+1}), walk-forward expandible con embargo, Purged K-Fold interno
(López de Prado 2018, sec. 7.4). HMM+GARCH se calibran al vuelo por activo (build_states_onthefly).

Enfoque de lectura. La pregunta NO es "¿bate AutoML a M10?" (mismo techo ZeroR), sino:
¿corrige AutoML al agente (M5) y a la regla STRATA (M8) tan bien o mejor que M10, y con mejor
Sharpe / drawdown? Por eso la tabla por activo incluye accuracy, AUC, Sharpe causal, equity, max DD y
Calmar para CADA estrategia (M5, M8, M10-XGB, AutoML, ZeroR, B&H), y McNemar de rescate de AutoML y de
M10 contra M5/M8.

H0 (universalidad §2 nivel 3). AutoML no bate a ZeroR causal en accuracy. Criterio de fracaso: si AutoML
batiera a ZeroR de forma robusta en varios activos, refutaría que XGBoost ya captura la señal.

Uso: python experiments/automl_m10.py [--panel SPY,NVDA,...] [--max-models 25]
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
from core import metrics
from core.backtest import run_backtest
from core.stats import mcnemar_test, sign_test
from core import h2o_automl as ha
import experiments.walkforward_robustez as wf
from falsacion.m10_configs.m10_v3_causal_panel import build_states_onthefly

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA",
         "QQQ", "DIA", "IWM", "XLF", "XLK"]
N0 = 150
STEP = 21
EMBARGO = 1          # walk-forward DESPLEGABLE: embargo=1 (horizonte=1), canónico M10 (CLAUDE.md §4, decisión #15)
N_SPLITS = 5
HOLDOUT_FRAC = None     # si >0: validación interna por holdout cronológico causal (no fold_column)
SORT_METRIC = "AUC"     # métrica de selección del leader (AUC | logloss | AUCPR)
INCLUDE_ALGOS = None    # acota familias H2O (p.ej. ["GBM","XGBoost","StackedEnsemble"])
ANN = np.sqrt(252)
B_BOOT = 2000
OUT = Path("outputs/experiments/automl_panel.json")

FULL_COLS = ([f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
             + ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"])
ARMS = ("m5", "m8", "m10_xgb", "automl", "zeror", "bh")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def automl_wf_p1(X: pd.DataFrame, y: pd.Series, max_models: int) -> tuple[pd.Series, list[dict]]:
    """p1 causal con H2O AutoML: cada paso entrena con [0:start-EMBARGO] (Purged K-Fold interno) y
    predice [start:start+STEP]. Registra el leader de cada reentreno (auditoría + multiple testing).

    Usa max_models (nº fijo de modelos, sin DeepLearning) → determinista dada la semilla. Con
    max_runtime_secs el resultado NO era reproducible (cambiaba entre corridas con la misma semilla)."""
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    leaders: list[dict] = []
    n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        leader, res = ha.train_h2o(X.iloc[:tr_end], y.iloc[:tr_end], use_fold_column=True,
                                   max_models=max_models, n_splits=N_SPLITS, embargo=EMBARGO, seed=config.SEED,
                                   holdout_frac=HOLDOUT_FRAC, sort_metric=SORT_METRIC, include_algos=INCLUDE_ALGOS)
        end = min(start + STEP, n)
        p1.iloc[start:end] = ha.predict_class1_proba(leader, X.iloc[start:end])
        leaders.append({"retrain_start": str(X.index[start].date()), "leader_id": res.leader_id,
                        "metric": res.leader_metric, "family": res.leader_id.split("_")[0]})
    return p1, leaders


M10_SEEDS = [config.SEED + i for i in range(10)]  # ensemble canónico de 10 semillas (m10_pivot_scan "ens")


def xgb_wf_p1(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """p1 causal del M10 CANÓNICO: ensemble de 10 XGBoost (semillas), WF expandible, embargo=1.

    Replica m10_pivot_scan config 'ens' (ALL22, 10 semillas, sin recency) = el M10 desplegable del TFG.
    Un solo XGBoost daba ~0.52 en SMCI; el ensemble + embargo=1 recupera el 0.552 canónico."""
    base = {k: v for k, v in wf.PARAMS.items() if k != "random_state"}
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**base, random_state=sd)
                 .fit(X.iloc[:tr_end], y.iloc[:tr_end])
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in M10_SEEDS]
        p1.iloc[start:end] = np.mean(preds, axis=0)
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
            "point": round(_sr(r_a) - _sr(r_b), 4)}


def run_ticker(ticker: str, max_models: int) -> dict:
    wf.TICKER = ticker
    wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    p1_automl, leaders = automl_wf_p1(X, y, max_models)
    p1_xgb = xgb_wf_p1(X, y)

    td = X.index[p1_automl.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())

    # ZeroR causal: clase mayoritaria del PASADO estricto en cada fecha (sin look-ahead).
    yb_full = (mv["r_next"] > 0).astype(int)
    zr = np.array([1.0 if (yb_full.loc[:t].iloc[:-1].mean() >= 0.5 if len(yb_full.loc[:t]) > 1 else True) else -1.0
                   for t in td])

    pos = {
        "m5": np.sign(mv.loc[td, "agent_size"].to_numpy()),
        "m8": np.sign(mv.loc[td, "final_size"].to_numpy()),
        "m10_xgb": np.sign(p1_xgb.loc[td].to_numpy() - 0.5),
        "automl": np.sign(p1_automl.loc[td].to_numpy() - 0.5),
        "zeror": zr,
        "bh": np.ones(len(td)),
    }
    correct = {k: (v == truth).astype(int) for k, v in pos.items()}

    def net_arm(w_td: np.ndarray, lag: int = 1) -> np.ndarray:
        w = pd.Series(0.0, index=m.index); w.loc[td] = w_td
        return run_backtest(oos_ret, w, signal_lag=lag)["net_return"].reindex(td).to_numpy()

    yt = y.loc[td]
    table = {}
    for k, v in pos.items():
        nr = pd.Series(net_arm(v, 1), index=td).dropna()
        eq = (1.0 + nr).cumprod()
        auc = None
        if k in ("automl", "m10_xgb") and yt.nunique() >= 2:
            auc = round(float(roc_auc_score(yt, (p1_automl if k == "automl" else p1_xgb).loc[td])), 4)
        table[k] = {"accuracy": round(float(correct[k].mean()), 4), "auc": auc,
                    "sharpe": round(_sr(nr.to_numpy()), 3),
                    "equity_final": round(float(eq.iloc[-1]), 4),
                    "max_dd": round(metrics.max_drawdown(eq), 4),
                    "calmar": round(metrics.calmar(nr), 3)}

    # McNemar de rescate: mcnemar_test(base, sup) → c = sup✓ & base✗ (rescate del supervisor).
    def mc(base: str, sup: str) -> dict:
        _, p, b, c = mcnemar_test(correct[base], correct[sup])
        return {"p": round(float(p), 4), "b_base_solo": int(b), "c_sup_solo": int(c)}

    # Matriz completa: cada no-trivial vs M5 (¿rescata al agente?) y vs ZeroR (¿bate el techo?).
    tests = {}
    for arm in ("m8", "m10_xgb", "automl"):
        tests[f"{arm}_vs_m5"] = mc("m5", arm)
    for arm in ("m5", "m8", "m10_xgb", "automl"):
        tests[f"{arm}_vs_zeror"] = mc("zeror", arm)
    tests["automl_vs_m8"] = mc("m8", "automl")
    tests["automl_vs_m10"] = mc("m10_xgb", "automl")
    # Sign test vs 0.5 (¿bate al azar direccional?) por brazo.
    for arm in ARMS:
        k_a, n_a, p_a, ci_a = sign_test(correct[arm])
        tests[f"{arm}_sign_vs_0.5"] = {"k": int(k_a), "n": int(n_a), "p": round(float(p_a), 4),
                                       "ci95": [round(float(ci_a[0]), 4), round(float(ci_a[1]), 4)]}

    # Matriz McNemar COMPLETA todos-contra-todos (15 pares) + vectores de acierto para tests post-hoc.
    mcnemar_matrix = {}
    for i, a in enumerate(ARMS):
        for bn in ARMS[i + 1:]:
            _, p, n_a, n_b = mcnemar_test(correct[a], correct[bn])
            mcnemar_matrix[f"{a}__{bn}"] = {"p": round(float(p), 4),
                                            f"{a}_solo": int(n_a), f"{bn}_solo": int(n_b)}
    correct_by_arm = {k: [int(x) for x in correct[k]] for k in ARMS}

    # --- OOS COMPLETO para no-learners (M5/M8/ZeroR/B&H no necesitan burn-in) ---
    idxf = mv.index
    truth_f = np.sign(mv["r_next"].to_numpy())
    ybf = (mv["r_next"] > 0).astype(int).to_numpy()
    zrf = np.where(pd.Series(ybf).expanding().mean().shift().fillna(0.5).to_numpy() >= 0.5, 1.0, -1.0)
    pos_f = {"m5": np.sign(mv["agent_size"].to_numpy()), "m8": np.sign(mv["final_size"].to_numpy()),
             "zeror": zrf, "bh": np.ones(len(mv))}
    correct_f = {k: (v == truth_f).astype(int) for k, v in pos_f.items()}

    def net_f(w_td: np.ndarray) -> np.ndarray:
        w = pd.Series(0.0, index=m.index); w.loc[idxf] = w_td
        return run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idxf).to_numpy()

    table_f, tests_f = {}, {}
    for k, v in pos_f.items():
        nrf = pd.Series(net_f(v), index=idxf).dropna(); eqf = (1.0 + nrf).cumprod()
        table_f[k] = {"accuracy": round(float(correct_f[k].mean()), 4),
                      "sharpe": round(_sr(nrf.to_numpy()), 3), "equity_final": round(float(eqf.iloc[-1]), 4),
                      "max_dd": round(metrics.max_drawdown(eqf), 4), "calmar": round(metrics.calmar(nrf), 3)}
    for arm in ("m5", "m8"):
        _, pf, bf, cf = mcnemar_test(correct_f["zeror"], correct_f[arm])
        tests_f[f"{arm}_vs_zeror"] = {"p": round(float(pf), 4), "zeror_solo": int(bf), f"{arm}_solo": int(cf)}
    _, pf, bf, cf = mcnemar_test(correct_f["m5"], correct_f["m8"])
    tests_f["m8_vs_m5"] = {"p": round(float(pf), 4), "m5_solo": int(bf), "m8_solo": int(cf)}
    for arm in ("m5", "m8", "zeror", "bh"):
        k_a, n_a, p_a, ci_a = sign_test(correct_f[arm])
        tests_f[f"{arm}_sign_vs_0.5"] = {"k": int(k_a), "n": int(n_a), "p": round(float(p_a), 4),
                                         "ci95": [round(float(ci_a[0]), 4), round(float(ci_a[1]), 4)]}
    full_oos = {"n": len(mv), "test_span": [str(idxf.min().date()), str(idxf.max().date())],
                "table": table_f, "tests": tests_f}

    # Serie diaria de retornos netos por brazo (incl. AutoML) → equity y bootstrap de riesgo en el notebook.
    def _jsafe(a) -> list:
        return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 6) for x in a]
    net_returns = {"dates": [str(d.date()) for d in td], **{k: _jsafe(net_arm(pos[k])) for k in pos}}

    dS = _paired_boot_dsharpe(net_arm(pos["automl"]), net_arm(pos["m5"]))
    acc = {k: table[k]["accuracy"] for k in table}
    fams = sorted({l["family"] for l in leaders})
    verdict = {
        "automl_bate_zeror": bool(acc["automl"] > acc["zeror"] and tests["automl_vs_zeror"]["p"] < 0.10),
        "automl_rescata_m5": bool(acc["automl"] > acc["m5"] and tests["automl_vs_m5"]["p"] < 0.10),
        "automl_rescata_m8": bool(acc["automl"] > acc["m8"] and tests["automl_vs_m8"]["p"] < 0.10),
        "automl_mejor_sharpe_que_m5": bool(table["automl"]["sharpe"] > table["m5"]["sharpe"]),
        "automl_mejor_dd_que_m5": bool(table["automl"]["max_dd"] > table["m5"]["max_dd"]),
    }
    return {
        "config": {"N0": N0, "step": STEP, "embargo": EMBARGO, "max_models": max_models,
                   "n_test": len(td), "n_retrains": len(leaders),
                   "test_span": [str(td.min().date()), str(td.max().date())]},
        "table": table, "tests": tests, "mcnemar_matrix": mcnemar_matrix,
        "full_oos_no_learners": full_oos,
        "correct_by_arm": correct_by_arm, "delta_sharpe_automl_vs_m5": dS,
        "net_returns": net_returns,
        "familias_ganadoras": fams, "leaders_por_reentreno": leaders, "verdict": verdict,
    }


def _print_table(tk: str, r: dict) -> None:
    t = r["table"]
    print(f"\n=== {tk} === test_span={r['config']['test_span']} n={r['config']['n_test']} "
          f"retrains={r['config']['n_retrains']} familias={r['familias_ganadoras']}")
    print(f"{'arm':9} {'acc':>6} {'auc':>6} {'sharpe':>7} {'equity':>7} {'maxDD':>7} {'calmar':>7}")
    for k in ARMS:
        a = t[k]
        print(f"{k:9} {a['accuracy']:>6} {str(a['auc']):>6} {a['sharpe']:>7} "
              f"{a['equity_final']:>7} {a['max_dd']:>7} {a['calmar']:>7}")
    tq = r["tests"]
    print("  McNemar p · vs ZeroR (¿bate el techo?) | vs M5 (¿rescata al agente?):")
    print(f"    M8   zeror p={tq['m8_vs_zeror']['p']} · m5 p={tq['m8_vs_m5']['p']}")
    print(f"    M10  zeror p={tq['m10_xgb_vs_zeror']['p']} · m5 p={tq['m10_xgb_vs_m5']['p']}")
    print(f"    AutoML zeror p={tq['automl_vs_zeror']['p']} · m5 p={tq['automl_vs_m5']['p']} · "
          f"vs M8 p={tq['automl_vs_m8']['p']} · vs M10 p={tq['automl_vs_m10']['p']}")
    print(f"  sign vs 0.5: AutoML p={tq['automl_sign_vs_0.5']['p']} ({tq['automl_sign_vs_0.5']['k']}/{tq['automl_sign_vs_0.5']['n']})")
    fo = r["full_oos_no_learners"]; ft = fo["table"]; ftq = fo["tests"]
    print(f"  -- OOS COMPLETO no-learners (n={fo['n']}, {fo['test_span']}) --")
    for k in ("m5", "m8", "zeror", "bh"):
        a = ft[k]
        print(f"    {k:6} acc={a['accuracy']} sharpe={a['sharpe']} maxDD={a['max_dd']} calmar={a['calmar']}")
    print(f"    sig: M8 vs ZeroR p={ftq['m8_vs_zeror']['p']} · M8 vs M5 p={ftq['m8_vs_m5']['p']} · "
          f"M8 sign p={ftq['m8_sign_vs_0.5']['p']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=",".join(PANEL))
    ap.add_argument("--max-models", type=int, default=25)
    ap.add_argument("--n0", type=int, default=150, help="burn-in WF; bajar para acercarse al OOS completo")
    ap.add_argument("--embargo", type=int, default=1, help="embargo WF desplegable (canónico M10 = 1)")
    ap.add_argument("--out", default=str(OUT), help="JSON de salida (workers paralelos → ficheros distintos)")
    ap.add_argument("--holdout-frac", type=float, default=0.0, help=">0 → holdout cronológico causal (Opción 1)")
    ap.add_argument("--sort-metric", default="AUC", help="AUC | logloss | AUCPR")
    ap.add_argument("--include-algos", default="", help="coma-sep, p.ej. GBM,XGBoost,StackedEnsemble")
    args = ap.parse_args()
    tickers = [t.strip() for t in args.panel.split(",") if t.strip()]
    out = Path(args.out)
    global N0, EMBARGO, HOLDOUT_FRAC, SORT_METRIC, INCLUDE_ALGOS
    N0 = args.n0
    EMBARGO = args.embargo
    HOLDOUT_FRAC = args.holdout_frac if args.holdout_frac > 0 else None
    SORT_METRIC = args.sort_metric
    INCLUDE_ALGOS = [a.strip() for a in args.include_algos.split(",") if a.strip()] or None

    config.set_seeds(config.SEED)
    meta = {"seed": config.SEED, "signal_lag": 1, "panel": tickers, "max_models": args.max_models,
            "embargo": EMBARGO, "n0": N0, "step": STEP, "sort_metric": SORT_METRIC,
            "include_algos": INCLUDE_ALGOS, "holdout_frac": HOLDOUT_FRAC,
            "scheme": "AutoML (H2O, Purged K-Fold, max_models DETERMINISTA) WF por activo",
            "states": "HMM+GARCH calibrados al vuelo por activo (build_states_onthefly)",
            "reproducibilidad": "max_models fijo + semilla → determinista; max_runtime_secs NO lo era",
            "pre_registro": "docstring de este fichero"}
    result = {"meta": meta, "por_activo": {}}
    # Resume: si el JSON de salida ya existe, conserva los activos ya hechos (no-error) y los salta.
    if out.exists():
        prev = json.loads(out.read_text()).get("por_activo", {})
        for tk, v in prev.items():
            if "error" not in v:
                result["por_activo"][tk] = v
        if result["por_activo"]:
            print(f"RESUME: {len(result['por_activo'])} activos ya hechos, se saltan: {list(result['por_activo'])}")
    try:
        for tk in tickers:
            if tk in result["por_activo"]:
                continue                                   # ya calculado (resume)
            print(f"\n########## {tk} (max_models={args.max_models}) ##########")
            try:
                r = run_ticker(tk, args.max_models)
                result["por_activo"][tk] = r
                _print_table(tk, r)
            except Exception as e:                       # un activo que falle no tumba el panel
                print(f"  !! {tk} FALLÓ: {type(e).__name__}: {e}")
                result["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False))  # guarda incremental
    finally:
        ha.shutdown_h2o()
    print(f"\nOK · {out} · {len([k for k,v in result['por_activo'].items() if 'error' not in v])} activos")


if __name__ == "__main__":
    main()
