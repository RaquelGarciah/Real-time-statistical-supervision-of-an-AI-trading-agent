"""AutoML-M10 como ENSEMBLE de semillas + análisis de sensibilidad a la semilla (honesto).

Motivación. La semilla de H2O AutoML es ruido, no un hiperparámetro del modelo. Elegir la semilla
que maximiza el OOS sería p-hacking (overfitting del test). En su lugar, este script:

  1. ENSEMBLE: fija 10 semillas A PRIORI (SEED+0..9 = 42..51, idénticas al ensemble canónico de
     M10-XGBoost en automl_m10.xgb_wf_p1) y promedia las probabilidades p1 entre semillas. El
     ensemble reduce varianza; NO maximiza el OOS. Es la versión defendible de "variar la semilla".
  2. SENSIBILIDAD: reporta por activo la distribución de accuracy entre las 10 semillas
     (mín/mediana/máx/std). Documenta la fragilidad del método ante la semilla — lo que un tribunal
     espera ver, no lo que se esconde.

Reusa toda la mecánica de experiments/automl_m10.py (mismo pipeline causal, ALL22 features, target
signo(r_{t+1}), WF expandible embargo=1, Purged K-Fold interno, build_states_onthefly). Mismas
métricas por estrategia (accuracy, AUC, Sharpe causal, equity, maxDD, Calmar) y misma batería de
tests (McNemar vs ZeroR/M5/M8/M10, sign test).

H0 (universalidad §2 nivel 3, SIN cambios). El ensemble de AutoML NO bate a ZeroR causal en accuracy.
Criterio de fracaso preservado: si el ensemble batiera a ZeroR de forma robusta refutaría que la
señal direccional ya está agotada.

Uso: python experiments/automl_seed_ensemble.py [--panel SPY,NVDA,...] [--n-seeds 10] [--max-models 25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from core import metrics
from core.backtest import run_backtest
from core.stats import mcnemar_test, sign_test
from core import h2o_automl as ha
import experiments.walkforward_robustez as wf
from falsacion.m10_configs.m10_v3_causal_panel import build_states_onthefly

import experiments.automl_m10 as am   # reusa FULL_COLS, ARMS, _sr, _paired_boot_dsharpe, xgb_wf_p1, N0...

OUT = Path("outputs/experiments/automl_seed_ensemble.json")


def _reset_h2o() -> None:
    """Fuerza un re-arranque limpio del clúster H2O. Espera a que la JVM vieja muera antes de devolver,
    para no solapar dos JVMs (cada una reserva el heap completo → pico de memoria y posible OOM)."""
    import time
    import h2o
    try:
        h2o.cluster().shutdown(prompt=False)
        time.sleep(4)                              # deja que la JVM libere el heap antes de re-init
    except Exception:
        pass
    ha._H2O_INIT = False


def _train_predict_resilient(X_tr, y_tr, X_oos, max_models, seed, n_try: int = 4):
    """train_h2o + predict con reintentos sobre clúster fresco ante CUALQUIER error de H2O.

    Dos fallos observados, ambos transitorios sobre el clúster persistente:
      - 'Local server has died. RIP' (H2OConnectionError): la JVM muere.
      - 'Job is missing' (H2OResponseError): carrera en el polling asíncrono de H2OAutoML; H2O
        recolecta el job antes de que el cliente lo consulte. Aparece de forma intermitente.
    Ante cualquiera, se reinicia el clúster (estado limpio) y se reintenta hasta n_try veces.
    NO se hace h2o.remove_all() (borra el registro de Jobs y provoca el 'Job is missing')."""
    from h2o.exceptions import H2OError
    for attempt in range(1, n_try + 1):
        try:
            leader, res = ha.train_h2o(X_tr, y_tr, use_fold_column=True, max_models=max_models,
                                       n_splits=am.N_SPLITS, embargo=am.EMBARGO, seed=seed)
            proba = ha.predict_class1_proba(leader, X_oos)
            return proba, res.leader_id.split("_")[0]
        except H2OError:
            if attempt == n_try:
                raise
            _reset_h2o()                           # clúster fresco y reintento


def automl_wf_p1_seeded(X: pd.DataFrame, y: pd.Series, max_models: int, seed: int) -> tuple[pd.Series, list[str]]:
    """p1 causal de AutoML para UNA semilla. Idéntico a am.automl_wf_p1 pero parametrizando la semilla
    (que allí está fija en config.SEED) y con limpieza de memoria H2O entre reentrenos. Devuelve
    (p1, familias_ganadoras)."""
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    fams: set[str] = set()
    n = len(X)
    for start in range(am.N0, n, am.STEP):
        tr_end = start - am.EMBARGO
        if tr_end < 50:
            continue
        end = min(start + am.STEP, n)
        proba, fam = _train_predict_resilient(X.iloc[:tr_end], y.iloc[:tr_end], X.iloc[start:end],
                                              max_models, seed)
        p1.iloc[start:end] = proba
        fams.add(fam)
    return p1, sorted(fams)


def run_ticker(ticker: str, seeds: list[int], max_models: int) -> dict:
    _reset_h2o()          # clúster fresco por activo: la JVM de larga vida se degradaba (4.7→0.8 retr/min)
    wf.TICKER = ticker
    wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[am.FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    # --- AutoML por semilla -> ensemble (promedio de probabilidades) ---
    # Clúster persistente (como el baseline, que corrió 180 reentrenos sin problema). Los fallos
    # transitorios de H2O ('Job is missing', 'server died') los absorbe _train_predict_resilient.
    p1_by_seed: dict[int, pd.Series] = {}
    fams_by_seed: dict[int, list[str]] = {}
    for sd in seeds:
        p1_s, fams_s = automl_wf_p1_seeded(X, y, max_models, sd)
        p1_by_seed[sd] = p1_s
        fams_by_seed[sd] = fams_s
    p1_automl = pd.concat([p1_by_seed[sd] for sd in seeds], axis=1).mean(axis=1)
    p1_xgb = am.xgb_wf_p1(X, y)

    td = X.index[p1_automl.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())

    # accuracy por semilla (sensibilidad) sobre la MISMA ventana td
    acc_by_seed = {}
    for sd in seeds:
        pos_s = np.sign(p1_by_seed[sd].loc[td].to_numpy() - 0.5)
        acc_by_seed[sd] = float((pos_s == truth).mean())
    accs = np.array(list(acc_by_seed.values()))

    # ZeroR causal: clase mayoritaria del pasado estricto (sin look-ahead).
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
                    "sharpe": round(am._sr(nr.to_numpy()), 3),
                    "equity_final": round(float(eq.iloc[-1]), 4),
                    "max_dd": round(metrics.max_drawdown(eq), 4),
                    "calmar": round(metrics.calmar(nr), 3)}

    def mc(base: str, sup: str) -> dict:
        _, p, b, c = mcnemar_test(correct[base], correct[sup])
        return {"p": round(float(p), 4), "b_base_solo": int(b), "c_sup_solo": int(c)}

    tests = {}
    for arm in ("m8", "m10_xgb", "automl"):
        tests[f"{arm}_vs_m5"] = mc("m5", arm)
    for arm in ("m5", "m8", "m10_xgb", "automl"):
        tests[f"{arm}_vs_zeror"] = mc("zeror", arm)
    tests["automl_vs_m8"] = mc("m8", "automl")
    tests["automl_vs_m10"] = mc("m10_xgb", "automl")
    for arm in am.ARMS:
        k_a, n_a, p_a, ci_a = sign_test(correct[arm])
        tests[f"{arm}_sign_vs_0.5"] = {"k": int(k_a), "n": int(n_a), "p": round(float(p_a), 4),
                                       "ci95": [round(float(ci_a[0]), 4), round(float(ci_a[1]), 4)]}

    mcnemar_matrix = {}
    for i, a in enumerate(am.ARMS):
        for bn in am.ARMS[i + 1:]:
            _, p, n_a, n_b = mcnemar_test(correct[a], correct[bn])
            mcnemar_matrix[f"{a}__{bn}"] = {"p": round(float(p), 4),
                                            f"{a}_solo": int(n_a), f"{bn}_solo": int(n_b)}
    correct_by_arm = {k: [int(x) for x in correct[k]] for k in am.ARMS}

    dS = am._paired_boot_dsharpe(net_arm(pos["automl"]), net_arm(pos["m5"]))
    acc = {k: table[k]["accuracy"] for k in table}
    fams_union = sorted({f for sd in seeds for f in fams_by_seed[sd]})
    verdict = {
        "ensemble_bate_zeror": bool(acc["automl"] > acc["zeror"] and tests["automl_vs_zeror"]["p"] < 0.10),
        "ensemble_rescata_m5": bool(acc["automl"] > acc["m5"] and tests["automl_vs_m5"]["p"] < 0.10),
        "ensemble_rescata_m8": bool(acc["automl"] > acc["m8"] and tests["automl_vs_m8"]["p"] < 0.10),
        "ensemble_mejor_sharpe_que_m5": bool(table["automl"]["sharpe"] > table["m5"]["sharpe"]),
        "ensemble_mejor_dd_que_m5": bool(table["automl"]["max_dd"] > table["m5"]["max_dd"]),
    }
    seed_sens = {
        "seeds": seeds,
        "automl_acc_by_seed": {str(sd): round(acc_by_seed[sd], 4) for sd in seeds},
        "acc_min": round(float(accs.min()), 4), "acc_median": round(float(np.median(accs)), 4),
        "acc_max": round(float(accs.max()), 4), "acc_std": round(float(accs.std(ddof=1)), 4),
        "ensemble_acc": acc["automl"],
        "zeror_acc": acc["zeror"],
        "n_seeds_beat_zeror": int((accs > acc["zeror"]).sum()),
    }
    return {
        "config": {"N0": am.N0, "step": am.STEP, "embargo": am.EMBARGO, "max_models": max_models,
                   "n_test": len(td), "n_seeds": len(seeds),
                   "test_span": [str(td.min().date()), str(td.max().date())]},
        "table": table, "tests": tests, "mcnemar_matrix": mcnemar_matrix,
        "correct_by_arm": correct_by_arm, "delta_sharpe_automl_vs_m5": dS,
        "familias_ganadoras": fams_union, "seed_sensibilidad": seed_sens, "verdict": verdict,
    }


def _print_table(tk: str, r: dict) -> None:
    t = r["table"]; s = r["seed_sensibilidad"]
    print(f"\n=== {tk} === test_span={r['config']['test_span']} n={r['config']['n_test']} "
          f"seeds={r['config']['n_seeds']} familias={r['familias_ganadoras']}")
    print(f"{'arm':9} {'acc':>6} {'auc':>6} {'sharpe':>7} {'equity':>7} {'maxDD':>7} {'calmar':>7}")
    for k in am.ARMS:
        a = t[k]
        print(f"{k:9} {a['accuracy']:>6} {str(a['auc']):>6} {a['sharpe']:>7} "
              f"{a['equity_final']:>7} {a['max_dd']:>7} {a['calmar']:>7}")
    print(f"  seed-sens AutoML: ens={s['ensemble_acc']} | min={s['acc_min']} med={s['acc_median']} "
          f"max={s['acc_max']} std={s['acc_std']} | ZeroR={s['zeror_acc']} | "
          f"semillas>ZeroR={s['n_seeds_beat_zeror']}/{r['config']['n_seeds']}")
    tq = r["tests"]
    print(f"  McNemar p: AutoML vs ZeroR={tq['automl_vs_zeror']['p']} · vs M5={tq['automl_vs_m5']['p']} · "
          f"vs M8={tq['automl_vs_m8']['p']} · vs M10={tq['automl_vs_m10']['p']} · "
          f"sign p={tq['automl_sign_vs_0.5']['p']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=",".join(am.PANEL))
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--max-models", type=int, default=25)
    ap.add_argument("--out", default=str(OUT), help="ruta JSON (workers paralelos usan ficheros distintos)")
    ap.add_argument("--resume", action="store_true", help="reusa tickers ya presentes en --out (tras un corte)")
    args = ap.parse_args()
    tickers = [t.strip() for t in args.panel.split(",") if t.strip()]
    seeds = [config.SEED + i for i in range(args.n_seeds)]
    out_path = Path(args.out)

    config.set_seeds(config.SEED)
    result = {"meta": {"seeds": seeds, "n_seeds": len(seeds), "signal_lag": 1, "panel": tickers,
                       "max_models": args.max_models,
                       "scheme": "AutoML ENSEMBLE de semillas (promedio de p1) sobre el pipeline causal de M10",
                       "semillas_fijadas": "a priori = SEED+0..n-1 (idénticas al ensemble M10-XGBoost); NO se elige por OOS",
                       "honestidad": "seed_sensibilidad reporta dist. accuracy (min/med/max/std) por activo",
                       "H0": "el ensemble NO bate a ZeroR causal (universalidad §2 nivel 3)"},
              "por_activo": {}}
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text()).get("por_activo", {})
        result["por_activo"] = {k: v for k, v in prev.items() if "error" not in v}
        print(f"[resume] reusando {len(result['por_activo'])} tickers: {list(result['por_activo'])}")
    try:
        for tk in tickers:
            if tk in result["por_activo"]:
                print(f"\n########## {tk} — ya hecho (resume), saltando ##########")
                continue
            print(f"\n########## {tk} (seeds={seeds}, max_models={args.max_models}) ##########")
            try:
                r = run_ticker(tk, seeds, args.max_models)
                result["por_activo"][tk] = r
                _print_table(tk, r)
            except Exception as e:
                print(f"  !! {tk} FALLÓ: {type(e).__name__}: {e}")
                result["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        ha.shutdown_h2o()
    ok = len([k for k, v in result["por_activo"].items() if "error" not in v])
    print(f"\nOK · {out_path} · {ok} activos")


if __name__ == "__main__":
    main()
