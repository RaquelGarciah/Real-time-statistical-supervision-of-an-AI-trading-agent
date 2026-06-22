"""Optuna como selección de hiperparámetros DEPLOYABLE para el meta-learner (SPY).

Pregunta: optimizando hiperparámetros del meta-learner (no la semilla por OOS, que NO sería deployable),
¿se bate a ZeroR? Diseño deployable y sin fuga:

- Optuna optimiza hiperparámetros XGBoost en una **partición de validación interna** carved del burn-in
  inicial [0:N0] (sub-train = primeros 70%, validación = últimos 30%, con embargo). NUNCA toca el OOS.
- Se congela la mejor config y se ejecuta el walk-forward DESPLEGABLE (embargo=1, reentreno mensual,
  ensemble de 10 semillas) sobre el OOS, evaluado UNA sola vez.
- Elegir la semilla por su accuracy OOS NO es deployable (requiere conocer el futuro) → se usa ensemble.

H0 (universalidad §2 nivel 3): ni con hiperparámetros optimizados el meta-learner bate a ZeroR causal.
Uso: python experiments/optuna_m10_spy.py [--ticker SPY] [--trials 80]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score

import config
from core.backtest import run_backtest
from core import metrics
from core.stats import mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from falsacion.m10_configs.m10_v3_causal_panel import build_states_onthefly

N0, STEP, EMBARGO = 150, 21, 1
SEEDS = [config.SEED + i for i in range(10)]
ANN = np.sqrt(252)
FULL_COLS = ([f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
             + ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"])
OUT = Path("outputs/experiments/optuna_m10.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def tune_on_burnin(Xb: pd.DataFrame, yb: pd.Series, n_trials: int) -> dict:
    """Optuna sobre validación interna del burn-in (sub-train 70% / val 30%, embargo). Sin OOS."""
    n = len(Xb); cut = int(n * 0.7)
    Xtr, ytr = Xb.iloc[:cut - EMBARGO], yb.iloc[:cut - EMBARGO]
    Xva, yva = Xb.iloc[cut:], yb.iloc[cut:]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=config.SEED)
        clf = xgb.XGBClassifier(**params).fit(Xtr, ytr)
        pred = (clf.predict_proba(Xva)[:, 1] > 0.5).astype(int)
        return accuracy_score(yva, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_val_acc": round(float(study.best_value), 4),
            "n_trials": n_trials, "val_n": int(len(Xva))}


def wf_ensemble(X: pd.DataFrame, y: pd.Series, params: dict) -> pd.Series:
    """Walk-forward desplegable (embargo=1) con la config congelada, ensemble de 10 semillas."""
    base = dict(params, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
    p1 = pd.Series(np.nan, index=X.index, dtype=float); n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**base, random_state=sd).fit(X.iloc[:tr_end], y.iloc[:tr_end])
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in SEEDS]
        p1.iloc[start:end] = np.mean(preds, axis=0)
    return p1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--trials", type=int, default=80)
    args = ap.parse_args()
    config.set_seeds(config.SEED)

    wf.TICKER = args.ticker; wf.reset_thresholds_cache()
    gamma, sigma, oos = build_states_onthefly(args.ticker)
    m = wf.run_master(gamma, sigma, oos, wf.load_agent(args.ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    # Tuning SOLO en el burn-in (deployable): [0:N0] con validación interna.
    tuned = tune_on_burnin(X.iloc[:N0], y.iloc[:N0], args.trials)
    print(f"Optuna best (val interna): acc={tuned['best_val_acc']} params={tuned['best_params']}")

    p1 = wf_ensemble(X, y, tuned["best_params"])
    td = X.index[p1.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    ybf = (mv["r_next"] > 0).astype(int).to_numpy()
    zr = np.where(pd.Series(ybf).expanding().mean().shift().fillna(0.5).to_numpy()[mv.index.get_indexer(td)] >= 0.5, 1.0, -1.0)
    pos = {"optuna": np.sign(p1.loc[td].to_numpy() - 0.5),
           "m5": np.sign(mv.loc[td, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[td, "final_size"].to_numpy()),
           "zeror": zr, "bh": np.ones(len(td))}
    correct = {k: (v == truth).astype(int) for k, v in pos.items()}

    def net(w):
        s = pd.Series(0.0, index=m.index); s.loc[td] = w
        return run_backtest(oos, s, signal_lag=1)["net_return"].reindex(td).to_numpy()

    table = {}
    for k, v in pos.items():
        nr = pd.Series(net(v), index=td).dropna(); eq = (1 + nr).cumprod()
        table[k] = {"accuracy": round(float(correct[k].mean()), 4), "sharpe": round(_sr(nr.to_numpy()), 3),
                    "max_dd": round(metrics.max_drawdown(eq), 4), "calmar": round(metrics.calmar(nr), 3),
                    "equity_final": round(float(eq.iloc[-1]), 4)}
    _, p_zr, b_zr, c_zr = mcnemar_test(correct["zeror"], correct["optuna"])
    k_s, n_s, p_s, ci_s = sign_test(correct["optuna"])

    print(f"\n=== {args.ticker} · Optuna-M10 deployable (embargo=1, ensemble 10) · n={len(td)} ===")
    for k in ("m5", "m8", "optuna", "zeror", "bh"):
        a = table[k]; print(f"  {k:7} acc={a['accuracy']} sharpe={a['sharpe']} maxDD={a['max_dd']} calmar={a['calmar']}")
    print(f"  Optuna vs ZeroR: McNemar p={round(float(p_zr),4)} (ZeroR solo {b_zr}, Optuna solo {c_zr})")
    print(f"  Optuna sign vs 0.5: p={round(float(p_s),4)} ({k_s}/{n_s}) IC95={[round(float(ci_s[0]),4),round(float(ci_s[1]),4)]}")
    bate = bool(table["optuna"]["accuracy"] > table["zeror"]["accuracy"] and p_zr < 0.10)
    print(f"  ¿Optuna BATE a ZeroR significativamente? {bate}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": {"ticker": args.ticker, "scheme": "Optuna en validación interna del burn-in (deployable), WF embargo=1 ensemble 10",
                                        "no_phacking": "hiperparámetros desde burn-in; semilla NO elegida por OOS (no deployable)",
                                        "seed": config.SEED}, "optuna": tuned, "table": table,
                               "optuna_vs_zeror": {"p": round(float(p_zr), 4), "zeror_solo": int(b_zr), "optuna_solo": int(c_zr)},
                               "optuna_sign": {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4)},
                               "bate_zeror": bate}, indent=2, ensure_ascii=False))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
