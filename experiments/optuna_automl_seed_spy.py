"""Optuna sobre la SEMILLA de H2O AutoML (SPY). EXPLORATORIO / best-case-by-search.

⚠️ ATENCIÓN: optimizar la semilla por accuracy OOS es p-hacking (selecciona mirando el OOS,
no es deployable). Esto NO es un resultado reportable en la memoria. Su único valor legítimo es
de SENSIBILIDAD: medir cuánto puede inflarse la accuracy de AutoML solo eligiendo la semilla
→ evidencia de fragilidad frente al techo ZeroR. Pedido explícito de Raquel (2026-06-22).

Config: max_models=20, embargo=1, include_algos=[GBM, StackedEnsemble], sort_metric=AUC.
Entorno: VM x86 → la "mejor semilla" NO es transferible al Mac (ARM).

Uso: python experiments/optuna_automl_seed_spy.py [--ticker SPY] [--trials 25]
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

import config
from core import h2o_automl as ha
import experiments.walkforward_robustez as wf
from falsacion.m10_configs.m10_v3_causal_panel import build_states_onthefly

N0, STEP, EMBARGO, N_SPLITS = 150, 21, 1, 5
MAX_MODELS = 20
INCLUDE_ALGOS = ["GBM", "StackedEnsemble"]
SORT_METRIC = "AUC"
FULL_COLS = ([f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
             + ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"])


def build_xy(ticker: str):
    wf.TICKER = ticker
    wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]
    y = (mv["r_next"] > 0).astype(int)
    return X, y, mv


def wf_acc(X: pd.DataFrame, y: pd.Series, mv: pd.DataFrame, seed: int) -> tuple[float, float]:
    """WF AutoML con una semilla dada. Devuelve (acc OOS, acc ZeroR causal) sobre las mismas fechas."""
    p1 = pd.Series(np.nan, index=X.index, dtype=float)
    n = len(X)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 50:
            continue
        leader, _ = ha.train_h2o(X.iloc[:tr_end], y.iloc[:tr_end], use_fold_column=True,
                                 max_models=MAX_MODELS, n_splits=N_SPLITS, embargo=EMBARGO, seed=seed,
                                 holdout_frac=0.0, sort_metric=SORT_METRIC, include_algos=INCLUDE_ALGOS)
        end = min(start + STEP, n)
        p1.iloc[start:end] = ha.predict_class1_proba(leader, X.iloc[start:end])

    td = X.index[p1.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    pred = np.where(p1[td].to_numpy() > 0.5, 1.0, -1.0)
    acc = float((pred == truth).mean())

    # ZeroR causal (clase mayoritaria del pasado estricto), idéntico a automl_m10.run_ticker
    yb = (mv["r_next"] > 0).astype(int)
    zr = np.array([1.0 if (yb.loc[:t].iloc[:-1].mean() >= 0.5 if len(yb.loc[:t]) > 1 else True) else -1.0 for t in td])
    zacc = float((zr == truth).mean())
    return acc, zacc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--out", default="outputs/experiments/automl_runs/optuna_SEED_phacking_SPY_mm20_GBM-SE_emb1_AUC_x86.json")
    args = ap.parse_args()

    X, y, mv = build_xy(args.ticker)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trials: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        seed = trial.suggest_int("seed", 0, 10000)
        acc, zacc = wf_acc(X, y, mv, seed)
        trials.append({"seed": seed, "acc": round(acc, 4), "zeror": round(zacc, 4), "beats_zeror": acc > zacc})
        # volcado incremental: parar a media no pierde lo hecho
        json.dump({"meta": META | {"trials_done": len(trials)}, "trials": trials,
                   "best": max(trials, key=lambda r: r["acc"])}, open(out_path, "w"), indent=2)
        print(f"trial {len(trials)}: seed={seed} acc={acc:.4f} zeror={zacc:.4f} {'BATE' if acc>zacc else '-'}")
        return acc

    global META
    META = {"WARNING": "EXPLORATORIO / p-hacking: semilla optimizada por accuracy OOS. NO reportable en la memoria.",
            "uso_legitimo": "analisis de sensibilidad: cuanto sube la accuracy solo eligiendo semilla = fragilidad",
            "ticker": args.ticker, "max_models": MAX_MODELS, "embargo": EMBARGO, "n0": N0, "step": STEP,
            "include_algos": INCLUDE_ALGOS, "sort_metric": SORT_METRIC, "n_trials": args.trials,
            "env": "VM x86 (e2-highmem-8) - semilla NO transferible al Mac ARM"}

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)

    best = max(trials, key=lambda r: r["acc"])
    print(f"\nBEST: seed={best['seed']} acc={best['acc']} zeror={best['zeror']} beats_zeror={best['beats_zeror']}")
    print(f"OK · {out_path}")


if __name__ == "__main__":
    main()
