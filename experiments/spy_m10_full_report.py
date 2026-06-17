"""M10 canónico (ALL22, ensemble 10 semillas) sobre SPY: cuadro completo de métricas vs M5/M8/B&H.

Arquitectura (especificada por Raquel): XGBoost binario (300 árboles, prof 4, lr 0.05, subsample 0.8,
colsample 0.8, reg_lambda 1.0), ensemble de 10 semillas (p1 = media de predict_proba). Features = ALL22
(15 agente + 7 STRATA/régimen). Walk-forward expandible: N0=150, STEP=21, tr_end=start−5 (embargo 5),
decisión = signo(p1−0.5), etiqueta y=1[r_{t+1}>0], signal_lag=1. Evaluación sobre [N0:fin].

Reporta, para M5 (agente), M8 (STRATA), M10 y B&H: accuracy direccional, Sharpe causal, equity final,
max drawdown; y para M10 además AUC, log-loss, Brier (es el único probabilístico). Tests pareados:
McNemar M10 vs {M5,M8,B&H}, sign test vs 0.5. DSR/PSR con caveat de multiplicidad.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config
from core.backtest import run_backtest
from core.metrics import equity_curve
from core.stats import deflated_sharpe, mcnemar_test, sign_test, stationary_bootstrap_ci
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_pivot_scan import SEEDS, N0, STEP, PARAMS
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SPY"
ANN = np.sqrt(252)
EMBARGO = 1
OUT = Path("outputs/experiments/spy_m10_full_report.json")


def wf_p1(X: pd.DataFrame, y: pd.Series, seeds, embargo: int) -> pd.Series:
    """Walk-forward expandible (solo pasado), ensemble de semillas, embargo parametrizable. tr_end=start−embargo."""
    n = len(X)
    p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr_end = start - embargo
        if tr_end < 60:
            continue
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**PARAMS, random_state=sd)
                 .fit(X.iloc[:tr_end], y.iloc[:tr_end]).predict_proba(X.iloc[start:end])[:, 1] for sd in seeds]
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p


def econ(nr: np.ndarray) -> dict:
    nrc = nr[~np.isnan(nr)]
    sr = float(nrc.mean() / nrc.std(ddof=1) * ANN) if nrc.std(ddof=1) > 0 else 0.0
    eq = equity_curve(pd.Series(nrc))
    dd = float((eq / eq.cummax() - 1).min())
    return {"sharpe": round(sr, 3), "equity_final": round(float(eq.iloc[-1]), 4), "max_drawdown": round(dd, 4)}


def main() -> None:
    config.set_seeds(config.SEED)
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    y_bin = (truth > 0).astype(int)

    # M10 (ALL22, ensemble 10 semillas), embargo=1
    p1 = wf_p1(mv[ALL22], y, SEEDS, EMBARGO).loc[idx].to_numpy()
    pos_m10 = np.where(p1 >= 0.5, 1.0, -1.0)

    # posiciones de referencia
    pos = {"M5": np.sign(mv.loc[idx, "agent_size"].to_numpy()),
           "M8": np.sign(mv.loc[idx, "final_size"].to_numpy()),
           "M10": pos_m10,
           "B&H": np.ones(len(idx))}
    nr_ref = {"M5": mv["nr_m5_causal"].reindex(idx).to_numpy(),
              "M8": mv["nr_m8_causal"].reindex(idx).to_numpy(),
              "B&H": oos_ret.reindex(idx).to_numpy()}
    w10 = pd.Series(0.0, index=mv.index); w10.loc[idx] = pos_m10
    nr_ref["M10"] = run_backtest(oos_ret, w10, signal_lag=1)["net_return"].reindex(idx).to_numpy()

    corr = {k: (v == truth).astype(int) for k, v in pos.items()}
    rep = {}
    for k in ["M5", "M8", "M10", "B&H"]:
        rep[k] = {"accuracy": round(float(corr[k].mean()), 4), **econ(nr_ref[k])}

    # métricas probabilísticas (solo M10)
    rep["M10"].update({
        "auc": round(float(roc_auc_score(y_bin, p1)), 4),
        "log_loss": round(float(log_loss(y_bin, np.clip(p1, 1e-6, 1 - 1e-6))), 4),
        "brier": round(float(brier_score_loss(y_bin, p1)), 4),
    })
    nrc = nr_ref["M10"][~np.isnan(nr_ref["M10"])]
    from scipy.stats import kurtosis, skew
    sr_raw = float(nrc.mean() / nrc.std(ddof=1)) if nrc.std(ddof=1) > 0 else 0.0
    rep["M10"]["dsr_ntrials1"] = round(float(deflated_sharpe(sr_raw, n_trials=1, n_obs=len(nrc),
                                       skew=float(skew(nrc)), kurt=float(kurtosis(nrc, fisher=False)))), 4)

    # tests pareados M10 vs resto + sign test + IC bootstrap del exceso de accuracy sobre 0.5
    tests = {}
    for opp in ["M5", "M8", "B&H"]:
        _, p_mc, b, c = mcnemar_test(corr[opp], corr["M10"])
        tests[f"M10_vs_{opp}_mcnemar_p"] = round(float(p_mc), 4)
        tests[f"M10_vs_{opp}_b_c"] = [int(b), int(c)]
    k, n, ps, ci = sign_test(corr["M10"])
    tests["M10_sign_vs_0.5_p"] = round(float(ps), 4)
    lo, hi, _ = stationary_bootstrap_ci(corr["M10"].astype(float) - 0.5, seed=config.SEED)
    tests["M10_exceso_acc_IC95"] = [round(float(lo), 4), round(float(hi), 4)]

    out = {"meta": {"ticker": TICKER, "n_eval": int(len(idx)), "n_seeds": len(SEEDS),
                    "features": "ALL22", "embargo": EMBARGO, "oos": [str(idx.min().date()), str(idx.max().date())],
                    "frac_up_bh": round(float((truth > 0).mean()), 4),
                    "nota_dsr": "DSR con n_trials=1 (PSR vs 0); el proyecto probó varias configs → DSR real menor"},
           "metricas": rep, "tests": tests}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"SPY · M10 ALL22 ensemble · OOS {out['meta']['oos']} · n={len(idx)} · B&H frac_up={out['meta']['frac_up_bh']}\n")
    print(f"{'modelo':<6}{'acc':>8}{'Sharpe':>9}{'equity':>9}{'maxDD':>9}{'AUC':>8}{'logloss':>9}{'Brier':>8}")
    for k in ["M5", "M8", "M10", "B&H"]:
        r = rep[k]
        print(f"{k:<6}{r['accuracy']:>8.4f}{r['sharpe']:>9.3f}{r['equity_final']:>9.4f}{r['max_drawdown']:>9.4f}"
              f"{r.get('auc', float('nan')):>8}{r.get('log_loss', float('nan')):>9}{r.get('brier', float('nan')):>8}")
    print("\nTests pareados (M10):")
    for kk, vv in tests.items():
        print(f"  {kk}: {vv}")
    print(f"\n>>> {OUT}")


if __name__ == "__main__":
    main()
