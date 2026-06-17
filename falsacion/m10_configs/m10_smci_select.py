"""Elegir la MEJOR estrategia M10 en SMCI por selección en VALIDACIÓN (no p-hacking: el test no se toca).

Split cronológico: VALIDACIÓN = [N0 : TEST_START], TEST = [TEST_START : fin] (intacto, una vez). Para cada
(config, burn-in) se mide accuracy en validación; se elige el mejor (accuracy primaria; desempate por Sharpe
de validación) y se reporta en test accuracy + Sharpe + equity, contra M5/M8/B&H. Walk-forward desplegable.

Uso: python experiments/m10_smci_select.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from core.backtest import run_backtest
from core.metrics import equity_curve
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SMCI"
STEP, EMBARGO = 21, 1   # embargo=1: horizonte de etiqueta=1 (ver BITACORA 2026-06-17, logic_esential §14b)
TEST_START = 250                                  # test = últimos ~150 días (intacto)
BURNINS = [100, 120, 140, 160, 180, 200]
N_SEEDS = 10
ANN = np.sqrt(252)
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
CONFIGS = {"base": [config.SEED], "ens": SEEDS}
OUT = Path("outputs/experiments/m10_smci_select.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def wf_p1(X, y, N0, seeds):
    """Walk-forward expandible (solo pasado) con burn-in N0; ensemble sobre seeds. p1 en [N0:fin]."""
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 50:
            continue
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(X.iloc[:tr], y.iloc[:tr])
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in seeds]
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p


def metrics(p, mv, oos_ret, lo, hi):
    """accuracy, Sharpe, equity de la posición sign(p-0.5) en el tramo [lo:hi] (días con predicción)."""
    sub = mv.index[lo:hi]
    pr = p.reindex(sub); ok = pr.notna()
    sub = sub[ok.to_numpy()]
    pos = np.where(pr.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    acc = float((pos == truth).mean())
    w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()
    eq = float(equity_curve(pd.Series(nr).dropna()).iloc[-1])
    return round(acc, 4), round(_sr(nr), 3), round(eq, 4), len(sub)


def ref_metrics(mv, oos_ret, lo, hi):
    sub = mv.index[lo:hi]; truth = np.sign(mv.loc[sub, "r_next"].to_numpy()); out = {}
    for k, col in (("m5", "agent_size"), ("m8", "final_size")):
        pos = np.sign(mv.loc[sub, col].to_numpy())
        nr = mv[f"nr_{k}_causal"].reindex(sub).to_numpy()
        out[k] = {"acc": round(float((pos == truth).mean()), 4), "sharpe": round(_sr(nr), 3),
                  "equity": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4)}
    nr = oos_ret.reindex(sub).to_numpy()
    out["bh"] = {"acc": round(float((np.ones(len(sub)) == truth).mean()), 4), "sharpe": round(_sr(nr), 3),
                 "equity": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4)}
    return out


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    n = len(mv); y = (mv["r_next"] > 0).astype(int)

    grid = []
    for cname, seeds in CONFIGS.items():
        for N0 in BURNINS:
            p = wf_p1(mv[ALL22], y, N0, seeds)
            va, vs, ve, vn = metrics(p, mv, oos_ret, N0, TEST_START)        # VALIDACIÓN [N0:TEST_START]
            ta, ts, te, tn = metrics(p, mv, oos_ret, TEST_START, n)         # TEST [TEST_START:fin]
            grid.append({"config": cname, "burnin": N0, "val_acc": va, "val_sharpe": vs, "val_equity": ve,
                         "val_n": vn, "test_acc": ta, "test_sharpe": ts, "test_equity": te, "test_n": tn})

    best = max(grid, key=lambda g: (g["val_acc"], g["val_sharpe"]))        # accuracy primaria; desempate Sharpe
    ref_val = ref_metrics(mv, oos_ret, best["burnin"], TEST_START)
    ref_test = ref_metrics(mv, oos_ret, TEST_START, n)

    # --- Diagnóstico honesto de la elegida en TEST: ¿habilidad o sesgo a corto en mercado bajista? ---
    from core.stats import block_permutation_test, mcnemar_test, sign_test
    p = wf_p1(mv[ALL22], y, best["burnin"], CONFIGS[best["config"]])
    sub = mv.index[TEST_START:n]; pr = p.reindex(sub); ok = pr.notna().to_numpy(); sub = sub[ok]
    pos = np.where(pr.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    corr = (pos == truth).astype(int)
    corr_bh = (np.ones(len(sub)) == truth).astype(int)
    corr_m5 = (np.sign(mv.loc[sub, "agent_size"].to_numpy()) == truth).astype(int)
    _, p_mc_bh, b_bh, c_bh = mcnemar_test(corr_bh, corr)
    _, p_bp_bh = block_permutation_test(corr, corr_bh, seed=config.SEED)
    _, p_mc_m5, _, _ = mcnemar_test(corr_m5, corr)
    k_s, n_s, p_s, ci_s = sign_test(corr)
    diag = {"frac_largo": round(float((pos > 0).mean()), 3), "frac_corto": round(float((pos < 0).mean()), 3),
            "frac_up_test": round(float((truth > 0).mean()), 3),
            "siempre_corto_acc": round(float((np.full(len(sub), -1.0) == truth).mean()), 4),
            "siempre_largo_bh_acc": round(float(corr_bh.mean()), 4),
            "mcnemar_vs_bh_p": round(float(p_mc_bh), 4), "block_perm_vs_bh_p": round(float(p_bp_bh), 4),
            "mcnemar_vs_m5_p": round(float(p_mc_m5), 4),
            "sign_vs_0.5_p": round(float(p_s), 4), "sign_ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}

    result = {"meta": {"ticker": TICKER, "seed": config.SEED, "signal_lag": 1, "n_oos": int(n),
                       "test_start_idx": TEST_START, "test_span": [str(mv.index[TEST_START].date()), str(mv.index[-1].date())],
                       "burnins": BURNINS, "configs": list(CONFIGS), "n_seeds": N_SEEDS,
                       "criterio": "elegir (config,burn-in) por accuracy en VALIDACIÓN; test intacto",
                       "regla": "test no se toca para elegir → selección honesta (validación≠test)"},
              "grid": grid, "elegida": best, "ref_validacion": ref_val, "ref_test": ref_test,
              "diagnostico_test": diag}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"SMCI · OOS n={n} · TEST=[{result['meta']['test_span'][0]}→{result['meta']['test_span'][1]}] (intacto)")
    print(f"\n{'config':6} {'N0':>4} | {'VAL acc':>7} {'SR':>6} {'eq':>6} (n) | {'TEST acc':>8} {'SR':>6} {'eq':>6} (n)")
    print("-" * 74)
    for g in grid:
        mark = "  <<< elegida" if g is best else ""
        print(f"{g['config']:6} {g['burnin']:>4} | {g['val_acc']:>7} {g['val_sharpe']:>+6.2f} {g['val_equity']:>6} ({g['val_n']:>3}) | "
              f"{g['test_acc']:>8} {g['test_sharpe']:>+6.2f} {g['test_equity']:>6} ({g['test_n']:>3}){mark}")
    print(f"\nELEGIDA en validación: {best['config']} / burn-in {best['burnin']} (val_acc={best['val_acc']}, val_SR={best['val_sharpe']})")
    print(f"  → TEST: acc={best['test_acc']} SR={best['test_sharpe']} eq={best['test_equity']}")
    print(f"  REF TEST: M5 acc={ref_test['m5']['acc']} SR={ref_test['m5']['sharpe']} | M8 acc={ref_test['m8']['acc']} SR={ref_test['m8']['sharpe']} | "
          f"B&H acc={ref_test['bh']['acc']} SR={ref_test['bh']['sharpe']} eq={ref_test['bh']['equity']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
