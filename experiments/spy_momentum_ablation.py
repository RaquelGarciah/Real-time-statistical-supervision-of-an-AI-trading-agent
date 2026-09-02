"""¿Cuánto del Sharpe de SPY/aug es momentum y cuánto aporta STRATA? Ablación de features.

Mismo walk-forward desplegable que m10_pivot_scan (config FIJA, todo el OOS = test, ensemble 10 semillas,
embargo 5), variando SOLO el conjunto de features. Aísla la contribución del momentum (REALSIG) frente a
las features STRATA/régimen y de agente. Comparación clave: momentum_solo vs all22+mom (si empatan, STRATA
no aporta sobre el momentum). McNemar pareado entre conjuntos y vs B&H. signal_lag=1, etiqueta signo(r_{t+1}).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis, skew as _skew

import config
from core.backtest import run_backtest
from core.metrics import equity_curve
from core.stats import deflated_sharpe, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import REALSIG, build_realsignal
from experiments.m10_pivot_scan import SEEDS, wf_p1_full, _sr, N0
from experiments.m10_valtest_casestudy import AGENT15, ALL22, STRATA_REGIME7

TICKER = "SPY"
ANN = np.sqrt(252)
FEATURE_SETS = {
    "momentum_solo": REALSIG,                       # 5: solo señal de tendencia, CERO STRATA
    "strata_regime7": STRATA_REGIME7,               # 7: scores STRATA + régimen, sin momentum ni agente
    "agent15": AGENT15,                             # 15: solo el agente LLM
    "all22": ALL22,                                 # 22: STRATA completo, sin momentum (= 'ens' del barrido)
    "strata7+mom": STRATA_REGIME7 + REALSIG,        # 12: STRATA + momentum, sin agente
    "all22+mom": ALL22 + REALSIG,                   # 27: el SPY/aug (Sharpe 2.56)
}
OUT = Path("outputs/experiments/spy_momentum_ablation.json")


def main() -> None:
    config.set_seeds(config.SEED)
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    _, ret_full = wf.load_features(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    mv[REALSIG] = build_realsignal(ret_full, mv.index)
    mv = mv.dropna(subset=REALSIG)
    y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    bh_corr = (np.ones(len(idx)) == truth).astype(int)
    bh_acc = float(bh_corr.mean())

    res, corr_by = {}, {}
    for name, cols in FEATURE_SETS.items():
        p = wf_p1_full(mv[cols], y, None, SEEDS).loc[idx]
        pos = np.where(p.to_numpy() >= 0.5, 1.0, -1.0)
        corr = (pos == truth).astype(int)
        corr_by[name] = corr
        w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()
        nrc = nr[~np.isnan(nr)]
        sr = float(nrc.mean() / nrc.std(ddof=1)) if nrc.std(ddof=1) > 0 else 0.0
        dsr = deflated_sharpe(sr, n_trials=len(FEATURE_SETS), n_obs=len(nrc),
                              skew=float(_skew(nrc)), kurt=float(_kurtosis(nrc, fisher=False)))
        k, n, ps, ci = sign_test(corr)
        _, p_bh, _, _ = mcnemar_test(bh_corr, corr)
        res[name] = {"n_features": len(cols), "accuracy": round(float(corr.mean()), 4),
                     "sharpe_causal": round(_sr(nr), 3), "dsr": round(float(dsr), 4),
                     "equity_final": round(float(equity_curve(pd.Series(nrc)).iloc[-1]), 4),
                     "sign_vs_azar_p": round(float(ps), 4), "mcnemar_vs_bh_p": round(float(p_bh), 4)}

    # ¿STRATA aporta sobre el momentum? McNemar pareado entre all22+mom y momentum_solo.
    _, p_inc, b, c = mcnemar_test(corr_by["momentum_solo"], corr_by["all22+mom"])
    incremento = {"mcnemar_all22mom_vs_momsolo_p": round(float(p_inc), 4),
                  "dias_solo_mom_acierta": int(b), "dias_all22mom_acierta": int(c),
                  "delta_accuracy": round(res["all22+mom"]["accuracy"] - res["momentum_solo"]["accuracy"], 4)}

    out = {"meta": {"ticker": TICKER, "seed": config.SEED, "n_seeds": len(SEEDS), "signal_lag": 1,
                    "n_eval": int(len(idx)), "oos": [str(idx.min().date()), str(idx.max().date())],
                    "bh_accuracy": round(bh_acc, 4)},
           "feature_sets": res, "aporta_strata_sobre_momentum": incremento}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"SPY  OOS {out['meta']['oos']}  n={len(idx)}  B&H acc={bh_acc:.4f}\n")
    print(f"{'conjunto':<16}{'#f':>4}{'acc':>9}{'Sharpe':>9}{'DSR':>8}{'equity':>9}{'sign_p':>9}{'vsBH_p':>9}")
    for name, r in res.items():
        print(f"{name:<16}{r['n_features']:>4}{r['accuracy']:>9.4f}{r['sharpe_causal']:>9.3f}"
              f"{r['dsr']:>8.3f}{r['equity_final']:>9.4f}{r['sign_vs_azar_p']:>9.4f}{r['mcnemar_vs_bh_p']:>9.4f}")
    print(f"\n¿STRATA aporta sobre momentum? all22+mom vs momentum_solo: "
          f"Δacc={incremento['delta_accuracy']:+.4f}, McNemar p={incremento['mcnemar_all22mom_vs_momsolo_p']}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
