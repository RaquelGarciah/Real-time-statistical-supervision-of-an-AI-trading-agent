"""¿El embargo afecta de verdad a M10, o es ruido? Barrido de embargo en SPY y SMCI.

M10 ALL22, ensemble 10 semillas, WF (N0=150, STEP=21). Se varía SOLO el embargo (tr_end=start−embargo)
en {1,2,3,5,10,21} y se mide accuracy/Sharpe OOS. Para juzgar si las diferencias son señal o ruido se
reporta la banda de ruido binomial (sqrt(0.25/n)) y la dispersión entre semillas individuales.

Importante: el embargo NO es un hiperparámetro de rendimiento; es control de fuga (López de Prado 2018,
sec. 7.4). Elegirlo por la accuracy OOS sería look-ahead. Este barrido es diagnóstico, no selección.
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
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_pivot_scan import SEEDS, N0, STEP, PARAMS
from experiments.m10_valtest_casestudy import ALL22

EMBARGOS = [1, 2, 3, 5, 10, 21]
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/embargo_sweep.json")


def wf_p1_seed(X, y, sd, embargo):
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr_end = start - embargo
        if tr_end < 60:
            continue
        end = min(start + STEP, n)
        p.iloc[start:end] = xgb.XGBClassifier(**PARAMS, random_state=sd).fit(
            X.iloc[:tr_end], y.iloc[:tr_end]).predict_proba(X.iloc[start:end])[:, 1]
    return p


def sharpe(nr):
    nrc = nr[~np.isnan(nr)]
    return float(nrc.mean() / nrc.std(ddof=1) * ANN) if nrc.std(ddof=1) > 0 else 0.0


def run_asset(tk):
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    n = len(idx)
    banda = float(np.sqrt(0.25 / n))                       # 1 SD binomial de una accuracy ~0.5
    res = {"n": n, "banda_ruido_1sd": round(banda, 4), "por_embargo": {}}
    for emb in EMBARGOS:
        accs_seed = []
        p1_ens = np.zeros(n)
        for sd in SEEDS:
            p1 = wf_p1_seed(mv[ALL22], y, sd, emb).loc[idx].to_numpy()
            accs_seed.append(float((np.where(p1 >= 0.5, 1.0, -1.0) == truth).mean()))
            p1_ens += p1
        p1_ens /= len(SEEDS)
        pos = np.where(p1_ens >= 0.5, 1.0, -1.0)
        w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()
        res["por_embargo"][str(emb)] = {
            "acc_ensemble": round(float((pos == truth).mean()), 4),
            "sharpe_ensemble": round(sharpe(nr), 3),
            "acc_seed_media": round(float(np.mean(accs_seed)), 4),
            "acc_seed_std": round(float(np.std(accs_seed)), 4),
            "acc_seed_min": round(float(np.min(accs_seed)), 4),
            "acc_seed_max": round(float(np.max(accs_seed)), 4),
        }
    return res


def main():
    config.set_seeds(config.SEED)
    res = {tk: run_asset(tk) for tk in ["SPY", "SMCI"]}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for tk, r in res.items():
        print(f"\n=== {tk} === (n={r['n']}, banda de ruido ±{r['banda_ruido_1sd']} 1SD; "
              f"rango embargo<banda ⇒ ruido)")
        print(f"{'embargo':<9}{'acc_ens':>9}{'Sharpe':>9}{'acc_seed(media±std)':>24}{'[min,max] semillas':>22}")
        for emb in EMBARGOS:
            e = r["por_embargo"][str(emb)]
            print(f"{emb:<9}{e['acc_ensemble']:>9.4f}{e['sharpe_ensemble']:>9.3f}"
                  f"{(str(e['acc_seed_media'])+'±'+str(e['acc_seed_std'])):>24}"
                  f"{('['+str(e['acc_seed_min'])+','+str(e['acc_seed_max'])+']'):>22}")
        accs = [r["por_embargo"][str(e)]["acc_ensemble"] for e in EMBARGOS]
        print(f"  rango accuracy entre embargos: {max(accs)-min(accs):.4f}  (banda ruido 1SD={r['banda_ruido_1sd']})")
    print(f"\n>>> {OUT}")


if __name__ == "__main__":
    main()
