"""Robustez al embargo: accuracy de M10 (ens) en SMCI para embargo ∈ {0,1,2,3,5,10}.

Evidencia de transparencia exigida por @rigor-matematico: muestra que la accuracy NO es monótona en el
embargo y que la significancia vs B&H es un pico aislado en embargo=1 (no una meseta) → embargo=1 se elige
por PRINCIPIO (horizonte=1), no por su p-valor. Walk-forward expandible, ensemble 10 semillas, todo el OOS.

Uso: python experiments/m10_smci_embargo.py
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
from core.stats import block_permutation_test, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SMCI"
N0, STEP = 150, 21
EMBARGOS = [0, 1, 2, 3, 5, 10]
SEEDS = [config.SEED + i for i in range(10)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
OUT = Path("outputs/experiments/m10_smci_embargo.json")


def main() -> None:
    config.set_seeds(config.SEED); wf.reset_thresholds_cache()
    g, s, o = build_states_onthefly(TICKER)
    m = wf.run_master(g, s, o, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    n = len(mv); y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]; truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    bh = (np.ones(len(idx)) == truth).astype(int)
    m5 = (np.sign(mv.loc[idx, "agent_size"].to_numpy()) == truth).astype(int)

    rows = []
    for E in EMBARGOS:
        p = pd.Series(np.nan, index=mv.index)
        for start in range(N0, n, STEP):
            tr = start - E
            if tr < 60:
                continue
            end = min(start + STEP, n)
            pr = [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(mv[ALL22].iloc[:tr], y.iloc[:tr])
                  .predict_proba(mv[ALL22].iloc[start:end])[:, 1] for sd in SEEDS]
            p.iloc[start:end] = np.mean(pr, axis=0)
        pos = np.where(p.reindex(idx).to_numpy() >= 0.5, 1.0, -1.0)
        corr = (pos == truth).astype(int)
        _, pmc, _, _ = mcnemar_test(bh, corr)
        _, pbp = block_permutation_test(corr, bh, seed=config.SEED)
        _, pbp5 = block_permutation_test(corr, m5, seed=config.SEED)
        _, _, paz, _ = sign_test(corr)
        rows.append({"embargo": E, "accuracy": round(float(corr.mean()), 4),
                     "frac_corto": round(float((pos < 0).mean()), 3),
                     "blockperm_vs_bh_p": round(float(pbp), 4), "mcnemar_vs_bh_p": round(float(pmc), 4),
                     "blockperm_vs_m5_p": round(float(pbp5), 4), "sign_vs_0.5_p": round(float(paz), 4)})

    bonf5 = round(min(r["blockperm_vs_bh_p"] for r in rows) * len(EMBARGOS), 4)
    result = {"meta": {"ticker": TICKER, "n_eval": int(len(idx)), "embargos": EMBARGOS, "n_seeds": len(SEEDS),
                       "frac_up": round(float((truth > 0).mean()), 3),
                       "lectura": "pico aislado de sig. en emb=1 (emb 0 y 2 no sig) → ruido; emb=1 por principio, no por p",
                       "bonferroni5_min_blockperm_vs_bh": bonf5,
                       "pre_registro": "BITACORA 2026-06-17"},
              "por_embargo": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"SMCI · robustez al embargo (ens, n_eval={len(idx)})")
    print(f"{'emb':>4} {'acc':>7} {'%corto':>7} {'bp vsBH':>8} {'McN vsBH':>9} {'bp vsM5':>8} {'sign0.5':>8}")
    for r in rows:
        print(f"{r['embargo']:>4} {r['accuracy']:>7} {r['frac_corto']:>7} {r['blockperm_vs_bh_p']:>8} "
              f"{r['mcnemar_vs_bh_p']:>9} {r['blockperm_vs_m5_p']:>8} {r['sign_vs_0.5_p']:>8}")
    print(f"Bonferroni-5 del mínimo p(block-perm vs B&H): {bonf5} → {'sig' if bonf5 < 0.10 else 'NO sig'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
