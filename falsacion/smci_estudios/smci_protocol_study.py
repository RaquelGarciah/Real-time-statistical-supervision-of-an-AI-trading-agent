"""Sensibilidad del protocolo + robustez por ventanas para SMCI (M10 walk-forward desplegable).

Responde: ¿mejora la accuracy si reentrenamos cada día (vs semanal/mensual)? ¿y con más burn-in? ¿el
acierto de M10 sobre B&H/M5 es consistente por sub-periodos o suerte de uno? Config M10 fija (300×4, 22
features, vanilla). Vuelca JSON para que decision_activo.ipynb haga las gráficas.

Uso: python experiments/smci_protocol_study.py
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
from core.stats import block_permutation_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SMCI"
EMBARGO = 5
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
              random_state=config.SEED, tree_method="hist")
OUT = Path("outputs/experiments/smci_protocol_study.json")


def wf_predict(mv, N0, step):
    """Walk-forward expandible (solo pasado): p OOS (NaN en burn-in). Reentrena cada `step` días desde N0."""
    X, y = mv[ALL22], (mv["r_next"] > 0).astype(int); n = len(mv)
    p = pd.Series(np.nan, index=mv.index)
    for start in range(N0, n, step):
        tr_end = start - EMBARGO
        if tr_end < 60:
            continue
        clf = xgb.XGBClassifier(**PARAMS).fit(X.iloc[:tr_end], y.iloc[:tr_end])
        end = min(start + step, n)
        p.iloc[start:end] = clf.predict_proba(X.iloc[start:end])[:, 1]
    return p


def acc_on(p, mv, start_pos):
    idx = mv.index[start_pos:]
    idx = idx[p.loc[idx].notna()]
    pred = np.sign(p.loc[idx].to_numpy() - 0.5)
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    return round(float((pred == truth).mean()), 4), idx


def ref_acc(mv, idx, col):
    pred = np.sign(mv.loc[idx, col].to_numpy()) if col != "bh" else np.ones(len(idx))
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    return round(float((pred == truth).mean()), 4)


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]; n = len(mv)

    # --- A. Frecuencia de reentreno: N0=150 fijo, step ∈ {1,5,10,21}, mismo tramo [150:fin] ---
    freq = []
    for step in (1, 5, 10, 21):
        p = wf_predict(mv, 150, step)
        a, idx = acc_on(p, mv, 150)
        freq.append({"step_dias": step, "n_reentrenos": len(range(150, n, step)), "n_test": len(idx),
                     "acc_m10": a, "acc_m5": ref_acc(mv, idx, "agent_size"),
                     "acc_m8": ref_acc(mv, idx, "final_size"), "acc_bh": ref_acc(mv, idx, "bh")})

    # --- B. Burn-in: step=21 fijo, N0 ∈ {120,150,200,250}, mismo tramo común [250:fin] (comparable) ---
    burn = []
    for N0 in (120, 150, 200, 250):
        p = wf_predict(mv, N0, 21)
        a, idx = acc_on(p, mv, 250)
        burn.append({"burn_in": N0, "n_test_comun": len(idx), "acc_m10": a,
                     "acc_bh": ref_acc(mv, idx, "bh")})

    # --- C. Robustez por ventanas (protocolo canónico N0=150/step=21): Δacc por ventana deslizante ---
    p_can = wf_predict(mv, 150, 21)
    idx = mv.index[150:][p_can.loc[mv.index[150:]].notna()]
    corr_m10 = (np.sign(p_can.loc[idx].to_numpy() - 0.5) == np.sign(mv.loc[idx, "r_next"].to_numpy())).astype(int)
    corr_bh = (np.ones(len(idx)) == np.sign(mv.loc[idx, "r_next"].to_numpy())).astype(int)
    corr_m5 = (np.sign(mv.loc[idx, "agent_size"].to_numpy()) == np.sign(mv.loc[idx, "r_next"].to_numpy())).astype(int)
    W, STEP_W = 63, 10
    win = []
    for s in range(0, len(idx) - W + 1, STEP_W):
        sl = slice(s, s + W)
        win.append({"ini": int(s),
                    "dacc_m10_bh": round(float(corr_m10[sl].mean() - corr_bh[sl].mean()), 4),
                    "dacc_m10_m5": round(float(corr_m10[sl].mean() - corr_m5[sl].mean()), 4),
                    "acc_m10": round(float(corr_m10[sl].mean()), 4), "acc_bh": round(float(corr_bh[sl].mean()), 4)})
    frac_bh = round(float(np.mean([w["dacc_m10_bh"] > 0 for w in win])), 3)
    frac_m5 = round(float(np.mean([w["dacc_m10_m5"] > 0 for w in win])), 3)
    # Significancia global (autocorr-robusta) sobre todo el tramo.
    _, p_bh = block_permutation_test(corr_m10, corr_bh)
    _, p_m5 = block_permutation_test(corr_m10, corr_m5)
    _, _, p_az, _ = sign_test(corr_m10)

    result = {"meta": {"ticker": TICKER, "embargo": EMBARGO, "config": "M10-WF 300x4 all22 vanilla",
                       "n_total": int(n), "seed": config.SEED, "ventana": W, "paso_ventana": STEP_W},
              "A_frecuencia_reentreno": freq, "B_burn_in": burn,
              "C_ventanas": {"per_window": win, "frac_ventanas_m10_gt_bh": frac_bh,
                             "frac_ventanas_m10_gt_m5": frac_m5, "n_ventanas": len(win),
                             "global_blockperm_p_vs_bh": round(float(p_bh), 4),
                             "global_blockperm_p_vs_m5": round(float(p_m5), 4),
                             "global_sign_vs_0.5_p": round(float(p_az), 4)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("A) Frecuencia de reentreno (N0=150, tramo [150:fin]):")
    for f in freq:
        print(f"   step={f['step_dias']:2}d ({f['n_reentrenos']:3} reentrenos) acc_M10={f['acc_m10']}  (M5={f['acc_m5']} M8={f['acc_m8']} B&H={f['acc_bh']})")
    print("B) Burn-in (step=21, tramo común [250:fin]):")
    for b in burn:
        print(f"   N0={b['burn_in']:3} acc_M10={b['acc_m10']}  (B&H={b['acc_bh']}, n={b['n_test_comun']})")
    print(f"C) Ventanas ({len(win)} de {W}d): M10>B&H en {frac_bh:.0%}, M10>M5 en {frac_m5:.0%}")
    print(f"   Global: block-perm vs B&H p={p_bh:.3f}, vs M5 p={p_m5:.3f}, sign vs 0.5 p={p_az:.3f}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
