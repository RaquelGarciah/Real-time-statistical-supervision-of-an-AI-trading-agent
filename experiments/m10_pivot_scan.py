"""Elección rigurosa del activo de pivote: ¿dónde la M10 DESPLEGABLE bate a M5/M8/B&H en accuracy?

M10 desplegable = walk-forward expandible (reentreno mensual, solo pasado) → TODO el OOS es test válido
para una config FIJA (no hay selección por activo → sin sobreajuste de selección; más potencia que la
loncha 40%). Tres configs fijas a priori (base / ensemble / +señal-real). Por activo×config: accuracy,
McNemar + block-permutation vs M5/M8/B&H, sign vs 0.5, Sharpe causal, equity, Deflated Sharpe. Holm-30
sobre la familia primaria M10-vs-B&H. Cohorte ex-ante B&H-débil (mecanística, anti-cherry-pick).

Pre-registro: BITACORA.md [2026-06-16]. Uso: python experiments/m10_pivot_scan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import kurtosis as _kurtosis, skew as _skew

import config
from core.backtest import run_backtest
from core.metrics import equity_curve
from core.stats import block_permutation_test, deflated_sharpe, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import REALSIG, build_realsignal, recency_weight
from experiments.m10_valtest_casestudy import ALL22

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
N0, STEP, EMBARGO = 150, 21, 5
N_SEEDS = 10
ANN = np.sqrt(252)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
CONFIGS = {
    "base": {"cols": ALL22, "hl": None, "seeds": [config.SEED]},
    "ens": {"cols": ALL22, "hl": None, "seeds": SEEDS},
    "aug": {"cols": ALL22 + REALSIG, "hl": None, "seeds": SEEDS},
}
OUT = Path("outputs/experiments/m10_pivot_scan.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def wf_p1_full(X: pd.DataFrame, y: pd.Series, hl, seeds) -> pd.Series:
    """Walk-forward expandible (solo pasado): p1 OOS para [N0:fin] (NaN en burn-in). Ensemble sobre seeds.

    Embargo=5 purga la frontera train/predict (etiqueta horizonte 1 día; López de Prado 2018, sec. 7.4).
    """
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 60:
            continue
        sw = recency_weight(tr_end, hl)
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**PARAMS, random_state=sd)
                 .fit(X.iloc[:tr_end], y.iloc[:tr_end], sample_weight=sw)
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in seeds]
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p


def run_asset(tk: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(tk)
    _, ret_full = wf.load_features(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    mv[REALSIG] = build_realsignal(ret_full, mv.index)
    mv = mv.dropna(subset=REALSIG)
    n = len(mv); y = (mv["r_next"] > 0).astype(int)
    idx_eval = mv.index[N0:]                                # días con predicción WF (burn-in fuera)
    truth = np.sign(mv.loc[idx_eval, "r_next"].to_numpy())

    ref = {"m5": np.sign(mv.loc[idx_eval, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[idx_eval, "final_size"].to_numpy()),
           "bh": np.ones(len(idx_eval))}
    acc_ref = {k: round(float((v == truth).mean()), 4) for k, v in ref.items()}
    corr_ref = {k: (v == truth).astype(int) for k, v in ref.items()}
    sr_ref = {"m5": round(_sr(mv["nr_m5_causal"].reindex(idx_eval).to_numpy()), 3),
              "m8": round(_sr(mv["nr_m8_causal"].reindex(idx_eval).to_numpy()), 3),
              "bh": round(_sr(oos_ret.reindex(idx_eval).to_numpy()), 3)}
    frac_up = round(float((truth == 1).mean()), 4)

    out_cfg = {}
    for cname, cf in CONFIGS.items():
        p = wf_p1_full(mv[cf["cols"]], y, cf["hl"], cf["seeds"]).loc[idx_eval]
        pos = np.where(p.to_numpy() >= 0.5, 1.0, -1.0)
        corr = (pos == truth).astype(int)
        acc = round(float(corr.mean()), 4)
        w = pd.Series(0.0, index=mv.index); w.loc[idx_eval] = pos
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx_eval).to_numpy()
        sr = _sr(nr)
        dsr = deflated_sharpe(float(nr[~np.isnan(nr)].mean() / nr[~np.isnan(nr)].std(ddof=1))
                              if nr[~np.isnan(nr)].std(ddof=1) > 0 else 0.0,
                              n_trials=len(CONFIGS), n_obs=int(np.isfinite(nr).sum()),
                              skew=float(_skew(nr[~np.isnan(nr)])), kurt=float(_kurtosis(nr[~np.isnan(nr)], fisher=False)))
        tests = {}
        for opp in ("m5", "m8", "bh"):
            _, p_mc, b, c = mcnemar_test(corr_ref[opp], corr)
            _, p_bp = block_permutation_test(corr, corr_ref[opp], seed=config.SEED)
            tests[f"vs_{opp}"] = {"mcnemar_p": round(float(p_mc), 4), "block_perm_p": round(float(p_bp), 4),
                                  "b_opp": int(b), "c_m10": int(c)}
        k_s, n_s, p_s, ci_s = sign_test(corr)
        tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4),
                            "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}
        out_cfg[cname] = {"accuracy": acc, "sharpe_causal": round(sr, 3),
                          "equity_final": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4),
                          "dsr": round(float(dsr), 4), "tests": tests,
                          "bate_todo_nominal": bool(acc > max(acc_ref["m5"], acc_ref["m8"], acc_ref["bh"]))}

    return {"n_eval": int(len(idx_eval)), "oos_span": [str(idx_eval.min().date()), str(idx_eval.max().date())],
            "frac_up": frac_up, "bh_debil": bool(acc_ref["bh"] <= 0.5),
            "acc_ref": acc_ref, "sharpe_ref": sr_ref, "configs": out_cfg}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "panel": PANEL, "N0": N0, "step": STEP,
                       "embargo": EMBARGO, "n_seeds": N_SEEDS, "configs": list(CONFIGS),
                       "scheme": "M10-WF desplegable, config FIJA a priori, todo el OOS = test; Holm-30 M10-vs-B&H",
                       "cohorte_exante": "B&H accuracy ≤ 0.5 (mal apostador direccional)",
                       "pre_registro": "BITACORA 2026-06-16"},
              "por_activo": {}}
    holm_pool = {}
    for tk in PANEL:
        try:
            r = run_asset(tk); result["por_activo"][tk] = r
            for cname, cd in r["configs"].items():
                holm_pool[f"{tk}__{cname}__vs_bh"] = cd["tests"]["vs_bh"]["mcnemar_p"]
            best = max(r["configs"].items(), key=lambda kv: kv[1]["accuracy"])
            cd = best[1]; t = cd["tests"]
            flag = "  <<<" if cd["bate_todo_nominal"] else ""
            print(f"{tk:5} {'B&Hdébil' if r['bh_debil'] else '        '} up={r['frac_up']} "
                  f"acc[M5={r['acc_ref']['m5']} M8={r['acc_ref']['m8']} BH={r['acc_ref']['bh']}] "
                  f"M10-best={best[0]}:{cd['accuracy']} (vsBH p={t['vs_bh']['mcnemar_p']} bp={t['vs_bh']['block_perm_p']} "
                  f"sign={t['vs_azar']['p']}) SR={cd['sharpe_causal']}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); result["por_activo"][tk] = {"error": repr(e)}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    result["holm_30_m10_vs_bh"] = holm

    # Caso fuerte: B&H-débil + bate a todo nominal + McNemar vs B&H bajo Holm-30 + sign vs 0.5 <0.10.
    fuerte, nominal = [], []
    for tk, r in result["por_activo"].items():
        if "error" in r:
            continue
        for cname, cd in r["configs"].items():
            if cd["bate_todo_nominal"]:
                nominal.append(f"{tk}/{cname}")
                if (r["bh_debil"] and holm.get(f"{tk}__{cname}__vs_bh", {}).get("reject")
                        and cd["tests"]["vs_azar"]["p"] < 0.10):
                    fuerte.append(f"{tk}/{cname}")
    result["caso_fuerte"] = fuerte
    result["bate_todo_nominal"] = nominal
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nBate a todo NOMINAL: {nominal or 'NINGUNO'}")
    print(f"Caso FUERTE (B&H-débil + Holm-30 vs B&H + sign vs 0.5): {fuerte or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
