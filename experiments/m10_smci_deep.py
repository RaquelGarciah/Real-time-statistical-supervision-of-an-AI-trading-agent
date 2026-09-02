"""SMCI a fondo: ¿se puede llevar la M10 DESPLEGABLE a batir a M5/M8/B&H en accuracy de forma SIGNIFICATIVA?

Configs FIJAS a priori (no tuneadas sobre los datos de SMCI → todo el OOS es test válido, ~250 días, más
potencia que la loncha 40%). Motivación a priori de cada palanca: ensemble = reduce varianza; señal real
(momentum/vol-rel/racha) = información direccional causal; quitar las 15 del agente = la ablación mostró que
son señal perdedora; recencia = no estacionariedad de SMCI. Holm sobre las configs probadas (familia
M10-vs-B&H) controla el coste de multiplicidad de haber probado varias.

Por config: accuracy de M10/M5/M8/B&H; McNemar + block-permutation (autocorr-robusto) M10 vs M5/M8/B&H;
sign vs 0.5; Sharpe causal (lag=1), equity, Deflated Sharpe (n_trials = nº configs).

Pre-registro: BITACORA.md [2026-06-16] (pivot-scan, restringido aquí a SMCI). Uso: python experiments/m10_smci_deep.py
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
from core.stats import block_permutation_test, deflated_sharpe, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import N0, REALSIG, build_realsignal, wf_p1
from experiments.m10_valtest_casestudy import ALL22, STRATA_REGIME7

TICKER = "SMCI"
N_SEEDS = 10
ANN = np.sqrt(252)
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
S1 = [config.SEED]
# Configs fijas a priori; thr=0.5 fijo (no se tunea umbral por activo → sin sobreajuste de selección).
CONFIGS = {
    "base":        {"cols": ALL22,                 "hl": None, "seeds": S1},
    "ens":         {"cols": ALL22,                 "hl": None, "seeds": SEEDS},
    "aug":         {"cols": ALL22 + REALSIG,       "hl": None, "seeds": SEEDS},
    "strata_real": {"cols": STRATA_REGIME7 + REALSIG, "hl": None, "seeds": SEEDS},
    "aug_recency": {"cols": ALL22 + REALSIG,       "hl": 252,  "seeds": SEEDS},
}
OUT = Path("outputs/experiments/m10_smci_deep.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    _, ret_full = wf.load_features(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    mv[REALSIG] = build_realsignal(ret_full, mv.index)
    mv = mv.dropna(subset=REALSIG)
    n = len(mv); y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]                                   # días con predicción WF (burn-in 150 fuera)
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())

    ref = {"m5": np.sign(mv.loc[idx, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[idx, "final_size"].to_numpy()), "bh": np.ones(len(idx))}
    acc_ref = {k: round(float((v == truth).mean()), 4) for k, v in ref.items()}
    corr_ref = {k: (v == truth).astype(int) for k, v in ref.items()}
    sr_ref = {"m5": round(_sr(mv["nr_m5_causal"].reindex(idx).to_numpy()), 3),
              "m8": round(_sr(mv["nr_m8_causal"].reindex(idx).to_numpy()), 3),
              "bh": round(_sr(oos_ret.reindex(idx).to_numpy()), 3)}
    eq_ref = {"m5": round(float(equity_curve(mv["nr_m5_causal"].reindex(idx).dropna()).iloc[-1]), 4),
              "m8": round(float(equity_curve(mv["nr_m8_causal"].reindex(idx).dropna()).iloc[-1]), 4),
              "bh": round(float(equity_curve(oos_ret.reindex(idx).dropna()).iloc[-1]), 4)}
    frac_up = round(float((truth == 1).mean()), 4)

    out_cfg, holm_pool = {}, {}
    for cname, cf in CONFIGS.items():
        p = wf_p1(mv[cf["cols"]], y, start_hi=n, pred_lo=N0, hl=cf["hl"], seeds=cf["seeds"]).reindex(idx)
        pos = np.where(p.to_numpy() >= 0.5, 1.0, -1.0)
        corr = (pos == truth).astype(int)
        acc = round(float(corr.mean()), 4)
        w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()
        nrc = nr[~np.isnan(nr)]
        sr = _sr(nr)
        dsr = deflated_sharpe(float(nrc.mean() / nrc.std(ddof=1)) if nrc.std(ddof=1) > 0 else 0.0,
                              n_trials=len(CONFIGS), n_obs=len(nrc),
                              skew=float(_skew(nrc)), kurt=float(_kurtosis(nrc, fisher=False)))
        tests = {}
        for opp in ("m5", "m8", "bh"):
            _, p_mc, b, c = mcnemar_test(corr_ref[opp], corr)
            _, p_bp = block_permutation_test(corr, corr_ref[opp], seed=config.SEED)
            tests[f"vs_{opp}"] = {"mcnemar_p": round(float(p_mc), 4), "block_perm_p": round(float(p_bp), 4),
                                  "b_opp": int(b), "c_m10": int(c)}
        k_s, n_s, p_s, ci_s = sign_test(corr)
        tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4),
                            "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}
        holm_pool[f"{cname}__vs_bh"] = tests["vs_bh"]["mcnemar_p"]
        out_cfg[cname] = {"features": "all22" if cf["cols"] == ALL22 else ("+".join(["all22", "real"]) if cf["cols"] == ALL22 + REALSIG else "strata7+real"),
                          "recency": cf["hl"], "n_seeds": len(cf["seeds"]), "accuracy": acc,
                          "sharpe_causal": round(sr, 3),
                          "equity_final": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4),
                          "dsr": round(float(dsr), 4), "tests": tests,
                          "bate_todo_nominal": bool(acc > max(acc_ref.values()))}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    fuerte = [c for c in out_cfg if out_cfg[c]["bate_todo_nominal"]
              and holm.get(f"{c}__vs_bh", {}).get("reject") and out_cfg[c]["tests"]["vs_azar"]["p"] < 0.10]
    nominal = [c for c in out_cfg if out_cfg[c]["bate_todo_nominal"]]

    result = {"meta": {"ticker": TICKER, "seed": config.SEED, "signal_lag": 1, "n_eval": int(len(idx)),
                       "oos_span": [str(idx.min().date()), str(idx.max().date())], "frac_up": frac_up,
                       "bh_debil": bool(acc_ref["bh"] <= 0.5), "N0": N0, "n_configs": len(CONFIGS),
                       "scheme": "M10-WF desplegable, configs FIJAS a priori, todo el OOS = test; Holm sobre configs (M10-vs-B&H)",
                       "pre_registro": "BITACORA 2026-06-16"},
              "acc_ref": acc_ref, "sharpe_ref": sr_ref, "equity_ref": eq_ref,
              "configs": out_cfg, "holm_vs_bh": holm,
              "bate_todo_nominal": nominal, "caso_fuerte": fuerte}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"SMCI · OOS {idx.min().date()}→{idx.max().date()} · n_eval={len(idx)} · up={frac_up} · B&H-débil={result['meta']['bh_debil']}")
    print(f"REF  acc[M5={acc_ref['m5']} M8={acc_ref['m8']} BH={acc_ref['bh']}]  SR[M5={sr_ref['m5']} M8={sr_ref['m8']} BH={sr_ref['bh']}]")
    for c, cd in out_cfg.items():
        t = cd["tests"]; rej = holm.get(f"{c}__vs_bh", {}).get("reject")
        flag = "  <<< bate todo" + ("+SIG" if c in fuerte else "") if cd["bate_todo_nominal"] else ""
        print(f"  {c:12} acc={cd['accuracy']} SR={cd['sharpe_causal']:+.2f} eq={cd['equity_final']} "
              f"vsBH(McN p={t['vs_bh']['mcnemar_p']} bp={t['vs_bh']['block_perm_p']} Holm.rej={rej}) "
              f"vsM5 p={t['vs_m5']['mcnemar_p']} sign p={t['vs_azar']['p']} DSR={cd['dsr']}{flag}")
    print(f"\nBate a todo NOMINAL: {nominal or 'NINGUNO'}")
    print(f"Caso FUERTE (bate todo + Holm vs B&H + sign vs 0.5): {fuerte or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
