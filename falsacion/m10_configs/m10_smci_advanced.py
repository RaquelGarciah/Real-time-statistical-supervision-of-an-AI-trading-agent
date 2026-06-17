"""SMCI a fondo, métodos avanzados para mejorar M10 (López de Prado y reformulaciones de target/arquitectura).

Todos walk-forward expandibles desplegables, config FIJA a priori (sin tuneo por activo → todo el OOS = test,
~250 días). Ensemble 10 semillas donde aplique. Métrica primaria = accuracy a COBERTURA COMPLETA vs
sign(r_{t+1}); Sharpe/equity/DSR como enriquecimiento. Holm sobre la familia método-vs-B&H.

Métodos:
  base / ens                 — referencia (1 seed / 10 seeds), ALL22.
  triple_barrier             — etiqueta TP=+kσ / SL=−kσ / barrera temporal H (López de Prado 2018, cap. 3);
                               se entrena con esa etiqueta pero se EVALÚA contra sign(r_{t+1}). embargo=H+1.
  regime_models              — 3 XGBoost ponderados por P_estado HMM en el fit; mezcla p1 = Σ_s P_s·model_s.
  stack_agent                — ALL22 + size del agente (M5) y supervisado (M8) como features (causal).
  vote_m5_m10                — cobertura completa = M10; "activos" = días en que M5 y M10 coinciden.
  abst_regime / abst_accord  — abstención condicional al régimen / al acuerdo de las 5 personalidades.

Pre-registro: BITACORA.md [2026-06-16] (m10-smci-advanced). Uso: python experiments/m10_smci_advanced.py
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
from experiments.m10_improve_smci import N0, STEP, EMBARGO, REALSIG, build_realsignal, wf_p1
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SMCI"
N_SEEDS = 10
H_TB, K_TB, EMBARGO_TB = 5, 1.0, 6          # triple-barrier: horizonte, k·σ, embargo = H+1 (anti-look-ahead)
ANN = np.sqrt(252)
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
REGIME_COLS = ["calm_prob", "stress_prob", "crisis_prob"]
OUT = Path("outputs/experiments/m10_smci_advanced.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def wf_regime(X: pd.DataFrame, y: pd.Series, P: np.ndarray, seeds) -> pd.Series:
    """3 XGBoost ponderados por P_estado (sample_weight) en el pasado; mezcla por P_estado del día (causal)."""
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 60:
            continue
        end = min(start + STEP, n); mix = np.zeros(end - start)
        for s in range(3):
            w_s = P[:tr, s]                                  # peso = prob del estado s en cada día pasado
            acc = np.zeros(end - start)
            for sd in seeds:
                clf = xgb.XGBClassifier(**PARAMS, random_state=sd)
                clf.fit(X.iloc[:tr], y.iloc[:tr], sample_weight=w_s)
                acc += clf.predict_proba(X.iloc[start:end])[:, 1]
            mix += P[start:end, s] * (acc / len(seeds))      # mezcla por P_estado del día predicho
        p.iloc[start:end] = mix
    return p


def triple_barrier_labels(rr: np.ndarray, sigma_d: np.ndarray) -> np.ndarray:
    """Etiqueta {-1,0,+1} por TP=+K·σ / SL=−K·σ sobre el camino r_{t+1..t+H}, o signo en la barrera temporal.

    López de Prado 2018, cap. 3 (triple-barrier). rr[i]=r_i; el camino de la posición abierta en i usa
    rr[i+1..i+H]. NaN en la cola (sin H días por delante).
    """
    n = len(rr); lab = np.full(n, np.nan)
    for i in range(n - H_TB):
        thr = K_TB * sigma_d[i]; cum = 0.0; hit = 0
        for h in range(1, H_TB + 1):
            cum += rr[i + h]
            if cum >= thr:
                hit = 1; break
            if cum <= -thr:
                hit = -1; break
        lab[i] = hit if hit != 0 else (1.0 if cum > 0 else (-1.0 if cum < 0 else 0.0))
    return lab


def wf_triple_barrier(X: pd.DataFrame, lab: np.ndarray, seeds) -> pd.Series:
    """WF con etiqueta triple-barrier. embargo=H+1: la etiqueta del último día de train no alcanza el bloque."""
    n = len(X); p = pd.Series(np.nan, index=X.index)
    ytb = (lab > 0).astype(int); ok = ~np.isnan(lab) & (lab != 0)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO_TB
        if tr < 60:
            continue
        msk = ok[:tr]
        if msk.sum() < 40:
            continue
        end = min(start + STEP, n)
        Xtr, ytr = X.iloc[:tr][msk], pd.Series(ytb[:tr])[msk]
        preds = [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(Xtr, ytr)
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in seeds]
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p


def evaluate(pos: np.ndarray, idx, truth, corr_ref, acc_ref, oos_ret, mv, n_trials, active=None) -> dict:
    """Battery común: accuracy (full + activos), Sharpe/equity/DSR, McNemar/block-perm vs M5/M8/B&H, sign."""
    corr = (pos == truth).astype(int)
    acc = round(float(corr.mean()), 4)
    w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()
    nrc = nr[~np.isnan(nr)]
    sr = _sr(nr)
    dsr = deflated_sharpe(float(nrc.mean() / nrc.std(ddof=1)) if nrc.std(ddof=1) > 0 else 0.0,
                          n_trials=n_trials, n_obs=len(nrc),
                          skew=float(_skew(nrc)), kurt=float(_kurtosis(nrc, fisher=False)))
    tests = {}
    for opp in ("m5", "m8", "bh"):
        _, p_mc, b, c = mcnemar_test(corr_ref[opp], corr)
        _, p_bp = block_permutation_test(corr, corr_ref[opp], seed=config.SEED)
        tests[f"vs_{opp}"] = {"mcnemar_p": round(float(p_mc), 4), "block_perm_p": round(float(p_bp), 4)}
    k_s, n_s, p_s, ci_s = sign_test(corr)
    tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4),
                        "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}
    out = {"accuracy": acc, "sharpe_causal": round(sr, 3),
           "equity_final": round(float(equity_curve(pd.Series(nr).dropna()).iloc[-1]), 4),
           "dsr": round(float(dsr), 4), "tests": tests,
           "bate_todo_nominal": bool(acc > max(acc_ref.values()))}
    if active is not None:                                    # accuracy en días activos + cobertura (abstención)
        cov = float(active.mean())
        out["coverage"] = round(cov, 3)
        out["accuracy_activos"] = round(float((pos[active] == truth[active]).mean()), 4) if active.sum() else None
    return out


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
    idx = mv.index[N0:]
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

    # --- p1 de cada método (full OOS) ---
    P = mv[REGIME_COLS].to_numpy()
    rr = oos_ret.reindex(mv.index).to_numpy()
    sigma_d = (mv["garch_sigma"].to_numpy() / ANN)            # σ diaria desde vol anualizada GARCH
    lab_tb = triple_barrier_labels(rr, sigma_d)

    p1 = {
        "base": wf_p1(mv[ALL22], y, n, N0, None, [config.SEED]),
        "ens": wf_p1(mv[ALL22], y, n, N0, None, SEEDS),
        "triple_barrier": wf_triple_barrier(mv[ALL22], lab_tb, SEEDS),
        "regime_models": wf_regime(mv[ALL22], y, P, SEEDS),
        "stack_agent": wf_p1(mv[ALL22 + ["agent_size", "final_size"]], y, n, N0, None, SEEDS),
    }
    n_methods = len(p1) + 1                                   # +vote (mismo p1 que ens, decisión distinta)

    out_cfg, holm_pool = {}, {}
    for name, p in p1.items():
        pp = p.reindex(idx).to_numpy()
        pos = np.where(pp >= 0.5, 1.0, -1.0)
        out_cfg[name] = evaluate(pos, idx, truth, corr_ref, acc_ref, oos_ret, mv, n_methods)
        holm_pool[f"{name}__vs_bh"] = out_cfg[name]["tests"]["vs_bh"]["mcnemar_p"]

    # --- vote_m5_m10: cobertura completa = M10(ens); activos = días en que M5 y M10 coinciden ---
    pe = p1["ens"].reindex(idx).to_numpy(); pos_e = np.where(pe >= 0.5, 1.0, -1.0)
    agree = pos_e == ref["m5"]
    out_cfg["vote_m5_m10"] = evaluate(pos_e, idx, truth, corr_ref, acc_ref, oos_ret, mv, n_methods, active=agree)
    holm_pool["vote_m5_m10__vs_bh"] = out_cfg["vote_m5_m10"]["tests"]["vs_bh"]["mcnemar_p"]

    # --- abstención condicional (Prioridad 2): se reporta accuracy en activos + cobertura (no full-coverage) ---
    conf = np.abs(pe - 0.5)                                   # confianza de M10
    tau0 = float(np.quantile(conf, 0.30))                     # umbral base (30% menos confiados)
    pmax = np.maximum(P[N0:, 0], P[N0:, 2])                   # max(P_calma, P_crisis) en idx
    tau_reg = tau0 * (1 - 0.5 * pmax)                         # abstener menos si régimen decisivo
    act_reg = conf >= tau_reg
    out_cfg["abst_regime"] = evaluate(pos_e, idx, truth, corr_ref, acc_ref, oos_ret, mv, n_methods, active=act_reg)
    # acuerdo de las 5 personalidades (señal robusta → abstener menos)
    signs = mv.loc[idx, [f"{nm}_sign" for nm in wf.PERS]].to_numpy()
    accord = np.abs(signs.sum(axis=1)) / 5.0                  # 0 (discrepan) .. 1 (todas iguales)
    tau_acc = tau0 * (1 - 0.5 * accord)
    act_acc = conf >= tau_acc
    out_cfg["abst_accord"] = evaluate(pos_e, idx, truth, corr_ref, acc_ref, oos_ret, mv, n_methods, active=act_acc)

    # --- Series de retorno neto causal por brazo (para equity curves del notebook) ---
    def _nr(pos_arr):
        w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos_arr
        return run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).fillna(0.0).to_numpy()
    pos_base = np.where(p1["base"].reindex(idx).to_numpy() >= 0.5, 1.0, -1.0)
    series = {"dates": [str(d.date()) for d in idx],
              "base": [round(float(x), 6) for x in _nr(pos_base)],
              "ens": [round(float(x), 6) for x in _nr(pos_e)],
              "m5": [round(float(x), 6) for x in mv["nr_m5_causal"].reindex(idx).fillna(0.0)],
              "m8": [round(float(x), 6) for x in mv["nr_m8_causal"].reindex(idx).fillna(0.0)],
              "bh": [round(float(x), 6) for x in oos_ret.reindex(idx).fillna(0.0)]}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    fuerte = [c for c in holm_pool if (cc := out_cfg[c.replace("__vs_bh", "")])["bate_todo_nominal"]
              and holm.get(c, {}).get("reject") and cc["tests"]["vs_azar"]["p"] < 0.10]
    nominal = [c for c in out_cfg if out_cfg[c]["bate_todo_nominal"]]
    # criterio secundario de Raquel: accuracy ≥ base y Sharpe y equity mejores con DSR>0
    ab = out_cfg["base"]
    economico = [c for c in out_cfg if out_cfg[c]["accuracy"] >= ab["accuracy"]
                 and out_cfg[c]["sharpe_causal"] > ab["sharpe_causal"]
                 and out_cfg[c]["equity_final"] > ab["equity_final"] and out_cfg[c]["dsr"] > 0]

    result = {"meta": {"ticker": TICKER, "seed": config.SEED, "signal_lag": 1, "n_eval": int(len(idx)),
                       "oos_span": [str(idx.min().date()), str(idx.max().date())], "frac_up": frac_up,
                       "bh_debil": bool(acc_ref["bh"] <= 0.5), "N0": N0, "n_methods_dsr": n_methods,
                       "triple_barrier": {"H": H_TB, "k": K_TB, "embargo": EMBARGO_TB},
                       "scheme": "M10-WF desplegable, métodos avanzados fijos a priori; todo el OOS=test; Holm método-vs-B&H",
                       "pre_registro": "BITACORA 2026-06-16 (m10-smci-advanced)"},
              "acc_ref": acc_ref, "sharpe_ref": sr_ref, "equity_ref": eq_ref,
              "metodos": out_cfg, "holm_vs_bh": holm, "series": series,
              "bate_todo_nominal": nominal, "caso_fuerte": [f.replace("__vs_bh", "") for f in fuerte],
              "mejora_economica_sobre_base": economico}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"SMCI · OOS {idx.min().date()}→{idx.max().date()} · n={len(idx)} · up={frac_up} · B&H-débil={result['meta']['bh_debil']}")
    print(f"REF acc[M5={acc_ref['m5']} M8={acc_ref['m8']} BH={acc_ref['bh']}] SR[M5={sr_ref['m5']} M8={sr_ref['m8']} BH={sr_ref['bh']}] eqBH={eq_ref['bh']}")
    for c, cd in out_cfg.items():
        t = cd["tests"]; rej = holm.get(f"{c}__vs_bh", {}).get("reject")
        extra = f" cov={cd.get('coverage')} accAct={cd.get('accuracy_activos')}" if "coverage" in cd else ""
        flag = "  <<<" + (" SIG" if c in [x.replace('__vs_bh','') for x in fuerte] else " nom") if cd["bate_todo_nominal"] else ""
        print(f"  {c:14} acc={cd['accuracy']} SR={cd['sharpe_causal']:+.2f} eq={cd['equity_final']} DSR={cd['dsr']} "
              f"vsBH(p={t['vs_bh']['mcnemar_p']} bp={t['vs_bh']['block_perm_p']} Holm={rej}) sign={t['vs_azar']['p']}{extra}{flag}")
    print(f"\nBate a todo NOMINAL: {nominal or 'NINGUNO'}")
    print(f"Caso FUERTE (sig): {[f.replace('__vs_bh','') for f in fuerte] or 'NINGUNO'}")
    print(f"Mejora ECONÓMICA sobre base (acc≥, SR↑, eq↑, DSR>0): {economico or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
