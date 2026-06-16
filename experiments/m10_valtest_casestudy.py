"""Caso de estudio: M10 desplegable (split validación/test 60/40) que bate a M5 y B&H en accuracy.

Split cronológico del OOS: validación = primeros 60% (optimizar config + calibrar), test = últimos 40%
(INTACTO, se toca una sola vez). Desplegable y honesto: "optimizo con el pasado, opero el futuro".
Optimización SOLO en validación sobre 6 configs (capacidad × feature set), selección por accuracy de
validación a cobertura COMPLETA (la abstención no manipula la selección); abstención 30% como overlay fijo.

Hipótesis ex-ante: en activos de B&H débil (B&H≤0.5 en validación) M10 bate a M5 y B&H en test.

Pre-registro: BITACORA.md [2026-06-15]. Uso: python experiments/m10_valtest_casestudy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

import config
from core.backtest import run_backtest
from core.stats import mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
VAL_FRAC = 0.60
ABSTAIN_Q = 0.30
ANN = np.sqrt(252)

AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA_REGIME7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA_REGIME7
FEATURE_SETS = {"all22": ALL22, "regime_strata7": STRATA_REGIME7, "agent15": AGENT15}
CAPACITIES = {"cap80x3": dict(n_estimators=80, max_depth=3), "cap300x4": dict(n_estimators=300, max_depth=4)}
_BASE = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
             objective="binary:logistic", eval_metric="logloss", random_state=config.SEED, tree_method="hist")
OUT = Path("outputs/experiments/m10_valtest_casestudy.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _fit_calibrated(Xtr: pd.DataFrame, ytr: pd.Series, cap: dict):
    """Entrena XGB en sub-split fit(80%)/calib(20%) de validación; isotónica + umbral del calib.

    Devuelve también la accuracy HELD-OUT en el calib (el XGB no vio esas obs) → criterio de selección
    honesto (no in-sample, que favorecería al modelo más sobreajustado).
    """
    n = len(Xtr); c = max(40, int(n * 0.8))
    if n - c < 15:
        c = n - 15
    clf = xgb.XGBClassifier(**{**_BASE, **cap}).fit(Xtr.iloc[:c], ytr.iloc[:c])
    p_c = clf.predict_proba(Xtr.iloc[c:])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_c, ytr.iloc[c:].to_numpy())
    p_c_cal = iso.transform(p_c)
    q_abs = float(np.quantile(np.abs(p_c_cal - 0.5), ABSTAIN_Q))
    truth_c = np.where(ytr.iloc[c:].to_numpy() == 1, 1.0, -1.0)
    # Selección sobre el XGB CRUDO held-out (el modelo no vio el calib); la isotónica queda FUERA de la
    # selección para no contaminarla (se ajusta en el mismo calib y solo se usa en test/abstención).
    acc_heldout = float((np.sign(p_c - 0.5) == truth_c).mean())
    return clf, iso, q_abs, acc_heldout, int(len(p_c))


def _predict(clf, iso, X: pd.DataFrame) -> np.ndarray:
    return iso.transform(clf.predict_proba(X)[:, 1])


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    n = len(mv); split = int(n * VAL_FRAC)
    val_idx, test_idx = mv.index[:split], mv.index[split:]

    yv = (mv.loc[val_idx, "r_next"] > 0).astype(int)
    yt_truth = np.sign(mv.loc[test_idx, "r_next"].to_numpy())
    yt = (mv.loc[test_idx, "r_next"] > 0).astype(int).to_numpy()

    # --- Optimización SOLO en validación: 6 configs, selección por accuracy HELD-OUT (XGB crudo) ---
    grid = []; n_calib = None
    for cname, cap in CAPACITIES.items():
        for fname, cols in FEATURE_SETS.items():
            clf, iso, q_abs, acc_ho, n_calib = _fit_calibrated(mv.loc[val_idx, cols], yv, cap)
            grid.append({"cap": cname, "features": fname, "acc_val_heldout": round(acc_ho, 4),
                         "_clf": clf, "_iso": iso, "_q": q_abs, "_cols": cols})
    # Salvaguarda de varianza de selección (B3): SE de una proporción en el calib (~48 días).
    se_sel = float(np.sqrt(0.25 / max(n_calib, 1)))
    med = float(np.median([g["acc_val_heldout"] for g in grid]))
    best = max(grid, key=lambda g: g["acc_val_heldout"])
    seleccion_informativa = bool(best["acc_val_heldout"] - med > se_sel)
    if not seleccion_informativa:
        # Selección no concluyente → config por defecto pre-registrada (cap80x3 / all22).
        best = next(g for g in grid if g["cap"] == "cap80x3" and g["features"] == "all22")

    # --- TEST (intacto, una vez): congela la config ganadora ---
    p_test = _predict(best["_clf"], best["_iso"], mv.loc[test_idx, best["_cols"]])
    act = np.abs(p_test - 0.5) >= best["_q"]          # overlay abstención 30% (umbral de validación)
    pos = {"m5": np.sign(mv.loc[test_idx, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[test_idx, "final_size"].to_numpy()),
           "m10": np.sign(p_test - 0.5),
           "bh": np.ones(len(test_idx))}

    acc_full = {k: round(float((v == yt_truth).mean()), 4) for k, v in pos.items()}
    acc_act = {k: (round(float((v[act] == yt_truth[act]).mean()), 4) if act.sum() else None) for k, v in pos.items()}
    frac_up_test = round(float((yt_truth == 1).mean()), 4)
    frac_up_val = round(float((mv.loc[val_idx, "r_next"] > 0).mean()), 4)
    bh_val = frac_up_val                               # B&H val accuracy = % alcistas en validación

    # McNemar M10 vs M5/B&H y sign test, a cobertura completa (B&H opera 100%).
    corr = {k: (pos[k] == yt_truth).astype(int) for k in pos}
    tests = {}
    for opp in ("m5", "m8", "bh"):
        _, p, b, cc = mcnemar_test(corr[opp], corr["m10"])      # b=opp✓&m10✗, c=m10✓&opp✗
        tests[f"vs_{opp}"] = {"mcnemar_p": float(p), "b_opp": int(b), "c_m10": int(cc)}
    k_s, n_s, p_s, ci_s = sign_test(corr["m10"])
    tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": float(p_s), "ci95": [float(ci_s[0]), float(ci_s[1])]}

    auc = round(float(roc_auc_score(yt, p_test)), 4) if len(np.unique(yt)) == 2 else None
    brier = round(float(np.mean((p_test - yt) ** 2)), 4)
    pcl = np.clip(p_test, 1e-9, 1 - 1e-9)
    logloss = round(float(-np.mean(yt * np.log(pcl) + (1 - yt) * np.log(1 - pcl))), 4)
    mcc = (round(float(matthews_corrcoef(yt_truth, pos["m10"])), 4)
           if len(np.unique(pos["m10"])) > 1 and len(np.unique(yt_truth)) > 1 else None)
    w = pd.Series(0.0, index=m.index); w.loc[test_idx] = np.clip(2 * p_test - 1, -1, 1)
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(test_idx).to_numpy()

    bate_full = bool(acc_full["m10"] > acc_full["m5"] and acc_full["m10"] > acc_full["bh"])
    return {"n_val": int(split), "n_test": int(n - split), "n_calib_seleccion": int(n_calib),
            "test_span": [str(test_idx.min().date()), str(test_idx.max().date())],
            "bh_val_accuracy": round(bh_val, 4), "es_cohorte_bh_debil": bool(bh_val <= 0.5),
            "config_elegida": {"cap": best["cap"], "features": best["features"], "acc_val_heldout": best["acc_val_heldout"]},
            "seleccion_informativa": seleccion_informativa, "se_seleccion": round(se_sel, 4),
            "grid_acc_val": [{"cap": g["cap"], "features": g["features"], "acc_val_heldout": g["acc_val_heldout"]} for g in grid],
            "coverage_test": round(float(act.mean()), 3), "frac_up_test": frac_up_test, "frac_up_val": frac_up_val,
            "accuracy_completa": acc_full, "accuracy_activos": acc_act,
            "auc_m10_test": auc, "brier_m10_test": brier, "logloss_m10_test": logloss, "mcc_m10_test": mcc,
            "sharpe_m10_test_ilustrativo": round(_sr(nr), 3),
            "tests_test": tests, "bate_m5_y_bh_completa": bate_full}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "panel": PANEL, "val_frac": VAL_FRAC,
                       "scheme": "split cronológico 60/40; opt 6 configs en validación (sel. acc val full); test intacto; abstención 30% overlay",
                       "pre_registro": "BITACORA 2026-06-15"},
              "por_activo": {}}
    holm_pool = {}
    for tk in PANEL:
        try:
            r = run_ticker(tk); result["por_activo"][tk] = r
            for opp in ("m5", "bh"):
                holm_pool[f"{tk}__vs_{opp}"] = r["tests_test"][f"vs_{opp}"]["mcnemar_p"]
            af = r["accuracy_completa"]; t = r["tests_test"]
            flag = "  <<< M10 > M5 y B&H (full)" if r["bate_m5_y_bh_completa"] else ""
            print(f"{tk:5} bhVal={r['bh_val_accuracy']:.3f}{'(débil)' if r['es_cohorte_bh_debil'] else '       '} "
                  f"cfg={r['config_elegida']['cap']}/{r['config_elegida']['features']:13} | TEST n={r['n_test']:3} "
                  f"M5={af['m5']:.3f} M8={af['m8']:.3f} M10={af['m10']:.3f} B&H={af['bh']:.3f}  "
                  f"McN(vsBH)p={t['vs_bh']['mcnemar_p']:.3f} sign p={t['vs_azar']['p']:.2f}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); result["por_activo"][tk] = {"error": repr(e)}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    result["holm_test_panel"] = holm
    sostenido = []
    for tk, r in result["por_activo"].items():
        if "error" in r:
            continue
        comp_ok = abs(r["frac_up_test"] - r["frac_up_val"]) < 0.05
        auc_ok = r["auc_m10_test"] is not None and r["auc_m10_test"] > 0.5      # cubre prior-flip (B5)
        if (r["es_cohorte_bh_debil"] and r["bate_m5_y_bh_completa"]
                and holm.get(f"{tk}__vs_bh", {}).get("reject")
                and r["tests_test"]["vs_azar"]["p"] < 0.10 and comp_ok and auc_ok):
            sostenido.append(tk)
    result["caso_estudio_sostenido"] = sostenido
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    bate = [tk for tk, r in result["por_activo"].items() if r.get("bate_m5_y_bh_completa")]
    print(f"\nM10 > M5 y B&H en test (cobertura completa): {bate or 'NINGUNO'}")
    print(f"Caso de estudio SOSTENIDO (cohorte B&H-débil + Holm vs B&H + sign vs 0.5 + sin sesgo): {sostenido or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
