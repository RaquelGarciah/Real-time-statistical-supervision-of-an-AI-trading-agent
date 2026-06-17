"""M10-v3 CAUSAL (desplegable): ¿hay un caso de estudio donde bate a todo en accuracy?

Implementa la M10-v3 (capacidad reducida + isotónica + abstención + renorm P95) pero **causal**: en cada
reentreno mensual, la isotónica y los umbrales (abstención, P95) se ajustan sobre un split de calibración
INTERNO del pasado, nunca sobre el OOS global (la versión documentada en M10_V3_GUIA.md usa estadísticas
globales = look-ahead, no desplegable).

Auditoría @rigor-matematico (B1–B5) aplicada:
- B1: Holm sobre TODOS los contrastes (10 activos × {vs M5, vs M8, vs B&H} = 30 tests).
- B2: accuracy en días activos Y a cobertura completa (sin abstención) → distinguir habilidad de "no jugar
  días difíciles".
- B3: composición direccional (% alcistas) de días activos vs global → detectar selección de días.
- B4: guarda P95 unificada. B5: AUC + Brier de v3 (la isotónica se vende como calibración).

Pre-registro: BITACORA.md [2026-06-15]. Uso: python experiments/m10_v3_causal_panel.py
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
from sklearn.metrics import roc_auc_score

import config
from config import CALIBRATION_END, STRATA_OOS_START
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.walkforward_m10_causal import FULL_COLS

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
N0, STEP, EMBARGO = 150, 21, 5
ABSTAIN_Q = 0.30
ANN = np.sqrt(252)
PARAMS_V3 = dict(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
                 eval_metric="logloss", random_state=config.SEED, tree_method="hist")
OUT = Path("outputs/experiments/m10_v3_causal_panel.json")


def _sr(a) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def build_states_onthefly(ticker: str):
    feat_df, ret = wf.load_features(ticker)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    return gamma, sigma, oos_ret


def m10_v3_causal(X: pd.DataFrame, y: pd.Series):
    """p_cal causal, máscara activa, dirección escalada P95, y traza del tamaño del set de calibración."""
    n = len(X)
    p_cal = pd.Series(np.nan, index=X.index)
    active = pd.Series(False, index=X.index)
    dir_scaled = pd.Series(0.0, index=X.index)
    calib_sizes = []
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 70:
            continue
        c = max(50, int(tr_end * 0.8))                 # fit [0:c]; calibra [c:tr_end] (pasado, leakage-free)
        if tr_end - c < 15:
            c = tr_end - 15
        calib_sizes.append(int(tr_end - c))
        clf = xgb.XGBClassifier(**PARAMS_V3).fit(X.iloc[:c], y.iloc[:c])
        p_c = clf.predict_proba(X.iloc[c:tr_end])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_c, y.iloc[c:tr_end].to_numpy())
        p_c_cal = iso.transform(p_c)
        q_abs = float(np.quantile(np.abs(p_c_cal - 0.5), ABSTAIN_Q))     # umbral abstención (pasado)
        p95 = float(np.quantile(np.abs(2 * p_c_cal - 1), 0.95))         # escala P95 (pasado)
        p95 = p95 if p95 > 1e-9 else 1.0                                 # guarda unificada (B4)
        end = min(start + STEP, n); idx = X.index[start:end]
        p_t = iso.transform(clf.predict_proba(X.iloc[start:end])[:, 1])
        p_cal.loc[idx] = p_t
        active.loc[idx] = np.abs(p_t - 0.5) >= q_abs
        dir_scaled.loc[idx] = np.clip((2 * p_t - 1) / p95, -1, 1)
    return p_cal, active, dir_scaled, calib_sizes


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    p_cal, active, dir_scaled, calib_sizes = m10_v3_causal(X, y)
    td = X.index[p_cal.notna()]
    act = active.loc[td].to_numpy()
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    yt = (mv.loc[td, "r_next"] > 0).astype(int).to_numpy()

    pos = {"m5": np.sign(mv.loc[td, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[td, "final_size"].to_numpy()),
           "m10v3": np.sign(p_cal.loc[td].to_numpy() - 0.5),
           "bh": np.ones(len(td))}

    # B2: accuracy en días ACTIVOS y a COBERTURA COMPLETA (v3 sin abstención), 4 brazos.
    acc_act = {k: (round(float((v[act] == truth[act]).mean()), 4) if act.sum() else None) for k, v in pos.items()}
    acc_full = {k: round(float((v == truth).mean()), 4) for k, v in pos.items()}

    # B3: composición direccional de días activos vs global.
    frac_up_full = round(float((truth == 1).mean()), 4)
    frac_up_active = round(float((truth[act] == 1).mean()), 4) if act.sum() else None

    # B5: AUC + Brier de v3 (calibración).
    pc = p_cal.loc[td].to_numpy()
    auc_full = round(float(roc_auc_score(yt, pc)), 4) if len(np.unique(yt)) == 2 else None
    auc_act = (round(float(roc_auc_score(yt[act], pc[act])), 4)
               if act.sum() and len(np.unique(yt[act])) == 2 else None)
    brier_full = round(float(np.mean((pc - yt) ** 2)), 4)

    # McNemar pareado en días activos vs cada brazo (b = opp✓&v3✗, c = v3✓&opp✗).
    corr = {k: (pos[k][act] == truth[act]).astype(int) for k in pos}
    tests = {}
    for opp in ("m5", "m8", "bh"):
        _, p, b, cc = mcnemar_test(corr[opp], corr["m10v3"])
        tests[f"vs_{opp}"] = {"mcnemar_p": float(p), "b_opp": int(b), "c_v3": int(cc)}
    k_s, n_s, p_s, ci_s = sign_test(corr["m10v3"])
    tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": float(p_s), "ci95": [float(ci_s[0]), float(ci_s[1])]}

    w = pd.Series(0.0, index=m.index); w.loc[td] = dir_scaled.loc[td].to_numpy()
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(td).to_numpy()

    bate_act = bool(acc_act["m10v3"] is not None
                    and acc_act["m10v3"] > max(acc_act["m5"], acc_act["m8"], acc_act["bh"]))
    bate_full = bool(acc_full["m10v3"] > max(acc_full["m5"], acc_full["m8"], acc_full["bh"]))
    return {"n_test": int(len(td)), "n_active": int(act.sum()), "coverage": round(float(act.mean()), 3),
            "calib_size_min": int(min(calib_sizes)) if calib_sizes else None,
            "calib_size_median": int(np.median(calib_sizes)) if calib_sizes else None,
            "test_span": [str(td.min().date()), str(td.max().date())],
            "frac_up_full": frac_up_full, "frac_up_active": frac_up_active,
            "accuracy_activos": acc_act, "accuracy_completa": acc_full,
            "auc_v3_full": auc_full, "auc_v3_activos": auc_act, "brier_v3_full": brier_full,
            "tests": tests, "sharpe_causal_v3_ilustrativo": round(_sr(nr), 3),
            "bate_todo_activos": bate_act, "bate_todo_completa": bate_full}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "panel": PANEL,
                       "scheme": "M10-v3 causal (XGB 80x3 + isotónica causal + abstención 30% causal + P95 causal); WF N0=150/21/embargo5",
                       "metrica": "accuracy en días ACTIVOS y a COBERTURA COMPLETA; pareada vs M5/M8/B&H; Holm sobre 30 contrastes",
                       "pre_registro": "BITACORA 2026-06-15 (enmiendas B1–B5 aplicadas)"},
              "por_activo": {}}
    holm_pool = {}                                       # B1: pool completo de 30 contrastes
    for tk in PANEL:
        try:
            r = run_ticker(tk); result["por_activo"][tk] = r
            for opp in ("m5", "m8", "bh"):
                holm_pool[f"{tk}__vs_{opp}"] = r["tests"][f"vs_{opp}"]["mcnemar_p"]
            a, af, t = r["accuracy_activos"], r["accuracy_completa"], r["tests"]
            flag = "  <<< v3 > TODO (act+full)" if (r["bate_todo_activos"] and r["bate_todo_completa"]) else ""
            print(f"{tk:5} cov={r['coverage']:.2f} up_act={r['frac_up_active']} up_full={r['frac_up_full']}  "
                  f"ACT v3={a['m10v3']} (M5{a['m5']} M8{a['m8']} BH{a['bh']}) | "
                  f"FULL v3={af['m10v3']} BH{af['bh']}  McN(vsBH)p={t['vs_bh']['mcnemar_p']:.3f} "
                  f"sign p={t['vs_azar']['p']:.2f}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); result["por_activo"][tk] = {"error": repr(e)}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    result["holm_30_contrastes"] = holm
    # Caso de estudio sostenido: bate a todo en activos Y a cobertura completa, McNemar vs B&H rechaza bajo
    # Holm-30, sign vs 0.5 < 0.10, y composición de días activos no sesgada (|up_act - up_full| < 0.05).
    sostenido = []
    for tk, r in result["por_activo"].items():
        if "error" in r:
            continue
        comp_ok = (r["frac_up_active"] is not None and abs(r["frac_up_active"] - r["frac_up_full"]) < 0.05)
        if (r["bate_todo_activos"] and r["bate_todo_completa"]
                and holm.get(f"{tk}__vs_bh", {}).get("reject")
                and r["tests"]["vs_azar"]["p"] < 0.10 and comp_ok):
            sostenido.append(tk)
    result["caso_estudio_sostenido"] = sostenido
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    bate = [tk for tk, r in result["por_activo"].items()
            if r.get("bate_todo_activos") and r.get("bate_todo_completa")]
    print(f"\nv3 > TODO en accuracy (activos Y completa): {bate or 'NINGUNO'}")
    print(f"Caso de estudio SOSTENIDO (+ Holm-30 vs B&H + sign vs 0.5 + sin sesgo de selección): {sostenido or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
