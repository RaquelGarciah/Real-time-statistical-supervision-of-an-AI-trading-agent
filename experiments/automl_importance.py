"""Importancia de features de AutoML (H2O), con el método correcto por tipo de modelo.

Para SPY (caso de estudio) y los activos donde AutoML destaca, ajusta H2O AutoML sobre la ventana válida
(in-sample, igual convención que m10_shap_priorflip → describe en qué se APOYA el modelo, no rendimiento) con
la config del panel (max_models=25, GBM/XGBoost/StackedEnsemble, AUC, Purged K-Fold) y extrae DOS lecturas:

  1. SHAP limpio: si el leader es StackedEnsemble (sin atribución exacta por feature), se coge el MEJOR
     XGBoost del leaderboard (`aml.get_best_model(algorithm="xgboost")`) y se le saca TreeSHAP
     (`predict_contributions`). Conecta con que el modelo canónico de H3 es M10-XGBoost.
  2. Permutation importance (model-agnostic) sobre el leader/ensemble: `h2o.permutation_importance`. Mide el
     efecto de cada feature REAL sobre el modelo final; sensible a features correladas (las nuestras lo están).

Ambas se agregan por bloque (agente vs régimen/vol/psa) → cuota STRATA. NOTA explícita para la memoria: un
ensemble NO admite atribución exacta por feature; por eso la interpretabilidad mecánica (SHAP) se hace sobre el
árbol individual y para el ensemble se reporta permutation importance.

Uso: python experiments/automl_importance.py [--panel SPY,MARA,UNG] [--max-models 25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core import h2o_automl as ha
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, ALL22, AGENT15, STRATA7

BLOQUES = {"agente": AGENT15,
           "régimen": ["calm_prob", "stress_prob", "crisis_prob", "ram_score"],
           "volatilidad": ["garch_sigma", "gso_score"],
           "psa": ["psa_score"]}
OUT = Path("outputs/experiments/automl_importance.json")


def _shares(imp_by_feat: dict) -> dict:
    """Normaliza a 1 y agrega por bloque + top + cuota STRATA."""
    tot = max(sum(abs(v) for v in imp_by_feat.values()), 1e-12)
    sh = {k: abs(v) / tot for k, v in imp_by_feat.items()}
    bloque = {b: round(float(sum(sh.get(f, 0.0) for f in feats)), 4) for b, feats in BLOQUES.items()}
    top = sorted(sh.items(), key=lambda kv: -kv[1])[:10]
    return {"bloques": bloque, "cuota_strata": round(float(sum(sh.get(f, 0.0) for f in STRATA7)), 4),
            "top10": [(f, round(s, 4)) for f, s in top]}


def importance_ticker(tk: str, max_models: int) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    X = m.loc[valid, ALL22]; y = (m.loc[valid, "r_next"] > 0).astype(int)

    h2o = ha._ensure_h2o()
    from h2o.automl import H2OAutoML
    keep, folds = ha.purged_kfold_fold_ids(len(X), n_splits=5, embargo=1)
    train_df = ha._to_h2o_frame(h2o, X.iloc[keep], y.iloc[keep], folds)
    aml = H2OAutoML(max_models=max_models, seed=config.SEED, sort_metric="AUC", nfolds=0,
                    include_algos=["GBM", "XGBoost", "StackedEnsemble"], verbosity=None)
    aml.train(x=ALL22, y="y", training_frame=train_df, fold_column="fold_id")
    leader = aml.leader
    leader_family = str(aml.leaderboard.as_data_frame(use_pandas=True).iloc[0]["model_id"]).split("_")[0]

    out = {"n": int(valid.sum()), "leader_family": leader_family}

    # 1) SHAP (TreeSHAP) del mejor modelo de ÁRBOL del leaderboard (XGBoost si existe; si no, GBM/DRF).
    #    En macOS H2O suele NO entrenar XGBoost → fallback a GBM (boosted trees, predict_contributions limpio).
    try:
        tree = None
        for algo in ("xgboost", "gbm", "drf"):
            try:
                tree = aml.get_best_model(algorithm=algo)
            except Exception:  # noqa: BLE001
                tree = None
            if tree is not None:
                break
        if tree is not None:
            cont = tree.predict_contributions(train_df).as_data_frame(use_pandas=True)
            cols = [c for c in cont.columns if c in ALL22]
            imp = {c: float(np.abs(cont[c].to_numpy()).mean()) for c in cols}
            out["shap_tree"] = {"modelo": str(tree.model_id), **_shares(imp)}
        else:
            out["shap_tree"] = {"warning": "ningún modelo de árbol (XGBoost/GBM/DRF) en el leaderboard"}
    except Exception as e:  # noqa: BLE001
        out["shap_tree"] = {"error": f"{type(e).__name__}: {e}"}

    # 2) Permutation importance (model-agnostic) sobre el leader/ensemble — método del modelo en H2O.
    try:
        pi = leader.permutation_importance(train_df, metric="logloss", n_samples=-1, n_repeats=5,
                                           use_pandas=True)
        pdf = pi if hasattr(pi, "columns") else pi.as_data_frame(use_pandas=True)
        # H2O (use_pandas): las VARIABLES van en el índice; columnas = Relative/Scaled/Percentage Importance.
        imp_col = next((c for c in pdf.columns if "elative" in c.lower()), pdf.columns[0])
        if any(str(v) in ALL22 for v in pdf.index):           # variables en el índice
            imp = {str(idx): float(pdf.loc[idx, imp_col]) for idx in pdf.index if str(idx) in ALL22}
        else:                                                  # fallback: variables en una columna
            vc = next((c for c in pdf.columns if "ariable" in c), pdf.columns[0])
            imp = {str(v): float(s) for v, s in zip(pdf[vc], pdf[imp_col]) if str(v) in ALL22}
        out["perm_importance_ensemble"] = {"metrica": "logloss", "n_repeats": 5, **_shares(imp)}
    except Exception as e:  # noqa: BLE001
        out["perm_importance_ensemble"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="SPY,MARA,UNG")
    ap.add_argument("--max-models", type=int, default=25)
    args = ap.parse_args()
    tickers = [t.strip() for t in args.panel.split(",") if t.strip()]

    config.set_seeds(config.SEED)
    res = {"meta": {"seed": config.SEED, "panel": tickers, "max_models": args.max_models,
                    "config": "H2O AutoML in-sample (importancia, no rendimiento); GBM/XGBoost/SE, AUC, Purged K-Fold emb=1",
                    "nota": "Ensemble sin atribución exacta por feature → SHAP sobre mejor XGBoost del leaderboard; "
                            "permutation importance sobre el ensemble (sensible a features correladas).",
                    "bloques": BLOQUES},
           "por_activo": {}}
    try:
        for tk in tickers:
            print(f"\n########## {tk} ##########", flush=True)
            try:
                r = importance_ticker(tk, args.max_models)
                res["por_activo"][tk] = r
                sx = r.get("shap_xgb_leaderboard", {}); pe = r.get("perm_importance_ensemble", {})
                print(f"  leader={r['leader_family']} | SHAP(XGB) cuotaSTRATA="
                      f"{sx.get('cuota_strata','?')} bloques={sx.get('bloques','?')}", flush=True)
                print(f"  PERM(ensemble) cuotaSTRATA={pe.get('cuota_strata','?')} bloques={pe.get('bloques','?')}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                res["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    finally:
        ha.shutdown_h2o()
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
