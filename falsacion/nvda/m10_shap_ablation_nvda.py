"""SHAP + ablación de M10 sobre NVDA: ¿el meta-learner usa el régimen/RAM, y aportan algo?

Cierra dos preguntas del apéndice NVDA (§A.2):
  (1) TreeSHAP pooled out-of-fold (mismo método que §11 del canónico, vía pred_contribs de XGBoost):
      ¿en qué features se apoya M10? Predicción si el régimen NO es direccional en NVDA: las features
      STRATA/régimen (incl. ram_score, que lleva embebido el prior Calma→long/Crisis→short) NO deben
      dominar; el peso debe recaer en las del agente (que son la señal perdedora) y/o repartirse plano.
  (2) Ablación con CPCV purgado: quitar ram_score —o todo el bloque de régimen— NO debe mover el acierto
      OOF. Eso confirma que el prior embebido en ram_score no es lo que hunde a M10: simplemente no hay
      señal de régimen que extraer.

Reutiliza las funciones del walk-forward (HMM_nvda recalibrado, override-C, 22 features). No toca nada
de SPY. Uso: python experiments/m10_shap_ablation_nvda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import config
from core.backtest import run_backtest
from core.cpcv import CombinatorialPurgedKFold
import experiments.walkforward_robustez as wf

wf.TICKER = "NVDA"          # override del global del módulo → build_market_states_oos usa hmm_nvda.pkl
PERS = wf.PERS
PARAMS = wf.PARAMS
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/m10_shap_ablation_nvda.json")

AGENT_COLS = [f"{nm}_{k}" for nm in PERS for k in ("sign", "size", "conf")]
STRATA_COLS = ["ram_score", "psa_score", "gso_score"]
REGIME_COLS = ["calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
FULL_COLS = AGENT_COLS + STRATA_COLS + ["calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]


def _familia(f: str) -> str:
    if f in STRATA_COLS:
        return "STRATA"
    if f in REGIME_COLS:
        return "régimen"
    return "personalidad"


def cpcv_oof_shap(Xm: pd.DataFrame, ym: pd.Series, collect_shap: bool = False):
    """Idéntico a §11 del canónico: OOF por CPCV purgado + |SHAP| pooled-OOF vía pred_contribs."""
    t1 = pd.Series(Xm.index, index=Xm.index).shift(-1).ffill()
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo=5)
    oof_sum = np.zeros(len(Xm)); oof_cnt = np.zeros(len(Xm))
    sabs = np.zeros(Xm.shape[1]); sn = 0
    for tr, te in cv.split(Xm, t1=t1):
        clf = xgb.XGBClassifier(**PARAMS)
        clf.fit(Xm.iloc[tr], ym.iloc[tr])
        oof_sum[te] += clf.predict_proba(Xm.iloc[te])[:, 1]; oof_cnt[te] += 1
        if collect_shap:
            ct = clf.get_booster().predict(
                xgb.DMatrix(Xm.iloc[te], feature_names=list(Xm.columns)), pred_contribs=True)
            sabs += np.abs(ct[:, :-1]).sum(0); sn += len(te)
    p1 = pd.Series(oof_sum / np.maximum(oof_cnt, 1), index=Xm.index)
    shap_mean = (sabs / max(sn, 1)) if collect_shap else None
    return p1, shap_mean


def main() -> None:
    gamma_df, sigma, oos_ret = wf.build_market_states_oos()
    m = wf.run_master(gamma_df, sigma, oos_ret, wf.load_agent("NVDA"))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    yb = (m.loc[valid, "r_next"] > 0).astype(int)
    truth = np.sign(m.loc[valid, "r_next"].to_numpy())

    def oof_metrics(cols: list[str]) -> dict:
        p1, _ = cpcv_oof_shap(m.loc[valid, cols], yb)
        pred = np.sign((p1 - 0.5).to_numpy())
        acc = float((pred == truth).mean())
        try:
            auc = float(roc_auc_score(yb.to_numpy(), p1.to_numpy()))
        except Exception:
            auc = float("nan")
        w = np.sign(p1 - 0.5)
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(m.index).to_numpy()
        nr = nr[~np.isnan(nr)]
        sharpe = float(nr.mean() / nr.std(ddof=1) * ANN) if nr.std(ddof=1) > 0 else 0.0
        return {"n_features": len(cols), "accuracy": round(acc, 4), "auc": round(auc, 4),
                "sharpe": round(sharpe, 3)}

    # --- (1) SHAP pooled-OOF sobre las 22 features ---
    _, shap_mean = cpcv_oof_shap(m.loc[valid, FULL_COLS], yb, collect_shap=True)
    ranking = sorted(({"feature": f, "familia": _familia(f), "mean_abs_shap": round(float(s), 5)}
                      for f, s in zip(FULL_COLS, shap_mean)), key=lambda d: -d["mean_abs_shap"])
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    fam_share = {fam: round(float(sum(r["mean_abs_shap"] for r in ranking if r["familia"] == fam)
                                  / max(sum(d["mean_abs_shap"] for d in ranking), 1e-12)), 4)
                 for fam in ("personalidad", "STRATA", "régimen")}
    ram_rank = next(r["rank"] for r in ranking if r["feature"] == "ram_score")

    # --- (2) Ablación con CPCV purgado ---
    ablation = {
        "full_22": oof_metrics(FULL_COLS),
        "sin_ram_score": oof_metrics([c for c in FULL_COLS if c != "ram_score"]),
        "sin_strata_scores": oof_metrics(AGENT_COLS + ["calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]),
        "sin_regimen_ni_strata_solo_agente": oof_metrics(AGENT_COLS),
        "solo_regimen_y_strata": oof_metrics(STRATA_COLS + ["calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]),
    }

    result = {
        "meta": {"ticker": "NVDA", "n_valid": int(valid.sum()), "n_features": len(FULL_COLS),
                 "oos": [str(oos_ret.index.min().date()), str(oos_ret.index.max().date())],
                 "seed": config.SEED, "shap": "TreeSHAP pooled out-of-fold (pred_contribs), idéntico a §11",
                 "hmm_file": wf._hmm_path("NVDA").name},
        "shap_ranking": ranking,
        "shap_familia_share": fam_share,
        "ram_score_rank": ram_rank,
        "ablation": ablation,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"n_valid={result['meta']['n_valid']}  ram_score_rank={ram_rank}/22  fam_share={fam_share}")
    print("Top-5 SHAP:", [(r["feature"], r["mean_abs_shap"]) for r in ranking[:5]])
    print("Ablación (accuracy / auc / sharpe):")
    for k, v in ablation.items():
        print(f"  {k:38} acc={v['accuracy']:.3f}  auc={v['auc']:.3f}  sharpe={v['sharpe']:+.2f}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
