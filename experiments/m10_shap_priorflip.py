"""¿De qué features depende M10 según el tipo de activo (leverage-effect vs prior-flip)?

Para cada activo: construye las 22 features (run_master, override-C) y la etiqueta direccional,
ajusta un XGBoost (PARAMS canónicos) sobre la ventana válida y calcula SHAP (TreeExplainer). Reporta
el peso (|SHAP| medio, normalizado a 1) de cada feature y por BLOQUES (agente / régimen / volatilidad /
psa), agregado por grupos de activos:
  - leverage-effect (régimen direccional y estable): SPY, QQQ, XLK, BAC, DIA, XLF
  - prior-flip (el signo del régimen se invierte fuera de muestra): MSTR, SMCI
  - inverso-estable (leverage inverso pero consistente; contexto): NVDA, TSLA, MARA, ROKU

CAVEAT (rigor): importancia IN-SAMPLE sobre la ventana OOS (describe en qué se APOYA el modelo, no es
claim de rendimiento); n≈250/activo y el grupo prior-flip tiene SOLO 2 activos → EXPLORATORIO, no
confirmatorio (garden of forking paths). Uso: python experiments/m10_shap_priorflip.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import config
from config import CALIBRATION_START
from core import data
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, ALL22, AGENT15, STRATA7, PARAMS

GRUPOS = {
    "leverage-effect": ["SPY", "QQQ", "XLK", "BAC", "DIA", "XLF"],
    "prior-flip": ["MSTR", "SMCI"],
    "inverso-estable": ["NVDA", "TSLA", "MARA", "ROKU"],
}
# bloques interpretables de las 22 features
BLOQUES = {
    "agente": AGENT15,
    "régimen": ["calm_prob", "stress_prob", "crisis_prob", "ram_score"],
    "volatilidad": ["garch_sigma", "gso_score"],
    "psa": ["psa_score"],
}
OUT = Path("outputs/experiments/m10_shap_priorflip.json")


def _shap_shares(tk: str) -> dict:
    """|SHAP| medio por feature (normalizado a 1) de un XGBoost sobre las 22 features del activo."""
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    X = m.loc[valid, ALL22]; y = (m.loc[valid, "r_next"] > 0).astype(int)
    clf = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(X, y)
    sv = shap.TreeExplainer(clf).shap_values(X)
    imp = np.abs(sv).mean(0)
    imp = imp / max(imp.sum(), 1e-12)
    shares = dict(zip(ALL22, imp.tolist()))
    bloque = {b: float(sum(shares[f] for f in feats)) for b, feats in BLOQUES.items()}
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:5]
    return {"n": int(valid.sum()), "shares": {k: round(v, 4) for k, v in shares.items()},
            "bloques": {k: round(v, 4) for k, v in bloque.items()},
            "top5": [(f, round(s, 4)) for f, s in top]}


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for grupo, tks in GRUPOS.items():
        for tk in tks:
            try:
                rows[tk] = {"grupo": grupo, **_shap_shares(tk)}
                t = rows[tk]
                print(f"{tk:5s} [{grupo:15s}] bloques: " +
                      " ".join(f"{b}={t['bloques'][b]:.2f}" for b in BLOQUES) +
                      " | top: " + ", ".join(f"{f}={s:.2f}" for f, s in t["top5"][:3]), flush=True)
            except Exception as ex:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)

    # medias por grupo: bloques + top features individuales
    por_grupo = {}
    for grupo, tks in GRUPOS.items():
        ok = [t for t in tks if t in rows]
        if not ok:
            continue
        blq = {b: round(float(np.mean([rows[t]["bloques"][b] for t in ok])), 4) for b in BLOQUES}
        feat_mean = {f: round(float(np.mean([rows[t]["shares"][f] for t in ok])), 4) for f in ALL22}
        top = sorted(feat_mean.items(), key=lambda kv: -kv[1])[:6]
        por_grupo[grupo] = {"activos": ok, "bloques": blq, "top6": top}

    res = {"meta": {"grupos": GRUPOS, "bloques": BLOQUES, "features": ALL22, "seed": config.SEED,
                    "nota": "SHAP in-sample (describe en qué se apoya M10, no rendimiento). n≈250/activo; "
                            "prior-flip n=2 → EXPLORATORIO, no confirmatorio. Exploratorio (docs/)."},
           "por_activo": rows, "por_grupo": por_grupo}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== PESO POR BLOQUE (|SHAP| medio normalizado), por grupo de activos ===")
    print(f"  {'grupo':16s}{'agente':>9s}{'régimen':>9s}{'volat.':>9s}{'psa':>7s}")
    for g in GRUPOS:
        if g not in por_grupo:
            continue
        b = por_grupo[g]["bloques"]
        print(f"  {g:16s}{b['agente']:>9.2f}{b['régimen']:>9.2f}{b['volatilidad']:>9.2f}{b['psa']:>7.2f}")
    print("\n=== TOP features por grupo ===")
    for g in GRUPOS:
        if g not in por_grupo:
            continue
        print(f"  {g:16s}: " + ", ".join(f"{f}={s:.2f}" for f, s in por_grupo[g]["top6"]))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
