"""¿Qué hace M10 en C2 (leverage inverso) y es real? Decomposición honesta.

C2 = {UNG, MSTR, SMCI}: el régimen falla (acc<0.5). Pregunta: ¿M10 mejora por habilidad real o
solo por montar la dirección dominante (drift) + ruido? Tres diagnósticos:

1. Fracción corta de M10 y accuracy condicionada a estar de acuerdo/en contra de ZeroR (la dirección
   mayoritaria). Si solo gana cuando coincide con la mayoría → es drift, no timing.
2. McNemar M10 vs ZeroR y M10 vs M5 (por activo + pooled clusterizado por fecha) → ¿el margen es
   significativo o cabe en el ruido?
3. Importancia de features de M10 (full-fit): cuota de las 7 de STRATA/régimen vs las 15 del agente.
   Si las de régimen pesan ~0 en C2 → el edge (si lo hay) es el canal del agente.

Uso: python experiments/c2_decompose.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from config import CALIBRATION_START
from core import data
from core.stats import mcnemar_test
from core.validation import panel_pooled_test
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22, AGENT15, STRATA7, PARAMS

C2 = ["UNG", "MSTR", "SMCI"]
OUT = Path("outputs/experiments/c2_decompose.json")


def _one(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    pos10 = np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    pos5 = np.sign(mv.loc[sub, "agent_size"].to_numpy())
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    c10 = (pos10 == truth).astype(int); c5 = (pos5 == truth).astype(int)
    c_zr = (np.full_like(truth, maj) == truth).astype(int)

    # 1) descomposición vs la dirección mayoritaria (drift)
    agree = pos10 == maj
    acc_agree = float((pos10[agree] == truth[agree]).mean()) if agree.any() else float("nan")
    acc_disagree = float((pos10[~agree] == truth[~agree]).mean()) if (~agree).any() else float("nan")

    # 2) McNemar vs M5 y vs ZeroR
    _, p_m5, b5, cc5 = mcnemar_test(c5, c10)
    _, p_zr, bz, cz = mcnemar_test(c_zr, c10)

    # 3) importancia de features (full-fit) por familia
    clf = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(mv[ALL22], y)
    try:
        import shap
        sv = np.abs(shap.TreeExplainer(clf).shap_values(mv[ALL22])).mean(axis=0)
        imp = pd.Series(sv, index=ALL22); metodo = "TreeSHAP"
    except Exception:  # noqa: BLE001
        imp = pd.Series(clf.feature_importances_, index=ALL22); metodo = "XGB gain"
    cuota_strata = float(imp[STRATA7].sum() / imp.sum())
    top = imp.sort_values(ascending=False).head(6)

    return {"n": int(len(sub)), "frac_up": round(frac_up, 4), "maj": maj,
            "acc_m10": round(float(c10.mean()), 4), "acc_m5": round(float(c5.mean()), 4),
            "acc_zeror": round(float(c_zr.mean()), 4),
            "m10_frac_corto": round(float((pos10 < 0).mean()), 4),
            "m10_frac_coincide_mayoria": round(float(agree.mean()), 4),
            "acc_m10_cuando_coincide_mayoria": round(acc_agree, 4),
            "acc_m10_cuando_contraria_mayoria": round(acc_disagree, 4),
            "mcnemar_m10_vs_m5_p": round(float(p_m5), 4), "mcnemar_m10_vs_zeror_p": round(float(p_zr), 4),
            "n_dias_disagree": int((~agree).sum()),
            "shap_cuota_strata7": round(cuota_strata, 4), "shap_metodo": metodo,
            "top6_features": {k: round(float(v), 5) for k, v in top.items()},
            "_dates": np.asarray(sub), "_d_zr": (c10 - c_zr).astype(float), "_d_m5": (c10 - c5).astype(float)}


def main() -> None:
    wf.reset_thresholds_cache()
    R = {tk: _one(tk) for tk in C2}
    for tk in C2:
        r = R[tk]
        print(f"\n=== {tk} (n={r['n']}, frac_up={r['frac_up']}, mayoría={'corto' if r['maj']<0 else 'largo'}) ===")
        print(f"  acc: M10={r['acc_m10']}  M5={r['acc_m5']}  ZeroR={r['acc_zeror']}")
        print(f"  M10 corto {r['m10_frac_corto']:.0%} · coincide con la mayoría {r['m10_frac_coincide_mayoria']:.0%}")
        print(f"  acc M10 cuando coincide={r['acc_m10_cuando_coincide_mayoria']} · "
              f"cuando va a contracorriente={r['acc_m10_cuando_contraria_mayoria']} (n={r['n_dias_disagree']} días)")
        print(f"  McNemar M10 vs M5 p={r['mcnemar_m10_vs_m5_p']} · vs ZeroR p={r['mcnemar_m10_vs_zeror_p']}")
        print(f"  cuota features STRATA/régimen ({r['shap_metodo']})={r['shap_cuota_strata7']} · top: {list(r['top6_features'])}")

    dates = np.concatenate([R[tk]["_dates"] for tk in C2])
    pooled_zr = panel_pooled_test(np.concatenate([R[tk]["_d_zr"] for tk in C2]), dates)
    pooled_m5 = panel_pooled_test(np.concatenate([R[tk]["_d_m5"] for tk in C2]), dates)
    out = {"meta": {"cluster": "C2", "activos": C2, "seed": config.SEED,
                    "nota": "diagnóstico honesto: ¿M10 en C2 es habilidad o drift+ruido? n por activo ~250"},
           "por_activo": {tk: {k: v for k, v in R[tk].items() if not k.startswith("_")} for tk in C2},
           "pooled_acc_m10_vs_zeror": pooled_zr, "pooled_acc_m10_vs_m5": pooled_m5}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== POOLED C2 (clusterizado por fecha) ===")
    print(f"  M10 vs ZeroR (acc): Δ={pooled_zr['delta']:+.4f} IC95=[{pooled_zr['ci_low']:+.4f},{pooled_zr['ci_high']:+.4f}] p={pooled_zr['p_greater']:.4f}")
    print(f"  M10 vs M5   (acc): Δ={pooled_m5['delta']:+.4f} IC95=[{pooled_m5['ci_low']:+.4f},{pooled_m5['ci_high']:+.4f}] p={pooled_m5['p_greater']:.4f}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
