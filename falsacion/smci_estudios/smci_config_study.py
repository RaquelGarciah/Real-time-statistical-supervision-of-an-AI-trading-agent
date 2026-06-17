"""Estudio de configuración sobre SMCI (caso de estudio elegido) para decidir la M10 visualmente.

Calcula, sobre SMCI, la accuracy direccional DESPLEGABLE (walk-forward expandible: burn-in N0=150,
reentreno mensual step=21, embargo=5, solo pasado) de:
  - Estrategias de referencia: M5 (agente), M8 (regla override-C), B&H (trivial siempre-largo).
  - M10-CPCV (marcado: VE EL FUTURO, no desplegable) para contraste.
  - M10 walk-forward (desplegable) bajo una rejilla de configs: capacidad × feature set × abstención/isotónica.
Vuelca JSON para que decision_activo.ipynb haga los gráficos de barras y se registre la decisión.

Uso: python experiments/smci_config_study.py
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

import config
from core.stats import block_permutation_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22, STRATA_REGIME7, AGENT15

TICKER = "SMCI"
N0, STEP, EMBARGO = 150, 21, 5
FEATURES = {"all22": ALL22, "regime7": STRATA_REGIME7, "agent15": AGENT15}
BASE = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", random_state=config.SEED, tree_method="hist")
OUT = Path("outputs/experiments/smci_config_study.json")


def wf_m10(mv, cols, n_est, depth, iso=False, abstain=0.0):
    """Walk-forward desplegable: devuelve p (prob OOS, NaN en burn-in) y máscara activa (abstención)."""
    X, y = mv[cols], (mv["r_next"] > 0).astype(int)
    n = len(mv); params = dict(BASE, n_estimators=n_est, max_depth=depth)
    p = pd.Series(np.nan, index=mv.index); active = pd.Series(True, index=mv.index)
    for start in range(N0, n, STEP):
        tr_end = start - EMBARGO
        if tr_end < 60:
            continue
        end = min(start + STEP, n); idx = mv.index[start:end]
        if iso:
            c = max(50, int(tr_end * 0.8))
            if tr_end - c < 15:
                c = tr_end - 15
            clf = xgb.XGBClassifier(**params).fit(X.iloc[:c], y.iloc[:c])
            pc = clf.predict_proba(X.iloc[c:tr_end])[:, 1]
            isor = IsotonicRegression(out_of_bounds="clip").fit(pc, y.iloc[c:tr_end].to_numpy())
            pt = isor.transform(clf.predict_proba(X.iloc[start:end])[:, 1])
            if abstain > 0:
                q = float(np.quantile(np.abs(isor.transform(pc) - 0.5), abstain))
                active.loc[idx] = np.abs(pt - 0.5) >= q
        else:
            clf = xgb.XGBClassifier(**params).fit(X.iloc[:tr_end], y.iloc[:tr_end])
            pt = clf.predict_proba(X.iloc[start:end])[:, 1]
        p.loc[idx] = pt
    return p, active


def acc_of(pred_sign, truth, mask=None):
    m = np.ones(len(truth), bool) if mask is None else mask
    return round(float((pred_sign[m] == truth[m]).mean()), 4) if m.sum() else None


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]

    # M10-WF vanilla (config elegida) para definir el tramo evaluado [N0:fin].
    p_van, _ = wf_m10(mv, ALL22, 300, 4)
    td = mv.index[p_van.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    yt = (mv.loc[td, "r_next"] > 0).astype(int).to_numpy()

    # Referencias en el MISMO tramo [N0:fin].
    refs = {"M5 (agente)": np.sign(mv.loc[td, "agent_size"].to_numpy()),
            "M8 (regla)": np.sign(mv.loc[td, "final_size"].to_numpy()),
            "B&H (trivial)": np.ones(len(td))}
    estrategias = {k: acc_of(v, truth) for k, v in refs.items()}
    estrategias["M10-WF (desplegable)"] = acc_of(np.sign(p_van.loc[td].to_numpy() - 0.5), truth)
    # M10-CPCV (VE EL FUTURO; solo contraste).
    p_cpcv = wf.cpcv_oof(mv[ALL22], (mv["r_next"] > 0).astype(int)).loc[td]
    estrategias["M10-CPCV (ve futuro)"] = acc_of(np.sign(p_cpcv.to_numpy() - 0.5), truth)

    # Rejilla de configs M10-WF desplegable (para elegir visualmente).
    grid = []
    for fname, cols in FEATURES.items():
        for (ne, dp, lab) in [(300, 4, "300x4"), (80, 3, "80x3")]:
            p, _ = wf_m10(mv, cols, ne, dp)
            grid.append({"config": f"{lab}/{fname}", "capacidad": lab, "features": fname,
                         "isotonica_abstencion": False,
                         "accuracy": acc_of(np.sign(p.loc[td].to_numpy() - 0.5), truth)})
    # Variantes con tus mejoras v3 (isotónica + abstención 30%) sobre las dos mejores capacidades, all22.
    for (ne, dp, lab) in [(300, 4, "300x4"), (80, 3, "80x3")]:
        p, act = wf_m10(mv, ALL22, ne, dp, iso=True, abstain=0.30)
        a = act.loc[td].to_numpy()
        grid.append({"config": f"{lab}/all22 +iso+abst30", "capacidad": lab, "features": "all22",
                     "isotonica_abstencion": True, "cobertura": round(float(a.mean()), 3),
                     "accuracy": acc_of(np.sign(p.loc[td].to_numpy() - 0.5), truth),
                     "accuracy_activos": acc_of(np.sign(p.loc[td].to_numpy() - 0.5), truth, a)})

    # Significancia de la config elegida (M10-WF vanilla) vs referencias.
    corr_m10 = (np.sign(p_van.loc[td].to_numpy() - 0.5) == truth).astype(int)
    tests = {}
    for k, v in refs.items():
        _, pbp = block_permutation_test(corr_m10, (v == truth).astype(int))
        tests[k] = {"blockperm_p": round(float(pbp), 4)}
    _, _, ps, _ = sign_test(corr_m10)
    tests["sign_vs_0.5_p"] = round(float(ps), 4)

    result = {"meta": {"ticker": TICKER, "n_burnin": N0, "step": STEP, "embargo": EMBARGO,
                       "n_test": int(len(td)), "test_span": [str(td.min().date()), str(td.max().date())],
                       "bh_accuracy": estrategias["B&H (trivial)"], "seed": config.SEED,
                       "config_elegida": "M10-WF 300x4 all22 (vanilla; las mejoras v3 no ayudan en SMCI)"},
              "estrategias": estrategias, "grid_configs": grid, "tests_m10wf_vanilla": tests}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"n_test={len(td)} (burn-in {N0})  B&H={estrategias['B&H (trivial)']}")
    print("Estrategias:", {k: v for k, v in estrategias.items()})
    print("Grid configs:")
    for g in grid:
        print(f"  {g['config']:24} acc={g['accuracy']}" + (f" (activos {g.get('accuracy_activos')}, cob {g.get('cobertura')})" if g["isotonica_abstencion"] else ""))
    print("Tests M10-WF vanilla:", tests)
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
