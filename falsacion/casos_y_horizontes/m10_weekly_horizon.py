"""Horizonte semanal (5 días): ¿M10 o M8 baten a M5 y B&H en accuracy direccional?

Hipótesis mecanística: el régimen HMM es persistente (multi-día) y el leverage effect opera a escala de
semanas → predecir el signo del retorno a 5 días debería tener más señal/ruido que el diario. Target
solapado diario (y5_t = signo de Σ r_{t+1..t+5}); split causal 60/40 con purga de 5 días (sin fuga del
solape); significancia por **block-permutation** (robusta a autocorrelación) + N_eff de Bartlett reportado.
Foco M10/M8 vs M5 y B&H, cohorte B&H-débil ex-ante. Config M10 FIJA (sin grid: ~80 semanas efectivas).

LIMITACIÓN: ~80 semanas efectivas → poca potencia (un positivo puede no ser significativo).
Pre-registro: BITACORA.md [2026-06-16]. Uso: python experiments/m10_weekly_horizon.py
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
from core.stats import block_permutation_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
H = 5                      # horizonte (días)
VAL_FRAC = 0.60
PURGE = 5                  # = horizonte: las etiquetas de train no solapan el test
PARAMS = dict(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
              random_state=config.SEED, tree_method="hist")
BLOCK_LEN = H + 5          # C1: ≥ horizonte de solape, fijo (no depende de √N por ticker)
OUT = Path("outputs/experiments/m10_weekly_horizon.json")

assert PURGE >= H, "La purga debe cubrir el horizonte del target (C4): si no, las etiquetas de train solapan el test."


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    r5 = oos_ret.rolling(H).sum().shift(-H).reindex(m.index)      # Σ r_{t+1..t+5} (target a 5 días)
    m = m.assign(r5=r5)
    valid = m["r5"].notna() & (np.sign(m["r5"]) != 0)
    mv = m.loc[valid]
    n = len(mv); split = int(n * VAL_FRAC)
    val_idx, test_idx = mv.index[:split], mv.index[split:]
    y = (mv["r5"] > 0).astype(int)

    # M10 causal: entrena en validación con PURGA de 5 (etiquetas no solapan test), predice test.
    tr = val_idx[:max(1, split - PURGE)]
    clf = xgb.XGBClassifier(**PARAMS).fit(mv.loc[tr, ALL22], y.loc[tr])
    p_test = clf.predict_proba(mv.loc[test_idx, ALL22])[:, 1]
    yt = y.loc[test_idx].to_numpy()
    truth = np.sign(mv.loc[test_idx, "r5"].to_numpy())

    pos = {"m5": np.sign(mv.loc[test_idx, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[test_idx, "final_size"].to_numpy()),
           "m10": np.where(p_test >= 0.5, 1.0, -1.0),               # C6: empate p=0.5 → long (no signo 0)
           "bh": np.ones(len(test_idx))}
    corr = {k: (v == truth).astype(int) for k, v in pos.items()}
    acc = {k: round(float(c.mean()), 4) for k, c in corr.items()}

    bh_val = round(float((mv.loc[val_idx, "r5"] > 0).mean()), 4)   # B&H semanal en validación (ex-ante)
    neff, rho = wf._n_eff_bartlett(corr["m10"].astype(float))      # C5: N_eff del brazo M10 (informativo)
    auc = round(float(roc_auc_score(yt, p_test)), 4) if len(np.unique(yt)) == 2 else None

    tests = {}
    for arm in ("m10", "m8"):
        for opp in ("m5", "bh"):
            _, p = block_permutation_test(corr[arm], corr[opp], block_len=BLOCK_LEN)   # stat = acc(arm)-acc(opp)
            tests[f"{arm}_vs_{opp}_blockperm_p"] = float(p)
        # C2: sign test SOLO sobre submuestra no solapada (1 de cada H) → respeta independencia
        _, _, p_s, _ = sign_test(corr[arm][::H])
        tests[f"{arm}_sign_vs_0.5_p_noverlap"] = float(p_s)

    return {"n_test": int(len(test_idx)), "n_eff_m10": round(float(neff), 1), "rho_lag1": round(float(rho), 3),
            "test_span": [str(test_idx.min().date()), str(test_idx.max().date())],
            "bh_val_5d": bh_val, "es_cohorte_bh_debil": bool(bh_val <= 0.5),
            "accuracy_5d": acc, "auc_m10": auc, "tests": tests,
            "m10_bate_m5_y_bh": bool(acc["m10"] > acc["m5"] and acc["m10"] > acc["bh"]),
            "m8_bate_m5_y_bh": bool(acc["m8"] > acc["m5"] and acc["m8"] > acc["bh"])}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "horizon_days": H, "val_frac": VAL_FRAC, "purge": PURGE,
                       "block_len": BLOCK_LEN, "panel": PANEL,
                       "metrica": "accuracy direccional a 5 días; significancia block-permutation (block_len fijo)",
                       "nota_significancia": ("Holm corrige vs B&H (benchmark duro); vs M5 es block-perm SIN Holm; "
                                              "sign test vs 0.5 SOLO sobre submuestra no solapada (1/H)."),
                       "limitacion": "~80 semanas efectivas; poca potencia → un negativo es 'no detectamos', no 'no existe'",
                       "pre_registro": "BITACORA 2026-06-16"},
              "por_activo": {}}
    holm_pool = {}
    for tk in PANEL:
        try:
            r = run_ticker(tk); result["por_activo"][tk] = r
            holm_pool[f"{tk}__m10_vs_bh"] = r["tests"]["m10_vs_bh_blockperm_p"]
            holm_pool[f"{tk}__m8_vs_bh"] = r["tests"]["m8_vs_bh_blockperm_p"]
            a = r["accuracy_5d"]; t = r["tests"]
            f10 = " M10>both" if r["m10_bate_m5_y_bh"] else ""
            f8 = " M8>both" if r["m8_bate_m5_y_bh"] else ""
            print(f"{tk:5} bhVal5d={r['bh_val_5d']:.3f}{'(débil)' if r['es_cohorte_bh_debil'] else '       '} "
                  f"TEST n={r['n_test']:3} M5={a['m5']:.3f} M8={a['m8']:.3f} M10={a['m10']:.3f} B&H={a['bh']:.3f} "
                  f"| bp(M10vBH)={t['m10_vs_bh_blockperm_p']:.3f} bp(M8vBH)={t['m8_vs_bh_blockperm_p']:.3f}{f10}{f8}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); result["por_activo"][tk] = {"error": repr(e)}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    result["holm_test_panel"] = holm
    sostenido = []
    for tk, r in result["por_activo"].items():
        if "error" in r or not r["es_cohorte_bh_debil"]:
            continue
        for arm in ("m10", "m8"):
            ok = (r[f"{arm}_bate_m5_y_bh"]
                  and holm.get(f"{tk}__{arm}_vs_bh", {}).get("reject")               # Holm-corregido vs B&H
                  and r["tests"][f"{arm}_vs_m5_blockperm_p"] < 0.10                   # C3: block-perm vs M5 (sin Holm)
                  and r["tests"][f"{arm}_sign_vs_0.5_p_noverlap"] < 0.10)             # skill vs azar (no solapado)
            if arm == "m10":
                ok = ok and (r["auc_m10"] is not None and r["auc_m10"] > 0.5)
            if ok:
                sostenido.append(f"{tk}:{arm}")
    result["caso_estudio_sostenido"] = sostenido
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    bate = [f"{tk}:{arm}" for tk, r in result["por_activo"].items() if "error" not in r
            for arm in ("m10", "m8") if r[f"{arm}_bate_m5_y_bh"]]
    print(f"\nBaten a M5 y B&H (nominal, a 5d): {bate or 'NINGUNO'}")
    print(f"Caso de estudio SOSTENIDO (cohorte + block-perm Holm vs B&H + sign vs 0.5 [+AUC>0.5 si M10]): {sostenido or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
