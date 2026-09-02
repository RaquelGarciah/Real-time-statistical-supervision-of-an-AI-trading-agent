"""Arreglo del bug del signo de RAM: M8 con signo de régimen DATA-DRIVEN por activo (vs hardcode).

`strata/detectors.py:209` hardcodea Calma→long/Crisis→short (leverage effect de SPY) para los 10 activos,
lo que viola CLAUDE.md §9 y está INVERTIDO en activos individuales (MARA: Crisis sube, Calma baja). Aquí se
recomputa la dirección de M8 con el signo correcto, **data-driven y congelado en calibración** (sign del
drift medio de cada estado): long_state = argmax μ_calib → +1, short_state = argmin μ_calib → −1, intermedio
→ 0 (neutro). Sin look-ahead (el signo sale solo de calibración). M8 es regla determinista → todo el OOS es
test válido. Se reporta prior-flip (dónde el signo de calib no transfiere al OOS).

Pre-registro: BITACORA.md [2026-06-16]. Uso: python experiments/m8_datadriven_sign.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, STRATA_OOS_START
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import block_permutation_test, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
TAU = 0.5
PRIOR_FLIP_DAYS = 60
OUT = Path("outputs/experiments/m8_datadriven_sign.json")


def build(ticker: str):
    """HMM K=3 + GARCH por activo; devuelve gamma_df OOS, sigma, oos_ret y μ por estado en calibración."""
    feat_df, ret = wf.load_features(ticker)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    states_c = hmm.predict_states(calib.to_numpy())
    rc = calib["r"].to_numpy()
    mu = {s: (float(rc[states_c == s].mean()) if (states_c == s).any() else 0.0) for s in range(3)}
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    return gamma, sigma, oos_ret, mu


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret, mu = build(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    truth = np.sign(mv["r_next"].to_numpy())

    # Signo DATA-DRIVEN congelado en calibración: long_s = argmax μ (+1), short_s = argmin μ (−1), medio → 0.
    long_s, short_s = max(mu, key=mu.get), min(mu, key=mu.get)
    mu_degenerate = bool(abs(mu[long_s] - mu[short_s]) < 1e-9)          # B3: μ empatados → no evaluable
    sign_dd = {s: (1 if s == long_s else -1 if s == short_s else 0) for s in range(3)}
    sign_hard = {0: 1, 1: 0, 2: -1}            # hardcode actual: Calma→+1, Crisis→−1, Estrés→0
    invertido = bool(any(sign_dd[s] != sign_hard[s] for s in range(3)))  # B3: los 3 estados

    G = mv[["calm_prob", "stress_prob", "crisis_prob"]].to_numpy()    # estados 0=Calma,1=Estrés,2=Crisis
    agent_sign = np.sign(mv["agent_size"].to_numpy())
    pl, ps = G[:, long_s], G[:, short_s]       # masa de los estados long/short data-driven
    # M8-dd = override-C FIEL con long_s/short_s data-driven (B1): inconsistencia = masa del estado extremo
    # OPUESTO al agente; regime_sign = signo del extremo dominante (generaliza calm_prob≥crisis_prob).
    incons = np.where(agent_sign > 0, ps, np.where(agent_sign < 0, pl, 0.0))
    override = incons >= TAU
    regime_sign = np.where(pl >= ps, 1.0, -1.0)
    m8dd = np.where(override, regime_sign, agent_sign)

    pos = {"m5": agent_sign,
           "m8_orig": np.sign(mv["final_size"].to_numpy()),
           "m8_dd": m8dd,
           "bh": np.ones(len(mv))}
    corr = {k: (v == truth).astype(int) for k, v in pos.items()}
    acc = {k: round(float(c.mean()), 4) for k, c in corr.items()}

    # prior-flip: DIAGNÓSTICO de los estados EXTREMOS (long_s/short_s) sobre los primeros 60 días OOS (B2):
    # ¿el signo de μ de calibración transfiere? No es garantía sobre todo el OOS; se reporta μ_OOS_60 para juzgar.
    oos_feat_idx = mv.index[mv.index >= pd.Timestamp(STRATA_OOS_START)][:PRIOR_FLIP_DAYS]
    flip = False; mu_oos = {}
    if len(oos_feat_idx) > 5:
        sub = m.loc[oos_feat_idx]
        dom60 = sub[["calm_prob", "stress_prob", "crisis_prob"]].to_numpy().argmax(1)
        r60 = sub["r_next"].to_numpy()
        for s in range(3):
            msk = (dom60 == s) & ~np.isnan(r60)
            mu_oos[str(s)] = round(float(r60[msk].mean()), 6) if msk.any() else None
        for s in (long_s, short_s):
            msk = (dom60 == s) & ~np.isnan(r60)
            if msk.any() and abs(mu[s]) > 1e-5 and np.sign(r60[msk].mean()) != np.sign(mu[s]):
                flip = True

    tests = {}
    for opp in ("m5", "bh"):
        _, pbp = block_permutation_test(corr["m8_dd"], corr[opp])
        _, pmc, b, c = mcnemar_test(corr[opp], corr["m8_dd"])
        tests[f"vs_{opp}"] = {"blockperm_p": float(pbp), "mcnemar_p": float(pmc),
                              "b_opp": int(b), "c_m8dd": int(c)}
    _, _, p_s, _ = sign_test(corr["m8_dd"])
    tests["sign_vs_0.5_p"] = float(p_s)

    bh_oos = acc["bh"]
    return {"n": int(len(mv)), "mu_state_calib": {str(s): round(mu[s], 6) for s in range(3)},
            "mu_state_oos60": mu_oos, "mu_degenerate": mu_degenerate,
            "long_state": int(long_s), "short_state": int(short_s),
            "sign_dd": {str(s): sign_dd[s] for s in range(3)}, "invertido_vs_hardcode": invertido,
            "prior_flip": bool(flip), "bh_oos_accuracy": bh_oos, "es_bh_debil": bool(bh_oos <= 0.5),
            "accuracy": acc, "n_override": int(override.sum()), "tests": tests,
            "m8dd_bate_m5_y_bh": bool(acc["m8_dd"] > acc["m5"] and acc["m8_dd"] > acc["bh"]),
            "m8dd_mejora_orig": bool(acc["m8_dd"] > acc["m8_orig"])}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "tau": TAU, "panel": PANEL,
                       "scheme": "M8 con signo régimen DATA-DRIVEN (argmax/argmin μ calib, congelado); todo el OOS = test",
                       "fix_de": "strata/detectors.py:209 (hardcode Calma→long/Crisis→short, viola CLAUDE.md §9)",
                       "pre_registro": "BITACORA 2026-06-16"},
              "por_activo": {}}
    holm_pool = {}
    for tk in PANEL:
        try:
            r = run_ticker(tk); result["por_activo"][tk] = r
            holm_pool[f"{tk}__vs_bh"] = r["tests"]["vs_bh"]["blockperm_p"]
            a = r["accuracy"]; t = r["tests"]
            tag = "INVERTIDO" if r["invertido_vs_hardcode"] else "         "
            flip = "FLIP" if r["prior_flip"] else "    "
            f = "  <<< M8dd > M5 y B&H" if r["m8dd_bate_m5_y_bh"] else ""
            print(f"{tk:5} {tag} {flip} sign_dd={r['sign_dd']} | M5={a['m5']:.3f} M8orig={a['m8_orig']:.3f} "
                  f"M8dd={a['m8_dd']:.3f} B&H={a['bh']:.3f} | bp(vsBH)={t['vs_bh']['blockperm_p']:.3f} "
                  f"sign={t['sign_vs_0.5_p']:.2f}{f}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); result["por_activo"][tk] = {"error": repr(e)}

    holm = wf._holm_bonferroni(holm_pool, alpha=0.10)
    result["holm_vs_bh_panel"] = holm
    sostenido = []
    for tk, r in result["por_activo"].items():
        if "error" in r:
            continue
        if (r["invertido_vs_hardcode"] and not r.get("mu_degenerate") and not r["prior_flip"]
                and r["m8dd_bate_m5_y_bh"]
                and holm.get(f"{tk}__vs_bh", {}).get("reject") and r["tests"]["sign_vs_0.5_p"] < 0.10):
            sostenido.append(tk)
    result["caso_estudio_sostenido"] = sostenido
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    inv = [tk for tk, r in result["por_activo"].items() if "error" not in r and r["invertido_vs_hardcode"]]
    mej = [tk for tk, r in result["por_activo"].items() if "error" not in r and r["m8dd_mejora_orig"]]
    bate = [tk for tk, r in result["por_activo"].items() if "error" not in r and r["m8dd_bate_m5_y_bh"]]
    print(f"\nActivos con signo INVERTIDO (el fix cambia algo): {inv}")
    print(f"M8dd mejora a M8 original (accuracy): {mej}")
    print(f"M8dd bate a M5 y B&H (nominal): {bate}")
    print(f"Caso de estudio SOSTENIDO (invertido + sin flip + Holm vs B&H + sign vs 0.5): {sostenido or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
