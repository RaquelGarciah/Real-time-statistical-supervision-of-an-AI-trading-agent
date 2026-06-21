"""Análisis inductivo de alcance: ¿qué naturaleza de activo predice que STRATA funcione?

No impone la hipótesis 'leverage fuerte ⇒ funciona'. La INDUCE: caracteriza cada activo
del panel por su naturaleza (clase, leverage effect, persistencia y fracción de régimen,
volatilidad, sesgo del agente) y la relaciona con los resultados (accuracy de M5/M8/M10,
direccionalidad cruda del régimen, intervención de RAM, valor añadido de STRATA), buscando
patrones entre naturaleza y resultado.

Solo usa los 10 activos con decisiones de agente cacheadas (no requiere LLM). El régimen y
el master se recomputan por activo (HMM/GARCH recalibrados ≤2024-09; sin M10 walk-forward,
así que es rápido). Las accuracies de M5/M8/M10 se leen del panel auditado.

Uso: python experiments/scope_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
CLASE = {"SPY": "índice", "XLE": "ETF sector", "UNG": "ETF commodity", "NVDA": "acción", "BAC": "acción",
         "TSLA": "acción", "MSTR": "acción/cripto", "SMCI": "acción", "ROKU": "acción", "MARA": "acción/cripto"}
REGNAMES = ["Calma", "Estrés", "Crisis"]
OUT = Path("outputs/experiments/scope_analysis.json")


def nature_and_outcome(ticker: str, lev: dict, pan: dict) -> dict:
    gamma, sigma, oos_ret = build_states(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    truth = np.sign(mv["r_next"].to_numpy())

    # --- Naturaleza estructural ---
    crisis_mean = lev[ticker]["crisis_mean"]
    lev_corr = lev[ticker]["leverage_corr"]
    # Signo direccional del régimen por calibración (prior data-driven): sign(media del régimen).
    sign_prior = {k: float(np.sign(lev[ticker]["media_regimen"][nm]))
                  for k, nm in enumerate(REGNAMES)}
    dom = mv["regime_dom"].to_numpy().astype(int)         # 0/1/2 = argmax del posterior
    oos_crisis_frac = float((dom == 2).mean())
    oos_calm_frac = float((dom == 0).mean())
    oos_vol = float(mv["garch_sigma"].mean())
    agent_short_frac = float((mv["agent_size"] < 0).mean())
    agent_abs_size = float(mv["agent_size"].abs().mean())
    ram_interv = float(mv["intervenido"].mean())

    # --- Direccionalidad CRUDA del régimen (lo que RAM explota) ---
    # Posición = signo de calibración del régimen dominante del día; accuracy vs r_{t+1}.
    pos_reg = np.array([sign_prior[d] for d in dom])
    mask = pos_reg != 0
    regime_dir_acc = float((pos_reg[mask] == truth[mask]).mean()) if mask.any() else float("nan")

    # --- Resultados (del panel auditado) ---
    h = pan[ticker]["headline"]
    acc_m5, acc_m8, acc_m10 = h["accuracy_m5"], h["accuracy_m8"], h["accuracy_m10"]
    skill_p = pan[ticker]["skill_vs_luck"]["sign_test"]["p_skill_1cola"]

    return {
        "clase": CLASE[ticker],
        "naturaleza": {
            "leverage_corr": lev_corr, "crisis_mean": crisis_mean,
            "regime_sign_prior": {REGNAMES[k]: sign_prior[k] for k in range(3)},
            "oos_crisis_frac": round(oos_crisis_frac, 4), "oos_calm_frac": round(oos_calm_frac, 4),
            "oos_vol_media": round(oos_vol, 4), "agent_short_frac": round(agent_short_frac, 4),
            "agent_abs_size": round(agent_abs_size, 4)},
        "resultado": {
            "regime_dir_acc": round(regime_dir_acc, 4), "ram_interv_frac": round(ram_interv, 4),
            "acc_m5": acc_m5, "acc_m8": acc_m8, "acc_m10": acc_m10, "skill_p_1cola": skill_p,
            "strata_valor_m8_m5": round(acc_m8 - acc_m5, 4),
            "strata_valor_m10_m5": round(acc_m10 - acc_m5, 4)},
    }


def main() -> None:
    wf.reset_thresholds_cache()
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    pan = json.load(open("outputs/experiments/quant_validation_panel.json"))["por_activo"]
    res = {}
    for tk in PANEL:
        res[tk] = nature_and_outcome(tk, lev, pan)
        r = res[tk]
        print(f"{tk:5s} {r['clase']:13s} lev={r['naturaleza']['leverage_corr']:+.3f} "
              f"crisisOOS={r['naturaleza']['oos_crisis_frac']:.2f} "
              f"regAcc={r['resultado']['regime_dir_acc']:.3f} "
              f"M5={r['resultado']['acc_m5']:.3f} M8={r['resultado']['acc_m8']:.3f} "
              f"M10={r['resultado']['acc_m10']:.3f} ΔM8={r['resultado']['strata_valor_m8_m5']:+.3f}")

    # --- Patrones (correlaciones exploratorias, n=10: ilustrativas, NO significativas) ---
    df = pd.DataFrame({t: {**res[t]["naturaleza"], **res[t]["resultado"], "clase": res[t]["clase"]}
                       for t in PANEL}).T
    num = df.drop(columns=["regime_sign_prior", "clase"]).astype(float)
    pares = [("leverage_corr", "regime_dir_acc"), ("leverage_corr", "acc_m10"),
             ("leverage_corr", "strata_valor_m8_m5"), ("oos_crisis_frac", "regime_dir_acc"),
             ("oos_crisis_frac", "strata_valor_m8_m5"), ("agent_short_frac", "acc_m10"),
             ("oos_vol_media", "acc_m10"), ("regime_dir_acc", "acc_m10")]
    correls = {f"{a} ~ {b}": round(float(num[a].corr(num[b])), 3) for a, b in pares}

    # Grupos por leverage de Black (criterio limpio): fuerte = lev_corr < -0.05.
    fuerte = [t for t in PANEL if res[t]["naturaleza"]["leverage_corr"] < -0.05]
    debil = [t for t in PANEL if res[t]["naturaleza"]["leverage_corr"] >= -0.05]
    def _avg(grp, path):
        sec, key = path
        return round(float(np.mean([res[t][sec][key] for t in grp])), 4) if grp else float("nan")
    grupos = {
        "leverage_fuerte": {"tickers": fuerte,
                            "regime_dir_acc_medio": _avg(fuerte, ("resultado", "regime_dir_acc")),
                            "acc_m10_medio": _avg(fuerte, ("resultado", "acc_m10")),
                            "strata_valor_m8_m5_medio": _avg(fuerte, ("resultado", "strata_valor_m8_m5"))},
        "leverage_debil": {"tickers": debil,
                           "regime_dir_acc_medio": _avg(debil, ("resultado", "regime_dir_acc")),
                           "acc_m10_medio": _avg(debil, ("resultado", "acc_m10")),
                           "strata_valor_m8_m5_medio": _avg(debil, ("resultado", "strata_valor_m8_m5"))},
    }

    out = {"meta": {"panel": PANEL, "n": len(PANEL), "seed": config.SEED,
                    "nota": "análisis inductivo de alcance; correlaciones con n=10 son ILUSTRATIVAS, no significativas; "
                            "régimen y master recomputados por activo; accuracies M5/M8/M10 del panel auditado",
                    "regime_dir_acc": "accuracy de seguir el signo de calibración del régimen dominante vs r_{t+1}"},
           "por_activo": res, "correlaciones": correls, "grupos_por_leverage": grupos}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== correlaciones (n=10, ilustrativas) ===")
    for k, v in correls.items():
        print(f"  {k:42s} {v:+.3f}")
    print("\n=== grupos por leverage de Black ===")
    for g, d in grupos.items():
        print(f"  {g:16s} {d['tickers']}")
        print(f"      regime_dir_acc={d['regime_dir_acc_medio']} · M10={d['acc_m10_medio']} · "
              f"ΔM8(STRATA)={d['strata_valor_m8_m5_medio']}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
