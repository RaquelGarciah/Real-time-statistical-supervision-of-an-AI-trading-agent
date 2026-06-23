"""Anatomía de un día de intervención de STRATA en SPY: cuándo RAM corrige bien al agente y cuándo no.

Hace tangible la mecánica del override-C (M8). Para cada día del OOS reconstruye: decisión del agente (M5),
régimen HMM dominante, score RAM, lo que votó cada una de las 5 personalidades, la decisión final de M8
(intervenida o no) y la verdad r_{t+1}. Selecciona dos días ilustrativos —una intervención ACERTADA (M8 corrige
y el agente se equivocaba) y una FALLIDA (M8 corrige pero el agente tenía razón)— y agrega el balance de las 121
intervenciones. Determinista (sin H2O). Uso: python experiments/spy_intervention_anatomy.py
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

REGIME = {0: "Calma", 1: "Estrés", 2: "Crisis"}
PERS = wf.PERS  # las 5 personalidades del agente
OUT = Path("outputs/experiments/spy_intervention_anatomy.json")


def main() -> None:
    config.set_seeds(config.SEED)
    wf.TICKER = "SPY"; wf.reset_thresholds_cache()
    g, sig, oos = build_states("SPY")
    m = wf.run_master(g, sig, oos, wf.load_agent("SPY"))
    m = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()

    m5 = np.sign(m["agent_size"].to_numpy()); m8 = np.sign(m["final_size"].to_numpy())
    truth = np.sign(m["r_next"].to_numpy())
    interv = m["intervenido"].astype(bool).to_numpy() & (m8 != m5)
    m8_hit = m8 == truth; m5_hit = m5 == truth

    # --- balance de las intervenciones ---
    ni = int(interv.sum())
    agg = {"n_dias": int(len(m)), "n_intervenciones": ni,
           "tasa_intervencion": round(ni / len(m), 4),
           "acc_M8_en_intervencion": round(float(m8_hit[interv].mean()), 4),
           "acc_M5_en_intervencion": round(float(m5_hit[interv].mean()), 4),
           "intervenciones_acertadas": int(np.sum(interv & m8_hit & ~m5_hit)),
           "intervenciones_fallidas": int(np.sum(interv & ~m8_hit & m5_hit)),
           "pnl_intervenciones": round(float((m8.astype(float) * m["r_next"].to_numpy())[interv].sum()
                                             - (m5.astype(float) * m["r_next"].to_numpy())[interv].sum()), 4)}

    def _day(i: int) -> dict:
        r = m.iloc[i]
        votos = {p: int(r[f"{p}_sign"]) for p in PERS}
        return {"fecha": str(m.index[i].date()), "regimen": REGIME[int(r["regime_dom"])],
                "ram_score": round(float(r["ram_score"]), 4), "crisis_prob": round(float(r["crisis_prob"]), 4),
                "garch_sigma": round(float(r["garch_sigma"]), 4),
                "agente_M5": int(m5[i]), "votos_personalidades": votos, "STRATA_M8": int(m8[i]),
                "r_next": round(float(r["r_next"]), 4), "verdad": int(truth[i]),
                "M8_acierta": bool(m8_hit[i]), "M5_acierta": bool(m5_hit[i])}

    # --- días ilustrativos: el de mayor |r_next| en cada categoría ---
    rnext_abs = np.abs(m["r_next"].to_numpy())
    idx_ok = np.where(interv & m8_hit & ~m5_hit)[0]
    idx_ko = np.where(interv & ~m8_hit & m5_hit)[0]
    caso_acierto = _day(int(idx_ok[np.argmax(rnext_abs[idx_ok])]))
    caso_fallo = _day(int(idx_ko[np.argmax(rnext_abs[idx_ko])]))

    # --- serie compacta para el timeline del notebook ---
    serie = {"dates": [str(d.date()) for d in m.index], "regime": [int(x) for x in m["regime_dom"]],
             "intervino": [bool(x) for x in interv], "m8_hit": [bool(x) for x in m8_hit],
             "m5_pos": [int(x) for x in m5], "m8_pos": [int(x) for x in m8],
             "ram_score": [round(float(x), 4) for x in m["ram_score"]],
             "r_next": [round(float(x), 5) for x in m["r_next"]]}

    res = {"meta": {"ticker": "SPY", "n_oos": int(len(m)), "modo": "override-C (RAM)",
                    "nota": "intervención = M8 cambia el SIGNO de la decisión del agente (was_intervened y "
                            "sign(final)≠sign(agente)). Verdad = signo de r_{t+1} (signal_lag=1)."},
           "balance_intervenciones": agg, "caso_acierto": caso_acierto, "caso_fallo": caso_fallo,
           "serie": serie}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"=== SPY · balance de las {ni} intervenciones (de {len(m)} días) ===")
    print(f"  acc M8 al intervenir = {agg['acc_M8_en_intervencion']} vs M5 = {agg['acc_M5_en_intervencion']}; "
          f"acertadas={agg['intervenciones_acertadas']} fallidas={agg['intervenciones_fallidas']} "
          f"P&L rescate={agg['pnl_intervenciones']:+.4f}")
    for tag, c in (("ACIERTO", caso_acierto), ("FALLO", caso_fallo)):
        print(f"\n--- Intervención {tag} · {c['fecha']} (régimen {c['regimen']}, RAM={c['ram_score']}) ---")
        print(f"  agente M5={c['agente_M5']:+d}  votos={c['votos_personalidades']}  →  STRATA M8={c['STRATA_M8']:+d}")
        print(f"  r_next={c['r_next']:+.4f} (verdad {c['verdad']:+d}) → M8 {'acierta' if c['M8_acierta'] else 'falla'}, "
              f"M5 {'acierta' if c['M5_acierta'] else 'falla'}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
