"""¿Dónde puede STRATA/M10 batir al agente? Barrido del panel: intervención de STRATA y discrepancia agente↔régimen.

Diagnóstico clave (visto en SMCI): si el agente ya está alineado con el régimen (p.ej. 95% corto en un activo
bajista), STRATA apenas interviene (M8≈M5) y M10 replica el sesgo del agente → nadie se separa. STRATA luce
donde el agente va A CONTRACORRIENTE del régimen (como en SPY: agente alcista, régimen bajista). Este barrido
mide, por activo: dirección del agente, tasa de intervención (M8≠M5), discrepancia agente↔régimen, y si M10/M8
baten al agente (accuracy + McNemar). Todo en la ventana desplegable [N0:fin].

Uso: python experiments/panel_intervention_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from core.stats import mcnemar_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import N0, wf_p1
from experiments.m10_valtest_casestudy import ALL22

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
OUT = Path("outputs/experiments/panel_intervention_scan.json")


def run_asset(tk: str) -> dict:
    wf.reset_thresholds_cache()
    g, s, o = build_states_onthefly(tk)
    m = wf.run_master(g, s, o, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    if len(mv) <= N0 + 20:
        return {"error": f"OOS corto (n={len(mv)})"}
    idx = mv.index[N0:]; sub = mv.loc[idx]
    a = np.sign(sub["agent_size"].to_numpy())          # M5
    f = np.sign(sub["final_size"].to_numpy())          # M8
    truth = np.sign(sub["r_next"].to_numpy())
    reg_dir = np.where(sub["calm_prob"].to_numpy() >= sub["crisis_prob"].to_numpy(), 1.0, -1.0)  # régimen: calma→largo / crisis→corto

    p = wf_p1(mv[ALL22], (mv["r_next"] > 0).astype(int), len(mv), N0, None, [config.SEED])
    m10 = np.where(p.reindex(idx).to_numpy() >= 0.5, 1.0, -1.0)

    acc = {"m5": round(float((a == truth).mean()), 4), "m8": round(float((f == truth).mean()), 4),
           "m10": round(float((m10 == truth).mean()), 4),
           "bh": round(float((np.ones(len(idx)) == truth).mean()), 4)}
    nonneu = a != 0
    _, p_m10_m5, b1, c1 = mcnemar_test((a == truth).astype(int), (m10 == truth).astype(int))
    _, p_m8_m5, b2, c2 = mcnemar_test((a == truth).astype(int), (f == truth).astype(int))
    return {"n_eval": int(len(idx)), "frac_up": round(float((truth > 0).mean()), 3),
            "agente_largo": round(float((a > 0).mean()), 3), "agente_corto": round(float((a < 0).mean()), 3),
            "agente_neutral": round(float((a == 0).mean()), 3),
            "intervencion_strata": round(float((a != f).mean()), 3),
            "discrepancia_agente_regimen": round(float((a[nonneu] != reg_dir[nonneu]).mean()), 3) if nonneu.any() else None,
            "accuracy": acc,
            "mcnemar_m10_vs_m5_p": round(float(p_m10_m5), 4), "m10_disc_bc": [int(b1), int(c1)],
            "mcnemar_m8_vs_m5_p": round(float(p_m8_m5), 4), "m8_disc_bc": [int(b2), int(c2)],
            "m10_bate_m5_nom": bool(acc["m10"] > acc["m5"]), "m8_bate_m5_nom": bool(acc["m8"] > acc["m5"])}


def main() -> None:
    res = {"meta": {"panel": PANEL, "N0": N0, "seed": config.SEED,
                    "regimen_dir": "calm_prob>=crisis_prob → largo (+1), si no corto (−1) [leverage effect]",
                    "nota": "STRATA luce donde el agente DISCREPA del régimen (alta discrepancia → más intervención → posible batir al agente)"},
           "por_activo": {}}
    for tk in PANEL:
        try:
            r = run_asset(tk); res["por_activo"][tk] = r
            if "error" in r:
                print(f"{tk:5} {r['error']}"); continue
            print(f"{tk:5} up={r['frac_up']:.2f} | agente: {r['agente_largo']:.0%}L/{r['agente_corto']:.0%}S "
                  f"| interv={r['intervencion_strata']:.0%} | discrep.régimen={r['discrepancia_agente_regimen']} "
                  f"| acc M5={r['accuracy']['m5']} M8={r['accuracy']['m8']} M10={r['accuracy']['m10']} "
                  f"| McN M10vsM5 p={r['mcnemar_m10_vs_m5_p']}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}"); res["por_activo"][tk] = {"error": repr(e)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    # Ranking por discrepancia agente↔régimen (donde STRATA tiene algo que corregir)
    ok = {k: v for k, v in res["por_activo"].items() if "error" not in v}
    rank = sorted(ok.items(), key=lambda kv: -(kv[1]["discrepancia_agente_regimen"] or 0))
    print("\nRanking por DISCREPANCIA agente↔régimen (candidatos a que STRATA/M10 aporte):")
    for tk, r in rank:
        print(f"  {tk:5} discrep={r['discrepancia_agente_regimen']}  interv={r['intervencion_strata']}  "
              f"M10>M5={r['m10_bate_m5_nom']} (p={r['mcnemar_m10_vs_m5_p']})  M8>M5={r['m8_bate_m5_nom']} (p={r['mcnemar_m8_vs_m5_p']})")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
