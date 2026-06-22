"""Diagnóstico: ¿por qué M8 (actuar solo en contradicción) es peor que seguir el régimen siempre?

Descompone la ventana de M8 en días que INTERVIENE (RAM mismatch dispara, τ=0.5) y días que NO. En los
días que NO interviene M8 sigue al agente. Medimos, en esos días: accuracy del agente vs accuracy que
habría dado el régimen (s_dom), y en qué fracción el agente realmente COINCIDE con el régimen. Si en los
días de no-intervención el agente es peor que el régimen y no coincide tanto como se cree, esa es la
razón del gap M8 (0.50) ↔ Régimen (0.54). Uso: python experiments/m8_nonintervention_diag.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CALIBRATION_START
from core import data
import experiments.walkforward_robustez as wf
from experiments.strata_u import _regime_drift
from experiments.strata_adaptada import _adapted_final, PANEL, N0


def _complete(tk):
    import glob
    return len(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))) >= 400


def _row(tk):
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    agents = wf.load_agent(tk)
    recs = {}
    for t in sorted(agents):
        if t in gamma.index and t in sigma.index:
            g = gamma.loc[t]
            recs[t] = (agents[t].size, float(g["Calma"]), float(g["Crisis"]), float(sigma.loc[t]))
    m = pd.DataFrame.from_dict(recs, orient="index",
                               columns=["agent_size", "calm_prob", "crisis_prob", "garch_sigma"])
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    mv = m[m["r_next"].notna() & (np.sign(m["r_next"].to_numpy()) != 0)].copy()
    sel = np.zeros(len(mv), dtype=bool); sel[N0:] = True
    truth = np.sign(mv["r_next"].to_numpy())
    agent = np.sign(mv["agent_size"].to_numpy())
    s_dom = reg.reindex(mv.index)["s_dom"].to_numpy()
    fs_m8 = _adapted_final(mv, "mismatch", "absolute", 0.50, None)
    fire = (np.sign(fs_m8) != agent)        # M8 cambió la dirección del agente = intervino
    # restringir a ventana de evaluación
    truth, agent, s_dom, fire = truth[sel], agent[sel], s_dom[sel], fire[sel]
    no = ~fire
    def acc(d, mask): return float((d[mask] == truth[mask]).mean()) if mask.sum() else float("nan")
    return {
        "n": int(sel.sum()), "interv": round(float(fire.mean()), 3),
        "acc_agente_noint": round(acc(agent, no), 3),      # agente en días que M8 NO interviene
        "acc_regimen_noint": round(acc(s_dom, no), 3),     # qué habría dado el régimen esos días
        "agente_coincide_regimen_noint": round(float((agent[no] == s_dom[no]).mean()), 3),
        "acc_M8_total": round(acc(np.sign(fs_m8)[sel], np.ones_like(truth, bool)), 3),
        "acc_regimen_total": round(acc(s_dom, np.ones_like(truth, bool)), 3),
    }


def main():
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            continue
        try:
            rows[tk] = _row(tk)
        except Exception as e:  # noqa: BLE001
            print(f"{tk} ERROR {e}")
    A = list(rows)
    print(f"\n{'activo':6s}{'%noint':>8s}{'acc_ag_noint':>14s}{'acc_reg_noint':>15s}"
          f"{'coincide':>10s}{'M8tot':>8s}{'regtot':>8s}")
    for tk in A:
        r = rows[tk]
        print(f"{tk:6s}{1-r['interv']:>8.0%}{r['acc_agente_noint']:>14.3f}{r['acc_regimen_noint']:>15.3f}"
              f"{r['agente_coincide_regimen_noint']:>10.0%}{r['acc_M8_total']:>8.3f}{r['acc_regimen_total']:>8.3f}")
    avg = lambda k: round(float(np.nanmean([rows[t][k] for t in A])), 3)
    medias = {k: avg(k) for k in ("interv", "acc_agente_noint", "acc_regimen_noint",
                                  "agente_coincide_regimen_noint", "acc_M8_total", "acc_regimen_total")}
    print(f"\n{'MEDIA':6s}{1-medias['interv']:>8.0%}{medias['acc_agente_noint']:>14.3f}"
          f"{medias['acc_regimen_noint']:>15.3f}{medias['agente_coincide_regimen_noint']:>10.0%}"
          f"{medias['acc_M8_total']:>8.3f}{medias['acc_regimen_total']:>8.3f}")
    out = Path("outputs/experiments/m8_nonintervention_diag.json")
    out.write_text(json.dumps({"meta": {"activos": A, "seed": config.SEED,
        "nota": "Días que M8 (mismatch τ=0.5) NO interviene: acc del agente vs acc del régimen (s_dom) "
                "y coincidencia agente↔régimen. Ventana mv[150:]. Exploratorio (docs/)."},
        "por_activo": rows, "medias": medias}, indent=2, ensure_ascii=False))
    print(f"\nLectura: en los días que M8 NO interviene, comparar acc del agente vs acc del régimen, y "
          "cuánto coincide el agente con el régimen. Si el agente es peor y NO coincide tanto, ahí está el gap.")
    print("OK ·", out)


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
