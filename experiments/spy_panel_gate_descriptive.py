"""Mecánica visual de STRATA: gate RAM por activo del panel + descriptivo de features en SPY. Determinista, sin H2O.

Tres piezas que en el notebook de SMCI estaban solo para SMCI y aquí se replican para el caso central (SPY) y el
panel-10:
  A. GATE RAM por activo (10): cuando RAM dispara (≥τ), ¿acierta más seguir al AGENTE o seguir al RÉGIMEN
     (override)? Más la tasa de intervención y la discrepancia agente↔régimen por activo → "donde el agente
     discrepa de un régimen acertado, STRATA interviene".
  B. DESCRIPTIVO SPY: para cada variable clave, su distribución condicionada al signo de r_{t+1} y el corte de un
     árbol depth-1 (accuracy univariante) — el "deber" que pidió el tutor, ahora sobre SPY.
  C. (la ablación del ensemble M10 ya está en detector_ablation_panel.json::ablacion_m10_spy; el notebook la pinta).

Reconstruye mv (22 features + r_next) con build_states + run_master (mismo pipeline canónico). reg_sign =
Calma≥Crisis → +1 (efecto leverage: alta vol↔bajista). Uso: python experiments/spy_panel_gate_descriptive.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import config
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import AGENT15, build_states

PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
TAU = 0.5
CAND = ["ram_score", "calm_prob", "crisis_prob", "garch_sigma", "psa_score", "gso_score", "stress_prob"] + AGENT15
OUT = Path("outputs/experiments/spy_panel_gate_descriptive.json")


def _gate(mv) -> dict:
    g_cols = mv[["calm_prob", "crisis_prob"]].to_numpy()
    reg_sign = np.where(g_cols[:, 0] >= g_cols[:, 1], 1.0, -1.0)
    agent_sign = np.sign(mv["agent_size"].to_numpy()); truth = np.sign(mv["r_next"].to_numpy())
    ram = mv["ram_score"].to_numpy(); hi = ram >= TAU
    out = {"tasa_intervencion": round(float(mv["intervenido"].astype(bool).mean()), 4),
           "discrepancia_agente_regimen": round(float((agent_sign != reg_sign).mean()), 4)}
    for lbl, msk in (("ram_bajo", ~hi), ("ram_alto", hi)):
        out[lbl] = {"n": int(msk.sum()),
                    "acc_seguir_agente": round(float((agent_sign[msk] == truth[msk]).mean()), 4) if msk.sum() else None,
                    "acc_seguir_regimen": round(float((reg_sign[msk] == truth[msk]).mean()), 4) if msk.sum() else None}
    return out


def main() -> None:
    config.set_seeds(config.SEED)
    res = {"meta": {"panel": PANEL10, "ram_tau": TAU,
                    "nota": "reg_sign: Calma≥Crisis→+1 (efecto leverage). gate: acc de seguir agente vs régimen "
                            "por nivel de RAM. descriptivo SPY: corte de árbol depth-1 por variable."},
           "gate_por_activo": {}, "descriptivo_spy": {}}

    mv_spy = None
    for tk in PANEL10:
        wf.TICKER = tk; wf.reset_thresholds_cache()
        g, sig, oos = build_states(tk)
        m = wf.run_master(g, sig, oos, wf.load_agent(tk))
        mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
        res["gate_por_activo"][tk] = _gate(mv)
        ga = res["gate_por_activo"][tk]
        print(f"{tk:5s} interv={ga['tasa_intervencion']:.0%} discrep={ga['discrepancia_agente_regimen']:.0%} | "
              f"RAM≥τ (n={ga['ram_alto']['n']}): agente={ga['ram_alto']['acc_seguir_agente']} "
              f"régimen={ga['ram_alto']['acc_seguir_regimen']}", flush=True)
        if tk == "SPY":
            mv_spy = mv

    # --- descriptivo SPY: variable vs signo r_{t+1}, corte árbol depth-1 ---
    yb = (mv_spy["r_next"] > 0).astype(int).to_numpy()
    cand = [c for c in CAND if c in mv_spy.columns][:9]
    res["descriptivo_spy"]["yb"] = [int(v) for v in yb]
    res["descriptivo_spy"]["variables"] = {}
    for col in cand:
        x = mv_spy[col].to_numpy(float)
        tr = DecisionTreeClassifier(max_depth=1, random_state=config.SEED).fit(x.reshape(-1, 1), yb)
        thr = float(tr.tree_.threshold[0]); acc1 = float((tr.predict(x.reshape(-1, 1)) == yb).mean())
        res["descriptivo_spy"]["variables"][col] = {"x": [round(float(v), 5) for v in x],
                                                    "thr": round(thr, 5) if thr > -2 else None,
                                                    "acc_univar": round(acc1, 4)}
    print("\nDescriptivo SPY (acc univariante por corte depth-1):")
    for c, d in res["descriptivo_spy"]["variables"].items():
        print(f"  {c:24s} acc={d['acc_univar']}  thr={d['thr']}")
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
