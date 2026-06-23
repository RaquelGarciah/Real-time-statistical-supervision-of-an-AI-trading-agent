"""SPY: variantes de intervención y sensibilidad a umbrales (robustez, NO tuning por resultado).

Dos preguntas que pidió Raquel para reforzar el caso SPY:
  A. ¿Qué pasa si, en lugar de hacer OVERRIDE (voltear al signo del régimen) en los días de intervención, STRATA
     se ABSTIENE (posición 0) o REDUCE (atenúa el tamaño)? → compara override-C (canónico) vs abstención vs
     reduce vs el agente. Muestra que el VALOR viene del override activo, no solo de "quitar" al agente.
  B. Sensibilidad a los umbrales (robustez): barrido del gate RAM τ ∈ {0.3..0.7} para M8 y del umbral del
     meta-learner p1* ∈ {0.45..0.55} para M10. Si el resultado es plano alrededor del valor canónico (τ=0.5,
     p1*=0.5), la elección ex-ante NO es un grado de libertad oculto (anti p-hacking): se reporta TODO el barrido,
     no se elige el mejor.

Ventana: la desplegable (tras burn-in N0=150), idéntica a la tabla del §3, para que las cifras sean comparables.
`signal_lag=1`, embargo=1. Uso: python experiments/spy_intervention_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core.backtest import run_backtest
from core.metrics import equity_curve, max_drawdown
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22

TICKER = "SPY"
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/spy_intervention_variants.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _metrics(oos_ret, full_idx, sub, w_sub) -> dict:
    """Métricas de una posición (alineada a sub) con signal_lag=1; accuracy sobre días con posición≠0."""
    w = pd.Series(0.0, index=full_idx); w.loc[sub] = w_sub
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).dropna()
    eq = equity_curve(nr)
    truth = np.sign(oos_ret.shift(-1).reindex(sub).to_numpy())
    pos = np.sign(np.asarray(w_sub, float)); m = pos != 0
    acc = float((pos[m] == truth[m]).mean()) if m.any() else float("nan")
    return {"accuracy": round(acc, 4), "n_pos": int(m.sum()), "frac_en_mercado": round(float(m.mean()), 3),
            "sharpe": round(_sr(nr.to_numpy()), 3), "max_dd": round(max_drawdown(eq), 4),
            "equity_final": round(float(eq.iloc[-1]), 4)}


def main() -> None:
    config.set_seeds(config.SEED)
    wf.TICKER = TICKER; wf.reset_thresholds_cache()
    gamma, sigma, oos = build_states(TICKER)
    agents = wf.load_agent(TICKER)

    # run_master canónico (τ=0.5) + ventana desplegable (sub) vía M10
    m = wf.run_master(gamma, sigma, oos, agents)
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p1 = wf_p1(mv[ALL22], y); sub = mv.index[p1.notna().to_numpy()]
    interv = mv.loc[sub, "intervenido"].astype(bool).to_numpy()
    agent = np.sign(mv.loc[sub, "agent_size"].to_numpy())
    final = np.sign(mv.loc[sub, "final_size"].to_numpy())

    # --- A. variantes de intervención (sobre sub) ---
    variantes = {
        "M5_agente": agent,
        "M8_override_C (canónico)": final,
        "M8_abstencion": np.where(interv, 0.0, agent),                 # flat en los días que intervendría
        "M8_reduce_0.5": np.where(interv, 0.5 * agent, agent),         # mismo signo, medio tamaño
    }
    res_var = {k: _metrics(oos, m.index, sub, w) for k, w in variantes.items()}

    # --- B1. sensibilidad al gate RAM τ (re-corre run_master con cada umbral) ---
    sweep_ram = []
    base = wf.RAM_THRESHOLDS
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
        wf.RAM_THRESHOLDS = (tau / 2, tau, 0.70)
        mt = wf.run_master(gamma, sigma, oos, agents)
        mvt = mt.reindex(sub)
        wsub = np.sign(mvt["final_size"].to_numpy())
        r = _metrics(oos, mt.index, sub, wsub)
        r["tau"] = tau; r["frac_interv"] = round(float(mvt["intervenido"].astype(bool).mean()), 3)
        sweep_ram.append(r)
    wf.RAM_THRESHOLDS = base

    # --- B2. sensibilidad al umbral del meta-learner p1* ---
    sweep_p1 = []
    p1s = p1.dropna().to_numpy()
    for thr in (0.45, 0.475, 0.5, 0.525, 0.55):
        wsub = np.where(p1s >= thr, 1.0, -1.0)
        r = _metrics(oos, m.index, sub, wsub); r["p1_thr"] = thr
        sweep_p1.append(r)

    out = {"meta": {"ticker": TICKER, "n_sub": int(len(sub)), "ventana": "desplegable (tras burn-in 150)",
                    "nota": "robustez/sensibilidad, NO selección por resultado; umbrales canónicos τ=0.5, p1*=0.5 "
                            "fijados ex-ante; se reporta el barrido completo (anti p-hacking)"},
           "variantes_intervencion": res_var, "sweep_ram_tau": sweep_ram, "sweep_m10_p1": sweep_p1}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("=== Variantes de intervención (SPY, ventana desplegable n=%d) ===" % len(sub))
    for k, v in res_var.items():
        print(f"  {k:26s} acc={v['accuracy']} Sharpe={v['sharpe']:+.2f} maxDD={v['max_dd']} eq={v['equity_final']} (en mercado {v['frac_en_mercado']:.0%})")
    print("\n=== Sensibilidad RAM τ (M8) ===")
    for r in sweep_ram: print(f"  τ={r['tau']}: acc={r['accuracy']} Sharpe={r['sharpe']:+.2f} interv={r['frac_interv']:.0%}")
    print("=== Sensibilidad p1* (M10) ===")
    for r in sweep_p1: print(f"  p1*={r['p1_thr']}: acc={r['accuracy']} Sharpe={r['sharpe']:+.2f}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
