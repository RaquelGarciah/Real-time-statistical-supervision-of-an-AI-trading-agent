"""STRATA-U: supervisión redefinida como batidor universal sobre los 12 activos disponibles.

Redefine la supervisión (NO es el override-C de M8). Arquitectura: núcleo estadístico
**régimen (HMM/Markov) + volatilidad (GARCH)**, con el agente como **tilt condicional gateado**.
Regla transparente, parámetros ex-ante idénticos para los 12 (sin tuning por activo). Todo causal
(signal_lag=1). Objetivos: (1) batir a M5 en los 12; (2) batir a ZeroR en el máximo de activos
(riesgo-ajustado + accuracy donde el régimen es fiable).

Dirección diaria d_t:
  - Gate de régimen CAUSAL y expansible: signo del régimen dominante = signo de la media de
    r_{t+1} de ese estado acumulada hasta t-1 (data-driven, sin look-ahead); se usa solo si la
    accuracy direccional del régimen acumulada hasta t-1 supera 0.5 y el posterior domina (≥τ).
    Esto apaga el canal solo donde el régimen deja de acertar (maneja el prior-flip).
  - Si no, drift causal (signo del retorno acumulado trailing) → sigue la tendencia del activo.
  - Tilt del agente: si el agente es fiable (hit-rate expansible > 0.5 con ≥N obs), su lado manda.
Tamaño: vol-target |w| = min(1, target_vol/σ_t). De-risk ×0.5 si nada es fiable y el régimen es ambiguo.

Uso: python experiments/strata_u.py
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
from config import CALIBRATION_END, CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
from core.stats import mcnemar_test
from core.validation import panel_pooled_test
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA"]

# --- Parámetros ex-ante, idénticos para los 12 (no se tunean al resultado) ---
TARGET_VOL = config.TARGET_VOL if hasattr(config, "TARGET_VOL") else 0.10
CAP = 1.0            # sin apalancamiento
TAU_CONF = 0.55      # confianza mínima del régimen dominante para actuar
DRIFT_L = 63         # ventana del drift causal (≈ 3 meses)
REG_REL_MIN = 0.50   # accuracy expansible mínima del régimen para fiarse
AGENT_REL_MIN = 0.50 # hit-rate expansible mínimo del agente para fiarse
AGENT_MIN_OBS = 30   # obs mínimas antes de fiarse del agente
DERISK = 0.5         # factor de de-risk cuando nada es fiable y el régimen es ambiguo
OUT = Path("outputs/experiments/strata_u_panel.json")


def _regime_drift(tk: str):
    """Series causales por fecha: signo del régimen dominante (expansible), gate de régimen, drift, γ_max."""
    feat_df, ret = wf.load_features(tk)
    gamma, sigma, oos_ret = build_states(tk)
    idx = gamma.index
    state = gamma.to_numpy().argmax(1)
    rnext = ret.shift(-1).reindex(idx).to_numpy()
    n = len(idx)
    sums = np.zeros(3); cnts = np.zeros(3); reg_ok = 0.0; reg_tot = 0.0
    s_dom = np.zeros(n); reg_gate = np.zeros(n)
    for i in range(n):
        st = int(state[i])
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        s_dom[i] = np.sign(means[st])                      # signo data-driven hasta t-1
        reg_gate[i] = reg_ok / reg_tot if reg_tot > 0 else 0.5
        if not np.isnan(rnext[i]):
            sums[st] += rnext[i]; cnts[st] += 1
            if s_dom[i] != 0:
                reg_ok += float(s_dom[i] == np.sign(rnext[i])); reg_tot += 1
    drift = pd.Series(ret.reindex(idx).to_numpy(), index=idx).rolling(DRIFT_L).sum().shift(1).to_numpy()
    return pd.DataFrame({"s_dom": s_dom, "reg_gate": reg_gate, "drift": np.sign(drift),
                         "gmax": gamma.to_numpy().max(1)}, index=idx), gamma, sigma, oos_ret


def _series(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())  # asegura OOS
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    R = reg.reindex(mv.index)
    truth = np.sign(mv["r_next"].to_numpy())
    agent_sign = np.sign(mv["agent_size"].to_numpy())
    sig = mv["garch_sigma"].to_numpy()

    # fiabilidad expansible del agente (causal)
    c_ag = (agent_sign == truth).astype(float)
    ag_gate = pd.Series(c_ag, index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    ag_obs = np.arange(len(mv))

    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy()
    drift = R["drift"].to_numpy(); gmax = R["gmax"].to_numpy()
    d = np.empty(len(mv))
    for i in range(len(mv)):
        reg_on = (reg_gate[i] > REG_REL_MIN) and (gmax[i] >= TAU_CONF) and (s_dom[i] != 0)
        base = s_dom[i] if reg_on else (drift[i] if drift[i] != 0 else 1.0)
        if (ag_obs[i] >= AGENT_MIN_OBS) and (ag_gate[i] > AGENT_REL_MIN) and (agent_sign[i] != 0):
            base = agent_sign[i]                          # agente fiable manda (supervisión condicional)
        d[i] = base if base != 0 else 1.0
    vol_scale = np.where(sig > 0, np.minimum(CAP, TARGET_VOL / sig), CAP)
    derisk = np.where((reg_gate <= REG_REL_MIN) & (ag_gate <= AGENT_REL_MIN) & (gmax < TAU_CONF), DERISK, 1.0)
    w = d * vol_scale * derisk

    # P&L causales
    def nr(pos):
        ws = pd.Series(0.0, index=mv.index); ws.loc[mv.index] = pos
        return run_backtest(oos_ret, ws, signal_lag=1)["net_return"].reindex(mv.index)
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    nr_u = run_backtest(oos_ret, pd.Series(w, index=mv.index), signal_lag=1)["net_return"].reindex(mv.index)
    nr_m5 = nr(agent_sign); nr_zr = nr(np.full_like(truth, maj)); nr_bh = nr(np.ones_like(truth))

    def stats(nrx, accpos):
        return {"acc": round(float((accpos == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
                "calmar": round(float(calmar(nrx)), 4), "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                "equity": round(float((1 + nrx.fillna(0)).prod()), 4)}
    su = stats(nr_u, np.sign(w.astype(float))); m5 = stats(nr_m5, agent_sign)
    zr = {"acc": round(max(frac_up, 1 - frac_up), 4), **{k: v for k, v in stats(nr_zr, np.full_like(truth, maj)).items() if k != "acc"}}
    bh = {"acc": round(frac_up, 4), **{k: v for k, v in stats(nr_bh, np.ones_like(truth)).items() if k != "acc"}}

    return {"dates": mv.index, "truth": truth,
            "c_u": (np.sign(w) == truth).astype(float), "c_m5": (agent_sign == truth).astype(float),
            "c_zr": (np.full_like(truth, maj) == truth).astype(float),
            "nr_u": nr_u.to_numpy(), "nr_m5": nr_m5.to_numpy(), "nr_zr": nr_zr.to_numpy(),
            "perfil": {"n": int(len(mv)), "frac_up": round(frac_up, 4),
                       "reg_on_frac": round(float(((reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF)).mean()), 4),
                       "agent_on_frac": round(float(((ag_obs >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN)).mean()), 4),
                       "strata_u": su, "m5": m5, "zeror": zr, "bh": bh}}


def main() -> None:
    wf.reset_thresholds_cache()
    por_activo = {}
    pool = {"acc_vs_m5": [], "acc_vs_zr": [], "pnl_vs_m5": [], "pnl_vs_zr": [], "dates": []}
    for tk in PANEL:
        try:
            s = _series(tk)
            por_activo[tk] = s["perfil"]
            pool["acc_vs_m5"].append(s["c_u"] - s["c_m5"]); pool["acc_vs_zr"].append(s["c_u"] - s["c_zr"])
            pool["pnl_vs_m5"].append(s["nr_u"] - s["nr_m5"]); pool["pnl_vs_zr"].append(s["nr_u"] - s["nr_zr"])
            pool["dates"].append(np.asarray(s["dates"]))
            p = s["perfil"]
            print(f"{tk:5s} acc U={p['strata_u']['acc']:.3f} M5={p['m5']['acc']:.3f} ZeroR={p['zeror']['acc']:.3f} | "
                  f"Sh U={p['strata_u']['sharpe']:+.2f} M5={p['m5']['sharpe']:+.2f} ZeroR={p['zeror']['sharpe']:+.2f} | "
                  f"reg_on={p['reg_on_frac']:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            por_activo[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)

    ok = [t for t in PANEL if "error" not in por_activo[t]]
    dates = np.concatenate(pool["dates"])
    pooled = {
        "acc_vs_m5": panel_pooled_test(np.concatenate(pool["acc_vs_m5"]), dates),
        "acc_vs_zeror": panel_pooled_test(np.concatenate(pool["acc_vs_zr"]), dates),
        "pnl_vs_m5": panel_pooled_test(np.concatenate(pool["pnl_vs_m5"]), dates),
        "pnl_vs_zeror": panel_pooled_test(np.concatenate(pool["pnl_vs_zr"]), dates)}

    def cobertura(base, metric):
        return int(sum(por_activo[t]["strata_u"][metric] > por_activo[t][base][metric] for t in ok))
    cov = {f"gana_{m}_vs_{b}": f"{cobertura(b, m)}/{len(ok)}"
           for b in ("m5", "zeror", "bh") for m in ("acc", "sharpe", "calmar")}
    # sign test 12/12 de Δacc y ΔSharpe vs M5
    from scipy.stats import binomtest
    pos_acc_m5 = sum(por_activo[t]["strata_u"]["acc"] >= por_activo[t]["m5"]["acc"] for t in ok)
    pos_sh_m5 = sum(por_activo[t]["strata_u"]["sharpe"] >= por_activo[t]["m5"]["sharpe"] for t in ok)

    res = {"meta": {"panel": PANEL, "n_activos": len(ok), "params": {
        "target_vol": TARGET_VOL, "cap": CAP, "tau_conf": TAU_CONF, "drift_L": DRIFT_L,
        "reg_rel_min": REG_REL_MIN, "agent_rel_min": AGENT_REL_MIN, "agent_min_obs": AGENT_MIN_OBS,
        "derisk": DERISK}, "seed": config.SEED, "signal_lag": 1,
        "nota": "STRATA-U: núcleo régimen(HMM)+vol(GARCH) con gate causal expansible + tilt agente; "
                "parámetros ex-ante idénticos para los 12, sin tuning por activo; exploratorio (docs/)"},
        "por_activo": por_activo, "cobertura": cov,
        "pooled": pooled,
        "obj1_m5": {"acc_ge_en": f"{pos_acc_m5}/{len(ok)}", "sharpe_ge_en": f"{pos_sh_m5}/{len(ok)}",
                    "sign_p_acc": round(float(binomtest(pos_acc_m5, len(ok), 0.5, "greater").pvalue), 4),
                    "sign_p_sharpe": round(float(binomtest(pos_sh_m5, len(ok), 0.5, "greater").pvalue), 4)},
        "veredicto": {
            "obj1_batir_m5": (pos_sh_m5 == len(ok) or pos_acc_m5 == len(ok)),
            "obj2_max_zeror_sharpe": cobertura("zeror", "sharpe"),
            "obj2_max_zeror_acc": cobertura("zeror", "acc")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    assert set(res) >= {"meta", "por_activo", "cobertura", "pooled", "obj1_m5", "veredicto"}
    print("\n=== COBERTURA (de {} activos) ===".format(len(ok)))
    for k, v in cov.items():
        print(f"  {k:22s} {v}")
    print("\n=== POOLED (clusterizado por fecha) ===")
    for k, v in pooled.items():
        print(f"  {k:14s} Δ={v['delta']:+.5f} IC95=[{v['ci_low']:+.5f},{v['ci_high']:+.5f}] "
              f"p_greater={v['p_greater']:.4f} (n={v['n_pairs']}, fechas={v['n_dates']})")
    print(f"\nOBJ1 batir M5: acc≥ en {res['obj1_m5']['acc_ge_en']} (sign p={res['obj1_m5']['sign_p_acc']}), "
          f"Sharpe≥ en {res['obj1_m5']['sharpe_ge_en']} (sign p={res['obj1_m5']['sign_p_sharpe']})")
    print(f"OBJ2 batir ZeroR: Sharpe {cov['gana_sharpe_vs_zeror']}, acc {cov['gana_acc_vs_zeror']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
