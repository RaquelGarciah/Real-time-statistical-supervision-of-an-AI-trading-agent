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


# Variantes de DISEÑO (ex-ante, motivadas por teoría; se reportan TODAS, no se elige a dedo):
#  - regime_flip : el régimen VOLTEA la dirección (corto en Crisis fiable). [STRATA-U v1]
#  - risk_overlay: dirección = agente fiable / drift; el régimen solo DE-RISKea en Crisis (no voltea).
#  - trend       : dirección = agente fiable / drift; sin régimen direccional (puro trend + vol-target).
MODES = ["regime_flip", "risk_overlay", "trend"]


def _series(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())  # asegura OOS
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    R = reg.reindex(mv.index)
    truth = np.sign(mv["r_next"].to_numpy())
    agent_sign = np.sign(mv["agent_size"].to_numpy())
    sig = mv["garch_sigma"].to_numpy()
    crisis = (mv["regime_dom"].to_numpy().astype(int) == 2)   # estado de mayor vol = Crisis

    c_ag = (agent_sign == truth).astype(float)
    ag_gate = pd.Series(c_ag, index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    ag_obs = np.arange(len(mv))
    agent_ok = (ag_obs >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN) & (agent_sign != 0)

    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy()
    drift = R["drift"].to_numpy(); gmax = R["gmax"].to_numpy()
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom != 0)
    drift_dir = np.where(drift != 0, drift, 1.0)
    vol_scale = np.where(sig > 0, np.minimum(CAP, TARGET_VOL / sig), CAP)

    def weights(mode: str):
        if mode == "regime_flip":
            base = np.where(reg_on, s_dom, drift_dir)
            base = np.where(agent_ok, agent_sign, base)
            fac = np.where((~reg_on) & (~agent_ok) & (gmax < TAU_CONF), DERISK, 1.0)
        else:  # risk_overlay / trend: dirección = agente fiable / drift (sigue la tendencia, como ZeroR)
            base = np.where(agent_ok, agent_sign, drift_dir)
            fac = np.where(crisis, DERISK, 1.0) if mode == "risk_overlay" else np.ones(len(mv))
        base = np.where(base == 0, 1.0, base)
        return base * vol_scale * fac

    def nr(pos):
        ws = pd.Series(0.0, index=mv.index); ws.loc[mv.index] = pos
        return run_backtest(oos_ret, ws, signal_lag=1)["net_return"].reindex(mv.index)

    def stats(nrx, accpos):
        return {"acc": round(float((accpos == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
                "calmar": round(float(calmar(nrx)), 4), "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                "equity": round(float((1 + nrx.fillna(0)).prod()), 4)}

    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    nr_m5 = nr(agent_sign); nr_zr = nr(np.full_like(truth, maj)); nr_bh = nr(np.ones_like(truth))
    m5 = stats(nr_m5, agent_sign)
    zr = {"acc": round(max(frac_up, 1 - frac_up), 4), **{k: v for k, v in stats(nr_zr, np.full_like(truth, maj)).items() if k != "acc"}}
    bh = {"acc": round(frac_up, 4), **{k: v for k, v in stats(nr_bh, np.ones_like(truth)).items() if k != "acc"}}

    out = {"dates": mv.index, "truth": truth, "agent_sign": agent_sign, "maj": maj,
           "nr_m5": nr_m5.to_numpy(), "nr_zr": nr_zr.to_numpy(), "c_m5": (agent_sign == truth).astype(float),
           "c_zr": (np.full_like(truth, maj) == truth).astype(float), "modes": {},
           "perfil_base": {"n": int(len(mv)), "frac_up": round(frac_up, 4),
                           "reg_on_frac": round(float(reg_on.mean()), 4),
                           "agent_on_frac": round(float(agent_ok.mean()), 4),
                           "m5": m5, "zeror": zr, "bh": bh}}
    for mode in MODES:
        w = weights(mode); nr_u = nr(w)
        out["modes"][mode] = {"stats": stats(nr_u, np.sign(w)),
                              "c_u": (np.sign(w) == truth).astype(float), "nr_u": nr_u.to_numpy()}
    return out


def main() -> None:
    from scipy.stats import binomtest
    wf.reset_thresholds_cache()
    S = {}
    for tk in PANEL:
        try:
            S[tk] = _series(tk)
            print(f"{tk:5s} OK (n={S[tk]['perfil_base']['n']}, reg_on={S[tk]['perfil_base']['reg_on_frac']:.2f}, "
                  f"agent_on={S[tk]['perfil_base']['agent_on_frac']:.2f})", flush=True)
        except Exception as e:  # noqa: BLE001
            S[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if "error" not in S[t]]
    dates = np.concatenate([np.asarray(S[t]["dates"]) for t in ok])

    por_modo = {}
    for mode in MODES:
        cob = {}
        for base in ("m5", "zeror", "bh"):
            for met in ("acc", "sharpe", "calmar"):
                cob[f"{met}_vs_{base}"] = int(sum(
                    S[t]["modes"][mode]["stats"][met] > S[t]["perfil_base"][base][met] for t in ok))
        pos_acc = sum(S[t]["modes"][mode]["stats"]["acc"] >= S[t]["perfil_base"]["m5"]["acc"] for t in ok)
        pos_sh = sum(S[t]["modes"][mode]["stats"]["sharpe"] >= S[t]["perfil_base"]["m5"]["sharpe"] for t in ok)
        pooled = {
            "acc_vs_m5": panel_pooled_test(np.concatenate([S[t]["modes"][mode]["c_u"] - S[t]["c_m5"] for t in ok]), dates),
            "acc_vs_zeror": panel_pooled_test(np.concatenate([S[t]["modes"][mode]["c_u"] - S[t]["c_zr"] for t in ok]), dates),
            "pnl_vs_m5": panel_pooled_test(np.concatenate([S[t]["modes"][mode]["nr_u"] - S[t]["nr_m5"] for t in ok]), dates),
            "pnl_vs_zeror": panel_pooled_test(np.concatenate([S[t]["modes"][mode]["nr_u"] - S[t]["nr_zr"] for t in ok]), dates)}
        por_modo[mode] = {
            "cobertura": {k: f"{v}/{len(ok)}" for k, v in cob.items()},
            "obj1_m5": {"acc_ge": f"{pos_acc}/{len(ok)}", "sharpe_ge": f"{pos_sh}/{len(ok)}",
                        "sign_p_acc": round(float(binomtest(pos_acc, len(ok), 0.5, "greater").pvalue), 4),
                        "sign_p_sharpe": round(float(binomtest(pos_sh, len(ok), 0.5, "greater").pvalue), 4)},
            "obj2_zeror": {"sharpe": cob["sharpe_vs_zeror"], "calmar": cob["calmar_vs_zeror"], "acc": cob["acc_vs_zeror"]},
            "pooled": pooled,
            "por_activo": {t: {"strata_u": S[t]["modes"][mode]["stats"], "m5": S[t]["perfil_base"]["m5"],
                               "zeror": S[t]["perfil_base"]["zeror"], "bh": S[t]["perfil_base"]["bh"]} for t in ok}}

    res = {"meta": {"panel": PANEL, "n_activos": len(ok), "modes": MODES, "params": {
        "target_vol": TARGET_VOL, "cap": CAP, "tau_conf": TAU_CONF, "drift_L": DRIFT_L,
        "reg_rel_min": REG_REL_MIN, "agent_rel_min": AGENT_REL_MIN, "agent_min_obs": AGENT_MIN_OBS,
        "derisk": DERISK}, "seed": config.SEED, "signal_lag": 1,
        "nota": "3 variantes de DISEÑO ex-ante (no tuning de parámetros al OOS); se reportan TODAS. "
                "Elegir la mejor variante por el conteo OOS vs ZeroR sería multiplicidad (3 pruebas): "
                "honestidad obliga a mostrarlas todas y descontar. Exploratorio (docs/)",
        "perfiles": {t: S[t]["perfil_base"] for t in ok}},
        "por_modo": por_modo}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    assert set(res) >= {"meta", "por_modo"}
    for mode in MODES:
        pm = por_modo[mode]
        print(f"\n=== {mode} ===")
        print(f"  OBJ1 batir M5: acc≥ {pm['obj1_m5']['acc_ge']} (p={pm['obj1_m5']['sign_p_acc']}), "
              f"Sharpe≥ {pm['obj1_m5']['sharpe_ge']} (p={pm['obj1_m5']['sign_p_sharpe']}) | "
              f"pooled acc Δ={pm['pooled']['acc_vs_m5']['delta']:+.4f} p={pm['pooled']['acc_vs_m5']['p_greater']:.4f}")
        print(f"  OBJ2 batir ZeroR: Sharpe {pm['obj2_zeror']['sharpe']}/{len(ok)} · "
              f"Calmar {pm['obj2_zeror']['calmar']}/{len(ok)} · acc {pm['obj2_zeror']['acc']}/{len(ok)} | "
              f"pooled Sh Δ={pm['pooled']['pnl_vs_zeror']['delta']:+.5f} p={pm['pooled']['pnl_vs_zeror']['p_greater']:.4f}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
