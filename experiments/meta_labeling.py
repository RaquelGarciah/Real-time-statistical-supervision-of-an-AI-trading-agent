"""Supervisión FIEL al agente vía meta-labeling: M8 y M_meta vs M5 / ZeroR (pooled, 12 activos).

Fiel al título del TFG ("supervisión de un agente"): el AGENTE da la dirección (lado) SIEMPRE;
la capa estadística (régimen HMM + vol GARCH) NO le cambia el signo, solo decide **si apostar
y cuánto** (act/size) — meta-labeling (López de Prado, AFML cap. 3). Se compara con M8
(override-C, lo que ya había) y con las baselines M5 (agente solo) y ZeroR.

M_meta (causal, signal_lag=1):
  side_t = signo(agent_size_t)   # el agente es el protagonista
  abstiene (act=0, flat) si: el agente NO es fiable (hit-rate expansible ≤ 0.5 con <N obs) o
    CONTRADICE a un régimen fiable (reg_on y signo del régimen ≠ lado del agente). Si no, apuesta.
  tamaño = act · vol_target/σ_t (vol-target).  NUNCA invierte la dirección del agente.

A diferencia de STRATA-U (régimen al mando), aquí el régimen solo VETA/atenúa al agente; el
agente sigue siendo el núcleo. Su mejora esperada es de RIESGO (evita las malas apuestas del
agente), no necesariamente de accuracy bruta.

Uso: python experiments/meta_labeling.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config
from config import CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
from core.validation import panel_pooled_test
import experiments.walkforward_robustez as wf
from experiments.strata_u import (_regime_drift, PANEL, TARGET_VOL, CAP, TAU_CONF,
                                   REG_REL_MIN, AGENT_REL_MIN, AGENT_MIN_OBS)

OUT = Path("outputs/experiments/meta_labeling_panel.json")


def _series(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    R = reg.reindex(mv.index)
    truth = np.sign(mv["r_next"].to_numpy())
    agent_sign = np.sign(mv["agent_size"].to_numpy())
    m8_sign = np.sign(mv["final_size"].to_numpy())          # M8 = override-C (lo que ya había)
    sig = mv["garch_sigma"].to_numpy()
    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy(); gmax = R["gmax"].to_numpy()
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom != 0)

    # fiabilidad expansible del agente (causal)
    ag_gate = pd.Series((agent_sign == truth).astype(float), index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    ag_obs = np.arange(len(mv))
    vol_scale = np.where(sig > 0, np.minimum(CAP, TARGET_VOL / sig), CAP)

    # --- M_meta: agente da el lado; veto/abstención por fiabilidad y contradicción con régimen ---
    veto = reg_on & (s_dom != agent_sign) & (agent_sign != 0)
    unreliable = (ag_obs < AGENT_MIN_OBS) | (ag_gate <= AGENT_REL_MIN)
    act = (~veto) & (~unreliable) & (agent_sign != 0)
    w_meta = np.where(act, agent_sign, 0.0) * vol_scale

    def nr(pos):
        ws = pd.Series(0.0, index=mv.index); ws.loc[mv.index] = pos
        return run_backtest(oos_ret, ws, signal_lag=1)["net_return"].reindex(mv.index)

    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    nr_m5 = nr(agent_sign); nr_m8 = mv["nr_m8_causal"].reindex(mv.index)
    nr_meta = nr(w_meta); nr_zr = nr(np.full_like(truth, maj))

    def stats(nrx, accpos):
        return {"acc": round(float((accpos == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
                "calmar": round(float(calmar(nrx)), 4), "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
                "equity": round(float((1 + nrx.fillna(0)).prod()), 4)}
    acc_bet = float((agent_sign[act] == truth[act]).mean()) if act.any() else float("nan")
    perfil = {"n": int(len(mv)), "frac_up": round(frac_up, 4), "frac_bet_meta": round(float(act.mean()), 4),
              "acc_meta_cuando_apuesta": round(acc_bet, 4),
              "m5": stats(nr_m5, agent_sign), "m8": stats(nr_m8, m8_sign),
              "meta": stats(nr_meta, np.sign(w_meta)),
              "zeror": {"acc": round(max(frac_up, 1 - frac_up), 4),
                        **{k: v for k, v in stats(nr_zr, np.full_like(truth, maj)).items() if k != "acc"}}}
    return {"dates": np.asarray(mv.index), "truth": truth,
            "c_m5": (agent_sign == truth).astype(float), "c_m8": (m8_sign == truth).astype(float),
            "c_meta": (np.sign(w_meta) == truth).astype(float),
            "c_zr": (np.full_like(truth, maj) == truth).astype(float),
            "nr_m5": nr_m5.to_numpy(), "nr_m8": nr_m8.to_numpy(), "nr_meta": nr_meta.to_numpy(),
            "nr_zr": nr_zr.to_numpy(), "perfil": perfil}


def main() -> None:
    wf.reset_thresholds_cache()
    S = {}
    for tk in PANEL:
        try:
            S[tk] = _series(tk)
            p = S[tk]["perfil"]
            print(f"{tk:5s} acc M5={p['m5']['acc']:.3f} M8={p['m8']['acc']:.3f} meta={p['meta']['acc']:.3f}"
                  f"(bet {p['frac_bet_meta']:.2f},acc_bet {p['acc_meta_cuando_apuesta']:.3f}) ZeroR={p['zeror']['acc']:.3f}"
                  f" | Sh M5={p['m5']['sharpe']:+.2f} M8={p['m8']['sharpe']:+.2f} meta={p['meta']['sharpe']:+.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            S[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if "error" not in S[t]]
    dates = np.concatenate([S[t]["dates"] for t in ok])

    def pooled(c_key, base_key):
        return panel_pooled_test(np.concatenate([S[t][c_key] - S[t][base_key] for t in ok]), dates)
    POOL = {
        "M8_vs_M5_acc": pooled("c_m8", "c_m5"), "M8_vs_M5_pnl": pooled("nr_m8", "nr_m5"),
        "Mmeta_vs_M5_acc": pooled("c_meta", "c_m5"), "Mmeta_vs_M5_pnl": pooled("nr_meta", "nr_m5"),
        "M8_vs_ZeroR_pnl": pooled("nr_m8", "nr_zr"), "Mmeta_vs_ZeroR_pnl": pooled("nr_meta", "nr_zr")}

    def cov(strat, base, met):
        return int(sum(S[t]["perfil"][strat][met] > S[t]["perfil"][base][met] for t in ok))
    def ge(strat, base, met):
        return int(sum(S[t]["perfil"][strat][met] >= S[t]["perfil"][base][met] for t in ok))
    n = len(ok)
    cobertura = {}
    for strat in ("m8", "meta"):
        for base in ("m5", "zeror"):
            for met in ("acc", "sharpe", "calmar"):
                cobertura[f"{strat}_gana_{met}_vs_{base}"] = f"{cov(strat, base, met)}/{n}"
    obj1 = {f"{strat}_vs_m5": {
        "acc_ge": f"{ge(strat,'m5','acc')}/{n}", "sharpe_ge": f"{ge(strat,'m5','sharpe')}/{n}",
        "sign_p_acc": round(float(binomtest(ge(strat,'m5','acc'), n, 0.5, 'greater').pvalue), 4),
        "sign_p_sharpe": round(float(binomtest(ge(strat,'m5','sharpe'), n, 0.5, 'greater').pvalue), 4)}
        for strat in ("m8", "meta")}

    res = {"meta": {"panel": PANEL, "n_activos": n, "seed": config.SEED, "signal_lag": 1,
                    "params": {"target_vol": TARGET_VOL, "tau_conf": TAU_CONF, "agent_min_obs": AGENT_MIN_OBS,
                               "agent_rel_min": AGENT_REL_MIN, "reg_rel_min": REG_REL_MIN},
                    "nota": "supervisión FIEL (agente da el lado siempre): M8 (override-C) y M_meta "
                            "(meta-labeling: régimen+vol solo gatean act/size, no invierten dirección). "
                            "M_meta abstiene (flat) → su accuracy bruta penaliza la abstención; su mejora "
                            "esperada es de RIESGO. Exploratorio (docs/)."},
           "por_activo": {t: S[t]["perfil"] for t in ok},
           "pooled": POOL, "cobertura": cobertura, "obj1_vs_m5": obj1}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== POOLED (clusterizado por fecha) ===")
    for k, v in POOL.items():
        print(f"  {k:20s} Δ={v['delta']:+.5f} IC95=[{v['ci_low']:+.5f},{v['ci_high']:+.5f}] p_greater={v['p_greater']:.4f}")
    print("\n=== OBJ1 (batir a M5, agente-céntrico) ===")
    for strat in ("m8", "meta"):
        o = obj1[f"{strat}_vs_m5"]
        print(f"  {strat:5s}: acc≥ {o['acc_ge']} (p={o['sign_p_acc']}) · Sharpe≥ {o['sharpe_ge']} (p={o['sign_p_sharpe']})")
    print("\n=== COBERTURA vs ZeroR (Sharpe/Calmar) ===")
    for strat in ("m8", "meta"):
        print(f"  {strat:5s}: Sharpe {cobertura[f'{strat}_gana_sharpe_vs_zeror']} · "
              f"Calmar {cobertura[f'{strat}_gana_calmar_vs_zeror']} · acc {cobertura[f'{strat}_gana_acc_vs_zeror']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
