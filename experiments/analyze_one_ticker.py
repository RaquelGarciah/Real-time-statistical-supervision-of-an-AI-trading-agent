"""Análisis de alcance de UN activo (se dispara cuando su caché de agente está completa).

Recomputa M5/M8/M10 y las métricas que responden a la hipótesis de los dos canales:
- Canal régimen (RAM/M8): accuracy direccional cruda del régimen y valor añadido de STRATA
  (ΔM8 = M8 − M5). Debería ser alto en activos de leverage fuerte (índices/ETF).
- Canal meta-aprendiz (M10): accuracy de M10, sesgo corto del agente y volatilidad.

Escribe outputs/experiments/scope_oneoff_<TICKER>.json. Uso:
    python experiments/analyze_one_ticker.py --ticker QQQ
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config
from core.backtest import run_backtest
from core.metrics import max_drawdown, equity_curve, sharpe
from core.stats import mcnemar_test, sign_test
from core.validation import hac_tstat
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import ALL22, build_states, wf_p1

REGNAMES = ["Calma", "Estrés", "Crisis"]


def _ensure_full_history(ticker: str) -> None:
    """Garantiza un parquet que cubra el OOS (los nuevos solo tienen el de calibración del screen)."""
    import datetime
    from core import data
    from config import CALIBRATION_START
    data.load_market_data(ticker, CALIBRATION_START, datetime.date.today().isoformat())


def analyze(ticker: str) -> dict:
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"].get(ticker, {})
    _ensure_full_history(ticker)
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)

    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    pos10 = np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    pos5 = np.sign(mv.loc[sub, "agent_size"].to_numpy())
    pos8 = np.sign(mv.loc[sub, "final_size"].to_numpy())
    c10 = (pos10 == truth).astype(int); c5 = (pos5 == truth).astype(int); c8 = (pos8 == truth).astype(int)
    acc_m5, acc_m8, acc_m10 = float(c5.mean()), float(c8.mean()), float(c10.mean())

    w10 = pd.Series(0.0, index=mv.index); w10.loc[sub] = pos10
    nr10 = run_backtest(oos_ret, w10, signal_lag=1)["net_return"].reindex(sub)
    _, hac_t, hac_p = hac_tstat(nr10.to_numpy())
    k_s, n_s, _, _ = sign_test(c10)
    p_skill = float(binomtest(int(k_s), int(n_s), 0.5, alternative="greater").pvalue)
    _, p_mc5, _, _ = mcnemar_test(c5, c10)

    # Canal régimen: accuracy direccional cruda del régimen (signo de calibración del régimen dominante).
    sign_prior = {k: float(np.sign(lev.get("media_regimen", {}).get(nm, 0.0))) for k, nm in enumerate(REGNAMES)}
    dom = mv.loc[sub, "regime_dom"].to_numpy().astype(int)
    pos_reg = np.array([sign_prior[d] for d in dom])
    mask = pos_reg != 0
    regime_dir_acc = float((pos_reg[mask] == truth[mask]).mean()) if mask.any() else float("nan")
    oos_crisis_frac = float((mv.loc[sub, "regime_dom"] == 2).mean())
    agent_short_frac = float((mv.loc[sub, "agent_size"] < 0).mean())

    # --- Tabla completa de estrategias (accuracy + Sharpe + equity + maxDD sobre el mismo tramo) ---
    frac_up = float((truth > 0).mean())
    maj_sign = 1.0 if frac_up >= 0.5 else -1.0
    posiciones = {"M5 (agente)": pos5, "M8 (STRATA)": pos8, "M10 (meta-learner)": pos10,
                  "Régimen (RAM crudo)": pos_reg, "B&H (siempre largo)": np.ones_like(truth),
                  "S&H (siempre corto)": -np.ones_like(truth),
                  "Mayoría (ZeroR/NIR)": np.full_like(truth, maj_sign)}
    estrategias = {}
    for nm, pos in posiciones.items():
        # Convención canónica del panel: accuracy direccional = sign(pos)==sign(truth) sobre TODOS
        # los días; un 'hold' (pos=0) cuenta como fallo (no toma dirección). Apples-to-apples.
        acc = float((pos == truth).mean())
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
        nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        estrategias[nm] = {"accuracy": round(acc, 4), "sharpe": round(float(sharpe(nr)), 4),
                           "equity": round(float((1 + nr.fillna(0)).prod()), 4),
                           "max_dd": round(float(max_drawdown(equity_curve(nr))), 4)}

    res = {
        "ticker": ticker, "n_eval": int(len(sub)),
        "naturaleza": {"leverage_corr": lev.get("leverage_corr"), "crisis_mean": lev.get("crisis_mean"),
                       "oos_crisis_frac": round(oos_crisis_frac, 4), "agent_short_frac": round(agent_short_frac, 4),
                       "oos_vol_media": round(float(mv.loc[sub, "garch_sigma"].mean()), 4)},
        "canal_regimen": {"regime_dir_acc": round(regime_dir_acc, 4), "acc_m5": round(acc_m5, 4),
                          "acc_m8": round(acc_m8, 4), "strata_valor_m8_m5": round(acc_m8 - acc_m5, 4)},
        "canal_m10": {"acc_m10": round(acc_m10, 4), "sharpe_m10": round(float(sharpe(nr10)), 4),
                      "max_drawdown": round(float(max_drawdown(equity_curve(nr10))), 4),
                      "skill_p_1cola": round(p_skill, 4), "hac_t": round(float(hac_t), 4),
                      "hac_p": round(float(hac_p), 4), "mcnemar_vs_m5_p": round(float(p_mc5), 4)},
        "estrategias": estrategias, "frac_up": round(frac_up, 4),
    }
    # Veredicto del canal régimen (vs líneas base del panel: fuerte regAcc~0.534/ΔM8~0.055, débil ~0.493/0.025).
    cr = res["canal_regimen"]
    res["lectura_canal_regimen"] = (
        "CONFIRMA canal régimen (régimen direccional >0.5 y ΔM8 alto)"
        if cr["regime_dir_acc"] > 0.50 and cr["strata_valor_m8_m5"] >= 0.04 else
        "NO confirma (régimen no direccional o STRATA no aporta)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ticker", required=True)
    tk = ap.parse_args().ticker
    res = analyze(tk)
    out = Path(f"outputs/experiments/scope_oneoff_{tk}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    cr, cm, nt = res["canal_regimen"], res["canal_m10"], res["naturaleza"]
    print(f"\n=== {tk} (n={res['n_eval']}) ===")
    print(f"  naturaleza: leverage={nt['leverage_corr']:+.3f} crisisOOS={nt['oos_crisis_frac']:.2f} "
          f"agente_corto={nt['agent_short_frac']:.2f}")
    print(f"  CANAL RÉGIMEN: regAcc={cr['regime_dir_acc']:.3f} M5={cr['acc_m5']:.3f} M8={cr['acc_m8']:.3f} "
          f"ΔM8={cr['strata_valor_m8_m5']:+.3f}")
    print(f"  CANAL M10: acc={cm['acc_m10']:.3f} Sharpe={cm['sharpe_m10']:+.2f} skill_p={cm['skill_p_1cola']:.3f} "
          f"HAC_t={cm['hac_t']:.2f}")
    print(f"  → {res['lectura_canal_regimen']}")
    print(f"\n  TODAS LAS ESTRATEGIAS (n={res['n_eval']}, frac_up={res['frac_up']}):")
    print(f"  {'estrategia':22s} {'acc':>6s} {'Sharpe':>8s} {'equity':>8s} {'maxDD':>8s}")
    for nm, d in sorted(res["estrategias"].items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {nm:22s} {d['accuracy']:6.3f} {d['sharpe']:8.2f} {d['equity']:8.3f} {d['max_dd']:8.2%}")
    print(f"OK · {out}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
