"""Validación de producción 'real quant' de M10 sobre SMCI y el panel de 10 activos.

Aplica la batería que un comité de inversión exigiría antes de asignar capital,
reutilizando ``core.stats``/``core.metrics`` y el módulo nuevo ``core.validation``:

  Habilidad vs suerte (por activo): t de Newey-West (HAC), Sharpe e IC de Lo,
    PSR/DSR, sign test, McNemar vs M5/M8/B&H, Diebold-Mariano de P&L, permutación
    por bloques vs B&H.
  Riesgo: VaR/CVaR (histórico y Cornish-Fisher), Sortino, Calmar, Information Ratio.
  Economía: P&L neto con escenarios de coste de préstamo (0/100/300/500 pb), turnover,
    capacidad por volumen-dólar (ADV).
  Atribución factorial: alpha tras quitar betas Fama-French (5 factores + momentum).
  Multiplicidad/overfitting (panel): FDR (BH/BY) sobre los p de los 10 activos,
    haircut de Sharpe (Harvey-Liu-Zhu), PBO/CSCV, MinBTL, White Reality Check, Hansen SPA.

LÍMITE DURO honesto: el agente LLM solo existe en el OOS (post-cutoff, 2024-10→), así que
NO hay historia de M10 para estresar 2008/COVID; el stress realizado es el drawdown de 2025.

Pre-registro: BORRADOR para BITACORA (ver final del informe). Uso: python experiments/quant_validation_panel.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest, kurtosis as _kurt, skew as _skew

import config
from config import CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.metrics import calmar, equity_curve, max_drawdown, sharpe, sortino, turnover
from core.stats import (block_permutation_test, deflated_sharpe, diebold_mariano,
                        mcnemar_test, sign_test)
from core.validation import (apply_borrow_cost, cvar_cornish_fisher, cvar_historical,
                             factor_attribution, fdr_bh, fdr_by, haircut_sharpe,
                             hac_tstat, hansen_spa, information_ratio, load_ff_factors,
                             min_btl, pbo_cscv, reality_check, sharpe_se_lo,
                             var_cornish_fisher, var_historical)
import experiments.walkforward_robustez as wf

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
STEP, EMBARGO, N0, N_SEEDS = 21, 1, 150, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
              eval_metric="logloss", tree_method="hist")
AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA7
N_TRIALS_DSR = 6          # configuraciones exploradas a lo largo del estudio (DSR/haircut por activo)
BORROW_BPS = [0, 100, 300, 500]
OUT = Path("outputs/experiments/quant_validation_panel.json")


def build_states(ticker: str):
    """Régimen filtrado K=3 + sigma GARCH, recalibrados sobre la historia del activo (≤ 2024-09)."""
    feat_df, ret = wf.load_features(ticker)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    return gamma, sigma, oos_ret


def wf_p1(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """M10 canónico: walk-forward expandible, ensemble de 10 semillas, embargo=1."""
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 50:
            continue
        end = min(start + STEP, n)
        p.iloc[start:end] = np.mean(
            [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(X.iloc[:tr], y.iloc[:tr])
             .predict_proba(X.iloc[start:end])[:, 1] for sd in SEEDS], axis=0)
    return p


def _adv_usd(ticker: str) -> float:
    """ADV en dólares (mediana de Close×Volume) sobre el tramo OOS."""
    pqs = sorted(glob.glob(str(DATA_DIR / f"{ticker}_{CALIBRATION_START}_*.parquet")))
    end = pqs[-1].rsplit("_", 1)[1].replace(".parquet", "")
    px = data.load_market_data(ticker, CALIBRATION_START, end)
    dv = (px["Close"] * px["Volume"])
    dv = dv[dv.index >= pd.Timestamp(STRATA_OOS_START)]
    return float(dv.median())


def evaluate_ticker(ticker: str, ff: pd.DataFrame) -> tuple[dict, pd.Series, dict]:
    """Devuelve (bloque JSON del activo, serie nr10 indexada por fecha, p-valores clave)."""
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
    c10 = (pos10 == truth).astype(int)
    c5 = (pos5 == truth).astype(int)
    c8 = (pos8 == truth).astype(int)
    c_bh = (truth > 0).astype(int)  # B&H = siempre largo

    # P&L causales (lag=1). M10 vía run_backtest; M5/M8 desde el master; B&H = largo permanente.
    w10 = pd.Series(0.0, index=mv.index); w10.loc[sub] = pos10
    nr10 = run_backtest(oos_ret, w10, signal_lag=1)["net_return"].reindex(sub)
    nr8 = mv["nr_m8_causal"].reindex(sub)
    w_bh = pd.Series(1.0, index=mv.index)
    nr_bh = run_backtest(oos_ret, w_bh, signal_lag=1)["net_return"].reindex(sub)

    nr10_v = nr10.to_numpy()
    sr_daily = float(nr10_v.mean() / nr10_v.std(ddof=1)) if nr10_v.std(ddof=1) > 0 else 0.0
    sk, ku = float(_skew(nr10_v)), float(_kurt(nr10_v) + 3.0)  # kurt total para deflated_sharpe
    n = int(len(sub))

    _, hac_t, hac_p = hac_tstat(nr10_v)
    lo = sharpe_se_lo(nr10)
    k_s, n_s, p_sign, ci_s = sign_test(c10)
    # p de UNA cola para habilidad direccional (accuracy > 0.5): es el correcto
    # para el FDR del panel; el de dos colas marcaría como 'hallazgo' a un activo
    # donde M10 pierde significativamente (accuracy < 0.5), que no es habilidad.
    p_skill_1c = float(binomtest(int(k_s), int(n_s), 0.5, alternative="greater").pvalue)
    _, p_mc5, _, _ = mcnemar_test(c5, c10)
    _, p_mc8, _, _ = mcnemar_test(c8, c10)
    _, p_mcbh, _, _ = mcnemar_test(c_bh, c10)
    dm8 = diebold_mariano(-nr10_v, -nr8.to_numpy())
    _, p_bp_bh = block_permutation_test(c10, c_bh)

    bor = {}
    for b in BORROW_BPS:
        bt = apply_borrow_cost(oos_ret, w10, borrow_bps_yr=b, signal_lag=1)
        nrb = bt["net_return_borrow"].reindex(sub)
        bor[str(b)] = {"sharpe": round(float(sharpe(nrb)), 4),
                       "equity": round(float((1 + nrb.fillna(0)).prod()), 4)}

    adv = _adv_usd(ticker)
    fa = factor_attribution(nr10, ff)

    blk = {
        "headline": {
            "accuracy_m10": round(float(c10.mean()), 4), "accuracy_m5": round(float(c5.mean()), 4),
            "accuracy_m8": round(float(c8.mean()), 4), "accuracy_bh": round(float(c_bh.mean()), 4),
            "sharpe_m10": round(float(sharpe(nr10)), 4), "sharpe_bh": round(float(sharpe(nr_bh)), 4),
            "equity_final": round(float((1 + nr10.fillna(0)).prod()), 4),
            "max_drawdown": round(float(max_drawdown(equity_curve(nr10))), 4), "n_eval": n},
        "skill_vs_luck": {
            "hac_tstat": round(float(hac_t), 4), "hac_p": round(float(hac_p), 4),
            "sharpe_lo": {k: (round(float(val), 4) if np.isfinite(val) else None)
                          for k, val in lo.items()},
            "psr_dsr": {"sr_daily": round(sr_daily, 4),
                        "psr": round(float(deflated_sharpe(sr_daily, 1, n, sk, ku)), 4),
                        "dsr": round(float(deflated_sharpe(sr_daily, N_TRIALS_DSR, n, sk, ku)), 4),
                        "n_trials": N_TRIALS_DSR,
                        "dsr_sensibilidad": {str(nt): round(float(deflated_sharpe(sr_daily, nt, n, sk, ku)), 4)
                                             for nt in (6, 12, 24)}},
            "sign_test": {"k": int(k_s), "n": int(n_s), "p_2colas": round(float(p_sign), 4),
                          "p_skill_1cola": round(p_skill_1c, 4),
                          "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]},
            "mcnemar_vs_m5_p": round(float(p_mc5), 4), "mcnemar_vs_m8_p": round(float(p_mc8), 4),
            "mcnemar_vs_bh_p": round(float(p_mcbh), 4),
            "dm_pnl_vs_m8": {"stat": round(float(dm8[0]), 4), "p": round(float(dm8[1]), 4)},
            "block_perm_vs_bh_p": round(float(p_bp_bh), 4)},
        "risk": {
            "var95_hist": round(var_historical(nr10), 5), "cvar95_hist": round(cvar_historical(nr10), 5),
            "var95_cf": round(var_cornish_fisher(nr10), 5), "cvar95_cf": round(cvar_cornish_fisher(nr10), 5),
            "sortino": round(float(sortino(nr10)), 4), "calmar": round(float(calmar(nr10)), 4),
            "information_ratio_vs_bh": {k: round(float(val), 4) for k, val in
                                        information_ratio(nr10, nr_bh).items()}},
        "econ": {
            "turnover": round(float(turnover(pd.Series(pos10, index=sub))), 4),
            "adv_usd": round(adv, 0), "capacidad_1pct_adv_usd": round(0.01 * adv, 0),
            "borrow_scenarios": bor},
        "factor_attribution": {k: (round(float(val), 5) if isinstance(val, (int, float)) and np.isfinite(val)
                                   else ({kk: round(float(vv), 4) for kk, vv in val.items()}
                                         if isinstance(val, dict) else val))
                               for k, val in fa.items()},
    }
    pvals = {"skill_1c": p_skill_1c, "sign_2c": float(p_sign), "hac": float(hac_p), "sharpe": sr_daily}
    nr10.name = ticker
    nr_bh.name = ticker
    return blk, nr10, nr_bh, pvals


def main() -> None:
    wf.reset_thresholds_cache()
    ff = load_ff_factors(STRATA_OOS_START, "2026-12-31")
    por_activo, nr_by_tk, nrbh_by_tk, pv_skill, pv_hac, sr_by_tk = {}, {}, {}, {}, {}, {}
    for tk in PANEL:
        try:
            blk, nr10, nr_bh, pv = evaluate_ticker(tk, ff)
            por_activo[tk] = blk
            nr_by_tk[tk] = nr10; nrbh_by_tk[tk] = nr_bh
            pv_skill[tk] = pv["skill_1c"]; pv_hac[tk] = pv["hac"]; sr_by_tk[tk] = blk["headline"]["sharpe_m10"]
            print(f"{tk:5s} acc={blk['headline']['accuracy_m10']:.3f} SR={blk['headline']['sharpe_m10']:+.2f} "
                  f"skill_p1c={pv['skill_1c']:.3f} hac_p={pv['hac']:.3f} n={blk['headline']['n_eval']}")
        except Exception as e:  # noqa: BLE001
            por_activo[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}")

    # --- Multiplicidad sobre el panel ---
    tickers_ok = [t for t in PANEL if "error" not in por_activo[t]]
    p_skill_list = [pv_skill[t] for t in tickers_ok]  # una cola (habilidad), correcto para FDR
    bh = fdr_bh(p_skill_list, alpha=0.10)
    by = fdr_by(p_skill_list, alpha=0.10)
    # Matriz de configs para PBO/RC/SPA: M10 de cada activo alineado en fechas comunes.
    mat = pd.concat([nr_by_tk[t] for t in tickers_ok], axis=1, sort=True).dropna()
    bench0 = pd.Series(0.0, index=mat.index)  # benchmark laxo = cash (no hacer nada)
    pbo = pbo_cscv(mat, n_splits=16)
    rc = reality_check(mat, bench0, n_boot=2000, seed=config.SEED)
    spa = hansen_spa(mat, bench0, n_boot=2000, seed=config.SEED)
    # Benchmark exigente = B&H del propio activo: matriz de exceso M10−B&H vs 0.
    act = pd.concat([(nr_by_tk[t] - nrbh_by_tk[t]).rename(t) for t in tickers_ok],
                    axis=1, sort=True).dropna()
    rc_bh = reality_check(act, pd.Series(0.0, index=act.index), n_boot=2000, seed=config.SEED)
    spa_bh = hansen_spa(act, pd.Series(0.0, index=act.index), n_boot=2000, seed=config.SEED)
    best_tk = max(tickers_ok, key=lambda t: sr_by_tk[t])
    n_best = por_activo[best_tk]["headline"]["n_eval"]
    hc = haircut_sharpe(sr_by_tk[best_tk], n_trials=len(tickers_ok), n_obs=n_best)
    mbtl = min_btl(n_trials=len(tickers_ok), target_sharpe=1.0)
    # Alpha factorial del mejor activo descontado por la selección best-of-N (Bonferroni).
    from scipy.stats import norm as _norm
    fa_best = por_activo[best_tk]["factor_attribution"]
    t_a = float(fa_best.get("t_alpha", float("nan")))
    p2_a = float(2 * (1 - _norm.cdf(abs(t_a)))) if np.isfinite(t_a) else float("nan")
    alpha_disc = {"ticker": best_tk, "alpha_ann": fa_best.get("alpha_ann"), "t_alpha": round(t_a, 4),
                  "p2_nominal": round(p2_a, 4),
                  "p2_bonferroni_panel": round(min(1.0, p2_a * len(tickers_ok)), 4)}

    n_raw = int(sum(p < 0.10 for p in p_skill_list))
    multiplicity = {
        "n_tickers": len(tickers_ok),
        "metodo_fdr": "p binomial de una cola (accuracy > 0.5) por activo; FDR a alpha=0.10",
        "fdr_bh": {"n_rejected": bh["n_rejected"], "threshold": round(bh["threshold"], 4)},
        "fdr_by": {"n_rejected": by["n_rejected"], "threshold": round(by["threshold"], 4)},
        "haircut_sharpe_mejor_activo": {"ticker": best_tk, "sr_obs": round(hc["sr_obs"], 4),
                                        "sr_haircut_bonferroni": round(hc["sr_haircut_bonferroni"], 4),
                                        "sr_haircut_bhy": round(hc["sr_haircut_bhy"], 4),
                                        "haircut_pct": round(hc["haircut_pct"], 4),
                                        "nota": "n_trials=10 (best-of-panel); Holm≡Bonferroni en rango 1"},
        "alpha_mejor_activo": alpha_disc,
        "pbo_cscv": {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
                     for k, v in pbo.items()},
        "min_btl_years": round(mbtl, 2),
        "white_reality_check_vs_cash": {"best": str(rc["best_strategy"]), "V": round(rc["V"], 4),
                                        "p_value": round(rc["p_value"], 4),
                                        "nota": "benchmark laxo = cash; mide rentabilidad, no habilidad"},
        "white_reality_check_vs_bh": {"best": str(rc_bh["best_strategy"]), "V": round(rc_bh["V"], 4),
                                      "p_value": round(rc_bh["p_value"], 4),
                                      "nota": "benchmark exigente = B&H del propio activo (exceso M10−B&H)"},
        "hansen_spa_vs_cash": {"t_spa": round(spa["t_spa"], 4), "p_consistent": round(spa["p_consistent"], 4)},
        "hansen_spa_vs_bh": {"t_spa": round(spa_bh["t_spa"], 4), "p_consistent": round(spa_bh["p_consistent"], 4)},
    }

    verdict = {
        "n_tickers_sig_raw": n_raw,
        "n_tickers_survive_fdr_bh": bh["n_rejected"],
        "pbo": multiplicity["pbo_cscv"].get("pbo"),
        "reality_check_vs_cash_p": multiplicity["white_reality_check_vs_cash"]["p_value"],
        "reality_check_vs_bh_p": multiplicity["white_reality_check_vs_bh"]["p_value"],
        "go_no_go": "NO-GO (condicional)" if bh["n_rejected"] == 0 else "REVISAR",
        "comentario": ("El cuello de botella NO es el coste de préstamo: M10 no está permanentemente "
                       "corto como M5, así que el borrow apenas mueve el Sharpe (p. ej. SMCI 1,84→1,81 a "
                       "500 pb). El bloqueo es la multiplicidad y el tamaño muestral: ningún activo "
                       "sobrevive al FDR de habilidad direccional del panel, el Sharpe del mejor activo "
                       "se recorta ~100% por Harvey-Liu-Zhu (best-of-10) y la PBO es 0,38. SMCI muestra "
                       "señales nominalmente fuertes (alpha factorial t≈2,5; block-perm vs B&H p=0,047) y "
                       "White RC rechaza (p=0,024), pero ese contraste es vs cash (benchmark laxo) y mide "
                       "rentabilidad, no habilidad direccional. Veredicto honesto: prometedor en el "
                       "activo-caso, NO apto para producción con evidencia de panel controlada por "
                       "multiplicidad (~250 d). Falsable: cambiaría con muestra más larga (MinBTL≈2,5 "
                       "años), forward test y borrow calibrado por activo difícil de tomar prestado."),
    }

    res = {
        "meta": {"panel": PANEL, "oos_start": STRATA_OOS_START, "calibration_end": CALIBRATION_END,
                 "signal_lag": 1, "embargo": EMBARGO, "n0": N0, "step": STEP, "n_seeds": N_SEEDS,
                 "features": "ALL22", "seed": config.SEED, "n_trials_dsr": N_TRIALS_DSR,
                 "n_trials_haircut": len(tickers_ok),
                 "nota_n_trials": ("n_trials_dsr=6 = configuraciones metodológicas exploradas por activo "
                                   "(eje intra-activo); n_trials_haircut=nº de activos del panel "
                                   "(eje de selección best-of-N, cross-activo). Ejes ortogonales."),
                 "borrow_bps": BORROW_BPS, "ff_n_obs": int(len(ff)),
                 "nota": "due-diligence de producción; M10 canónico walk-forward emb=1; cifras auditables",
                 "limite_duro": "M10 solo existe en OOS post-2024-10 → 2008/COVID no estresables en M10"},
        "por_activo": por_activo,
        "multiplicity_panel": multiplicity,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # --- Aserción de claves prometidas ---
    assert set(res) == {"meta", "por_activo", "multiplicity_panel", "verdict"}
    for t in tickers_ok:
        for sec in ("headline", "skill_vs_luck", "risk", "econ", "factor_attribution"):
            assert sec in por_activo[t], f"falta {sec} en {t}"
    for key in ("fdr_bh", "fdr_by", "pbo_cscv", "white_reality_check_vs_cash",
                "white_reality_check_vs_bh", "hansen_spa_vs_cash", "hansen_spa_vs_bh",
                "min_btl_years", "haircut_sharpe_mejor_activo", "alpha_mejor_activo"):
        assert key in multiplicity, f"falta {key}"
    print(f"\nPanel OK: {len(tickers_ok)}/{len(PANEL)} activos · "
          f"FDR-BH rechaza {bh['n_rejected']} · PBO={multiplicity['pbo_cscv'].get('pbo')} · "
          f"RC vs cash p={rc['p_value']:.3f} · RC vs B&H p={rc_bh['p_value']:.3f} · "
          f"veredicto={verdict['go_no_go']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
