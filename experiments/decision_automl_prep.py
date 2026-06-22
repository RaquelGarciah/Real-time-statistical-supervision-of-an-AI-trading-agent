"""Preparación del notebook `decision_automl`: efecto de STRATA + significancia de riesgo (sin H2O).

Para los 15 activos del panel, con el M10 CANÓNICO (ensemble 10 XGBoost, embargo=1, ALL22) reusado de
`quant_validation_panel`, calcula de forma determinista:

  1. Métricas por estrategia (M5, M8, M10, ZeroR, B&H) sobre la ventana del walk-forward (~250 d):
     accuracy, Sharpe, max drawdown, Calmar, equity final.
  2. Ablación agente15 vs STRATA7 vs ALL22 (mismo WF ensemble): Δacc y ΔSharpe = cuánto aporta STRATA.
  3. TreeSHAP (XGBoost canónico, ALL22) → importancia por bloque (agente vs régimen/vol/psa) + top features.
     Interpretabilidad limpia sobre árbol (no sobre ensemble) → evidencia de universalidad (CLAUDE.md §2.3).
  4. Significancia de RIESGO: bootstrap estacionario PAREADO (Politis-Romano 1994) de ΔSharpe y ΔmaxDD de
     M8 y M10 vs M5, B&H y ZeroR, por activo y POOLED (retornos netos concatenados). IC95 → marca dónde
     excluye 0. (La significancia de accuracy se deja como línea futura: ventana corta n≈250.)

AutoML NO se recomputa aquí (requiere H2O); sus cifras vienen del panel mm25 ya auditado. La importancia de
AutoML va en `automl_importance.py`.

Honestidad: ganar a ZeroR en accuracy es NOMINAL (no significativo, ventana corta). El valor robusto y
contrastable es el RIESGO (Sharpe/maxDD). Uso: python experiments/decision_automl_prep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from core.backtest import run_backtest
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22, AGENT15, STRATA7, PARAMS

PANEL = ["SPY", "QQQ", "DIA", "IWM", "XLE", "XLF", "XLK", "NVDA", "BAC", "TSLA",
         "MSTR", "SMCI", "ROKU", "MARA", "UNG"]
ARMS = ["m5", "m8", "m10", "zeror", "bh"]
SUP = ["m8", "m10"]            # estrategias STRATA (supervisor) que se contrastan en riesgo
BASE = ["m5", "bh", "zeror"]   # comparadores: agente, comprar-y-mantener, naïve mayoritario
ANN = np.sqrt(252)
B_BOOT = 2000
# Bloques interpretables de las 22 features (idéntico a m10_shap_priorflip).
BLOQUES = {"agente": AGENT15,
           "régimen": ["calm_prob", "stress_prob", "crisis_prob", "ram_score"],
           "volatilidad": ["garch_sigma", "gso_score"],
           "psa": ["psa_score"]}
OUT = Path("outputs/experiments/decision_automl_prep.json")


def _sr(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / s * ANN) if s > 0 else 0.0


def _maxdd(r: np.ndarray) -> float:
    r = np.nan_to_num(r)
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _calmar(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    mdd = _maxdd(r)
    if mdd == 0:
        return float("nan")
    ann_ret = eq[-1] ** (252.0 / len(r)) - 1.0
    return float(ann_ret / abs(mdd))


def _boot_paired(r_a: np.ndarray, r_b: np.ndarray, stat, seed: int) -> dict:
    """Bootstrap estacionario PAREADO de la mediana de stat(a)−stat(b) (Politis-Romano 1994)."""
    n = len(r_a)
    block = max(2, int(round(np.sqrt(n))))
    p = 1.0 / block
    rng = np.random.default_rng(seed)
    d = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        u = rng.random(n - 1)
        jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        d[i] = stat(r_a[idx]) - stat(r_b[idx])
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return {"median": round(float(np.median(d)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "point": round(stat(r_a) - stat(r_b), 4), "sig": bool(lo > 0 or hi < 0)}


def _net(oos_ret: pd.Series, full_idx: pd.Index, sub: pd.Index, pos: np.ndarray) -> np.ndarray:
    w = pd.Series(0.0, index=full_idx)
    w.loc[sub] = pos
    return run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()


def run_ticker(tk: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)

    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0

    pos = {"m5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
           "m8": np.sign(mv.loc[sub, "final_size"].to_numpy()),
           "m10": np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0),
           "zeror": np.full_like(truth, maj),
           "bh": np.ones_like(truth)}
    nr = {k: _net(oos_ret, m.index, sub, v) for k, v in pos.items()}
    acc = {k: round(float((v == truth).mean()), 4) for k, v in pos.items()}
    shp = {k: round(_sr(nr[k]), 3) for k in ARMS}
    mdd = {k: round(_maxdd(nr[k]), 4) for k in ARMS}
    cal = {k: round(_calmar(nr[k]), 3) for k in ARMS}
    eqf = {k: round(float(np.cumprod(1.0 + np.nan_to_num(nr[k]))[-1]), 4) for k in ARMS}

    # --- Ablación agente15 / strata7 / all22 (mismo WF ensemble) ---
    abl_acc, abl_shp = {}, {}
    for nm, cols in (("agente15", AGENT15), ("strata7", STRATA7), ("all22", ALL22)):
        pa = p1 if nm == "all22" else wf_p1(mv[cols], y)
        sa = mv.index[pa.notna().to_numpy()]
        ta = np.sign(mv.loc[sa, "r_next"].to_numpy())
        posa = np.where(pa.dropna().to_numpy() >= 0.5, 1.0, -1.0)
        abl_acc[nm] = round(float((posa == ta).mean()), 4)
        abl_shp[nm] = round(_sr(_net(oos_ret, m.index, sa, posa)), 3)
    ablation = {"acc": abl_acc, "sharpe": abl_shp,
                "d_acc_strata": round(abl_acc["all22"] - abl_acc["agente15"], 4),
                "d_sharpe_strata": round(abl_shp["all22"] - abl_shp["agente15"], 3)}

    # --- TreeSHAP (XGBoost canónico full-fit) → cuota por bloque + top features ---
    clf = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(mv[ALL22], y)
    try:
        import shap
        sv = shap.TreeExplainer(clf).shap_values(mv[ALL22])
        imp = np.abs(sv).mean(0); metodo = "media |TreeSHAP|"
    except Exception as e:  # noqa: BLE001
        imp = clf.feature_importances_; metodo = f"XGB gain (shap no disp.: {type(e).__name__})"
    imp = imp / max(imp.sum(), 1e-12)
    shares = dict(zip(ALL22, imp.tolist()))
    bloques = {b: round(float(sum(shares[f] for f in feats)), 4) for b, feats in BLOQUES.items()}
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:10]
    shap_out = {"metodo": metodo, "bloques": bloques,
                "cuota_strata": round(float(sum(shares[f] for f in STRATA7)), 4),
                "top10": [(f, round(s, 4)) for f, s in top]}

    # --- Bootstrap de riesgo: ΔSharpe y ΔmaxDD de M8/M10 vs M5/B&H/ZeroR ---
    boot = {}
    for s in SUP:
        for b in BASE:
            boot[f"{s}_vs_{b}"] = {"dSharpe": _boot_paired(nr[s], nr[b], _sr, config.SEED),
                                   "dMaxDD": _boot_paired(nr[s], nr[b], _maxdd, config.SEED)}

    return {"clase": _CLASE.get(tk, "?"), "n": int(len(sub)), "frac_up": round(frac_up, 4),
            "accuracy": acc, "sharpe": shp, "max_dd": mdd, "calmar": cal, "equity_final": eqf,
            "ablation": ablation, "shap": shap_out, "boot": boot,
            "net_returns": {k: [round(float(x), 6) for x in np.nan_to_num(nr[k])] for k in ARMS}}


_CLASE = {"SPY": "índice", "QQQ": "índice", "DIA": "índice", "IWM": "índice", "XLE": "ETF sect.",
          "XLF": "ETF sect.", "XLK": "ETF sect.", "UNG": "ETF commod.", "NVDA": "acción",
          "BAC": "acción", "TSLA": "acción", "MSTR": "cripto-px", "SMCI": "acción",
          "ROKU": "acción", "MARA": "cripto-px"}


def main() -> None:
    config.set_seeds(config.SEED)
    res = {"meta": {"seed": config.SEED, "panel": PANEL, "arms": ARMS, "bloques": BLOQUES,
                    "m10": "canónico: ensemble 10 XGBoost, embargo=1, N0=150, ALL22 (quant_validation_panel.wf_p1)",
                    "ventana": "walk-forward M10 (~250 d tras burn-in 150)",
                    "nota_significancia": "accuracy vs ZeroR = NOMINAL (no sig., n≈250 → futuro); "
                                          "RIESGO (Sharpe/maxDD) sí se contrasta con bootstrap pareado.",
                    "automl": "no recomputado aquí (requiere H2O); ver panel mm25 + automl_importance.py"},
           "por_activo": {}}
    for tk in PANEL:
        try:
            r = run_ticker(tk)
            res["por_activo"][tk] = r
            sig = [k for k, v in r["boot"].items() if v["dSharpe"]["sig"]]
            print(f"{tk:5s} acc M5={r['accuracy']['m5']:.3f} M8={r['accuracy']['m8']:.3f} "
                  f"M10={r['accuracy']['m10']:.3f} ZeroR={r['accuracy']['zeror']:.3f} | "
                  f"ΔSTRATA acc={r['ablation']['d_acc_strata']:+.3f} cuotaSTRATA_SHAP={r['shap']['cuota_strata']:.2f} | "
                  f"riesgo sig: {sig}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}

    # --- POOLED: bootstrap de riesgo sobre retornos netos concatenados de todos los activos ---
    ok = [t for t in PANEL if "error" not in res["por_activo"][t]]
    pooled_nr = {k: np.concatenate([np.array(res["por_activo"][t]["net_returns"][k]) for t in ok]) for k in ARMS}
    pooled_boot = {}
    for s in SUP:
        for b in BASE:
            pooled_boot[f"{s}_vs_{b}"] = {"dSharpe": _boot_paired(pooled_nr[s], pooled_nr[b], _sr, config.SEED),
                                          "dMaxDD": _boot_paired(pooled_nr[s], pooled_nr[b], _maxdd, config.SEED)}
    res["pooled"] = {"n_total": int(len(pooled_nr["m5"])), "n_activos": len(ok), "boot": pooled_boot}

    # medias de panel (sin retornos crudos)
    res["medias"] = {m: {k: round(float(np.mean([res["por_activo"][t][m][k] for t in ok])), 4)
                         for k in ARMS} for m in ("accuracy", "sharpe", "max_dd", "calmar")}
    res["medias"]["d_acc_strata"] = round(float(np.mean([res["por_activo"][t]["ablation"]["d_acc_strata"] for t in ok])), 4)
    res["medias"]["cuota_strata_shap"] = round(float(np.mean([res["por_activo"][t]["shap"]["cuota_strata"] for t in ok])), 4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n=== POOLED (n={res['pooled']['n_total']}, {len(ok)} activos) — riesgo M8/M10 vs M5/B&H/ZeroR ===")
    for k, v in pooled_boot.items():
        ds, dm = v["dSharpe"], v["dMaxDD"]
        print(f"  {k:14s} ΔSharpe={ds['point']:+.3f} IC{ds['ci95']} {'SIG' if ds['sig'] else '—'} | "
              f"ΔmaxDD={dm['point']:+.3f} IC{dm['ci95']} {'SIG' if dm['sig'] else '—'}")
    print(f"\nmedias acc: {res['medias']['accuracy']}")
    print(f"ΔSTRATA acc medio={res['medias']['d_acc_strata']} · cuota STRATA SHAP media={res['medias']['cuota_strata_shap']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
