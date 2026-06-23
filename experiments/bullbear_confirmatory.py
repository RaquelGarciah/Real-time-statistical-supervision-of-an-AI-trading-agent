"""Rescate por régimen (alcista/bajista) y confirmatorio de ΔSharpe — SPY y POOLED-10. Sin re-entrenar.

Reconstruye la PARTE B del estudio canónico (walkforward_robustez) a nivel de marco práctico, en dos planos:

  A. CONFIRMATORIO (todo el OOS desplegable): mediana de ΔSharpe(sup−M5) por bootstrap estacionario pareado
     (Politis-Romano 1994), IC95 + cota inferior Bonferroni para la familia de confirmatorios (M8/M10/AutoML
     vs M5) → el veredicto H1_b usa la cota Bonferroni (controla FWER del OR). Más Deflated Sharpe (Bailey &
     López de Prado 2014) de cada brazo, n_trials = configuraciones exploradas en el estudio.
  B. POR RÉGIMEN (alcista vs bajista, tendencia 21d causal): McNemar pareado (sup vs M5, p_adj Holm sobre la
     familia régimen×contraste), block-permutation (blinda McNemar contra autocorrelación) y ΔSharpe puntual.
     Es la prueba de si el rescate sobrevive un test EN CADA régimen o solo agregando.

Plano único ±1: la posición es direccional, así que el retorno diario se reconstruye EXACTO desde el acierto
canónico del panel: r_t = (2·acierto_t − 1)·|r_{t+1}|, con |r_{t+1}| y el signo de la verdad desde
`wf.load_features`. Pooled = concatenación de los 10 con posiciones ±1: mismo MÉTODO de bootstrap que el pooled
de riesgo canónico (decision_automl_prep, 15 activos, retorno neto causal con leverage), pero DISTINTO universo
(10 vs 15) y distinta serie (±1 vs net-causal) → no debe cruzarse numéricamente con aquel.

Determinista (seed fijo). Uso: python experiments/bullbear_confirmatory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

import config
import experiments.walkforward_robustez as wf
from config import STRATA_OOS_START
from core.stats import block_permutation_test, deflated_sharpe, mcnemar_test

PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
ANR = json.load(open("outputs/experiments/automl_net_returns.json"))["por_activo"]
PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
PAIRS = [("m8", "m5"), ("m10_xgb", "m5"), ("automl", "m5")]  # familia confirmatoria → Bonferroni m=3
ARMS = ["m5", "m8", "m10_xgb", "automl"]
NAME = {"m5": "M5", "m8": "M8", "m10_xgb": "M10", "automl": "AutoML"}
N_TRIALS_DSR = 6           # configuraciones metodológicas exploradas por activo (coherente con quant_validation_panel)
ALPHA = 0.05
M_BONF = len(PAIRS)        # 3 confirmatorios (M8/M10/AutoML vs M5)
Q_BONF = (ALPHA / M_BONF) / 2.0   # cuantil inferior de la cota Bonferroni (IC 1-α/m, dos colas)
B_REPS = 2000
ANN = np.sqrt(252.0)
OUT = Path("outputs/experiments/bullbear_confirmatory.json")


def _sr(a) -> float:
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _truth(tk: str, dates: list) -> np.ndarray:
    """Signo de r_{t+1} alineado a `dates` (la verdad direccional)."""
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    return oos.shift(-1).reindex(pd.to_datetime(dates)).to_numpy()


def _trend(tk: str, dates: list) -> np.ndarray:
    """Tendencia a 21 días (media móvil causal, shift-1) alineada a `dates`."""
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    tr = oos.rolling(21, min_periods=5).mean().shift(1)
    return tr.reindex(pd.to_datetime(dates)).to_numpy()


def _daily_returns(tk: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Para un activo: retorno diario reconstruido por brazo + tendencia + máscara válida."""
    pan = json.load(open(PANEL_FILE))["por_activo"][tk]
    dates = ANR[tk]["dates"]; rnext = _truth(tk, dates); absr = np.abs(rnext)
    ret = {a: (2 * np.asarray(pan["correct_by_arm"][a], float) - 1) * absr for a in ARMS}
    corr = {a: np.asarray(pan["correct_by_arm"][a], float) for a in ARMS}
    tr = _trend(tk, dates)
    valid = ~np.isnan(rnext) & (np.sign(rnext) != 0) & ~np.isnan(tr)
    return {"ret": ret, "corr": corr}, tr, valid


def _boot_delta_sharpe(r_a: np.ndarray, r_b: np.ndarray) -> dict:
    """Bootstrap estacionario PAREADO de la mediana ΔSharpe(a−b) (Politis-Romano 1994; convención canónica)."""
    n = len(r_a); block = max(2, int(round(np.sqrt(n)))); p = 1.0 / block
    rng = np.random.default_rng(config.SEED); deltas = np.empty(B_REPS)
    for i in range(B_REPS):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        deltas[i] = _sr(r_a[idx]) - _sr(r_b[idx])
    return {"median_delta_sharpe": round(float(np.median(deltas)), 4),
            "ci95_low": round(float(np.quantile(deltas, 0.025)), 4),
            "ci95_high": round(float(np.quantile(deltas, 0.975)), 4),
            "ci_bonf_low": round(float(np.quantile(deltas, Q_BONF)), 4),
            "point": round(_sr(r_a) - _sr(r_b), 4), "block_len": int(block), "n_obs": int(n)}


def _dsr(r: np.ndarray) -> dict:
    """Deflated Sharpe: P(SR_verdadero > 0) con n_trials configs exploradas (Bailey & López de Prado 2014)."""
    r = r[~np.isnan(r)]; sd = r.std(ddof=1)
    sr_d = float(r.mean() / sd) if sd > 0 else 0.0
    sk = float(stats.skew(r)); ku = float(stats.kurtosis(r) + 3.0)
    return {"sr_daily": round(sr_d, 4), "sharpe_ann": round(sr_d * ANN, 3),
            "dsr": round(float(deflated_sharpe(sr_d, N_TRIALS_DSR, len(r), sk, ku)), 4), "n": int(len(r))}


def _holm(pvals: dict) -> dict:
    """Corrección Holm-Bonferroni (1979) sobre una familia de p-valores {clave: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); adj[k] = round(running, 4)
    return adj


def _confirmatory(ret: dict, label: str) -> dict:
    out = {"_label": label, "pairs": {}, "dsr": {}}
    for a, b in PAIRS:
        out["pairs"][f"{NAME[a]}_vs_{NAME[b]}"] = _boot_delta_sharpe(ret[a], ret[b])
    for a in ARMS:
        out["dsr"][NAME[a]] = _dsr(ret[a])
    return out


def _by_regime(ret: dict, corr: dict, tr: np.ndarray) -> dict:
    masks = {"alcista": tr > 0, "bajista": tr < 0}
    raw_p = {}; cells = {}
    for reg, msk in masks.items():
        cells[reg] = {"n": int(msk.sum()), "contrastes": {}}
        for a, b in PAIRS:
            ca, cb = corr[a][msk].astype(bool), corr[b][msk].astype(bool)
            _, pmc, bb, cc = mcnemar_test(ca, cb)
            _, pbp = block_permutation_test(ca, cb)
            dS = _sr(ret[a][msk]) - _sr(ret[b][msk])
            key = f"{NAME[a]}_vs_{NAME[b]}"
            raw_p[f"{reg}|{key}"] = float(pmc)
            cells[reg]["contrastes"][key] = {"mcnemar_p": round(float(pmc), 4), "blockperm_p": round(float(pbp), 4),
                                             "delta_sharpe": round(float(dS), 4), "b": bb, "c": cc}
    p_holm = _holm(raw_p)  # familia régimen×contraste (6)
    for reg in masks:
        for a, b in PAIRS:
            key = f"{NAME[a]}_vs_{NAME[b]}"
            cells[reg]["contrastes"][key]["mcnemar_p_holm"] = p_holm[f"{reg}|{key}"]
    return cells


def main() -> None:
    config.set_seeds(config.SEED)
    # --- carga por activo ---
    per = {tk: _daily_returns(tk) for tk in PANEL10}

    # SPY
    d, tr, val = per["SPY"]
    spy_ret = {a: d["ret"][a][val] for a in ARMS}; spy_corr = {a: d["corr"][a][val] for a in ARMS}; spy_tr = tr[val]

    # POOLED-10: concatenación (misma convención que el bootstrap pooled de riesgo)
    pool_ret = {a: [] for a in ARMS}; pool_corr = {a: [] for a in ARMS}; pool_tr = []
    for tk in PANEL10:
        d, tr, val = per[tk]
        for a in ARMS:
            pool_ret[a].append(d["ret"][a][val]); pool_corr[a].append(d["corr"][a][val])
        pool_tr.append(tr[val])
    pool_ret = {a: np.concatenate(pool_ret[a]) for a in ARMS}
    pool_corr = {a: np.concatenate(pool_corr[a]) for a in ARMS}
    pool_tr = np.concatenate(pool_tr)

    res = {"meta": {"panel": PANEL10, "pares": [f"{NAME[a]}_vs_{NAME[b]}" for a, b in PAIRS],
                    "alpha": ALPHA, "m_bonferroni": M_BONF, "q_bonf": round(Q_BONF, 5),
                    "n_trials_dsr": N_TRIALS_DSR, "b_reps": B_REPS, "seed": config.SEED,
                    "nota": "PARTE B confirmatoria (mediana ΔSharpe + cota Bonferroni + DSR) y rescate por "
                            "régimen (McNemar Holm + block-perm + ΔSharpe). Retornos ±1 reconstruidos del "
                            "acierto canónico; pooled = concatenación de los 10 con posiciones ±1 (mismo método "
                            "de bootstrap que el pooled de riesgo de 15, pero distinto universo y serie → no "
                            "cruzar numéricamente con el pooled-15 net-causal)."},
           "confirmatorio": {"SPY": _confirmatory(spy_ret, "SPY"),
                             "POOLED10": _confirmatory(pool_ret, "POOLED10")},
           "por_regimen": {"SPY": _by_regime(spy_ret, spy_corr, spy_tr),
                           "POOLED10": _by_regime(pool_ret, pool_corr, pool_tr)}}

    # --- reporte ---
    for scope in ("SPY", "POOLED10"):
        c = res["confirmatorio"][scope]
        print(f"\n===== {scope} · CONFIRMATORIO (todo el OOS) · cota Bonferroni q={Q_BONF:.4f} (m={M_BONF}) =====")
        for k, v in c["pairs"].items():
            ok = "H1_b SÍ" if v["ci_bonf_low"] > 0 else "H1_b NO"
            print(f"  {k:14s} mediana ΔSharpe={v['median_delta_sharpe']:+.2f} "
                  f"IC95=[{v['ci95_low']:+.2f},{v['ci95_high']:+.2f}] cotaBonf={v['ci_bonf_low']:+.2f} → {ok}")
        print("  DSR: " + " ".join(f"{NAME[a]}={c['dsr'][NAME[a]]['dsr']:.3f}" for a in ARMS))
        r = res["por_regimen"][scope]
        print(f"--- {scope} · POR RÉGIMEN (alcista n={r['alcista']['n']}, bajista n={r['bajista']['n']}) ---")
        for reg in ("alcista", "bajista"):
            for k, v in r[reg]["contrastes"].items():
                print(f"  {reg:8s} {k:14s} McNemar p_Holm={v['mcnemar_p_holm']:.4f} "
                      f"block-perm p={v['blockperm_p']:.4f} ΔSharpe={v['delta_sharpe']:+.2f}")
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
