"""Validación walk-forward / robustez multi-ventana de STRATA (SPY).

Pendiente nº1 del tutor: "lánzalo en diferentes años/momentos; puede que tuvieras suerte". Por el
constraint duro de que el agente LLM solo existe en el OOS (post-cutoff de DeepSeek, 2024-10→), la
validación se PARTE EN DOS:

  PARTE A — robustez del MODELO de régimen (24 años, SIN agente). Rolling-origin / time-series CV
    (Tashman 2000; Bergmeir-Benítez 2012) sobre 2000–2024-09: held-out log-likelihood de HMM K∈{2,3,4}
    por origen anual (incl. 2008/2020/2022) + informatividad direccional del régimen (mapeo CONGELADO
    en el tramo de ajuste). Mide robustez INTER-ÉPOCA: aquí recae la respuesta al "tuviste suerte".

  PARTE B — robustez del RESCATE (M8 y M10 vs M5, SOLO dentro del OOS). LÍMITE DURO: el agente no existe
    antes de 2024-10, así que las "ventanas" son SUB-TROZOS SOLAPADOS de un único tramo de ~18 meses, NO
    años distintos. Mide ESTABILIDAD INTRA-OOS, no robustez inter-época.

    TESTS CONFIRMATORIOS (dictan el veredicto B): mediana de ΔSharpe con IC95 por bootstrap estacionario
    PAREADO (índices de bloque compartidos entre los dos brazos, recomputando Sharpe por brazo; bloque √N,
    Politis-Romano 1994) que excluya 0, para **M8−M5** y **M10−M5**. M10 = CPCV-OOF purgado (López de
    Prado): única vía en 18 meses; es validación cruzada, NO walk-forward estricto.

    EXPLORATORIO/SANITY (no veredicto): sliding/anchor/disjoint por ventana, McNemar por ventana y
    estratificado por régimen (Holm), Deflated Sharpe, sanity dual same-day/causal, tabla maestra, panel.

Pre-registro: BITACORA.md [2026-06-09] [Pre-registro] + enmienda [2026-06-10] (M10 vs M5, sliding 150/15).
Uso: ``python experiments/walkforward_robustez.py``
"""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raíz del repo en el path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest, kurtosis as _kurtosis, skew as _skew, spearmanr

import config
from config import (
    CACHE_AGENT_DIR,
    CACHE_MODELS_DIR,
    CALIBRATION_END,
    CALIBRATION_START,
    DATA_DIR,
    STRATA_OOS_START,
)
from core import data, features
from core.backtest import run_backtest
from core.cpcv import CombinatorialPurgedKFold
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.metrics import classification_metrics, equity_curve
from core.stats import block_permutation_test, deflated_sharpe, mcnemar_test, sign_test
from strata.detectors import reset_thresholds_cache
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

# --- Configuración congelada (pre-registro + enmienda 2026-06-10) --------------------------
TICKER = "SPY"
PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
PERS = list(config.ACTIVE_PERSONALITIES)
TAU_RAM = 0.5
RAM_THRESHOLDS = (TAU_RAM / 2, TAU_RAM, 0.70)  # gate efectivo = medium = τ (entrada 2026-06-09)

# Parte A — rolling-origin del modelo de régimen.
A_K_GRID = (2, 3, 4)
A_ORIGINS = tuple(range(2008, 2024))   # orígenes anuales: incluyen 2008/2020/2022
A_HORIZON_DAYS = 252
A_K3_DOM_FRAC_REF = 0.70

# Parte B — rolling-origin del rescate dentro del OOS (enmienda: sliding 150/15, menos solape).
B_WINDOW = 150
B_SLIDING_STEP = 15
B_ANCHOR_STEP = 20
ALPHA = 0.10
B_BOOT_REPS = 2000
N_TRIALS_DSR = 3
STRATUM_MIN_N = 60
PANEL_PRIOR_OOS_DAYS = 60

# XGBoost de M10 (idéntico a experiments/m10_k2.py).
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
              eval_metric="logloss", random_state=config.SEED, tree_method="hist")

# Mapa brazo → columna de sizing en el master.
_SIZE_COL = {"m5": "agent_size", "m8": "final_size", "m10": "m10_size"}
ANN = np.sqrt(252)
OUT = Path("outputs/experiments/walkforward_robustez.json")


def _hash_dir(path: Path, pattern: str = "*") -> str:
    h = hashlib.sha256()
    for fp in sorted(path.glob(pattern)):
        if fp.is_file():
            h.update(fp.name.encode()); h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def _sr(a) -> float:
    """Sharpe anualizado de un array de retornos diarios (criterio de metrics.sharpe)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _n_eff_bartlett(x) -> tuple[float, float]:
    """N efectivo de Bartlett (1946): N·(1−ρ̂)/(1+ρ̂), ρ̂=autocorr lag-1. Solo informativo."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]; n = len(x)
    if n < 3 or x.std() == 0:
        return float(n), 0.0
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1]); rho = max(min(rho, 0.999), -0.999)
    return float(n * (1 - rho) / (1 + rho)), rho


def _holm_bonferroni(pvals: dict, alpha: float = ALPHA) -> dict:
    """Holm-Bonferroni (Holm 1979): {p_raw, p_adj, reject} por etiqueta, FWER ≤ alpha."""
    valid = [(k, v) for k, v in pvals.items() if v is not None and not np.isnan(v)]
    m = len(valid); out: dict = {}
    if m == 0:
        return {k: {"p_raw": None, "p_adj": None, "reject": False} for k in pvals}
    still, prev = True, 0.0
    for i, (k, p) in enumerate(sorted(valid, key=lambda kv: kv[1])):
        p_adj = max(prev, min(1.0, (m - i) * p)); prev = p_adj
        rej = still and (p <= alpha / (m - i))
        if not rej:
            still = False
        out[k] = {"p_raw": float(p), "p_adj": float(p_adj), "reject": bool(rej)}
    for k in pvals:
        out.setdefault(k, {"p_raw": None, "p_adj": None, "reject": False})
    return out


# ============================================================================================
# Carga de datos
# ============================================================================================

def load_agent(ticker: str) -> dict:
    out: dict = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        d = json.load(open(fp))
        pers = {k: PersonalityOutput(name=k, action=v["action"], size=v["size"],
                                     confidence=v["confidence"], reasoning="")
                for k, v in d.get("personalities", {}).items()}
        out[pd.Timestamp(d["date"])] = AgentOutput(
            date=d["date"], ticker=d["ticker"], action=d["action"], size=d["size"],
            confidence=d["confidence"], reasoning="", personalities=pers)
    return out


def load_features(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    parquets = sorted(glob.glob(str(DATA_DIR / f"{ticker}_{CALIBRATION_START}_*.parquet")))
    data_end = parquets[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(ticker, CALIBRATION_START, data_end)
    ret = features.log_returns(prices["Close"])
    rv21 = features.realized_vol_annualized(ret, window=21)
    feat_df = pd.concat([ret.rename("r"), rv21.rename("rv")], axis=1).dropna()
    return feat_df, ret


def build_market_states_oos() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Régimen filtrado canónico K=3 + sigma GARCH causal + oos_ret, reusando caché FIJA pre-OOS.

    Calibración 2000–2024-09 (hmm.pkl/garch_SPY.pkl): ANTERIOR a toda ventana OOS → sin look-ahead.
    """
    feat_df, ret = load_features(TICKER)
    hmm = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
    garch = pickle.load(open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "rb"))
    gamma = hmm.predict_proba_filtered(feat_df.to_numpy())
    gamma_df = pd.DataFrame(gamma, index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    return gamma_df, sigma, oos_ret


# ============================================================================================
# Master M5/M8 (+ 22 features de M10) y CPCV
# ============================================================================================

def run_master(gamma_df: pd.DataFrame, sigma: pd.Series, oos_ret: pd.Series, agents: dict) -> pd.DataFrame:
    """Recorre el OOS día a día con override-C (M8); guarda M5/M8, severidades y las 22 features de M10.

    Series netas causales (lag=1, único válido) y same-day (lag=0, sanity) de M5 y M8 vía run_backtest.
    """
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute",
                           psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
                           ram_thresholds=RAM_THRESHOLDS)
    rows, sizing_hist = [], []
    for t in sorted(agents):
        if t not in gamma_df.index or t not in sigma.index:
            continue
        a = agents[t]; sizing_hist.append(a.size); g = gamma_df.loc[t]
        ms = {"regime": {"calm_prob": float(g["Calma"]), "stress_prob": float(g["Estrés"]),
                         "crisis_prob": float(g["Crisis"]), "viterbi_state": int(np.argmax(g.values))},
              "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, sizing_hist)
        row = {"date": t, "agent_size": a.size, "final_size": dec.final_size,
               "ram_sev": dec.detectors["ram"].severity, "regime_dom": int(np.argmax(g.values)),
               "intervenido": dec.was_intervened,
               "ram_score": dec.detectors["ram"].score, "psa_score": dec.detectors["psa"].score,
               "gso_score": dec.detectors["gso"].score, "calm_prob": float(g["Calma"]),
               "stress_prob": float(g["Estrés"]), "crisis_prob": float(g["Crisis"]),
               "garch_sigma": float(sigma.loc[t])}
        for nm in PERS:
            po = a.personalities.get(nm)
            row[f"{nm}_sign"] = 0.0 if po is None else (1.0 if po.action == "long" else -1.0 if po.action == "short" else 0.0)
            row[f"{nm}_size"] = 0.0 if po is None else float(po.size)
            row[f"{nm}_conf"] = 0.0 if po is None else float(po.confidence)
        rows.append(row)
    m = pd.DataFrame(rows).set_index("date")
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    m["r_curr"] = oos_ret.reindex(m.index)
    m["y"] = np.sign(m["r_next"])
    for arm, col in (("m5", "agent_size"), ("m8", "final_size")):
        for lag, suf in ((1, "causal"), (0, "sameday")):
            m[f"nr_{arm}_{suf}"] = run_backtest(oos_ret, m[col], signal_lag=lag)["net_return"].reindex(m.index)
    return m


def cpcv_oof(Xm: pd.DataFrame, ym: pd.Series) -> pd.Series:
    """Probabilidad out-of-fold de subida con CPCV purgado (López de Prado 2018; = m10_k2.py)."""
    t1 = pd.Series(Xm.index, index=Xm.index).shift(-1).ffill()
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, embargo=5)
    s = np.zeros(len(Xm)); c = np.zeros(len(Xm))
    for tr, te in cv.split(Xm, t1=t1):
        clf = xgb.XGBClassifier(**PARAMS); clf.fit(Xm.iloc[tr], ym.iloc[tr])
        s[te] += clf.predict_proba(Xm.iloc[te])[:, 1]; c[te] += 1
    return pd.Series(s / np.maximum(c, 1), index=Xm.index)


def add_m10(m: pd.DataFrame, oos_ret: pd.Series) -> pd.DataFrame:
    """Añade M10 (XGBoost-CPCV sobre las 22 features STRATA): m10_size + series netas causal/sameday."""
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    yb = (m.loc[valid, "r_next"] > 0).astype(int)
    cols = [f"{nm}_{k}" for nm in PERS for k in ("sign", "size", "conf")] + \
           ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
    p1 = cpcv_oof(m.loc[valid, cols], yb)
    m["m10_size"] = np.sign(p1 - 0.5).reindex(m.index).fillna(0.0)
    for lag, suf in ((1, "causal"), (0, "sameday")):
        m[f"nr_m10_{suf}"] = run_backtest(oos_ret, m["m10_size"], signal_lag=lag)["net_return"].reindex(m.index)
    return m


def paired_daily_returns(m: pd.DataFrame, pair: tuple[str, str], lag: int = 1) -> pd.DataFrame:
    """Retornos netos pareados de dos brazos: ret_a, ret_b, diff = a − b (causal lag=1 / same-day lag=0)."""
    suf = "causal" if lag == 1 else "sameday"
    a, b = pair
    out = pd.DataFrame({"ret_a": m[f"nr_{a}_{suf}"], "ret_b": m[f"nr_{b}_{suf}"]})
    out["diff"] = out["ret_a"] - out["ret_b"]
    return out.dropna()


def directional_correct(m: pd.DataFrame, size_col: str) -> np.ndarray:
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    pred = np.sign(m.loc[valid, size_col].to_numpy()); truth = np.sign(m.loc[valid, "r_next"].to_numpy())
    return (pred == truth).astype(int)


# ============================================================================================
# PARTE A — robustez del modelo de régimen (24 años, sin agente)
# ============================================================================================

def part_a_heldout_ll(feat_df: pd.DataFrame) -> dict:
    """Held-out LL/obs rodante de HMM K∈{2,3,4} sobre 2000–2024-09 (DESCRIPTIVO)."""
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    X = calib.to_numpy(); idx = calib.index; rows = []
    for origin in A_ORIGINS:
        cut = pd.Timestamp(f"{origin}-01-01"); n_fit = int((idx < cut).sum())
        eval_pos = np.where(idx >= cut)[0]
        if n_fit < 250 or len(eval_pos) < 30:
            continue
        eval_pos = eval_pos[:A_HORIZON_DAYS]
        for K in A_K_GRID:
            h = RegimeHMM(n_states=K, seed=config.SEED).fit(X[:n_fit])
            ll = float(h.model.score(h._standardize(X[eval_pos])) / len(eval_pos))
            rows.append({"origin": origin, "K": K, "ll_por_obs": round(ll, 4), "n_obs": int(len(eval_pos))})
    by_o: dict = {}
    for r in rows:
        by_o.setdefault(r["origin"], {})[r["K"]] = r["ll_por_obs"]
    dom = sum(1 for d in by_o.values() if d.get(3, -np.inf) > d.get(2, -np.inf))
    return {"per_origin_K": rows, "k3_domina_frac": round(dom / max(len(by_o), 1), 3),
            "k3_domina_frac_ref": A_K3_DOM_FRAC_REF, "n_origins": len(by_o),
            "label_switch_control": "estados ordenados por sigma ascendente en RegimeHMM.fit"}


def part_a_directional(feat_df: pd.DataFrame, ret: pd.Series) -> dict:
    """Informatividad direccional del régimen por ventana (gate τ=0.5), mapeo CONGELADO en el ajuste."""
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    cret = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
    X = calib.to_numpy(); idx = calib.index; rfit = calib["r"].to_numpy(); rows, frozen = [], {}
    for origin in A_ORIGINS:
        cut = pd.Timestamp(f"{origin}-01-01"); n_fit = int((idx < cut).sum())
        eval_pos = np.where(idx >= cut)[0]
        if n_fit < 250 or len(eval_pos) < 30:
            continue
        eval_pos = eval_pos[:A_HORIZON_DAYS]
        h = RegimeHMM(n_states=3, seed=config.SEED).fit(X[:n_fit])
        sf = h.predict_states(X[:n_fit])
        mu = {s: (float(rfit[:n_fit][sf == s].mean()) if (sf == s).any() else 0.0) for s in range(3)}
        long_s, short_s = max(mu, key=mu.get), min(mu, key=mu.get)
        frozen[str(origin)] = {"long_state": int(long_s), "short_state": int(short_s),
                               "mu_state": {str(k): round(v, 6) for k, v in mu.items()}}
        g = h.predict_proba_filtered(X[:eval_pos[-1] + 1])[eval_pos]
        conf, dom = g.max(1), g.argmax(1)
        r_next = cret.shift(-1).reindex(idx[eval_pos]).to_numpy()
        pos = np.where(dom == long_s, 1.0, np.where(dom == short_s, -1.0, 0.0))
        gate = (conf >= TAU_RAM) & ~np.isnan(r_next) & (pos != 0) & (np.sign(r_next) != 0)
        acc = float((np.sign(pos[gate]) == np.sign(r_next[gate])).mean()) if gate.sum() else float("nan")
        rows.append({"origin": origin, "acc_at_gate": None if np.isnan(acc) else round(acc, 4), "n_obs": int(gate.sum())})
    accs = [r["acc_at_gate"] for r in rows if r["acc_at_gate"] is not None]
    ind = np.array([1 if a >= 0.5 else 0 for a in accs], dtype=bool)
    frac = float(ind.mean()) if ind.size else float("nan")
    k, n, p, ci = sign_test(ind) if ind.size else (0, 0, float("nan"), (float("nan"), float("nan")))
    return {"per_origin": rows, "frac_windows_acc_ge_0p5": round(frac, 3) if ind.size else None,
            "sign_test": {"k": int(k), "n": int(n), "p": float(p), "ci95": [float(ci[0]), float(ci[1])]},
            "direction_map_frozen": frozen}


# ============================================================================================
# PARTE B — confirmatorio + exploratorio, PARAMETRIZADO por par de brazos
# ============================================================================================

def part_b_confirmatory(m: pd.DataFrame, pair: tuple[str, str]) -> dict:
    """Bootstrap estacionario PAREADO (índices compartidos) de la mediana ΔSharpe(a−b). Confirmatorio."""
    pdr = paired_daily_returns(m, pair, lag=1)
    r_a, r_b = pdr["ret_a"].to_numpy(), pdr["ret_b"].to_numpy(); n = len(r_a)
    block = max(2, int(round(np.sqrt(n)))); p = 1.0 / block
    rng = np.random.default_rng(config.SEED); deltas = np.empty(B_BOOT_REPS)
    for i in range(B_BOOT_REPS):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(0, n)
        u = rng.random(n - 1); jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        deltas[i] = _sr(r_a[idx]) - _sr(r_b[idx])
    point = _sr(r_a) - _sr(r_b)
    return {"pair": f"{pair[0]}-{pair[1]}", "median_delta_sharpe": round(float(np.median(deltas)), 4),
            "ci95_boot": {"low": round(float(np.quantile(deltas, 0.025)), 4),
                          "high": round(float(np.quantile(deltas, 0.975)), 4), "point": round(point, 4)},
            # Cota inferior Bonferroni para 2 confirmatorios (M8-M5, M10-M5): IC 97.5% → cuantil 0.0125.
            # El VEREDICTO usa esta cota (controla FWER del OR); el IC95 se reporta para transparencia.
            "ci_bonf2_low": round(float(np.quantile(deltas, 0.0125)), 4),
            "block_len": int(block), "n_obs": int(n), "n_reps": B_BOOT_REPS,
            "variant": "paired_shared_block_indices_recompute_sharpe_per_arm"}


def _window_starts(n: int, window: int, step: int, anchor: bool) -> list:
    if anchor:
        return [(0, end) for end in range(window, n + 1, step)]
    return [(s, s + window) for s in range(0, n - window + 1, step)]


def per_window_metrics(m: pd.DataFrame, windows: list, pair: tuple[str, str]) -> list:
    """EXPLORATORIO. Por sub-ventana: ΔSharpe(a−b), ΔAccuracy(a−b), McNemar, n_obs."""
    a, b = pair; ca, cb = _SIZE_COL[a], _SIZE_COL[b]; out = []
    for ini, fin in windows:
        sub = m.iloc[ini:fin]
        sr_a, sr_b = _sr(sub[f"nr_{a}_causal"].to_numpy()), _sr(sub[f"nr_{b}_causal"].to_numpy())
        corr_a, corr_b = directional_correct(sub, ca), directional_correct(sub, cb)
        acc_a = float(corr_a.mean()) if corr_a.size else float("nan")
        acc_b = float(corr_b.mean()) if corr_b.size else float("nan")
        _, pmc, bb, cc = (mcnemar_test(corr_b.astype(bool), corr_a.astype(bool)) if corr_a.size
                          else (float("nan"), float("nan"), 0, 0))
        out.append({"ini": int(ini), "fin": int(fin), "dsharpe": round(sr_a - sr_b, 4),
                    "dacc": round(acc_a - acc_b, 4), "mcnemar_p": float(pmc), "n_obs": int(corr_a.size)})
    return out


def aggregate_windows(per_window: list) -> dict:
    ds = np.array([w["dsharpe"] for w in per_window if not np.isnan(w["dsharpe"])])
    if ds.size == 0:
        return {"frac_positive": None, "n_eff_bartlett": 0.0, "rho_lag1": 0.0,
                "sign_test_neff": None, "median_delta_sharpe": None}
    frac = float((ds > 0).mean()); n_eff, rho = _n_eff_bartlett(ds)
    n_eff_r = max(1, int(round(n_eff))); k_eff = int(round(frac * n_eff_r))
    return {"frac_positive": round(frac, 3), "n_eff_bartlett": round(float(n_eff), 2),
            "rho_lag1": round(float(rho), 3),
            "sign_test_neff": {"k_eff": k_eff, "n_eff": n_eff_r,
                               "p": float(binomtest(k_eff, n_eff_r, 0.5, alternative="two-sided").pvalue)},
            "median_delta_sharpe": round(float(np.median(ds)), 4)}


def stratified_mcnemar(m: pd.DataFrame, pair: tuple[str, str]) -> dict:
    """EXPLORATORIO. McNemar pooled (a vs b) por estrato régimen y signo del drift + Holm."""
    a, b = pair; ca, cb = _SIZE_COL[a], _SIZE_COL[b]
    res: dict = {"pair": f"{a}-{b}", "regime": {}, "drift": {}, "holm_bonferroni": {}}
    pvals: dict = {}; names = {0: "Calma", 1: "Estrés", 2: "Crisis"}
    for s, nm in names.items():
        sub = m[m["regime_dom"] == s]
        c_a, c_b = directional_correct(sub, ca), directional_correct(sub, cb)
        if c_a.size == 0:
            res["regime"][nm] = {"mcnemar_p": None, "n_obs": 0, "b": 0, "c": 0}; continue
        _, p, bb, cc = mcnemar_test(c_b.astype(bool), c_a.astype(bool))
        bp = block_permutation_test(c_b, c_a)[1] if c_a.size > 1 else float("nan")
        res["regime"][nm] = {"mcnemar_p": float(p), "block_perm_p": float(bp), "n_obs": int(c_a.size),
                             "b": int(bb), "c": int(cc)}
        pvals[f"regime_{nm}"] = p
    drift_roll = m["r_curr"].rolling(21, min_periods=5).mean()
    label = np.where(drift_roll > 0, "alcista", "bajista")
    for nm in ("alcista", "bajista"):
        sub = m[label == nm]
        c_a, c_b = directional_correct(sub, ca), directional_correct(sub, cb); n_obs = int(c_a.size)
        if n_obs == 0:
            res["drift"][nm] = {"mcnemar_p": None, "n_obs": 0, "median_delta_sharpe": None,
                                "inconclusivo_por_n": True, "b": 0, "c": 0}; continue
        _, p, bb, cc = mcnemar_test(c_b.astype(bool), c_a.astype(bool))
        bp = block_permutation_test(c_b, c_a)[1] if n_obs > 1 else float("nan")  # √N: corrige autocorr (pre-registro)
        sr_a, sr_b = _sr(sub[f"nr_{a}_causal"].to_numpy()), _sr(sub[f"nr_{b}_causal"].to_numpy())
        res["drift"][nm] = {"mcnemar_p": float(p), "block_perm_p": float(bp), "n_obs": n_obs,
                            "median_delta_sharpe": round(sr_a - sr_b, 4),
                            "inconclusivo_por_n": bool(n_obs < STRATUM_MIN_N), "b": int(bb), "c": int(cc)}
        pvals[f"drift_{nm}"] = p
    res["holm_bonferroni"] = _holm_bonferroni(pvals)
    return res


def _oos_m5_m8(tk: str) -> dict:
    """M5/M8 frescos en el OOS de un activo del panel (HMM K=3 + GARCH re-ajustados, τ=0.5)."""
    feat_df, ret = load_features(tk)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]; sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    mtk = run_master(gamma, sigma, oos_ret, load_agent(tk))
    sr5, sr8 = _sr(mtk["nr_m5_causal"].to_numpy()), _sr(mtk["nr_m8_causal"].to_numpy())
    st_cal = hmm.predict_states(calib.to_numpy()); rc = calib["r"].to_numpy()
    mu_cal = {s: (float(rc[st_cal == s].mean()) if (st_cal == s).any() else 0.0) for s in range(3)}
    oos_feat = feat_df.loc[feat_df.index >= pd.Timestamp(STRATA_OOS_START)].iloc[:PANEL_PRIOR_OOS_DAYS]
    flip = False
    if len(oos_feat) > 5:
        st60 = hmm.predict_states(oos_feat.to_numpy()); r60 = oos_feat["r"].to_numpy()
        for s in (0, 2):
            if (st60 == s).any() and abs(mu_cal[s]) > 1e-5 and np.sign(r60[st60 == s].mean()) != np.sign(mu_cal[s]):
                flip = True
    return {"drift_oos": round(float(oos_ret.mean() * 252), 3), "delta_sharpe_M8_M5": round(sr8 - sr5, 3),
            "prior_flip_calib_oos": bool(flip)}


def panel_drift_correlation() -> dict:
    """EXPLORATORIO. Por activo: drift_oos vs ΔSharpe(M8−M5); Spearman ρ con IC (sin p; n=10)."""
    rows = []
    for tk in PANEL:
        try:
            r = _oos_m5_m8(tk); r["ticker"] = tk; rows.append(r)
            print(f"  panel {tk:6} drift={r['drift_oos']:+.2f} ΔSharpe(M8-M5)={r['delta_sharpe_M8_M5']:+.2f} flip={r['prior_flip_calib_oos']}")
        except Exception as e:  # noqa: BLE001
            print(f"  panel {tk}: ERROR {e!r}")
    dr = np.array([r["drift_oos"] for r in rows]); dd = np.array([r["delta_sharpe_M8_M5"] for r in rows])
    rho = float(spearmanr(dr, dd)[0]) if len(rows) > 2 else float("nan")
    rng = np.random.default_rng(config.SEED); boots = []
    for _ in range(2000):
        ix = rng.integers(0, len(dr), len(dr))
        if len(np.unique(dr[ix])) > 1 and len(np.unique(dd[ix])) > 1:
            boots.append(spearmanr(dr[ix], dd[ix])[0])
    ci = [round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)] if boots else [None, None]
    return {"per_asset": rows,
            "spearman_drift_vs_delta": {"rho": round(rho, 3) if not np.isnan(rho) else None, "ci95": ci},
            "note": "n=10 subpotente: solo signo de rho + IC bootstrap; sin p-valor."}


# ============================================================================================
# Sanity dual + tabla maestra + Deflated Sharpe
# ============================================================================================

def sanity_dual(m: pd.DataFrame) -> dict:
    c, s = paired_daily_returns(m, ("m8", "m5"), lag=1), paired_daily_returns(m, ("m8", "m5"), lag=0)
    sr8c, sr5c = _sr(c["ret_a"].to_numpy()), _sr(c["ret_b"].to_numpy())
    sr8s, sr5s = _sr(s["ret_a"].to_numpy()), _sr(s["ret_b"].to_numpy())
    return {"sharpe_causal": {"m5": round(sr5c, 3), "m8": round(sr8c, 3)},
            "sharpe_same_day": {"m5": round(sr5s, 3), "m8": round(sr8s, 3)},
            "sign_consistent": bool(np.sign(sr8c - sr5c) == np.sign(sr8s - sr5s))}


def master_table(m: pd.DataFrame) -> dict:
    """TABLA MAESTRA (acc+auc+log_loss+brier+mcc+sharpe+equity) global para M5, M8, M10."""
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    y = (m.loc[valid, "r_next"] > 0).astype(int).to_numpy(); out = {}
    for arm, col, nrcol in (("m5", "agent_size", "nr_m5_causal"), ("m8", "final_size", "nr_m8_causal"),
                            ("m10", "m10_size", "nr_m10_causal")):
        score = (np.sign(m.loc[valid, col].to_numpy()) + 1.0) / 2.0
        cm = classification_metrics(y, score); nr = m[nrcol].dropna()
        out[arm] = {"accuracy": round(cm["accuracy"], 4), "auc": cm["auc"], "log_loss": cm["log_loss"],
                    "brier": round(cm["brier"], 4), "mcc": round(cm["mcc"], 4),
                    "sharpe": round(_sr(nr.to_numpy()), 3),
                    "equity_final": round(float(equity_curve(nr).iloc[-1]), 4)}
    return out


def deflated_sharpe_arm(m: pd.DataFrame, arm: str) -> dict:
    nr = m[f"nr_{arm}_causal"].dropna().to_numpy()
    sr = float(nr.mean() / nr.std(ddof=1)) if nr.std(ddof=1) > 0 else 0.0
    dsr = deflated_sharpe(sr, n_trials=N_TRIALS_DSR, n_obs=len(nr),
                          skew=float(_skew(nr)), kurt=float(_kurtosis(nr, fisher=False)))
    return {"dsr": round(float(dsr), 4), "n_trials": N_TRIALS_DSR, "n_obs": int(len(nr)), "sr_observed": round(sr, 4)}


# ============================================================================================
# Orquestación
# ============================================================================================

def main() -> None:
    reset_thresholds_cache()
    feat_df, ret = load_features(TICKER)
    gamma_df, sigma, oos_ret = build_market_states_oos()
    m = run_master(gamma_df, sigma, oos_ret, load_agent(TICKER))
    m = add_m10(m, oos_ret)
    n = len(m)
    print(f"master OK · n={n} días · OOS {oos_ret.index.min().date()}→{oos_ret.index.max().date()} · M10 añadido")

    part_a = {"heldout_ll": part_a_heldout_ll(feat_df), "directional": part_a_directional(feat_df, ret)}
    print(f"Parte A · k3_domina_frac={part_a['heldout_ll']['k3_domina_frac']} "
          f"dir_frac≥0.5={part_a['directional']['frac_windows_acc_ge_0p5']}")

    CONF_PAIRS = [("m8", "m5"), ("m10", "m5"), ("m10", "m8")]
    part_b_conf = {f"{a}_vs_{b}": part_b_confirmatory(m, (a, b)) for a, b in CONF_PAIRS}
    for key, cb in part_b_conf.items():
        ci = cb["ci95_boot"]
        print(f"Confirmatorio {key}: mediana ΔSharpe={cb['median_delta_sharpe']:+.2f} "
              f"IC95=[{ci['low']:+.2f},{ci['high']:+.2f}] {'INCLUYE 0' if ci['low'] <= 0 <= ci['high'] else 'EXCLUYE 0'}")

    sliding = _window_starts(n, B_WINDOW, B_SLIDING_STEP, anchor=False)        # 150/15 (enmienda)
    sliding_legacy = _window_starts(n, 120, 5, anchor=False)                   # 120/5 (reportado, no veredicto)
    anchor = _window_starts(n, B_WINDOW, B_ANCHOR_STEP, anchor=True)
    disjoint = _window_starts(n, B_WINDOW, B_WINDOW, anchor=False)

    def pack(windows):
        out = {}
        for a, b in CONF_PAIRS:
            w = per_window_metrics(m, windows, (a, b))
            d = {"windows": w, **aggregate_windows(w)}
            if a == "m10":  # M10 por ventana = corte descriptivo de un OOF global, NO walk-forward estricto
                d["m10_cv_not_walkforward"] = True
            out[f"{a}_vs_{b}"] = d
        return out
    part_b_sliding, part_b_anchor, part_b_disjoint = pack(sliding), pack(anchor), pack(disjoint)
    part_b_sliding_legacy = pack(sliding_legacy)  # ventana antigua 120/5, reportada para transparencia
    print(f"Sliding 150/15 ({len(sliding)} ventanas): "
          + "  ".join(f"{k} frac+={v['frac_positive']}" for k, v in part_b_sliding.items()))
    print(f"Sliding 120/5 (legacy, {len(sliding_legacy)} ventanas): "
          + "  ".join(f"{k} frac+={v['frac_positive']}" for k, v in part_b_sliding_legacy.items()))

    strat = {f"{a}_vs_{b}": stratified_mcnemar(m, (a, b)) for a, b in [("m8", "m5"), ("m10", "m5")]}
    panel = panel_drift_correlation()
    sanity = sanity_dual(m)
    master = master_table(m)
    dsr = {"m8": deflated_sharpe_arm(m, "m8"), "m10": deflated_sharpe_arm(m, "m10")}

    # --- Veredicto: dos confirmatorios (M8 vs M5, M10 vs M5) + falsificación por brazo ---
    da = part_a["directional"]
    part_a_consistente = (da["frac_windows_acc_ge_0p5"] is not None and da["frac_windows_acc_ge_0p5"] > 0.5
                          and da["sign_test"]["p"] < ALPHA)
    # Veredicto usa la cota Bonferroni (IC 97.5%/brazo) por los 2 confirmatorios → controla FWER del OR.
    h1_b_m8 = part_b_conf["m8_vs_m5"]["ci_bonf2_low"] > 0
    h1_b_m10 = part_b_conf["m10_vs_m5"]["ci_bonf2_low"] > 0

    def _falsif(strat_pair):
        baj = strat_pair["drift"].get("bajista", {"inconclusivo_por_n": True, "median_delta_sharpe": None})
        return bool((not baj["inconclusivo_por_n"]) and baj["median_delta_sharpe"] is not None
                    and baj["median_delta_sharpe"] < 0)
    falsif_m8 = _falsif(strat["m8_vs_m5"]); falsif_m10 = _falsif(strat["m10_vs_m5"])
    sp = panel["spearman_drift_vs_delta"]
    limite_panel = (sp["rho"] is not None and sp["rho"] > 0 and sp["ci95"][0] is not None and sp["ci95"][0] > 0)
    if (h1_b_m8 or h1_b_m10) and part_a_consistente:
        comp = "robustez_sostenida"
    elif part_a_consistente:
        comp = "modelo_sostenido_rescate_no_concluyente"
    elif h1_b_m8 or h1_b_m10:
        comp = "rescate_sin_modelo_investigar"
    else:
        comp = "robustez_no_sostenida"

    result = {
        "meta": {"ticker": TICKER, "panel": PANEL, "oos_start": str(oos_ret.index.min().date()),
                 "oos_end": str(oos_ret.index.max().date()), "n_days": int(n),
                 "n_obs": int(m["r_next"].notna().sum()), "signal_lag": 1, "ram_tau": TAU_RAM, "alpha": ALPHA,
                 "seed": config.SEED, "b_window": B_WINDOW, "b_sliding_step": B_SLIDING_STEP,
                 "n_sliding_windows": len(sliding), "stratum_min_n": STRATUM_MIN_N, "n_trials_dsr": N_TRIALS_DSR,
                 "limite_18m": "rescate medible solo en sub-trozos del OOS (~18 meses); NO años distintos",
                 "calibration_window": [CALIBRATION_START, CALIBRATION_END],
                 "hash_cache_agent": _hash_dir(CACHE_AGENT_DIR / TICKER), "hash_cache_models": _hash_dir(CACHE_MODELS_DIR)},
        "part_a": part_a, "part_b_confirmatory": part_b_conf, "part_b_sliding": part_b_sliding,
        "part_b_sliding_legacy_120_5": part_b_sliding_legacy,
        "part_b_anchor": part_b_anchor, "part_b_disjoint": part_b_disjoint, "stratified_mcnemar": strat,
        "deflated_sharpe": dsr, "sanity_dual": sanity, "master_table": master, "panel_drift": panel,
        "verdict": {"part_a_consistente": bool(part_a_consistente), "h1_b_m8_vs_m5": bool(h1_b_m8),
                    "h1_b_m10_vs_m5": bool(h1_b_m10), "falsif_spy_m8_bajista": bool(falsif_m8),
                    "falsif_spy_m10_bajista": bool(falsif_m10), "limite_panel_drift": bool(limite_panel),
                    "composicion": comp, "comentario": "Veredicto neutral; causa la dictamina @rigor-matematico."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # --- Validación de claves ---
    loaded = json.loads(OUT.read_text())
    for key in ("meta", "part_a", "part_b_confirmatory", "part_b_sliding", "part_b_anchor", "part_b_disjoint",
                "stratified_mcnemar", "deflated_sharpe", "sanity_dual", "master_table", "panel_drift", "verdict"):
        assert key in loaded, f"Falta clave: {key}"
    assert loaded["meta"]["signal_lag"] == 1
    for pk in ("m8_vs_m5", "m10_vs_m5"):
        assert pk in loaded["part_b_confirmatory"], f"Falta confirmatorio {pk}"
        assert "low" in loaded["part_b_confirmatory"][pk]["ci95_boot"]
        assert "ci_bonf2_low" in loaded["part_b_confirmatory"][pk], "Falta cota Bonferroni del veredicto"
        assert pk in loaded["stratified_mcnemar"] and "holm_bonferroni" in loaded["stratified_mcnemar"][pk]
        assert "bajista" in loaded["stratified_mcnemar"][pk]["drift"]
    assert "m10_vs_m5" in loaded["part_b_sliding_legacy_120_5"], "Falta la ventana legacy 120/5"
    assert loaded["part_b_sliding"]["m10_vs_m5"].get("m10_cv_not_walkforward") is True, "Falta flag M10=CV"
    for blk in ("part_b_sliding", "part_b_sliding_legacy_120_5", "part_b_anchor", "part_b_disjoint"):
        for pk in ("m8_vs_m5", "m10_vs_m5"):
            assert "n_eff" not in loaded[blk][pk]
            for w in loaded[blk][pk]["windows"]:
                assert "n_obs" in w
    for arm in ("m5", "m8", "m10"):
        for k in ("accuracy", "auc", "log_loss", "brier", "mcc", "sharpe", "equity_final"):
            assert k in loaded["master_table"][arm]
    assert "p" not in loaded["panel_drift"]["spearman_drift_vs_delta"]
    for k in ("part_a_consistente", "h1_b_m8_vs_m5", "h1_b_m10_vs_m5", "composicion"):
        assert k in loaded["verdict"]

    print(f"\nOK · {OUT} · h1_b(M8vsM5)={h1_b_m8} h1_b(M10vsM5)={h1_b_m10} composicion={comp}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
