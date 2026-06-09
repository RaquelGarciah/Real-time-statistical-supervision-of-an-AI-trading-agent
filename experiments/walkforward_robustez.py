"""Validación walk-forward / robustez multi-ventana de STRATA (SPY).

Pendiente nº1 del tutor: "lánzalo en diferentes años/momentos; puede que tuvieras
suerte en el periodo". Por el constraint duro de que el agente LLM solo existe en el OOS
post-cutoff de DeepSeek (2024-10→), la validación se PARTE EN DOS:

  PARTE A — robustez del MODELO de régimen (24 años, SIN agente). EXPLORATORIO/descriptivo.
    Rolling-origin / time-series CV (Tashman 2000; Bergmeir-Benítez 2012) sobre 2000–2024-09:
    held-out log-likelihood por observación de HMM K∈{2,3,4} en orígenes anuales (incluye
    2008/2020/2022) + informatividad direccional del régimen (acierto del mapeo régimen→signo,
    CONGELADO en el tramo de ajuste, entre días con confianza ≥ τ=0.5). Mide robustez
    INTER-RÉGIMEN/INTER-ÉPOCA: aquí recae enteramente la respuesta al "tuviste suerte".

  PARTE B — robustez del RESCATE (M8 vs M5, SOLO dentro del OOS). Mide ESTABILIDAD INTRA-OOS:
    re-muestreos de un ÚNICO OOS alcista (~400 días). NO es robustez inter-régimen ni
    inter-época (todas las sub-ventanas viven en el mismo tramo alcista). Es un test de
    fragilidad de la lectura global, no de generalización temporal.

    TEST CONFIRMATORIO ÚNICO (el que dicta el veredicto B): mediana de ΔSharpe(M8−M5) con IC95
    por bootstrap estacionario que excluye 0 por arriba, bootstrapeando la SERIE DIARIA de la
    diferencia de retornos pareada (no la serie pre-agregada de ΔSharpe por ventana) y
    recomputando Sharpe en cada réplica con ÍNDICES DE BLOQUE COMPARTIDOS entre los dos brazos.
    Bloque medio √N (Politis-Romano 1994).

    EXPLORATORIO/SANITY (no entra al veredicto): sliding/anchor/disjoint por ventana, McNemar
    por ventana, McNemar estratificado (Holm-Bonferroni), Deflated Sharpe de M8, sanity dual
    same-day/causal, tabla maestra agregada, correlación drift-ΔSharpe del panel.

NO-INDEPENDENCIA de ventanas solapadas: el confirmatorio NO usa la serie de ΔSharpe por
ventana (bootstrapea la serie diaria pareada → no infla N). Los descriptivos por ventana usan
N efectivo de Bartlett (1946) N·(1−ρ)/(1+ρ), NO el apaño N/(window/step).

Pre-registro: BITACORA.md [2026-06-09] [Pre-registro] - Validación walk-forward / robustez.

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
from core import data, features, metrics
from core.backtest import run_backtest
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.metrics import classification_metrics, equity_curve
from core.stats import block_permutation_test, deflated_sharpe, mcnemar_test, sign_test
from strata.detectors import reset_thresholds_cache
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

# --- Configuración congelada (pre-registro) -----------------------------------------------
TICKER = "SPY"
PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
TAU_RAM = 0.5
RAM_THRESHOLDS = (TAU_RAM / 2, TAU_RAM, 0.70)  # gate efectivo = medium = τ (entrada 2026-06-09)

# Parte A — rolling-origin del modelo de régimen (EXPLORATORIO/descriptivo).
A_K_GRID = (2, 3, 4)
A_ORIGINS = tuple(range(2008, 2024))   # orígenes anuales: incluyen 2008/2020/2022
A_HORIZON_DAYS = 252                   # bloque de evaluación held-out (un año bursátil)
A_K3_DOM_FRAC_REF = 0.70               # REFERENCIA descriptiva del "K=3≫K2" global (NO umbral)

# Parte B — rolling-origin del rescate dentro del OOS.
B_WINDOW = 120
B_SLIDING_STEP = 5
B_ANCHOR_STEP = 20
ALPHA = 0.10                           # nivel pre-declarado (baja potencia con N≈400)
B_BOOT_REPS = 2000                     # réplicas del bootstrap estacionario confirmatorio
N_TRIALS_DSR = 3                       # configuraciones de ventana (sliding/anchor/disjoint)
STRATUM_MIN_N = 60                     # tamaño mínimo del estrato bajista para concluir falsif.
PANEL_PRIOR_OOS_DAYS = 60              # ventana OOS para chequear estabilidad de signo (lección #6)

ANN = np.sqrt(252)
OUT = Path("outputs/experiments/walkforward_robustez.json")


def _hash_dir(path: Path, pattern: str = "*") -> str:
    """Hash de reproducibilidad del contenido de un directorio de caché."""
    h = hashlib.sha256()
    for fp in sorted(path.glob(pattern)):
        if fp.is_file():
            h.update(fp.name.encode())
            h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def _sr(a: np.ndarray) -> float:
    """Sharpe anualizado de un array de retornos diarios (mismo criterio que metrics.sharpe)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _n_eff_bartlett(x: np.ndarray) -> tuple[float, float]:
    """N efectivo de Bartlett (1946) para una serie autocorrelacionada.

    ``N_eff = N · (1 − ρ̂) / (1 + ρ̂)`` con ρ̂ = autocorrelación lag-1. Reemplaza al apaño
    ``N/(window/step)`` (sin base, hacía N_eff≈2 e imposibilitaba cruzar α). Devuelve
    ``(n_eff, rho_lag1)``. Solo informativo (los descriptivos por ventana no deciden).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3 or x.std() == 0:
        return float(n), 0.0
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    rho = max(min(rho, 0.999), -0.999)
    n_eff = n * (1 - rho) / (1 + rho)
    return float(n_eff), rho


def _holm_bonferroni(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni (Holm 1979) sobre un conjunto de p-valores etiquetados.

    Devuelve por etiqueta ``{p_raw, p_adj, reject}`` controlando el FWER al nivel ``alpha``.
    Aplicado a los McNemar estratificados (régimen + signo de drift).
    """
    valid = [(k, v) for k, v in pvals.items() if v is not None and not np.isnan(v)]
    m = len(valid)
    out: dict[str, dict] = {}
    if m == 0:
        return {k: {"p_raw": None, "p_adj": None, "reject": False} for k in pvals}
    ordered = sorted(valid, key=lambda kv: kv[1])
    still_reject = True
    prev_adj = 0.0
    for i, (k, p) in enumerate(ordered):
        p_adj = max(prev_adj, min(1.0, (m - i) * p))  # step-down monótono
        prev_adj = p_adj
        reject = still_reject and (p <= alpha / (m - i))
        if not reject:
            still_reject = False
        out[k] = {"p_raw": float(p), "p_adj": float(p_adj), "reject": bool(reject)}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p_raw": None, "p_adj": None, "reject": False}
    return out


# ============================================================================================
# Carga de datos (idéntica al notebook canónico y a psa_gso_threshold_sensitivity.py)
# ============================================================================================

def load_agent(ticker: str) -> dict[pd.Timestamp, AgentOutput]:
    """Decisiones del agente desde cache/agent/<ticker>/ (mismo loader que el canónico)."""
    out: dict[pd.Timestamp, AgentOutput] = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        d = json.load(open(fp))
        pers = {
            k: PersonalityOutput(name=k, action=v["action"], size=v["size"],
                                 confidence=v["confidence"], reasoning="")
            for k, v in d.get("personalities", {}).items()
        }
        out[pd.Timestamp(d["date"])] = AgentOutput(
            date=d["date"], ticker=d["ticker"], action=d["action"], size=d["size"],
            confidence=d["confidence"], reasoning="", personalities=pers)
    return out


def load_features(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (feat_df[r, rv], log_returns) del histórico completo 2000→data_end."""
    parquets = sorted(glob.glob(str(DATA_DIR / f"{ticker}_{CALIBRATION_START}_*.parquet")))
    data_end = parquets[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(ticker, CALIBRATION_START, data_end)
    ret = features.log_returns(prices["Close"])
    rv21 = features.realized_vol_annualized(ret, window=21)
    feat_df = pd.concat([ret.rename("r"), rv21.rename("rv")], axis=1).dropna()
    return feat_df, ret


def build_market_states_oos() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Régimen filtrado canónico (K=3) + sigma GARCH causal + oos_ret, reusando caché."""
    feat_df, ret = load_features(TICKER)
    hmm = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
    garch = pickle.load(open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "rb"))
    gamma = hmm.predict_proba_filtered(feat_df.to_numpy())
    gamma_df = pd.DataFrame(gamma, index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    return gamma_df, sigma, oos_ret


# ============================================================================================
# Construcción del master M5/M8 en el OOS (mismo cableado que notebooks/_build.py §7)
# ============================================================================================

def run_master(gamma_df: pd.DataFrame, sigma: pd.Series, oos_ret: pd.Series,
               agents: dict) -> pd.DataFrame:
    """Recorre el OOS día a día con override-C (M8) y devuelve master.

    Columnas: ``agent_size`` (M5), ``final_size`` (M8), ``ram_sev``, ``regime_dom``,
    ``intervenido``; ``r_next`` = r_{t+1}, ``r_curr`` = r_t, ``y`` = signo(r_next), y las
    series de retorno netas causales (lag=1) y same-day (lag=0) de cada brazo, vía run_backtest
    (signal_lag=1 es el único válido; lag=0 es sanity).
    """
    sup = StrataSupervisor(
        mode="override", override_variant="C", gso_mode="absolute",
        psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
        ram_thresholds=RAM_THRESHOLDS,
    )
    rows, sizing_hist = [], []
    for t in sorted(agents):
        if t not in gamma_df.index or t not in sigma.index:
            continue
        a = agents[t]
        sizing_hist.append(a.size)
        g = gamma_df.loc[t]
        ms = {"regime": {"calm_prob": float(g["Calma"]), "stress_prob": float(g["Estrés"]),
                         "crisis_prob": float(g["Crisis"]),
                         "viterbi_state": int(np.argmax(g.values))},
              "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, sizing_hist)
        rows.append({
            "date": t, "agent_size": a.size, "final_size": dec.final_size,
            "ram_sev": dec.detectors["ram"].severity,
            "regime_dom": int(np.argmax(g.values)),  # 0 Calma, 1 Estrés, 2 Crisis
            "intervenido": dec.was_intervened,
        })
    m = pd.DataFrame(rows).set_index("date")
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    m["r_curr"] = oos_ret.reindex(m.index)
    m["y"] = np.sign(m["r_next"])
    # Series de retorno netas (run_backtest aplica signal_lag y costes), alineadas a m.index.
    for arm, col in (("m5", "agent_size"), ("m8", "final_size")):
        for lag, suf in ((1, "causal"), (0, "sameday")):
            nr = run_backtest(oos_ret, m[col], signal_lag=lag)["net_return"]
            m[f"nr_{arm}_{suf}"] = nr.reindex(m.index)
    return m


def paired_daily_returns(m: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Retornos diarios netos pareados de M5 y M8 (causal lag=1 o same-day lag=0).

    Devuelve un DataFrame con ``ret_m5``, ``ret_m8`` y ``diff`` = ret_m8 − ret_m5, indexado por
    fecha, sin NaN. Base del bootstrap CONFIRMATORIO (se remuestrean los dos brazos con índices
    compartidos, no la serie de ΔSharpe por ventana).
    """
    suf = "causal" if lag == 1 else "sameday"
    out = pd.DataFrame({"ret_m5": m[f"nr_m5_{suf}"], "ret_m8": m[f"nr_m8_{suf}"]})
    out["diff"] = out["ret_m8"] - out["ret_m5"]
    return out.dropna()


def directional_correct(m: pd.DataFrame, size_col: str) -> np.ndarray:
    """Aciertos direccionales pareados: sign(size_t) == sign(r_next_t), sobre r_next≠0."""
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    pred = np.sign(m.loc[valid, size_col].to_numpy())
    truth = np.sign(m.loc[valid, "r_next"].to_numpy())
    return (pred == truth).astype(int)


# ============================================================================================
# PARTE A — robustez del modelo de régimen (24 años, sin agente) · EXPLORATORIO/descriptivo
# ============================================================================================

def part_a_heldout_ll(feat_df: pd.DataFrame) -> dict:
    """Held-out LL/obs rodante de HMM K∈{2,3,4} sobre 2000–2024-09 (Tashman 2000). DESCRIPTIVO."""
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    X = calib.to_numpy()
    idx = calib.index
    rows = []
    for origin in A_ORIGINS:
        cut = pd.Timestamp(f"{origin}-01-01")
        n_fit = int((idx < cut).sum())
        eval_pos = np.where(idx >= cut)[0]
        if n_fit < 250 or len(eval_pos) < 30:
            continue
        eval_pos = eval_pos[:A_HORIZON_DAYS]
        for K in A_K_GRID:
            h = RegimeHMM(n_states=K, seed=config.SEED).fit(X[:n_fit])  # estados ordenados por σ asc
            ll = float(h.model.score(h._standardize(X[eval_pos])) / len(eval_pos))
            rows.append({"origin": origin, "K": K, "ll_por_obs": round(ll, 4),
                         "n_obs": int(len(eval_pos))})
    by_origin: dict[int, dict] = {}
    for r in rows:
        by_origin.setdefault(r["origin"], {})[r["K"]] = r["ll_por_obs"]
    dom = sum(1 for d in by_origin.values() if d.get(3, -np.inf) > d.get(2, -np.inf))
    frac = dom / max(len(by_origin), 1)
    return {"per_origin_K": rows, "k3_domina_frac": round(float(frac), 3),
            "k3_domina_frac_ref": A_K3_DOM_FRAC_REF, "n_origins": len(by_origin),
            "label_switch_control": "estados ordenados por sigma ascendente en RegimeHMM.fit"}


def part_a_directional(feat_df: pd.DataFrame, ret: pd.Series) -> dict:
    """Informatividad direccional del régimen por ventana (gate τ=0.5). ANCLA descriptiva de A.

    Mapeo régimen→dirección CONGELADO con las medias de retorno por estado del tramo de ajuste
    [inicio, origen]; aplicado al held-out (sin look-ahead). Acierto entre días con confianza ≥ τ.
    """
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    cret = ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)]
    X = calib.to_numpy()
    idx = calib.index
    rfit_all = calib["r"].to_numpy()
    rows, frozen = [], {}
    for origin in A_ORIGINS:
        cut = pd.Timestamp(f"{origin}-01-01")
        n_fit = int((idx < cut).sum())
        eval_pos = np.where(idx >= cut)[0]
        if n_fit < 250 or len(eval_pos) < 30:
            continue
        eval_pos = eval_pos[:A_HORIZON_DAYS]
        h = RegimeHMM(n_states=3, seed=config.SEED).fit(X[:n_fit])
        states_fit = h.predict_states(X[:n_fit])
        mu = {s: (float(rfit_all[:n_fit][states_fit == s].mean()) if (states_fit == s).any() else 0.0)
              for s in range(3)}
        long_s, short_s = max(mu, key=mu.get), min(mu, key=mu.get)  # mapeo CONGELADO por media
        frozen[str(origin)] = {"long_state": int(long_s), "short_state": int(short_s),
                               "mu_state": {str(k): round(v, 6) for k, v in mu.items()}}
        g = h.predict_proba_filtered(X[:eval_pos[-1] + 1])[eval_pos]  # filtrado causal
        conf, dom = g.max(1), g.argmax(1)
        r_next = cret.shift(-1).reindex(idx[eval_pos]).to_numpy()
        pos = np.where(dom == long_s, 1.0, np.where(dom == short_s, -1.0, 0.0))
        gate = (conf >= TAU_RAM) & ~np.isnan(r_next) & (pos != 0) & (np.sign(r_next) != 0)
        acc = float((np.sign(pos[gate]) == np.sign(r_next[gate])).mean()) if gate.sum() else float("nan")
        rows.append({"origin": origin, "acc_at_gate": None if np.isnan(acc) else round(acc, 4),
                     "n_obs": int(gate.sum())})
    accs = [r["acc_at_gate"] for r in rows if r["acc_at_gate"] is not None]
    ind = np.array([1 if a >= 0.5 else 0 for a in accs], dtype=bool)
    frac = float(ind.mean()) if ind.size else float("nan")
    k, n, p, ci = sign_test(ind) if ind.size else (0, 0, float("nan"), (float("nan"), float("nan")))
    return {"per_origin": rows, "frac_windows_acc_ge_0p5": round(frac, 3) if ind.size else None,
            "sign_test": {"k": int(k), "n": int(n), "p": float(p), "ci95": [float(ci[0]), float(ci[1])]},
            "direction_map_frozen": frozen}


# ============================================================================================
# PARTE B confirmatorio — mediana ΔSharpe(M8−M5) por bootstrap sobre la serie diaria pareada
# ============================================================================================

def part_b_confirmatory(m: pd.DataFrame) -> dict:
    """TEST CONFIRMATORIO ÚNICO del veredicto B (Politis-Romano 1994, bloque √N).

    Bootstrap estacionario PAREADO: por réplica genera UN vector de índices de bloque y lo
    aplica a AMBOS brazos (M5 y M8), recomputando ``sharpe(ret_m8) − sharpe(ret_m5)``. NO
    bootstrapea ΔSharpe por ventana ni ``sharpe(diff)`` (el Sharpe de la diferencia ≠ diferencia
    de Sharpes). Éxito B = ``ci95_boot["low"] > 0``.
    """
    pdr = paired_daily_returns(m, lag=1)
    r5, r8 = pdr["ret_m5"].to_numpy(), pdr["ret_m8"].to_numpy()
    n = len(r5)
    block = max(2, int(round(np.sqrt(n))))
    p = 1.0 / block
    rng = np.random.default_rng(config.SEED)
    deltas = np.empty(B_BOOT_REPS)
    for i in range(B_BOOT_REPS):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(0, n)
        u = rng.random(n - 1)
        jumps = rng.integers(0, n, n - 1)
        for t in range(1, n):
            idx[t] = jumps[t - 1] if u[t - 1] < p else (idx[t - 1] + 1) % n
        deltas[i] = _sr(r8[idx]) - _sr(r5[idx])  # MISMOS índices en ambos brazos
    point = _sr(r8) - _sr(r5)
    return {"median_delta_sharpe": round(float(np.median(deltas)), 4),
            "ci95_boot": {"low": round(float(np.quantile(deltas, 0.025)), 4),
                          "high": round(float(np.quantile(deltas, 0.975)), 4),
                          "point": round(point, 4)},
            "block_len": int(block), "n_obs": int(n), "n_reps": B_BOOT_REPS,
            "variant": "paired_shared_block_indices_recompute_sharpe_per_arm"}


# ============================================================================================
# PARTE B exploratorio — sub-ventanas, McNemar, estratificación, panel · NO entra al veredicto
# ============================================================================================

def _window_starts(n: int, window: int, step: int, anchor: bool) -> list[tuple[int, int]]:
    """Índices (ini, fin) de las sub-ventanas. anchor=True → ventanas crecientes desde 0."""
    if anchor:
        return [(0, end) for end in range(window, n + 1, step)]
    return [(s, s + window) for s in range(0, n - window + 1, step)]


def per_window_metrics(m: pd.DataFrame, oos_ret: pd.Series,
                       windows: list[tuple[int, int]]) -> list[dict]:
    """EXPLORATORIO. Por sub-ventana: ΔSharpe(M8−M5), ΔAccuracy(M8−M5), McNemar y ``n_obs``."""
    out = []
    for ini, fin in windows:
        sub = m.iloc[ini:fin]
        sr5, sr8 = _sr(sub["nr_m5_causal"].to_numpy()), _sr(sub["nr_m8_causal"].to_numpy())
        c5, c8 = directional_correct(sub, "agent_size"), directional_correct(sub, "final_size")
        acc5 = float(c5.mean()) if c5.size else float("nan")
        acc8 = float(c8.mean()) if c8.size else float("nan")
        _, pmc, b, c = (mcnemar_test(c5.astype(bool), c8.astype(bool)) if c8.size
                        else (float("nan"), float("nan"), 0, 0))
        out.append({"ini": int(ini), "fin": int(fin), "dsharpe": round(sr8 - sr5, 4),
                    "dacc": round(acc8 - acc5, 4), "mcnemar_p": float(pmc),
                    "n_obs": int(c8.size)})
    return out


def aggregate_windows(per_window: list[dict]) -> dict:
    """EXPLORATORIO. Agregados descriptivos por ventana con N efectivo de Bartlett."""
    ds = np.array([w["dsharpe"] for w in per_window if not np.isnan(w["dsharpe"])])
    if ds.size == 0:
        return {"frac_positive": None, "n_eff_bartlett": 0.0, "rho_lag1": 0.0,
                "sign_test_neff": None, "median_delta_sharpe": None}
    frac_pos = float((ds > 0).mean())
    n_eff, rho = _n_eff_bartlett(ds)
    n_eff_r = max(1, int(round(n_eff)))
    k_eff = int(round(frac_pos * n_eff_r))
    p_neff = float(binomtest(k_eff, n_eff_r, 0.5, alternative="two-sided").pvalue)
    return {"frac_positive": round(frac_pos, 3),
            "n_eff_bartlett": round(float(n_eff), 2), "rho_lag1": round(float(rho), 3),
            "sign_test_neff": {"k_eff": k_eff, "n_eff": n_eff_r, "p": p_neff},
            "median_delta_sharpe": round(float(np.median(ds)), 4)}


def stratified_mcnemar(m: pd.DataFrame) -> dict:
    """EXPLORATORIO. McNemar pooled M8 vs M5 por estrato (régimen y signo del drift) + Holm."""
    res: dict = {"regime": {}, "drift": {}, "holm_bonferroni": {}}
    pvals: dict[str, float] = {}
    names = {0: "Calma", 1: "Estrés", 2: "Crisis"}
    for s, nm in names.items():
        sub = m[m["regime_dom"] == s]
        c5, c8 = directional_correct(sub, "agent_size"), directional_correct(sub, "final_size")
        if c8.size == 0:
            res["regime"][nm] = {"mcnemar_p": None, "n_obs": 0, "b": 0, "c": 0}
            continue
        _, p, b, c = mcnemar_test(c5.astype(bool), c8.astype(bool))
        bp = block_permutation_test(c5, c8)[1] if c8.size > 1 else float("nan")
        res["regime"][nm] = {"mcnemar_p": float(p), "block_perm_p": float(bp),
                             "n_obs": int(c8.size), "b": int(b), "c": int(c)}
        pvals[f"regime_{nm}"] = p
    drift_roll = m["r_curr"].rolling(21, min_periods=5).mean()
    label = np.where(drift_roll > 0, "alcista", "bajista")
    for nm in ("alcista", "bajista"):
        sub = m[label == nm]
        c5, c8 = directional_correct(sub, "agent_size"), directional_correct(sub, "final_size")
        n_obs = int(c8.size)
        if n_obs == 0:
            res["drift"][nm] = {"mcnemar_p": None, "n_obs": 0, "median_delta_sharpe": None,
                                "inconclusivo_por_n": True, "b": 0, "c": 0}
            continue
        _, p, b, c = mcnemar_test(c5.astype(bool), c8.astype(bool))
        sr5, sr8 = _sr(sub["nr_m5_causal"].to_numpy()), _sr(sub["nr_m8_causal"].to_numpy())
        res["drift"][nm] = {"mcnemar_p": float(p), "n_obs": n_obs,
                            "median_delta_sharpe": round(sr8 - sr5, 4),
                            "inconclusivo_por_n": bool(n_obs < STRATUM_MIN_N),
                            "b": int(b), "c": int(c)}
        pvals[f"drift_{nm}"] = p
    res["holm_bonferroni"] = _holm_bonferroni(pvals)
    return res


def _oos_m5_m8(tk: str) -> dict:
    """M5/M8 frescos en el OOS de un activo del panel (re-ajuste HMM K=3 + GARCH, τ=0.5)."""
    feat_df, ret = load_features(tk)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()),
                         index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
    mtk = run_master(gamma, sigma, oos_ret, load_agent(tk))
    sr5, sr8 = _sr(mtk["nr_m5_causal"].to_numpy()), _sr(mtk["nr_m8_causal"].to_numpy())
    # LECCIÓN #6: signo de medias por régimen calib vs primeros PANEL_PRIOR_OOS_DAYS del OOS.
    st_cal = hmm.predict_states(calib.to_numpy())
    rc = calib["r"].to_numpy()
    mu_cal = {s: (float(rc[st_cal == s].mean()) if (st_cal == s).any() else 0.0) for s in range(3)}
    oos_feat = feat_df.loc[feat_df.index >= pd.Timestamp(STRATA_OOS_START)].iloc[:PANEL_PRIOR_OOS_DAYS]
    flip = False
    if len(oos_feat) > 5:
        st60 = hmm.predict_states(oos_feat.to_numpy())
        r60 = oos_feat["r"].to_numpy()
        for s in (0, 2):  # Calma y Crisis (los direccionales)
            if (st60 == s).any() and abs(mu_cal[s]) > 1e-5:
                if np.sign(r60[st60 == s].mean()) != np.sign(mu_cal[s]):
                    flip = True
    return {"drift_oos": round(float(oos_ret.mean() * 252), 3),
            "delta_sharpe_M8_M5": round(sr8 - sr5, 3), "prior_flip_calib_oos": bool(flip)}


def panel_drift_correlation() -> dict:
    """EXPLORATORIO/descriptivo. Por activo: drift_oos vs ΔSharpe(M8−M5); Spearman ρ con IC (sin p)."""
    rows = []
    for tk in PANEL:
        try:
            r = _oos_m5_m8(tk)
            r["ticker"] = tk
            rows.append(r)
            print(f"  panel {tk:6} drift={r['drift_oos']:+.2f} ΔSharpe(M8-M5)={r['delta_sharpe_M8_M5']:+.2f} "
                  f"prior_flip={r['prior_flip_calib_oos']}")
        except Exception as e:  # noqa: BLE001
            print(f"  panel {tk}: ERROR {e!r}")
    dr = np.array([r["drift_oos"] for r in rows])
    dd = np.array([r["delta_sharpe_M8_M5"] for r in rows])
    rho = float(spearmanr(dr, dd)[0]) if len(rows) > 2 else float("nan")
    rng = np.random.default_rng(config.SEED)
    boots = []
    for _ in range(2000):
        ix = rng.integers(0, len(dr), len(dr))
        if len(np.unique(dr[ix])) > 1 and len(np.unique(dd[ix])) > 1:
            boots.append(spearmanr(dr[ix], dd[ix])[0])
    ci = ([round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)]
          if boots else [None, None])
    return {"per_asset": rows,
            "spearman_drift_vs_delta": {"rho": round(rho, 3) if not np.isnan(rho) else None, "ci95": ci},
            "note": "n=10 subpotente: solo signo de rho + IC bootstrap; sin p-valor."}


# ============================================================================================
# Sanity dual + tabla maestra (agregado OOS global; NUNCA por sub-ventana — lección #11)
# ============================================================================================

def sanity_dual(m: pd.DataFrame) -> dict:
    """SANITY. Sharpe agregado de M5/M8 en causal y same-day; verifica que el signo no se invierte."""
    c, s = paired_daily_returns(m, lag=1), paired_daily_returns(m, lag=0)
    sr5c, sr8c = _sr(c["ret_m5"].to_numpy()), _sr(c["ret_m8"].to_numpy())
    sr5s, sr8s = _sr(s["ret_m5"].to_numpy()), _sr(s["ret_m8"].to_numpy())
    return {"sharpe_causal": {"m5": round(sr5c, 3), "m8": round(sr8c, 3)},
            "sharpe_same_day": {"m5": round(sr5s, 3), "m8": round(sr8s, 3)},
            "sign_consistent": bool(np.sign(sr8c - sr5c) == np.sign(sr8s - sr5s))}


def master_table(m: pd.DataFrame) -> dict:
    """TABLA MAESTRA (lección #11) en el AGREGADO OOS GLOBAL para M5 y M8."""
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    y = (m.loc[valid, "r_next"] > 0).astype(int).to_numpy()
    out = {}
    for arm, col, nrcol in (("m5", "agent_size", "nr_m5_causal"),
                            ("m8", "final_size", "nr_m8_causal")):
        score = ((np.sign(m.loc[valid, col].to_numpy()) + 1.0) / 2.0)  # -1→0, 0→0.5, +1→1
        cm = classification_metrics(y, score)
        nr = m[nrcol].dropna()
        out[arm] = {"accuracy": round(cm["accuracy"], 4), "auc": cm["auc"],
                    "log_loss": cm["log_loss"], "brier": round(cm["brier"], 4),
                    "mcc": round(cm["mcc"], 4), "sharpe": round(_sr(nr.to_numpy()), 3),
                    "equity_final": round(float(equity_curve(nr).iloc[-1]), 4)}
    return out


def deflated_sharpe_m8(m: pd.DataFrame) -> dict:
    """Deflated Sharpe de M8 (Bailey & López de Prado 2014) con n_trials = configuraciones."""
    nr = m["nr_m8_causal"].dropna().to_numpy()
    sr_daily = float(nr.mean() / nr.std(ddof=1)) if nr.std(ddof=1) > 0 else 0.0
    dsr = deflated_sharpe(sr_daily, n_trials=N_TRIALS_DSR, n_obs=len(nr),
                          skew=float(_skew(nr)), kurt=float(_kurtosis(nr, fisher=False)))
    return {"dsr": round(float(dsr), 4), "n_trials": N_TRIALS_DSR, "n_obs": int(len(nr)),
            "sr_observed": round(sr_daily, 4)}


# ============================================================================================
# Orquestación
# ============================================================================================

def main() -> None:
    reset_thresholds_cache()
    feat_df, ret = load_features(TICKER)
    gamma_df, sigma, oos_ret = build_market_states_oos()
    agents = load_agent(TICKER)
    m = run_master(gamma_df, sigma, oos_ret, agents)
    n = len(m)
    print(f"master OK · n={n} días · OOS {oos_ret.index.min().date()}→{oos_ret.index.max().date()}")

    part_a = {
        "heldout_ll": part_a_heldout_ll(feat_df),
        "directional": part_a_directional(feat_df, ret),
    }
    print(f"Parte A · k3_domina_frac={part_a['heldout_ll']['k3_domina_frac']} "
          f"dir_frac_acc≥0.5={part_a['directional']['frac_windows_acc_ge_0p5']}")

    part_b_conf = part_b_confirmatory(m)
    print(f"Parte B confirmatorio · mediana ΔSharpe={part_b_conf['median_delta_sharpe']} "
          f"IC95=[{part_b_conf['ci95_boot']['low']}, {part_b_conf['ci95_boot']['high']}]")

    sliding = _window_starts(n, B_WINDOW, B_SLIDING_STEP, anchor=False)
    anchor = _window_starts(n, B_WINDOW, B_ANCHOR_STEP, anchor=True)
    disjoint = _window_starts(n, B_WINDOW, B_WINDOW, anchor=False)

    pw_sliding = per_window_metrics(m, oos_ret, sliding)
    pw_anchor = per_window_metrics(m, oos_ret, anchor)
    pw_disjoint = per_window_metrics(m, oos_ret, disjoint)

    part_b_sliding = {"windows": pw_sliding, **aggregate_windows(pw_sliding)}
    part_b_anchor = {"windows": pw_anchor, **aggregate_windows(pw_anchor)}
    part_b_disjoint = {"windows": pw_disjoint, **aggregate_windows(pw_disjoint)}
    print(f"Parte B sliding · frac_positive={part_b_sliding['frac_positive']} "
          f"(referencia proyecto anterior: 0.737)")

    strat = stratified_mcnemar(m)
    panel = panel_drift_correlation()
    sanity = sanity_dual(m)
    master = master_table(m)
    dsr = deflated_sharpe_m8(m)

    # --- Veredicto (criterios PRE-DECLARADOS) ---
    da = part_a["directional"]
    part_a_consistente = (da["frac_windows_acc_ge_0p5"] is not None
                          and da["frac_windows_acc_ge_0p5"] > 0.5
                          and da["sign_test"]["p"] < ALPHA)
    h1_b = part_b_conf["ci95_boot"]["low"] > 0
    bajista = strat["drift"].get("bajista", {"inconclusivo_por_n": True, "median_delta_sharpe": None})
    falsif_spy = ((not bajista["inconclusivo_por_n"]) and bajista["median_delta_sharpe"] is not None
                  and bajista["median_delta_sharpe"] < 0)
    sp = panel["spearman_drift_vs_delta"]
    limite_panel = (sp["rho"] is not None and sp["rho"] > 0
                    and sp["ci95"][0] is not None and sp["ci95"][0] > 0)
    if part_a_consistente and h1_b:
        composicion = "robustez_sostenida"
    elif part_a_consistente and not h1_b:
        composicion = "modelo_sostenido_rescate_no_concluyente"
    elif (not part_a_consistente) and h1_b:
        composicion = "contradiccion_investigar"
    else:
        composicion = "robustez_no_sostenida"

    result = {
        "meta": {
            "ticker": TICKER, "panel": PANEL,
            "oos_start": str(oos_ret.index.min().date()),
            "oos_end": str(oos_ret.index.max().date()),
            "n_days": int(n), "n_obs": int(m["r_next"].notna().sum()),
            "signal_lag": 1, "ram_tau": TAU_RAM, "alpha": ALPHA, "seed": config.SEED,
            "b_window": B_WINDOW, "b_sliding_step": B_SLIDING_STEP, "b_anchor_step": B_ANCHOR_STEP,
            "bartlett_note": "N_eff descriptivo = N·(1−ρ)/(1+ρ); confirmatorio NO usa ventanas",
            "stratum_min_n": STRATUM_MIN_N, "n_trials_dsr": N_TRIALS_DSR,
            "calibration_window": [CALIBRATION_START, CALIBRATION_END],
            "hash_cache_agent": _hash_dir(CACHE_AGENT_DIR / TICKER),
            "hash_cache_models": _hash_dir(CACHE_MODELS_DIR),
        },
        "part_a": part_a,
        "part_b_confirmatory": part_b_conf,
        "part_b_sliding": part_b_sliding,
        "part_b_anchor": part_b_anchor,
        "part_b_disjoint": part_b_disjoint,
        "stratified_mcnemar": strat,
        "deflated_sharpe_m8": dsr,
        "sanity_dual": sanity,
        "master_table": master,
        "panel_drift": panel,
        "verdict": {
            "part_a_consistente": bool(part_a_consistente),
            "h1_b_sostenida": bool(h1_b),
            "falsif_spy_estrato_bajista": bool(falsif_spy),
            "limite_panel_drift": bool(limite_panel),
            "composicion": composicion,
            "comentario": "Veredicto neutral; la causa la dictamina @rigor-matematico (paso 5).",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # --- Validación final: las claves contractadas en el pre-registro existen ---
    loaded = json.loads(OUT.read_text())
    for key in ("meta", "part_a", "part_b_confirmatory", "part_b_sliding", "part_b_anchor",
                "part_b_disjoint", "stratified_mcnemar", "deflated_sharpe_m8", "sanity_dual",
                "master_table", "panel_drift", "verdict"):
        assert key in loaded, f"Falta la clave de primer nivel: {key}"
    assert loaded["meta"]["signal_lag"] == 1, "signal_lag debe ser 1 (causal)"
    assert "n_obs" in loaded["meta"], "Falta meta.n_obs"
    for key in ("heldout_ll", "directional"):
        assert key in loaded["part_a"], f"Falta part_a.{key}"
    assert "k3_domina_frac" in loaded["part_a"]["heldout_ll"]
    assert "label_switch_control" in loaded["part_a"]["heldout_ll"]
    assert "direction_map_frozen" in loaded["part_a"]["directional"]
    assert "frac_windows_acc_ge_0p5" in loaded["part_a"]["directional"]
    for key in ("median_delta_sharpe", "ci95_boot", "block_len", "n_obs"):
        assert key in loaded["part_b_confirmatory"], f"Falta part_b_confirmatory.{key}"
    assert "low" in loaded["part_b_confirmatory"]["ci95_boot"]
    for blk in ("part_b_sliding", "part_b_anchor", "part_b_disjoint"):
        for key in ("windows", "frac_positive", "n_eff_bartlett", "rho_lag1",
                    "sign_test_neff", "median_delta_sharpe"):
            assert key in loaded[blk], f"Falta {blk}.{key}"
        assert "n_eff" not in loaded[blk], f"{blk} no debe usar el apaño n_eff = N/(window/step)"
        for w in loaded[blk]["windows"]:
            assert "n_obs" in w, f"Falta n_obs en una ventana de {blk}"
    assert "holm_bonferroni" in loaded["stratified_mcnemar"]
    assert "bajista" in loaded["stratified_mcnemar"]["drift"]
    assert "inconclusivo_por_n" in loaded["stratified_mcnemar"]["drift"]["bajista"]
    assert "sign_consistent" in loaded["sanity_dual"]
    for arm in ("m5", "m8"):
        for key in ("accuracy", "auc", "log_loss", "brier", "mcc", "sharpe", "equity_final"):
            assert key in loaded["master_table"][arm], f"Falta master_table.{arm}.{key}"
    assert "dsr" in loaded["deflated_sharpe_m8"]
    sp_chk = loaded["panel_drift"]["spearman_drift_vs_delta"]
    assert "rho" in sp_chk and "ci95" in sp_chk
    assert "p" not in sp_chk, "panel n=10 subpotente: NO se reporta p-valor"
    for key in ("part_a_consistente", "h1_b_sostenida", "falsif_spy_estrato_bajista",
                "limite_panel_drift", "composicion"):
        assert key in loaded["verdict"], f"Falta verdict.{key}"

    print(f"\nOK · {OUT} · A_consistente={result['verdict']['part_a_consistente']} "
          f"H1_B={result['verdict']['h1_b_sostenida']} "
          f"composicion={result['verdict']['composicion']}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
