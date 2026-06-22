"""Optimizar la detección de régimen: reducir el RETARDO del HMM sin look-ahead — EXPLORATORIO.

El detector RAM (HMM gaussiano de 3 estados) ya se infiere de forma CAUSAL
(``predict_proba_filtered``, forward-only; ver core/hmm.py y test_filtered_no_lookahead).
El retardo no es look-ahead: es retardo real de reactividad con dos fuentes atacables:
  (1) la feature de volatilidad ``RV^21 = std(r_{t-20:t})·√252`` → ~10 días de lag medio;
  (2) la inercia de la matriz de transición (diagonal alta) → el argmax tarda en voltear.

Este script prueba variantes de estimación de régimen que reducen ese retardo manteniendo
causalidad estricta, mide el trade-off lag-vs-ruido y evalúa el resultado downstream con la
estrategia "Régimen" PURA (posición = signo del régimen dominante data-driven, sin agente ni
vol-target) sobre los 13 activos con todas las métricas.

EXPLORATORIO: NO toca cache/models/*.pkl (los HMM se recalibran en memoria por activo), NO
toca docs canónicos. Resultado en outputs/experiments/optimizar_deteccion_regimenes.json y
documentado en docs/optimizar_deteccion_regimenes_EXPLORATORIO.md.

Uso: python experiments/optimizar_deteccion_regimenes.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features
from core.backtest import run_backtest
from core.hmm import RegimeHMM
from core.metrics import calmar, equity_curve, max_drawdown, sharpe, sortino, turnover
from core.stats import deflated_sharpe, mcnemar_test, sign_test
from core.validation import panel_pooled_test

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA", "IWM"]

# Variantes (todas causales). La 2ª columna del input al HMM SIEMPRE es la vol que ordena
# los estados (Calma/Estrés/Crisis), para que el etiquetado sea comparable entre variantes.
#   vol  : ("rolling", w) → realized_vol_annualized; ("ewma", lam) → ewma_vol_annualized.
#   extra: 3ª columna informativa (solo V3): acelera la reacción sin mover la etiqueta.
#   rule : "argmax" o ("crisis_th", θ) → declara Crisis si crisis_prob > θ (más reactivo).
VARIANTS: dict[str, dict] = {
    "V0_rv21":     {"vol": ("rolling", 21), "rule": "argmax"},   # control canónico
    "V1a_rv10":    {"vol": ("rolling", 10), "rule": "argmax"},
    "V1b_rv5":     {"vol": ("rolling", 5),  "rule": "argmax"},
    "V2_ewma094":  {"vol": ("ewma", 0.94),  "rule": "argmax"},
    "V2b_ewma097": {"vol": ("ewma", 0.97),  "rule": "argmax"},
    "V3_multi":    {"vol": ("rolling", 21), "extra": ("ewma", 0.94), "rule": "argmax"},
    "V4a_th04":    {"vol": ("rolling", 21), "rule": ("crisis_th", 0.4)},
    "V4b_th03":    {"vol": ("rolling", 21), "rule": ("crisis_th", 0.3)},
}
BASE = "V0_rv21"
DD_THR = -0.10        # umbral de "entrada en drawdown" para el lag de onset
ONSET_HORIZON = 63    # días bursátiles de horizonte para detectar la Crisis tras el onset
XCORR_KMAX = 20       # rango de lag de la cross-correlación régimen↔drawdown
OUT = Path("outputs/experiments/optimizar_deteccion_regimenes.json")


# ---------------------------------------------------------------------------
# Régimen por variante (recalibrado en memoria, NUNCA serializado)
# ---------------------------------------------------------------------------

def _load_close(tk: str) -> pd.Series:
    """Serie de cierre 2000→hoy (último parquet largo del activo)."""
    pqs = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))
    end = pqs[-1].rsplit("_", 1)[1].replace(".parquet", "")
    return data.load_market_data(tk, CALIBRATION_START, end)["Close"]


def _vol(ret: pd.Series, spec: tuple) -> pd.Series:
    kind, par = spec
    if kind == "rolling":
        return features.realized_vol_annualized(ret, window=int(par))
    if kind == "ewma":
        return features.ewma_vol_annualized(ret, lam=float(par))
    raise ValueError(f"spec de vol desconocido: {spec}")


def _build_feat(ret: pd.Series, v: dict) -> pd.DataFrame:
    """Matriz de features de la variante; col 1 = vol de ordenamiento (invariante)."""
    cols = {"r": ret, "rv": _vol(ret, v["vol"])}
    if "extra" in v:
        cols["rv_extra"] = _vol(ret, v["extra"])
    df = pd.DataFrame(cols).dropna()
    assert df.columns[1] == "rv", "la 2ª columna debe ser la vol que ordena los estados"
    return df


def _fit_gamma(feat_df: pd.DataFrame) -> pd.DataFrame:
    """HMM K=3 calibrado ≤ 2024-09 (en memoria) + posterior FILTRADO causal sobre toda la serie."""
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    gamma = hmm.predict_proba_filtered(feat_df.to_numpy())
    return pd.DataFrame(gamma, index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])


def _state(gamma: pd.DataFrame, rule) -> np.ndarray:
    """Estado por día según la regla de la variante (argmax o umbral en crisis_prob)."""
    arr = gamma.to_numpy()
    am = arr.argmax(1)
    if rule == "argmax":
        return am
    _, th = rule  # ("crisis_th", θ)
    return np.where(arr[:, 2] > th, 2, am)


def _s_dom_causal(state: np.ndarray, ret: pd.Series, idx: pd.Index) -> np.ndarray:
    """Signo del régimen dominante, expansible y causal (bucle de strata_u._regime_drift).

    s_dom[t] = sign( media de r_{·+1} del estado state[t], acumulada solo hasta t-1 ).
    Data-driven (maneja el prior-flip por activo) y sin look-ahead.
    """
    rnext = ret.shift(-1).reindex(idx).to_numpy()
    n = len(idx)
    sums = np.zeros(3); cnts = np.zeros(3)
    s_dom = np.zeros(n)
    for i in range(n):
        st = int(state[i])
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        s_dom[i] = np.sign(means[st])
        if not np.isnan(rnext[i]):
            sums[st] += rnext[i]; cnts[st] += 1
    return s_dom


# ---------------------------------------------------------------------------
# Métricas de LAG (sobre la serie completa 2000→hoy: capta 2008/2020/2022)
# ---------------------------------------------------------------------------

def _xcorr_kstar(sev: np.ndarray, mag: np.ndarray, kmax: int) -> int:
    """k que maximiza corr(sev_t, mag_{t+k}). k>0 ⇒ el régimen ANTICIPA el drawdown."""
    a0 = sev - sev.mean(); b0 = mag - mag.mean()
    best_k, best = 0, -2.0
    for k in range(-kmax, kmax + 1):
        if k >= 0:
            a, b = a0[: len(a0) - k], b0[k:]
        else:
            a, b = a0[-k:], b0[: len(b0) + k]
        if len(a) < 10 or a.std() == 0 or b.std() == 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best:
            best, best_k = c, k
    return best_k


def _onset_lag(state: np.ndarray, dd: np.ndarray) -> dict:
    """Lag de detección sobre eventos objetivos de 'entrada en drawdown' (dd cruza DD_THR).

    Para cada cruce dd_{t-1}>thr ≥ dd_t, días bursátiles hasta la primera Crisis dentro de
    ONSET_HORIZON. Eventos sin detección → censurados (cuentan en la tasa de detección).
    Estos cruces incluyen automáticamente GFC-2008/COVID-2020/2022 en los índices.
    """
    entries = np.where((dd[1:] <= DD_THR) & (dd[:-1] > DD_THR))[0] + 1
    lags, det = [], 0
    for e in entries:
        win = state[e: min(e + ONSET_HORIZON, len(state))]
        hit = np.where(win == 2)[0]
        if hit.size:
            lags.append(int(hit[0])); det += 1
    n_ev = int(len(entries))
    return {"onset_n_eventos": n_ev,
            "onset_lag_mediana": round(float(np.median(lags)), 1) if lags else None,
            "onset_deteccion_rate": round(det / n_ev, 3) if n_ev else None}


def _lag_metrics(gamma: pd.DataFrame, state: np.ndarray, close: pd.Series) -> dict:
    arr = gamma.to_numpy()
    n = len(state)
    trans = int((state[1:] != state[:-1]).sum())
    runs, i = [], 0
    while i < n:
        if state[i] == 2:
            j = i
            while j < n and state[j] == 2:
                j += 1
            runs.append(j - i); i = j
        else:
            i += 1
    dd = (close / close.cummax() - 1.0).to_numpy()
    crisis = state == 2
    sev = arr[:, 1] + 2 * arr[:, 2]               # severidad esperada del estado (causal)
    out = {"whipsaw_por_anyo": round(trans / (n / 252.0), 2),
           "crisis_run_mediana": float(np.median(runs)) if runs else 0.0,
           "crisis_frac": round(float(crisis.mean()), 3),
           "crisis_precision": round(float((dd[crisis] <= -0.05).mean()), 3) if crisis.any() else None,
           "xcorr_kstar": _xcorr_kstar(sev, -dd, XCORR_KMAX)}
    out.update(_onset_lag(state, dd))
    return out


# ---------------------------------------------------------------------------
# Evaluación downstream: estrategia "Régimen" pura sobre el OOS
# ---------------------------------------------------------------------------

def _metr(nr: pd.Series, pos: np.ndarray, truth: np.ndarray) -> dict:
    eq = equity_curve(nr)
    return {"acc": round(float((np.sign(pos) == truth).mean()), 4),
            "sharpe": round(float(sharpe(nr)), 4),
            "sortino": round(float(sortino(nr)), 4),
            "equity": round(float((1 + nr.fillna(0)).prod()), 4),
            "maxdd": round(float(max_drawdown(eq)), 4),
            "calmar": round(float(calmar(nr)), 4),
            "turnover": round(float(turnover(pd.Series(pos))), 4),
            "n": int(len(nr))}


def _oos_eval(ret: pd.Series, idx: pd.Index):
    """Conjunto OOS común a todas las variantes (depende solo de precios): dates, truth, máscara."""
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    r_next = ret.shift(-1).reindex(oos.index)
    mask = r_next.notna() & (np.sign(r_next) != 0)
    ev = oos.index[mask]
    truth = np.sign(r_next.reindex(ev).to_numpy())
    return oos, ev, truth


def _downstream(pos_full: pd.Series, oos: pd.Series, ev: pd.Index, truth: np.ndarray):
    p = pos_full.reindex(ev).fillna(0.0).to_numpy()
    nr = run_backtest(oos, pos_full, signal_lag=1)["net_return"].reindex(ev)
    stats = _metr(nr, p, truth)
    k, n, sp, _ = sign_test((np.sign(p) == truth).astype(bool))
    sr = stats["sharpe"]
    dsr = float(deflated_sharpe(sr / np.sqrt(252), len(VARIANTS), n,
                               skew=float(pd.Series(nr).skew()),
                               kurt=float(pd.Series(nr).kurt() + 3.0))) if n > 2 and not np.isnan(sr) else None
    return stats, {"correct": (np.sign(p) == truth).astype(float), "nr": nr.to_numpy(),
                   "sign_p": round(sp, 4), "dsr": round(dsr, 4) if dsr is not None else None}


def run_asset(tk: str) -> dict:
    close = _load_close(tk)
    ret = features.log_returns(close).dropna()
    res = {"variantes": {}, "_arr": {}}
    # OOS común y baselines (no dependen de la variante)
    sample_idx = _build_feat(ret, VARIANTS[BASE]).index
    oos, ev, truth = _oos_eval(ret, sample_idx)
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    bh_pos = pd.Series(1.0, index=oos.index)
    zr_pos = pd.Series(maj, index=oos.index)
    bh_st, _ = _downstream(bh_pos, oos, ev, truth)
    zr_st, _ = _downstream(zr_pos, oos, ev, truth)
    res["baselines"] = {"B&H": {**bh_st, "acc": round(frac_up, 4)},
                        "ZeroR": {**zr_st, "acc": round(max(frac_up, 1 - frac_up), 4)}}
    res["frac_up"] = round(frac_up, 4)
    for name, v in VARIANTS.items():
        gamma = _fit_gamma(_build_feat(ret, v))
        state = _state(gamma, v["rule"])
        s_dom = _s_dom_causal(state, ret, gamma.index)
        lag = _lag_metrics(gamma, state, close.reindex(gamma.index))
        down, arr = _downstream(pd.Series(s_dom, index=gamma.index), oos, ev, truth)
        res["variantes"][name] = {"lag": lag, "downstream": down,
                                  "tests": {"sign_p": arr["sign_p"], "dsr": arr["dsr"]}}
        res["_arr"][name] = {"correct": arr["correct"], "nr": arr["nr"], "dates": ev}
    return res


# ---------------------------------------------------------------------------
# Agregación cross-activo
# ---------------------------------------------------------------------------

def main() -> None:
    config.set_seeds(config.SEED)
    S = {}
    for tk in PANEL:
        try:
            S[tk] = run_asset(tk)
            v0 = S[tk]["variantes"][BASE]
            print(f"{tk:5s} OK  lag_med(V0)={v0['lag']['onset_lag_mediana']} "
                  f"whip(V0)={v0['lag']['whipsaw_por_anyo']} sharpe(V0)={v0['downstream']['sharpe']:+.3f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            S[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if "error" not in S[t]]

    def _mean(name: str, getter) -> float:
        vals = [getter(S[t]["variantes"][name]) for t in ok if getter(S[t]["variantes"][name]) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    lag_global, tradeoff = {}, {}
    for name in VARIANTS:
        lag_global[name] = {
            "onset_lag_mediana": _mean(name, lambda d: d["lag"]["onset_lag_mediana"]),
            "onset_deteccion_rate": _mean(name, lambda d: d["lag"]["onset_deteccion_rate"]),
            "xcorr_kstar": _mean(name, lambda d: d["lag"]["xcorr_kstar"]),
            "whipsaw_por_anyo": _mean(name, lambda d: d["lag"]["whipsaw_por_anyo"]),
            "crisis_run_mediana": _mean(name, lambda d: d["lag"]["crisis_run_mediana"]),
            "crisis_precision": _mean(name, lambda d: d["lag"]["crisis_precision"]),
            "turnover_oos": _mean(name, lambda d: d["downstream"]["turnover"]),
            "sharpe_oos": _mean(name, lambda d: d["downstream"]["sharpe"]),
            "acc_oos": _mean(name, lambda d: d["downstream"]["acc"])}
        tradeoff[name] = {"lag_medio_onset": lag_global[name]["onset_lag_mediana"],
                          "whipsaw_por_anyo": lag_global[name]["whipsaw_por_anyo"],
                          "sharpe_medio": lag_global[name]["sharpe_oos"]}

    # Tests pooled de cada variante contra V0 (panel cross-activo, clusterizado por fecha)
    panel_tests = {}
    for name in VARIANTS:
        if name == BASE:
            continue
        acc_d, pnl_d, dts = [], [], []
        sh_gt = lag_gt = 0
        for t in ok:
            a = S[t]["_arr"]; v = S[t]["variantes"]
            acc_d.append(a[name]["correct"] - a[BASE]["correct"])
            pnl_d.append(a[name]["nr"] - a[BASE]["nr"])
            dts.append(np.asarray(a[name]["dates"]))
            sh_gt += int(v[name]["downstream"]["sharpe"] > v[BASE]["downstream"]["sharpe"])
            lv, l0 = v[name]["lag"]["onset_lag_mediana"], v[BASE]["lag"]["onset_lag_mediana"]
            lag_gt += int(lv is not None and l0 is not None and lv < l0)
        acc_d = np.concatenate(acc_d); pnl_d = np.concatenate(pnl_d); dts = np.concatenate(dts)
        panel_tests[name] = {
            "acc_vs_V0": panel_pooled_test(acc_d, dts),
            "pnl_vs_V0": panel_pooled_test(pnl_d, dts),
            "sharpe_gt_V0": f"{sh_gt}/{len(ok)}",
            "lag_menor_que_V0": f"{lag_gt}/{len(ok)}"}

    por_activo = {t: {"frac_up": S[t]["frac_up"], "baselines": S[t]["baselines"],
                      "variantes": {n: {"lag": S[t]["variantes"][n]["lag"],
                                        "downstream": S[t]["variantes"][n]["downstream"],
                                        "tests": S[t]["variantes"][n]["tests"]}
                                    for n in VARIANTS}} for t in ok}

    res = {
        "meta": {
            "panel": PANEL, "n_activos": len(ok),
            "variantes": {n: {k: vv for k, vv in v.items()} for n, v in VARIANTS.items()},
            "base": BASE, "seed": config.SEED, "signal_lag": 1,
            "n_states": 3, "covariance_type": "full", "n_seeds": 10,
            "calibration_end": CALIBRATION_END, "oos_start": STRATA_OOS_START,
            "dd_thr": DD_THR, "onset_horizon": ONSET_HORIZON,
            "estrategia": "Régimen pura (s_dom expansible causal), sin agente ni vol-target",
            "onset_def": "evento = día en que el drawdown desde el máximo móvil cruza por debajo "
                         "de -10% (objetivo y reproducible; incluye GFC-2008/COVID-2020/2022 en índices)",
            "xcorr_conv": "corr(sev_t, |dd_{t+k}|); k>0 ⇒ el régimen anticipa el drawdown",
            "nota": "EXPLORATORIO (docs/). NO toca cache/models/*.pkl: los HMM se recalibran en "
                    "memoria por activo. Se reportan TODAS las variantes; la multiplicidad de haber "
                    "explorado varias configs se descuenta con DSR (n_trials=nº variantes) y el "
                    "pooled vs V0 es el test honesto de mejora del panel."},
        "lag_global": lag_global,
        "tradeoff": tradeoff,
        "panel_tests": panel_tests,
        "por_activo": por_activo}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=_json_default))
    assert set(res) >= {"meta", "lag_global", "tradeoff", "panel_tests", "por_activo"}

    # Resumen por consola: lag de onset y Sharpe medio por variante
    print(f"\n{'variante':14s} {'lag_onset':>10s} {'detec':>7s} {'whip/año':>9s} "
          f"{'xcorr_k':>8s} {'sharpe':>8s} {'acc':>7s}")
    for n in VARIANTS:
        g = lag_global[n]
        print(f"{n:14s} {str(g['onset_lag_mediana']):>10s} {str(g['onset_deteccion_rate']):>7s} "
              f"{str(g['whipsaw_por_anyo']):>9s} {str(g['xcorr_kstar']):>8s} "
              f"{str(g['sharpe_oos']):>8s} {str(g['acc_oos']):>7s}")
    print("\npooled vs V0 (acc Δ, p_greater):")
    for n, pt in panel_tests.items():
        print(f"  {n:14s} accΔ={pt['acc_vs_V0']['delta']:+.4f} p={pt['acc_vs_V0']['p_greater']:.3f} | "
              f"sharpe>V0 {pt['sharpe_gt_V0']} | lag<V0 {pt['lag_menor_que_V0']}")
    print(f"\nOK · {OUT}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"no serializable: {type(o)}")


if __name__ == "__main__":
    main()
