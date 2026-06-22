"""Búsqueda exhaustiva de ACCURACY direccional causal — EXPLORATORIO.

Objetivo: encontrar una señal direccional que mejore la accuracy frente al bar honesto (ZeroR causal,
clase mayoritaria expansible — corto en los bajistas, largo en los alcistas), por activo y pooled.
Todo causal: la decisión en t usa solo información ≤ t y se evalúa contra sign(r_{t+1}).

Señales (todas devuelven ±1 por día, sin look-ahead):
  - ZeroR_exp : signo de la suma de signos de r hasta t (clase mayoritaria expansible). EL BAR.
  - RegK2..K5 : signo del régimen dominante data-driven (s_dom expansible) con HMM de K estados.
                K es la palanca señalada como importante y no explorada a fondo.
  - Trend20/63/126 : signo del retorno acumulado trailing (momentum de series temporales).
  - MA50_200 : cruce de medias (Golden/Death cross) — signo de SMA50 − SMA200.
  - Reg3+Trend : tendencia por defecto; en Crisis fiable (crisis_prob>τ) se voltea a corto.
  - Trend+RegOff : tendencia, salvo que el régimen Crisis fiable la apague (a corto).

Para cada señal: accuracy OOS, sign-test, McNemar vs ZeroR_exp por activo y pooled cross-activo.
EXPLORATORIO (docs/). No toca caché canónica. Uso: python experiments/regime_accuracy_search.py
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
from core.hmm import RegimeHMM
from core.stats import mcnemar_test, sign_test
from core.validation import panel_pooled_test

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA", "IWM"]
K_VALUES = [2, 3, 4, 5]
TREND_W = [20, 63, 126]
TAU_CRISIS = 0.5      # confianza mínima para que el régimen Crisis apague/voltee la tendencia
OUT = Path("outputs/experiments/regime_accuracy_search.json")


def _load_close(tk: str) -> pd.Series:
    pqs = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))
    end = pqs[-1].rsplit("_", 1)[1].replace(".parquet", "")
    return data.load_market_data(tk, CALIBRATION_START, end)["Close"]


def _feat(ret: pd.Series) -> pd.DataFrame:
    rv = features.realized_vol_annualized(ret, window=21)
    return pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()


def _s_dom(state: np.ndarray, ret: pd.Series, idx: pd.Index, k: int) -> np.ndarray:
    """Signo del régimen dominante, expansible y causal, para K estados."""
    rnext = ret.shift(-1).reindex(idx).to_numpy()
    sums = np.zeros(k); cnts = np.zeros(k); s = np.zeros(len(idx))
    for i in range(len(idx)):
        st = int(state[i])
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        s[i] = np.sign(means[st])
        if not np.isnan(rnext[i]):
            sums[st] += rnext[i]; cnts[st] += 1
    return s


def _regime(ret: pd.Series, k: int):
    """HMM de K estados (recalibrado en memoria ≤2024-09) → state filtrado + crisis_prob."""
    feat_df = _feat(ret)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=k, seed=config.SEED).fit(calib.to_numpy())
    gamma = hmm.predict_proba_filtered(feat_df.to_numpy())
    state = gamma.argmax(1)
    crisis_prob = gamma[:, -1]                      # estado de mayor vol (último tras el orden)
    return feat_df.index, state, crisis_prob


def run_asset(tk: str) -> dict:
    close = _load_close(tk)
    ret = features.log_returns(close).dropna()
    fidx = _feat(ret).index
    oos_mask = fidx >= pd.Timestamp(STRATA_OOS_START)
    oos_idx = fidx[oos_mask]
    rnext = ret.shift(-1).reindex(oos_idx)
    ev_mask = rnext.notna() & (np.sign(rnext) != 0)
    ev = oos_idx[ev_mask]
    truth = np.sign(rnext.reindex(ev).to_numpy())

    sig = {}  # nombre -> señal ±1 alineada a ev

    # ZeroR causal: clase mayoritaria expansible (signo de cumsum de sign(r) hasta t)
    sgn = np.sign(ret.reindex(fidx).to_numpy())
    maj = pd.Series(np.sign(np.cumsum(np.nan_to_num(sgn))), index=fidx)
    maj = maj.replace(0, 1.0)
    sig["ZeroR_exp"] = maj.reindex(ev).to_numpy()

    # Régimen K-sweep
    reg_state = {}
    for k in K_VALUES:
        idx_k, state_k, cp_k = _regime(ret, k)
        s = _s_dom(state_k, ret, idx_k, k)
        ser = pd.Series(s, index=idx_k)
        sig[f"RegK{k}"] = np.where(ser.reindex(ev).fillna(0).to_numpy() != 0,
                                   ser.reindex(ev).fillna(0).to_numpy(), 1.0)
        reg_state[k] = (idx_k, state_k, cp_k)

    # Trend (momentum de series temporales): signo del retorno acumulado trailing (causal)
    for w in TREND_W:
        tr = ret.rolling(w).sum().reindex(ev)            # incluye r_t (conocido al cierre t)
        sig[f"Trend{w}"] = np.where(np.sign(tr.fillna(0).to_numpy()) != 0,
                                    np.sign(tr.fillna(0).to_numpy()), 1.0)

    # Cruce de medias 50/200
    sma50 = close.rolling(50).mean(); sma200 = close.rolling(200).mean()
    ma = np.sign((sma50 - sma200).reindex(ev).fillna(0).to_numpy())
    sig["MA50_200"] = np.where(ma != 0, ma, 1.0)

    # Combos régimen(3) + tendencia(63)
    idx3, state3, cp3 = reg_state[3]
    cp3s = pd.Series(cp3, index=idx3).reindex(ev).fillna(0).to_numpy()
    crisis_on = cp3s > TAU_CRISIS
    trend63 = sig["Trend63"]
    sig["Trend_RegFlip"] = np.where(crisis_on, -1.0, trend63)    # en Crisis fiable, corto
    # versión que en Crisis va a la dirección del régimen (no corto fijo)
    sdom3 = pd.Series(_s_dom(state3, ret, idx3, 3), index=idx3).reindex(ev).fillna(0).to_numpy()
    sig["Trend_RegDir"] = np.where(crisis_on & (sdom3 != 0), sdom3, trend63)

    # --- Meta-señales causales: "sigue al mejor experto" y voto ponderado por accuracy ---
    # Expertos base (señales ±1 ya alineadas a ev). En cada día i, su accuracy hasta i-1 manda.
    experts = ["ZeroR_exp", "RegK3", "Trend63", "Trend126", "MA50_200"]
    E = np.vstack([sig[e] for e in experts])              # (n_exp, T)
    correct_e = (E == truth).astype(float)               # acierto por experto y día
    T = len(truth)
    cum = np.cumsum(correct_e, axis=1)                    # aciertos acumulados inclusivos
    best = np.zeros(T); wvote = np.zeros(T)
    for i in range(T):
        if i == 0:                                        # sin historia → ZeroR
            best[i] = sig["ZeroR_exp"][i]; wvote[i] = sig["ZeroR_exp"][i]; continue
        acc_prev = cum[:, i - 1] / i                      # accuracy hasta i-1 (causal)
        best[i] = E[int(np.argmax(acc_prev)), i]
        w = np.clip(acc_prev - 0.5, 0, None)              # peso = ventaja sobre 0.5
        v = np.sum(w * E[:, i])
        wvote[i] = np.sign(v) if v != 0 else sig["ZeroR_exp"][i]
    sig["BestExpert"] = best
    sig["WeightVote"] = wvote

    out = {}
    zr = (sig["ZeroR_exp"] == truth).astype(float)
    for name, s in sig.items():
        correct = (s == truth).astype(float)
        k_, n_, sp, _ = sign_test(correct.astype(bool))
        _, mp, b, c = mcnemar_test(correct.astype(bool), zr.astype(bool))
        out[name] = {"acc": round(float(correct.mean()), 4), "sign_p": round(sp, 4),
                     "mcnemar_vs_zeror_p": round(mp, 4), "b_c": [b, c],
                     "_correct": correct, "_dates": ev}
    return {"signals": out, "frac_up": round(float((truth > 0).mean()), 4), "n": int(len(ev))}


def main() -> None:
    config.set_seeds(config.SEED)
    S = {}
    for tk in PANEL:
        try:
            S[tk] = run_asset(tk)
            zr = S[tk]["signals"]["ZeroR_exp"]["acc"]
            best = max((v["acc"], n) for n, v in S[tk]["signals"].items() if n != "ZeroR_exp")
            print(f"{tk:5s} OK  ZeroR_exp={zr:.3f} | mejor={best[1]} {best[0]:.3f} (n={S[tk]['n']})", flush=True)
        except Exception as e:  # noqa: BLE001
            S[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if "error" not in S[t]]
    names = [n for n in S[ok[0]]["signals"]]

    # Pooled cross-activo de cada señal vs ZeroR_exp (clusterizado por fecha)
    pooled = {}
    for name in names:
        if name == "ZeroR_exp":
            continue
        dts = np.concatenate([np.asarray(S[t]["signals"][name]["_dates"]) for t in ok])
        delta = np.concatenate([S[t]["signals"][name]["_correct"] - S[t]["signals"]["ZeroR_exp"]["_correct"] for t in ok])
        beat = sum(S[t]["signals"][name]["acc"] > S[t]["signals"]["ZeroR_exp"]["acc"] for t in ok)
        pooled[name] = {**panel_pooled_test(delta, dts), "acc_gt_zeror": f"{beat}/{len(ok)}",
                        "acc_media": round(float(np.mean([S[t]["signals"][name]["acc"] for t in ok])), 4)}
    zr_mean = round(float(np.mean([S[t]["signals"]["ZeroR_exp"]["acc"] for t in ok])), 4)

    por_activo = {t: {"frac_up": S[t]["frac_up"], "n": S[t]["n"],
                      "signals": {n: {kk: v[kk] for kk in ("acc", "sign_p", "mcnemar_vs_zeror_p", "b_c")}
                                  for n, v in S[t]["signals"].items()}} for t in ok}

    res = {"meta": {"panel": PANEL, "n_activos": len(ok), "k_values": K_VALUES, "trend_w": TREND_W,
                    "tau_crisis": TAU_CRISIS, "seed": config.SEED, "signal_lag": 1,
                    "bar": "ZeroR_exp (clase mayoritaria expansible, causal)",
                    "objetivo": "ACCURACY direccional; rival = ZeroR causal (no B&H)",
                    "nota": "EXPLORATORIO (docs/). Se reportan TODAS las señales; el pooled vs ZeroR y "
                            "el conteo acc>ZeroR son los tests honestos. Multiplicidad alta (descontar)."},
           "zeror_acc_media": zr_mean, "pooled_vs_zeror": pooled, "por_activo": por_activo}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=_jd))
    assert set(res) >= {"meta", "pooled_vs_zeror", "por_activo"}

    print(f"\nZeroR_exp acc media = {zr_mean}")
    print(f"{'señal':14s} {'acc media':>10s} {'acc>ZeroR':>10s} {'Δpooled':>9s} {'p_greater':>10s}")
    for name in names:
        if name == "ZeroR_exp":
            continue
        p = pooled[name]
        print(f"{name:14s} {p['acc_media']:>10.4f} {p['acc_gt_zeror']:>10s} {p['delta']:>+9.4f} {p['p_greater']:>10.4f}")
    print(f"\nOK · {OUT}")


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    main()
