"""Régimen como DIAL DE RIESGO (no interruptor de dirección) — EXPLORATORIO.

Diagnóstico previo (docs/optimizar_deteccion_regimenes_EXPLORATORIO.md): en un OOS con tendencia
no se puede batir a "siempre largo" en accuracy direccional, y voltear el régimen a corto es ruido.
PERO el régimen detecta los drawdowns con alta precisión (crisis_precision ~0,93). La hipótesis de
mejora: usar el régimen para REDUCIR exposición en Crisis (manteniéndose largo) + vol-target mejora
el resultado RIESGO-AJUSTADO (Sharpe/Calmar/drawdown) frente a B&H, aunque no la dirección.

Estrategias (todas largas en dirección = capturan la tendencia; difieren en el TAMAÑO):
  - B&H        : +1, tamaño pleno (baseline).
  - VolTgt     : +1, vol-target |w|=min(1, target/σ_t) (¿ayuda el sizing solo?).
  - Overlay50  : VolTgt × 0.5 en días de Crisis (de-risk parcial).
  - OverlayFlat: VolTgt × 0   en días de Crisis (a liquidez en Crisis).
  - RegFlip    : s_dom × VolTgt (el direccional de antes; referencia de que es peor).
Todo causal (régimen filtrado, σ GARCH causal, signal_lag=1). Objetivo: Sharpe/Calmar/maxdd vs B&H.

EXPLORATORIO: NO toca caché canónica ni docs canónicos. Uso: python experiments/regime_risk_overlay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe, sortino, turnover
from core.stats import deflated_sharpe
from core.validation import panel_pooled_test
from experiments.quant_validation_panel import build_states
import experiments.walkforward_robustez as wf

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA", "IWM"]
TARGET_ANN = 0.15                       # objetivo de volatilidad ANUAL (σ del GARCH ya viene anualizada)
CAP = 1.0                               # sin apalancamiento
STRATS = ["B&H", "VolTgt", "Overlay50", "OverlayFlat", "RegFlip"]
OUT = Path("outputs/experiments/regime_risk_overlay.json")


def _s_dom_causal(state: np.ndarray, ret: pd.Series, idx: pd.Index) -> np.ndarray:
    """Signo del régimen dominante, expansible y causal (idéntico a strata_u._regime_drift)."""
    rnext = ret.shift(-1).reindex(idx).to_numpy()
    sums = np.zeros(3); cnts = np.zeros(3); s = np.zeros(len(idx))
    for i in range(len(idx)):
        st = int(state[i])
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        s[i] = np.sign(means[st])
        if not np.isnan(rnext[i]):
            sums[st] += rnext[i]; cnts[st] += 1
    return s


def _metr(nr: pd.Series, pos: np.ndarray, truth: np.ndarray, n_trials: int) -> dict:
    eq = equity_curve(nr)
    sr = float(sharpe(nr))
    dsr = (float(deflated_sharpe(sr / np.sqrt(252), n_trials, len(nr),
                                skew=float(pd.Series(nr).skew()), kurt=float(pd.Series(nr).kurt() + 3.0)))
           if len(nr) > 2 and not np.isnan(sr) else None)
    return {"acc": round(float((np.sign(pos) == truth).mean()), 4),
            "sharpe": round(sr, 4), "sortino": round(float(sortino(nr)), 4),
            "calmar": round(float(calmar(nr)), 4), "maxdd": round(float(max_drawdown(eq)), 4),
            "equity": round(float((1 + nr.fillna(0)).prod()), 4),
            "turnover": round(float(turnover(pd.Series(pos))), 4),
            "dsr": round(dsr, 4) if dsr is not None else None, "n": int(len(nr))}


def run_asset(tk: str) -> dict:
    gamma, sigma, oos_ret = build_states(tk)
    _, ret = wf.load_features(tk)                       # ret completo (2000→hoy) para s_dom expansible
    state = gamma.to_numpy().argmax(1)
    oos = oos_ret
    idx = oos.index
    crisis_oos = (gamma.reindex(idx).to_numpy().argmax(1) == 2)
    sig = sigma.reindex(idx).to_numpy()
    vol_scale = np.where(sig > 0, np.minimum(CAP, TARGET_ANN / sig), CAP)
    s_dom_full = _s_dom_causal(state, ret.reindex(gamma.index), gamma.index)
    s_dom = pd.Series(s_dom_full, index=gamma.index).reindex(idx).fillna(0.0).to_numpy()

    r_next = oos.shift(-1)
    mask = r_next.notna() & (np.sign(r_next) != 0)
    ev = idx[mask]
    truth = np.sign(r_next.reindex(ev).to_numpy())

    pos = {
        "B&H": np.ones(len(idx)),
        "VolTgt": vol_scale.copy(),
        "Overlay50": vol_scale * np.where(crisis_oos, 0.5, 1.0),
        "OverlayFlat": vol_scale * np.where(crisis_oos, 0.0, 1.0),
        "RegFlip": np.where(s_dom != 0, s_dom, 1.0) * vol_scale,
    }
    out = {}
    for name, w in pos.items():
        ws = pd.Series(w, index=idx)
        nr = run_backtest(oos, ws, signal_lag=1)["net_return"].reindex(ev)
        p = ws.reindex(ev).fillna(0.0).to_numpy()
        out[name] = {"stats": _metr(nr, p, truth, len(STRATS)), "nr": nr.to_numpy(), "dates": ev}
    return {"strats": out, "frac_up": round(float((truth > 0).mean()), 4),
            "crisis_frac_oos": round(float(crisis_oos.mean()), 4),
            "vol_scale_med": round(float(np.median(vol_scale)), 4)}


def main() -> None:
    config.set_seeds(config.SEED)
    S = {}
    for tk in PANEL:
        try:
            S[tk] = run_asset(tk)
            b = S[tk]["strats"]["B&H"]["stats"]; o = S[tk]["strats"]["Overlay50"]["stats"]
            print(f"{tk:5s} OK  crisis_oos={S[tk]['crisis_frac_oos']:.2f} volscl={S[tk]['vol_scale_med']:.2f} | "
                  f"B&H Sh={b['sharpe']:+.2f} mdd={b['maxdd']:+.2f} | Ovl50 Sh={o['sharpe']:+.2f} mdd={o['maxdd']:+.2f}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            S[tk] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if "error" not in S[t]]

    # Conteos y pooled vs B&H
    resumen = {}
    for name in STRATS:
        if name == "B&H":
            continue
        sh_gt = sum(S[t]["strats"][name]["stats"]["sharpe"] > S[t]["strats"]["B&H"]["stats"]["sharpe"] for t in ok)
        ca_gt = sum(S[t]["strats"][name]["stats"]["calmar"] > S[t]["strats"]["B&H"]["stats"]["calmar"] for t in ok)
        # maxdd "mejor" = menos negativo (más cerca de 0)
        dd_bt = sum(S[t]["strats"][name]["stats"]["maxdd"] > S[t]["strats"]["B&H"]["stats"]["maxdd"] for t in ok)
        dts = np.concatenate([np.asarray(S[t]["strats"][name]["dates"]) for t in ok])
        pnl_d = np.concatenate([S[t]["strats"][name]["nr"] - S[t]["strats"]["B&H"]["nr"] for t in ok])
        resumen[name] = {"sharpe_gt_BH": f"{sh_gt}/{len(ok)}", "calmar_gt_BH": f"{ca_gt}/{len(ok)}",
                         "maxdd_mejor_BH": f"{dd_bt}/{len(ok)}",
                         "pnl_vs_BH": panel_pooled_test(pnl_d, dts),
                         "sharpe_medio": round(float(np.mean([S[t]["strats"][name]["stats"]["sharpe"] for t in ok])), 4),
                         "maxdd_medio": round(float(np.mean([S[t]["strats"][name]["stats"]["maxdd"] for t in ok])), 4)}
    resumen["B&H"] = {"sharpe_medio": round(float(np.mean([S[t]["strats"]["B&H"]["stats"]["sharpe"] for t in ok])), 4),
                      "maxdd_medio": round(float(np.mean([S[t]["strats"]["B&H"]["stats"]["maxdd"] for t in ok])), 4)}

    por_activo = {t: {"frac_up": S[t]["frac_up"], "crisis_frac_oos": S[t]["crisis_frac_oos"],
                      "vol_scale_med": S[t]["vol_scale_med"],
                      "strats": {n: S[t]["strats"][n]["stats"] for n in STRATS}} for t in ok}

    res = {"meta": {"panel": PANEL, "n_activos": len(ok), "estrategias": STRATS,
                    "target_vol_ann": TARGET_ANN, "cap": CAP, "seed": config.SEED, "signal_lag": 1,
                    "objetivo": "RIESGO-AJUSTADO (Sharpe/Calmar/maxdd) vs B&H; dirección = siempre larga",
                    "tesis": "el régimen como dial de riesgo (de-risk en Crisis) + vol-target, no como "
                             "interruptor de dirección. Régimen filtrado causal + σ GARCH causal.",
                    "nota": "EXPLORATORIO (docs/). No toca caché canónica. DSR n_trials=nº estrategias."},
           "resumen_vs_BH": resumen, "por_activo": por_activo}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=_jd))
    assert set(res) >= {"meta", "resumen_vs_BH", "por_activo"}

    print(f"\n{'estrategia':12s} {'Sh medio':>9s} {'mdd medio':>10s} {'Sh>BH':>7s} {'Calmar>BH':>10s} "
          f"{'mdd mejor':>10s} {'pnlΔ p':>8s}")
    print(f"{'B&H':12s} {resumen['B&H']['sharpe_medio']:>9.3f} {resumen['B&H']['maxdd_medio']:>10.3f}")
    for n in STRATS:
        if n == "B&H":
            continue
        r = resumen[n]
        print(f"{n:12s} {r['sharpe_medio']:>9.3f} {r['maxdd_medio']:>10.3f} {r['sharpe_gt_BH']:>7s} "
              f"{r['calmar_gt_BH']:>10s} {r['maxdd_mejor_BH']:>10s} {r['pnl_vs_BH']['p_greater']:>8.3f}")
    print(f"\nOK · {OUT}")


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    main()
