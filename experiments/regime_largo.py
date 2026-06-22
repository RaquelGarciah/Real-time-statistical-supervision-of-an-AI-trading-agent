"""Backtest LARGO del canal de régimen (sin agente), sobre la ventana común a los 15 activos.

Motivación (observación de Raquel): "Régimen" (seguir el signo del régimen dominante) y las triviales
B&H/ZeroR NO dependen del agente LLM → se pueden backtestear en cualquier ventana con precios, incluida
una con crisis reales. El OOS del TFG (post-2024-10) ha sido casi siempre alcista y sin crash, así que el
valor del régimen —esquivar caídas en Crisis— nunca se materializa. Aquí se prueba con COVID-2020 y el
bear-2022 dentro.

Diseño causal (sin look-ahead):
  - Ventana común: los 15 activos existen desde 2017-09-28 (ROKU es el más joven).
  - Calibración por activo con su PROPIA historia hasta 2019-12-31 (HMM K=3 + GARCH).
  - Test común: 2020-01-01 → 2024-09-30 (incluye crash COVID y bear-2022).
  - s_dom expansible y causal: signo de la media de r_{t+1} del estado dominante, sembrado con las medias
    de calibración (≤2019) y actualizado día a día con info hasta t-1. Maneja el prior-flip.
Estrategias (todas con el MISMO sizing vol-target → Sharpe/maxDD comparables; accuracy es independiente):
  Régimen (dir=s_dom), B&H (+1), ZeroR (clase mayoritaria realizada del test = NIR). Además se reporta el
  B&H 1x (comprar y mantener real) como referencia, y un desglose por sub-periodos de crisis.

Uso: python experiments/regime_largo.py
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
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
import experiments.walkforward_robustez as wf

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]
CALIB_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2024-09-30")
TARGET_VOL, CAP = 0.10, 1.0
# sub-periodos de crisis para el desglose (esquivar caídas es donde el régimen debe lucir)
CRISIS = {"COVID-2020": ("2020-02-15", "2020-04-30"), "bear-2022": ("2022-01-01", "2022-10-15")}
OUT = Path("outputs/experiments/regime_largo.json")


def _states_causal(tk: str):
    """HMM K=3 + GARCH calibrados con la historia del activo ≤2019; γ filtrado (causal) y σ sobre test.
    s_dom expansible (sembrado con calibración, actualizado hasta t-1). Devuelve series sobre el test."""
    feat, ret = wf.load_features(tk)
    calib_feat = feat.loc[feat.index <= CALIB_END]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib_feat.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= CALIB_END])
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat.to_numpy()), index=feat.index,
                         columns=["Calma", "Estrés", "Crisis"])
    state = gamma.to_numpy().argmax(1)
    rnext_full = ret.shift(-1).reindex(feat.index).to_numpy()

    # siembra de medias por estado con calibración (≤2019), causal
    sums, cnts = np.zeros(3), np.zeros(3)
    calib_mask = feat.index <= CALIB_END
    for i in np.where(calib_mask)[0]:
        if not np.isnan(rnext_full[i]):
            sums[state[i]] += rnext_full[i]; cnts[state[i]] += 1

    test_pos = np.where((feat.index >= TEST_START) & (feat.index <= TEST_END))[0]
    idx = feat.index[test_pos]
    s_dom = np.zeros(len(test_pos))
    for j, i in enumerate(test_pos):
        st = state[i]
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        s_dom[j] = np.sign(means[st])               # signo causal (info hasta t-1)
        if not np.isnan(rnext_full[i]):
            sums[st] += rnext_full[i]; cnts[st] += 1
    oos_ret = ret.reindex(idx)
    sigma = garch.forecast_path(oos_ret)
    return idx, oos_ret, sigma, pd.Series(np.where(s_dom == 0, 1.0, s_dom), index=idx), gamma.reindex(idx)


def _metr(nrx, dir_arr, truth):
    return {"acc": round(float((dir_arr == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
            "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
            "calmar": round(float(calmar(nrx)), 4), "equity": round(float((1 + nrx.fillna(0)).prod()), 4)}


def _row(tk: str) -> dict:
    idx, oos_ret, sigma, s_dom, gamma = _states_causal(tk)
    rnext = oos_ret.shift(-1)
    valid = rnext.notna() & (np.sign(rnext) != 0)
    sub = idx[valid.to_numpy()]
    truth = np.sign(rnext.reindex(sub).to_numpy())
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    vs = np.where(sigma.reindex(sub).to_numpy() > 0,
                  np.minimum(CAP, TARGET_VOL / sigma.reindex(sub).to_numpy()), CAP)
    dirs = {"Régimen": s_dom.reindex(sub).to_numpy(),
            "B&H": np.ones_like(truth), "ZeroR": np.full_like(truth, maj)}

    def nr(d, sized=True):
        w = pd.Series(0.0, index=idx); w.loc[sub] = d * (vs if sized else 1.0)
        return run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)

    est = {nm: _metr(nr(d), d, truth) for nm, d in dirs.items()}
    est["ZeroR"]["acc"] = round(max(frac_up, 1 - frac_up), 4)
    est["B&H_1x"] = _metr(nr(np.ones_like(truth), sized=False), np.ones_like(truth), truth)  # comprar y mantener real

    # desglose por crisis: maxDD y retorno de Régimen vs B&H (vol-target) en cada sub-periodo
    crisis = {}
    for name, (a, b) in CRISIS.items():
        m = (sub >= pd.Timestamp(a)) & (sub <= pd.Timestamp(b))
        if m.sum() < 5:
            continue
        out = {}
        for nm in ("Régimen", "B&H"):
            nrx = nr(dirs[nm]).reindex(sub)[m]
            out[nm] = {"ret": round(float((1 + nrx.fillna(0)).prod() - 1), 4),
                       "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4)}
        crisis[name] = {"n": int(m.sum()), **out}

    # fracción de días que Régimen se pone corto (cuándo se separa de B&H)
    short_frac = round(float((s_dom.reindex(sub).to_numpy() < 0).mean()), 4)
    return {"n": int(len(sub)), "frac_up": round(frac_up, 4), "short_frac": short_frac,
            "estrategias": est, "crisis": crisis}


def main() -> None:
    rows = {}
    for tk in PANEL:
        try:
            rows[tk] = _row(tk)
            r = rows[tk]; e = r["estrategias"]
            print(f"{tk:5s} n={r['n']} up={r['frac_up']:.2f} short={r['short_frac']:.0%} | "
                  f"Rég acc={e['Régimen']['acc']:.3f} Sh={e['Régimen']['sharpe']:+.2f} dd={e['Régimen']['maxdd']:.0%} | "
                  f"B&H Sh={e['B&H']['sharpe']:+.2f} dd={e['B&H']['maxdd']:.0%} | ZeroR Sh={e['ZeroR']['sharpe']:+.2f}",
                  flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    A = list(rows)
    def avg(s, m): return round(float(np.mean([rows[t]["estrategias"][s][m] for t in A])), 4)
    medias = {s: {m: avg(s, m) for m in ("acc", "sharpe", "maxdd", "calmar", "equity")}
              for s in ("Régimen", "B&H", "ZeroR", "B&H_1x")}
    # conteo Régimen > B&H y > ZeroR
    cob = {b: {m: f"{sum(rows[t]['estrategias']['Régimen'][m] > rows[t]['estrategias'][b][m] for t in A)}/{len(A)}"
               for m in ("sharpe", "calmar", "acc")} for b in ("B&H", "ZeroR")}
    res = {"meta": {"activos": A, "calib_end": str(CALIB_END.date()), "test": f"{TEST_START.date()}..{TEST_END.date()}",
                    "sizing": "vol-target (target_vol/σ) para Régimen/B&H/ZeroR; B&H_1x = comprar y mantener 1x",
                    "causal": "HMM/GARCH calibrados ≤2019 por activo; s_dom expansible; signal_lag=1",
                    "seed": config.SEED, "crisis": CRISIS,
                    "nota": "Canal de régimen SIN agente → ventana larga con crisis. Exploratorio (docs/)."},
           "por_activo": rows, "medias": medias, "cobertura_regimen": cob}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== MEDIAS (test 2020–2024, sizing vol-target) ===")
    print(f"  {'estrategia':10s}{'acc':>8s}{'Sharpe':>9s}{'maxDD':>9s}{'Calmar':>9s}{'equity':>9s}")
    for s in ("Régimen", "B&H", "ZeroR", "B&H_1x"):
        r = medias[s]
        print(f"  {s:10s}{r['acc']:>8.3f}{r['sharpe']:>9.2f}{r['maxdd']:>8.1%}{r['calmar']:>9.2f}{r['equity']:>9.3f}")
    print("\n  Régimen bate a B&H:", cob["B&H"], "| a ZeroR:", cob["ZeroR"])
    print("\n=== CRISIS: retorno y maxDD Régimen vs B&H (vol-target) ===")
    for name in CRISIS:
        rr = [rows[t]["crisis"].get(name) for t in A if name in rows[t]["crisis"]]
        if not rr:
            continue
        rg_dd = np.mean([x["Régimen"]["maxdd"] for x in rr]); bh_dd = np.mean([x["B&H"]["maxdd"] for x in rr])
        rg_r = np.mean([x["Régimen"]["ret"] for x in rr]); bh_r = np.mean([x["B&H"]["ret"] for x in rr])
        print(f"  {name:11s} Régimen: ret={rg_r:+.1%} maxDD={rg_dd:.1%}  |  B&H: ret={bh_r:+.1%} maxDD={bh_dd:.1%}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
