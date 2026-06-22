"""Tabla de estrategias × panel completo (15 activos) sobre la ventana desplegable [150:] (~250 días).

Computa de forma autónoma (solo HMM compartido + precios + decisiones del agente, sin el pipeline
de detectores) las estrategias que NO necesitan GARCH/override:
  - B&H        : siempre largo (+1).
  - ZeroR      : clase mayoritaria (max(frac_up, 1-frac_up)) — el naïve que pide el tutor.
  - M5         : agente LLM solo (signo del size).
  - Régimen    : RAM crudo = dirección del régimen por leverage effect (calm_prob>=crisis_prob → +1).

M8 (regla STRATA) y M10 (meta-learner) requieren el pipeline de detectores+override; para los 10
activos originales se reusan de panel_intervention_scan.json (misma ventana). Para los 5 nuevos
(QQQ, DIA, IWM, XLF, XLK) quedan pendientes de reconstruir ese pipeline.

Uso: python experiments/panel_all_strategies.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CACHE_MODELS_DIR, DATA_DIR, CALIBRATION_START, STRATA_OOS_START
from core import data, features

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA",
         "QQQ", "DIA", "IWM", "XLF", "XLK"]
N0 = 150  # burn-in → ventana desplegable ~250 días, coherente con panel_intervention_scan
OUT = Path("outputs/experiments/panel_all_strategies.json")
HMM = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
SCAN = json.load(open("outputs/experiments/panel_intervention_scan.json"))["por_activo"]


def regime_df(tk: str) -> tuple[pd.DataFrame, pd.Series]:
    parquets = sorted(glob.glob(str(DATA_DIR / f"{tk}_{CALIBRATION_START}_*.parquet")))
    data_end = parquets[-1].rsplit("_", 1)[1].replace(".parquet", "")
    ret = features.log_returns(data.load_market_data(tk, CALIBRATION_START, data_end)["Close"])
    rv = features.realized_vol_annualized(ret, window=21)
    feat = pd.concat([ret.rename("r"), rv.rename("rv")], axis=1).dropna()
    gamma = HMM.predict_proba_filtered(feat.to_numpy())  # filtrado = causal
    g = pd.DataFrame(gamma, index=feat.index, columns=["Calma", "Estrés", "Crisis"])
    return g, ret


def agent_size(tk: str) -> pd.Series:
    s = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))):
        d = json.load(open(fp))
        s[pd.Timestamp(d["date"])] = float(d["size"])
    return pd.Series(s).sort_index()


def run(tk: str) -> dict:
    g, ret = regime_df(tk)
    a = agent_size(tk)
    idx = a.index
    truth = np.sign(ret.shift(-1).reindex(idx).to_numpy())          # r_{t+1}
    m5 = np.sign(a.to_numpy())
    gi = g.reindex(idx)
    reg = np.where(gi["Calma"].to_numpy() >= gi["Crisis"].to_numpy(), 1.0, -1.0)  # leverage effect
    keep = np.isfinite(truth) & (truth != 0)
    truth, m5, reg = truth[keep], m5[keep], reg[keep]
    # ventana desplegable [N0:]
    truth, m5, reg = truth[N0:], m5[N0:], reg[N0:]
    n = len(truth)
    frac_up = float((truth > 0).mean())
    acc = {
        "n": n, "frac_up": round(frac_up, 3),
        "bh": round(frac_up, 4),                                   # siempre largo
        "zeror": round(max(frac_up, 1 - frac_up), 4),              # clase mayoritaria
        "m5": round(float((m5 == truth).mean()), 4),
        "regimen": round(float((reg == truth).mean()), 4),
    }
    sc = SCAN.get(tk, {}).get("accuracy", {})
    acc["m8"] = sc.get("m8")    # de panel_intervention_scan (solo 10 originales)
    acc["m10"] = sc.get("m10")
    return acc


def main() -> None:
    config.set_seeds(config.SEED)
    res = {}
    hdr = f"{'tk':5} {'n':>3} {'up%':>5} | {'B&H':>6} {'ZeroR':>6} {'M5':>6} {'Régim':>6} {'M8':>6} {'M10':>6}"
    print(hdr); print("-" * len(hdr))
    for tk in PANEL:
        r = run(tk); res[tk] = r
        f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "  —  "
        print(f"{tk:5} {r['n']:>3} {r['frac_up']*100:>4.0f}% | {f(r['bh'])} {f(r['zeror'])} "
              f"{f(r['m5'])} {f(r['regimen'])} {f(r['m8'])} {f(r['m10'])}")

    # Resumen: medianas y nº de activos donde cada estrategia bate a B&H y a ZeroR
    def med(k):
        v = [res[t][k] for t in PANEL if isinstance(res[t][k], (int, float))]
        return round(float(np.median(v)), 4)
    print("-" * len(hdr))
    print(f"{'MED':5} {'':>3} {'':>5} | {med('bh'):.3f} {med('zeror'):.3f} {med('m5'):.3f} {med('regimen'):.3f}")
    reg_beats_bh = sum(res[t]["regimen"] > res[t]["bh"] for t in PANEL)
    reg_beats_zr = sum(res[t]["regimen"] > res[t]["zeror"] for t in PANEL)
    m5_beats_bh = sum(res[t]["m5"] > res[t]["bh"] for t in PANEL)
    print(f"\nRégimen bate a B&H en {reg_beats_bh}/15 · a ZeroR en {reg_beats_zr}/15 · M5 bate a B&H en {m5_beats_bh}/15")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": {"panel": PANEL, "N0": N0,
                   "regimen": "RAM crudo: calm_prob>=crisis_prob → largo (leverage effect)",
                   "nota_m8_m10": "M8/M10 solo para los 10 originales (panel_intervention_scan.json); 5 nuevos pendientes de pipeline"},
                   "por_activo": res}, indent=2, ensure_ascii=False))
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
