"""Recon: ¿en qué activo del panel la M10 causal (desplegable) bate a B&H en accuracy?

Exploratorio barato (pre-registro BITACORA [2026-06-15]) previo a decidir si hace falta la M10-v3 causal
completa. Por activo: HMM K=3 + GARCH ajustados al vuelo (como `_oos_m5_m8` del panel), master M5/M8 + 22
features, M10 causal walk-forward (vanilla, reentreno mensual, solo pasado), y accuracy direccional de
M5/M8/M10-WF/M10-CPCV/B&H en el tramo de test `[N0:fin]`. Reporta los 10 activos (anti-cherry-pick).

Uso: python experiments/m10_causal_panel_recon.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, STRATA_OOS_START
from core.garch import GARCHModel
from core.hmm import RegimeHMM
import experiments.walkforward_robustez as wf
from experiments.walkforward_m10_causal import FULL_COLS, expanding_wf_p1

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
OUT = Path("outputs/experiments/m10_causal_panel_recon.json")


def build_states_onthefly(ticker: str):
    """HMM K=3 + GARCH ajustados sobre la calibración del PROPIO activo (idéntico a _oos_m5_m8)."""
    feat_df, ret = wf.load_features(ticker)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    return gamma, sigma, oos_ret


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)

    p1_wf = expanding_wf_p1(X, y)
    p1_cpcv = wf.cpcv_oof(X, y)
    td = X.index[p1_wf.notna()]
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())

    acc = {
        "m5": float((np.sign(mv.loc[td, "agent_size"].to_numpy()) == truth).mean()),
        "m8": float((np.sign(mv.loc[td, "final_size"].to_numpy()) == truth).mean()),
        "m10_cpcv": float((np.sign(p1_cpcv.loc[td].to_numpy() - 0.5) == truth).mean()),
        "m10_wf": float((np.sign(p1_wf.loc[td].to_numpy() - 0.5) == truth).mean()),
        "bh": float((truth == 1).mean()),                     # B&H acierta los días alcistas
    }
    acc = {k: round(v, 4) for k, v in acc.items()}
    best = max(("m5", "m8", "m10_cpcv", "m10_wf", "bh"), key=lambda k: acc[k])
    return {"n_test": int(len(td)), "test_span": [str(td.min().date()), str(td.max().date())],
            "accuracy": acc, "best_arm": best,
            "m10wf_bate_bh": bool(acc["m10_wf"] > acc["bh"]),
            "m10wf_bate_todo": bool(acc["m10_wf"] > max(acc["m5"], acc["m8"], acc["bh"]))}


def main() -> None:
    result = {"meta": {"seed": config.SEED, "signal_lag": 1, "panel": PANEL,
                       "scheme": "causal expanding WF (N0=150,step=21,embargo=5); HMM+GARCH on-the-fly por activo",
                       "nota": "Exploratorio (recon). Reporta los 10; B&H accuracy = % días alcistas en el tramo.",
                       "pre_registro": "BITACORA 2026-06-15"},
              "por_activo": {}}
    for tk in PANEL:
        try:
            r = run_ticker(tk); result["por_activo"][tk] = r
            a = r["accuracy"]
            flag = "  <<< M10-WF > B&H" if r["m10wf_bate_bh"] else ""
            print(f"{tk:5} n={r['n_test']:3}  M5={a['m5']:.3f} M8={a['m8']:.3f} "
                  f"M10cpcv={a['m10_cpcv']:.3f} M10wf={a['m10_wf']:.3f} B&H={a['bh']:.3f}  best={r['best_arm']}{flag}")
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5} ERROR {e!r}")
            result["por_activo"][tk] = {"error": repr(e)}

    cand = [tk for tk, r in result["por_activo"].items() if r.get("m10wf_bate_bh")]
    cand_todo = [tk for tk, r in result["por_activo"].items() if r.get("m10wf_bate_todo")]
    result["candidatos_m10wf_gt_bh"] = cand
    result["candidatos_m10wf_gt_todo"] = cand_todo
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nCandidatos M10-WF > B&H: {cand or 'NINGUNO'}")
    print(f"Candidatos M10-WF > TODO (M5,M8,B&H): {cand_todo or 'NINGUNO'}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
