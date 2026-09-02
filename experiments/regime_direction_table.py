"""¿El régimen filtrado del HMM predice dirección, y transfiere de calibración al OOS? SPY vs SMCI.

Por régimen (argmax del posterior FILTRADO, causal): retorno medio de r_{t+1} y % de días al alza,
en calibración (2000→2024-09) y en OOS (2024-10→). Si el signo del retorno medio por régimen se
mantiene calib→OOS, el régimen predice dirección y transfiere (SPY, leverage effect); si cambia de
signo, no transfiere (prior-flip; SMCI, sin leverage effect). Ilustra que el HMM da VOLATILIDAD y solo
mapea a dirección donde el leverage effect existe.
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
from core.hmm import RegimeHMM
import experiments.walkforward_robustez as wf

NAMES = ["Calma", "Estrés", "Crisis"]
OUT = Path("outputs/experiments/regime_direction_table.json")


def per_regime(tk: str) -> dict:
    feat_df, ret = wf.load_features(tk)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()),
                         index=feat_df.index, columns=NAMES)
    reg = gamma.values.argmax(1)
    reg = pd.Series(reg, index=feat_df.index)
    r_same = ret                    # r_t : MISMO día (leverage effect, contemporáneo, NO tradeable)
    r_next = ret.shift(-1)          # r_{t+1} : día SIGUIENTE (causal, lo único tradeable)
    out = {}
    for split, mask in [("calib", feat_df.index <= pd.Timestamp(CALIBRATION_END)),
                        ("oos", feat_df.index >= pd.Timestamp(STRATA_OOS_START))]:
        idx = feat_df.index[mask]
        d = {}
        for s, nm in enumerate(NAMES):
            sel = idx[(reg.reindex(idx) == s).to_numpy()]
            rs, rn = r_same.reindex(sel).dropna(), r_next.reindex(sel).dropna()
            d[nm] = {"n": int(len(rn)),
                     "ret_mismo_dia": round(float(rs.mean()), 6) if len(rs) else None,
                     "ret_dia_sig": round(float(rn.mean()), 6) if len(rn) else None,
                     "frac_sube_sig": round(float((rn > 0).mean()), 4) if len(rn) else None}
        out[split] = d
    return out


def main() -> None:
    res = {tk: per_regime(tk) for tk in ["SPY", "SMCI"]}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for tk, r in res.items():
        print(f"\n=== {tk} ===  (ret MISMO día = leverage contemporáneo; ret día SIG = causal/tradeable)")
        print(f"{'régimen':<8}{'mismo_cal':>11}{'sig_cal':>11}{'mismo_oos':>11}{'sig_oos':>11}  ¿signo causal transfiere?")
        for nm in NAMES:
            c, o = r["calib"][nm], r["oos"][nm]
            sc = np.sign(c["ret_dia_sig"]) if c["ret_dia_sig"] is not None else 0
            so = np.sign(o["ret_dia_sig"]) if o["ret_dia_sig"] is not None else 0
            flip = "transfiere" if sc == so and sc != 0 else "FLIP"
            print(f"{nm:<8}{c['ret_mismo_dia']:>11.5f}{c['ret_dia_sig']:>11.5f}"
                  f"{o['ret_mismo_dia']:>11.5f}{o['ret_dia_sig']:>11.5f}  {flip}")
    print(f"\n>>> {OUT}")


if __name__ == "__main__":
    main()
