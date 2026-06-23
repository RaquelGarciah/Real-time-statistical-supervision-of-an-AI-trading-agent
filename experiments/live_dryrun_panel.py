"""Dry-run de producción del panel: ¿qué decidiría el kill-switch HOY, activo por activo?

No es un experimento de tesis: es una FOTO operativa para validar el protocolo de despliegue (hueco #3)
antes de cablearlo al notebook. Para cada activo computa, desde lógica ya implementada y sin tocar el
agente, las tres condiciones pre-registradas que apagan la capa de override de STRATA:

  1. GATE identificable (core.regime_gate.calibrate_gate, ex-ante sobre calibración): ¿el régimen se
     vuelve MÁS fiable al subir la confianza (pendiente b>0) y el cruce de 0.5 cae en (0,1)? Si no,
     el régimen no es direccionalmente informativo → la capa se ABSTIENE (no fuerza τ=0).
  2. LEVERAGE suficiente (leverage_screen): crisis_mean<0 (régimen direccional por leverage effect).
     Si el leverage es débil/inverso, RAM no es proxy de dirección → fuera de alcance.
  3. PRIOR-FLIP: la apuesta de dirección dominante del régimen, ¿sigue acertando ≥0.5 en el OOS, o
     se invirtió respecto a calibración? Un flip de signo → desconfiar (mecanismo de falsación).

Más un monitor de DRIFT (frecuencia del régimen de crisis en calibración vs OOS). Verdict por activo:
GO (override activo) / ABSTAIN (solo warn, sin override) con el motivo. Uso: python experiments/live_dryrun_panel.py
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
from core.regime_gate import calibrate_gate, directional_reliability
from experiments.quant_validation_panel import build_states, wf

PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
EXCL5 = ["MSTR", "NVDA", "BAC", "TSLA", "IWM"]
OUT = Path("outputs/live/dryrun_panel.json")
LEV = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]


TAU = 0.5  # umbral del gate RAM desplegado: el override-C dispara con confianza direccional > τ.


def _fired_acc(calm, crisis, r_next):
    """Confianza, acierto y accuracy del subconjunto DISPARADO (conf>τ) — la apuesta real del override."""
    conf, correct = directional_reliability(calm, crisis, r_next)
    fired = conf > TAU
    acc_fired = float(correct[fired].mean()) if fired.any() else float("nan")
    return acc_fired, float(fired.mean()), conf, correct, fired


def evaluate(tk: str) -> dict:
    feat_df, ret = wf.load_features(tk)
    gamma, sigma, oos_ret = build_states(tk)
    r_next = ret.shift(-1)
    calib = gamma.index[gamma.index <= pd.Timestamp(CALIBRATION_END)]
    oos = gamma.index[gamma.index >= pd.Timestamp(STRATA_OOS_START)]

    # (1) Identificabilidad EX-ANTE como la usa STRATA: ¿el subconjunto disparado (conf>τ=0.5) bate 0.5
    #     direccional en CALIBRACIÓN? + pendiente logística b>0 (el régimen gana fiabilidad con la confianza).
    rc = r_next.reindex(calib); mc = rc.notna() & (np.sign(rc) != 0)
    calm_c, crisis_c, rc = gamma.loc[calib, "Calma"].to_numpy()[mc.to_numpy()], \
        gamma.loc[calib, "Crisis"].to_numpy()[mc.to_numpy()], rc[mc].to_numpy()
    acc_calib_fired, frac_fired, _, _, _ = _fired_acc(calm_c, crisis_c, rc)
    g = calibrate_gate(calm_c, crisis_c, rc)
    identifiable = bool(acc_calib_fired > 0.5 and g.slope > 0 and frac_fired > 0.05)

    # (2) Leverage = el MECANISMO que explica la identificabilidad (descriptivo, no gate independiente).
    lv = LEV.get(tk, {}); crisis_mean = lv.get("crisis_mean", float("nan"))

    # (3) Monitor OOS — prior-flip: el subconjunto disparado, ¿deja de batir 0.5 en OOS (signo invertido)?
    ro = r_next.reindex(oos); mo = ro.notna() & (np.sign(ro) != 0)
    acc_oos_fired, _, _, _, _ = _fired_acc(gamma.loc[oos, "Calma"].to_numpy()[mo.to_numpy()],
                                           gamma.loc[oos, "Crisis"].to_numpy()[mo.to_numpy()], ro[mo].to_numpy())
    prior_flip = bool(identifiable and acc_oos_fired < 0.5)

    # Drift: frecuencia del régimen de crisis calibración vs OOS.
    crisis_calib = float((gamma.loc[calib].to_numpy().argmax(1) == 2).mean())
    crisis_oos = float((gamma.loc[oos].to_numpy().argmax(1) == 2).mean())

    reasons = []
    if not identifiable:
        reasons.append("régimen no informativo ex-ante (subconjunto disparado no bate 0.5)")
    if prior_flip:
        reasons.append("prior-flip (dirección disparada invertida en OOS)")
    verdict = "GO" if not reasons else "ABSTAIN"
    return {
        "verdict": verdict, "motivo": reasons or ["régimen informativo ex-ante · sin prior-flip"],
        "gate": {"identifiable": identifiable, "acc_calib_disparado": round(acc_calib_fired, 3),
                 "frac_disparo": round(frac_fired, 3), "slope": round(g.slope, 3),
                 "frac_identified_robusto": round(g.frac_identified, 3)},
        "leverage": {"crisis_mean": crisis_mean, "leverage_corr": lv.get("leverage_corr")},
        "monitor_prior_flip": {"acc_oos_disparado": round(acc_oos_fired, 3), "flip": prior_flip},
        "monitor_drift_crisis": {"calib": round(crisis_calib, 3), "oos": round(crisis_oos, 3),
                                 "ratio": round(crisis_oos / crisis_calib, 2) if crisis_calib > 0 else None},
    }


def main() -> None:
    res = {}
    print(f"{'activo':6s} {'grupo':5s} {'verdict':8s} {'ident':>5s} accCal disp%  lev(cm)   accOOS  flip  drift | motivo")
    for grp, tks in (("CUERPO", PANEL10), ("APEND", EXCL5)):
        for tk in tks:
            try:
                r = res[tk] = evaluate(tk)
                g = r["gate"]; pf = r["monitor_prior_flip"]
                print(f"{tk:6s} {grp[:5]:5s} {r['verdict']:8s} "
                      f"{'sí' if g['identifiable'] else 'NO':>5s} {g['acc_calib_disparado']:.3f} {g['frac_disparo']:.2f}  "
                      f"{r['leverage']['crisis_mean']:+.4f}  {pf['acc_oos_disparado']:.3f}  "
                      f"{'SÍ' if pf['flip'] else '·':>3s}   ×{r['monitor_drift_crisis']['ratio']}  "
                      f"| {', '.join(r['motivo'])}", flush=True)
            except Exception as e:  # noqa: BLE001
                res[tk] = {"error": f"{type(e).__name__}: {e}"}
                print(f"{tk:6s} ERROR {type(e).__name__}: {e}", flush=True)

    body = [t for t in PANEL10 if "verdict" in res[t]]
    go = [t for t in body if res[t]["verdict"] == "GO"]
    abst = [t for t in body if res[t]["verdict"] == "ABSTAIN"]
    out = {"meta": {"panel": PANEL10, "apendice": EXCL5, "oos_start": STRATA_OOS_START,
                    "nota": "dry-run del kill-switch (gate identificable + leverage + prior-flip + drift); "
                            "sin agente; ex-ante donde aplica. NO es backtest de P&L."},
           "por_activo": res,
           "resumen": {"cuerpo_GO": go, "cuerpo_ABSTAIN": abst,
                       "n_GO": len(go), "n_ABSTAIN": len(abst)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n=== resumen cuerpo (10): GO={len(go)} {go} · ABSTAIN={len(abst)} {abst} ===")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
