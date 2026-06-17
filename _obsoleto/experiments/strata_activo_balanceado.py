"""STRATA sobre un activo BALANCEADO (clases sube/baja ~50/50): M5 vs M8 vs M10.

El tutor ([21:51],[22:01], Reunion_Dani_2026-06-16) pidio un activo balanceado para que el
baseline mayoritario valga ~50% (en SPY el baseline trivial "siempre largo" acierta 56.9% y
hunde a M8/M10 por comparacion). Diagnostico (class_balance_diagnostic.json): TSLA y UNG son
los mas balanceados (desbalanceo_abs=0.0012). Caso central TSLA; robustez UNG (--ticker).

Mide: accuracy direccional M5/M8/M10 vs baseline mayoritario; McNemar pareado M8 vs M5 y
M10 vs M5; Diebold-Mariano M10 vs M8 (universalidad: M10 no debe batir a los detectores
clasicos); chequeo prior-flip de RAM (leverage effect, Black 1976; Christie 1982). En
TSLA/UNG el leverage effect es debil/nulo, asi que prior_flip=true es resultado ESPERADO y
documenta la limitacion honesta del mecanismo RAM fuera de indices.

Reutiliza m10_v3_causal_panel (loader, run_master, m10_v3_causal, build_states_onthefly)
restringido a un activo. HMM/GARCH on-the-fly (cache/models es SPY-centrico).

Pre-registro: BITACORA.md [2026-06-16] strata_activo_balanceado.
NO ejecuta nada al importarse; solo main() bajo __main__. PENDIENTE de auditoria rigor.
Uso: python experiments/strata_activo_balanceado.py --ticker TSLA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CALIBRATION_END, CALIBRATION_START, STRATA_OOS_START
from core.stats import diebold_mariano, mcnemar_test, sign_test, stationary_bootstrap_ci
import experiments.walkforward_robustez as wf
import experiments.m10_v3_causal_panel as m10
from experiments.walkforward_m10_causal import FULL_COLS

REGIMES = ["Calma", "Estrés", "Crisis"]
OUT = Path("outputs/experiments/strata_activo_balanceado.json")


def prior_flip_check(gamma: pd.DataFrame, ret: pd.Series, m: pd.DataFrame) -> dict:
    """Signo de la dir. media por regimen en CALIBRACION vs OOS; prior_flip si difiere en el dominante.

    regime_sign_k = sign(E[r | estado dominante = k]). Calibracion: ret <= CALIBRATION_END.
    OOS: ret en STRATA_OOS_START.. El regimen dominante = estado con mas masa de intervencion RAM
    en el OOS (donde RAM realmente actua). prior_flip si sign cambia ahi (leverage effect roto).
    """
    state = gamma.values.argmax(axis=1)
    state = pd.Series(state, index=gamma.index)
    calib_mask = ret.index <= pd.Timestamp(CALIBRATION_END)
    oos_mask = ret.index >= pd.Timestamp(STRATA_OOS_START)

    sign_calib, sign_oos = {}, {}
    for k, name in enumerate(REGIMES):
        idx_c = state.reindex(ret.index)[calib_mask] == k
        idx_o = state.reindex(ret.index)[oos_mask] == k
        rc = ret[calib_mask][idx_c.values]
        ro = ret[oos_mask][idx_o.values]
        sign_calib[name] = float(np.sign(rc.mean())) if len(rc) else 0.0
        sign_oos[name] = float(np.sign(ro.mean())) if len(ro) else 0.0

    # Regimen dominante = donde RAM interviene mas en el OOS (ram_sev medium/high).
    ram_fire = m[m["ram_sev"].isin(["medium", "high"])]
    if len(ram_fire):
        dom = int(ram_fire["regime_dom"].mode().iloc[0])
    else:
        dom = int(state.reindex(m.index).mode().iloc[0])
    dom_name = REGIMES[dom]
    flip = bool(sign_calib[dom_name] != 0 and sign_oos[dom_name] != 0
                and sign_calib[dom_name] != sign_oos[dom_name])
    return {"regime_sign_calib": sign_calib, "regime_sign_oos": sign_oos,
            "regime_dominante": dom_name, "prior_flip": flip,
            "comentario_leverage_effect": (
                "El proxy direccional del HMM se invierte fuera de muestra en el regimen "
                "dominante: RAM carece de base direccional aqui (leverage effect debil/nulo, "
                "esperado en activos no-indice; Black 1976, Christie 1982)." if flip else
                "El signo por regimen se mantiene OOS: el proxy direccional del HMM resiste.")}


def run_ticker(ticker: str) -> dict:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = m10.build_states_onthefly(ticker)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(ticker))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    truth = np.sign(mv["r_next"].to_numpy())

    # M10-v3 causal (mismo esquema que el panel) restringido a este activo.
    X = mv[FULL_COLS]; y = (mv["r_next"] > 0).astype(int)
    p_cal, active, dir_scaled, _ = m10.m10_v3_causal(X, y)
    td = X.index[p_cal.notna()]
    mvt = mv.loc[td]; truth_t = np.sign(mvt["r_next"].to_numpy())

    pos = {"m5": np.sign(mvt["agent_size"].to_numpy()),
           "m8": np.sign(mvt["final_size"].to_numpy()),
           "m10": np.sign(p_cal.loc[td].to_numpy() - 0.5)}
    corr = {k: (v == truth_t).astype(int) for k, v in pos.items()}

    frac_sube = float((truth == 1).mean())
    baseline_may = max(frac_sube, 1.0 - frac_sube)
    accuracy = {k: float((v == truth_t).mean()) for k, v in pos.items()}
    accuracy["baseline_mayoritario"] = baseline_may

    # McNemar pareado: b = primero acierta y segundo falla.
    def _mc(a, b):
        _, p, bb, cc = mcnemar_test(corr[a], corr[b])
        return {"p": float(p), "b": int(bb), "c": int(cc), "n_disc": int(bb + cc)}
    mcnemar = {"m8_vs_m5": _mc("m8", "m5"), "m10_vs_m5": _mc("m10", "m5")}

    # Sign test contra 0.5.
    def _sign(arr):
        k, n, p, ci = sign_test(arr)
        return {"k": int(k), "n": int(n), "p": float(p), "ci95": [float(ci[0]), float(ci[1])]}
    sign_t = {"m8_vs_05": _sign(corr["m8"]), "m5_vs_05": _sign(corr["m5"])}

    # Diebold-Mariano M10 vs M8 sobre perdida de error direccional (1-acierto)^2 = (1-acierto).
    loss_m10 = (1 - corr["m10"]).astype(float)
    loss_m8 = (1 - corr["m8"]).astype(float)
    dm_stat, dm_p = diebold_mariano(loss_m10, loss_m8, h=1)
    dm = {"m10_vs_m8": {"stat": float(dm_stat), "p": float(dm_p)}}

    # IC95 bootstrap del Delta-accuracy M8-M5 (pareado dia a dia).
    diff = (corr["m8"] - corr["m5"]).astype(float)
    lo, hi, mean = stationary_bootstrap_ci(diff, np.mean, n=2000, seed=config.SEED)
    ci_delta = {"low": float(lo), "high": float(hi), "mean": float(mean),
                "excluye_cero": not (lo <= 0 <= hi)}

    # Holm sobre el pool de contrastes del activo.
    holm = wf._holm_bonferroni(
        {"m8_vs_m5": mcnemar["m8_vs_m5"]["p"], "m10_vs_m5": mcnemar["m10_vs_m5"]["p"],
         "m10_vs_m8_dm": dm["m10_vs_m8"]["p"]}, alpha=0.10)

    pf = prior_flip_check(gamma, oos_ret, m)

    # Veredictos pre-registrados.
    h1a = bool(mcnemar["m8_vs_m5"]["p"] < 0.10 and mcnemar["m8_vs_m5"]["b"] > mcnemar["m8_vs_m5"]["c"])
    h1b = bool(sign_t["m8_vs_05"]["p"] < 0.10 and accuracy["m8"] >= baseline_may + 0.02)
    h1c = bool(dm["m10_vs_m8"]["p"] > 0.10)  # universalidad: NO se rechaza igualdad
    refuta_a = bool(mcnemar["m8_vs_m5"]["p"] < 0.10 and mcnemar["m8_vs_m5"]["c"] > mcnemar["m8_vs_m5"]["b"])

    if h1a and not pf["prior_flip"]:
        generalizacion = "fuerte"
    elif h1a and pf["prior_flip"]:
        generalizacion = "por_modulacion"
    else:
        generalizacion = "limitacion_honesta"

    interp = ("STRATA EMPEORA al agente (McNemar c>b significativo): hipotesis H1(a) refutada."
              if refuta_a else
              {"fuerte": "STRATA corrige al agente con mecanismo RAM intacto (sin prior-flip).",
               "por_modulacion": "STRATA corrige pero el RAM tiene prior-flip: la mejora viene "
                                 "de PSA/GSO o azar, exige atribucion de P&L por detector.",
               "limitacion_honesta": "STRATA no corrige significativamente y hay prior-flip: "
                                     "limitacion especifica de activos sin leverage effect."}[generalizacion])

    return {
        "balanceo": {"frac_sube": frac_sube, "frac_baja": 1.0 - frac_sube,
                     "baseline_mayoritario_acc": baseline_may},
        "accuracy": accuracy,
        "mcnemar": mcnemar,
        "sign_test": sign_t,
        "diebold_mariano": dm,
        "holm": holm,
        "ci_delta_acc_m8_m5": ci_delta,
        "prior_flip": pf,
        "n_test": int(len(td)),
        "test_span": [str(td.min().date()), str(td.max().date())],
        "verdict": {"h1a_corrige": h1a, "h1b_bate_azar": h1b, "h1c_universalidad": h1c,
                    "prior_flip": pf["prior_flip"], "interpretacion": interp,
                    "generalizacion": generalizacion},
    }


def main(ticker: str = "TSLA") -> None:
    r = run_ticker(ticker)
    result = {
        "meta": {"ticker": ticker, "oos_start": STRATA_OOS_START,
                 "oos_end": r["test_span"][1], "n_days": r["n_test"],
                 "signal_lag": 1, "ram_tau": 0.5, "override_variant": "C",
                 "calibration_window": [CALIBRATION_START, CALIBRATION_END], "seed": config.SEED,
                 "scheme": "M5/M8 override-C dia a dia + M10-v3 causal (XGB 80x3 + isotonica + "
                           "abstencion 30% + P95, todo en pasado; WF N0=150/21/embargo5)"},
        **{k: r[k] for k in ("balanceo", "accuracy", "mcnemar", "sign_test", "diebold_mariano",
                             "holm", "ci_delta_acc_m8_m5", "prior_flip", "verdict")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Validacion final: claves contractadas en el pre-registro.
    loaded = json.loads(OUT.read_text())
    for key in ("meta", "balanceo", "accuracy", "mcnemar", "sign_test", "diebold_mariano",
                "holm", "ci_delta_acc_m8_m5", "prior_flip", "verdict"):
        assert key in loaded, f"Falta clave de primer nivel: {key}"
    for k in ("m5", "m8", "m10", "baseline_mayoritario"):
        assert k in loaded["accuracy"], f"Falta accuracy.{k}"
    for k in ("m8_vs_m5", "m10_vs_m5"):
        assert k in loaded["mcnemar"], f"Falta mcnemar.{k}"
    assert "m10_vs_m8" in loaded["diebold_mariano"], "Falta diebold_mariano.m10_vs_m8"
    for k in ("regime_sign_calib", "regime_sign_oos", "regime_dominante", "prior_flip"):
        assert k in loaded["prior_flip"], f"Falta prior_flip.{k}"
    for k in ("h1a_corrige", "h1b_bate_azar", "h1c_universalidad", "prior_flip",
              "interpretacion", "generalizacion"):
        assert k in loaded["verdict"], f"Falta verdict.{k}"
    assert loaded["meta"]["signal_lag"] == 1, "signal_lag debe ser 1 (causal)"
    print(f"OK · {OUT} · {ticker} · generalizacion={result['verdict']['generalizacion']} · "
          f"prior_flip={result['prior_flip']['prior_flip']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="TSLA")
    args = ap.parse_args()
    config.set_seeds(config.SEED)
    main(args.ticker)
