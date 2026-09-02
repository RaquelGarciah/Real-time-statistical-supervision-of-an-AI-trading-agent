"""Sensibilidad del percentil de corte de PSA/GSO con curva de VALIDACION y de TEST.

El tutor (Reunion_Dani_2026-06-16, [05:20],[09:05],[30:44]) pidio: elegir el percentil
de corte mirando SOLO una ventana de validacion (ultimo anio de la calibracion) y
comprobar en test (OOS) que la curva accuracy-vs-percentil "va de la mano". Eje X:
percentiles P50..P95 (paso 5); eje Y: accuracy direccional; dos series val/test.

Difiere de psa_gso_threshold_sensitivity.py (E4): aquel barria SOLO el OOS de SPY como
diagnostico anti-look-ahead. Este introduce particion temporal val/test y un criterio de
generalizacion pre-registrado (rho>=0.5 Y |gap|<=0.05 en el p* elegido en validacion).
RAM fijo en tau=0.5; solo se mueven PSA/GSO.

Activo balanceado por defecto TSLA (caso central); UNG via --ticker (robustez). HMM/GARCH
se calibran on-the-fly por activo (cache/models es SPY-centrico, no reutilizable aqui).

Pre-registro: BITACORA.md [2026-06-16] umbral_psa_gso_valtest.
NO ejecuta nada al importarse; solo main() bajo __main__. PENDIENTE de auditoria rigor.
Uso: python experiments/umbral_psa_gso_valtest.py --ticker TSLA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config
from config import CACHE_AGENT_DIR, CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START, STRATA_OOS_START
from core.garch import GARCHModel
from core.hmm import RegimeHMM
from core.stats import sign_test, stationary_bootstrap_ci
from strata.detectors import reset_thresholds_cache
import experiments.walkforward_robustez as wf
from experiments.psa_gso_threshold_sensitivity import run_master  # inyecta psa_thr/gso_thr via hooks

TAU_RAM = 0.5
RAM_THRESHOLDS = (TAU_RAM / 2, TAU_RAM, 0.70)
GRID_PCTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
VAL_START, VAL_END = "2023-10-01", "2024-09-30"   # ultimo anio DENTRO de la calibracion
RHO_MIN, GAP_MAX = 0.5, 0.05


def build_states_full(ticker: str):
    """HMM/GARCH calibrados <= CALIBRATION_END (on-the-fly) y devuelve regimen filtrado, sigma, ret.

    Identico esquema a m10_v3_causal_panel.build_states_onthefly pero extendiendo sigma a TODO
    el rango (validacion + test), no solo al OOS, para poder evaluar la ventana de validacion.
    """
    feat_df, ret = wf.load_features(ticker)
    calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
    hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
    garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
    gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index,
                         columns=["Calma", "Estrés", "Crisis"])
    sigma = garch.forecast_path(ret[ret.index >= pd.Timestamp(VAL_START)])
    return gamma, sigma, ret


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for fp in sorted(path.glob("*")):
        if fp.is_file():
            h.update(fp.name.encode()); h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def thresholds_at(detector: str, pct: int, dist: dict) -> tuple[float, float, float]:
    """Umbral a percentil pct (mismas reglas que E4: PSA mueve 'high', GSO mueve 'medium')."""
    p = float(dist[detector]["score_distribution"][f"p{pct}"])
    if detector == "psa":
        return (0.0, p, p)
    return (p, p, float(dist[detector]["score_distribution"]["max"]))


def acc_directional(m: pd.DataFrame, win_mask: pd.Series) -> tuple[float, int]:
    """Accuracy direccional sign(final_size)==sign(r_next) en la ventana, dias con r_next!=0."""
    sub = m.loc[win_mask]
    valid = sub["r_next"].notna() & (np.sign(sub["r_next"]) != 0)
    pred = np.sign(sub.loc[valid, "final_size"].to_numpy())
    truth = np.sign(sub.loc[valid, "r_next"].to_numpy())
    return (float((pred == truth).mean()) if valid.sum() else float("nan"), int(valid.sum()))


def main(ticker: str = "TSLA") -> None:
    out = Path(f"outputs/experiments/umbral_psa_gso_valtest.json")
    gamma, sigma, ret = build_states_full(ticker)
    agents = wf.load_agent(ticker)
    dist = json.load(open(CACHE_MODELS_DIR / "strata_thresholds.json"))

    # ret_next para toda la ventana evaluable (val+test); run_master de E4 calcula r_next sobre oos.
    # Aqui pasamos como "oos_ret" el tramo VAL_START.. (incluye validacion y test) para una sola pasada.
    eval_ret = ret[ret.index >= pd.Timestamp(VAL_START)]
    val_mask_idx = (eval_ret.index >= pd.Timestamp(VAL_START)) & (eval_ret.index <= pd.Timestamp(VAL_END))
    test_mask_idx = eval_ret.index >= pd.Timestamp(STRATA_OOS_START)

    curvas: dict[str, list] = {"psa": [], "gso": []}
    for detector in ("psa", "gso"):
        for pct in GRID_PCTILES:
            thr = thresholds_at(detector, pct, dist)
            reset_thresholds_cache()
            m = run_master(gamma, sigma, eval_ret, agents,
                           psa_thr=thr if detector == "psa" else None,
                           gso_thr=thr if detector == "gso" else None)
            vmask = pd.Series(m.index.isin(eval_ret.index[val_mask_idx]), index=m.index)
            tmask = pd.Series(m.index.isin(eval_ret.index[test_mask_idx]), index=m.index)
            acc_v, n_v = acc_directional(m, vmask)
            acc_t, n_t = acc_directional(m, tmask)
            curvas[detector].append({"pctile": pct, "acc_val": acc_v, "acc_test": acc_t,
                                     "n_val": n_v, "n_test": n_t})

    # Seleccion del p* SOLO en validacion (argmax acc_val) y comprobacion en test.
    seleccion, generalizacion = {}, {}
    for detector in ("psa", "gso"):
        pts = curvas[detector]
        accs_v = np.array([p["acc_val"] for p in pts])
        accs_t = np.array([p["acc_test"] for p in pts])
        i_star = int(np.nanargmax(accs_v))
        p_star = pts[i_star]["pctile"]
        gap = abs(accs_v[i_star] - accs_t[i_star])
        rho = float(scipy_stats.pearsonr(accs_v, accs_t)[0])

        # sign test de la accuracy de test en p* contra 0.5 (descartar azar a 50/50).
        thr = thresholds_at(detector, p_star, dist)
        reset_thresholds_cache()
        m_star = run_master(gamma, sigma, eval_ret, agents,
                            psa_thr=thr if detector == "psa" else None,
                            gso_thr=thr if detector == "gso" else None)
        tmask = pd.Series(m_star.index.isin(eval_ret.index[test_mask_idx]), index=m_star.index)
        sub = m_star.loc[tmask]
        valid = sub["r_next"].notna() & (np.sign(sub["r_next"]) != 0)
        correct = (np.sign(sub.loc[valid, "final_size"].to_numpy())
                   == np.sign(sub.loc[valid, "r_next"].to_numpy())).astype(int)
        k_s, n_s, p_s, ci_s = sign_test(correct)
        lo, hi, _ = stationary_bootstrap_ci(
            (correct - 0.5).astype(float), np.mean, n=2000, seed=config.SEED)

        seleccion[detector] = {
            "p_star": p_star, "acc_val_star": float(accs_v[i_star]),
            "acc_test_star": float(accs_t[i_star]), "gap_abs": float(gap),
            "seleccion_en_borde": bool(p_star in (GRID_PCTILES[0], GRID_PCTILES[-1])),
            "sign_test_test": {"k": int(k_s), "n": int(n_s), "p": float(p_s),
                               "ci95": [float(ci_s[0]), float(ci_s[1])]},
            "ci95_gap_low": float(lo), "ci95_gap_high": float(hi),
        }
        generalizacion[detector] = {
            "rho_pearson": rho, "rho_ge_0p5": bool(rho >= RHO_MIN),
            "gap_le_0p05": bool(gap <= GAP_MAX),
            "generaliza": bool(rho >= RHO_MIN and gap <= GAP_MAX),
        }

    h1 = all(g["generaliza"] for g in generalizacion.values())
    result = {
        "meta": {
            "ticker": ticker, "val_window": [VAL_START, VAL_END],
            "test_window": [STRATA_OOS_START, str(eval_ret.index.max().date())],
            "signal_lag": 1, "ram_tau": TAU_RAM, "override_variant": "C",
            "grid_pctiles": GRID_PCTILES, "seed": config.SEED,
            "n_val_days": int(val_mask_idx.sum()), "n_test_days": int(test_mask_idx.sum()),
            "calibration_window": [CALIBRATION_START, CALIBRATION_END],
            "hash_cache_agent": _hash_dir(CACHE_AGENT_DIR / ticker),
        },
        "curvas": curvas,
        "seleccion": seleccion,
        "generalizacion": generalizacion,
        "verdict": {
            "h1_sostenida": h1,
            "detectores_que_generalizan": [d for d, g in generalizacion.items() if g["generaliza"]],
            "comentario_neutral": ("El percentil elegido en validacion generaliza al test en "
                                   "ambos detectores." if h1 else
                                   "Al menos un detector no generaliza (rho<0.5 o gap>0.05): "
                                   "la seleccion por validacion no transfiere al OOS."),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Validacion final: claves contractadas en el pre-registro.
    loaded = json.loads(out.read_text())
    for key in ("meta", "curvas", "seleccion", "generalizacion", "verdict"):
        assert key in loaded, f"Falta clave de primer nivel: {key}"
    for det in ("psa", "gso"):
        assert loaded["curvas"][det], f"Curva {det} vacia"
        for pt in loaded["curvas"][det]:
            for k in ("pctile", "acc_val", "acc_test", "n_val", "n_test"):
                assert k in pt, f"Falta curvas.{det}[*].{k}"
        for k in ("p_star", "acc_val_star", "acc_test_star", "gap_abs", "seleccion_en_borde"):
            assert k in loaded["seleccion"][det], f"Falta seleccion.{det}.{k}"
        for k in ("rho_pearson", "rho_ge_0p5", "gap_le_0p05", "generaliza"):
            assert k in loaded["generalizacion"][det], f"Falta generalizacion.{det}.{k}"
    assert loaded["meta"]["signal_lag"] == 1, "signal_lag debe ser 1 (causal)"
    print(f"OK · {out} · {ticker} · h1_sostenida={result['verdict']['h1_sostenida']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="TSLA")
    args = ap.parse_args()
    config.set_seeds(config.SEED)
    main(args.ticker)
