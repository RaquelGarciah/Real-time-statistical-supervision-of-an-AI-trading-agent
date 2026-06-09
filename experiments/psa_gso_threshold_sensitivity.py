"""E4 — Sensibilidad de los umbrales de PSA y GSO sobre SPY (DIAGNÓSTICO, no selección).

Barre hacia abajo los umbrales de severidad de PSA y GSO (P50/P75/P90/P95/P99 de su
score en calibración) para FORZAR que disparen en el OOS de SPY, y mide cómo cambian
nº de intervenciones de cada detector, accuracy direccional, hit_rate, Sharpe causal,
turnover y McNemar M8 vs M5. RAM se mantiene FIJO en τ=0.5 (solo se mueven PSA y GSO).

Es sensibilidad, NO recalibración: se reportan TODOS los puntos del barrido. PROHIBIDO
adoptar el que maximiza accuracy/Sharpe OOS como nuevo default (sería look-ahead). El
default sigue siendo P95/P99 ex-ante de cache/models/strata_thresholds.json.

Mecánica que justifica la hipótesis nula (strata/intervention.py, override, en cascada):
GSO medium/high recorta magnitud (bounded_size, capa hacia abajo, NO voltea signo);
RAM medium/high reorienta a regime_sign·bound (único que voltea, sobrescribe a GSO);
PSA HIGH multiplica el size por 0.5 (freno, NO voltea). Luego PSA/GSO solo modulan
magnitud: H1 honesta = bajar sus umbrales NO mejora el accuracy direccional.

Pre-registro: BITACORA.md [2026-06-09] [Pre-registro] E4.
Uso: ``python experiments/psa_gso_threshold_sensitivity.py``
"""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START, DATA_DIR, STRATA_OOS_START
from core import data, features, metrics
from core.backtest import run_backtest
from core.stats import mcnemar_test, stationary_bootstrap_ci
from strata.detectors import reset_thresholds_cache
from strata.strata import StrataSupervisor
from strata.types import AgentOutput, PersonalityOutput

TICKER = "SPY"
TAU_RAM = 0.5
RAM_THRESHOLDS = (TAU_RAM / 2, TAU_RAM, 0.70)  # fijo: solo se mueven PSA/GSO
# Percentiles del barrido. P95/P99 = default ex-ante; los más bajos fuerzan disparo.
SWEEP_PCTILES = [50, 75, 90, 95, 99]
OUT = Path("outputs/experiments/psa_gso_threshold_sensitivity.json")


def _hash_dir(path: Path, pattern: str = "*") -> str:
    h = hashlib.sha256()
    for fp in sorted(path.glob(pattern)):
        if fp.is_file():
            h.update(fp.name.encode()); h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def load_agent(ticker: str) -> dict:
    """Decisiones del agente desde cache/agent/<ticker>/ (mismo loader que el canónico)."""
    out = {}
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        d = json.load(open(fp))
        pers = {k: PersonalityOutput(name=k, action=v["action"], size=v["size"],
                                     confidence=v["confidence"], reasoning="")
                for k, v in d.get("personalities", {}).items()}
        out[pd.Timestamp(d["date"])] = AgentOutput(
            date=d["date"], ticker=d["ticker"], action=d["action"], size=d["size"],
            confidence=d["confidence"], reasoning="", personalities=pers)
    return out


def build_market_states() -> tuple[dict, pd.Series, pd.Series]:
    """Reusa HMM/GARCH cacheados (NO recalcula) y devuelve regímenes filtrados, sigma y oos_ret.

    Régimen FILTRADO causal (gamma_df) y sigma GARCH causal: idéntico a notebooks/_build.py §2.
    """
    spy_parquets = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))
    data_end = spy_parquets[-1].rsplit("_", 1)[1].replace(".parquet", "")
    prices = data.load_market_data(TICKER, CALIBRATION_START, data_end)
    ret = features.log_returns(prices["Close"])
    rv21 = features.realized_vol_annualized(ret, window=21)
    feat_df = pd.concat([ret.rename("r"), rv21.rename("rv")], axis=1).dropna()

    hmm = pickle.load(open(CACHE_MODELS_DIR / "hmm.pkl", "rb"))
    garch = pickle.load(open(CACHE_MODELS_DIR / f"garch_{TICKER}.pkl", "rb"))
    gamma = hmm.predict_proba_filtered(feat_df.to_numpy())
    gamma_df = pd.DataFrame(gamma, index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
    oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    sigma = garch.forecast_path(oos_ret)
    return gamma_df, sigma, oos_ret


def thresholds_at(detector: str, pct: int) -> tuple[float, float, float]:
    """Umbrales (low, medium, high) a percentil ``pct`` del score de calibración.

    Lee la score_distribution ex-ante de strata_thresholds.json (p50/.../p99/max). El umbral
    que se mueve depende de DÓNDE interviene cada detector (strata/intervention.py):

    - **PSA** solo frena en severidad ``high`` → el barrido mueve ``high`` (low=0, medium=high=
      p{pct}); con ``_severity_from_levels`` un score ≥ p{pct} cae directo a ``high`` y PSA dispara.
    - **GSO** interviene en ``medium``+ → el barrido mueve ``medium`` (low=medium=p{pct}, high=max).

    Bajar ``pct`` ⇒ umbral más bajo ⇒ el detector dispara más. Decisión metodológica
    (distinto umbral por detector) documentada en el pre-registro de BITACORA.
    """
    dist = json.load(open(CACHE_MODELS_DIR / "strata_thresholds.json"))[detector]["score_distribution"]
    p = float(dist[f"p{pct}"])
    if detector == "psa":
        return (0.0, p, p)                 # score ≥ p{pct} → high (PSA frena)
    return (p, p, float(dist["max"]))      # score ≥ p{pct} → medium+ (GSO recorta)


def run_master(gamma_df, sigma, oos_ret, agents, psa_thr=None, gso_thr=None) -> pd.DataFrame:
    """Recorre el OOS día a día con override-C, RAM fijo y umbrales PSA/GSO inyectados.

    Los umbrales ``psa_thr``/``gso_thr`` se inyectan vía los hooks ``psa_thresholds``/
    ``gso_thresholds`` del ``StrataSupervisor`` (paralelos a ``ram_thresholds``), que los
    propaga a los detectores; estos re-clasifican la severidad con ``_severity_from_levels``
    sobre el score (invariante al umbral), sin recalcular HMM/GARCH ni tocar el JSON global.
    """
    sup = StrataSupervisor(
        mode="override", override_variant="C", gso_mode="absolute",
        psa_signal="cp_prob", psa_hazard=config.BOCPD_HAZARD,
        ram_thresholds=RAM_THRESHOLDS,
        psa_thresholds=psa_thr, gso_thresholds=gso_thr,
    )
    rows, sizing_hist = [], []
    for t in sorted(agents):
        if t not in gamma_df.index or t not in sigma.index:
            continue
        a = agents[t]; sizing_hist.append(a.size)
        g = gamma_df.loc[t]
        ms = {"regime": {"calm_prob": float(g["Calma"]), "stress_prob": float(g["Estrés"]),
                         "crisis_prob": float(g["Crisis"]), "viterbi_state": int(np.argmax(g.values))},
              "garch_vol_annualized": float(sigma.loc[t])}
        dec = sup.supervise(a, ms, sizing_hist)
        rows.append({"date": t, "agent_size": a.size, "final_size": dec.final_size,
                     "psa_score": dec.detectors["psa"].score, "psa_sev": dec.detectors["psa"].severity,
                     "gso_score": dec.detectors["gso"].score, "gso_sev": dec.detectors["gso"].severity,
                     "ram_sev": dec.detectors["ram"].severity, "intervenido": dec.was_intervened})
    m = pd.DataFrame(rows).set_index("date")
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    m["y"] = np.sign(m["r_next"])
    return m


def directional_correct(m: pd.DataFrame, size_col: str) -> np.ndarray:
    """Aciertos direccionales pareados: sign(size_t) == sign(r_next_t), sobre días con r_next≠0."""
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    pred = np.sign(m.loc[valid, size_col].to_numpy())
    truth = np.sign(m.loc[valid, "r_next"].to_numpy())
    return (pred == truth).astype(int)


def stats_for(m: pd.DataFrame, oos_ret: pd.Series, size_col: str) -> dict:
    """accuracy, MCC, hit_rate, Sharpe causal, turnover, equity €1000 y nº días válidos."""
    from sklearn.metrics import matthews_corrcoef

    bt = run_backtest(oos_ret, m[size_col], signal_lag=1)  # signal_lag=1: causal, NO same-day
    correct = directional_correct(m, size_col)
    # MCC sobre la dirección up/down (robusto a desbalanceo n_up≠n_down; lección #11). Se
    # computa solo en días con apuesta direccional (sign≠0) para mantener el problema binario.
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    nz = valid & (np.sign(m[size_col]) != 0)
    try:
        mcc = float(matthews_corrcoef(np.sign(m.loc[nz, "r_next"]), np.sign(m.loc[nz, size_col])))
    except Exception:
        mcc = float("nan")
    return {
        "accuracy": float(correct.mean()),
        "mcc": mcc,
        "n_valid": int(valid.sum()),
        "hit_rate": metrics.hit_rate(bt["net_return"]),
        "sharpe_causal": metrics.sharpe(bt["net_return"]),
        "turnover": metrics.turnover(m[size_col]),
        "equity_final_1000": float(1000.0 * bt["equity"].iloc[-1]),
        "_correct": correct,            # interno, para McNemar; se elimina antes de serializar
        "_net_return": bt["net_return"],  # interno, para el IC bootstrap
    }


def mcnemar_dict(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    # mcnemar_test devuelve (estadístico, p, b, c); b = a✓&b✗, c = a✗&b✓.
    stat, p, b, c = mcnemar_test(correct_a, correct_b)
    return {"p": float(p), "b": int(b), "c": int(c), "n_disc": int(b + c)}


def main():
    gamma_df, sigma, oos_ret = build_market_states()
    agents = load_agent(TICKER)

    # M5 (agente solo) y M8 base (override-C con PSA/GSO al default ex-ante P95/P99).
    reset_thresholds_cache()
    m_base = run_master(gamma_df, sigma, oos_ret, agents)
    s_m5 = stats_for(m_base, oos_ret, "agent_size")
    s_base = stats_for(m_base, oos_ret, "final_size")
    base_vs_m5 = mcnemar_dict(s_base["_correct"], s_m5["_correct"])

    sweep = []
    refuta = 0
    for detector in ("psa", "gso"):
        for pct in SWEEP_PCTILES:
            # thresholds_at mueve el umbral en la severidad donde cada detector interviene
            # (PSA: high; GSO: medium). Bajar pct ⇒ dispara más.
            thr = thresholds_at(detector, pct)
            reset_thresholds_cache()
            psa_thr = thr if detector == "psa" else None
            gso_thr = thr if detector == "gso" else None
            m = run_master(gamma_df, sigma, oos_ret, agents, psa_thr=psa_thr, gso_thr=gso_thr)
            s = stats_for(m, oos_ret, "final_size")
            mc_m5 = mcnemar_dict(s["_correct"], s_m5["_correct"])
            mc_base = mcnemar_dict(s["_correct"], s_base["_correct"])
            d_acc = s["accuracy"] - s_base["accuracy"]
            # Regla de refutación pre-registrada (coherente en signo): un punto REFUTA H1 si
            # mejora la dirección de forma significativa, es decir Δacc>+0.02 Y McNemar vs base
            # p<0.10 Y los discordantes favorecen al sweep (b>c, pues mcnemar_dict(sweep,base)
            # define b = sweep✓&base✗). Así un punto que EMPEORA con p bajo no cuenta como refute.
            refuta_pt = bool(d_acc > 0.02 and mc_base["p"] < 0.10 and mc_base["b"] > mc_base["c"])
            if refuta_pt:
                refuta += 1
            # El detector interviene en 'high' (PSA) o 'medium'+ (GSO): contamos lo pertinente.
            n_fire = int((m["psa_sev"] == "high").sum()) if detector == "psa" \
                else int((m["gso_sev"].isin(["medium", "high"])).sum())
            sweep.append({
                "detector": detector, "pctile": pct,
                "umbral_movido": "high" if detector == "psa" else "medium",
                "thresh": thr[2] if detector == "psa" else thr[1],
                "n_intervenciones_detector": n_fire,
                "accuracy": s["accuracy"], "mcc": s["mcc"], "n_valid": s["n_valid"],
                "delta_accuracy_vs_base": float(d_acc),
                "hit_rate": s["hit_rate"], "sharpe_causal": s["sharpe_causal"],
                "turnover": s["turnover"], "equity_final_1000": s["equity_final_1000"],
                "mcnemar_vs_m5": mc_m5, "mcnemar_vs_base": mc_base,
                "refuta_h1": refuta_pt,
            })

    # IC bootstrap del ΔSharpe del punto más extremo (P50) vs base, por detector.
    ci_extreme = []
    for detector in ("psa", "gso"):
        thr = thresholds_at(detector, 50)
        reset_thresholds_cache()
        m = run_master(gamma_df, sigma, oos_ret, agents,
                       psa_thr=thr if detector == "psa" else None,
                       gso_thr=thr if detector == "gso" else None)
        s = stats_for(m, oos_ret, "final_size")
        diff = (s["_net_return"] - s_base["_net_return"]).dropna().to_numpy()
        lo, hi, mean = stationary_bootstrap_ci(diff, np.mean, n=2000, seed=config.SEED)
        ci_extreme.append({"detector": detector, "pctile": 50,
                           "delta_sharpe": s["sharpe_causal"] - s_base["sharpe_causal"],
                           "ci95_low_meandiff": float(lo), "ci95_high_meandiff": float(hi),
                           "excluye_cero": not (lo <= 0 <= hi)})

    max_dacc = max((p["delta_accuracy_vs_base"] for p in sweep), default=0.0)
    _metric_keys = ("accuracy", "mcc", "n_valid", "hit_rate", "sharpe_causal", "turnover", "equity_final_1000")
    result = {
        "meta": {
            "ticker": TICKER, "oos_start": str(oos_ret.index.min().date()),
            "oos_end": str(oos_ret.index.max().date()), "n_days": int(len(m_base)),
            "signal_lag": 1, "ram_tau": TAU_RAM, "override_variant": "C",
            "calibration_window": [CALIBRATION_START, CALIBRATION_END], "seed": config.SEED,
            # n_trials = grados de libertad del barrido (5 pct × 2 detectores). Declarado
            # ex-ante para el Deflated Sharpe del paso 5 si se mencionara algún Sharpe favorable.
            "n_trials": len(SWEEP_PCTILES) * 2,
            "hash_cache_agent": _hash_dir(CACHE_AGENT_DIR / TICKER),
            "hash_cache_models": _hash_dir(CACHE_MODELS_DIR),
        },
        "m5": {k: s_m5[k] for k in _metric_keys},
        "base": {
            "threshold_label": "P95/P99 ex-ante",
            "n_psa_medium_plus": int((m_base["psa_sev"].isin(["medium", "high"])).sum()),
            "n_psa_high": int((m_base["psa_sev"] == "high").sum()),
            "n_gso_medium_plus": int((m_base["gso_sev"].isin(["medium", "high"])).sum()),
            **{k: s_base[k] for k in _metric_keys},
            "mcnemar_vs_m5": base_vs_m5,
        },
        "sweep": sweep,
        "ci_sharpe_extreme_vs_base": ci_extreme,
        "verdict": {
            "h1_sostenida": refuta == 0,
            "n_puntos_refutan": refuta,
            "max_delta_accuracy": float(max_dacc),
            # Comentario NEUTRAL: reporta el hecho, no prejuzga la causa (la causa se dictamina
            # en el paso 5 de auditoría de resultados, no se pre-escribe aquí).
            "comentario": ("Ningún punto del barrido cruza el umbral de refutación: bajar los "
                           "umbrales de PSA/GSO no mejora el acierto direccional (consistente con "
                           "que solo modulan magnitud)."
                           if refuta == 0 else
                           f"{refuta} punto(s) cruzan el umbral de refutación (Δacc>0.02, McNemar "
                           "vs base p<0.10, b>c): requiere análisis en la auditoría de resultados."),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Validación final: las claves del JSON contractadas en el pre-registro existen.
    loaded = json.loads(OUT.read_text())
    for key in ("meta", "m5", "base", "sweep", "ci_sharpe_extreme_vs_base", "verdict"):
        assert key in loaded, f"Falta la clave de primer nivel: {key}"
    for key in ("accuracy", "mcc", "hit_rate", "sharpe_causal", "turnover", "mcnemar_vs_m5"):
        assert key in loaded["base"], f"Falta base.{key}"
    assert loaded["sweep"], "El barrido está vacío"
    for p in loaded["sweep"]:
        for key in ("detector", "pctile", "umbral_movido", "n_intervenciones_detector",
                    "accuracy", "mcc", "n_valid", "delta_accuracy_vs_base", "sharpe_causal",
                    "turnover", "mcnemar_vs_m5", "mcnemar_vs_base", "refuta_h1"):
            assert key in p, f"Falta sweep[*].{key}"
    assert loaded["meta"]["signal_lag"] == 1, "signal_lag debe ser 1 (causal)"
    assert loaded["meta"]["n_trials"] == len(SWEEP_PCTILES) * 2, "n_trials mal declarado"
    print(f"OK · {OUT} · {len(sweep)} puntos · H1 sostenida={result['verdict']['h1_sostenida']}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
