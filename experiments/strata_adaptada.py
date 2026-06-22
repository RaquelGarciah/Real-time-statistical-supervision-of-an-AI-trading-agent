"""STRATA adaptada: ¿se alcanza el rendimiento de STRATA-U DESDE el supervisor real de Raquel
(override-C) moviendo umbrales y haciendo que RAM dispare más, sin cambiar la lógica de intervención?

Diagnóstico (verificado en el código). El RAM histórico (modo "mismatch") solo dispara cuando el
agente CONTRADICE al régimen (score = masa de probabilidad en el régimen incoherente). Por eso bajar
el umbral de RAM solo intensifica los overrides de contradicción y NO aprovecha el régimen cuando el
agente coincide o se abstiene. Para explotar el régimen en muchos más casos se añade a RAM el modo
"regime" (score = confianza del régimen direccional dominante max(P(Calma),P(Crisis))): RAM dispara
con la confianza del régimen INDEPENDIENTEMENTE del signo del agente, y el override-C —idéntico—
impone regime_sign·bound. El signo del régimen es data-driven por activo (signo de la media del
estado en calibración, leverage_screen.json), no hardcodeado leverage (CLAUDE.md §9).

Las configs adaptadas se calculan con `_adapted_final`, que reproduce el override-C del
StrataSupervisor (GSO paso 1 + RAM paso 2) de forma VECTORIZADA, omitiendo solo el freno PSA (×0.5 en
transición): PSA no cambia la dirección ni el vol-target y es lo único O(n²) del supervisor. Se valida
con un sanity-check: la config mismatch/absolute/τ=0.5 reproduce la accuracy de M8 de fair_sizing.

Salida: TODOS los activos × TODAS las métricas (acc/Sharpe/maxDD/Calmar/equity) × TODAS las estrategias
(M5/M8/M10/STRATA-U/Régimen/B&H/S&H/ZeroR + configs adaptadas). Sizing JUSTO (mismo vol-target para
todas) → Sharpe/maxDD aíslan la dirección; accuracy es independiente del sizing. M5/M8/STRATA-U/
Régimen/triviales se recomputan aquí (instantáneo); M10 (XGBoost walk-forward, lento) se LEE de
fair_sizing_compare.json sobre la MISMA ventana (mv[150:]). También se reporta el sizing NATIVO del
supervisor para las configs adaptadas.

Uso: python experiments/strata_adaptada.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from config import CACHE_AGENT_DIR, CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import calmar, equity_curve, max_drawdown, sharpe
import experiments.walkforward_robustez as wf
from experiments.strata_u import (_regime_drift, TARGET_VOL, CAP, TAU_CONF, REG_REL_MIN,
                                   AGENT_REL_MIN, AGENT_MIN_OBS)

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA",
         "IWM", "XLF", "XLK"]
N0 = 150
OUT = Path("outputs/experiments/strata_adaptada.json")
FAIR = Path("outputs/experiments/fair_sizing_compare.json")   # fuente de M10 (misma ventana)

REF = ["M5", "M8", "M10", "STRATA-U", "Régimen", "B&H", "S&H", "ZeroR"]

# Configs del supervisor real. tau = corte de override (medium); ram_thresholds=(tau/2, tau, 0.70).
#   mode_ram: "mismatch" (histórico, dispara en contradicción) | "regime" (dispara por confianza).
#   gso: "absolute" (M8, solo capa sobre-exposición) | "relative" (vol-target permanente).
#   sign: "lev" (leverage hardcoded) | "dd" (data-driven, signo de la media del estado en calibración).
CFGS = {
    "A_mm_τ50_abs":   dict(mode_ram="mismatch", gso="absolute", tau=0.50, sign="lev"),   # ≡ M8 (sanity)
    "A_mm_τ30_abs":   dict(mode_ram="mismatch", gso="absolute", tau=0.30, sign="lev"),   # solo bajar umbral RAM
    "A_mm_τ30_rel":   dict(mode_ram="mismatch", gso="relative", tau=0.30, sign="lev"),   # + vol-target
    "A_reg_lev_τ50":  dict(mode_ram="regime",   gso="relative", tau=0.50, sign="lev"),   # régimen confianza, leverage
    "A_reg_lev_τ45":  dict(mode_ram="regime",   gso="relative", tau=0.45, sign="lev"),
    "A_reg_dd_τ45":   dict(mode_ram="regime",   gso="relative", tau=0.45, sign="dd"),    # régimen confianza, data-driven
    "A_reg_dd_τ50":   dict(mode_ram="regime",   gso="relative", tau=0.50, sign="dd"),
    "A_reg_dd_τ55":   dict(mode_ram="regime",   gso="relative", tau=0.55, sign="dd"),
    # signo CAUSAL EXPANSIBLE (s_dom de STRATA-U) inyectado en el override-C → ¿reproduce STRATA-U?
    "A_reg_sdom_τ45": dict(mode_ram="regime",   gso="relative", tau=0.45, sign="sdom"),
    "A_reg_sdom_τ00": dict(mode_ram="regime",   gso="relative", tau=0.00, sign="sdom"),
}
ALL_STRATS = REF + list(CFGS)

# Umbral GSO absoluto medium = P99 de cache/models/strata_thresholds.json (recorta solo aquí).
GSO_ABS_MEDIUM = 5.5653

_LEV = None
_FAIR = None


def _sign_map(tk: str) -> dict:
    """Signo data-driven del régimen por estado = signo de la media de retorno del estado en calibración."""
    global _LEV
    if _LEV is None:
        _LEV = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    mr = _LEV[tk]["media_regimen"]
    return {"Calma": float(np.sign(mr["Calma"])), "Crisis": float(np.sign(mr["Crisis"]))}


def _complete(tk):
    import glob
    return len(glob.glob(str(CACHE_AGENT_DIR / tk / f"{tk}_*.json"))) >= 400


def _adapted_final(mv: pd.DataFrame, mode_ram: str, gso: str, tau: float, sign_map: dict | None,
                   sign_arr: np.ndarray | None = None) -> np.ndarray:
    """override-C VECTORIZADO (RAM modo `mode_ram` + GSO `gso`), idéntico a StrataSupervisor salvo el
    freno PSA (×0.5 en transición), que se omite (no cambia dirección ni vol-target).

    Paso 1 (GSO): relative reescala al vol-target (sign·bound) cuando |log2(|size|/bound)|≥0.585
    (severidad ≥ medium); absolute solo recorta sobre-exposición cuando score ≥ P99. Paso 2 (RAM):
    si el score de RAM ≥ τ (medium), el tamaño pasa a regime_sign·bound (override-C).
    """
    size = mv["agent_size"].to_numpy(float)
    calm = mv["calm_prob"].to_numpy(float); crisis = mv["crisis_prob"].to_numpy(float)
    sigma = mv["garch_sigma"].to_numpy(float)
    bound = np.where(sigma > 0, np.minimum(1.0, TARGET_VOL / sigma), 1.0)
    asize = np.abs(size); asign = np.sign(size)
    final = size.copy()

    if gso == "relative":
        ratio = asize / np.maximum(bound, 1e-3)
        dev = np.where(ratio > 0, np.abs(np.log2(np.where(ratio > 0, ratio, 1.0))), np.inf)
        gso_fire = (ratio <= 0) | (dev >= 0.585)
        final = np.where(gso_fire, asign * bound, final)
    else:  # absolute
        score = np.maximum(0.0, asize - bound) / np.maximum(bound, 1e-3)
        final = np.where(score >= GSO_ABS_MEDIUM, asign * np.minimum(asize, bound), final)

    dom_calm = calm >= crisis
    pdom = np.where(dom_calm, calm, crisis)
    if mode_ram == "regime":
        rscore = np.minimum(1.0, pdom)
    else:
        rscore = np.minimum(1.0, np.where(asign < 0, calm, 0.0) + np.where(asign > 0, crisis, 0.0))
    ram_fire = rscore >= tau
    if sign_arr is not None:                      # signo causal expansible por día (s_dom de STRATA-U)
        rsign = np.where(sign_arr != 0, sign_arr, np.where(dom_calm, 1.0, -1.0))
    elif sign_map is not None:                    # signo data-driven estático de calibración
        rsign = np.where(dom_calm, sign_map["Calma"], sign_map["Crisis"])
        rsign = np.where(rsign == 0, np.where(dom_calm, 1.0, -1.0), rsign)
    else:                                         # leverage por defecto
        rsign = np.where(dom_calm, 1.0, -1.0)
    final = np.where(ram_fire, rsign * bound, final)
    return final


def _metr(nrx, dir_arr, truth):
    return {"acc": round(float((dir_arr == truth).mean()), 4), "sharpe": round(float(sharpe(nrx)), 4),
            "maxdd": round(float(max_drawdown(equity_curve(nrx))), 4),
            "calmar": round(float(calmar(nrx)), 4),
            "equity": round(float((1 + nrx.fillna(0)).prod()), 4)}


def _row(tk: str) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    reg, gamma, sigma, oos_ret = _regime_drift(tk)
    agents = wf.load_agent(tk)
    recs = {}
    for t in sorted(agents):
        if t in gamma.index and t in sigma.index:
            g = gamma.loc[t]
            recs[t] = (agents[t].size, float(g["Calma"]), float(g["Crisis"]), float(sigma.loc[t]))
    m = pd.DataFrame.from_dict(recs, orient="index",
                               columns=["agent_size", "calm_prob", "crisis_prob", "garch_sigma"])
    m["r_next"] = oos_ret.shift(-1).reindex(m.index)
    mv = m[m["r_next"].notna() & (np.sign(m["r_next"].to_numpy()) != 0)].copy()
    sub = mv.index[N0:]
    sel = np.zeros(len(mv), dtype=bool); sel[N0:] = True
    truth = np.sign(mv["r_next"].to_numpy())[sel]
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    vs = np.where(mv["garch_sigma"].to_numpy() > 0,
                  np.minimum(CAP, TARGET_VOL / mv["garch_sigma"].to_numpy()), CAP)[sel]

    # --- STRATA-U (regime_flip, dirección) ---
    R = reg.reindex(mv.index)
    s_dom = R["s_dom"].to_numpy(); reg_gate = R["reg_gate"].to_numpy()
    drift = np.where(R["drift"].to_numpy() != 0, R["drift"].to_numpy(), 1.0); gmax = R["gmax"].to_numpy()
    reg_on = (reg_gate > REG_REL_MIN) & (gmax >= TAU_CONF) & (s_dom != 0)
    agent_sign = np.sign(mv["agent_size"].to_numpy())
    ag_gate = pd.Series((agent_sign == np.sign(mv["r_next"].to_numpy())).astype(float),
                        index=mv.index).expanding().mean().shift(1).fillna(0.5).to_numpy()
    agent_ok = (np.arange(len(mv)) >= AGENT_MIN_OBS) & (ag_gate > AGENT_REL_MIN) & (agent_sign != 0)
    su = np.where(reg_on, s_dom, drift); su = np.where(agent_ok, agent_sign, su); su = np.where(su == 0, 1.0, su)

    dirs = {
        "M5": agent_sign[sel],
        "Régimen": s_dom[sel], "STRATA-U": su[sel],
        "B&H": np.ones_like(truth), "S&H": -np.ones_like(truth), "ZeroR": np.full_like(truth, maj)}
    # M8 = config mismatch/absolute/τ=0.5 (idéntico al override-C de run_master salvo PSA)
    fs_m8 = _adapted_final(mv, "mismatch", "absolute", 0.50, None)
    dirs["M8"] = np.sign(fs_m8)[sel]

    sm = _sign_map(tk)
    native = {"M5": mv["agent_size"].to_numpy()[sel], "M8": fs_m8[sel]}
    for name, cfg in CFGS.items():
        fs = _adapted_final(mv, cfg["mode_ram"], cfg["gso"], cfg["tau"],
                            sm if cfg["sign"] == "dd" else None,
                            sign_arr=s_dom if cfg["sign"] == "sdom" else None)
        dirs[name] = np.sign(fs)[sel]; native[name] = fs[sel]

    # --- métricas con sizing JUSTO (vol-target) para todas las direcciones recomputadas aquí ---
    def fair_metr(d):
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = d * vs
        nrx = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        return _metr(nrx, d, truth)

    est = {nm: fair_metr(d) for nm, d in dirs.items()}
    est["ZeroR"]["acc"] = round(max(frac_up, 1 - frac_up), 4)
    # M10: leído de fair_sizing_compare.json (misma ventana, mismo sizing); equity no disponible allí
    global _FAIR
    if _FAIR is None:
        _FAIR = json.load(open(FAIR))["por_activo"]
    m10 = dict(_FAIR[tk]["estrategias"]["M10"]); m10.setdefault("equity", None)
    est["M10"] = m10

    # --- métricas con sizing NATIVO del supervisor (configs + M5/M8) ---
    est_nat = {}
    for nm, fs in native.items():
        w = pd.Series(0.0, index=mv.index); w.loc[sub] = fs
        nrx = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
        est_nat[nm] = _metr(nrx, np.sign(fs), truth)

    # sanity: A_mm_τ50_abs (≡M8) reproduce la accuracy de M8 de fair_sizing
    m8_fair = _FAIR[tk]["estrategias"]["M8"]["acc"]
    sanity = round(abs(est["A_mm_τ50_abs"]["acc"] - m8_fair), 4)

    return {"n": int(sel.sum()), "frac_up": round(frac_up, 4), "clase": _LEV[tk]["clase"],
            "sign_map": sm, "fair": est, "nativo": est_nat, "sanity_m8_absdiff": sanity,
            "interv": {nm: round(float((dirs[nm] != dirs["M5"]).mean()), 4) for nm in CFGS}}


def main() -> None:
    wf.reset_thresholds_cache()
    rows = {}
    for tk in PANEL:
        if not _complete(tk):
            print(f"{tk:5s} (sin caché completa, omitido)"); continue
        try:
            rows[tk] = _row(tk); e = rows[tk]["fair"]
            best = max(CFGS, key=lambda c: e[c]["acc"])
            print(f"{tk:5s} n={rows[tk]['n']} up={rows[tk]['frac_up']:.2f} sane={rows[tk]['sanity_m8_absdiff']:.3f} | "
                  f"M8={e['M8']['acc']:.3f} U={e['STRATA-U']['acc']:.3f} reg={e['Régimen']['acc']:.3f} "
                  f"dd45={e['A_reg_dd_τ45']['acc']:.3f} dd50={e['A_reg_dd_τ50']['acc']:.3f} dd55={e['A_reg_dd_τ55']['acc']:.3f} "
                  f"| best={best}({e[best]['acc']:.3f})", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{tk:5s} ERROR {type(ex).__name__}: {ex}", flush=True)
    A = list(rows)

    def avg(strat, met):
        vals = [rows[t]["fair"][strat].get(met) for t in A]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    medias = {s: {m: avg(s, m) for m in ("acc", "sharpe", "maxdd", "calmar", "equity")} for s in ALL_STRATS}

    def cnt(s, met, ref, ge=True):
        c = 0
        for t in A:
            a = rows[t]["fair"][s].get(met); b = rows[t]["fair"][ref].get(met)
            if a is None or b is None:
                continue
            c += (a >= b - 1e-9) if ge else (a > b)
        return f"{c}/{len(A)}"
    cob = {s: {"acc≥M5": cnt(s, "acc", "M5"), "sharpe≥M5": cnt(s, "sharpe", "M5"),
               "acc≥M8": cnt(s, "acc", "M8"), "sharpe≥M8": cnt(s, "sharpe", "M8"),
               "acc≥STRATA-U": cnt(s, "acc", "STRATA-U"), "sharpe≥STRATA-U": cnt(s, "sharpe", "STRATA-U"),
               "acc>ZeroR": cnt(s, "acc", "ZeroR", ge=False), "sharpe>ZeroR": cnt(s, "sharpe", "ZeroR", ge=False)}
           for s in CFGS}

    res = {"meta": {"activos": A, "ventana": "mv[150:] (≡ ventana M10 de fair_sizing_compare.json)",
                    "sizing_fair": "mismo vol-target (target_vol/σ) para TODAS → aísla dirección",
                    "m10_fuente": "fair_sizing_compare.json (no se recomputa)",
                    "configs": CFGS, "estrategias": ALL_STRATS, "seed": config.SEED, "signal_lag": 1,
                    "nota": "STRATA adaptada = MISMO override-C; solo cambia ram_mode, umbral RAM, "
                            "gso_mode y signo data-driven (PSA omitido, no afecta dirección). Exploratorio (docs/)."},
           "por_activo": rows, "medias_fair": medias, "cobertura": cob}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== MEDIAS (sizing justo vol-target para TODAS) ===")
    print(f"  {'estrategia':14s}{'acc':>8s}{'Sharpe':>9s}{'maxDD':>9s}{'Calmar':>9s}{'equity':>9s}")
    for s in ALL_STRATS:
        r = medias[s]
        eq = f"{r['equity']:>9.3f}" if r["equity"] is not None else f"{'—':>9s}"
        print(f"  {s:14s}{r['acc']:>8.3f}{r['sharpe']:>9.2f}{r['maxdd']:>8.1%}{r['calmar']:>9.2f}{eq}")
    print("\n=== COBERTURA de cada config adaptada (de", len(A), "activos) ===")
    for s in CFGS:
        c = cob[s]
        print(f"  {s:14s} acc≥M8={c['acc≥M8']} Sh≥M8={c['sharpe≥M8']} | "
              f"acc≥U={c['acc≥STRATA-U']} Sh≥U={c['sharpe≥STRATA-U']} | "
              f"acc≥M5={c['acc≥M5']} | acc>ZeroR={c['acc>ZeroR']} Sh>ZeroR={c['sharpe>ZeroR']}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
