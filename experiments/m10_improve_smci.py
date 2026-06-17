"""Mejora honesta de la accuracy de M10 sobre SMCI (split validación/test, desplegable).

Las palancas son HIPERPARÁMETROS, se eligen SOLO en validación (primeros 60% del OOS) y se reportan en
test (últimos 40%, intacto, se toca una vez). El walk-forward sigue reentrenando sobre el pasado expandible
en el test (eso es el despliegue; arregla el colapso del modelo congelado 60/40 = 0.150). Lo que se congela
de validación son: feature set, decaimiento de recencia y umbral; NO un modelo.

Palancas (las 5 que pidió Raquel):
  1. Umbral ≠ 0.5 (grid pre-registrado, elegido en validación).
  2. Selección de features (all22 / régimen+STRATA-7 / agente-15 / STRATA7+señal real / all22+señal real).
  3. Pesos por recencia (no estacionariedad de SMCI): semivida {flat, 252, 126}.
  4. Ensemble de semillas (10) para reducir varianza — aplicado en todo; se reporta 1-seed como referencia.
  5. Features con señal real, CAUSALES (momentum 5/21/63, vol relativa rv21/rv63, racha de signo).

Selección en validación = walk-forward dentro de [N0:split] (nunca ve el test). Test = walk-forward con
pasado expandible (incluye validación) prediciendo [split:fin]. Disciplina validación≠test.

Pre-registro: BITACORA.md [2026-06-16]. Uso: python experiments/m10_improve_smci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from core.features import realized_vol_annualized
from core.stats import block_permutation_test, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import AGENT15, ALL22, STRATA_REGIME7

TICKER = "SMCI"
N0, STEP, EMBARGO = 150, 21, 1   # embargo=1: horizonte de etiqueta=1 (Tashman 2000; LdP 2018 §7.4). Ver BITACORA 2026-06-17
VAL_FRAC = 0.60
N_SEEDS = 10
THR_GRID = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
HALFLIVES = {"flat": None, "hl252": 252, "hl126": 126}
REALSIG = ["mom5", "mom21", "mom63", "relvol", "streak"]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
OUT = Path("outputs/experiments/m10_improve_smci.json")


def build_realsignal(ret: pd.Series, index: pd.Index) -> pd.DataFrame:
    """Features causales conocidas en t (predicen r_{t+1}): momentum, vol relativa, racha de signo.

    Todas usan solo información hasta t (rolling sin shift negativo) → sin look-ahead (A1 rigor). La racha
    usa np.sign(r): un día de retorno EXACTAMENTE nulo (sign=0) reinicia la racha a 0 y se propaga; es una
    decisión de diseño (días planos = ruptura de tendencia), no un bug. En SMCI los retornos nulos son ~0.
    """
    rv21 = realized_vol_annualized(ret, window=21)
    rv63 = realized_vol_annualized(ret, window=63)
    sign = np.sign(ret)
    # racha: longitud (con signo) de la serie de retornos del mismo signo terminando en t
    streak = sign.copy().astype(float)
    vals = sign.to_numpy()
    run = np.zeros(len(vals))
    for i in range(len(vals)):
        run[i] = vals[i] if i == 0 or vals[i] != vals[i - 1] else run[i - 1] + vals[i]
    streak = pd.Series(run, index=ret.index)
    df = pd.DataFrame({"mom5": ret.rolling(5).sum(), "mom21": ret.rolling(21).sum(),
                       "mom63": ret.rolling(63).sum(), "relvol": rv21 / rv63, "streak": streak})
    return df.reindex(index)


def recency_weight(tr_end: int, hl) -> np.ndarray | None:
    """Pesos por recencia para fit([:tr_end]): w_i = 0.5^(antigüedad_i / semivida). flat → None."""
    if hl is None:
        return None
    ages = (tr_end - 1) - np.arange(tr_end)
    return 0.5 ** (ages / hl)


def wf_p1(X: pd.DataFrame, y: pd.Series, start_hi: int, pred_lo: int, hl, seeds) -> pd.Series:
    """Walk-forward expandible (solo pasado). Reentrena en range(N0, start_hi, STEP); predice [pred_lo:].

    Ensemble: promedia predict_proba de `seeds`. start_hi acota dónde puede empezar a reentrenar (en
    validación = split, para no tocar el test). pred_lo descarta predicciones anteriores a ese índice.

    Embargo: la etiqueta y_i = 1[r_{i+1}>0] tiene HORIZONTE 1 día. En walk-forward rolling-origin (Tashman
    2000) el test es siempre futuro respecto al train → no hay solape bidireccional (eso es lo que motiva el
    embargo de CPCV, López de Prado 2018 §7.4). El único solape es el de la etiqueta de horizonte 1, que se
    purga con EMBARGO=1 (mínimo correcto = horizonte). El "embargo≥5" de CLAUDE.md §4 es regla de CPCV (folds
    interleaved) / etiquetas multi-día, no de este WF. Validez con hueco mínimo bajo residuos no correlados:
    Bergmeir, Hyndman & Koo (2018). Decisión y respaldo: BITACORA 2026-06-17, logic_esential §14b.
    """
    n = len(X)
    p = pd.Series(np.nan, index=X.index)
    for start in range(N0, start_hi, STEP):
        tr_end = start - EMBARGO
        if tr_end < 60:
            continue
        sw = recency_weight(tr_end, hl)
        end = min(start + STEP, n)
        preds = []
        for sd in seeds:
            clf = xgb.XGBClassifier(**PARAMS, random_state=sd)
            clf.fit(X.iloc[:tr_end], y.iloc[:tr_end], sample_weight=sw)
            preds.append(clf.predict_proba(X.iloc[start:end])[:, 1])
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p.iloc[pred_lo:] if pred_lo else p


def acc_at(p: pd.Series, thr: float, truth: np.ndarray) -> float:
    pos = np.where(p.to_numpy() >= thr, 1.0, -1.0)
    return float((pos == truth).mean())


def main() -> None:
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    _, ret_full = wf.load_features(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    rs = build_realsignal(ret_full, mv.index)
    mv[REALSIG] = rs
    mv = mv.dropna(subset=REALSIG)                       # descarta días sin histórico de momentum (inicio 2007 → ninguno)
    n = len(mv)
    assert 360 <= n <= 440, f"B5: n={n} difiere >10% de 400 pre-registrado; el split se movería sin traza"
    # B2: split alineado a múltiplo de STEP desde N0 → en test el primer reentreno cae EXACTO en split
    # (entrena [:split−EMBARGO], predice desde split). Evita el acoplamiento de rejilla en la frontera 60/40.
    split = N0 + STEP * ((int(n * VAL_FRAC) - N0) // STEP)
    y = (mv["r_next"] > 0).astype(int)
    truth = np.sign(mv["r_next"].to_numpy())

    feature_sets = {"all22": ALL22, "regime_strata7": STRATA_REGIME7, "agent15": AGENT15,
                    "strata7+real": STRATA_REGIME7 + REALSIG, "all22+real": ALL22 + REALSIG}

    # ---- Selección SOLO en validación: WF dentro de [N0:split], predice [N0:split] ----
    val_truth = truth[N0:split]
    grid = []
    for fname, cols in feature_sets.items():
        X = mv[cols]
        for dname, hl in HALFLIVES.items():
            p_val = wf_p1(X, y, start_hi=split, pred_lo=N0, hl=hl, seeds=SEEDS)
            p_val = p_val.iloc[:split - N0]              # solo días de validación
            best_thr, best_acc = max(((t, acc_at(p_val, t, val_truth)) for t in THR_GRID), key=lambda z: z[1])
            grid.append({"features": fname, "recency": dname, "thr": best_thr,
                         "acc_val": round(best_acc, 4), "acc_val_thr05": round(acc_at(p_val, 0.5, val_truth), 4)})

    best = max(grid, key=lambda g: g["acc_val"])
    base = next(g for g in grid if g["features"] == "all22" and g["recency"] == "flat")

    # ---- TEST (intacto, una vez): WF con pasado expandible (incl. validación), predice [split:fin] ----
    def test_p1(cols, hl, seeds):
        return wf_p1(mv[cols], y, start_hi=n, pred_lo=split, hl=hl, seeds=seeds)

    test_truth = truth[split:]
    p_sel = test_p1(feature_sets[best["features"]], HALFLIVES[best["recency"]], SEEDS)
    p_base = test_p1(ALL22, None, [config.SEED])                       # M10-base: all22, flat, 1 seed, thr 0.5
    p_sel_1seed = test_p1(feature_sets[best["features"]], HALFLIVES[best["recency"]], [config.SEED])

    pos = {
        "m10_sel": np.where(p_sel.to_numpy() >= best["thr"], 1.0, -1.0),
        "m10_base": np.where(p_base.to_numpy() >= 0.5, 1.0, -1.0),
        "m5": np.sign(mv["agent_size"].to_numpy()[split:]),
        "m8": np.sign(mv["final_size"].to_numpy()[split:]),
        "bh": np.ones(n - split),
    }
    acc = {k: round(float((v == test_truth).mean()), 4) for k, v in pos.items()}
    acc["m10_sel_1seed"] = round(acc_at(p_sel_1seed, best["thr"], test_truth), 4)

    corr = {k: (v == test_truth).astype(int) for k, v in pos.items()}
    tests = {}
    for opp in ("m5", "m8", "bh"):
        _, p_mc, b, c = mcnemar_test(corr[opp], corr["m10_sel"])
        _, p_bp = block_permutation_test(corr["m10_sel"], corr[opp], seed=config.SEED)
        tests[f"vs_{opp}"] = {"mcnemar_p": round(float(p_mc), 4), "block_perm_p": round(float(p_bp), 4),
                              "b_opp": int(b), "c_m10": int(c)}
    k_s, n_s, p_s, ci_s = sign_test(corr["m10_sel"])
    tests["vs_azar"] = {"k": int(k_s), "n": int(n_s), "p": round(float(p_s), 4),
                        "ci95": [round(float(ci_s[0]), 4), round(float(ci_s[1]), 4)]}
    # B4: Holm cubre SOLO los 3 McNemar (familia "bate a competidores"). El sign-test vs 0.5 es un sanity
    # ORTOGONAL (null distinto: ¿mejor que moneda?), NO confirmatorio → fuera de la familia Holm.
    holm = wf._holm_bonferroni({f"vs_{o}": tests[f"vs_{o}"]["mcnemar_p"] for o in ("m5", "m8", "bh")}, alpha=0.10)

    frac_up_val = round(float((truth[:split] == 1).mean()), 4)
    frac_up_test = round(float((test_truth == 1).mean()), 4)
    # A2: cobertura de M5/M8 en test (frac de días con apuesta direccional; M10/B&H apuestan siempre).
    cov = {"m5": round(float((np.sign(mv["agent_size"].to_numpy()[split:]) != 0).mean()), 3),
           "m8": round(float((np.sign(mv["final_size"].to_numpy()[split:]) != 0).mean()), 3)}
    n_combos = len(grid) * len(THR_GRID)

    result = {
        "meta": {"ticker": TICKER, "seed": config.SEED, "signal_lag": 1, "n_valid": int(n),
                 "n_val": int(split), "n_test": int(n - split), "N0": N0, "step": STEP, "embargo": EMBARGO,
                 "n_seeds": N_SEEDS, "thr_grid": THR_GRID, "halflives": list(HALFLIVES),
                 "test_span": [str(mv.index[split].date()), str(mv.index[-1].date())],
                 "frac_up_val": frac_up_val, "frac_up_test": frac_up_test, "cobertura_test_m5_m8": cov,
                 "split_efectivo": int(split), "n_combinaciones_grid": int(n_combos),
                 "acc_val_es_maximo_sobre_grid": True,
                 "nota_b3": f"acc_val del ganador = MÁXIMO sobre {n_combos} combinaciones (5 features × 3 recencias "
                            f"× 11 umbrales) en ~{split - N0} días → optimista, NO estimador insesgado; el test es la lectura honesta.",
                 "nota_b4": "Holm cubre los 3 McNemar (familia bate-competidores); sign-test vs 0.5 = sanity ortogonal, fuera de Holm.",
                 "nota_a2": "M10/B&H apuestan el 100% de los días; M5/M8 pueden ser neutrales (size=0 cuenta como fallo). n_obs común → comparación pareada; ver cobertura_test_m5_m8.",
                 "nota_m8": "M8 = override-C canónico (signo de régimen hardcoded Calma→long/Crisis→short). En SMCI (leverage débil) ese signo puede estar invertido (ver m8_datadriven_sign); se compara contra el M8 tal cual está desplegado.",
                 "scheme": "split 60/40 alineado a STEP; palancas elegidas SOLO en validación (WF interno); test WF pasado-expandible, una vez",
                 "pre_registro": "BITACORA 2026-06-16 (enmiendas B1–B5 aplicadas)"},
        "grid_validacion": sorted(grid, key=lambda g: -g["acc_val"]),
        "config_elegida": best, "config_base": base,
        "test": {"accuracy": acc, "tests": tests, "holm_3": holm,
                 "bate_m5_m8_bh": bool(acc["m10_sel"] > max(acc["m5"], acc["m8"], acc["bh"])),
                 "mejora_sobre_base": round(acc["m10_sel"] - acc["m10_base"], 4)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"VALIDACIÓN ({split - N0} días eval) — top configs por acc_val:")
    for g in result["grid_validacion"][:6]:
        print(f"   {g['features']:14} {g['recency']:5} thr={g['thr']} acc_val={g['acc_val']} (thr0.5={g['acc_val_thr05']})")
    print(f"\nCONFIG ELEGIDA: {best['features']} / {best['recency']} / thr={best['thr']} (acc_val={best['acc_val']})")
    print(f"\nTEST ({n - split} días, {result['meta']['test_span'][0]}→{result['meta']['test_span'][1]}, up={frac_up_test}):")
    print(f"   M10-sel={acc['m10_sel']}  M10-base={acc['m10_base']}  M10-sel(1seed)={acc['m10_sel_1seed']}")
    print(f"   M5={acc['m5']}  M8={acc['m8']}  B&H={acc['bh']}")
    print(f"   bate a todo: {result['test']['bate_m5_m8_bh']}  · mejora sobre base: {result['test']['mejora_sobre_base']:+}")
    print(f"   McNemar vs B&H p={tests['vs_bh']['mcnemar_p']} (Holm rej={holm['vs_bh']['reject']})  "
          f"vs M5 p={tests['vs_m5']['mcnemar_p']}  sign vs 0.5 p={tests['vs_azar']['p']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
