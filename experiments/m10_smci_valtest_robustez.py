"""Robustez del resultado de SMCI a la partición validación/test.

Resultado PRINCIPAL = M10 desplegable (ens, embargo=1) sobre TODO el OOS (250 d): accuracy 0.552, bate a
M5/M8/B&H nominal. Este script lo RESPALDA mostrando que la conclusión "M10 gana a todo" es **invariante a la
partición**: con 3 splits cronológicos estándar pre-especificados (60/40, 70/30, 80/20; burn-in 150 fijo),
M10 bate a M5/M8/B&H **tanto en validación como en test** en los tres casos.

NO es split-shopping: (i) los ratios son estándar y se fijan a priori; (ii) la lectura es la CONSISTENCIA, no
quedarse con el split de mayor accuracy; (iii) el número headline sigue siendo el de todo el OOS. Se reporta
que al achicar el test la accuracy sube pero pierde potencia (sign p sube), para no sobrevender.

El p1 del walk-forward (ensemble 10 semillas, embargo=1, burn-in 150) se calcula UNA vez y se reparte en las
ventanas de cada split. Uso: python experiments/m10_smci_valtest_robustez.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import binomtest

import config
from core.stats import block_permutation_test, mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_valtest_casestudy import ALL22
import experiments.m10_smci_select as S   # wf_p1, metrics, ref_metrics, SEEDS, EMBARGO

TICKER = "SMCI"
N0 = 150                       # burn-in (entrenamiento, no se puntúa)
SPLITS = [0.6, 0.7, 0.8]       # fracción de los días puntuables que va a VALIDACIÓN (resto = test)
OUT = Path("outputs/experiments/m10_smci_valtest_robustez.json")


def _tests(p, mv, lo, hi):
    """Significancia en la ventana [lo:hi]: M10 vs B&H/M5 (McNemar+block-perm) y sign vs 0.5."""
    idx = mv.index[lo:hi]; pr = p.reindex(idx); ok = pr.notna().to_numpy(); idx = idx[ok]
    pos = np.where(pr.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    corr = (pos == truth).astype(int)
    bh = (np.ones(len(idx)) == truth).astype(int)
    m5 = (np.sign(mv.loc[idx, "agent_size"].to_numpy()) == truth).astype(int)
    _, p_bh, _, _ = mcnemar_test(bh, corr)
    _, p_bp = block_permutation_test(corr, bh, seed=config.SEED)
    _, p_m5, _, _ = mcnemar_test(m5, corr)
    _, _, p_az, _ = sign_test(corr)
    return {"mcnemar_vs_bh_p": round(float(p_bh), 4), "block_perm_vs_bh_p": round(float(p_bp), 4),
            "mcnemar_vs_m5_p": round(float(p_m5), 4), "sign_vs_0.5_p": round(float(p_az), 4),
            "frac_corto_m10": round(float((pos < 0).mean()), 3),
            "siempre_corto_acc": round(float((np.full(len(idx), -1.0) == truth).mean()), 4)}


def _window(p, mv, o, lo, hi):
    a, sr, eq, nn = S.metrics(p, mv, o, lo, hi)
    ref = S.ref_metrics(mv, o, lo, hi)
    frac_up = float((np.sign(mv["r_next"].to_numpy()[lo:hi]) > 0).mean())
    nir = max(frac_up, 1 - frac_up)                       # clase mayoritaria (ZeroR / no-information rate)
    # test binomial unilateral (Kuhn 2008): ¿accuracy de M10 > NIR? y referencia vs 0.5.
    k = int(round(a * nn))
    p_nir = float(binomtest(k, nn, nir, alternative="greater").pvalue) if nn > 0 else float("nan")
    p_05 = float(binomtest(k, nn, 0.5, alternative="greater").pvalue) if nn > 0 else float("nan")
    gana = bool(a > max(ref["m5"]["acc"], ref["m8"]["acc"], ref["bh"]["acc"], round(nir, 4)))
    return {"n": int(nn), "frac_up": round(frac_up, 3), "m10": {"acc": a, "sharpe": sr, "equity": eq},
            "m5": ref["m5"], "m8": ref["m8"], "bh": ref["bh"],
            "majority": {"acc": round(nir, 4), "dir": "corto" if frac_up < 0.5 else "largo"},
            "binom_m10_vs_nir_p": round(p_nir, 4), "binom_m10_vs_0.5_p": round(p_05, 4),
            "m10_gana_a_todo": gana}


def main() -> None:
    config.set_seeds(config.SEED); wf.reset_thresholds_cache()
    assert S.EMBARGO == 1, "este respaldo es para el modelo definitivo embargo=1"
    g, s, o = build_states_onthefly(TICKER)
    m = wf.run_master(g, s, o, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    n = len(mv); y = (mv["r_next"] > 0).astype(int); scored = n - N0
    p = S.wf_p1(mv[ALL22], y, N0, S.SEEDS)        # un solo walk-forward; las ventanas solo cambian dónde se corta

    full = _window(p, mv, o, N0, n)               # PRINCIPAL: todo el OOS
    splits = []
    for f in SPLITS:
        v_end = N0 + int(scored * f)
        val = _window(p, mv, o, N0, v_end)
        test = _window(p, mv, o, v_end, n); test["tests"] = _tests(p, mv, v_end, n)
        splits.append({"frac_val": f, "v_end_idx": int(v_end),
                       "val_span": [str(mv.index[N0].date()), str(mv.index[v_end - 1].date())],
                       "test_span": [str(mv.index[v_end].date()), str(mv.index[-1].date())],
                       "validacion": val, "test": test,
                       "m10_gana_ambas": bool(val["m10_gana_a_todo"] and test["m10_gana_a_todo"])})

    result = {"meta": {"ticker": TICKER, "embargo": S.EMBARGO, "n_seeds": len(S.SEEDS), "burn_in": N0,
                       "oos_n": int(n), "scored": int(scored), "splits_val_frac": SPLITS,
                       "modelo": "M10-WF ensemble (XGBoost 300x4, 10 semillas, 22 features STRATA, umbral 0.5)",
                       "nota": "PRINCIPAL = todo el OOS; los splits son RESPALDO de robustez (no split-shopping): ratios estándar a priori, lectura = consistencia",
                       "pre_registro": "BITACORA 2026-06-17"},
              "principal_todo_oos": full, "robustez_splits": splits,
              "m10_gana_a_todo_en_todos": bool(full["m10_gana_a_todo"] and all(s["m10_gana_ambas"] for s in splits))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"PRINCIPAL · todo el OOS (n={full['n']}): M10={full['m10']['acc']}  M5={full['m5']['acc']} "
          f"M8={full['m8']['acc']} B&H={full['bh']['acc']} MAYORÍA={full['majority']['acc']}({full['majority']['dir']})  "
          f"gana_a_todo={full['m10_gana_a_todo']}  | binom M10 vs NIR p={full['binom_m10_vs_nir_p']} (vs 0.5 p={full['binom_m10_vs_0.5_p']})")
    print(f"\n{'split':>7} {'VAL M10/B&H/MAY':>18} {'gana':>5} | {'TEST M10/B&H/MAY':>19} {'gana':>5} {'binom vs NIR':>13}")
    print("-" * 78)
    for sp in splits:
        v, t = sp["validacion"], sp["test"]
        print(f"{int(sp['frac_val']*100):>4}/{int((1-sp['frac_val'])*100):<2} "
              f"{v['m10']['acc']:>5}/{v['bh']['acc']}/{v['majority']['acc']:<6} {str(v['m10_gana_a_todo']):>5} | "
              f"{t['m10']['acc']:>5}/{t['bh']['acc']}/{t['majority']['acc']:<6} {str(t['m10_gana_a_todo']):>5} {t['binom_m10_vs_nir_p']:>13}")
    print(f"\nM10 gana a TODO en el OOS completo Y en validación y test de los 3 splits: {result['m10_gana_a_todo_en_todos']}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
