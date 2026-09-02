"""¿M10 gana de forma CONSISTENTE en SMCI o solo en un tramo? Rolling-window sobre todo el OOS desplegable.

M10 = ensemble (10 semillas, 22 features), walk-forward desplegable, evaluado en [N0:fin] (250 d). Se calcula
accuracy rodante (ventanas deslizantes de varios tamaños) de M10/M5/M8/B&H, la fracción de ventanas en que M10
bate a cada uno, y la serie de accuracy rodante para graficar dónde funciona y dónde no. Mide ESTABILIDAD
intra-OOS (no robustez inter-época: el agente solo existe en el OOS post-cutoff del LLM).

Uso: python experiments/m10_smci_rolling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from core.stats import block_permutation_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import N0, wf_p1
from experiments.m10_valtest_casestudy import ALL22

TICKER = "SMCI"
N_SEEDS = 10
WINDOWS = [42, 63, 84]                                # 2, 3, 4 meses de mercado
STEP_W = 5
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
OUT = Path("outputs/experiments/m10_smci_rolling.json")


def main() -> None:
    config.set_seeds(config.SEED); wf.reset_thresholds_cache()
    g, s, o = build_states_onthefly(TICKER)
    m = wf.run_master(g, s, o, wf.load_agent(TICKER))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    idx = mv.index[N0:]; sub = mv.loc[idx]
    truth = np.sign(sub["r_next"].to_numpy())

    p = wf_p1(mv[ALL22], (mv["r_next"] > 0).astype(int), len(mv), N0, None, SEEDS).reindex(idx)
    m10 = np.where(p.to_numpy() >= 0.5, 1.0, -1.0)
    corr = {"m10": (m10 == truth).astype(int),
            "m5": (np.sign(sub["agent_size"].to_numpy()) == truth).astype(int),
            "m8": (np.sign(sub["final_size"].to_numpy()) == truth).astype(int),
            "bh": (np.ones(len(idx)) == truth).astype(int)}
    acc_full = {k: round(float(v.mean()), 4) for k, v in corr.items()}

    # Fracción de ventanas en que M10 bate a cada brazo, por tamaño de ventana
    frac = {}
    for W in WINDOWS:
        wins = range(0, len(idx) - W + 1, STEP_W)
        gt = {opp: [] for opp in ("bh", "m5", "m8")}
        for s0 in wins:
            sl = slice(s0, s0 + W)
            a10 = corr["m10"][sl].mean()
            for opp in gt:
                gt[opp].append(a10 > corr[opp][sl].mean())
        frac[W] = {"n_ventanas": len(list(wins)),
                   **{f"m10_gt_{opp}": round(float(np.mean(v)), 3) for opp, v in gt.items()}}

    # Serie de accuracy rodante 63d (para el gráfico) + significancia global autocorr-robusta
    W = 63; xs, roll = [], {k: [] for k in corr}
    for s0 in range(0, len(idx) - W + 1, STEP_W):
        sl = slice(s0, s0 + W); xs.append(str(idx[s0 + W - 1].date()))
        for k in corr:
            roll[k].append(round(float(corr[k][sl].mean()), 4))
    _, p_bh = block_permutation_test(corr["m10"], corr["bh"], seed=config.SEED)
    _, p_m5 = block_permutation_test(corr["m10"], corr["m5"], seed=config.SEED)
    _, _, p_az, _ = sign_test(corr["m10"])

    result = {"meta": {"ticker": TICKER, "n_eval": int(len(idx)), "n_seeds": N_SEEDS, "windows": WINDOWS,
                       "step": STEP_W, "oos_span": [str(idx.min().date()), str(idx.max().date())],
                       "nota": "estabilidad intra-OOS (no robustez inter-época: agente solo existe en OOS)"},
              "accuracy_global": acc_full,
              "frac_ventanas_m10_gana": frac,
              "rolling63": {"fecha_fin": xs, **roll},
              "significancia_global": {"block_perm_vs_bh_p": round(float(p_bh), 4),
                                       "block_perm_vs_m5_p": round(float(p_m5), 4),
                                       "sign_vs_0.5_p": round(float(p_az), 4)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"SMCI ens · OOS [{result['meta']['oos_span'][0]}→{result['meta']['oos_span'][1]}] n={len(idx)}")
    print(f"accuracy global: M10={acc_full['m10']} M5={acc_full['m5']} M8={acc_full['m8']} B&H={acc_full['bh']}")
    print("Fracción de ventanas en que M10 GANA:")
    for W, d in frac.items():
        print(f"  W={W}d ({d['n_ventanas']} ventanas): vs B&H {d['m10_gt_bh']:.0%}  vs M5 {d['m10_gt_m5']:.0%}  vs M8 {d['m10_gt_m8']:.0%}")
    print(f"Significancia global (autocorr-robusta): vs B&H p={p_bh:.3f}  vs M5 p={p_m5:.3f}  sign vs 0.5 p={p_az:.3f}")
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
