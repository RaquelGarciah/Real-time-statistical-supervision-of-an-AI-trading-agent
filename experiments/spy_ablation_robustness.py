"""Robustez de la ablación SPY frente al azar de las semillas.

La ablación (spy_momentum_ablation) usó UN ensemble de 10 semillas. Aquí se repite con 5 bloques de
semillas DISJUNTOS (42-51, 52-61, ...) para ver si las dos conclusiones aguantan o fueron un sorteo
afortunado:
  C1: STRATA+régimen añade accuracy direccional sobre el momentum puro (strata7+mom > momentum_solo).
  C2: quitar las 15 features del agente mejora (strata7+mom >= all22+mom).
Mismo walk-forward causal (embargo 5, signal_lag=1, todo el OOS = test). Reporta la distribución de
accuracy/Sharpe por config y el McNemar del incremento STRATA-sobre-momentum en cada bloque.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from core.backtest import run_backtest
from core.stats import mcnemar_test, sign_test
import experiments.walkforward_robustez as wf
from experiments.m10_v3_causal_panel import build_states_onthefly
from experiments.m10_improve_smci import REALSIG, build_realsignal
from experiments.m10_pivot_scan import wf_p1_full, _sr, N0
from experiments.m10_valtest_casestudy import ALL22, STRATA_REGIME7

TICKER = "SPY"
N_BLOCKS, SEEDS_PER_BLOCK = 5, 10
SEED_BLOCKS = [[config.SEED + b * SEEDS_PER_BLOCK + i for i in range(SEEDS_PER_BLOCK)] for b in range(N_BLOCKS)]
CONFIGS = {"momentum_solo": REALSIG, "strata7+mom": STRATA_REGIME7 + REALSIG, "all22+mom": ALL22 + REALSIG}
OUT = Path("outputs/experiments/spy_ablation_robustness.json")


def main() -> None:
    config.set_seeds(config.SEED)
    wf.reset_thresholds_cache()
    gamma, sigma, oos_ret = build_states_onthefly(TICKER)
    _, ret_full = wf.load_features(TICKER)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid].copy()
    mv[REALSIG] = build_realsignal(ret_full, mv.index)
    mv = mv.dropna(subset=REALSIG)
    y = (mv["r_next"] > 0).astype(int)
    idx = mv.index[N0:]
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy())
    bh_acc = float((np.ones(len(idx)) == truth).mean())

    per_block = []
    for b, seeds in enumerate(SEED_BLOCKS):
        row = {"block": b, "seeds": [seeds[0], seeds[-1]]}
        corr = {}
        for name, cols in CONFIGS.items():
            p = wf_p1_full(mv[cols], y, None, seeds).loc[idx]
            pos = np.where(p.to_numpy() >= 0.5, 1.0, -1.0)
            corr[name] = (pos == truth).astype(int)
            w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
            nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()
            _, _, ps, _ = sign_test(corr[name])
            row[name] = {"acc": round(float(corr[name].mean()), 4), "sharpe": round(_sr(nr), 3),
                         "sign_p": round(float(ps), 4)}
        _, p_inc, b_mom, c_str = mcnemar_test(corr["momentum_solo"], corr["strata7+mom"])
        row["strata_sobre_mom"] = {"delta_acc": round(row["strata7+mom"]["acc"] - row["momentum_solo"]["acc"], 4),
                                   "mcnemar_p": round(float(p_inc), 4),
                                   "dias_solo_mom": int(b_mom), "dias_strata": int(c_str)}
        row["quitar_agente_mejora"] = bool(row["strata7+mom"]["sharpe"] >= row["all22+mom"]["sharpe"])
        per_block.append(row)
        print(f"bloque {b} (seeds {seeds[0]}-{seeds[-1]}): "
              f"mom={row['momentum_solo']['acc']}/SR{row['momentum_solo']['sharpe']} "
              f"str+mom={row['strata7+mom']['acc']}/SR{row['strata7+mom']['sharpe']} "
              f"all22+mom={row['all22+mom']['acc']}/SR{row['all22+mom']['sharpe']} | "
              f"Δacc={row['strata_sobre_mom']['delta_acc']:+} p={row['strata_sobre_mom']['mcnemar_p']} "
              f"quita_agente_mejora={row['quitar_agente_mejora']}")

    def stats_of(name, field):
        v = np.array([blk[name][field] for blk in per_block])
        return {"media": round(float(v.mean()), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    deltas = np.array([blk["strata_sobre_mom"]["delta_acc"] for blk in per_block])
    pvals = np.array([blk["strata_sobre_mom"]["mcnemar_p"] for blk in per_block])
    resumen = {
        "acc_momentum_solo": stats_of("momentum_solo", "acc"),
        "acc_strata7+mom": stats_of("strata7+mom", "acc"),
        "sharpe_strata7+mom": stats_of("strata7+mom", "sharpe"),
        "C1_strata_sobre_mom": {"delta_acc_media": round(float(deltas.mean()), 4),
                                "delta_acc_min": round(float(deltas.min()), 4),
                                "bloques_delta_positivo": int((deltas > 0).sum()),
                                "bloques_mcnemar_sig_0.10": int((pvals < 0.10).sum())},
        "C2_quitar_agente_mejora": int(sum(blk["quitar_agente_mejora"] for blk in per_block)),
        "n_bloques": N_BLOCKS,
    }
    out = {"meta": {"ticker": TICKER, "n_eval": int(len(idx)), "bh_acc": round(bh_acc, 4),
                    "n_blocks": N_BLOCKS, "seeds_per_block": SEEDS_PER_BLOCK,
                    "oos": [str(idx.min().date()), str(idx.max().date())]},
           "por_bloque": per_block, "resumen": resumen}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nC1 (STRATA>mom): Δacc media {resumen['C1_strata_sobre_mom']['delta_acc_media']:+}, "
          f"positivo en {resumen['C1_strata_sobre_mom']['bloques_delta_positivo']}/{N_BLOCKS}, "
          f"McNemar<0.10 en {resumen['C1_strata_sobre_mom']['bloques_mcnemar_sig_0.10']}/{N_BLOCKS}")
    print(f"C2 (quitar agente mejora Sharpe): {resumen['C2_quitar_agente_mejora']}/{N_BLOCKS}")
    print(f">>> {OUT}")


if __name__ == "__main__":
    main()
