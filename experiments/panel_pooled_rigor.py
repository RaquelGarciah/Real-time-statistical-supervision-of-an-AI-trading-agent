"""¿Rescata STRATA (M8) al agente (M5) de forma significativa al AGRUPAR el panel?

Motivación. Por activo, McNemar M8 vs M5 está infrapotenciado (N≈250-400, casi todos no
significativos). Pero la hipótesis del TFG es global ("STRATA rescata al agente cuando éste
pierde"), así que el contraste correcto agrupa los 10 activos. M8 es una REGLA FIJA sin
entrenamiento → no necesita burn-in: se usa TODO el OOS (más potencia que la ventana [150:]
que solo hacía falta para el XGBoost de M10).

Tres niveles de rigor, de menos a más conservador:
  1. McNemar pooled (suma de pares discordantes). Asume días independientes.
  2. Combinación de Fisher de los 10 McNemar por activo (cada activo = estudio independiente).
  3. Sign-flip CLUSTERIZADO por fecha (Canay-Romano-Shaikh 2017): robusto a CUALQUIER
     correlación entre activos en la MISMA fecha — responde a la objeción "te has inflado el N
     porque el mismo día todos los activos se mueven juntos". La fecha es la unidad independiente.

Datos: outputs_canonicos/decision_level/<TK>_panel.csv (size_M5, size_M8, r_next por día).

Uso: python experiments/panel_pooled_rigor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

import config

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA"]
DL = Path("_archivo_proyecto_anterior/outputs_canonicos/decision_level")
OUT = Path("outputs/experiments/panel_pooled_rigor.json")
N_PERM = 100_000


def load_asset(tk: str) -> pd.DataFrame:
    """Día-a-día: dirección M5, M8 y verdad. Descarta días con r_next=0 o NaN (no informan)."""
    df = pd.read_csv(DL / f"{tk}_panel.csv", index_col=0, parse_dates=True)
    truth = np.sign(df["r_next"].to_numpy())
    keep = np.isfinite(truth) & (truth != 0)
    out = pd.DataFrame({
        "m5_ok": (np.sign(df["size_M5"].to_numpy()) == truth).astype(int),
        "m8_ok": (np.sign(df["size_M8"].to_numpy()) == truth).astype(int),
    }, index=df.index)[keep]
    return out


def main() -> None:
    config.set_seeds(config.SEED)
    rng = np.random.default_rng(config.SEED)

    per_asset, frames = {}, []
    for tk in PANEL:
        d = load_asset(tk)
        d["tk"] = tk
        frames.append(d)
        b = int(((d["m5_ok"] == 1) & (d["m8_ok"] == 0)).sum())   # M8 rompe
        c = int(((d["m5_ok"] == 0) & (d["m8_ok"] == 1)).sum())   # M8 arregla
        n = len(d)
        p_two = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        p_one = binomtest(c, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        per_asset[tk] = {"n": n, "acc_m5": round(d["m5_ok"].mean(), 4), "acc_m8": round(d["m8_ok"].mean(), 4),
                         "b_rompe": b, "c_arregla": c, "mcnemar_p2": round(float(p_two), 4),
                         "mcnemar_p1_mejora": round(float(p_one), 4)}

    big = pd.concat(frames)
    N = len(big)
    B = int(((big["m5_ok"] == 1) & (big["m8_ok"] == 0)).sum())
    C = int(((big["m5_ok"] == 0) & (big["m8_ok"] == 1)).sum())
    acc_m5, acc_m8 = float(big["m5_ok"].mean()), float(big["m8_ok"].mean())

    # --- Nivel 1: McNemar pooled (exacto e independiente) ---
    p_pool_2 = float(binomtest(min(B, C), B + C, 0.5).pvalue)
    p_pool_1 = float(binomtest(C, B + C, 0.5, alternative="greater").pvalue)

    # --- Nivel 2: Fisher sobre los 10 p-valores 1-cola por activo ---
    pvec = np.array([per_asset[tk]["mcnemar_p1_mejora"] for tk in PANEL])
    pvec = np.clip(pvec, 1e-12, 1.0)
    fisher_stat = float(-2.0 * np.log(pvec).sum())
    p_fisher = float(chi2.sf(fisher_stat, df=2 * len(pvec)))

    # --- Nivel 3: sign-flip clusterizado por FECHA ---
    # contribución diaria por par discordante: +1 si M8 arregla, -1 si M8 rompe, 0 concordante
    big2 = big.assign(contrib=big["m8_ok"].to_numpy() - big["m5_ok"].to_numpy(),
                      tk=big["tk"])  # contrib en {-1,0,1}: +1 M8 arregla, -1 rompe
    net = big2.groupby(level=0)["contrib"].sum().to_numpy().astype(float)  # neto por fecha
    T_obs = float(net.sum())  # = C - B
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, net.size))
    T_null = signs @ net
    p_cluster = float((np.abs(T_null) >= abs(T_obs) - 1e-9).mean())          # 2-colas
    p_cluster_1 = float((T_null >= T_obs - 1e-9).mean())                      # 1-cola (hipótesis direccional)
    n_dates = int(net.size)

    # Leave-one-asset-out: ¿lo sostiene un solo activo? (re-clusteriza por fecha sin cada activo)
    loo = {}
    for drop in PANEL:
        sub = big2[big2["tk"] != drop]
        nd = sub.groupby(level=0)["contrib"].sum().to_numpy().astype(float)
        sg = rng.choice([-1.0, 1.0], size=(N_PERM, nd.size))
        loo[drop] = round(float((np.abs(sg @ nd) >= abs(nd.sum()) - 1e-9).mean()), 4)

    res = {
        "meta": {"panel": PANEL, "fuente": str(DL), "ventana": "OOS completo (M8 es regla fija, sin burn-in)",
                 "N_ticker_dias": N, "n_fechas": n_dates, "n_perm": N_PERM, "seed": config.SEED},
        "pooled": {
            "acc_m5": round(acc_m5, 4), "acc_m8": round(acc_m8, 4), "delta_acc": round(acc_m8 - acc_m5, 4),
            "b_rompe": B, "c_arregla": C, "discordantes": B + C,
            "nivel1_mcnemar_pooled_p2": round(p_pool_2, 5), "nivel1_p1_mejora": round(p_pool_1, 5),
            "nivel2_fisher_stat": round(fisher_stat, 3), "nivel2_fisher_p": round(p_fisher, 5),
            "nivel3_cluster_fecha_Tobs": T_obs, "nivel3_cluster_fecha_p2": round(p_cluster, 5),
            "nivel3_cluster_fecha_p1_direccional": round(p_cluster_1, 5),
        },
        "leave_one_out_cluster_p2": loo,
        "por_activo": per_asset,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"N={N} ticker-días sobre {n_dates} fechas | acc M5={acc_m5:.4f} → M8={acc_m8:.4f} (Δ={acc_m8-acc_m5:+.4f})")
    print(f"Discordantes: M8 rompe={B}  arregla={C}")
    print(f"  Nivel 1  McNemar pooled        : p2={p_pool_2:.5f}  p1(mejora)={p_pool_1:.5f}")
    print(f"  Nivel 2  Fisher (10 activos)    : χ²={fisher_stat:.2f}  p={p_fisher:.5f}")
    print(f"  Nivel 3  sign-flip por FECHA     : T={T_obs:.0f}  p2={p_cluster:.5f}  p1(direccional)={p_cluster_1:.5f}   <-- robusto a correlación entre activos")
    print("  Leave-one-out (cluster p2): " + "  ".join(f"sin {k}={v}" for k, v in loo.items()))
    print(f"OK · {OUT}")


if __name__ == "__main__":
    main()
