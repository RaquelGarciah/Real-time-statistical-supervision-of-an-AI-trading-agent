"""Tabla completa de estrategias (ventana 250) + clustering de activos + regla candidata.

Para los 12 activos con caché de agente completa: calcula sobre la MISMA ventana de evaluación
(la del walk-forward de M10, ~250 d tras burn-in) la accuracy y el Sharpe de las 6 estrategias
(M5, M8, M10, régimen, B&H, ZeroR), más features de NATURALEZA del activo (pre-especificadas,
ex-ante donde es posible: leverage de Black, media del régimen Crisis, fracción de Crisis OOS,
vol media, sesgo corto del agente). Luego agrupa los activos por su naturaleza (KMeans, k elegido
por silueta) y caracteriza cada grupo por qué estrategia funciona → regla candidata.

HONESTIDAD: n=12 → el clustering es EXPLORATORIO/descriptivo, no confirmatorio; la regla es una
HIPÓTESIS pre-registrable para test futuro, no un resultado probado. Features pre-especificadas
para evitar el dragado de datos (garden of forking paths). Exploratorio (docs/).

Uso: python experiments/strategy_clustering.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from config import CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import sharpe
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states, wf_p1, ALL22

PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA"]
REGNAMES = ["Calma", "Estrés", "Crisis"]
N0 = 150
CLASE = {"SPY": "índice", "QQQ": "índice", "DIA": "índice", "XLE": "ETF sect.", "XLF": "ETF sect.",
         "XLK": "ETF sect.", "UNG": "ETF commod.", "NVDA": "acción", "BAC": "acción", "TSLA": "acción",
         "MSTR": "cripto-px", "SMCI": "acción", "ROKU": "acción", "MARA": "cripto-px", "IWM": "índice"}
OUT = Path("outputs/experiments/strategy_clustering.json")
FIG = Path("outputs/experiments/strategy_clusters.png")
# Features de NATURALEZA para agrupar (ex-ante/estructurales; pre-especificadas):
CLUS_FEATS = ["leverage_corr", "crisis_mean", "oos_crisis_frac", "oos_vol", "agent_short_frac"]


def _row(tk: str, lev: dict) -> dict:
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = (mv["r_next"] > 0).astype(int)
    p1 = wf_p1(mv[ALL22], y)
    sub = mv.index[p1.notna().to_numpy()]
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    sign_prior = {k: float(np.sign(lev["media_regimen"][nm])) for k, nm in enumerate(REGNAMES)}
    dom = mv.loc[sub, "regime_dom"].to_numpy().astype(int)

    pos = {"M5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
           "M8": np.sign(mv.loc[sub, "final_size"].to_numpy()),
           "M10": np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0),
           "Régimen": np.array([sign_prior[d] for d in dom]),
           "B&H": np.ones_like(truth)}
    frac_up = float((truth > 0).mean()); maj = 1.0 if frac_up >= 0.5 else -1.0
    pos["ZeroR"] = np.full_like(truth, maj)

    def nr(p):
        ws = pd.Series(0.0, index=mv.index); ws.loc[sub] = p
        return run_backtest(oos_ret, ws, signal_lag=1)["net_return"].reindex(sub)
    acc = {k: round(float((v == truth).mean()), 4) for k, v in pos.items()}
    shp = {k: round(float(sharpe(nr(v))), 4) for k, v in pos.items()}

    return {"clase": CLASE.get(tk, "?"), "n": int(len(sub)), "frac_up": round(frac_up, 4),
            "acc": acc, "sharpe": shp,
            "nat": {"leverage_corr": lev["leverage_corr"], "crisis_mean": lev["crisis_mean"],
                    "oos_crisis_frac": round(float((dom == 2).mean()), 4),
                    "oos_vol": round(float(mv.loc[sub, "garch_sigma"].mean()), 4),
                    "agent_short_frac": round(float((mv.loc[sub, "agent_size"] < 0).mean()), 4),
                    "regime_dir_acc": acc["Régimen"]}}


def main() -> None:
    wf.reset_thresholds_cache()
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    rows = {}
    for tk in PANEL:
        try:
            rows[tk] = _row(tk, lev[tk])
            r = rows[tk]
            print(f"{tk:5s} {r['clase']:10s} acc M5={r['acc']['M5']:.3f} M8={r['acc']['M8']:.3f} "
                  f"M10={r['acc']['M10']:.3f} Rég={r['acc']['Régimen']:.3f} ZeroR={r['acc']['ZeroR']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if t in rows]

    # --- matriz de features de naturaleza (estandarizada) ---
    X = np.array([[rows[t]["nat"][f] if f != "agent_short_frac" else rows[t]["nat"][f] for f in CLUS_FEATS] for t in ok])
    Xs = StandardScaler().fit_transform(X)
    # elegir k por silueta (k=2..4)
    sil = {}
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit(Xs)
        sil[k] = float(silhouette_score(Xs, km.labels_))
    k_best = max(sil, key=sil.get)
    km = KMeans(n_clusters=k_best, random_state=config.SEED, n_init=10).fit(Xs)
    labels = km.labels_

    clusters = {}
    for c in range(k_best):
        members = [ok[i] for i in range(len(ok)) if labels[i] == c]
        # naturaleza media + estrategia ganadora media del grupo
        nat_mean = {f: round(float(np.mean([rows[t]["nat"][f] for t in members])), 4) for f in CLUS_FEATS}
        acc_mean = {s: round(float(np.mean([rows[t]["acc"][s] for t in members])), 4)
                    for s in ("M5", "M8", "M10", "Régimen", "B&H", "ZeroR")}
        shp_mean = {s: round(float(np.mean([rows[t]["sharpe"][s] for t in members])), 4)
                    for s in ("M5", "M8", "M10", "Régimen", "B&H", "ZeroR")}
        # mejor estrategia NO trivial (excluye B&H/ZeroR) por accuracy y por Sharpe
        no_triv = ("M5", "M8", "M10", "Régimen")
        best_acc = max(no_triv, key=lambda s: acc_mean[s]); best_shp = max(no_triv, key=lambda s: shp_mean[s])
        clusters[f"C{c}"] = {"activos": members, "naturaleza_media": nat_mean,
                             "acc_media": acc_mean, "sharpe_media": shp_mean,
                             "mejor_acc_no_trivial": best_acc, "mejor_sharpe_no_trivial": best_shp,
                             "delta_M8_M5_acc": round(acc_mean["M8"] - acc_mean["M5"], 4),
                             "delta_M10_M8_acc": round(acc_mean["M10"] - acc_mean["M8"], 4)}

    # figura PCA 2D
    pca = PCA(n_components=2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in range(k_best):
        mask = labels == c
        ax.scatter(pca[mask, 0], pca[mask, 1], s=120, label=f"C{c}")
    for i, t in enumerate(ok):
        ax.annotate(t, (pca[i, 0], pca[i, 1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Clustering de activos por naturaleza (k={k_best}, silueta={sil[k_best]:.2f})")
    ax.legend(); plt.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110, bbox_inches="tight"); plt.close(fig)

    res = {"meta": {"panel": ok, "n_activos": len(ok), "ventana": "M10 walk-forward (~250 d tras burn-in 150)",
                    "cluster_features": CLUS_FEATS, "k_silueta": sil, "k_elegido": k_best, "seed": config.SEED,
                    "aviso": "n=12 → clustering EXPLORATORIO/descriptivo, no confirmatorio; la regla es HIPÓTESIS "
                             "pre-registrable, no probada. Features pre-especificadas (anti-dragado). Exploratorio (docs/)."},
           "por_activo": rows, "clusters": clusters}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n=== CLUSTERING (k={k_best}, silueta={sil[k_best]:.2f}; siluetas {sil}) ===")
    for c, d in clusters.items():
        print(f"\n{c}: {d['activos']}")
        nm = d["naturaleza_media"]
        print(f"   naturaleza: leverage={nm['leverage_corr']:+.3f} crisis_mean={nm['crisis_mean']:+.5f} "
              f"crisisOOS={nm['oos_crisis_frac']:.2f} vol={nm['oos_vol']:.2f} agente_corto={nm['agent_short_frac']:.2f}")
        print(f"   acc media: " + " ".join(f"{s}={d['acc_media'][s]:.3f}" for s in ("M5","M8","M10","Régimen","ZeroR")))
        print(f"   → mejor no-trivial: acc={d['mejor_acc_no_trivial']} sharpe={d['mejor_sharpe_no_trivial']} "
              f"· ΔM8-M5={d['delta_M8_M5_acc']:+.3f} ΔM10-M8={d['delta_M10_M8_acc']:+.3f}")
    print(f"\nfigura: {FIG}\nOK · {OUT}")


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
