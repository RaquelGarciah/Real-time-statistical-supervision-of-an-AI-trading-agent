"""Clustering por naturaleza SOBRE LOS 10 del cuerpo + análisis por grupo (acc por estrategia, drift, cuota SHAP).

Raquel se quedó con 10 activos como cohorte del marco práctico, así que el clustering se re-ejecuta SOLO con esos
10 (la versión de 15 se conserva en strategy_clustering15.json). Mismas 5 features de naturaleza y mismos métodos
(kmeans/ward/gmm/spectral) que el de 15; n=10 → EXPLORATORIO, no confirmatorio.

Tres piezas por grupo (réplica de exploracion_estrategias, una por grupo):
  1. accuracy MEDIA por estrategia por grupo (¿qué canal gana en cada naturaleza?).
  2. accuracy de cada estrategia según COINCIDA con el drift (tendencia 21d), por activo del grupo.
  3. cuota SHAP de las features de STRATA por activo del grupo.

Posiciones ±1 reconstruidas del acierto canónico (panel_mm25); drift = signo de la tendencia 21d (causal);
cuota SHAP de decision_automl_prep. Determinista. Uso: python experiments/cluster_panel10.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config
import experiments.walkforward_robustez as wf
from config import STRATA_OOS_START

PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
FEATS = ["leverage_corr", "crisis_mean", "oos_crisis_frac", "oos_vol", "agent_short_frac"]
ARMS = ["m5", "m8", "m10_xgb", "automl", "zeror", "bh"]
NAME = {"m5": "M5", "m8": "M8", "m10_xgb": "M10", "automl": "AutoML", "zeror": "ZeroR", "bh": "B&H"}
MAIN = ["m5", "m8", "m10_xgb", "automl"]
PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
OUT = Path("outputs/experiments/cluster_panel10.json")


def _trend_sign(tk: str, dates: list) -> np.ndarray:
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    tr = oos.rolling(21, min_periods=5).mean().shift(1).reindex(pd.to_datetime(dates))
    return np.sign(tr.to_numpy())


def _truth(tk: str, dates: list) -> np.ndarray:
    _, ret = wf.load_features(tk)
    oos = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
    return oos.shift(-1).reindex(pd.to_datetime(dates)).to_numpy()


def main() -> None:
    config.set_seeds(config.SEED)
    nat15 = json.load(open("outputs/experiments/strategy_clustering15.json"))["por_activo"]
    pan = json.load(open(PANEL_FILE))["por_activo"]
    dp = json.load(open("outputs/experiments/decision_automl_prep.json"))["por_activo"]
    anr = json.load(open("outputs/experiments/automl_net_returns.json"))["por_activo"]

    # --- naturaleza (5 features) y estandarización sobre los 10 ---
    X = np.array([[nat15[a]["nat"][f] for f in FEATS] for a in PANEL10])
    Xs = StandardScaler().fit_transform(X)

    # --- clustering: kmeans/ward/gmm/spectral, k=2,3,4 ---
    clustering = {}
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit_predict(Xs)
        wd = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        gm = GaussianMixture(n_components=k, random_state=config.SEED, n_init=5).fit(Xs)
        gml = gm.predict(Xs)
        sp = SpectralClustering(n_clusters=k, random_state=config.SEED, affinity="rbf",
                                assign_labels="discretize").fit_predict(Xs)
        clustering[f"k{k}"] = {
            "kmeans": {"labels": km.tolist(), "silhouette": round(float(silhouette_score(Xs, km)), 4)},
            "ward": {"labels": wd.tolist(), "silhouette": round(float(silhouette_score(Xs, wd)), 4)},
            "gmm": {"labels": gml.tolist(), "silhouette": round(float(silhouette_score(Xs, gml)), 4),
                    "bic": round(float(gm.bic(Xs)), 2)},
            "spectral": {"labels": sp.tolist(), "silhouette": round(float(silhouette_score(Xs, sp)), 4)}}

    lab3 = np.array(clustering["k3"]["kmeans"]["labels"])
    concord = {"kmeans~ward": round(float(adjusted_rand_score(lab3, clustering["k3"]["ward"]["labels"])), 4),
               "kmeans~gmm": round(float(adjusted_rand_score(lab3, clustering["k3"]["gmm"]["labels"])), 4),
               "kmeans~spectral": round(float(adjusted_rand_score(lab3, clustering["k3"]["spectral"]["labels"])), 4)}

    pca = PCA(n_components=2).fit_transform(Xs)

    # --- acc/sharpe por estrategia por activo + drift-coincidencia + cuota SHAP ---
    per_asset = {}
    for tk in PANEL10:
        t = pan[tk]["table"]; cba = pan[tk]["correct_by_arm"]
        dates = anr[tk]["dates"]; rnext = _truth(tk, dates); absr = np.abs(rnext); dr = _trend_sign(tk, dates)
        valid = ~np.isnan(rnext) & (np.sign(rnext) != 0)
        acc = {NAME[a]: round(float(t[a]["accuracy"]), 4) for a in ARMS}
        shp = {NAME[a]: round(float(t[a]["sharpe"]), 3) for a in ARMS}
        drift = {}
        for a in MAIN:
            c = np.asarray(cba[a], float); pos = np.sign(rnext * (2 * c - 1))
            coin = valid & ~np.isnan(dr) & (pos == dr); cont = valid & ~np.isnan(dr) & (pos == -dr)
            drift[NAME[a]] = {"acc_coincide": round(float(c[coin].mean()), 4) if coin.sum() else None,
                              "acc_contra": round(float(c[cont].mean()), 4) if cont.sum() else None,
                              "n_coincide": int(coin.sum()), "n_contra": int(cont.sum())}
        per_asset[tk] = {"acc": acc, "sharpe": shp, "drift": drift,
                         "shap_cuota_strata": round(float(dp[tk]["shap"]["cuota_strata"]), 4)}

    # --- perfiles por grupo (consenso k=3) ---
    perfiles = {}
    for c in sorted(set(lab3)):
        acts = [PANEL10[i] for i in range(len(PANEL10)) if lab3[i] == c]
        nat_med = {f: round(float(np.mean([nat15[a]["nat"][f] for a in acts])), 4) for f in FEATS}
        acc_med = {NAME[a]: round(float(np.mean([per_asset[x]["acc"][NAME[a]] for x in acts])), 4) for a in ARMS}
        shp_med = {NAME[a]: round(float(np.mean([per_asset[x]["sharpe"][NAME[a]] for x in acts])), 3) for a in ARMS}
        no_triv = [NAME[a] for a in ARMS if a not in ("zeror", "bh")]
        perfiles[f"C{c}"] = {"activos": acts, "naturaleza_media": nat_med, "acc_media": acc_med,
                             "sharpe_media": shp_med,
                             "mejor_acc_no_trivial": max(no_triv, key=lambda s: acc_med[s]),
                             "mejor_sharpe_no_trivial": max(no_triv, key=lambda s: shp_med[s])}

    res = {"meta": {"panel": PANEL10, "n_activos": 10, "cluster_features": FEATS, "seed": config.SEED,
                    "estrategias": [NAME[a] for a in ARMS], "ventana": "desplegable (panel_mm25)",
                    "X_estandarizada": [[round(float(v), 4) for v in row] for row in Xs],
                    "pca2d": [[round(float(v), 4) for v in row] for row in pca],
                    "aviso": "n=10 → EXPLORATORIO/descriptivo, no confirmatorio. Versión de 15 conservada en "
                             "strategy_clustering15.json.",
                    "drift": "coincide = signo de la posición ±1 == signo de la tendencia 21d (causal)."},
           "clustering": clustering, "concordancia_k3_randajustado": concord,
           "perfiles_k3": {"kmeans": perfiles}, "por_activo": per_asset}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("=== Clustering de los 10 (k=3, kmeans) ===")
    for c, d in perfiles.items():
        print(f"  {c}: {d['activos']}")
        print(f"     leverage={d['naturaleza_media']['leverage_corr']:+.3f} vol={d['naturaleza_media']['oos_vol']:.2f} "
              f"agente_corto={d['naturaleza_media']['agent_short_frac']:.2f} → mejor acc={d['mejor_acc_no_trivial']} "
              f"Sharpe={d['mejor_sharpe_no_trivial']}")
    print(f"\nSilhouette k3: kmeans={clustering['k3']['kmeans']['silhouette']} ward={clustering['k3']['ward']['silhouette']} "
          f"gmm={clustering['k3']['gmm']['silhouette']} spectral={clustering['k3']['spectral']['silhouette']}")
    print(f"Concordancia (Rand) k3: {concord}")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
