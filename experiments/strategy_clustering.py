"""Agrupación de los 15 activos por su NATURALEZA y qué estrategia funciona mejor en cada grupo.

Reúne, por activo: features de naturaleza (ex-ante: leverage de Black y media del régimen Crisis, de
leverage_screen.json; OOS: fracción de Crisis, vol media, sesgo corto del agente) y la accuracy/Sharpe de
cada estrategia (M5/M8/M10/Régimen/ZeroR/B&H de decision_automl_prep.json + AutoML del panel mm25). Luego
prueba VARIOS algoritmos de agrupación adecuados a la naturaleza de los datos (n=15 pequeño, ~5 features
continuas y correladas) y muestra cómo separa cada uno — la elección final del algoritmo la decide Raquel.

  - KMeans (centroides, esférico)            - Agglomerative/Ward (jerárquico, distancias)
  - GaussianMixture (probabilístico, +BIC)   - Spectral (no convexo, k-NN)
  DBSCAN se descarta (con n=15 y sin densidad clara no forma clusters estables) — se comenta, no se usa.

Para cada método y k∈{2,3,4}: etiquetas + silhouette (+ BIC en GMM). Concordancia entre métodos (Rand
ajustado) a k=3. Perfil por cluster (naturaleza media + estrategia ganadora) a k=3 por método.

HONESTIDAD: n=15 → EXPLORATORIO/descriptivo, no confirmatorio; la "regla por grupo" es HIPÓTESIS
pre-registrable, no probada. Features pre-especificadas (anti-dragado). Uso: python experiments/strategy_clustering.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import config
from config import CALIBRATION_START
from core import data
from core.backtest import run_backtest
from core.metrics import sharpe
import experiments.walkforward_robustez as wf
from experiments.quant_validation_panel import build_states

PANEL = ["SPY", "QQQ", "DIA", "IWM", "XLE", "XLF", "XLK", "NVDA", "BAC", "TSLA",
         "MSTR", "SMCI", "ROKU", "MARA", "UNG"]
REGNAMES = ["Calma", "Estrés", "Crisis"]
N0 = 150
STRATS = ["M5", "M8", "M10", "Régimen", "AutoML", "ZeroR", "B&H"]
NO_TRIV = ["M5", "M8", "M10", "Régimen", "AutoML"]
CLUS_FEATS = ["leverage_corr", "crisis_mean", "oos_crisis_frac", "oos_vol", "agent_short_frac"]
CLASE = {"SPY": "índice", "QQQ": "índice", "DIA": "índice", "IWM": "índice", "XLE": "ETF sect.",
         "XLF": "ETF sect.", "XLK": "ETF sect.", "UNG": "ETF commod.", "NVDA": "acción", "BAC": "acción",
         "TSLA": "acción", "MSTR": "cripto-px", "SMCI": "acción", "ROKU": "acción", "MARA": "cripto-px"}
PREP = "outputs/experiments/decision_automl_prep.json"
PANEL_AUTOML = "outputs/experiments/automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json"
OUT = Path("outputs/experiments/strategy_clustering15.json")


def _nature_and_regime(tk: str, lev: dict) -> dict:
    """Naturaleza OOS (Crisis frac, vol, sesgo corto) + acc/Sharpe de la estrategia Régimen (sign prior)."""
    data.load_market_data(tk, CALIBRATION_START, datetime.date.today().isoformat())
    gamma, sigma, oos_ret = build_states(tk)
    m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    sub = mv.index[N0:]                                   # ventana desplegable (≡ M10)
    truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
    dom = mv.loc[sub, "regime_dom"].to_numpy().astype(int)
    sign_prior = {k: float(np.sign(lev["media_regimen"][nm])) for k, nm in enumerate(REGNAMES)}
    pos_reg = np.array([sign_prior.get(d, 1.0) for d in dom])
    w = pd.Series(0.0, index=m.index); w.loc[sub] = pos_reg
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub)
    return {"reg_acc": round(float((pos_reg == truth).mean()), 4), "reg_sharpe": round(float(sharpe(nr)), 4),
            "nat": {"leverage_corr": lev["leverage_corr"], "crisis_mean": lev["crisis_mean"],
                    "oos_crisis_frac": round(float((dom == 2).mean()), 4),
                    "oos_vol": round(float(mv.loc[sub, "garch_sigma"].mean()), 4),
                    "agent_short_frac": round(float((mv.loc[sub, "agent_size"] < 0).mean()), 4)}}


def _profiles(labels: np.ndarray, ok: list, rows: dict, k: int) -> dict:
    prof = {}
    for c in range(k):
        members = [ok[i] for i in range(len(ok)) if labels[i] == c]
        if not members:
            continue
        nat = {f: round(float(np.mean([rows[t]["nat"][f] for t in members])), 4) for f in CLUS_FEATS}
        accm = {s: round(float(np.mean([rows[t]["acc"][s] for t in members])), 4) for s in STRATS}
        shpm = {s: round(float(np.mean([rows[t]["sharpe"][s] for t in members])), 4) for s in STRATS}
        prof[f"C{c}"] = {"activos": members, "naturaleza_media": nat, "acc_media": accm, "sharpe_media": shpm,
                         "mejor_acc_no_trivial": max(NO_TRIV, key=lambda s: accm[s]),
                         "mejor_sharpe_no_trivial": max(NO_TRIV, key=lambda s: shpm[s])}
    return prof


def main() -> None:
    config.set_seeds(config.SEED)
    wf.reset_thresholds_cache()
    lev = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
    prep = json.load(open(PREP))["por_activo"]
    autx = json.load(open(PANEL_AUTOML))["por_activo"]

    rows = {}
    for tk in PANEL:
        try:
            nr = _nature_and_regime(tk, lev[tk])
            p = prep[tk]; a = autx[tk]["table"]
            acc = {"M5": p["accuracy"]["m5"], "M8": p["accuracy"]["m8"], "M10": p["accuracy"]["m10"],
                   "Régimen": nr["reg_acc"], "AutoML": a["automl"]["accuracy"],
                   "ZeroR": p["accuracy"]["zeror"], "B&H": p["accuracy"]["bh"]}
            shp = {"M5": p["sharpe"]["m5"], "M8": p["sharpe"]["m8"], "M10": p["sharpe"]["m10"],
                   "Régimen": nr["reg_sharpe"], "AutoML": a["automl"]["sharpe"],
                   "ZeroR": p["sharpe"]["zeror"], "B&H": p["sharpe"]["bh"]}
            rows[tk] = {"clase": CLASE.get(tk, "?"), "nat": nr["nat"], "acc": acc, "sharpe": shp}
            print(f"{tk:5s} {rows[tk]['clase']:10s} lev={nr['nat']['leverage_corr']:+.3f} "
                  f"acc M8={acc['M8']:.3f} M10={acc['M10']:.3f} AutoML={acc['AutoML']:.3f} ZeroR={acc['ZeroR']:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
    ok = [t for t in PANEL if t in rows]

    X = np.array([[rows[t]["nat"][f] for f in CLUS_FEATS] for t in ok])
    Xs = StandardScaler().fit_transform(X)

    # --- VARIOS algoritmos (no se elige automáticamente: lo decide Raquel) ---
    methods = {}
    for k in (2, 3, 4):
        algos = {}
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit(Xs)
        algos["kmeans"] = {"labels": km.labels_.tolist(), "silhouette": round(float(silhouette_score(Xs, km.labels_)), 4)}
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xs)
        algos["ward"] = {"labels": ag.labels_.tolist(), "silhouette": round(float(silhouette_score(Xs, ag.labels_)), 4)}
        gm = GaussianMixture(n_components=k, random_state=config.SEED, n_init=5).fit(Xs)
        gl = gm.predict(Xs)
        algos["gmm"] = {"labels": gl.tolist(), "silhouette": round(float(silhouette_score(Xs, gl)), 4),
                        "bic": round(float(gm.bic(Xs)), 2)}
        try:
            sp = SpectralClustering(n_clusters=k, random_state=config.SEED, affinity="nearest_neighbors",
                                    n_neighbors=5, assign_labels="kmeans").fit(Xs)
            algos["spectral"] = {"labels": sp.labels_.tolist(),
                                 "silhouette": round(float(silhouette_score(Xs, sp.labels_)), 4)}
        except Exception as e:  # noqa: BLE001
            algos["spectral"] = {"error": f"{type(e).__name__}: {e}"}
        methods[f"k{k}"] = algos

    # Concordancia entre métodos a k=3 (Rand ajustado)
    lab3 = {mth: methods["k3"][mth]["labels"] for mth in ("kmeans", "ward", "gmm", "spectral")
            if "labels" in methods["k3"][mth]}
    concord = {}
    mlist = list(lab3)
    for i in range(len(mlist)):
        for j in range(i + 1, len(mlist)):
            concord[f"{mlist[i]}~{mlist[j]}"] = round(float(adjusted_rand_score(lab3[mlist[i]], lab3[mlist[j]])), 3)

    # Perfiles por cluster a k=3, para cada método (naturaleza + estrategia ganadora)
    profiles_k3 = {mth: _profiles(np.array(lab3[mth]), ok, rows, 3) for mth in lab3}

    res = {"meta": {"panel": ok, "n_activos": len(ok), "cluster_features": CLUS_FEATS, "seed": config.SEED,
                    "ventana": "desplegable M10 (~250 d tras burn-in 150)",
                    "estrategias": STRATS, "X_estandarizada": Xs.round(4).tolist(),
                    "metodos": "kmeans, ward, gmm(+BIC), spectral; DBSCAN descartado (n=15 sin densidad)",
                    "decision": "la elección del algoritmo/k la decide Raquel; aquí se muestran todos.",
                    "aviso": "n=15 → EXPLORATORIO/descriptivo, no confirmatorio; regla = HIPÓTESIS pre-registrable."},
           "por_activo": rows, "clustering": methods, "concordancia_k3_randajustado": concord,
           "perfiles_k3": profiles_k3}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== SILHOUETTE por método y k (mayor = mejor separación) ===")
    for k in ("k2", "k3", "k4"):
        row = " ".join(f"{mth}={methods[k][mth].get('silhouette','—')}" for mth in ("kmeans", "ward", "gmm", "spectral"))
        print(f"  {k}: {row}")
    print(f"\nConcordancia k=3 (Rand ajustado): {concord}")
    print("\n=== Perfiles k=3 (KMeans) — naturaleza y mejor estrategia por grupo ===")
    for c, d in profiles_k3.get("kmeans", {}).items():
        print(f"  {c}: {d['activos']} → mejor acc={d['mejor_acc_no_trivial']} sharpe={d['mejor_sharpe_no_trivial']} "
              f"(lev={d['naturaleza_media']['leverage_corr']:+.3f} vol={d['naturaleza_media']['oos_vol']:.2f})")
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
