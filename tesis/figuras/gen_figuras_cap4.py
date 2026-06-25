# -*- coding: utf-8 -*-
"""Generador de las 15 figuras del Capítulo 4 (marco práctico, panel canónico-10).

Reproduce los plots del notebook builder (`notebooks/_build_STRATA_marco_practico.py`)
y los guarda como PDF vectoriales en `tesis/figuras/cap4_*.pdf`, sin título embebido
(el caption va en LaTeX), con ejes/leyendas en español y coma decimal.

Fuente de verdad de cifras: `tesis/figuras/REQUERIDAS_cap4.md`. Cuando una cifra del
builder no cuadra con REQUERIDAS, se usa la fuente que indica REQUERIDAS (ver notas en
cada figura). Determinista: las semillas viven en `config.py`; el único cálculo en vivo
es el HMM de SPY (F4.1), que es determinista por construcción.

Uso:  python tesis/figuras/gen_figuras_cap4.py
"""
import os
import sys
import json
import warnings
from pathlib import Path

# --- Bootstrap raíz + carga de TODOS los JSON auditados (reutiliza la celda del builder) ---
_ROOT = Path(__file__).resolve().parent
while not (_ROOT / "config.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

FIGDIR = _ROOT / "tesis" / "figuras"


def _load(p):
    return json.load(open(p))


DP = _load("outputs/experiments/decision_automl_prep.json")
PAN = _load("outputs/experiments/automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")["por_activo"]
IMP = _load("outputs/experiments/automl_importance.json")["por_activo"]
CLU = _load("outputs/experiments/strategy_clustering15.json")
CL10 = _load("outputs/experiments/cluster_panel10.json")
ANR = _load("outputs/experiments/automl_net_returns.json")["por_activo"]
MECH = _load("outputs/experiments/mechanism_panel.json")["por_activo"]
DET = _load("outputs/experiments/detector_analysis_SPY.json")
DETXLE = _load("outputs/experiments/detector_analysis_XLE.json")
DETMAR = _load("outputs/experiments/detector_analysis_MARA.json")
BBC = _load("outputs/experiments/bullbear_confirmatory.json")
REGDID = _load("outputs/experiments/regime_did_learners.json")
EQT = _load("outputs/experiments/equivalence_tost.json")
ABT = _load("outputs/experiments/alfa_beta_lectura.json")
RDT = _load("outputs/experiments/regime_direction_table.json")
SPYIV = _load("outputs/experiments/spy_intervention_variants.json")
NAT = {a: CLU["por_activo"][a]["nat"] for a in CLU["por_activo"]}
CONF = _load("outputs/experiments/confusion_panel.json")
IANA = _load("outputs/experiments/spy_intervention_anatomy.json")
GATE = _load("outputs/experiments/spy_panel_gate_descriptive.json")
LEV10 = _load("outputs/experiments/leverage_law_panel10.json")  # F4.13: ley leverage sobre los 10
# --- JSON adicionales para las 16 figuras nuevas (cuerpo + anexo) ---
PANROB = _load("outputs/experiments/panel_robustness.json")              # rodante + val/test + bull/bear (panel 10)
CALW = _load("outputs/experiments/calib_window_panel.json")             # robustez a la ventana de calibración
SPYME = _load("outputs/experiments/spy_mechanism_extras.json")          # SPY: daily + SHAP dependency + cuota rodante
THR = _load("cache/models/strata_thresholds.json")                      # umbrales ex-ante PSA/GSO (calib 2000–2024-09)
DETABL = _load("outputs/experiments/detector_ablation_panel.json")      # activación detectores (10) + ablación M10 (SPY)
AABL = _load("outputs/experiments/automl_ablation_detectors.json")["ablacion_automl_spy"]  # ablación AutoML (SPY)
CL10PA = CL10["por_activo"]                                             # cuota SHAP por activo (panel 10)
CL10PROF = CL10["perfiles_k3"]["kmeans"]                                # perfiles por grupo (clustering canónico 10)

PANEL10 = ["SPY", "QQQ", "XLF", "DIA", "XLK", "XLE", "ROKU", "SMCI", "MARA", "UNG"]
DPA = DP["por_activo"]
COL = {"M5": "#9e9e9e", "M8": "#f0a830", "M10": "#2c7fb8", "AutoML": "#27ae60", "ZeroR": "#7d3c98", "B&H": "#c0392b"}
REGCOL = {0: "#2e9e4f", 1: "#e8a33d", 2: "#c0392b"}
REGNAME = {0: "Calma", 1: "Estrés", 2: "Crisis"}
PKEY = {"M5": "m5", "M8": "m8", "M10": "m10_xgb", "AutoML": "automl", "ZeroR": "zeror", "B&H": "bh"}

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

# --- Formateadores con coma decimal (español) ---
_coma = mticker.FuncFormatter(lambda v, _p: f"{v:.2f}".replace(".", ","))
_coma1 = mticker.FuncFormatter(lambda v, _p: f"{v:.1f}".replace(".", ","))


def _c(x, n=2):
    """Formatea un número con coma decimal (para textos dentro de la figura)."""
    return f"{x:.{n}f}".replace(".", ",")


def _save(fig, nombre):
    p = FIGDIR / f"{nombre}.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {nombre}.pdf")


# ════════════════════════════════════════════════════════════════════════════
# F4.1 — cap4_regimenes_spy : regímenes HMM sombreados sobre el precio de SPY
# Fuente: HMM en vivo (build_states, determinista) + regime_direction_table.json
# ════════════════════════════════════════════════════════════════════════════
def f4_1():
    from experiments.quant_validation_panel import build_states
    gamma, sigma, oos_ret = build_states("SPY")
    g = gamma.reindex(oos_ret.index).dropna()
    dom = g.values.argmax(1)
    px = (1 + oos_ret.reindex(g.index)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(px.index, px.values, color="#222", lw=1.1, label="SPY (nivel B&H)")
    for st in (0, 1, 2):
        ax.fill_between(px.index, float(px.min()), float(px.max()),
                        where=(dom == st), color=REGCOL[st], alpha=0.12, step="mid")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGCOL[st], alpha=0.35) for st in (0, 1, 2)]
    ax.legend(handles + [plt.Line2D([], [], color="#222", lw=1.1)],
              ["Calma", "Estrés", "Crisis", "SPY (nivel B&H)"], fontsize=8, loc="upper left", ncol=2)
    ax.set_ylabel("nivel (1 € invertido)")
    ax.yaxis.set_major_formatter(_coma)
    _save(fig, "cap4_regimenes_spy")


# ════════════════════════════════════════════════════════════════════════════
# F4.2 — cap4_scores_detectores : distribución de scores RAM/PSA/GSO con su umbral
# Fuente: detector_analysis_SPY.json (scores + umbrales ex-ante)
# RAM τ=0,50 ; PSA P95=0,023 ; GSO P95=2,371
# ════════════════════════════════════════════════════════════════════════════
def f4_2():
    sc = DET["scores"]
    thr = sc["umbrales"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.3))
    # RAM (casi binario)
    axes[0].hist(sc["ram_score"], bins=30, color="#2c7fb8", alpha=.75)
    axes[0].axvline(thr["RAM_tau"], color="k", ls="--", lw=1.2,
                    label=f"τ = {_c(thr['RAM_tau'])}")
    axes[0].set_xlabel("RAM score")
    axes[0].set_ylabel("nº de días")
    axes[0].legend(fontsize=9)
    # PSA y GSO con P95/P99 (escala log para ver la cola)
    for ax, key, name, p95, p99 in [
        (axes[1], "psa_score", "PSA", thr["PSA_p95"], thr["PSA_p99"]),
        (axes[2], "gso_score", "GSO", thr["GSO_p95"], thr["GSO_p99"]),
    ]:
        ax.hist(sc[key], bins=30, color="#7d3c98", alpha=.75)
        ax.axvline(p95, color="k", ls="--", lw=1, label=f"P95 = {_c(p95, 3 if name == 'PSA' else 2)}")
        ax.axvline(p99, color="k", ls=":", lw=1, label=f"P99 = {_c(p99, 2)}")
        ax.set_xlabel(f"{name} score")
        ax.set_ylabel("nº de días")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(_coma1)
    fig.tight_layout()
    _save(fig, "cap4_scores_detectores")


# ════════════════════════════════════════════════════════════════════════════
# F4.3 — cap4_confusion_spy : matriz agente (M5) vs intervención (M8)
# sobre las 121 intervenciones (71 aciertos de M8 / 50 fallos)
# Fuente: spy_intervention_anatomy.json (serie de días intervenidos)
# ════════════════════════════════════════════════════════════════════════════
def f4_3():
    s = IANA["serie"]
    iv = np.array(s["intervino"]).astype(bool)
    m8h = np.array(s["m8_hit"]).astype(bool)
    m5 = np.array(s["m5_pos"], float)
    rn = np.array(s["r_next"], float)
    m5h = np.sign(m5) == np.sign(rn)
    # Filas: M8 acierta / M8 falla ; Columnas: M5 acierta / M5 falla
    M = np.array([
        [int((iv & m8h & m5h).sum()), int((iv & m8h & ~m5h).sum())],
        [int((iv & ~m8h & m5h).sum()), int((iv & ~m8h & ~m5h).sum())],
    ])
    bal = IANA["balance_intervenciones"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["agente (M5)\nacierta", "agente (M5)\nfalla"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["intervención\n(M8) acierta", "intervención\n(M8) falla"], fontsize=9)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=16,
                    color="white" if M[i, j] > M.max() * 0.6 else "black")
    ax.set_xlabel(f"{bal['n_intervenciones']} días de intervención · "
                  f"M8 acierta {bal['intervenciones_acertadas']} / falla {bal['intervenciones_fallidas']}",
                  fontsize=9)
    fig.tight_layout()
    _save(fig, "cap4_confusion_spy")


# ════════════════════════════════════════════════════════════════════════════
# F4.4 — cap4_equity_spy : equity de las 6 estrategias en SPY (n=251)
# Fuente: automl_net_returns.json (AutoML) + decision_automl_prep.json (resto)
# AutoML gana: Sharpe +2,68 vs trivial +2,21 ; equity 1,38× vs 1,30×
# ════════════════════════════════════════════════════════════════════════════
def f4_4():
    nr = DPA["SPY"]["net_returns"]
    serie = {"M5": nr["m5"], "M8": nr["m8"], "M10": nr["m10"],
             "AutoML": ANR["SPY"]["automl"], "ZeroR": nr["zeror"], "B&H": nr["bh"]}
    eqf = {s: float(np.cumprod(1 + np.nan_to_num(np.array(v, float)))[-1]) for s, v in serie.items()}
    win = max(eqf, key=eqf.get)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for s, v in serie.items():
        eq = np.cumprod(1 + np.nan_to_num(np.array(v, float)))
        ax.plot(eq, color=COL[s], lw=2.6 if s == win else 1.2,
                alpha=1 if s in (win, "M5") else .85,
                label=f"{s} (×{_c(eq[-1])})" + ("  ★" if s == win else ""))
    ax.axhline(1, color="k", lw=.6, alpha=.5)
    ax.set_xlabel("días del OOS")
    ax.set_ylabel("capital (1 € inicial)")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, "cap4_equity_spy")


# ════════════════════════════════════════════════════════════════════════════
# F4.5 — cap4_sensibilidad_umbrales : accuracy vs umbral (meseta = robustez)
# Fuente: spy_intervention_variants.json (sweep_ram_tau, sweep_m10_p1)
# ════════════════════════════════════════════════════════════════════════════
def f4_5():
    sr, sp = SPYIV["sweep_ram_tau"], SPYIV["sweep_m10_p1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    axes[0].plot([r["tau"] for r in sr], [r["accuracy"] for r in sr], "o-",
                 color="#f0a830", label="accuracy M8")
    axes[0].axvline(0.5, color="k", ls="--", lw=.8, label="τ canónico = 0,5")
    axes[0].set_xlabel("gate RAM τ")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    axes[1].plot([r["p1_thr"] for r in sp], [r["accuracy"] for r in sp], "o-",
                 color="#2c7fb8", label="accuracy M10")
    axes[1].axvline(0.5, color="k", ls="--", lw=.8, label="p1* canónico = 0,5")
    axes[1].set_xlabel("umbral del meta-learner p1*")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.xaxis.set_major_formatter(_coma1)
        ax.yaxis.set_major_formatter(_coma)
    fig.tight_layout()
    _save(fig, "cap4_sensibilidad_umbrales")


# ════════════════════════════════════════════════════════════════════════════
# F4.6 — cap4_gate_ram : tasa de intervención vs discrepancia agente↔régimen
# 1 punto por activo (Pearson r=0,93). Fuente: spy_panel_gate_descriptive.json
# ════════════════════════════════════════════════════════════════════════════
def f4_6():
    gp = GATE["gate_por_activo"]
    disc = np.array([gp[t]["discrepancia_agente_regimen"] for t in PANEL10])
    interv = np.array([gp[t]["tasa_intervencion"] for t in PANEL10])
    r, p = pearsonr(disc, interv)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.scatter(disc, interv, s=85, color="#2c3e50", edgecolor="k", lw=.5, zorder=3)
    for t, x, y in zip(PANEL10, disc, interv):
        ax.annotate(t, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    b1, b0 = np.polyfit(disc, interv, 1)
    xs = np.linspace(disc.min(), disc.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.5, ls="--",
            label=f"Pearson r = {_c(r)} (p < 0,001)")
    ax.set_xlabel("discrepancia agente ↔ régimen")
    ax.set_ylabel("tasa de intervención de M8")
    ax.xaxis.set_major_formatter(_coma)
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "cap4_gate_ram")


# ════════════════════════════════════════════════════════════════════════════
# F4.7 — cap4_heatmap_accuracy : heatmap accuracy activo×estrategia centrado en 0,5
# Fuente: panel mm25 (PAN). Color = distancia a la trivial (0,5)
# ════════════════════════════════════════════════════════════════════════════
def f4_7():
    acc = pd.DataFrame({s: {a: PAN[a]["table"][PKEY[s]]["accuracy"] for a in PANEL10} for s in COL}).loc[PANEL10]
    M = acc[list(COL)].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    norm = TwoSlopeNorm(vmin=float(M.min().min()), vcenter=0.5, vmax=float(M.max().max()))
    im = ax.imshow(M.values, cmap="RdYlBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(COL)))
    ax.set_xticklabels(list(COL))
    ax.set_yticks(range(len(PANEL10)))
    ax.set_yticklabels(PANEL10, fontsize=8)
    ax.grid(False)
    for i in range(len(PANEL10)):
        for j in range(len(COL)):
            ax.text(j, i, _c(M.values[i, j]), ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, shrink=.8)
    cb.ax.yaxis.set_major_formatter(_coma)
    cb.set_label("accuracy (centro = 0,5)")
    fig.tight_layout()
    _save(fig, "cap4_heatmap_accuracy")


# ════════════════════════════════════════════════════════════════════════════
# F4.8 — cap4_forest_pooled : forest plot ΔSharpe pooled-10 (IC95 simple)
# Fuente: bullbear_confirmatory.json -> confirmatorio.POOLED10
# M8 +0,61 [0,05, 1,22], M10 +1,11 [0,39, 1,84], AutoML +1,10 [0,40, 1,85].
# Los tres IC excluyen el cero. (REQUERIDAS los cita como +0,60/+1,12/+1,08 — misma
# cifra a redondeo; uso los valores exactos del JSON.)
# ════════════════════════════════════════════════════════════════════════════
def f4_8():
    c = BBC["confirmatorio"]["POOLED10"]["pairs"]
    labs = ["M8_vs_M5", "M10_vs_M5", "AutoML_vs_M5"]
    nice = {"M8_vs_M5": "M8 vs M5", "M10_vs_M5": "M10 vs M5", "AutoML_vs_M5": "AutoML vs M5"}
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, k in enumerate(labs):
        v = c[k]
        med = v["median_delta_sharpe"]
        lo, hi = v["ci95_low"], v["ci95_high"]
        col = "#27ae60"  # los tres excluyen el cero
        ax.plot([lo, hi], [i, i], color=col, lw=2.5)
        ax.plot(med, i, "o", color=col, ms=8, zorder=4)
        ax.text(hi + 0.05, i,
                f"Δ = {_c(med)}  IC95 [{_c(lo)}, {_c(hi)}]",
                va="center", fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([nice[k] for k in labs])
    ax.set_xlim(-0.5, 3.4)
    ax.set_xlabel("ΔSharpe (mediana, bootstrap estacionario pareado) — pooled-10 (n = 2493)")
    ax.xaxis.set_major_formatter(_coma)
    fig.tight_layout()
    _save(fig, "cap4_forest_pooled")


# ════════════════════════════════════════════════════════════════════════════
# F4.9 — cap4_equity_panel : equity por activo con la ganadora destacada
# Fuente: automl_net_returns.json + decision_automl_prep.json
# ════════════════════════════════════════════════════════════════════════════
def f4_9():
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.ravel()
    handles_lab = {}
    for ax, a in zip(axes, PANEL10):
        nr = DPA[a]["net_returns"]
        ser = {"M5": nr["m5"], "M8": nr["m8"], "M10": nr["m10"],
               "AutoML": ANR[a]["automl"], "ZeroR": nr["zeror"], "B&H": nr["bh"]}
        eqf = {s: float(np.cumprod(1 + np.nan_to_num(np.array(v, float)))[-1]) for s, v in ser.items()}
        win = max(eqf, key=eqf.get)
        for s, v in ser.items():
            eq = np.cumprod(1 + np.nan_to_num(np.array(v, float)))
            ln, = ax.plot(eq, color=COL[s], lw=2.4 if s == win else .9,
                          alpha=1 if s == win else .8)
            handles_lab[s] = ln
        ax.axhline(1, color="k", lw=.5, alpha=.5)
        ax.set_title(f"{a}  (★ {win})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(_coma1)
    axes[0].set_ylabel("capital (1 €)")
    axes[5].set_ylabel("capital (1 €)")
    fig.legend([handles_lab[s] for s in COL], list(COL), fontsize=8, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "cap4_equity_panel")


# ════════════════════════════════════════════════════════════════════════════
# F4.10 — cap4_tost_2x2 : diagrama 2×2 del TOST (accuracy/Sharpe; superior./equiv.)
# Fuente: equivalence_tost.json -> POOLED10 (M10 y AutoML vs la regla M8)
# ════════════════════════════════════════════════════════════════════════════
def f4_10():
    pares = {"M10_vs_M8": ("M10 vs M8", "#2c7fb8"), "AutoML_vs_M8": ("AutoML vs M8", "#27ae60")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met, mlab in [(axes[0], "accuracy", "Δaccuracy (aprendiz − regla M8)"),
                          (axes[1], "sharpe", "ΔSharpe (aprendiz − regla M8)")]:
        for i, (k, (nm, col)) in enumerate(pares.items()):
            x = EQT["POOLED10"][k][met]
            lo, hi, pt = x["ci90_low"], x["ci90_high"], x["point"]
            delta = x["delta_preReg"]
            ax.plot([lo, hi], [i, i], color=col, lw=2.6)
            ax.plot(pt, i, "o", color=col, ms=9, zorder=4)
            ax.text(hi, i + 0.12,
                    f"Δ = {_c(pt, 3 if met == 'accuracy' else 2)}  IC90 [{_c(lo, 3 if met == 'accuracy' else 2)}, "
                    f"{_c(hi, 3 if met == 'accuracy' else 2)}]", va="bottom", ha="right", fontsize=8)
        ax.axvline(0, color="k", lw=.9)
        # banda de equivalencia pre-registrada (−δ, +δ)
        ax.axvspan(-delta, delta, color="#bbb", alpha=0.25, label=f"banda equivalencia ±{_c(delta, 3 if met == 'accuracy' else 2)}")
        ax.set_yticks(range(len(pares)))
        ax.set_yticklabels([v[0] for v in pares.values()])
        ax.set_ylim(-0.6, len(pares) - 0.3)
        ax.set_xlabel(mlab)
        ax.xaxis.set_major_formatter(_coma)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "cap4_tost_2x2")


# ════════════════════════════════════════════════════════════════════════════
# F4.11 — cap4_did_regimen : ΔSharpe de M10 y AutoML por régimen + DiD ΔΔSharpe +1,37
# Fuente: bullbear_confirmatory.json (por_regimen POOLED10) + regime_did_learners.json
# ════════════════════════════════════════════════════════════════════════════
def f4_11():
    reg = BBC["por_regimen"]["POOLED10"]
    pares = ["M8_vs_M5", "M10_vs_M5", "AutoML_vs_M5"]
    nice = {"M8_vs_M5": "M8", "M10_vs_M5": "M10", "AutoML_vs_M5": "AutoML"}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    # ΔSharpe por régimen (barras alcista/bajista) frente a M5
    x = np.arange(len(pares))
    w = 0.38
    alc = [reg["alcista"]["contrastes"][p]["delta_sharpe"] for p in pares]
    baj = [reg["bajista"]["contrastes"][p]["delta_sharpe"] for p in pares]
    ax.bar(x - w / 2, alc, w, color="#2c7fb8", edgecolor="k", lw=.5,
           label=f"alcista (n={reg['alcista']['n']})")
    ax.bar(x + w / 2, baj, w, color="#c0392b", edgecolor="k", lw=.5, hatch="//",
           label=f"bajista (n={reg['bajista']['n']})")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[p] for p in pares])
    ax.set_ylabel("ΔSharpe vs M5")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_did_regimen")


def n17_anexo_did():
    # Panel del DiD pooled-10 (M10−AutoML: alcista − bajista) = +1,37, antes a la derecha de F4.11
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    d = REGDID["POOLED10"]
    col = "#27ae60" if d["ci95_low"] > 0 else "#c0392b"
    ax.plot([d["ci95_low"], d["ci95_high"]], [0, 0], color=col, lw=3)
    ax.plot(d["did_point"], 0, "o", color=col, ms=10, zorder=4)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks([0])
    ax.set_yticklabels(["DiD pooled-10"])
    ax.set_ylim(-1, 1)
    ax.text(d["did_point"], 0.25,
            f"ΔΔSharpe = {_c(d['did_point'])}\nIC95 [{_c(d['ci95_low'])}, {_c(d['ci95_high'])}]  p = {_c(d['p_one_sided_did_gt_0'], 3)}",
            ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("DiD de Sharpe (M10 − AutoML: alcista − bajista)")
    ax.xaxis.set_major_formatter(_coma)
    fig.tight_layout()
    _save(fig, "cap4_anexo_did")


# ════════════════════════════════════════════════════════════════════════════
# F4.12 — cap4_atribucion_capas : izq. atribución P&L por detector (RAM 100%, PSA/GSO 0)
# der. timeline diario rescate riesgo (M8) vs accuracy (aprendiz)
# Fuente: detector_analysis_SPY.json (atribución) + spy_intervention_anatomy.json (serie)
# ════════════════════════════════════════════════════════════════════════════
def f4_12():
    at = DET["atribucion_pnl"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    # IZQ: atribución de P&L de rescate por detector
    ax = axes[0]
    dets = ["RAM", "PSA", "GSO"]
    pnl = [at["pnl_dias_RAM_disparado"], at["pnl_dias_PSA_disparado"], at["pnl_dias_GSO_disparado"]]
    cols = ["#2c7fb8", "#7d3c98", "#c0392b"]
    ax.bar(dets, pnl, color=cols, edgecolor="k", lw=.5)
    for i, v in enumerate(pnl):
        ax.text(i, v + 0.005, _c(v, 3), ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("P&L de rescate atribuido")
    ax.yaxis.set_major_formatter(_coma)
    ax.set_xlabel("RAM concentra todo el rescate; PSA/GSO inertes en este OOS")
    # DER: timeline diario de las intervenciones de M8 (verde acierta / rojo falla)
    ax2 = axes[1]
    s = IANA["serie"]
    x = pd.to_datetime(s["dates"])
    iv = np.array(s["intervino"]).astype(bool)
    hit = np.array(s["m8_hit"]).astype(bool)
    rn = np.abs(np.array(s["r_next"], float)) * 100
    cum = np.cumsum(s["r_next"])
    ax2.plot(x, cum, color="#bbb", lw=1, label="SPY (retorno acumulado)")
    for msk, col, lab in [(iv & hit, "#27ae60", "intervención (M8) acierta"),
                          (iv & ~hit, "#c0392b", "intervención (M8) falla")]:
        ax2.scatter(x[msk], cum[msk], s=20 + 8 * rn[msk], color=col, alpha=.75,
                    edgecolor="k", lw=.3, label=lab)
    ax2.set_ylabel("retorno acumulado")
    ax2.yaxis.set_major_formatter(_coma)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "cap4_atribucion_capas")


# ════════════════════════════════════════════════════════════════════════════
# F4.13 — cap4_scatter_leverage : scatter leverage↔rescate (10 activos)
# recta r=−0,56 p=0,093 ; ROKU señalado como excepción
# Fuente: leverage_law_panel10.json (REQUERIDAS exige el *10, no MECH ni el panel-15)
# ════════════════════════════════════════════════════════════════════════════
def f4_13():
    pa = LEV10["por_activo"]
    ley = LEV10["ley_leverage"]
    activos = [a for a in PANEL10 if a in pa]
    lev = np.array([pa[a]["leverage_corr"] for a in activos])
    res = np.array([pa[a]["rescate_aprendiz"] for a in activos])
    r, p = ley["pearson_r"], ley["pearson_p"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for a, x, y in zip(activos, lev, res):
        exc = (a == ley.get("excepcion"))
        ax.scatter(x, y, s=95, color="#e67e22" if exc else "#27ae60",
                   edgecolor="k", lw=.6, zorder=3)
        ax.annotate(a + ("  (excepción)" if exc else ""), (x, y), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    b1, b0 = np.polyfit(lev, res, 1)
    xs = np.linspace(lev.min(), lev.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.5, ls="--",
            label=f"Pearson r = {_c(r)} (p = {_c(p, 3)})")
    ax.set_xlabel("leverage de Black (corr. retorno–volatilidad; más negativo = leverage estándar más fuerte)")
    ax.set_ylabel("Δaccuracy (mejor aprendiz − agente M5)")
    ax.xaxis.set_major_formatter(_coma)
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "cap4_scatter_leverage")


# ════════════════════════════════════════════════════════════════════════════
# F4.14 — cap4_pca_clusters : PCA 2D de la naturaleza (10 activos), 3 clusters,
# PC1≈leverage (r=0,84). Fuente: cluster_panel10.json + mechanism_panel.json
# ════════════════════════════════════════════════════════════════════════════
def f4_14():
    clus = CL10["clustering"]
    ok = CL10["meta"]["panel"]
    lab = np.array(clus["k3"]["kmeans"]["labels"])
    pca = np.array(CL10["meta"]["pca2d"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    # izq: PCA 2D con los 3 clusters
    ax = axes[0]
    for cgrp in sorted(set(lab)):
        idx = np.where(lab == cgrp)[0]
        ax.scatter(pca[idx, 0], pca[idx, 1], s=95, label=f"C{cgrp}", edgecolor="k", lw=.4)
    for i, a in enumerate(ok):
        ax.annotate(a, (pca[i, 0], pca[i, 1]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.xaxis.set_major_formatter(_coma1)
    ax.yaxis.set_major_formatter(_coma1)
    ax.legend(fontsize=8)
    # der: PC1 vs leverage (r≈0,84)
    ax2 = axes[1]
    levv = np.array([MECH[a]["leverage_corr"] for a in ok])
    rpc, ppc = pearsonr(pca[:, 0], levv)
    ax2.scatter(pca[:, 0], levv, s=80, color="#2c7fb8", edgecolor="k", lw=.5)
    for i, a in enumerate(ok):
        ax2.annotate(a, (pca[i, 0], levv[i]), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    b1, b0 = np.polyfit(pca[:, 0], levv, 1)
    xs = np.linspace(pca[:, 0].min(), pca[:, 0].max(), 50)
    ax2.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.3, ls="--",
             label=f"Pearson r = {_c(rpc)} (p = {_c(ppc, 3)})")
    ax2.set_xlabel("PC1 (eje principal de la naturaleza)")
    ax2.set_ylabel("leverage de Black (corr. retorno–vol.)")
    ax2.xaxis.set_major_formatter(_coma1)
    ax2.yaxis.set_major_formatter(_coma)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "cap4_pca_clusters")


# ════════════════════════════════════════════════════════════════════════════
# F4.15 — cap4_casos : dos casos trabajados (XLE régimen / MARA leverage invertido)
# dirección agente vs supervisor vs régimen. Fuente: detector_analysis_XLE/MARA.json
# ════════════════════════════════════════════════════════════════════════════
def f4_15():
    casos = [("XLE", DETXLE, "canal RÉGIMEN (leverage presente)"),
             ("MARA", DETMAR, "canal ML (leverage invertido)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for ax, (tk, D, sub) in zip(axes, casos):
        iv = D["intervencion"]
        # acierto al intervenir: agente (M5) vs supervisor regla (M8)
        vals = [iv["acc_M5_si_interviene"], iv["acc_M8_si_interviene"]]
        cols = ["#9e9e9e", "#f0a830"]
        bars = ax.bar(["agente (M5)", "supervisor (M8)"], vals, color=cols, edgecolor="k", lw=.5)
        ax.axhline(0.5, color="k", ls="--", lw=1, label="azar (0,5)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, _c(v, 3), ha="center", fontsize=10)
        ax.set_ylim(0, max(0.7, max(vals) + 0.08))
        ax.set_ylabel("accuracy en los días intervenidos")
        ax.yaxis.set_major_formatter(_coma)
        ok = iv["acc_M8_si_interviene"] >= 0.5
        ax.set_title(f"{tk} · {sub}", fontsize=10)
        ax.set_xlabel(
            f"M8 interviene {iv['tasa_intervencion']*100:.0f}% ({iv['n_intervenciones']} días) · "
            + ("la regla corrige (>0,5)" if ok else "la regla mete ruido (<0,5); rescata el aprendiz"),
            fontsize=8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_casos")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  16 FIGURAS NUEVAS — cuerpo (1–6) y anexo (7–16) del Capítulo 4            ║
# ║  Misma estética: _save / _c / _coma / COL / REGCOL. Sin título embebido.  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_REGUP = lambda s: "LARGO ▲" if s > 0 else ("CORTO ▼" if s < 0 else "NEUTRAL")
_REGCL = lambda s: "#27ae60" if s > 0 else ("#c0392b" if s < 0 else "#bbb")


# ════════════════════════════════════════════════════════════════════════════
# N1 — cap4_anatomia_dia : anatomía de un día ACERTADO y uno FALLIDO + balance
# Builder ≈L259-L315 (diagrama de flujo agente→STRATA→resultado de los dos casos)
# Fuente: spy_intervention_anatomy.json (caso_acierto, caso_fallo, balance)
# ════════════════════════════════════════════════════════════════════════════
def n1_anatomia_dia():
    ca, cf = IANA["caso_acierto"], IANA["caso_fallo"]
    ba = IANA["balance_intervenciones"]

    def _flow(ax, c, tag):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(f"{tag} · {c['fecha']} · régimen {c['regimen']} · RAM = {_c(c['ram_score'])}", fontsize=10)
        # 1) agente (M5) + votos de las 5 personalidades
        ax.text(1.6, 7.6, "AGENTE (M5)", ha="center", fontsize=8, color="#555")
        ax.text(1.6, 6.7, _REGUP(c["agente_M5"]), ha="center", fontsize=12, fontweight="bold",
                color="white", bbox=dict(boxstyle="round", fc=_REGCL(c["agente_M5"]), ec="k"))
        for i, (_p, s) in enumerate(c["votos_personalidades"].items()):
            ax.scatter(0.5 + i * 0.55, 5.4, s=90, color=_REGCL(s), edgecolor="k", lw=.4)
        ax.text(1.6, 4.7, "votos 5 pers.", ha="center", fontsize=6.5, color="#888")
        # 2) RAM voltea -> STRATA (M8)
        ax.annotate("", xy=(4.6, 6.7), xytext=(2.9, 6.7), arrowprops=dict(arrowstyle="-|>", lw=2, color="#2c3e50"))
        ax.text(3.75, 7.2, "RAM voltea", ha="center", fontsize=7, color="#2c3e50")
        ax.text(6.1, 7.6, "STRATA (M8)", ha="center", fontsize=8, color="#555")
        ax.text(6.1, 6.7, _REGUP(c["STRATA_M8"]), ha="center", fontsize=12, fontweight="bold",
                color="white", bbox=dict(boxstyle="round", fc=_REGCL(c["STRATA_M8"]), ec="k"))
        # 3) resultado al día siguiente
        ax.annotate("", xy=(8.4, 6.7), xytext=(7.3, 6.7), arrowprops=dict(arrowstyle="-|>", lw=2, color="#2c3e50"))
        mk, mkc = ("✓", "#27ae60") if c["M8_acierta"] else ("✗", "#c0392b")
        ax.text(9.2, 6.7, mk, ha="center", va="center", fontsize=26, color=mkc, fontweight="bold")
        ax.bar(9.2, c["r_next"] * 30, width=0.7, bottom=3.0, color=_REGCL(c["verdad"]), edgecolor="k")
        ax.text(9.2, 2.3, f"r_next\n{_c(c['r_next'] * 100, 1)} %", ha="center", fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    _flow(axes[0], ca, "ACIERTO")
    _flow(axes[1], cf, "FALLO")
    fig.text(0.5, 0.01,
             f"Balance de las {ba['n_intervenciones']} intervenciones: M8 acierta {_c(ba['acc_M8_en_intervencion'] * 100, 0)} % "
             f"vs agente {_c(ba['acc_M5_en_intervencion'] * 100, 0)} % · "
             f"{ba['intervenciones_acertadas']} aciertan / {ba['intervenciones_fallidas']} fallan",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "cap4_anatomia_dia")


# ════════════════════════════════════════════════════════════════════════════
# N2 — cap4_confusion_spy_6 : matrices de confusión SPY de las 6 estrategias
# Builder ≈L483-L497. Fuente: confusion_panel.json (spy_por_estrategia)
# ════════════════════════════════════════════════════════════════════════════
def n2_confusion_spy_6():
    spc = CONF["spy_por_estrategia"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.ravel()
    for ax, s in zip(axes, ["M5", "M8", "M10", "AutoML", "ZeroR", "B&H"]):
        cm = spc[s]
        M = np.array([[cm["TP"], cm["FP"]], [cm["FN"], cm["TN"]]])
        ax.imshow(M, cmap="Blues")
        ax.set_title(f"{s}\nacc = {_c(cm['accuracy'])}", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["real ↑", "real ↓"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["pred L", "pred S"], fontsize=8)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=11,
                        color="white" if M[i, j] > M.max() * 0.6 else "black")
    fig.tight_layout()
    _save(fig, "cap4_confusion_spy_6")


# ════════════════════════════════════════════════════════════════════════════
# N3 — cap4_activacion_panel : tasa de disparo RAM/PSA/GSO + intervención M8 (panel 10)
# Builder ≈L665-L678. Fuente: detector_ablation_panel.json (activacion_panel)
# ════════════════════════════════════════════════════════════════════════════
def n3_activacion_panel():
    act = DETABL["activacion_panel"]
    A = pd.DataFrame(act).T.loc[PANEL10]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(PANEL10))
    w = 0.25
    ax.bar(x - w, A["RAM"], w, color="#2c7fb8", label="RAM (régimen)")
    ax.bar(x, A["PSA"], w, color="#7d3c98", label="PSA (cambio opinión)")
    ax.bar(x + w, A["GSO"], w, color="#c0392b", label="GSO (volatilidad)")
    ax.set_xticks(x)
    ax.set_xticklabels(PANEL10, rotation=45)
    ax.set_ylabel("tasa de disparo (OOS)")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_activacion_panel")


# ════════════════════════════════════════════════════════════════════════════
# N4 — cap4_naturaleza_panel : naturaleza de los 10 activos (2×2)
# Builder ≈L710-L723. Fuente: strategy_clustering15.json -> nat por activo (NAT)
# ════════════════════════════════════════════════════════════════════════════
def n4_naturaleza_panel():
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.4))
    axes = axes.ravel()
    specs = [("leverage_corr", "Leverage de Black (corr. retorno–vol.; < 0 = estándar)", "#2c7fb8"),
             ("oos_crisis_frac", "Fracción de días en Crisis (OOS)", "#c0392b"),
             ("agent_short_frac", "Sesgo corto del agente (frac. días corto)", "#9e9e9e"),
             ("oos_vol", "Volatilidad media OOS (σ GARCH anualizada)", "#7d3c98")]
    order = sorted(PANEL10, key=lambda a: NAT[a]["leverage_corr"])
    for ax, (key, ttl, c) in zip(axes, specs):
        ax.bar(order, [NAT[a][key] for a in order], color=c)
        ax.set_title(ttl, fontsize=9.5)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.yaxis.set_major_formatter(_coma)
        if key == "leverage_corr":
            ax.axhline(0, color="k", lw=.6)
    fig.tight_layout()
    _save(fig, "cap4_naturaleza_panel")


# ════════════════════════════════════════════════════════════════════════════
# N5 — cap4_robustez_calib : robustez a la ventana de calibración (M10 acc)
# Builder ≈L1405-L1426. Fuente: calib_window_panel.json (por_activo)
# ════════════════════════════════════════════════════════════════════════════
def n5_robustez_calib():
    ca = CALW["por_activo"]
    rows = []
    for tk, ws in ca.items():
        for w in ws:
            if "m10_acc" in w:
                rows.append({"activo": tk, "inicio_calib": w["start"], "M10_acc": w["m10_acc"]})
    RC = pd.DataFrame(rows)
    piv = RC.pivot(index="activo", columns="inicio_calib", values="M10_acc")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for tk in piv.index:
        ax.plot([s[:4] for s in piv.columns], piv.loc[tk].values, marker="o", label=tk)
    ax.axhline(0.5, color="k", ls=":", lw=.8, label="azar (0,5)")
    ax.set_xlabel("inicio de la ventana de calibración")
    ax.set_ylabel("accuracy de M10 (OOS fijo)")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, "cap4_robustez_calib")


# ════════════════════════════════════════════════════════════════════════════
# N6 — cap4_robustez_panel : robustez de panel (rodante / val-test / por régimen)
# Builder ≈L1213-L1252. Fuente: panel_robustness.json
# ════════════════════════════════════════════════════════════════════════════
def n6_robustez_panel():
    PR = PANROB["por_activo"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    # (a) accuracy rodante SPY: agente vs mejor STRATA vs trivial
    sp = PR["SPY"]["rolling"]
    xs = pd.to_datetime(sp["dates"])
    axes[0].plot(xs, sp["serie"]["m5"], color=COL["M5"], lw=1.4, label="M5 (agente)")
    axes[0].plot(xs, sp["serie"][sp["mejor_strata"]], color="#27ae60", lw=1.4, label=f"mejor STRATA ({sp['mejor_strata']})")
    axes[0].plot(xs, sp["serie"][sp["trivial"]], color=COL["ZeroR"], lw=1.4, label=f"trivial ({sp['trivial']})")
    axes[0].axhline(0.5, color="k", ls=":", lw=.8)
    axes[0].set_ylabel(f"accuracy rodante (ventana {sp['ventana']} d)")
    axes[0].yaxis.set_major_formatter(_coma)
    axes[0].legend(fontsize=7)
    # (b) fracción de ventanas rodantes en que la mejor STRATA bate al agente, por activo
    fr = {a: PR[a]["rolling"]["frac_ventanas_mejorSTRATA_gt_M5"] for a in PANEL10}
    axes[1].bar(list(fr), list(fr.values()), color=["#27ae60" if v >= 0.5 else "#bbb" for v in fr.values()])
    axes[1].axhline(0.5, color="k", ls="--", lw=.8)
    axes[1].set_ylabel("frac. ventanas: mejor STRATA > agente")
    axes[1].yaxis.set_major_formatter(_coma)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    # (c) McNemar del rescate por régimen (pooled 10): alcista vs bajista
    pt = PANROB["pooled_bullbear"]["tests"]
    labs = list(pt)
    nice = [k.replace("_xgb", "").replace("_vs_m5", "").replace("_", " ") for k in labs]
    pv = [pt[k]["mcnemar_p"] for k in labs]
    axes[2].bar(range(len(labs)), pv, color=["#27ae60" if pt[k]["sig_0.10"] else "#c0392b" for k in labs])
    axes[2].axhline(0.10, color="k", ls="--", lw=.8, label="α = 0,10")
    axes[2].set_xticks(range(len(labs)))
    axes[2].set_xticklabels(nice, rotation=40, ha="right", fontsize=7)
    axes[2].set_ylabel("p de McNemar (rescate vs M5)")
    axes[2].yaxis.set_major_formatter(_coma)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_robustez_panel")


# ════════════════════════════════════════════════════════════════════════════
# N7 — cap4_anexo_confusion_panel : matrices de confusión panel (mejor STRATA, 2×5)
# Builder ≈L820-L833. Fuente: confusion_panel.json (panel_mejor_strata)
# ════════════════════════════════════════════════════════════════════════════
def n7_anexo_confusion_panel():
    pc = CONF["panel_mejor_strata"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    for ax, tk in zip(axes, PANEL10):
        cm = pc[tk]
        M = np.array([[cm["TP"], cm["FP"]], [cm["FN"], cm["TN"]]])
        ax.imshow(M, cmap="Blues")
        ax.set_title(f"{tk} · {cm['estrategia']}\nacc = {_c(cm['accuracy'])}", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["↑", "↓"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L", "S"], fontsize=8)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > M.max() * 0.6 else "black")
    fig.tight_layout()
    _save(fig, "cap4_anexo_confusion_panel")


# ════════════════════════════════════════════════════════════════════════════
# N8 — cap4_anexo_ablacion : ablación de features (agente-15 / STRATA-7 / 22) M10 y AutoML
# Builder ≈L401-L424. Fuente: detector_ablation_panel.json + automl_ablation_detectors.json
# ════════════════════════════════════════════════════════════════════════════
def n8_anexo_ablacion():
    ab = DETABL["ablacion_m10_spy"]
    zeror = PAN["SPY"]["table"]["zeror"]["accuracy"]
    sets = ["solo agente (15)", "solo STRATA (7)", "ALL22 (canónico)"]
    m10v = [ab[s]["accuracy"] for s in sets]
    amlv = [AABL[s]["accuracy"] for s in sets]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(sets))
    w = 0.38
    ax.bar(x - w / 2, m10v, w, color="#1a5276", edgecolor="k", lw=.8, label="M10 (XGBoost, params fijos)")
    ax.bar(x + w / 2, amlv, w, color="#16a085", edgecolor="k", lw=.8, label="AutoML (H2O, el ganador)")
    ax.axhline(0.5, color="k", ls="--", lw=.8, label="azar (0,5)")
    ax.axhline(zeror, color="#e67e22", ls=":", lw=1.4, label=f"ZeroR = {_c(zeror, 3)}")
    for i in range(len(sets)):
        ax.text(i - w / 2, m10v[i] + 0.003, _c(m10v[i], 3), ha="center", fontsize=8)
        ax.text(i + w / 2, amlv[i] + 0.003, _c(amlv[i], 3), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["solo agente", "solo STRATA", "las 22"])
    ax.set_ylim(0.40, 0.62)
    ax.set_ylabel("accuracy (OOS desplegable)")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    _save(fig, "cap4_anexo_ablacion")


# ════════════════════════════════════════════════════════════════════════════
# N9 — cap4_anexo_shap_dependency : cómo USA el modelo cada señal STRATA (color=régimen)
# Builder ≈L600-L613. Fuente: spy_mechanism_extras.json (shap_dependency)
# ════════════════════════════════════════════════════════════════════════════
def n9_anexo_shap_dependency():
    dep = SPYME["shap_dependency"]
    feats = list(dep)
    fig, axes = plt.subplots(1, len(feats), figsize=(4.3 * len(feats), 3.4))
    for ax, f in zip(axes, feats):
        xs = np.array(dep[f]["x"])
        sh = np.array(dep[f]["shap"])
        rg = np.array(dep[f]["regime"])
        for k in (0, 1, 2):
            m = rg == k
            if m.any():
                ax.scatter(xs[m], sh[m], s=18, color=REGCOL[k], alpha=.7, label=REGNAME[k])
        ax.axhline(0, color="k", lw=.5)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP (→ prob. subida)")
        ax.set_title(f, fontsize=9)
        ax.xaxis.set_major_formatter(_coma1)
        ax.yaxis.set_major_formatter(_coma)
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "cap4_anexo_shap_dependency")


# ════════════════════════════════════════════════════════════════════════════
# N10 — cap4_anexo_shap_cuota : cuota SHAP de STRATA por activo (+ ablación) en el panel
# Builder ≈L745-L776. Fuente: decision_automl_prep.json (ablation + shap por activo)
# ════════════════════════════════════════════════════════════════════════════
def n10_anexo_shap_cuota():
    cuota = {a: DPA[a]["shap"]["cuota_strata"] for a in PANEL10}
    dacc = {a: DPA[a]["ablation"]["d_acc_strata"] for a in PANEL10}
    order = sorted(PANEL10, key=lambda a: cuota[a])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    # izq: cuota SHAP de STRATA por activo
    axes[0].bar(order, [cuota[a] for a in order], color="#16a085", edgecolor="k", lw=.4)
    axes[0].axhline(0.5, color="k", ls="--", lw=.8, label="50 %")
    for i, a in enumerate(order):
        axes[0].text(i, cuota[a] + 0.01, _c(cuota[a]), ha="center", fontsize=7)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("cuota SHAP de las features STRATA")
    axes[0].yaxis.set_major_formatter(_coma)
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].legend(fontsize=8)
    # der: ablación — Δaccuracy al añadir STRATA al vector del agente
    oa = sorted(PANEL10, key=lambda a: dacc[a])
    axes[1].barh(oa, [dacc[a] for a in oa],
                 color=["#27ae60" if dacc[a] > 0 else "#c0392b" for a in oa], edgecolor="k", lw=.4)
    axes[1].axvline(0, color="k", lw=.6)
    axes[1].set_xlabel("Δaccuracy al añadir STRATA (las 22 − agente-15)")
    axes[1].xaxis.set_major_formatter(_coma)
    fig.tight_layout()
    _save(fig, "cap4_anexo_shap_cuota")


# ════════════════════════════════════════════════════════════════════════════
# N11 — cap4_anexo_shap_rodante : cuota SHAP rodante (estabilidad temporal)
# Builder ≈L859-L869. Fuente: spy_mechanism_extras.json (shap_rolling)
# ════════════════════════════════════════════════════════════════════════════
def n11_anexo_shap_rodante():
    cu = SPYME["shap_rolling"]["cuota_strata"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(range(len(cu)), cu, "o-", color="#2c7fb8")
    ax.axhline(float(np.mean(cu)), color="#c0392b", ls="--", lw=1, label=f"media {_c(float(np.mean(cu)))}")
    ax.axhline(0.5, color="k", ls=":", lw=.8, label="0,5")
    ax.set_ylim(0, 1)
    ax.set_xlabel("reentreno walk-forward")
    ax.set_ylabel("cuota STRATA en |SHAP|")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_anexo_shap_rodante")


# ════════════════════════════════════════════════════════════════════════════
# N12 — cap4_anexo_regimen_direccion : régimen × dirección (calib y OOS)
# leverage contemporáneo (mismo día) sí; predicción (mañana ≈ 0,5) no.
# Builder ≈L443-L454. Fuente: regime_direction_table.json (SPY)
# ════════════════════════════════════════════════════════════════════════════
def n12_anexo_regimen_direccion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    rgs = ["Calma", "Estrés", "Crisis"]
    for ax, win, ttl in [(axes[0], "calib", "Calibración (n grande)"), (axes[1], "oos", "OOS")]:
        d = RDT["SPY"][win]
        same = [d[r]["ret_mismo_dia"] for r in rgs]
        nxt = [d[r]["frac_sube_sig"] for r in rgs]
        x = np.arange(3)
        ax2 = ax.twinx()
        b1 = ax.bar(x - 0.2, same, 0.4, color="#2c7fb8", label="ret. mismo día (leverage)")
        b2 = ax2.bar(x + 0.2, nxt, 0.4, color="#c0392b", label="frac. sube día sig.")
        ax2.axhline(0.5, color="k", ls=":", lw=.8)
        ax2.set_ylim(0.3, 0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(rgs)
        ax.set_title(ttl, fontsize=9)
        ax.axhline(0, color="k", lw=.5)
        ax.set_ylabel("ret. mismo día", color="#2c7fb8")
        ax2.set_ylabel("frac. sube mañana", color="#c0392b")
        ax.yaxis.set_major_formatter(_coma)
        ax2.yaxis.set_major_formatter(_coma)
        if win == "calib":
            ax.legend([b1, b2], ["ret. mismo día (leverage)", "frac. sube día sig."], fontsize=7, loc="upper right")
    fig.tight_layout()
    _save(fig, "cap4_anexo_regimen_direccion")


# ════════════════════════════════════════════════════════════════════════════
# N13 — cap4_anexo_grupos : perfil por grupo del clustering (accuracy y Sharpe medios)
# Builder ≈L1086-L1119. Fuente: cluster_panel10.json (perfiles_k3.kmeans)
# Subfigura más informativa: accuracy y Sharpe medios por estrategia, por grupo.
# ════════════════════════════════════════════════════════════════════════════
def n13_anexo_grupos():
    prof = CL10PROF
    groups = list(prof)
    ss = ["M5", "M8", "M10", "AutoML", "ZeroR", "B&H"]
    fig, axes = plt.subplots(2, len(groups), figsize=(14, 6.4), sharey="row")
    for j, g in enumerate(groups):
        # fila 0: accuracy media
        am = prof[g]["acc_media"]
        vals = [am[s] for s in ss]
        axes[0, j].bar(ss, vals, color=[COL[s] for s in ss], edgecolor="k", lw=.4)
        axes[0, j].axhline(0.5, color="k", ls="--", lw=.8)
        best = prof[g]["mejor_acc_no_trivial"]
        axes[0, j].text(ss.index(best), am[best] + 0.004, "★", ha="center", color="#c0392b", fontsize=12)
        axes[0, j].set_title(f"{g}: {', '.join(prof[g]['activos'])}\nlev = {_c(prof[g]['naturaleza_media']['leverage_corr'])} "
                             f"vol = {_c(prof[g]['naturaleza_media']['oos_vol'])}", fontsize=8)
        axes[0, j].tick_params(axis="x", rotation=45, labelsize=7)
        axes[0, j].yaxis.set_major_formatter(_coma)
        # fila 1: Sharpe medio
        sm = prof[g]["sharpe_media"]
        sv = [sm[s] for s in ss]
        axes[1, j].bar(ss, sv, color=[COL[s] for s in ss], edgecolor="k", lw=.4)
        axes[1, j].axhline(0, color="k", lw=.8)
        bs = prof[g]["mejor_sharpe_no_trivial"]
        axes[1, j].text(ss.index(bs), sm[bs] + 0.04, "★", ha="center", color="#c0392b", fontsize=12)
        axes[1, j].tick_params(axis="x", rotation=45, labelsize=7)
        axes[1, j].yaxis.set_major_formatter(_coma)
    axes[0, 0].set_ylabel("accuracy media")
    axes[1, 0].set_ylabel("Sharpe medio")
    fig.tight_layout()
    _save(fig, "cap4_anexo_grupos")


# ════════════════════════════════════════════════════════════════════════════
# N14 — cap4_anexo_psa_gso : PSA/GSO dormidos (scores OOS vs umbral ex-ante + sesgo agente)
# Builder ≈L331-L362. Fuente: detector_analysis_SPY.json (scores) + mechanism_panel.json
# ════════════════════════════════════════════════════════════════════════════
def n14_anexo_psa_gso():
    sc = DET["scores"]
    thr = sc["umbrales"]
    shorts = {a: MECH[a]["agente_frac_corto"] for a in PANEL10}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    # PSA y GSO: dónde caen los scores OOS frente al umbral ex-ante (calibrado con crisis)
    for ax, key, name, p95, p99 in [(axes[0], "psa_score", "PSA", thr["PSA_p95"], thr["PSA_p99"]),
                                    (axes[1], "gso_score", "GSO", thr["GSO_p95"], thr["GSO_p99"])]:
        ax.hist(sc[key], bins=30, color="#7d3c98", alpha=.7)
        ax.axvline(p95, color="k", ls="--", lw=1, label=f"P95 = {_c(p95, 3 if name == 'PSA' else 2)}")
        ax.axvline(p99, color="k", ls=":", lw=1, label=f"P99 = {_c(p99, 2)}")
        ax.set_xlabel(f"{name} score (OOS SPY)")
        ax.set_ylabel("nº de días")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(_coma1)
    # sesgo persistente del agente: PSA (cambio estructural) no tiene qué detectar
    order = sorted(PANEL10, key=lambda a: shorts[a])
    axes[2].bar(order, [shorts[a] for a in order], color="#9e9e9e")
    axes[2].axhline(0.5, color="k", ls="--", lw=.8, label="0,5 (sin sesgo)")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("fracción de días CORTO")
    axes[2].yaxis.set_major_formatter(_coma)
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_anexo_psa_gso")


# ════════════════════════════════════════════════════════════════════════════
# N15 — cap4_anexo_confusion_m10_regimen : acierto M5 vs M10 por régimen (SPY)
# Builder ≈L581-L598. Fuente: spy_mechanism_extras.json (daily)
# ════════════════════════════════════════════════════════════════════════════
def n15_anexo_confusion_m10_regimen():
    d = SPYME["daily"]
    reg = np.array(d["regime"])
    truth = np.array(d["truth"])
    cm5 = (np.array(d["m5_pos"]) == truth)
    cm10 = (np.array(d["m10_pos"]) == truth)
    rows = []
    for k in (0, 1, 2):
        msk = reg == k
        if msk.sum() >= 1:
            rows.append({"régimen": REGNAME[k], "n": int(msk.sum()),
                         "acc_M5": float(cm5[msk].mean()), "acc_M10": float(cm10[msk].mean())})
    RM = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(RM))
    ax.bar(x - 0.2, RM["acc_M5"], 0.4, color=COL["M5"], label="M5 (agente)")
    ax.bar(x + 0.2, RM["acc_M10"], 0.4, color=COL["M10"], label="M10 (aprendiz)")
    ax.axhline(0.5, color="k", ls=":", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['régimen']}\n(n = {r['n']})" for _, r in RM.iterrows()])
    ax.set_ylabel("accuracy")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_anexo_confusion_m10_regimen")


# ════════════════════════════════════════════════════════════════════════════
# N16 — cap4_anexo_mcnemar : McNemar del rescate (p por estrategia, pooled 10)
# Builder ≈L1246-L1249. Fuente: panel_robustness.json (pooled_bullbear.tests)
# ════════════════════════════════════════════════════════════════════════════
def n16_anexo_mcnemar():
    pt = PANROB["pooled_bullbear"]["tests"]
    labs = list(pt)
    nice = [k.replace("_xgb", "").replace("_vs_m5", "").replace("_", " ") for k in labs]
    pv = [pt[k]["mcnemar_p"] for k in labs]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(len(labs)), pv, color=["#27ae60" if pt[k]["sig_0.10"] else "#c0392b" for k in labs])
    ax.axhline(0.10, color="k", ls="--", lw=.8, label="α = 0,10")
    for i, p in enumerate(pv):
        ax.text(i, p + 0.002, _c(p, 3), ha="center", fontsize=8)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(nice, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("p de McNemar (rescate vs M5)")
    ax.yaxis.set_major_formatter(_coma)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cap4_anexo_mcnemar")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5 FIGURAS DE APOYO — arquitectura, protocolo temporal y naturaleza        ║
# ║  Misma estética: _save / _c / _coma / COL / REGCOL. Sin título embebido.   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


def _caja(ax, x, y, w, h, texto, fc, ec="#333", fs=9, tc="black", lw=1.2):
    """Caja de esquina redondeada centrada en (x, y) con texto multilínea."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def _flecha(ax, xy_o, xy_d, texto="", fs=7.5, color="#2c3e50"):
    ax.add_patch(FancyArrowPatch(xy_o, xy_d, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color, zorder=1))
    if texto:
        mx, my = (xy_o[0] + xy_d[0]) / 2, (xy_o[1] + xy_d[1]) / 2
        ax.text(mx + 0.15, my, texto, ha="left", va="center", fontsize=fs, color="#555", zorder=4)


# ════════════════════════════════════════════════════════════════════════════
# S1 — strata_arquitectura : diagrama de bloques del sistema STRATA
# Agente LLM -> 3 detectores (RAM/PSA/GSO) -> M8 / aprendiz -> posición w*
# Esquemático, sin datos.
# ════════════════════════════════════════════════════════════════════════════
def strata_arquitectura():
    fig, ax = plt.subplots(figsize=(10, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # 1) agente LLM
    _caja(ax, 5, 9.2, 8.6, 1.0,
          "Agente LLM — 5 personalidades\n(Buffett · Wood · Druckenmiller · Burry · Ackman)",
          fc="#e8eef4", fs=9.5)
    _flecha(ax, (5, 8.7), (5, 7.85), "decisión diaria:  signo · tamaño · confianza")
    # 2) tres detectores ortogonales
    det = [(2.0, "RAM\nrégimen (HMM)", REGCOL[2]),
           (5.0, "PSA\ncambio estructural (BOCPD)", REGCOL[1]),
           (8.0, "GSO\nvolatilidad (GARCH)", REGCOL[0])]
    for cx, txt, base in det:
        _caja(ax, cx, 7.1, 2.7, 1.1, txt, fc=base + "33", ec=base, fs=9, lw=1.6)
    # llaves desde el agente a cada detector
    for cx, _t, _b in det:
        _flecha(ax, (5, 7.85), (cx, 7.68))
    ax.text(5, 6.35, "señales  RAM / PSA / GSO  ∈ [0, 1]", ha="center", fontsize=8.5, color="#555")
    for cx, _t, _b in det:
        _flecha(ax, (cx, 6.55), (cx, 5.85) if cx == 5 else ((3.2, 5.4) if cx == 2 else (6.8, 5.4)))
    # 3) los dos supervisores
    _caja(ax, 3.1, 4.7, 3.3, 1.1,
          "M8\nregla de umbrales fijos\n(ex-ante, no aprende)", fc=COL["M8"] + "44", ec=COL["M8"], fs=8.7, lw=1.6)
    _caja(ax, 6.9, 4.7, 3.3, 1.1,
          "M10 / AutoML\nmeta-learner (aprende)\nXGBoost · H2O", fc=COL["M10"] + "44", ec=COL["M10"], fs=8.7, lw=1.6)
    _flecha(ax, (3.1, 4.15), (4.7, 3.25))
    _flecha(ax, (6.9, 4.15), (5.3, 3.25))
    # 4) posición supervisada
    _caja(ax, 5, 2.55, 4.4, 1.0, "posición supervisada  w*\n(warn · reduce · override)",
          fc="#2c3e50", ec="#2c3e50", fs=10.5, tc="white")
    fig.tight_layout()
    _save(fig, "strata_arquitectura")


# ════════════════════════════════════════════════════════════════════════════
# S2 — strata_timeline : protocolo temporal (calibración una vez / OOS desplegable)
# Esquemático, sin datos.
# ════════════════════════════════════════════════════════════════════════════
def strata_timeline():
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    y = 2.0
    # barra de calibración (2000-01 -> 2024-09)
    cal = Rectangle((0.3, y - 0.45), 7.7, 0.9, fc="#9fb6c9", ec="#2c3e50", lw=1.2, zorder=2)
    ax.add_patch(cal)
    ax.text(4.15, y, "Calibración (24 años)\nHMM / GARCH / BOCPD se estiman UNA vez;\numbrales cerrados ex-ante",
            ha="center", va="center", fontsize=8.5, zorder=3)
    # barra OOS (2024-10 -> cierre)
    oos = Rectangle((8.2, y - 0.45), 3.5, 0.9, fc="#27ae6055", ec="#27ae60", lw=1.4, zorder=2)
    ax.add_patch(oos)
    ax.text(9.95, y + 0.62, "Evaluación OOS\n(fuera de muestra)", ha="center", va="bottom",
            fontsize=8.5, color="#1e7a45", zorder=3)
    # tramos internos del OOS: burn-in y walk-forward
    ax.add_patch(Rectangle((8.2, y - 0.45), 0.55, 0.9, fc="#bbbbbb", ec="#27ae60", lw=.8, hatch="//", zorder=3))
    ax.text(8.47, y - 0.85, "burn-in\n150 d\n(no puntúa)", ha="center", va="top", fontsize=6.8, color="#555")
    ax.text(10.25, y, "walk-forward rodante\n(251 d desplegable)\nembargo = 1",
            ha="center", va="center", fontsize=7.3, zorder=4)
    # línea de corte calibración | OOS
    ax.axvline(8.1, color="#c0392b", ls="--", lw=1.4, ymin=0.18, ymax=0.82, zorder=1)
    ax.text(8.1, y + 1.18, "corte de conocimiento\ndel LLM  (2024-09 / 2024-10)",
            ha="center", va="bottom", fontsize=8, color="#c0392b")
    # eje temporal abajo
    ax.annotate("", xy=(11.9, 0.7), xytext=(0.3, 0.7),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
    for xpos, lab in [(0.3, "2000-01"), (8.1, "2024-09"), (11.7, "cierre TFG")]:
        ax.plot([xpos, xpos], [0.62, 0.78], color="#333", lw=1)
        ax.text(xpos, 0.45, lab, ha="center", va="top", fontsize=8)
    ax.text(6, 0.18, "tiempo →", ha="center", fontsize=8, color="#333")
    fig.tight_layout()
    _save(fig, "strata_timeline")


# ════════════════════════════════════════════════════════════════════════════
# S3 — strata_walkforward : esquema rolling-origin (ventana expandible, embargo=1)
# Esquemático, sin datos.
# ════════════════════════════════════════════════════════════════════════════
def strata_walkforward():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_filas = 5
    x0 = 0.5          # origen común de todos los train (izquierda)
    train0 = 3.0      # longitud inicial del train
    crece = 1.15      # crecimiento del train por fila
    avance = 1.15     # avance del test por fila
    gap = 0.25        # huequitos purga y embargo
    test_w = 1.3
    col = {"train": "#2c7fb8", "purga": "#e8a33d", "embargo": "#c0392b", "test": "#27ae60"}
    for i in range(n_filas):
        fila = n_filas - 1 - i  # de abajo (corta) a arriba (larga)
        y = i
        tlen = train0 + crece * i
        xt = x0
        ax.add_patch(Rectangle((xt, y + 0.12), tlen, 0.76, fc=col["train"], ec="k", lw=.5))
        xp = xt + tlen
        ax.add_patch(Rectangle((xp, y + 0.12), gap, 0.76, fc=col["purga"], ec="k", lw=.5))
        xe = xp + gap
        ax.add_patch(Rectangle((xe, y + 0.12), gap, 0.76, fc=col["embargo"], ec="k", lw=.5))
        xs = xe + gap
        ax.add_patch(Rectangle((xs, y + 0.12), test_w, 0.76, fc=col["test"], ec="k", lw=.5))
        ax.text(x0 - 0.2, y + 0.5, f"reentreno {i + 1}", ha="right", va="center", fontsize=8)
    ax.set_xlim(-1.7, x0 + train0 + crece * (n_filas - 1) + 2 * gap + test_w + 0.5)
    ax.set_ylim(-1.3, n_filas + 0.2)
    ax.axis("off")
    # leyenda
    handles = [Rectangle((0, 0), 1, 1, fc=col[k], ec="k") for k in ("train", "purga", "embargo", "test")]
    ax.legend(handles, ["train (se EXPANDE)", "purga", "embargo = 1 día", "test (futuro)"],
              loc="lower left", fontsize=8, ncol=4, bbox_to_anchor=(0.0, -0.04))
    # eje tiempo
    ax.annotate("", xy=(ax.get_xlim()[1] - 0.3, -0.55), xytext=(x0, -0.55),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
    ax.text((x0 + ax.get_xlim()[1]) / 2, -0.85, "tiempo →  (origen rodante; la ventana nunca retrocede: causal)",
            ha="center", fontsize=8.5, color="#333")
    fig.tight_layout()
    _save(fig, "strata_walkforward")


# ════════════════════════════════════════════════════════════════════════════
# S4 — leverage_spy_vs_accion : leverage effect FUERTE (SPY) vs DÉBIL (ROKU)
# scatter r_t vs ΔRV21_{t+1} con recta, dos paneles. RV21 = sqrt(suma r^2 últimos 21).
# Fuente: build_states (en vivo, determinista) + leverage_corr de NAT.
# ════════════════════════════════════════════════════════════════════════════
def leverage_spy_vs_accion():
    from experiments.quant_validation_panel import build_states
    pares = [("SPY", "índice (leverage de Black fuerte)", "#2c7fb8"),
             ("ROKU", "acción volátil (leverage casi nulo)", "#c0392b")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    usado_fallback = False
    for ax, (tk, sub, c) in zip(axes, pares):
        _, _, oos_ret = build_states(tk)
        r = oos_ret.dropna()
        rv = np.sqrt((r ** 2).rolling(21).sum())          # RV21_t
        drv = rv.shift(-1) - rv                            # ΔRV_{t+1}
        df = pd.concat([r.rename("r"), drv.rename("d")], axis=1).dropna()
        x, yv = df["r"].values, df["d"].values
        ax.scatter(x, yv, s=14, color=c, alpha=.55, edgecolor="none")
        b1, b0 = np.polyfit(x, yv, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="k", lw=1.8, ls="--")
        ax.axhline(0, color="#888", lw=.6)
        ax.axvline(0, color="#888", lw=.6)
        lev = NAT[tk]["leverage_corr"]
        ax.set_title(f"{tk} · {sub}", fontsize=9.5)
        ax.set_xlabel("retorno del día  r$_t$")
        ax.set_ylabel("cambio de volatilidad  ΔRV$_{t+1}$")
        ax.xaxis.set_major_formatter(_coma)
        ax.yaxis.set_major_formatter(_coma)
        ax.text(0.04, 0.95, f"leverage corr = {_c(lev, 3)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec=c, alpha=.9))
    fig.tight_layout()
    _save(fig, "leverage_spy_vs_accion")
    return "scatter" if not usado_fallback else "fallback"


# ════════════════════════════════════════════════════════════════════════════
# S5 — naturaleza_bloques : naturaleza de los 10 activos coloreada por cluster
# C0 índices / C1 leverage invertido / C2 volátiles. Orden agrupado por cluster.
# Fuente: NAT (leverage_corr, oos_crisis_frac) + cluster_panel10.json (asignación)
# ════════════════════════════════════════════════════════════════════════════
def naturaleza_bloques():
    # asignación de cluster por activo (labels en el orden de meta.panel)
    panel = CL10["meta"]["panel"]
    labels = CL10["clustering"]["k3"]["kmeans"]["labels"]
    cl = dict(zip(panel, labels))
    CNAME = {0: "C0 · índices", 1: "C1 · leverage invertido", 2: "C2 · volátiles"}
    CCOL = {0: "#2c7fb8", 1: "#e8a33d", 2: "#c0392b"}
    # orden de activos agrupado por cluster (0, 1, 2) y, dentro, por leverage
    order = sorted(PANEL10, key=lambda a: (cl[a], NAT[a]["leverage_corr"]))
    cols = [CCOL[cl[a]] for a in order]
    specs = [("leverage_corr", "Leverage de Black (corr. retorno–vol.; < 0 = estándar)"),
             ("oos_crisis_frac", "Fracción de días en Crisis (OOS)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, (key, ttl) in zip(axes, specs):
        ax.bar(order, [NAT[a][key] for a in order], color=cols, edgecolor="k", lw=.5)
        ax.set_title(ttl, fontsize=9.5)
        ax.tick_params(axis="x", rotation=45, labelsize=8.5)
        ax.yaxis.set_major_formatter(_coma)
        if key == "leverage_corr":
            ax.axhline(0, color="k", lw=.7)
    handles = [Rectangle((0, 0), 1, 1, fc=CCOL[k], ec="k") for k in (0, 1, 2)]
    fig.legend(handles, [CNAME[k] for k in (0, 1, 2)], fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "naturaleza_bloques")


if __name__ == "__main__":
    print("Generando las 15 figuras del Capítulo 4 en", FIGDIR)
    for fn in (f4_1, f4_2, f4_3, f4_4, f4_5, f4_6, f4_7, f4_8,
               f4_9, f4_10, f4_11, f4_12, f4_13, f4_14, f4_15):
        fn()
    print("Hecho: 15 figuras cap4_*.pdf")
    print("\nGenerando las figuras nuevas (cuerpo + anexo)")
    for fn in (n1_anatomia_dia, n2_confusion_spy_6, n3_activacion_panel, n4_naturaleza_panel,
               n5_robustez_calib, n6_robustez_panel, n7_anexo_confusion_panel,
               n8_anexo_ablacion, n9_anexo_shap_dependency, n10_anexo_shap_cuota,
               n11_anexo_shap_rodante, n12_anexo_regimen_direccion, n13_anexo_grupos,
               n14_anexo_psa_gso, n15_anexo_confusion_m10_regimen, n16_anexo_mcnemar,
               n17_anexo_did):
        fn()
    print("Hecho: figuras nuevas cap4_*.pdf")

    print("\nGenerando las 5 figuras de apoyo (arquitectura / protocolo / naturaleza)")
    strata_arquitectura()
    strata_timeline()
    strata_walkforward()
    modo = leverage_spy_vs_accion()
    naturaleza_bloques()
    print(f"Hecho: figuras de apoyo (leverage_spy_vs_accion -> modo '{modo}')")
