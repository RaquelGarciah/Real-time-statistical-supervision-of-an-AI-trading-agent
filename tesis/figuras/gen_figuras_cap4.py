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
# F4.8 — cap4_forest_pooled : forest plot ΔSharpe pooled-10 + cota Bonferroni
# Fuente: bullbear_confirmatory.json -> confirmatorio.POOLED10
# M8 +0,61 [0,05, 1,22] (no pasa Bonferroni), M10 +1,11 [0,39, 1,84],
# AutoML +1,10 [0,40, 1,85]. (REQUERIDAS los cita como +0,60/+1,12/+1,08 — misma cifra
# a redondeo; uso los valores exactos del JSON, que es la fuente que indica REQUERIDAS.)
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
        bonf = v["ci_bonf_low"]
        pasa = bonf > 0
        col = "#27ae60" if pasa else "#c0392b"
        ax.plot([lo, hi], [i, i], color=col, lw=2.5)
        ax.plot(med, i, "o", color=col, ms=8, zorder=4)
        ax.plot(bonf, i, "|", color="k", ms=22, mew=2, zorder=5)
        ax.text(hi + 0.05, i,
                f"Δ = {_c(med)}  IC95 [{_c(lo)}, {_c(hi)}]  ·  cota Bonferroni {_c(bonf)} "
                + ("(pasa)" if pasa else "(no pasa)"),
                va="center", fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([nice[k] for k in labs])
    ax.set_xlim(-0.5, 3.4)
    ax.set_xlabel("ΔSharpe (mediana, bootstrap estacionario pareado) — pooled-10 (n = 2493)")
    ax.xaxis.set_major_formatter(_coma)
    ax.plot([], [], "|", color="k", ms=12, mew=2, label="cota Bonferroni (1−α/m, m=3)")
    ax.legend(fontsize=8, loc="lower right")
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    # panel izq: ΔSharpe por régimen (barras alcista/bajista)
    ax = axes[0]
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
    # panel der: DiD pooled-10 (M10−AutoML: alcista − bajista) = +1,37
    ax2 = axes[1]
    d = REGDID["POOLED10"]
    col = "#27ae60" if d["ci95_low"] > 0 else "#c0392b"
    ax2.plot([d["ci95_low"], d["ci95_high"]], [0, 0], color=col, lw=3)
    ax2.plot(d["did_point"], 0, "o", color=col, ms=10, zorder=4)
    ax2.axvline(0, color="k", lw=.8)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["DiD pooled-10"])
    ax2.set_ylim(-1, 1)
    ax2.text(d["did_point"], 0.25,
             f"ΔΔSharpe = {_c(d['did_point'])}\nIC95 [{_c(d['ci95_low'])}, {_c(d['ci95_high'])}]  p = {_c(d['p_one_sided_did_gt_0'], 3)}",
             ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("DiD de Sharpe (M10 − AutoML: alcista − bajista)")
    ax2.xaxis.set_major_formatter(_coma)
    fig.tight_layout()
    _save(fig, "cap4_did_regimen")


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


if __name__ == "__main__":
    print("Generando las 15 figuras del Capítulo 4 en", FIGDIR)
    for fn in (f4_1, f4_2, f4_3, f4_4, f4_5, f4_6, f4_7, f4_8,
               f4_9, f4_10, f4_11, f4_12, f4_13, f4_14, f4_15):
        fn()
    print("Hecho: 15 figuras cap4_*.pdf")
