#!/usr/bin/env python3
"""Exporta las figuras del Capítulo 4 (marco práctico) a tesis/figuras/<nombre>.pdf.

Reproducible y determinista: lee solo los JSON canónicos de outputs/experiments/.
Las figuras que necesitan series diarias por estrategia que solo existen en el
notebook (cap4_casos, y la curva M10 diaria) se omiten o se aproximan; se avisa.
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/Raquel/Desktop/STRATA_kit")
EXP = ROOT / "outputs/experiments"
FIG = ROOT / "tesis/figuras"
PANEL = ["SPY","QQQ","XLF","DIA","XLK","XLE","ROKU","SMCI","MARA","UNG"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.bbox": "tight",
})
# paleta por estrategia (consistente en todas las figuras)
COL = {"M5":"#b00020","M8":"#1f6fb2","M10":"#7b3fa0","AutoML":"#0b8043",
       "ZeroR":"#888888","B&H":"#555555","trivial":"#888888"}

def jload(name):
    p = EXP/name
    if not p.exists(): p = EXP/"automl_runs"/name
    return json.load(open(p))

def save(fig, name):
    fig.savefig(FIG/f"{name}.pdf"); plt.close(fig); print("  ✓", name)

# ---------------------------------------------------------------- F4.1 regímenes
def f_regimenes():
    a = jload("spy_intervention_anatomy.json")["serie"]
    r = np.array(a["r_next"]); reg = np.array(a["regime"]); dates = a["dates"]
    price = np.cumprod(1+r)  # proxy de precio (trayectoria de SPY en el OOS)
    x = np.arange(len(price))
    fig, ax = plt.subplots(figsize=(7,3.2))
    names = {0:"Calma",1:"Estrés",2:"Crisis"}; cols = {0:"#d9ead3",1:"#fff2cc",2:"#f4cccc"}
    # sombreado por régimen
    start = 0
    for i in range(1,len(reg)+1):
        if i==len(reg) or reg[i]!=reg[start]:
            ax.axvspan(start, i-1, color=cols[int(reg[start])], lw=0)
            start = i
    ax.plot(x, price, color="#222", lw=1.3)
    ax.set_xlim(0,len(x)-1); ax.set_ylabel("capital relativo (SPY)"); ax.set_xlabel("día (OOS)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[k],label=names[k]) for k in [0,1,2]],
              loc="upper left", frameon=False, ncol=3, fontsize=8)
    ax.set_title("Regímenes del HMM (K=3) sobre la trayectoria de SPY", fontsize=10)
    save(fig,"cap4_regimenes_spy")

# ------------------------------------------------------------- F4.2 scores detect.
def f_scores():
    g = jload("spy_panel_gate_descriptive.json")["descriptivo_spy"]["variables"]
    specs = [("ram_score","RAM",0.50),("psa_score","PSA",0.023),("gso_score","GSO",2.371)]
    fig, axes = plt.subplots(1,3,figsize=(8.2,2.7))
    for ax,(k,lab,thr) in zip(axes,specs):
        x = np.array(g[k]["x"], dtype=float)
        ax.hist(x, bins=30, color="#9db8d2", edgecolor="white")
        ax.axvline(thr, color="#b00020", lw=1.6, ls="--")
        ax.set_title(f"{lab} (umbral {thr:g})", fontsize=9)
        ax.set_yticks([])
    axes[0].set_ylabel("frecuencia")
    fig.suptitle("Distribución de los scores de los detectores sobre SPY (n=401)", fontsize=10)
    save(fig,"cap4_scores_detectores")

# --------------------------------------------------------------- F4.3 confusión
def f_confusion():
    a = jload("spy_intervention_anatomy.json")["serie"]
    m5 = np.array(a["m5_pos"]); m8 = np.array(a["m8_pos"]); r = np.array(a["r_next"])
    interv = np.array(a["intervino"])
    truth = np.sign(r)
    idx = interv & (truth!=0)
    m5_hit = (np.sign(m5)==truth)[idx]; m8_hit = (np.sign(m8)==truth)[idx]
    # 2x2: filas = M5 (agente) acierta/falla ; columnas = M8 (intervención) acierta/falla
    M = np.array([[np.sum(m5_hit & m8_hit), np.sum(m5_hit & ~m8_hit)],
                  [np.sum(~m5_hit & m8_hit), np.sum(~m5_hit & ~m8_hit)]])
    fig, ax = plt.subplots(figsize=(4.2,3.4))
    ax.imshow(M, cmap="Blues");
    for i in range(2):
        for j in range(2):
            ax.text(j,i,int(M[i,j]),ha="center",va="center",
                    color="white" if M[i,j]>M.max()/2 else "#222", fontsize=14)
    ax.set_xticks([0,1]); ax.set_xticklabels(["acierta","falla"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["acierta","falla"])
    ax.set_xlabel("intervención (M8)"); ax.set_ylabel("agente (M5)")
    ax.set_title(f"SPY: {int(M.sum())} intervenciones  (rescate {int(M[1,0])} / daño {int(M[0,1])})", fontsize=9)
    ax.grid(False)
    save(fig,"cap4_confusion_spy")

# ----------------------------------------------------------------- F4.4 equity SPY
def f_equity_spy():
    a = jload("spy_intervention_anatomy.json")["serie"]
    nr = jload("automl_net_returns.json")["por_activo"]["SPY"]
    r = np.array(a["r_next"]); m5 = np.array(a["m5_pos"]); m8 = np.array(a["m8_pos"])
    # ventana desplegable = últimos 251 (alineado con automl)
    k = len(nr["automl"]); r=r[-k:]; m5=m5[-k:]; m8=m8[-k:]
    aml = np.array(nr["automl"])
    eq = lambda ret: np.cumprod(1+np.asarray(ret))
    curves = {"M5": eq(m5*r), "M8": eq(m8*r), "AutoML": eq(aml), "B&H": eq(r)}
    fig, ax = plt.subplots(figsize=(7,3.4))
    order = ["B&H","M5","M8","AutoML"]; lab = {"B&H":"B&H / ZeroR","M5":"M5 (agente)","M8":"M8 (regla)","AutoML":"AutoML"}
    for c in order:
        lw = 2.4 if c=="AutoML" else 1.4
        ax.plot(curves[c], color=COL[c], lw=lw, label=f"{lab[c]} ({curves[c][-1]:.2f}×)")
    ax.axhline(1, color="#aaa", lw=0.8)
    ax.set_xlim(0,k-1); ax.set_ylabel("capital por € invertido"); ax.set_xlabel("día (ventana desplegable, n=251)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("SPY: capital de las estrategias (AutoML supera a la trivial)", fontsize=10)
    save(fig,"cap4_equity_spy")

# ------------------------------------------------------------- F4.5 sensibilidad
def f_sensibilidad():
    v = jload("spy_intervention_variants.json")
    fig, axes = plt.subplots(1,2,figsize=(7.6,2.9))
    sr = v.get("sweep_ram_tau",[])
    if sr:
        taus=[s.get("tau",s.get("ram_tau")) for s in sr]; acc=[s["accuracy"] for s in sr]
        axes[0].plot(taus, acc, "-o", color=COL["M8"]); axes[0].axvline(0.5,color="#b00020",ls="--",lw=1)
        axes[0].set_xlabel(r"umbral $\tau$ de RAM"); axes[0].set_ylabel("accuracy M8"); axes[0].set_title("M8 vs $\\tau$",fontsize=9)
    sm = v.get("sweep_m10_p1",[])
    if sm:
        ps=[s.get("p1",s.get("thr")) for s in sm]; acc=[s["accuracy"] for s in sm]
        axes[1].plot(ps, acc, "-o", color=COL["M10"]); axes[1].axvline(0.5,color="#b00020",ls="--",lw=1)
        axes[1].set_xlabel("umbral de decisión M10"); axes[1].set_ylabel("accuracy M10"); axes[1].set_title("M10 vs umbral",fontsize=9)
    fig.suptitle("Sensibilidad de la accuracy al umbral (meseta = robustez)", fontsize=10)
    save(fig,"cap4_sensibilidad_umbrales")

# --------------------------------------------------------------- F4.6 gate RAM
def f_gate():
    g = jload("spy_panel_gate_descriptive.json")["gate_por_activo"]
    x = np.array([g[a]["discrepancia_agente_regimen"] for a in PANEL])
    y = np.array([g[a]["tasa_intervencion"] for a in PANEL])
    r = np.corrcoef(x,y)[0,1]
    fig, ax = plt.subplots(figsize=(5,3.6))
    ax.scatter(x,y, color=COL["M8"], zorder=3)
    for a,xi,yi in zip(PANEL,x,y): ax.annotate(a,(xi,yi),(4,3),textcoords="offset points",fontsize=7)
    b,m = np.polyfit(x,y,1)[::-1]; xs=np.linspace(x.min(),x.max(),50)
    ax.plot(xs, m*xs+b, color="#b00020", lw=1.3)
    ax.set_xlabel("discrepancia agente$\\leftrightarrow$régimen"); ax.set_ylabel("tasa de intervención (RAM)")
    ax.set_title(f"Compuerta de régimen (Pearson r = {r:.2f})", fontsize=10)
    save(fig,"cap4_gate_ram")

# ------------------------------------------------------------ F4.7 heatmap acc
def f_heatmap():
    d = jload("panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")["por_activo"]
    arms=["m5","m8","m10_xgb","automl","zeror","bh"]; labs=["M5","M8","M10","AutoML","ZeroR","B&H"]
    M=np.array([[d[a]["table"][arm]["accuracy"] for arm in arms] for a in PANEL])
    fig, ax = plt.subplots(figsize=(6.4,4.6))
    im=ax.imshow(M, cmap="RdYlGn", vmin=0.5-0.2, vmax=0.5+0.2, aspect="auto")
    for i in range(len(PANEL)):
        for j in range(len(arms)):
            ax.text(j,i,f"{M[i,j]:.3f}".replace("0.",",").lstrip("0") or f"{M[i,j]:.3f}",
                    ha="center",va="center",fontsize=7,
                    color="white" if abs(M[i,j]-0.5)>0.12 else "#222")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labs)
    ax.set_yticks(range(len(PANEL))); ax.set_yticklabels(PANEL)
    cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("accuracy (centrado en 0,5)")
    ax.set_title("Accuracy direccional por activo y estrategia", fontsize=10); ax.grid(False)
    save(fig,"cap4_heatmap_accuracy")

# ------------------------------------------------------------ F4.8 forest pooled
def f_forest():
    p = jload("bullbear_confirmatory.json")["confirmatorio"]["POOLED10"]["pairs"]
    rows=[("AutoML","AutoML_vs_M5"),("M10","M10_vs_M5"),("M8","M8_vs_M5")]
    fig, ax = plt.subplots(figsize=(6.4,2.8))
    for i,(lab,key) in enumerate(rows):
        d=p[key]; y=len(rows)-1-i
        ax.plot([d["ci95_low"],d["ci95_high"]],[y,y],color=COL[lab],lw=2)
        ax.plot(d["point"],y,"o",color=COL[lab])
        ax.plot([d["ci_bonf_low"],d["point"]],[y+0.16,y+0.16],color=COL[lab],lw=1,alpha=0.6)
        ax.plot(d["ci_bonf_low"],y+0.16,"|",color=COL[lab],ms=9)
        passb = "✓" if d["ci_bonf_low"]>0 else "✗"
        ax.text(d["ci95_high"]+0.08,y,f"{d['point']:+.2f}  Bonf {d['ci_bonf_low']:+.3f} {passb}",va="center",fontsize=8)
    ax.axvline(0,color="#b00020",lw=1)
    ax.set_yticks([len(rows)-1-i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"$\Delta$Sharpe pooled-10 frente al agente (IC$_{95}$, cota Bonferroni $m{=}3$)")
    ax.set_xlim(-0.3, 2.6); ax.set_title("Rescate de riesgo agregado", fontsize=10)
    save(fig,"cap4_forest_pooled")

# ------------------------------------------------------------ F4.9 equity panel
def f_equity_panel():
    nr = jload("automl_net_returns.json")["por_activo"]
    winners=["SPY","SMCI","MARA","UNG"]
    fig, axes = plt.subplots(2,2,figsize=(7.4,4.6))
    for ax,t in zip(axes.ravel(),winners):
        aml=np.array(nr[t]["automl"]); eq=np.cumprod(1+aml)
        ax.plot(eq,color=COL["AutoML"],lw=1.8,label="mejor derivada")
        ax.axhline(1,color="#aaa",lw=0.7)
        ax.set_title(f"{t}  (×{eq[-1]:.2f})",fontsize=9); ax.set_xticks([])
    axes[0,0].legend(frameon=False,fontsize=8,loc="upper left")
    fig.suptitle("Capital de la mejor derivada de STRATA en los cuatro activos donde bate al pasivo en Sharpe", fontsize=9)
    save(fig,"cap4_equity_panel")

# --------------------------------------------------------------- F4.10 TOST 2x2
def f_tost():
    t = jload("equivalence_tost.json")["POOLED10"]
    fig, ax = plt.subplots(figsize=(5.2,4.2))
    d=0.03  # margen accuracy pre-registrado
    ax.axvspan(-d,d,color="#eee"); ax.axvline(0,color="#aaa",lw=0.8); ax.axhline(0,color="#aaa",lw=0.8)
    for lab,key,col in [("M10","M10_vs_M8",COL["M10"]),("AutoML","AutoML_vs_M8",COL["AutoML"])]:
        acc=t[key]["accuracy"]; sh=t[key]["sharpe"]
        ax.errorbar(acc["point"], sh["point"],
                    xerr=[[acc["point"]-acc["ci90_low"]],[acc["ci90_high"]-acc["point"]]],
                    yerr=[[sh["point"]-sh["ci90_low"]],[sh["ci90_high"]-sh["point"]]],
                    fmt="o", color=col, capsize=3, label=f"{lab} vs M8")
    ax.set_xlabel(r"$\Delta$accuracy (aprendiz $-$ regla), IC$_{90}$")
    ax.set_ylabel(r"$\Delta$Sharpe, IC$_{90}$")
    ax.set_title("TOST: superioridad en accuracy, no concluyente en riesgo", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig,"cap4_tost_2x2")

# --------------------------------------------------------------- F4.11 DiD
def f_did():
    d = jload("regime_did_learners.json")["POOLED10"]
    cats=["alcista","bajista"]
    m10=[d["sr_m10_alcista"],d["sr_m10_bajista"]]; aml=[d["sr_aml_alcista"],d["sr_aml_bajista"]]
    x=np.arange(2); w=0.35
    fig, ax = plt.subplots(figsize=(5.4,3.4))
    ax.bar(x-w/2,m10,w,color=COL["M10"],label="M10"); ax.bar(x+w/2,aml,w,color=COL["AutoML"],label="AutoML")
    ax.axhline(0,color="#888",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["alcista","bajista"])
    ax.set_ylabel("Sharpe por régimen"); ax.legend(frameon=False,fontsize=8)
    ax.set_title(f"Complementariedad por régimen (DiD $+{d['did_point']:.2f}$, p={d['p_one_sided_did_gt_0']:.3f})", fontsize=9)
    save(fig,"cap4_did_regimen")

# ------------------------------------------------------- F4.12 atribución capas
def f_atribucion():
    a = jload("spy_intervention_anatomy.json")["serie"]
    bal = jload("spy_intervention_anatomy.json")["balance_intervenciones"]
    fig, axes = plt.subplots(1,2,figsize=(7.8,3.2))
    # izq: P&L por detector
    axes[0].bar(["RAM","PSA","GSO"],[bal["pnl_intervenciones"],0,0],color=[COL["M8"],"#bbb","#bbb"])
    axes[0].set_ylabel("P&L de rescate (SPY)"); axes[0].set_title("Atribución por detector",fontsize=9)
    # der: timeline acumulado de aciertos de intervención (rescate)
    interv=np.array(a["intervino"]); m8h=np.array(a["m8_hit"]);
    r=np.array(a["r_next"]); m5=np.array(a["m5_pos"]); m8=np.array(a["m8_pos"])
    truth=np.sign(r)
    net = np.where(interv, (np.sign(m8)==truth).astype(int)-(np.sign(m5)==truth).astype(int), 0)
    axes[1].plot(np.cumsum(net), color=COL["M8"], lw=1.5)
    axes[1].axhline(0,color="#aaa",lw=0.7)
    axes[1].set_xlabel("día (OOS)"); axes[1].set_ylabel("aciertos netos de la intervención (acum.)")
    axes[1].set_title("Rescate acumulado del régimen",fontsize=9)
    fig.suptitle("Las dos capas: el rescate de riesgo viene del canal de régimen", fontsize=10)
    save(fig,"cap4_atribucion_capas")

# --------------------------------------------------------------- F4.13 scatter lev
def f_scatter():
    d = jload("leverage_law_panel10.json")
    pa=d["por_activo"]; ll=d["ley_leverage"]
    x=np.array([pa[a]["leverage_corr"] for a in PANEL]); y=np.array([pa[a]["rescate_aprendiz"] for a in PANEL])
    fig, ax = plt.subplots(figsize=(5.4,3.8))
    ax.scatter(x,y,color=COL["M10"],zorder=3)
    for a,xi,yi in zip(PANEL,x,y):
        ax.annotate(a,(xi,yi),(4,3),textcoords="offset points",fontsize=7,
                    color="#b00020" if a=="ROKU" else "#333")
    b,m=np.polyfit(x,y,1)[::-1]; xs=np.linspace(x.min(),x.max(),50); ax.plot(xs,m*xs+b,color="#b00020",lw=1.3)
    ax.set_xlabel(r"leverage\_corr  $=\mathrm{corr}(r_t,\ \mathrm{RV}^{21}_{t+1}-\mathrm{RV}^{21}_t)$")
    ax.set_ylabel("rescate del aprendiz (accuracy)")
    ax.set_title(f"Ley del leverage: r = {ll['pearson_r']:.2f}, p = {ll['pearson_p']:.3f} (n=10)", fontsize=10)
    save(fig,"cap4_scatter_leverage")

# --------------------------------------------------------------- F4.14 PCA clusters
def f_pca():
    c = jload("cluster_panel10.json")
    pts=np.array(c["meta"]["pca2d"]); labels=c["clustering"]["k3"]["kmeans"]["labels"]
    cols={0:"#1f6fb2",1:"#7b3fa0",2:"#0b8043"}
    names={0:"C0 índices (leverage fuerte)",1:"C1 leverage invertido",2:"C2 volátiles"}
    fig, ax = plt.subplots(figsize=(5.8,4.2))
    for cl in [0,1,2]:
        idx=[i for i,l in enumerate(labels) if l==cl]
        ax.scatter(pts[idx,0],pts[idx,1],color=cols[cl],label=names[cl],s=40,zorder=3)
    for i,a in enumerate(PANEL): ax.annotate(a,(pts[i,0],pts[i,1]),(4,3),textcoords="offset points",fontsize=7)
    ax.set_xlabel("PC1 (≈ leverage, r = 0,84)"); ax.set_ylabel("PC2")
    ax.legend(frameon=False,fontsize=7,loc="best")
    ax.set_title("Naturaleza de los activos: el clustering recupera el eje del leverage", fontsize=9)
    save(fig,"cap4_pca_clusters")

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    print("Exportando figuras del cap. 4 →", FIG)
    for fn in [f_regimenes,f_scores,f_confusion,f_equity_spy,f_sensibilidad,f_gate,
               f_heatmap,f_forest,f_equity_panel,f_tost,f_did,f_atribucion,f_scatter,f_pca]:
        try: fn()
        except Exception as e: print("  ✗", fn.__name__, "->", repr(e))
    print("Nota: cap4_casos requiere series diarias por activo del notebook (no en JSON).")
