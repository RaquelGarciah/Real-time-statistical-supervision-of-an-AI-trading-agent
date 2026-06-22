"""Genera notebooks/STRATA_adaptada.ipynb — ¿se alcanza STRATA-U DESDE el supervisor real (override-C)?

Cuaderno EXPLORATORIO (no canónico). Responde a la pregunta de Raquel: ¿se puede llevar la STRATA
actual (M8, override-C) al rendimiento de STRATA-U moviendo umbrales y haciendo que RAM dispare más,
sin cambiar la lógica de intervención? Lee outputs/experiments/strata_adaptada.json (configs del
StrataSupervisor real, todas las métricas, todos los activos) y lo presenta visual y exhaustivo.

Uso: python notebooks/_build_strata_adaptada.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
def md(t): cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))


md(r"""# STRATA adaptada · ¿se alcanza STRATA-U desde el supervisor real (override-C)?

**Cuaderno EXPLORATORIO — NO canónico.** Trabajo en pruebas (rama `feat/quant-validation-panel`).
No es referencia de la memoria hasta validar.

**La pregunta (Raquel).** Tengo dos estrategias deterministas muy parecidas: **M8** (mi STRATA actual:
override-C, supervisa al agente) y **STRATA-U** (régimen al mando + vol-target). No puedo presentar las
dos. ¿Puedo **adaptar M8 hasta STRATA-U** moviendo umbrales y haciendo que **RAM dispare más** —sin
cambiar la lógica de intervención— y quedarme con **una sola** estrategia parametrizada?

**Diagnóstico (verificado en el código).** El RAM histórico (modo *mismatch*) solo dispara cuando el
agente **contradice** al régimen (su score es la masa de probabilidad del régimen incoherente). Por eso
**bajar el umbral de RAM no basta**: solo intensifica los overrides de contradicción y no aprovecha el
régimen cuando el agente coincide o se abstiene. Para explotar el régimen *en más casos* se añade a RAM
el modo **regime** (score = confianza del régimen direccional dominante `max(P(Calma),P(Crisis))`):
RAM dispara con la **confianza** del régimen, con independencia del signo del agente, y el override-C
—idéntico— impone `regime_sign · bound`.

**Lo que vamos a ver (verificado).** Se alcanza —y se iguala exactamente a *Régimen* / se supera a
STRATA-U— **solo** combinando TRES cosas, no una:
1. RAM en modo **regime** (disparar por confianza, no por contradicción),
2. **signo del régimen causal-expansible** `s_dom` (data-driven que se actualiza en OOS y maneja el
   *prior-flip*) — el signo **estático** de calibración o el leverage **no llegan**,
3. **τ → 0** (el régimen lidera cada día) + GSO `relative` (vol-target).

Con eso, la STRATA adaptada y STRATA-U son **dos puntos de un mismo eje** (la **tasa de intervención**,
gobernada por τ): M8 en el extremo conservador (agente por defecto), STRATA-U en el agresivo (régimen
al mando). **Una sola** estrategia, un dial. Caveat honesto: a τ=0 el agente queda fuera → deja de ser
*supervisión del agente* y pasa a ser *timing de régimen*; y **nadie bate a ZeroR**.""")

code(r"""# --- Bootstrap raíz + carga del JSON auditado ---
import os, sys, json, warnings
from pathlib import Path
_ROOT = Path.cwd()
while not (_ROOT / "config.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
os.chdir(_ROOT); sys.path.insert(0, str(_ROOT)); warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

D = json.load(open("outputs/experiments/strata_adaptada.json"))
PA = D["por_activo"]; ASSETS = list(PA)
MED = D["medias_fair"]; COB = D["cobertura"]; CFGS = D["meta"]["configs"]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
print("activos:", len(ASSETS), "·", ", ".join(ASSETS))
print("ventana:", D["meta"]["ventana"])
print("sizing:", D["meta"]["sizing_fair"])
print("sanity M8 (|acc_adaptada − acc_M8_fair|, debe ser ~0):",
      {a: PA[a]["sanity_m8_absdiff"] for a in ASSETS})""")

# ── §1 Mecanismo ──
md(r"""## §1. El mecanismo: por qué bajar el umbral no basta

| | RAM **mismatch** (histórico, M8) | RAM **regime** (adaptado) |
|---|---|---|
| **Score** | masa de prob. del régimen *incoherente* con el agente | confianza del régimen dominante `max(P(Calma),P(Crisis))` |
| **Dispara cuando** | el agente **contradice** al régimen | el régimen es **confiado** (mire lo que mire el agente) |
| **Agente coincide** | no dispara → pasa el agente | dispara → impone el régimen (re-vol-targeted) |
| **Agente se abstiene (flat)** | no dispara → queda flat | dispara → **toma la posición del régimen** |
| **Override** | `regime_sign · bound` (override-C, idéntico) | `regime_sign · bound` (override-C, idéntico) |

La **lógica de intervención (override-C) no cambia**; cambia *cuándo* se activa RAM y *qué signo* usa.
El dial es **τ** (umbral de RAM): cuanto menor, más días lidera el régimen → mayor tasa de intervención.""")

code(r"""# Configs probadas (todas son el MISMO StrataSupervisor, override-C; solo cambian estos campos)
cfg_tbl = pd.DataFrame(CFGS).T
cfg_tbl.columns = ["RAM mode", "GSO", "τ", "signo régimen"]
cfg_tbl["signo régimen"] = cfg_tbl["signo régimen"].map(
    {"lev": "leverage (hardcoded)", "dd": "data-driven estático (calib.)", "sdom": "causal expansible s_dom"})
cfg_tbl""")

# ── §2 Tabla exhaustiva ──
md(r"""## §2. Tabla EXHAUSTIVA: todos los activos × todas las métricas × todas las estrategias

Sizing **justo** (mismo vol-target `target_vol/σ` para todas) → Sharpe/maxDD/Calmar comparables y
aíslan la **dirección**; la accuracy es independiente del sizing. M5/M8/M10/STRATA-U/Régimen/triviales
+ las configs adaptadas. M10 se lee de `fair_sizing_compare.json` (misma ventana).""")

code(r"""METRICS = ["acc", "sharpe", "maxdd", "calmar"]
STRATS = list(PA[ASSETS[0]]["fair"])
# orden legible: referencias primero, luego configs adaptadas
REF = ["M5", "M8", "M10", "STRATA-U", "Régimen", "B&H", "S&H", "ZeroR"]
ADA = [s for s in STRATS if s not in REF]
ORDER = REF + ADA

def table(metric):
    df = pd.DataFrame({s: {a: PA[a]["fair"][s].get(metric) for a in ASSETS} for s in ORDER}).loc[ASSETS, ORDER]
    df.loc["— MEDIA —"] = {s: MED[s][metric] for s in ORDER}
    return df

for mt, name in [("acc", "ACCURACY direccional"), ("sharpe", "SHARPE"),
                 ("maxdd", "MAX DRAWDOWN"), ("calmar", "CALMAR")]:
    print(f"\n================  {name}  ================")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}", "display.max_columns", None,
                           "display.width", 200):
        print(table(mt))""")

md(r"""### Mapas de calor (accuracy y Sharpe)

Azul = mejor (accuracy > 0,5 / Sharpe > 0). La columna **ZeroR** es el listón duro (no-information-rate);
**Régimen** y **A_reg_sdom_τ00** son las que pelean por la cabeza entre las no triviales.""")

code(r"""SHOW = ["M5", "M8", "M10", "STRATA-U", "Régimen", "ZeroR", "B&H",
        "A_mm_τ30_abs", "A_reg_lev_τ50", "A_reg_dd_τ50", "A_reg_sdom_τ45", "A_reg_sdom_τ00"]
for metric, center, title in [("acc", 0.5, "Accuracy direccional"), ("sharpe", 0.0, "Sharpe")]:
    M = pd.DataFrame({s: {a: PA[a]["fair"][s][metric] for a in ASSETS} for s in SHOW}).loc[ASSETS, SHOW]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    lo, hi = M.min().min(), M.max().max()
    norm = TwoSlopeNorm(vmin=min(lo, center-1e-3), vcenter=center, vmax=max(hi, center+1e-3))
    im = ax.imshow(M.values, cmap="RdYlBu", norm=norm, aspect="auto")
    ax.set_xticks(range(len(SHOW))); ax.set_xticklabels(SHOW, rotation=45, ha="right")
    ax.set_yticks(range(len(ASSETS))); ax.set_yticklabels(ASSETS)
    for i in range(len(ASSETS)):
        for j in range(len(SHOW)):
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{title} por activo (sizing justo vol-target)"); fig.colorbar(im, shrink=0.8)
    plt.tight_layout(); plt.show()""")

# ── §3 El eje τ ──
md(r"""## §3. El eje: **tasa de intervención** (M8 ↔ STRATA-U son dos puntos del mismo dial)

La fracción de días en que la dirección final difiere de la del agente. M8 interviene ~1/3 de los días
(solo contradicciones fuertes); a τ→0 con régimen al mando se interviene ~4/5 (el régimen lidera).
**No son dos estrategias: son el mismo supervisor con la palanca en sitios distintos.**""")

code(r"""iv = pd.DataFrame({a: PA[a]["interv"] for a in ASSETS}).T
iv_mean = iv.mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.barh(range(len(iv_mean)), iv_mean.values, color="#f0a830")
ax.set_yticks(range(len(iv_mean))); ax.set_yticklabels(iv_mean.index)
for i, v in enumerate(iv_mean.values): ax.text(v+0.005, i, f"{v:.0%}", va="center", fontsize=9)
ax.set_xlabel("tasa de intervención media (dirección ≠ agente)")
ax.set_title("Eje de intervención: de M8 (agente por defecto) a régimen al mando")
plt.tight_layout(); plt.show()
print("M8 interviene de media:", f"{iv['A_mm_τ50_abs'].mean():.0%}",
      "· A_reg_sdom_τ00:", f"{iv['A_reg_sdom_τ00'].mean():.0%}")""")

# ── §4 El hallazgo decisivo ──
md(r"""## §4. El hallazgo decisivo: **qué** hace falta para alcanzar STRATA-U

Comparamos en la media del panel (sizing justo) los tres ingredientes, aislados:

- **Bajar solo el umbral** (mismatch, τ 0,5→0,3): no mueve la accuracy (sigue siendo M8).
- **Modo regime con signo estático** (leverage o calibración data-driven): mejora algo, **no llega**.
- **Modo regime + signo causal `s_dom` + τ→0**: **iguala a Régimen y supera a STRATA-U**.

El cuello de botella **no es el umbral**: es (a) que RAM dispare por confianza y (b) que el signo del
régimen sea **causal-expansible** (no el estático de calibración, que se equivoca en los activos con
*prior-flip*).""")

code(r"""key = ["M8", "A_mm_τ30_abs", "A_reg_lev_τ50", "A_reg_dd_τ50",
       "A_reg_sdom_τ45", "A_reg_sdom_τ00", "STRATA-U", "Régimen", "ZeroR"]
lab = ["M8\n(mismatch)", "solo bajar\numbral", "regime\nleverage", "regime\ncalib (dd)",
       "regime sdom\nτ=0.45", "regime sdom\nτ→0", "STRATA-U", "Régimen", "ZeroR\n(trivial)"]
acc = [MED[k]["acc"] for k in key]; sh = [MED[k]["sharpe"] for k in key]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
cols = ["#9e9e9e", "#9e9e9e", "#c9a0dc", "#c9a0dc", "#7fb3d5", "#2c7fb8", "#27ae60", "#c0392b", "#000000"]
for ax, vals, ttl, base in [(axes[0], acc, "Accuracy media", 0.5), (axes[1], sh, "Sharpe medio", 0.0)]:
    ax.bar(range(len(key)), vals, color=cols)
    ax.axhline(base, color="k", lw=0.8, ls=":")
    ax.axhline(MED["STRATA-U"][ttl.split()[0].lower().replace("accuracy","acc")], color="#27ae60", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(range(len(key))); ax.set_xticklabels(lab, fontsize=7.5)
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.3f}" if base==0.5 else f"{v:.2f}", ha="center",
                                         va="bottom" if v>=base else "top", fontsize=7.5)
    ax.set_title(ttl + " (sizing justo)")
plt.tight_layout(); plt.show()

# verificación exacta: A_reg_sdom_τ00 ≡ Régimen (mismo signo cada día + vol-target)
same = all(abs(PA[a]["fair"]["A_reg_sdom_τ00"]["acc"] - PA[a]["fair"]["Régimen"]["acc"]) < 1e-9 for a in ASSETS)
print("A_reg_sdom_τ00 == Régimen en accuracy (todos los activos)?", same,
      "→ a τ=0 el override-C con signo s_dom ES seguir el régimen cada día (con vol-target).")""")

# ── §5 Cobertura ──
md(r"""## §5. Cobertura: ¿en cuántos activos cada config iguala/supera a M8, STRATA-U, M5 y ZeroR?

`acc≥STRATA-U` y `Sh≥STRATA-U` miden si la adaptada **alcanza** STRATA-U por activo. Solo
`A_reg_sdom_τ00` lo hace de forma generalizada (11/13). `acc>ZeroR` y `Sh>ZeroR` confirman el muro:
**ninguna config bate a ZeroR** (0/13 la mejor).""")

code(r"""cob = pd.DataFrame(COB).T
cob = cob[["acc≥M5", "acc≥M8", "sharpe≥M8", "acc≥STRATA-U", "sharpe≥STRATA-U", "acc>ZeroR", "sharpe>ZeroR"]]
print(cob.to_string())""")

# ── §6 Conclusión ──
md(r"""## §6. Conclusión (verificada, sin adornos)

1. **Sí se puede unificar M8 y STRATA-U en UNA sola estrategia parametrizada** del mismo supervisor
   (override-C intacto). El **dial es τ** (la tasa de intervención): M8 = extremo conservador (agente
   por defecto, override solo en contradicción fuerte, ~34 % de intervención); STRATA-U ≈ el extremo
   agresivo (régimen al mando, ~78 %). Esto **elimina la redundancia** de presentar dos cosas casi
   iguales: es **una** con una palanca.

2. **Para llegar a STRATA-U NO basta con mover umbrales.** Hace falta, a la vez: (a) RAM en modo
   **regime** (disparar por confianza), (b) **signo causal-expansible** del régimen `s_dom` (el estático
   de calibración / leverage **no llega**: se equivoca en los activos con *prior-flip*), (c) **τ→0** +
   GSO `relative`. Con las tres, la STRATA adaptada **iguala exactamente a Régimen y supera a STRATA-U
   en 11/13** (acc 0,537 / Sharpe 0,77 vs 0,530 / 0,73).

3. **Caveat honesto (no negociable).** A τ=0 el agente queda **fuera** (la dirección es la del régimen
   cada día): deja de ser *supervisión del agente* y pasa a ser **timing de régimen**. Y como ya
   sabíamos, **nadie bate a ZeroR** (0/13). Así que la elección de marco para la memoria sigue en pie:
   **supervisión del agente → M8** (bate al agente, pooled-significativo); **timing → régimen/STRATA-U
   (= STRATA adaptada con τ→0)**, mejor señal no trivial pero que no supera a las triviales.

4. **Implicación de implementación.** El modo `regime` y el `regime_sign_map` estático ya están en
   `StrataSupervisor` (retrocompatibles, tests verdes). Para que el supervisor produzca *nativamente*
   STRATA-U falta alimentarle el signo `s_dom` **causal por día** (hoy se demuestra en el harness
   vectorizado `experiments/strata_adaptada.py`); es una extensión pequeña, no un método nuevo.""")

nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path("notebooks/STRATA_adaptada.ipynb")
nbf.write(nb, str(out))
print("escrito", out)
