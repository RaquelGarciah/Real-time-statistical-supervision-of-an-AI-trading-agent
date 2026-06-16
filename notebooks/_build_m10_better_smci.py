"""Builder de notebooks/m10_better_smci.ipynb: búsqueda exhaustiva de la mejor M10 desplegable en SMCI.

Documenta, con gráficas y conclusiones honestas, TODO lo probado para mejorar la accuracy de M10 en SMCI:
(A) tuning en validación (fracasa por sobreajuste de selección), (B) configs fijas a priori (techo 0.524),
(C) métodos avanzados (triple-barrier, modelos por régimen, stacking, voting, abstención). Lee tres JSON
(no recomputa): m10_improve_smci, m10_smci_deep, m10_smci_advanced. Claims auditados por @rigor-matematico
(permitidos/prohibidos). Patrón md()/code().
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []


def md(t: str) -> None:
    cells.append(new_markdown_cell(t))


def code(t: str) -> None:
    cells.append(new_code_cell(t))


md(r"""# Mejorar M10 en SMCI — búsqueda exhaustiva (caso de estudio de un activo)

**TFG STRATA · Raquel García.** El proyecto se centra en **un activo** (SMCI). Objetivo: encontrar la mejor
**M10 desplegable** (meta-learner XGBoost sobre features STRATA, walk-forward, solo pasado) que bata a M5
(agente), M8 (regla) y B&H (comprar-y-mantener). **Métrica clave: accuracy direccional**; se enriquece con
**Sharpe** y **equity** (la economía es ilustración, no prueba — CLAUDE.md §4).

**Por qué SMCI:** el tribunal puede tumbar un modelo si una estrategia **trivial** (B&H) lo bate. En activos
alcistas (SPY B&H≈0.57) eso pasa. SMCI tiene **B&H≈0.48** (sin deriva regalada) → benchmark **justo**.

**Disciplina de rigor (toda cifra cumple):**
- **Desplegable:** walk-forward expandible, burn-in 150 d, reentreno mensual (21 d), embargo 5 d, **solo
  pasado**. Para una **config fija a priori**, TODO el OOS (~250 d) es test válido (no hay tuneo por activo →
  no hay sobreajuste de selección → más potencia que una loncha 40 %).
- **Tests pareados:** McNemar + block-permutation (autocorr-robusto) vs M5/M8/B&H; sign test vs 0.5; **Holm**
  sobre la familia método-vs-B&H; **Deflated Sharpe (DSR)** por el nº de métodos probados.
- **Pre-registro** en BITACORA antes de mirar resultados; criterios de éxito/fracaso numéricos.
- **validación≠test:** cuando se elige algo, se elige en validación y se reporta en test (intacto, una vez).

> **Adelanto honesto del veredicto.** Ninguna de las **12 variantes desplegables** bate a B&H en accuracy de
> forma **significativa** (Holm p_adj=1.0; sign vs 0.5 p≥0.49 en todas). El **techo** es **0.524 nominal**
> (base = ensemble), que supera *nominalmente* a M5/M8/B&H. La única mejora robusta es el **ensemble** de 10
> semillas: misma accuracy, mejor Sharpe/equity **nominal** (DSR<0.5 → no significativo tras deflactar). La
> dirección diaria de SMCI es **casi-eficiente** para estos detectores: la contribución es **metodológica** y
> un **negativo honesto pre-registrado**, coherente con la tesis (el leverage effect que hace funcionar a
> STRATA en SPY es débil en un stock individual — CLAUDE.md §3).""")

code(r"""import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
os.chdir(_ROOT)                                   # ejecutar desde la raíz del repo (rutas relativas)

IMP = json.load(open("outputs/experiments/m10_improve_smci.json"))   # A: tuning en validación
DEEP = json.load(open("outputs/experiments/m10_smci_deep.json"))     # B: configs fijas a priori
ADV = json.load(open("outputs/experiments/m10_smci_advanced.json"))  # C: métodos avanzados
SEL = json.load(open("outputs/experiments/m10_smci_select.json"))    # E: selección de burn-in en validación
PAN = json.load(open("outputs/experiments/panel_intervention_scan.json"))  # F: intervención y discrepancia panel

m = ADV["meta"]
print(f"Activo: {m['ticker']}  ·  OOS test desplegable: {m['oos_span'][0]} → {m['oos_span'][1]}  (n={m['n_eval']} días)")
print(f"frac. días alcistas = {m['frac_up']}  ·  B&H accuracy = {ADV['acc_ref']['bh']}  (≈ moneda → benchmark justo)")
print(f"Referencias  accuracy: M5={ADV['acc_ref']['m5']}  M8={ADV['acc_ref']['m8']}  B&H={ADV['acc_ref']['bh']}")
print(f"Referencias  Sharpe:   M5={ADV['sharpe_ref']['m5']}  M8={ADV['sharpe_ref']['m8']}  B&H={ADV['sharpe_ref']['bh']}")""")

# ---------------------------------------------------------------------------------------------
md(r"""## §A · Intento 1 — *tuning* en validación: **FRACASA** (sobreajuste de selección)

Primer intento: partir el OOS en **validación (60 %)** + **test (40 %)** y elegir, **solo en validación**, la
mejor combinación de 5 palancas (umbral≠0.5, selección de features, recencia, ensemble, features de señal
real) sobre un grid de **165 combinaciones**. El walk-forward sigue reentrenando en el test (despliegue real).

**Resultado:** la config ganadora en validación se desploma en test. Es la firma del **sobreajuste de
selección**: maximizar accuracy sobre 165 celdas en ~84 días de validación selecciona **ruido**. Por eso NO
se hace *p-hacking*: el test (intacto, leído una vez) es la única lectura honesta.""")

code(r"""best = IMP["config_elegida"]; t = IMP["test"]["accuracy"]
labels = ["val (ganador,\nmáx 165 combos)", "test (config\nelegida)", "test (M10-base\nsin tuning)"]
vals = [best["acc_val"], t["m10_sel"], t["m10_base"]]
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(labels, vals, color=["#f0a830", "#d65a4a", "#2c7fb7"], edgecolor="black", lw=0.7)
ax.axhline(0.5, color="black", ls="--", lw=1, label="azar (0.5)")
ax.axhline(t["bh"], color="#4caf50", ls=":", lw=1.3, label=f"B&H ({t['bh']})")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylim(0.40, 0.62); ax.set_ylabel("accuracy direccional")
ax.set_title(f"SMCI · el tuning en validación NO transfiere a test (Δ = {IMP['test']['mejora_sobre_base']:+})")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"Config elegida en validación: {best['features']} / recencia {best['recency']} / umbral {best['thr']}")
print(f"  acc validación = {best['acc_val']}  →  acc test = {t['m10_sel']}   (mejora sobre base = {IMP['test']['mejora_sobre_base']:+})")
print(f"  El M10-base SIN tuning rinde {t['m10_base']} en el mismo test → el tuning PERJUDICA. Lección: validación≠test.")""")

# ---------------------------------------------------------------------------------------------
md(r"""## §B · Intento 2 — configs **fijas a priori** sobre todo el OOS: techo **0.524**

Sin tuneo por activo: se fijan 5 configs **motivadas a priori** (no elegidas sobre los datos) y se evalúan
sobre **todo el OOS** (250 d, más potencia). Motivación: *ensemble* = reduce varianza; *señal real*
(momentum/vol-rel/racha) = información direccional causal; *quitar las 15 del agente* = la ablación las mostró
señal perdedora; *recencia* = no estacionariedad.

**Resultado:** el techo de accuracy es **0.524** (base y ensemble); las features de señal real (`aug`,
`strata7+real`) y la recencia **no suben** la accuracy — incluso bajan. `strata7+real` además colapsa en
Sharpe/equity, patrón compatible con **prior-flip** del signo de régimen en SMCI (stock individual).""")

code(r"""cfgs = DEEP["configs"]; orden = ["base", "ens", "aug", "strata_real", "aug_recency"]
orden = [c for c in orden if c in cfgs]
acc = [cfgs[c]["accuracy"] for c in orden]
sr = [cfgs[c]["sharpe_causal"] for c in orden]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
b1 = a1.bar(orden, acc, color="#2c7fb7", edgecolor="black", lw=0.7)
a1.axhline(0.5, color="black", ls="--", lw=1, label="azar")
a1.axhline(DEEP["acc_ref"]["bh"], color="#4caf50", ls=":", lw=1.3, label=f"B&H ({DEEP['acc_ref']['bh']})")
a1.axhline(DEEP["acc_ref"]["m8"], color="#f0a830", ls=":", lw=1.1, label=f"M8 ({DEEP['acc_ref']['m8']})")
for b, v in zip(b1, acc):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
a1.set_ylim(0.46, 0.55); a1.set_ylabel("accuracy"); a1.set_title("Accuracy por config (todo el OOS)")
a1.legend(fontsize=8); a1.tick_params(axis="x", rotation=15)
b2 = a2.bar(orden, sr, color=["#2c7fb7" if s >= 0 else "#d65a4a" for s in sr], edgecolor="black", lw=0.7)
a2.axhline(0, color="black", lw=0.8)
for b, v in zip(b2, sr):
    a2.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=9)
a2.set_ylabel("Sharpe causal (ilustrativo)"); a2.set_title("Sharpe por config — el ensemble destaca")
a2.tick_params(axis="x", rotation=15); plt.tight_layout(); plt.show()
print("Significancia vs B&H (McNemar / Holm-reject) — NINGUNA significativa:")
for c in orden:
    rej = DEEP["holm_vs_bh"].get(f"{c}__vs_bh", {}).get("reject")
    print(f"  {c:12} acc={cfgs[c]['accuracy']}  McNemar vs B&H p={cfgs[c]['tests']['vs_bh']['mcnemar_p']}  Holm-reject={rej}")""")

# ---------------------------------------------------------------------------------------------
md(r"""## §C · Intento 3 — **métodos avanzados** (literatura y reformulaciones)

Se prueban métodos de la literatura y reformulaciones de target/arquitectura, todos **desplegables** y fijos
a priori:

| Método | Idea | Cita |
|---|---|---|
| **triple_barrier** | etiqueta TP=+kσ / SL=−kσ / barrera temporal H=5 (denoising del label) | López de Prado 2018, cap. 3 |
| **regime_models** | 3 XGBoost ponderados por P(estado) del HMM; mezcla p1 = Σ_s P_s·model_s | — |
| **stack_agent** | añadir size del agente (M5) y supervisado (M8) como features de M10 | stacking causal |
| **vote_m5_m10** | acuerdo M5–M10 → mayor confianza (días activos) | — |
| **abst_regime / abst_accord** | abstener menos si el régimen es decisivo / si las 5 personalidades coinciden | — |

**Resultado:** ninguno mejora la accuracy. `triple_barrier` (0.488) y `stack_agent` (0.492) la **degradan**;
`regime_models` (0.50) no aporta. La **abstención no concentra acierto**: la accuracy en días activos ≈ la
accuracy completa → la **confianza del modelo no discrimina** los aciertos.""")

code(r"""met = ADV["metodos"]
orden = ["base", "ens", "triple_barrier", "regime_models", "stack_agent", "vote_m5_m10", "abst_regime", "abst_accord"]
orden = [c for c in orden if c in met]
acc = [met[c]["accuracy"] for c in orden]
cols = ["#2c7fb7" if met[c]["bate_todo_nominal"] else "#9e9e9e" for c in orden]
fig, ax = plt.subplots(figsize=(12, 4.4))
bars = ax.bar(orden, acc, color=cols, edgecolor="black", lw=0.7)
ax.axhline(0.5, color="black", ls="--", lw=1, label="azar (0.5)")
ax.axhline(ADV["acc_ref"]["bh"], color="#4caf50", ls=":", lw=1.3, label=f"B&H ({ADV['acc_ref']['bh']})")
ax.axhline(ADV["acc_ref"]["m8"], color="#f0a830", ls=":", lw=1.1, label=f"M8 ({ADV['acc_ref']['m8']})")
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0.46, 0.55); ax.set_ylabel("accuracy direccional (todo el OOS)")
ax.set_title("SMCI · accuracy de los métodos avanzados (azul = bate a todo nominal)")
ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=20); plt.tight_layout(); plt.show()""")

md(r"""### §C.1 · Tabla de significancia (todos los métodos)

Ninguno rechaza H0 vs B&H bajo Holm; ninguno bate al azar (sign test vs 0.5). El Sharpe se reporta **siempre
con su DSR** (Deflated Sharpe): DSR<0.5 ⇒ la ventaja **no es distinguible de la suerte** tras deflactar por el
nº de métodos probados.""")

code(r"""print(f"{'método':16} {'acc':>6} {'SR':>7} {'equity':>7} {'DSR':>6} {'McN vsBH':>9} {'blkperm':>8} {'Holm':>6} {'sign0.5':>8}")
print("-" * 82)
for c in orden:
    cd = met[c]; tb = cd["tests"]["vs_bh"]; rej = ADV["holm_vs_bh"].get(f"{c}__vs_bh", {}).get("reject")
    print(f"{c:16} {cd['accuracy']:>6} {cd['sharpe_causal']:>+7.2f} {cd['equity_final']:>7} {cd['dsr']:>6} "
          f"{tb['mcnemar_p']:>9} {tb['block_perm_p']:>8} {str(rej):>6} {cd['tests']['vs_azar']['p']:>8}")
print(f"\nBate a todo NOMINAL: {ADV['bate_todo_nominal']}")
print(f"Caso FUERTE (significativo): {ADV['caso_fuerte'] or 'NINGUNO'}")""")

md(r"""### §C.2 · La mejora que **sí** cuenta: ensemble (Sharpe/equity a igual accuracy)

El **ensemble** de 10 semillas mantiene la accuracy de la base (0.524) y mejora Sharpe (0.73→1.23) y equity
(1.32×→1.98×) por **reducción de varianza**. Cumple el criterio de Raquel (a igual accuracy, ganar en
Sharpe/equity cuenta) — pero a nivel **nominal/ilustrativo**: el **DSR<0.5** dice que la ventaja económica
**no sobrevive a la deflación** por multiplicidad. Se conserva como el mejor entregable, con esta etiqueta.

*(`vote_m5_m10`, `abst_regime`, `abst_accord` tienen Sharpe/equity idénticos al ensemble por construcción:
usan la misma posición; solo cambian la cobertura. No son mejoras económicas adicionales.)*""")

code(r"""S = ADV["series"]; dates = np.array(S["dates"], dtype="datetime64[D]")
def equity(nr):
    return np.cumprod(1.0 + np.array(nr))
fig, ax = plt.subplots(figsize=(12, 4.8))
estilos = {"ens": ("#2c7fb7", 2.2, "M10 ensemble"), "base": ("#7fb2d6", 1.4, "M10 base (1 seed)"),
           "m8": ("#f0a830", 1.4, "M8 (regla)"), "m5": ("#9e9e9e", 1.2, "M5 (agente)"),
           "bh": ("#4caf50", 1.6, "B&H (trivial)")}
for k, (c, lw, lab) in estilos.items():
    eq = equity(S[k])
    ax.plot(dates, eq, color=c, lw=lw, label=f"{lab} → {eq[-1]:.2f}×")
ax.axhline(1.0, color="black", lw=0.7, ls="--")
ax.set_ylabel("equity (€1 inicial)"); ax.set_title("SMCI · curvas de equity (causal, lag=1) — el ensemble lidera (DSR<0.5: ilustrativo)")
ax.legend(fontsize=9, loc="upper left"); plt.tight_layout(); plt.show()

ec = met["ens"]
print(f"Ensemble: accuracy {ec['accuracy']} (= base) · Sharpe {ec['sharpe_causal']:+} · equity {ec['equity_final']}× · DSR {ec['dsr']} (<0.5 → no significativo)")""")

md(r"""### §C.3 · Por qué la abstención no ayuda

Si la **confianza** del modelo (|p1−0.5|) discriminara los aciertos, abstenerse en los días dudosos subiría la
accuracy en los días en que **sí** se apuesta. No ocurre: la accuracy en días activos ≈ la completa.""")

code(r"""abst = [c for c in ("abst_regime", "abst_accord", "vote_m5_m10") if c in met]
x = np.arange(len(abst)); w = 0.35
acc_full = [met[c]["accuracy"] for c in abst]
acc_act = [met[c].get("accuracy_activos") for c in abst]
cov = [met[c].get("coverage") for c in abst]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w / 2, acc_full, w, label="accuracy completa", color="#9e9e9e", edgecolor="black", lw=0.6)
ax.bar(x + w / 2, acc_act, w, label="accuracy días activos", color="#2c7fb7", edgecolor="black", lw=0.6)
ax.axhline(0.5, color="black", ls="--", lw=1)
for i, c in enumerate(abst):
    ax.text(i, max(acc_full[i], acc_act[i]) + 0.004, f"cob={cov[i]:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(abst); ax.set_ylim(0.46, 0.56)
ax.set_ylabel("accuracy"); ax.set_title("Abstención: activos ≈ completa → la confianza no discrimina")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------------------------------------
md(r"""## §E · Elegir la mejor estrategia por **selección en validación** (no p-hacking)

Elegir el **burn-in** y la **config** mirando un conjunto de **validación** (y reportar en **test** intacto)
**no es p-hacking** — es selección de hiperparámetros, lo correcto. P-hacking sería elegirlos mirando el test.

**Protocolo:** validación = `[burn-in : día 250]`, test = `[día 250 : fin]` (últimos ~150 días, **intacto**,
se toca una vez). Para cada (config, burn-in) se mide accuracy en validación; se elige el mejor; se reporta en
test. Barrido de burn-ins **{100,…,200}** × {base 1-seed, ensemble 10-seed}.""")

code(r"""grid = SEL["grid"]; best = SEL["elegida"]; BG = SEL["meta"]["burnins"]
fig, ax = plt.subplots(figsize=(10, 4.3))
x = np.arange(len(BG)); w = 0.38
for i, (cfg, col) in enumerate((("base", "#7fb2d6"), ("ens", "#2c7fb7"))):
    vals = [next(g["val_acc"] for g in grid if g["config"] == cfg and g["burnin"] == b) for b in BG]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=f"{cfg}", color=col, edgecolor="black", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.5, color="black", ls="--", lw=1, label="azar (0.5)")
ax.set_xticks(x); ax.set_xticklabels(BG); ax.set_xlabel("burn-in (días de entrenamiento inicial)")
ax.set_ylim(0.40, 0.58); ax.set_ylabel("accuracy en VALIDACIÓN")
ax.set_title(f"SMCI · accuracy en validación por burn-in → elegido: {best['config']}/{best['burnin']}")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"Elegido en validación: {best['config']} / burn-in {best['burnin']}  (val_acc={best['val_acc']}, val_n días variable)")
print("OJO: la validación de burn-in alto tiene MUY pocos días (p.ej. N0=200 → ~50 d) → selección ruidosa.")""")

md(r"""### §E.1 · La estrategia elegida en **test** — y el matiz que la define

La elegida (base, burn-in 200) en test: **accuracy 0.56, Sharpe +2.19, equity 2.55×**, batiendo a M5/M8
(0.533) y a B&H (0.447, equity 0.48×). **Pero** el test resultó un tramo **bajista** (solo 44.7 % de días
suben) y M10 está **58 % corto**. El benchmark trivial **"siempre corto" saca 0.553** ≈ M10 (0.56). Es el
**problema de SPY al revés**: M10 "gana a B&H" sobre todo por estar **corto en un mercado que cae**.""")

code(r"""d = SEL["diagnostico_test"]; rt = SEL["ref_test"]; b = SEL["elegida"]
labels = ["M10\n(elegida)", "M5\n(agente)", "M8\n(regla)", "B&H\n(siempre largo)", "SIEMPRE\nCORTO"]
acc = [b["test_acc"], rt["m5"]["acc"], rt["m8"]["acc"], rt["bh"]["acc"], d["siempre_corto_acc"]]
cols = ["#2c7fb7", "#9e9e9e", "#f0a830", "#4caf50", "#c44e52"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
bars = a1.bar(labels, acc, color=cols, edgecolor="black", lw=0.7)
a1.axhline(0.5, color="black", ls="--", lw=1)
for bb, v in zip(bars, acc):
    a1.text(bb.get_x() + bb.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
a1.set_ylim(0.40, 0.60); a1.set_ylabel("accuracy (test, n=150)")
a1.set_title("Accuracy en test — M10 ≈ 'siempre corto' (mercado bajista)")
# Sharpe + equity
arms = ["M10", "M5", "M8", "B&H"]
sr = [b["test_sharpe"], rt["m5"]["sharpe"], rt["m8"]["sharpe"], rt["bh"]["sharpe"]]
eq = [b["test_equity"], rt["m5"]["equity"], rt["m8"]["equity"], rt["bh"]["equity"]]
xx = np.arange(len(arms)); w = 0.38
a2.bar(xx - w / 2, sr, w, label="Sharpe", color="#2c7fb7", edgecolor="black", lw=0.6)
a2.bar(xx + w / 2, eq, w, label="equity (×)", color="#f0a830", edgecolor="black", lw=0.6)
a2.axhline(0, color="black", lw=0.8); a2.set_xticks(xx); a2.set_xticklabels(arms)
a2.set_title("Sharpe y equity en test (ilustrativo)"); a2.legend(fontsize=8)
for i, (s, e) in enumerate(zip(sr, eq)):
    a2.text(i - w / 2, s + 0.05, f"{s:+.1f}", ha="center", fontsize=8)
    a2.text(i + w / 2, e + 0.05, f"{e:.1f}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()
print(f"M10 elegida: {b['test_acc']} acc · posición {d['frac_corto']:.0%} corto · días alcistas {d['frac_up_test']:.0%}")
print(f"Significancia en test:  vs B&H  McNemar p={d['mcnemar_vs_bh_p']}  block-perm p={d['block_perm_vs_bh_p']}  → BATE a B&H")
print(f"                        vs M5   McNemar p={d['mcnemar_vs_m5_p']}  → NO bate al agente")
print(f"                        vs 0.5  sign test p={d['sign_vs_0.5_p']}  IC95={d['sign_ci95']}  → NO bate a la moneda")""")

md(r"""**Lectura honesta de §E.** Elegir el burn-in en validación es legítimo y la estrategia resultante **bate a
B&H de forma significativa** en el test (block-perm p=0.007). Pero: (i) **no bate al agente** (M5, p=0.71) ni a
una **moneda** (sign test p=0.16); (ii) el test es bajista y M10 está net-short, así que "siempre corto" (0.553)
casi iguala a M10 (0.56) → la ventaja sobre B&H es **el sesgo a corto en un mercado que cae**, no habilidad
direccional fina; (iii) la validación de burn-in alto son ~50 días → selección ruidosa. **Defendible como
"M10 bate al pasivo en el periodo de test"**, siempre que se presente con el benchmark "siempre-corto" al lado;
**no** como "habilidad direccional significativa".""")

# ---------------------------------------------------------------------------------------------
md(r"""## §F · El punto clave: por qué M5, M8 y M10 **no se separan** en SMCI

Mirando las curvas (§C.2) M8 y M5 casi coinciden y M10 no bate al agente. La causa está en **cómo se
posiciona el agente** en SMCI:

- **El agente (M5) está 95 % CORTO** en SMCI (2 % largo, 3 % neutral). Es bajista casi permanente.
- **STRATA interviene solo el 3 % de los días** (M8 ≠ M5) → por eso M8 ≈ M5. *Override-C* dispara cuando el
  agente es **incoherente con el régimen** (que tira a corto en alta vol, *leverage effect*). Pero el agente
  **ya está corto** → coincide con el régimen → **no hay nada que corregir**.
- **M10 también es corto-sesgado** (58 %) → coincide con el agente el 43 % de los días y los discordantes de
  McNemar están equilibrados (b=65, c=75, p=0.45) → **no hay potencia para separarlos**.

En SMCI, **M5, M8 y M10 son la misma apuesta** ("corto SMCI") → ninguno se separa. STRATA rescata donde el
agente va **a contracorriente** del régimen (hay algo que corregir), no donde ya está alineado.""")

code(r"""smci = PAN["por_activo"]["SMCI"]; spy = PAN["por_activo"]["SPY"]
print("                         SPY (caso central)      SMCI")
print(f"  agente corto/largo     {spy['agente_corto']:.0%} corto / {spy['agente_largo']:.0%} largo      {smci['agente_corto']:.0%} corto / {smci['agente_largo']:.0%} largo")
print(f"  discrepancia régimen   {spy['discrepancia_agente_regimen']:.2f}                   {smci['discrepancia_agente_regimen']:.2f}")
print(f"  intervención STRATA    {spy['intervencion_strata']:.0%}                    {smci['intervencion_strata']:.0%}")
print(f"  acc  M5 → M10          {spy['accuracy']['m5']} → {spy['accuracy']['m10']}        {smci['accuracy']['m5']} → {smci['accuracy']['m10']}")
print(f"  McNemar M10 vs M5      p={spy['mcnemar_m10_vs_m5_p']}  (rescate SIG)    p={smci['mcnemar_m10_vs_m5_p']}  (sin rescate)")""")

md(r"""### §F.1 · El panel lo confirma: STRATA aporta donde el agente discrepa del régimen

Barremos los 10 activos midiendo **discrepancia agente↔régimen** (cuánto va el agente a contracorriente),
**intervención de STRATA**, y si **M10 rescata al agente** (Δacc y McNemar). SMCI está al fondo de la
discrepancia → por eso no hay rescate. **SPY es el único con rescate significativo** (M10 vs M5 p=0.0005):
cumple las dos condiciones — el agente discrepa **y** el régimen acierta (leverage effect fuerte en el índice).""")

code(r"""pa = PAN["por_activo"]; tickers = [t for t in PAN["meta"]["panel"] if "error" not in pa[t]]
tickers = sorted(tickers, key=lambda t: -pa[t]["intervencion_strata"])
interv = [pa[t]["intervencion_strata"] for t in tickers]
disc = [pa[t]["discrepancia_agente_regimen"] for t in tickers]
x = np.arange(len(tickers)); w = 0.4
fig, ax = plt.subplots(figsize=(12, 4.3))
ax.bar(x - w / 2, disc, w, label="discrepancia agente↔régimen", color="#2c7fb7", edgecolor="black", lw=0.6)
ax.bar(x + w / 2, interv, w, label="intervención STRATA (M8≠M5)", color="#f0a830", edgecolor="black", lw=0.6)
for i, t in enumerate(tickers):
    if t in ("SMCI", "SPY"):
        ax.annotate(t, (i, max(disc[i], interv[i]) + 0.03), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(tickers); ax.set_ylim(0, 1.05)
ax.set_title("Panel · discrepancia agente↔régimen e intervención STRATA (SMCI al fondo → sin rescate)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# Rescate de M10 sobre el agente, coloreado por significancia
gap = [round(pa[t]["accuracy"]["m10"] - pa[t]["accuracy"]["m5"], 3) for t in tickers]
sig = [pa[t]["mcnemar_m10_vs_m5_p"] < 0.10 for t in tickers]
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(x, gap, color=["#2ca02c" if s else "#bbbbbb" for s in sig], edgecolor="black", lw=0.6)
ax.axhline(0, color="black", lw=0.8)
for i, t in enumerate(tickers):
    ax.text(i, gap[i] + (0.004 if gap[i] >= 0 else -0.012), f"{gap[i]:+.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tickers)
ax.set_ylabel("acc(M10) − acc(M5)"); ax.set_title("Rescate de M10 sobre el agente (verde = McNemar p<0.10 → SOLO SPY)")
plt.tight_layout(); plt.show()
print("Solo SPY tiene rescate significativo (M10 vs M5 p=0.0005). SMCI: Δ=+0.04 nominal, p=0.45 (no sig).")""")

md(r"""**Conclusión de §F.** El barrido **valida el mecanismo del TFG**: STRATA/M10 rescata al agente **solo donde
el agente fila a contracorriente de un régimen que acierta** — y eso ocurre en **SPY** (índice, leverage effect
fuerte; M10 vs M5 p=0.0005), no en SMCI (el agente ya está corto, alineado con el régimen → intervención 3 %,
sin rescate). Es la **explicación honesta** de por qué SPY es el caso central y por qué en SMCI los tres modelos
se confunden. *(Pista para otro activo: ROKU es el stock individual más "tipo-SPY" —alcista, agente 97 % corto,
intervención 88 %— pero su rescate aún no es significativo: M10 vs M5 p=0.13.)*""")

# ---------------------------------------------------------------------------------------------
md(r"""## §D · Conclusiones honestas (claims auditados por @rigor-matematico)

**Lo que se puede afirmar (PERMITIDO):**
- En SMCI (OOS n=250, B&H≈0.48 → benchmark justo), la M10-WF **desplegable base/ensemble** alcanza accuracy
  **0.524**, superior **nominalmente** a M5 (0.484), M8 (0.496) y B&H (0.484).
- **Ninguna de las 12 variantes desplegables** bate a B&H en accuracy de forma **significativa** (McNemar Holm
  p_adj=1.0; sign vs 0.5 p≥0.49 en todas). El **techo** es **0.524 nominal**.
- **Triple-barrier** (López de Prado 2018, cap. 3; embargo=H+1, **sin look-ahead** — verificado) **no mejora**
  la dirección a 1 día (0.488). `regime_models` (0.50) y `stack_agent` (0.492) tampoco; varios la degradan.
- El **ensemble** de 10 semillas preserva la accuracy y mejora **nominalmente** Sharpe (0.73→1.23) y equity
  (1.32×→1.98×) por reducción de varianza → se conserva como entregable (DSR<0.5: ilustrativo, no significativo).
- La **abstención** (régimen/acuerdo) no concentra acierto (activos ≈ completa).

**Lo que NO se puede afirmar (PROHIBIDO):** que M10 bate a M5/M8/B&H **significativamente** en accuracy; que
bate al **azar**; que el ensemble mejora **significativamente** Sharpe/equity (DSR<0.5); reportar cualquier
Sharpe sin su DSR.

**Veredicto de cierre (negativo honesto, pre-registrado).** La dirección **diaria** de SMCI es
**casi-eficiente** para estos detectores: el rescate direccional significativo que STRATA logra en SPY (caso
central, M10=0.539; leverage effect) **no** aparece en un stock individual con leverage débil — limitación ya
prevista en CLAUDE.md §3. La contribución de este cuaderno es **metodológica**: (i) demostración del
**sobreajuste de selección** y de la disciplina **validación≠test** (§A); (ii) un **mapa exhaustivo** de 12
métodos desplegables con su significancia (§B–§C); (iii) el **ensemble** como mejor M10 desplegable, con
ventaja nominal en accuracy y en riesgo-retorno honestamente etiquetada.

**Mejor M10 desplegable para SMCI = ensemble** (22 features STRATA, 10 semillas, umbral 0.5): accuracy 0.524
(> M5/M8/B&H nominal), Sharpe +1.23, equity 1.98×. *Trabajo futuro:* OOS más largo para ganar potencia (el
agente solo existe post-cutoff del LLM → límite duro de muestra).

**Matiz de §E (selección en validación).** Si se elige el burn-in en validación (legítimo), la estrategia
resultante (base, burn-in 200) **bate a B&H de forma significativa** en el tramo de test (block-perm p=0.007).
Pero ese test es **bajista** y M10 está net-short, así que "siempre corto" (0.553) casi iguala a M10 (0.56), y
M10 **no** bate al agente (p=0.71) ni a una moneda (p=0.16). Es el problema de SPY **al revés**: la ventaja
sobre el pasivo es el **sesgo a corto en un mercado que cae**, no habilidad direccional. Defendible como "bate
al pasivo en el periodo", no como "habilidad significativa".""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parent / "m10_better_smci.ipynb"
nbf.write(nb, str(out))
print(f"Notebook escrito: {out}  ({len(cells)} celdas)")
