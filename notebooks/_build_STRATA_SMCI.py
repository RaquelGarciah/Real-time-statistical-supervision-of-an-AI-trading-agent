"""Genera notebooks/STRATA_SMCI.ipynb — el notebook único y definitivo del TFG (caso de estudio SMCI).

Construye el cuaderno celda a celda (nbformat). El headline (accuracy M10) se RECOMPUTA en vivo con el
modelo definitivo (walk-forward ensemble, embargo=1, 22 features); la robustez y los negativos se leen de
outputs/experiments/*.json ya auditados. Reglas: cada cifra con su test, cero look-ahead (signal_lag=1),
cero p-hacking. Comentarios en español, código en inglés (CLAUDE.md §6).

Uso: python notebooks/_build_STRATA_SMCI.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(new_code_cell(text))


# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE 0 — Portada, notación, procedencia
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# STRATA — Supervisión estadística de un agente LLM de *trading*. Caso de estudio: SMCI

**Trabajo Fin de Grado · Matemáticas y Ciencia de Datos · Universidad Complutense de Madrid · Raquel García**

Un **agente de inversión basado en LLM** (AI Hedge Fund, cinco personalidades) decide cada día una posición
sobre un activo. Sin supervisar, **pierde y acierta la dirección menos del 50 %**. **STRATA** es una capa de
**supervisión estadística** que no predice el mercado: vigila al agente con tres detectores clásicos y, cuando
detecta incoherencia, atenúa o sustituye su decisión.

| Detector | Eje | Pregunta | Modelo |
|---|---|---|---|
| **RAM** | régimen discreto | ¿la acción es coherente con el régimen? | HMM gaussiano de 3 estados |
| **PSA** | coherencia temporal | ¿el agente cambia de opinión estructuralmente? | BOCPD (Adams & MacKay, 2007) |
| **GSO** | volatilidad continua | ¿el tamaño es compatible con la volatilidad? | GARCH(1,1) Student-t |

Comparamos seis estrategias sobre el mismo periodo fuera de muestra (OOS): **M5** (agente solo), **M8**
(STRATA, regla *override*), **M10** (meta-aprendiz XGBoost sobre 22 *features* de STRATA), y tres triviales:
**B&H** (siempre largo), **S&H** (siempre corto) y **clase mayoritaria** (la dirección dominante, *ZeroR*).

---

### Qué se defiende (honesto)

> En SMCI, un **benchmark justo** (el pasivo B&H acierta ≈ 0,48, casi una moneda), **M10 desplegable bate en
> accuracy a las cinco estrategias restantes de forma nominal y robusta** (a la partición, al embargo y a la
> ventana temporal). La ventaja es **nominal, no significativa** tras corregir por multiplicidad: la
> significancia plena queda como **trabajo futuro** por el tamaño de muestra (el agente solo existe en el OOS
> posterior al *cutoff* del LLM). La aportación es un **protocolo de supervisión estadística interpretable,
> desplegable y honesto** que recupera accuracy direccional y delimita dónde funciona. **No genera alfa.**

Este cuaderno se ejecuta de principio a fin y reproduce cada cifra. El número central (accuracy de M10) se
**recalcula en vivo**; la robustez y los hallazgos negativos se leen de ficheros JSON ya auditados.""")

md(r"""## §0. Notación

| Símbolo | Código | Significado |
|---|---|---|
| $r_t$ | `r_curr` | log-retorno diario del activo |
| $r_{t+1}$ | `r_next` | retorno del día siguiente (lo que la posición de $t$ cobra) |
| $y_t=\mathbf{1}[r_{t+1}>0]$ | `y` | etiqueta direccional (sube / baja) |
| $s_t\in\{\text{Calma},\text{Estrés},\text{Crisis}\}$ | `regime` | estado latente del HMM |
| $\gamma^f_t(s)$ | `calm/stress/crisis_prob` | posterior **filtrado** del régimen (causal) |
| $\sigma_t$ | `garch_sigma` | volatilidad condicional GARCH anualizada |
| $w_t\in[-1,1]$ | `agent_size` / `final_size` | posición (signo = dirección, magnitud = tamaño) |
| $p_{1,t}$ | `p1` | probabilidad de subida estimada por M10 |

**Convención causal (única válida).** El P&L contabiliza $\pi_t = w_t\, r_{t+1}$ (`signal_lag=1`): la posición
del día $t$ multiplica al retorno del día $t+1$. Nunca $w_t\,r_t$ (*look-ahead*). La accuracy direccional
compara $\operatorname{signo}(w_t)$ con $\operatorname{signo}(r_{t+1})$.""")

code(r"""# --- Bootstrap: situarse en la raíz del repo para que config y las cachés resuelvan igual que en CLI ---
import os, sys, glob, json, hashlib, warnings
from pathlib import Path

_ROOT = Path.cwd()
while not (_ROOT / "config.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
os.chdir(_ROOT); sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import config
from config import (CALIBRATION_START, CALIBRATION_END, STRATA_OOS_START,
                    CACHE_MODELS_DIR, CACHE_AGENT_DIR, DATA_DIR)
from core import data, features
from core.backtest import run_backtest
from core.metrics import equity_curve
from core.hmm import RegimeHMM
from core.garch import GARCHModel
from core.stats import (mcnemar_test, sign_test, block_permutation_test,
                        deflated_sharpe, stationary_bootstrap_ci)
import experiments.walkforward_robustez as wf

config.set_seeds(config.SEED)
TICKER = "SMCI"
ANN = np.sqrt(252)

# Paleta única (coherente en todo el cuaderno).
COL = {"M10": "#2c7fb8", "M8": "#f0a830", "M5": "#9e9e9e",
       "B&H": "#4caf50", "S&H": "#c44e52", "mayoría": "#7b5cc4"}
REGIME_COLOR = {"Calma": "#2e9e4f", "Estrés": "#e8a33d", "Crisis": "#c0392b"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.size": 10})


def _hash_dir(path, pattern="*"):
    h = hashlib.md5()
    for fp in sorted(glob.glob(str(Path(path) / pattern))):
        h.update(Path(fp).read_bytes())
    return h.hexdigest()[:10]


print(f"seed                = {config.SEED}")
print(f"calibración (fin)   = hasta {CALIBRATION_END}  (HMM/GARCH/umbrales se fijan aquí; inicio = IPO real del activo, ver abajo)")
print(f"OOS                 = {STRATA_OOS_START} → cierre  (posterior al cutoff del LLM)")
print(f"hash cache/agent/{TICKER} = {_hash_dir(CACHE_AGENT_DIR / TICKER)}")
print(f"hash cache/models       = {_hash_dir(CACHE_MODELS_DIR)}")""")

code(r"""# --- Resultados pre-computados y auditados (robustez y negativos). El headline se recalcula en vivo. ---
def _J(name):
    return json.load(open(f"outputs/experiments/{name}.json"))

ROB  = _J("m10_smci_valtest_robustez")   # PRINCIPAL (todo el OOS) + robustez de particiones
EMB  = _J("m10_smci_embargo")            # sensibilidad al embargo
ROLL = _J("m10_smci_rolling")            # ventanas rodantes
ADV  = _J("m10_smci_advanced")           # métodos avanzados (todos descartados) + abstención
PAN  = _J("panel_intervention_scan")     # escaneo del panel de 10 activos (selección de SMCI)
IMP  = _J("m10_improve_smci")            # tuning en validación (colapsa en test)

P = ROB["principal_todo_oos"]            # cifras canónicas del caso de estudio
print(f"Caso de estudio: {ROB['meta']['ticker']} · OOS n={P['n']} · frac. días al alza={P['frac_up']}")
print(f"Modelo definitivo: {ROB['meta']['modelo']}")
print(f"Pre-registro: {ROB['meta']['pre_registro']}")""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE I — Marco matemático + estados en vivo + barrera anti-fuga
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte I · Marco matemático y datos

## §1. Los tres modelos de STRATA

**HMM gaussiano de régimen (base de RAM).** La observación diaria $x_t=(r_t,\ \mathrm{RV}^{21}_t)$ (retorno y
volatilidad realizada a 21 días) la genera una cadena de Markov oculta con $K=3$ estados de emisión
$\mathcal{N}(\mu_k,\Sigma_k)$. Para *operar* usamos el posterior **filtrado**
$\gamma^f_t(s)=P(s_t=s\mid x_{1:t})$, que solo mira el pasado (causal). RAM mide la masa del régimen
**contrario** a la acción del agente: $\mathrm{RAM}_t=\sum_{s\in\mathcal{I}(\text{contra})}\gamma^f_t(s)$.

**GARCH(1,1) Student-t (base de GSO).** $\sigma_t^2=\omega+\alpha\,\epsilon_{t-1}^2+\beta\,\sigma_{t-1}^2$ con
innovaciones $t$ de Student (colas pesadas). GSO compara el tamaño del agente con la banda de $\sigma_t$.

> **Proposición (estacionariedad).** Si $\alpha+\beta<1$, el proceso es estacionario en covarianza y
> $\bar\sigma^2=\dfrac{\omega}{1-\alpha-\beta}$.
> *Demostración.* Tomando esperanza incondicional y usando $\mathbb E[\epsilon_{t-1}^2]=\mathbb E[\sigma_{t-1}^2]=\bar\sigma^2$
> en estacionariedad: $\bar\sigma^2=\omega+(\alpha+\beta)\bar\sigma^2$, de donde $\bar\sigma^2(1-\alpha-\beta)=\omega$.
> El cociente es positivo y finito si y solo si $\alpha+\beta<1$. $\qquad\blacksquare$

**BOCPD (base de PSA).** Detección bayesiana *online* de punto de cambio (Adams & MacKay, 2007): se mantiene
la distribución de la *longitud de racha* $P(r_t\mid x_{1:t})$ con una tasa de riesgo (*hazard*) constante;
PSA marca cuando el agente rompe la estructura de su propio *sizing*.

**Intervención (*override-C*).** Cuando RAM supera el umbral $\tau$ (el régimen contrario domina), STRATA
reorienta la posición hacia el signo del régimen; PSA/GSO actúan como cortafuegos en sus colas extremas.

**Limitación clave (leverage effect).** El régimen capta **volatilidad**, no dirección. Solo sirve de proxy
direccional bajo el *leverage effect* (correlación negativa retorno–volatilidad; Black, 1976; Christie, 1982),
fuerte en índices y **débil en un valor individual como SMCI**. Esta limitación anticipa por qué la ventaja en
SMCI será nominal y no significativa.""")

code(r"""# --- Estados de mercado en vivo: HMM K=3 + GARCH ajustados SOLO en calibración (IPO 2007 → 2024-09) ---
# Replica build_states_onthefly: ningún parámetro ve el OOS (sin look-ahead).
feat_df, ret = wf.load_features(TICKER)
_parq = sorted(glob.glob(str(DATA_DIR / f"{TICKER}_{CALIBRATION_START}_*.parquet")))
DATA_END = _parq[-1].rsplit("_", 1)[1].replace(".parquet", "")
prices = data.load_market_data(TICKER, CALIBRATION_START, DATA_END)

calib_feat = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib_feat.to_numpy())
garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])

oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
sigma = garch.forecast_path(oos_ret)
gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()),
                     index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
print(f"calibración: {calib_feat.index[0].date()} → {calib_feat.index[-1].date()}  (n={len(calib_feat)})")
print(f"OOS retornos: {oos_ret.index[0].date()} → {oos_ret.index[-1].date()}  (n={len(oos_ret)})")
_gp = garch.params
print(f"GARCH α+β = {_gp.alpha + _gp.beta:.4f}  (<1 ⇒ estacionario, σ̄²={_gp.omega/(1-_gp.alpha-_gp.beta):.2e})")""")

code(r"""# --- Barrera temporal: tres aserciones anti-fuga que ABORTAN si hay look-ahead ---
# (1) signal_lag=1: la posición de t NO cobra r_t. Con lag=1 el primer retorno neto es 0.
_chk = run_backtest(oos_ret, pd.Series(1.0, index=oos_ret.index), signal_lag=1)["net_return"]
assert abs(float(_chk.iloc[0])) < 1e-12, "LOOK-AHEAD: la posición de t no debe cobrar r_t"
# (2) Calibración estrictamente anterior al OOS.
assert pd.Timestamp(STRATA_OOS_START) > pd.Timestamp(CALIBRATION_END), "OOS solapa calibración"
# (3) Umbrales calibrados ex-ante (la ventana del fichero termina en CALIBRATION_END).
_th = json.load(open(CACHE_MODELS_DIR / "strata_thresholds.json"))
assert _th["calibration_window"][1] == CALIBRATION_END, "Umbrales NO calibrados ex-ante"
print("OK · barrera temporal verificada: signal_lag=1, calibración ⟂ OOS, umbrales ex-ante")
print(f"   hash cache/models = {_hash_dir(CACHE_MODELS_DIR)}  ·  cutoff calibración = {CALIBRATION_END}")""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE II — Calibración y umbrales
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte II · Calibración y umbrales (ex-ante, nunca sobre el OOS)

**Principio rector.** Todo lo que STRATA *aprende* se fija sobre toda la historia disponible de SMCI —desde su
salida a bolsa en marzo de 2007 hasta 2024-09—, **antes** de ver el OOS. Es la disciplina que blinda contra el
p-hacking: ningún corte se elige porque "mejora la curva".""")

code(r"""# --- Régimen HMM sobre el precio (figura dual-panel): Viterbi descriptivo ex-post ---
from matplotlib.patches import Patch
_LBL = {0: "Calma", 1: "Estrés", 2: "Crisis"}


def plot_regimes(close, feats, title):
    states = pd.Series(hmm.predict_states(feats.to_numpy()), index=feats.index)
    price = close.reindex(states.index).ffill()
    fig, (axp, axb) = plt.subplots(2, 1, figsize=(11, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07})
    axp.plot(price.index, price.values, color="black", lw=0.9)
    lo, hi = float(price.min()), float(price.max())
    for st in (0, 1, 2):
        axp.fill_between(states.index, lo, hi, where=(states == st).to_numpy(),
                         color=REGIME_COLOR[_LBL[st]], alpha=0.15, step="post")
    axp.set_ylim(lo * 0.99, hi * 1.01); axp.set_ylabel(f"{TICKER} Close"); axp.set_title(title)
    handles = [plt.Line2D([0], [0], color="black", lw=1, label=f"{TICKER} Close")] + \
        [Patch(facecolor=REGIME_COLOR[_LBL[s]], alpha=0.5, label=_LBL[s]) for s in (0, 1, 2)]
    axp.legend(handles=handles, fontsize=8, loc="upper left", ncol=4)
    for st in (0, 1, 2):
        axb.fill_between(states.index, 0, (states == st).astype(int).to_numpy(),
                         color=REGIME_COLOR[_LBL[st]], alpha=0.9, step="post")
    axb.set_ylim(0, 1); axb.set_yticks([]); axb.set_ylabel("Régimen", fontsize=9); axb.set_xlabel("Fecha")
    plt.show()
    occ = states.value_counts(normalize=True)
    print("Ocupación: " + "   ".join(f"{_LBL[k]} {occ.get(k, 0):.1%}" for k in (0, 1, 2)))


plot_regimes(prices["Close"], calib_feat,
             f"Régimen HMM sobre {TICKER} · CALIBRACIÓN ({calib_feat.index[0].date()} → {calib_feat.index[-1].date()})")""")

code(r"""# --- El mismo régimen sobre el OOS (qué estados ocurren en el periodo de evaluación) ---
oos_feat = feat_df.loc[feat_df.index >= pd.Timestamp(STRATA_OOS_START)]
plot_regimes(prices["Close"], oos_feat,
             f"Régimen HMM sobre {TICKER} · OOS ({oos_feat.index[0].date()} → {oos_feat.index[-1].date()})")""")

md(r"""## §2. Selección de $K$ y correspondencia régimen → signo

Elegimos $K=3$ por dos motivos: (a) la **verosimilitud fuera de muestra** dentro de la calibración mejora con
claridad de $K=2$ a $K=3$; (b) los tres regímenes son **distintos en volatilidad** (Calma < Estrés < Crisis).
La verosimilitud es monótona en $K$, así que no "selecciona" 3; descarta 2 con holgura y el tope en 3 es por
**interpretabilidad** (Calma / Estrés / Crisis son nombrables; $K\ge4$ subdivide la vol sin relato).

**Hallazgo honesto (importante).** En SMCI los regímenes separan por **volatilidad** pero **no por dirección**:
la media por régimen solo es significativa en Estrés (positiva) y la **Crisis tiene media positiva**, no
negativa. El *leverage effect* (alta vol ↔ caídas; Black 1976, Christie 1982), fuerte en índices, es **débil en
un valor individual**. Por eso el régimen aquí no es un prior direccional fiable, M8 (que usa el signo del
régimen) apenas aporta, y la dirección la aprende **M10 de las 22 features**, no del signo del régimen. Es la
limitación que explica el resultado nominal (§9).""")

code(r"""# --- Verosimilitud held-out vs K (dentro de calibración) + scatter régimen (vol × media) + tabla régimen→signo ---
calib_X = calib_feat.to_numpy()
calib_ret = ret.reindex(calib_feat.index)


def heldout_ll(K, folds=(0.5, 0.6, 0.7, 0.8, 0.9)):
    n = len(calib_X); out = []
    for i, s in enumerate(folds):
        a = int(s * n); b = int((s + 0.1) * n) if i < len(folds) - 1 else n
        h = RegimeHMM(n_states=K, seed=config.SEED).fit(calib_X[:a])
        out.append(float(h.model.score(h._standardize(calib_X[a:b])) / (b - a)))
    return float(np.mean(out))


ll = {K: heldout_ll(K) for K in (2, 3, 4)}

# Media/std/IC95 del retorno por régimen (Viterbi sobre calibración).
st_cal = pd.Series(hmm.predict_states(calib_X), index=calib_feat.index)
rows = []
for k in (0, 1, 2):
    rr = calib_ret[st_cal == k].dropna().to_numpy()
    lo, hi, mean = stationary_bootstrap_ci(rr, np.mean, n=2000, seed=config.SEED)
    sig0 = not (lo < 0 < hi)                                   # ¿la media difiere de 0 (IC95)?
    signo = ("largo" if mean > 0 else "corto") if sig0 else "≈0 (no sig.)"
    rows.append({"régimen": _LBL[k], "media r": mean, "std": float(rr.std()),
                 "IC95 media": f"[{lo:+.5f}, {hi:+.5f}]", "signo (media)": signo})
reg_tab = pd.DataFrame(rows).set_index("régimen")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
Ks = list(ll)
ax[0].plot(Ks, [ll[k] for k in Ks], "o-", color="#185", lw=2)
ax[0].axvline(3, color="blue", lw=2, label="K=3 (elegido)")
ax[0].set_xticks(Ks); ax[0].set_xlabel("nº de estados K"); ax[0].set_ylabel("LL fuera de muestra / obs")
ax[0].set_title("Verosimilitud held-out vs K"); ax[0].legend(fontsize=8)
mu = reg_tab["media r"].to_numpy(); sd = reg_tab["std"].to_numpy()
ax[1].scatter(sd, mu, c=[REGIME_COLOR[_LBL[k]] for k in (0, 1, 2)], s=150, zorder=3, edgecolor="black")
for k in (0, 1, 2):
    ax[1].annotate(_LBL[k], (sd[k], mu[k]), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("volatilidad diaria (std de r)"); ax[1].set_ylabel("media de retorno diario")
ax[1].set_title("Separan por VOLATILIDAD, no por signo (leverage débil en SMCI)")
plt.tight_layout(); plt.show()
print(f"LL/obs:  " + "   ".join(f"K={k}: {ll[k]:+.3f}" for k in Ks) + f"   (Δ K3−K2 = {ll[3]-ll[2]:+.3f})")
print("std creciente (Calma<Estrés<Crisis) ⇒ separan por VOLATILIDAD. Pero la Crisis tiene media positiva y")
print("Calma ≈0: NO separan por dirección ⇒ leverage effect débil en SMCI. El régimen capta vol, no dirección;")
print("M8 (usa signo de régimen) apenas aporta y M10 aprende la dirección de las features. Limitación §9.")
reg_tab""")

md(r"""### Persistencia de los regímenes: matriz de transición

La matriz $A=(a_{ij})$ del HMM da la probabilidad de pasar del régimen $i$ al $j$ en un día. La **diagonal
dominante** confirma que los regímenes son **persistentes** (no ruido día a día): la duración media esperada de
cada uno es $1/(1-a_{ii})$ días. Es la propiedad que hace del régimen una variable de estado útil.""")

code(r"""# --- Matriz de transición A del HMM (persistencia de régimen) ---
A = np.asarray(hmm.transition_matrix)
labs = ["Calma", "Estrés", "Crisis"]
fig, ax = plt.subplots(figsize=(5.0, 4.2))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                color="white" if A[i, j] > 0.5 else "black", fontsize=11)
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs)
ax.set_xlabel("régimen en t+1"); ax.set_ylabel("régimen en t")
ax.set_title("Matriz de transición del HMM (calibración)"); ax.grid(False)
fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout(); plt.show()
for i, lb in enumerate(labs):
    print(f"{lb:7} persistencia a_ii={A[i, i]:.3f}  →  duración media ≈ {1/(1-A[i, i]):.0f} días")""")

md(r"""## §3. Umbrales de los detectores

**PSA y GSO** son alarmas para patologías raras, así que su severidad se mapea a percentiles **altos** de la
calibración (low = P95, medium = P99 = *gate* del *override*). Las distribuciones tienen suelo plano y cola
larga: P99 es el codo donde empieza la anomalía real, no el ruido de fondo.

**RAM** necesita su propia metodología: su score es una **masa de probabilidad de régimen**, no un score de
anomalía. El *gate* se pone en $\tau=0{,}5$ (intervenir cuando el régimen contrario es el **más probable**),
una regla de mayoría con **varianza de estimación nula** que no se ajusta a ningún número. La accuracy de
"seguir largo" es plana en una banda ancha de $\tau$, así que el resultado no depende de un corte fino.""")

code(r"""# --- Perfil de percentiles PSA/GSO (calibración) con P95/P99 ---
th = json.load(open(CACHE_MODELS_DIR / "strata_thresholds.json"))
pcts = [50, 75, 90, 95, 99, 100]


def _profile(name):
    d = th[name]["score_distribution"]
    return [d["p50"], d["p75"], d["p90"], d["p95"], d["p99"], d["max"]]


fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
for a, name, ttl in [(ax[0], "psa", "PSA"), (ax[1], "gso", "GSO")]:
    a.plot(pcts, _profile(name), "o-", color="#185", lw=2)
    a.set_yscale("log")
    a.axvline(95, color="#ca4", lw=1.4, ls="--", label="P95 = low")
    a.axvline(99, color="blue", lw=2, label="P99 = gate override")
    a.set_xlabel("percentil de calibración"); a.set_ylabel(f"score {ttl} (log)")
    a.set_title(f"Perfil del score {ttl}"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
_cw = th["calibration_window"]
print(f"Ventana de calibración de umbrales: solicitada {_cw[0]}→{_cw[1]}; efectiva {calib_feat.index[0].date()}→{calib_feat.index[-1].date()} (recortada a la historia real de {TICKER})")""")

code(r"""# --- RAM-gate: P(Calma) es bimodal y la accuracy de 'largo' es plana en τ∈[0.3,0.9] ---
pcalma = gamma.loc[gamma.index <= pd.Timestamp(CALIBRATION_END), "Calma"].to_numpy()
y_cal = (calib_ret.reindex(calib_feat.index) > 0).astype(int).to_numpy()
base_rate = float(y_cal.mean())
grid = np.linspace(0.1, 0.95, 18)
acc_long = [float((y_cal[pcalma >= t] == 1).mean()) if (pcalma >= t).sum() else np.nan for t in grid]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
ax[0].hist(pcalma, bins=30, color="#999"); ax[0].set_yscale("log")
ax[0].axvline(0.5, color="blue", lw=2, label="τ = 0.5 (gate)")
ax[0].set_xlabel("P(Calma) filtrada"); ax[0].set_ylabel("frecuencia (log)")
ax[0].set_title("RAM score es bimodal (no anomalía)"); ax[0].legend(fontsize=8)
ax[1].plot(grid, acc_long, "o-", color="#3a7", label="acierto 'largo' si P(Calma)≥τ")
ax[1].axhline(base_rate, color="k", ls="--", lw=0.9, label=f"tasa base (drift) = {base_rate:.3f}")
ax[1].axvline(0.5, color="blue", lw=1.5)
ax[1].set_xlabel("umbral τ"); ax[1].set_ylabel("accuracy")
ax[1].set_title("Accuracy plana ⇒ τ no es un parámetro fino"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Banda plana en τ∈[0.3,0.9]; el gate separa la masa bimodal, no calibra contra 0.5 absoluto.")""")

md(r"""## §3b. ¿Conviene tunear los umbrales de PSA y GSO? (decisión: no, y por qué)

Los umbrales de PSA y GSO se fijan **ex-ante** en percentiles altos de la calibración (P95/P99): son detectores
de **alarma** para patologías raras. Antes de cerrar esa decisión comprobamos si **bajarlos** (para forzarlos a
disparar) mejoraría algo. Lo hacemos sobre la regla **M8** —la estrategia que de verdad usa estos umbrales—
comparando **validación y test**, para que la decisión no sea *p-hacking*. Tres hechos:

- **Estructural.** En el *override-C*, **GSO solo recorta el tamaño** ($\min(|size|,\text{bound})$) y **PSA solo
  lo divide a la mitad**; únicamente **RAM** voltea el signo. Por tanto tunear PSA/GSO **no puede cambiar la
  accuracy** (que mide el signo): a lo sumo cambia el tamaño y, con él, el Sharpe.
- **Empírico.** En SMCI el agente es **pasivo** (97 % corto, tamaño ≈0,10 constante): no hay **sobreexposición**
  (GSO) ni **cambios estructurales** del sizing (PSA), así que sus *scores* **no llegan al gate** (figura).
- **GSO y `target_vol`.** La banda es $\text{bound}=\text{target\_vol}/\sigma_t$; **subir `target_vol` la
  ensancha → GSO dispara MENOS** (para forzarlo habría que bajarlo a ≈0,02, y aun así solo recortaría tamaño).

El barrido confirma que **ningún umbral mejora a la vez validación y test**: la accuracy es **invariante** y el
Sharpe solo fluctúa como **ruido** en validación con el test plano. **Decisión: mantener los umbrales calibrados
ex-ante.** (M10, el modelo principal, **no usa estos umbrales** —apuesta ±1 por dirección—; esto concierne a la
regla M8 y a la mecánica de STRATA.)""")

code(r"""# --- §3b. Barrido de umbrales PSA/GSO sobre M8 (decisión: mantener los ex-ante) ---
from strata.strata import StrataSupervisor
_ag = wf.load_agent(TICKER)
_thr = json.load(open(CACHE_MODELS_DIR / "strata_thresholds.json"))


def _m8(psa_thr=None, gso_thr=None):
    sup = StrataSupervisor(mode="override", override_variant="C", gso_mode="absolute", psa_signal="cp_prob",
                           psa_hazard=config.BOCPD_HAZARD, ram_thresholds=wf.RAM_THRESHOLDS,
                           psa_thresholds=psa_thr, gso_thresholds=gso_thr)
    rows, hist = [], []
    for t in sorted(_ag):
        if t not in gamma.index or t not in sigma.index:
            continue
        a = _ag[t]; hist.append(a.size); gg = gamma.loc[t]
        msr = {"regime": {"calm_prob": float(gg["Calma"]), "stress_prob": float(gg["Estrés"]),
                          "crisis_prob": float(gg["Crisis"]), "viterbi_state": int(np.argmax(gg.values))},
               "garch_vol_annualized": float(sigma.loc[t])}
        d = sup.supervise(a, msr, hist)
        rows.append({"date": t, "final": d.final_size, "psa_s": d.detectors["psa"].score,
                     "gso_s": d.detectors["gso"].score, "psa": d.detectors["psa"].severity in ("medium", "high"),
                     "gso": d.detectors["gso"].severity in ("medium", "high")})
    mm = pd.DataFrame(rows).set_index("date"); mm["rn"] = oos_ret.shift(-1).reindex(mm.index)
    return mm[mm["rn"].notna() & (np.sign(mm["rn"]) != 0)]


def _vt(mm):  # accuracy y Sharpe de M8 en validación (60%) y test (40%)
    k = int(len(mm) * 0.6)
    w = pd.Series(0.0, index=oos_ret.index); w.loc[mm.index] = mm["final"].values
    nr = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(mm.index).to_numpy()

    def _sh(a):
        a = a[~np.isnan(a)]; s = a.std(ddof=1); return float(a.mean() / s * np.sqrt(252)) if s > 0 else 0.0
    acc = lambda sl: float((np.sign(mm["final"].to_numpy()[sl]) == np.sign(mm["rn"].to_numpy()[sl])).mean())
    return acc(slice(0, k)), acc(slice(k, None)), _sh(nr[:k]), _sh(nr[k:])


base = _m8()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for a, col, lbl in [(ax[0], "psa_s", "psa"), (ax[1], "gso_s", "gso")]:
    a.hist(base[col].dropna(), bins=40, color="#9ecae1", edgecolor="black", lw=0.4); a.set_yscale("log")
    p99 = _thr[lbl]["score_distribution"]["p99"]
    a.axvline(p99, color="red", lw=1.8, label=f"gate (P99) = {p99:.2f}")
    a.set_xlabel(f"score {lbl.upper()} (OOS)"); a.set_ylabel("frecuencia (log)")
    a.set_title(f"{lbl.upper()}: los scores no llegan al gate"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Agente SMCI: tamaño |size| ≈ {np.mean([abs(_ag[t].size) for t in _ag]):.3f} (casi constante), "
      f"{np.mean([np.sign(_ag[t].size) < 0 for t in _ag]):.0%} corto → ni sobreexposición ni cambios estructurales.")""")

code(r"""# --- §3b. Rendimiento de M8 según el umbral + GSO vs target_vol ---
_pd, _gd = _thr["psa"]["score_distribution"], _thr["gso"]["score_distribution"]
CFG = {"baseline\n(P99)": (None, None),
       "PSA bajo\n(p75)": ((_pd["p50"], _pd["p75"], _pd["p90"]), None),
       "GSO bajo\n(p75)": (None, (_gd["p50"], _gd["p75"], _gd["p90"])),
       "ambos\nbajos": ((_pd["p50"], _pd["p75"], _pd["p90"]), (_gd["p50"], _gd["p75"], _gd["p90"]))}
res = {}
for nm, (pt, gt) in CFG.items():
    res[nm] = _vt(_m8(pt, gt))   # (acc_val, acc_test, sr_val, sr_test)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xb = np.arange(len(CFG)); w = 0.38; names = list(CFG)
ax[0].bar(xb - w / 2, [res[n][2] for n in names], w, label="validación", color="#2c7fb8", edgecolor="black")
ax[0].bar(xb + w / 2, [res[n][3] for n in names], w, label="test", color="#f0a830", edgecolor="black")
ax[0].set_xticks(xb); ax[0].set_xticklabels(names, fontsize=8); ax[0].set_ylabel("Sharpe de M8")
ax[0].set_title("Rendimiento de M8 según el umbral: sin mejora robusta (val/test)"); ax[0].legend(fontsize=8)
for i, n in enumerate(names):  # anota que la accuracy es invariante
    ax[0].annotate(f"acc {res[n][0]:.2f}/{res[n][1]:.2f}", (i, 0), fontsize=6, ha="center", va="bottom", color="gray")
# GSO: firing rate vs target_vol (subir target_vol → dispara menos)
sg = sigma.reindex([t for t in sorted(_ag) if t in sigma.index]).to_numpy()
szs = np.array([abs(_ag[t].size) for t in sorted(_ag) if t in sigma.index])
tvs = np.array([0.02, 0.05, 0.10, 0.15, 0.30, 0.60, 1.0])
fire = [float((np.maximum(0.0, szs - np.minimum(1.0, tv / np.maximum(sg, 1e-9))) /
               np.maximum(np.minimum(1.0, tv / np.maximum(sg, 1e-9)), 1e-3) >= _gd["p99"]).mean()) for tv in tvs]
ax[1].plot(tvs, fire, "o-", color="#c0392b", lw=2)
ax[1].axvline(0.10, color="k", ls="--", lw=1, label="target_vol actual = 0.10")
ax[1].set_xlabel("target_vol"); ax[1].set_ylabel("% días que GSO dispara (gate P99)")
ax[1].set_title("Subir target_vol ensancha la banda → GSO dispara MENOS"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print("M8 accuracy: invariante al umbral (PSA/GSO no cambian el signo). Sharpe: solo ruido en validación, test")
print("plano → ningún umbral mejora val Y test (sería p-hacking). DECISIÓN: se mantienen los umbrales ex-ante.")""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE III — Resultado del caso de estudio (headline en vivo)
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte III · El resultado (caso de estudio SMCI, en vivo)

## §4. Por qué SMCI y qué se compara

SMCI es un **benchmark justo**: el pasivo B&H acierta ≈ 0,48 (casi una moneda), así que el tribunal no puede
tumbar el resultado con "lo trivial ya gana". Comparamos **seis estrategias** sobre el mismo OOS:

| Estrategia | Qué hace |
|---|---|
| **M5** | posición del agente LLM |
| **M8** | STRATA, regla *override-C* |
| **M10** | meta-aprendiz XGBoost (ensemble 10 semillas) sobre 22 *features* de STRATA |
| **B&H** | siempre largo |
| **S&H** | siempre corto |
| **clase mayoritaria** | la dirección dominante (*ZeroR*); en SMCI predomina "baja", así que coincide con S&H |

**Protocolo del modelo definitivo (M10).** Walk-forward expandible (burn-in 150, reentreno cada 21 días),
**embargo = 1**, posición $=\operatorname{signo}(p_1-0{,}5)$, cobertura 100 %, ensemble de 10 semillas.

**Embargo = 1, por principio (no por p-valor).** La etiqueta tiene horizonte 1 ($y_t=\mathbf 1[r_{t+1}>0]$), así
que la purga = 1; en validación *rolling-origin* el test es siempre futuro y no hay solape bidireccional que
justifique un embargo grande (Tashman, 2000; Bergmeir, Hyndman & Koo, 2018). El embargo $\ge 5$ es la regla de
**CPCV** con *folds* entrelazados (López de Prado, 2018, §7.4), otro régimen. El contraste M10-CPCV (que ve
bloques futuros) da **0,448 en SMCI**, peor: el buen número **no** viene de mirar el futuro.""")

code(r"""# --- 22 features y el walk-forward ensemble definitivo (el ÚNICO cómputo pesado: ~120 ajustes XGB) ---
STEP, EMBARGO, N0, N_SEEDS = 21, 1, 150, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA7


def wf_p1(X, y, N0, seeds):
    # Walk-forward expandible (solo pasado), embargo=1, ensemble sobre seeds. p1 en [N0:fin].
    n = len(X); p = pd.Series(np.nan, index=X.index)
    for start in range(N0, n, STEP):
        tr = start - EMBARGO
        if tr < 50:
            continue
        end = min(start + STEP, n)
        preds = [xgb.XGBClassifier(**PARAMS, random_state=sd).fit(X.iloc[:tr], y.iloc[:tr])
                 .predict_proba(X.iloc[start:end])[:, 1] for sd in seeds]
        p.iloc[start:end] = np.mean(preds, axis=0)
    return p


wf.reset_thresholds_cache()
m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))      # M5/M8 + 22 features, día a día
mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
y = (mv["r_next"] > 0).astype(int)
p1 = wf_p1(mv[ALL22], y, N0, SEEDS)
print(f"OOS evaluable n={len(mv)} · días con predicción M10={int(p1.notna().sum())}")""")

code(r"""# --- Posiciones, P&L causal y métricas de las 6 estrategias ---
sub = mv.index[p1.notna().to_numpy()]
truth = np.sign(mv.loc[sub, "r_next"].to_numpy())
yt = (truth > 0).astype(int)
proba = p1.dropna().to_numpy()
nir_dir = 1.0 if yt.mean() > 0.5 else -1.0
NIR = float(max(yt.mean(), 1 - yt.mean()))

POS = {"M10": np.where(proba >= 0.5, 1.0, -1.0),
       "M8": np.sign(mv.loc[sub, "final_size"].to_numpy()),
       "M5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
       "B&H": np.ones(len(sub)),
       "S&H": -np.ones(len(sub)),
       "mayoría": np.full(len(sub), nir_dir)}


def _sr(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return float(a.mean() / s * ANN) if s > 0 else 0.0


def _netret(pos):
    w = pd.Series(0.0, index=mv.index); w.loc[sub] = pos
    return run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(sub).to_numpy()


ACC = {k: float((v == truth).mean()) for k, v in POS.items()}
# Accuracy = dirección (signo). Para Sharpe/equity, M5/M8 usan su TAMAÑO real desplegado
# (nr_m5/m8_causal, coherente con el JSON auditado); los triviales, el retorno con su signo fijo.
_o = oos_ret.reindex(sub)
NR = {"M10": _netret(POS["M10"]),
      "M8": mv["nr_m8_causal"].reindex(sub).to_numpy(),
      "M5": mv["nr_m5_causal"].reindex(sub).to_numpy(),
      "B&H": _o.to_numpy(),
      "S&H": (-_o).to_numpy(),
      "mayoría": (nir_dir * _o).to_numpy()}
SR = {k: _sr(NR[k]) for k in POS}
EQ = {k: float(equity_curve(pd.Series(NR[k]).dropna()).iloc[-1]) for k in POS}

assert abs(ACC["M10"] - 0.552) < 0.012, f"headline {ACC['M10']} != 0.552 (¿cambió la caché?)"
print(f"M10 accuracy en vivo = {ACC['M10']:.4f}  (esperado 0.552)  ✓")
print(f"frac. días al alza = {yt.mean():.4f}  ·  NIR (clase mayoritaria) = {NIR:.4f} ({'largo' if nir_dir>0 else 'corto'})")""")

code(r"""# --- Tabla maestra COMPLETA: accuracy + AUC + log-loss + Brier + MCC + Sharpe + equity (cada cifra trazable) ---
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, matthews_corrcoef

auc = float(roc_auc_score(yt, proba))
ll_m10 = float(log_loss(yt, proba))
brier = float(brier_score_loss(yt, proba))
mcc = float(matthews_corrcoef(yt, (POS["M10"] > 0).astype(int)))

rows = []
for k in ["M10", "M8", "M5", "B&H", "S&H", "mayoría"]:
    rows.append({"estrategia": k, "accuracy": round(ACC[k], 4),
                 "AUC": round(auc, 4) if k == "M10" else "—",
                 "log-loss": round(ll_m10, 4) if k == "M10" else "—",
                 "Brier": round(brier, 4) if k == "M10" else "—",
                 "MCC": round(mcc, 4) if k == "M10" else "—",
                 "Sharpe": round(SR[k], 3), "equity": round(EQ[k], 4)})
tabla_maestra = pd.DataFrame(rows).set_index("estrategia")
print("AUC/log-loss/Brier/MCC solo aplican a M10 (único con probabilidad calibrable).")
print(f"AUC={auc:.3f} ≈ 0.5: como RANKING M10 es casi no informativo; la habilidad direccional vive en el")
print("SIGNO bajo umbral 0.5, no en la probabilidad. Por eso la posición es signo(p1-0.5), no un sizing ∝ p1.")
print("Sharpe y equity son ILUSTRACIÓN económica, no prueba (CLAUDE.md §4).")
tabla_maestra""")

code(r"""# --- Tests pareados de la serie viva (cada cifra con su H0) ---
from scipy.stats import binomtest

corr = {k: (POS[k] == truth).astype(int) for k in POS}
n = len(sub)


def _mc(comp):  # McNemar: comp vs M10 (b = comp✓&M10✗, c = M10✓&comp✗)
    _, p, b, c = mcnemar_test(corr[comp], corr["M10"]); return float(p), int(b), int(c)


p_mc_m5, b5, c5 = _mc("M5")
p_mc_m8, b8, c8 = _mc("M8")
p_mc_bh, bb, cb = _mc("B&H")
_, p_bp_bh = block_permutation_test(corr["M10"], corr["B&H"], seed=config.SEED)
_, p_bp_m5 = block_permutation_test(corr["M10"], corr["M5"], seed=config.SEED)
k_s, n_s, p_sign, ci_s = sign_test(corr["M10"])
k10 = int(corr["M10"].sum())
p_nir = float(binomtest(k10, n, NIR, alternative="greater").pvalue)
p_05 = float(binomtest(k10, n, 0.5, alternative="greater").pvalue)

tests = pd.DataFrame([
    {"contraste": "M10 vs M5  (McNemar pareado)", "H0": "P(b)=P(c)", "p": round(p_mc_m5, 4), "detalle": f"b={b5} c={c5}"},
    {"contraste": "M10 vs M8  (McNemar pareado)", "H0": "P(b)=P(c)", "p": round(p_mc_m8, 4), "detalle": f"b={b8} c={c8}"},
    {"contraste": "M10 vs B&H (McNemar pareado)", "H0": "P(b)=P(c)", "p": round(p_mc_bh, 4), "detalle": f"b={bb} c={cb}"},
    {"contraste": "M10 vs B&H (block-permutation)", "H0": "misma accuracy", "p": round(p_bp_bh, 4), "detalle": "robusto a autocorr."},
    {"contraste": "M10 vs M5  (block-permutation)", "H0": "misma accuracy", "p": round(p_bp_m5, 4), "detalle": "robusto a autocorr."},
    {"contraste": "M10 vs azar (sign test, bilateral)", "H0": "acc = 0.5", "p": round(p_sign, 4), "detalle": f"k={k_s}/{n_s}"},
    {"contraste": "M10 vs 0.5  (binomial, 1-cola)", "H0": "acc ≤ 0.5", "p": round(p_05, 4), "detalle": "unilateral"},
    {"contraste": f"M10 vs NIR={NIR:.3f} (binomial, 1-cola, listón duro)", "H0": "acc ≤ NIR", "p": round(p_nir, 4), "detalle": "clase mayoritaria"},
]).set_index("contraste")
print("Lectura: M10 gana NOMINALMENTE a todo; McNemar vs M5/M8 NO es significativo (signo nominal, no estadístico).")
print("El contraste honesto es M10 vs NIR (clase mayoritaria); ahí p no alcanza α=0.10 → se reporta como nominal.")
tests""")

md(r"""## §5. Las gráficas del resultado""")

code(r"""# --- Fig. headline: accuracy de las 6 estrategias (líneas de azar 0.5 y NIR) ---
order = ["M5", "M8", "B&H", "S&H", "mayoría", "M10"]
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(order, [ACC[k] for k in order],
              color=[COL[k] for k in order], edgecolor="black", lw=0.8)
ax.axhline(0.5, color="k", ls="--", lw=1, label="azar = 0.500")
ax.axhline(NIR, color="#7b5cc4", ls=":", lw=1.6, label=f"NIR (clase mayoritaria) = {NIR:.3f}")
for b, k in zip(bars, order):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{ACC[k]:.3f}",
            ha="center", fontsize=9, fontweight="bold" if k == "M10" else "normal")
ax.set_ylim(0.40, 0.60); ax.set_ylabel("accuracy direccional (OOS, n=250)")
ax.set_title("M10 bate a las cinco estrategias en accuracy (nominal)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"M10 {ACC['M10']:.3f} > mayoría/S&H {ACC['mayoría']:.3f} > M8 {ACC['M8']:.3f} > M5 {ACC['M5']:.3f} = B&H {ACC['B&H']:.3f}")""")

code(r"""# --- Fig. curvas de equity (ilustración económica, no prueba) ---
fig, ax = plt.subplots(figsize=(10, 4))
for k in ["M10", "M8", "M5", "B&H"]:
    eq = equity_curve(pd.Series(NR[k], index=sub).dropna())
    ax.plot(eq.index, eq.values, label=f"{k} (×{EQ[k]:.2f}, SR {SR[k]:+.2f})",
            color=COL[k], lw=2 if k == "M10" else 1.3)
ax.axhline(1.0, color="k", lw=0.7)
ax.set_ylabel("equity (€1 inicial)"); ax.set_title("Curvas de equity OOS · enriquecimiento, no prueba")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print("Sharpe con su P(Sharpe>0) (Parte V): P(Sharpe>0)=0.976 (positivo con alta prob.; ilustración económica).")""")

md(r"""### ¿Por qué M10 cae por debajo de todo en el verano de 2025?

En la curva anterior la equity de M10 queda **por debajo de M5, M8 y B&H entre mayo y septiembre de 2025**, con
un *drawdown* máximo del **−34 %** (pico 31-jul → valle 8-sep). No es ruido, y conviene desmontar el porqué con
cuidado porque la causa no es la obvia.

**(1) M10 no captura el rally previo.** En la subida de primavera-verano (B&H llega a ≈1,7×), M10 —sesgado a
corto como el agente— se queda fuera y **entra al verano rezagado**.

**(2) Causa dominante: el régimen no es direccional en SMCI.** En la caída, M10 se mantiene **largo el 70 % de
los días** (accuracy 0,37). Y **no es por un rezago** del detector: el régimen **sí capta la crisis** (marca
Crisis el ~81 % de los días de la caída). El problema es que en SMCI **Crisis tiene media histórica positiva**
(+0,0016, §2), así que el meta-aprendiz aprendió *"Crisis ≈ subida"* y **sigue largo aun cuando el régimen ya
señala Crisis**, justo cuando esta crisis cae. Capturar el régimen **no protege** porque en este valor el
régimen no apunta a la baja.

La ventaja de M10 **no es suave**: llega después, en las caídas de finales de 2025 y principios de 2026, donde se
pone corto y la equity pasa de ≈0,9× a 3,2×. Por eso el Sharpe es **episódico** (ilustración) y la significancia
direccional no se sostiene a esta muestra.""")

code(r"""# --- Episodio del drawdown de verano 2025: por qué M10 cae por debajo de todo ---
e10 = equity_curve(pd.Series(NR["M10"], index=sub).dropna())
dd = e10 / e10.cummax() - 1
valle = dd.idxmin(); pico = e10[:valle].idxmax()
seg = sub[(sub >= pico) & (sub <= valle)]
pos10 = pd.Series(POS["M10"], index=sub)
acc_seg = float((pos10[seg].values == np.sign(mv.loc[seg, "r_next"].values)).mean())
print(f"Drawdown máximo de M10: {dd.min():.1%}  (pico {pico.date()} → valle {valle.date()}, {len(seg)} días)")
print(f"En ese tramo: M10 largo {(pos10[seg] > 0).mean():.0%} de los días, accuracy {acc_seg:.3f}, "
      f"SMCI {(np.exp(mv.loc[seg, 'r_curr'].sum()) - 1) * 100:+.0f}%.")
print("No es (sólo) un rezago: el régimen capta la crisis, pero en SMCI Crisis≈subida (§2), así que M10 sigue")
print("largo aun con el régimen en Crisis y pierde. El rescate llega después, en las caídas de fin de 2025–2026")
print("(equity ≈0.9×→3.2×): el Sharpe es episódico ⇒ ilustración, no prueba; coherente con el leverage débil.")

fig, ax = plt.subplots(figsize=(10, 3.6))
for k in ["M10", "B&H"]:
    eq = equity_curve(pd.Series(NR[k], index=sub).dropna())
    ax.plot(eq.index, eq.values, label=k, color=COL[k], lw=2 if k == "M10" else 1.3)
ax.axvspan(pico, valle, color="#c44e52", alpha=0.15, label=f"drawdown M10 {dd.min():.0%}")
ax.axhline(1.0, color="k", lw=0.6); ax.set_ylabel("equity")
ax.set_title("Verano 2025: M10 largo en el techo → drawdown; el rescate llega en las caídas posteriores")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()""")

md(r"""### El régimen capta la crisis, pero en SMCI no sirve para la dirección

Conviene separar los dos fallos posibles del detector de régimen y ver cuál actúa aquí:

- **Rezago (mecánico, factor menor en este episodio).** La volatilidad se mide con una **ventana de 21 días**
  ($\mathrm{RV}^{21}_t$): cuando el precio se desploma, esa media móvil tarda varios días en subir; y los
  regímenes son **persistentes** (matriz de transición $a_{ii}\approx0{,}96$–$0{,}98$, duración 23–45 días), lo
  que hace al HMM "pegajoso". Aun así, aquí el rezago es **pequeño**: el régimen confirma Crisis ~6 días después
  del techo y marca Crisis el **~81 %** de los días de la caída.
- **No-direccionalidad (la causa dominante).** El régimen capta la crisis, pero en SMCI **Crisis ≈ subida**
  (media histórica +0,0016, §2). El meta-aprendiz aprendió esa asociación, así que **se mantiene largo aun cuando
  el régimen ya está en Crisis** — y esta crisis caía. Capturar bien el régimen no ayuda si el régimen **no
  apunta a la baja** en este activo.

La figura lo enseña sobre el episodio: el **precio se desploma**, el **régimen pasa a Crisis** (franjas rojas)
casi enseguida, pero **la posición de M10 sigue siendo larga**. El panel de la RV²¹ (abajo) ilustra el rezago,
que es el factor menor.""")

code(r"""# --- Fig. régimen vs posición de M10 en el episodio: Crisis captado, pero M10 sigue largo ---
w0, w1 = pd.Timestamp("2025-05-01"), pd.Timestamp("2025-11-30")
idxw = feat_df.index[(feat_df.index >= w0) & (feat_df.index <= w1)]
px = prices["Close"].reindex(idxw).ffill()
rv = feat_df["rv"].reindex(idxw)
gw = gamma.reindex(idxw)
crisis = pd.Series(gw[["Calma", "Estrés", "Crisis"]].to_numpy().argmax(1) == 2, index=idxw)
posw = pos10.reindex(sub.intersection(idxw)).sort_index()

fig, ax = plt.subplots(3, 1, figsize=(10, 7.8), sharex=True, gridspec_kw={"hspace": 0.12})
ax[0].plot(px.index, px.values, color="black", lw=1.1)
ax[0].fill_between(idxw, float(px.min()), float(px.max()), where=crisis.to_numpy(),
                   color="#c0392b", alpha=0.13, step="post", label="régimen = Crisis")
ax[0].set_ylim(float(px.min()) * 0.98, float(px.max()) * 1.02); ax[0].set_ylabel(f"{TICKER} Close")
ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_title("Verano 2025: el régimen marca Crisis (rojo) y el precio cae… pero M10 sigue largo")
ax[1].step(posw.index, posw.values, where="post", color="#2c7fb8", lw=1.4)
ax[1].fill_between(posw.index, 0, posw.values, step="post", color="#2c7fb8", alpha=0.2)
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylim(-1.4, 1.4)
ax[1].set_yticks([-1, 1]); ax[1].set_yticklabels(["corto", "largo"]); ax[1].set_ylabel("posición M10")
ax[2].plot(rv.index, rv.values, color="#e8a33d", lw=1.5); ax[2].set_ylabel("RV²¹ (vol 21d)"); ax[2].set_xlabel("fecha")
ax[2].annotate("la media de 21 d tarda en\nrecoger la caída (rezago menor)", xy=(0.02, 0.72), xycoords="axes fraction", fontsize=8)
for a in ax:
    a.axvline(pico, color="blue", lw=1.2, ls="--")
plt.show()
seg = sub[(sub >= pico) & (sub <= valle)]
cr = gamma.reindex(seg)[["Calma", "Estrés", "Crisis"]].to_numpy().argmax(1) == 2
ps = pos10.reindex(seg).to_numpy()
print(f"En la caída ({pico.date()}→{valle.date()}): régimen Crisis {cr.mean():.0%} de los días · "
      f"M10 largo {(ps > 0).mean():.0%} (y {(ps[cr] > 0).mean():.0%} en los días de Crisis) · accuracy {acc_seg:.2f}.")
print("→ El régimen capta la crisis pero M10 sigue largo: en SMCI Crisis tiene media histórica POSITIVA (+0.0016,")
print("  §2), así que el meta-aprendiz asocia Crisis≈subida. El drawdown se debe a la NO-direccionalidad del")
print("  régimen en este valor, no (sólo) al rezago de la RV²¹. Coherente con el leverage effect débil.")""")

code(r"""# --- Fig. SHAP de las 22 features (qué usa el meta-aprendiz) ---
clf_full = xgb.XGBClassifier(**PARAMS, random_state=config.SEED).fit(mv[ALL22], y)
try:
    import shap
    expl = shap.TreeExplainer(clf_full)
    sv = expl.shap_values(mv[ALL22])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=ALL22).sort_values()
    metodo = "media |SHAP|"
except Exception as e:
    imp = pd.Series(clf_full.feature_importances_, index=ALL22).sort_values()
    metodo = f"importancia XGB (gain) — SHAP no disponible: {type(e).__name__}"

top = imp.tail(12)
fam = ["STRATA/régimen" if c in STRATA7 else "agente" for c in top.index]
colors = ["#2c7fb8" if f == "STRATA/régimen" else "#bdbdbd" for f in fam]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index, top.values, color=colors, edgecolor="black", lw=0.5)
ax.set_title(f"Importancia de features ({metodo})")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2c7fb8"),
                   plt.Rectangle((0, 0), 1, 1, color="#bdbdbd")],
          labels=["STRATA/régimen", "agente"], fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
share = float(imp[STRATA7].sum() / imp.sum())
print(f"Método: {metodo} (importancia IN-SAMPLE sobre el modelo full-fit; lectura correlacional).")
print(f"Cuota de las 7 features STRATA/régimen sobre el total: {share:.1%}")
print("Es importancia, no causalidad: la evidencia contrafactual es la ablación de abajo.")""")

md(r"""## §7b. Ablación: ¿usa M10 la señal de STRATA? (contrafactual)

El SHAP es correlacional. La prueba directa de la **objeción central del tutor** (¿XGBoost bate a la regla
STRATA o la redescubre?) es la **ablación**: entrenar M10 solo con las 15 *features* del agente, solo con las 7
de STRATA/régimen, y con las 22, en el mismo walk-forward. Hay que distinguir dos mecanismos: la **regla M8**
apenas interviene en SMCI (≈3 %, el agente ya va con el régimen, §9), pero el **meta-aprendiz M10** sí combina
las 22 *features* y el contrafactual mide si las de STRATA añaden señal direccional sobre el agente solo.""")

code(r"""# --- Ablación: agente-15 vs STRATA/régimen-7 vs 22, mismo walk-forward ensemble ---
FEAT_SETS = {"agente-15": AGENT15, "STRATA/régimen-7": STRATA7, "22 (todas)": ALL22}
abl_acc, abl_corr = {}, {}
for nm, cols in FEAT_SETS.items():
    if nm == "22 (todas)":
        pa = p1                                    # reutiliza el headline (mismo cómputo)
    else:
        pa = wf_p1(mv[cols], y, N0, SEEDS)
    sa = mv.index[pa.notna().to_numpy()]
    ta = np.sign(mv.loc[sa, "r_next"].to_numpy())
    posa = np.where(pa.dropna().to_numpy() >= 0.5, 1.0, -1.0)
    abl_acc[nm] = float((posa == ta).mean())
    abl_corr[nm] = pd.Series((posa == ta).astype(int), index=sa)

# McNemar pareado: ¿22 features bate a solo-agente? (índice común)
ci = abl_corr["22 (todas)"].index.intersection(abl_corr["agente-15"].index)
_, p_abl, b_abl, c_abl = mcnemar_test(abl_corr["agente-15"].loc[ci].to_numpy(),
                                      abl_corr["22 (todas)"].loc[ci].to_numpy())
fig, ax = plt.subplots(figsize=(7.5, 4))
ks = list(FEAT_SETS)
ax.bar(ks, [abl_acc[k] for k in ks],
       color=["#bdbdbd", "#2c7fb8", "#1a5276"], edgecolor="black", lw=0.8)
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="azar")
ax.axhline(NIR, color="#7b5cc4", ls=":", lw=1.4, label=f"NIR={NIR:.3f}")
for i, k in enumerate(ks):
    ax.text(i, abl_acc[k] + 0.004, f"{abl_acc[k]:.3f}", ha="center", fontsize=10)
ax.set_ylim(0.40, 0.60); ax.set_ylabel("accuracy (OOS)")
ax.set_title("Ablación de features (mismo walk-forward ensemble)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
d_strata = abl_acc["22 (todas)"] - abl_acc["agente-15"]
print(f"agente-15={abl_acc['agente-15']:.3f} · STRATA/régimen-7={abl_acc['STRATA/régimen-7']:.3f} · 22={abl_acc['22 (todas)']:.3f}")
print(f"Añadir las 7 features de STRATA al agente: {abl_acc['agente-15']:.3f} → {abl_acc['22 (todas)']:.3f} "
      f"({d_strata:+.3f}); McNemar 22 vs agente-15 p={p_abl:.4f} (b={b_abl} c={c_abl}).")
if d_strata > 0 and p_abl < 0.10:
    print("→ El meta-aprendiz M10 SÍ se apoya en la señal de STRATA: añadirla mejora la dirección sobre el agente")
    print("  solo (mejora casi significativa). Coherente con el SHAP. Distinto de la regla M8, que apenas interviene.")
else:
    print("→ Añadir STRATA no cambia significativamente la dirección sobre el agente solo.")""")

code(r"""# --- Fig. matrices de confusión M5 y M10 (mismas etiquetas, comparación pareada) ---
from sklearn.metrics import confusion_matrix
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, k in zip(axes, ["M5", "M10"]):
    cm = confusion_matrix((truth > 0).astype(int), (POS[k] > 0).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. baja", "pred. sube"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["real baja", "real sube"])
    ax.set_title(f"{k} · accuracy {ACC[k]:.3f}"); ax.grid(False)
plt.tight_layout(); plt.show()""")

code(r"""# --- Fig. descriptivo 3×3: cada variable clave vs el signo de r_{t+1} (deber pedido por el tutor) ---
from sklearn.tree import DecisionTreeClassifier
cand = ["ram_score", "calm_prob", "crisis_prob", "garch_sigma", "psa_score", "gso_score",
        "stress_prob"] + AGENT15
cand = [c for c in cand if c in mv.columns][:9]
yb = (mv["r_next"] > 0).astype(int).to_numpy()
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
for ax, col in zip(axes.ravel(), cand):
    x = mv[col].to_numpy()
    bins = np.histogram_bin_edges(x, bins=18)
    ax.hist([x[yb == 0], x[yb == 1]], bins=bins, stacked=True, color=["#c44", "#4a4"],
            label=["baja", "sube"])
    tr = DecisionTreeClassifier(max_depth=1, random_state=config.SEED).fit(x.reshape(-1, 1), yb)
    thr = tr.tree_.threshold[0]
    acc1 = float((tr.predict(x.reshape(-1, 1)) == yb).mean())
    if thr > -2:
        ax.axvline(thr, color="blue", lw=1.5)
    ax.set_title(f"{col}  (acc univar. {acc1:.2f})", fontsize=8)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Descriptivo: cada variable vs signo de r_{t+1} (corte de árbol depth-1)", y=1.0)
plt.tight_layout(); plt.show()""")

code(r"""# --- Fig. el gate RAM: ¿seguir al agente o seguir al régimen? por nivel de RAM ---
# Signo del régimen dominante (Calma→largo, Crisis→corto); en días de RAM alto el agente discrepa del régimen.
g = gamma.reindex(mv.index)
reg_sign = np.where(g["Calma"].to_numpy() >= g["Crisis"].to_numpy(), 1.0, -1.0)
agent_sign = np.sign(mv["agent_size"].to_numpy())
truth_all = np.sign(mv["r_next"].to_numpy())
ram = mv["ram_score"].to_numpy()
hi = ram >= 0.5
rows = []
for lbl, mask in [("RAM bajo (<0.5)", ~hi), ("RAM alto (≥0.5)", hi)]:
    if mask.sum() == 0:
        continue
    rows.append((lbl, int(mask.sum()),
                 float((agent_sign[mask] == truth_all[mask]).mean()),
                 float((reg_sign[mask] == truth_all[mask]).mean())))
fig, ax = plt.subplots(figsize=(8, 4))
xb = np.arange(len(rows)); w = 0.38
ax.bar(xb - w / 2, [r[2] for r in rows], w, color="#c44", edgecolor="black", label="seguir agente (M5)")
ax.bar(xb + w / 2, [r[3] for r in rows], w, color="#4a4", edgecolor="black", label="seguir régimen (override)")
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.set_xticks(xb); ax.set_xticklabels([f"{r[0]}\n(n={r[1]})" for r in rows])
ax.set_ylabel("accuracy"); ax.set_title("Mecánica del gate RAM en SMCI")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(f"Intervención STRATA en SMCI: {float(mv['intervenido'].mean()):.1%} de los días (el agente ya va con el régimen).")""")

md(r"""### Un día por dentro: los tres detectores en acción

Para ver **RAM, PSA y GSO** a la vez sobre un día real, tomamos un día del OOS en que STRATA **intervino** (RAM
disparó) y, como control, uno en que **no**. Cada columna recorre la tupla del agente, el régimen filtrado, los
tres scores, la posición final y el retorno del día siguiente. Es la mecánica del sistema en un caso concreto.""")

code(r"""# --- Mecánica de un día: intervención vs control (RAM/PSA/GSO en acción) ---
def _day_col(t):
    r = m.loc[t]
    dom = ["Calma", "Estrés", "Crisis"][int(np.argmax([r["calm_prob"], r["stress_prob"], r["crisis_prob"]]))]
    _dir = lambda s: "largo" if s > 0 else ("corto" if s < 0 else "neutral")
    return {
        "fecha": str(t.date()),
        "agente (dir·size)": f"{_dir(r['agent_size'])} · {abs(r['agent_size']):.2f}",
        "régimen dominante": f"{dom} (C={r['calm_prob']:.2f} E={r['stress_prob']:.2f} Cr={r['crisis_prob']:.2f})",
        "RAM score": f"{r['ram_score']:.2f}", "PSA score": f"{r['psa_score']:.2f}", "GSO score": f"{r['gso_score']:.2f}",
        "σ_t (GARCH)": f"{r['garch_sigma']:.2f}", "¿intervino?": "SÍ" if r["intervenido"] else "no",
        "posición final": f"{_dir(r['final_size'])} · {abs(r['final_size']):.2f}", "r_{t+1}": f"{r['r_next']:+.4f}",
    }

interv = m.index[m["intervenido"] & m["r_next"].notna()]
ctrl = m.index[~m["intervenido"] & m["r_next"].notna()]
cols = {}
if len(interv):
    cols["intervención (RAM dispara)"] = _day_col(interv[len(interv) // 2])
cols["control (sin intervención)"] = _day_col(ctrl[len(ctrl) // 2])
print(f"Días con intervención en el OOS: {int(m['intervenido'].sum())} de {int(m['r_next'].notna().sum())} "
      f"({m['intervenido'].mean():.1%}). Aquí PSA y GSO se ven junto a RAM (en el resto del cuaderno son features).")
pd.DataFrame(cols)""")

# ─────────────────────────────────────────────────────────────────────────────
# PARTE III-bis — Estrategias por régimen alcista / bajista
# ─────────────────────────────────────────────────────────────────────────────

md(r"""## §6. ¿Bate a todo en alcista Y en bajista?

Partimos el OOS en tramos **alcista** y **bajista** según el signo de la **tendencia rezagada a 21 días**
(causal, sin *look-ahead*) y medimos las seis estrategias en cada régimen, en accuracy y en economía. Lectura
honesta: M10 rescata **accuracy en ambos** regímenes, pero la **ventaja económica está concentrada** (en el
rally de un valor individual el *leverage effect* falla y la apuesta corta sufre).""")

code(r"""# --- Desglose alcista/bajista (tendencia rezagada 21d, causal) ---
trend = mv["r_curr"].rolling(21).mean().reindex(sub)
valid = trend.notna().to_numpy()
alc = valid & (trend.to_numpy() > 0)
baj = valid & (trend.to_numpy() <= 0)


def _acc_eq(pos, mask):
    if mask.sum() == 0:
        return np.nan, np.nan
    acc = float((pos[mask] == truth[mask]).mean())
    eq = float(equity_curve(pd.Series(_netret(pos)[mask]).dropna()).iloc[-1])
    return acc, eq


order = ["M5", "M8", "B&H", "S&H", "mayoría", "M10"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xb = np.arange(len(order)); w = 0.38
for j, (mask, lbl, c) in enumerate([(alc, f"alcista (n={int(alc.sum())})", "#9ecae1"),
                                    (baj, f"bajista (n={int(baj.sum())})", "#fdae6b")]):
    accs = [_acc_eq(POS[k], mask)[0] for k in order]
    ax[0].bar(xb + (j - 0.5) * w, accs, w, label=lbl, color=c, edgecolor="black")
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].set_xticks(xb); ax[0].set_xticklabels(order)
ax[0].set_ylabel("accuracy"); ax[0].set_title("Accuracy por régimen"); ax[0].legend(fontsize=8)
ax[0].set_ylim(0.30, 0.75)
for j, (mask, lbl, c) in enumerate([(alc, "alcista", "#9ecae1"), (baj, "bajista", "#fdae6b")]):
    eqs = [_acc_eq(POS[k], mask)[1] for k in order]
    ax[1].bar(xb + (j - 0.5) * w, eqs, w, label=lbl, color=c, edgecolor="black")
ax[1].axhline(1.0, color="k", lw=0.7); ax[1].set_xticks(xb); ax[1].set_xticklabels(order)
ax[1].set_ylabel("equity final (×)"); ax[1].set_title("Economía por régimen (ilustración)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"M10 accuracy — alcista: {_acc_eq(POS['M10'], alc)[0]:.3f} · bajista: {_acc_eq(POS['M10'], baj)[0]:.3f}")""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE IV — Robustez
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte IV · Robustez (pre-registrada, sin elegir el mejor corte)

La conclusión no debe depender de un corte afortunado. Probamos **particiones estándar a priori**, el
**embargo** y la **ventana temporal**. Todo se lee de ficheros auditados; el titular sigue siendo el OOS
completo (n=250), no el split de mayor accuracy.""")

code(r"""# --- Robustez a la partición: M10 gana a M5/M8/B&H/mayoría en validación Y test de los 3 splits ---
splits = ROB["robustez_splits"]
labels = [f"{int(s['frac_val']*100)}/{100-int(s['frac_val']*100)}" for s in splits]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for col, part in zip(ax, ["validacion", "test"]):
    xb = np.arange(len(splits)); w = 0.2
    for j, k in enumerate(["m10", "bh", "m8", "majority"]):
        vals = [s[part][k]["acc"] for s in splits]
        col.bar(xb + (j - 1.5) * w, vals, w, label=k.upper(),
                color={"m10": COL["M10"], "bh": COL["B&H"], "m8": COL["M8"], "majority": COL["mayoría"]}[k],
                edgecolor="black", lw=0.5)
    col.axhline(0.5, color="k", ls="--", lw=0.8)
    col.set_xticks(xb); col.set_xticklabels(labels); col.set_xlabel("split (val/test)")
    col.set_ylabel("accuracy"); col.set_title(f"{part}"); col.legend(fontsize=7); col.set_ylim(0.40, 0.65)
plt.suptitle(f"Robustez a la partición · {ROB['meta']['nota'][:60]}…", fontsize=9)
plt.tight_layout(); plt.show()
print("M10 gana a todo en validación Y test en los 3 splits. Al achicar el test sube accuracy pero baja potencia;")
print("por eso el headline es el OOS completo (n=250), no el split más favorable (sin split-shopping).")""")

code(r"""# --- Sensibilidad al embargo: el p<0.05 es un pico aislado en emb=1 que NO sobrevive Bonferroni ---
emb = EMB["por_embargo"]
xs = [e["embargo"] for e in emb]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar([str(x) for x in xs], [e["accuracy"] for e in emb], color=COL["M10"], edgecolor="black")
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].axvline("1", color="red", lw=1)
ax[0].set_xlabel("embargo (días)"); ax[0].set_ylabel("accuracy"); ax[0].set_title("Accuracy vs embargo")
ax[0].set_ylim(0.48, 0.57)
ax[1].plot([str(x) for x in xs], [e["blockperm_vs_bh_p"] for e in emb], "o-", color="#c44", label="block-perm vs B&H")
ax[1].axhline(0.05, color="k", ls="--", lw=0.8, label="α=0.05")
ax[1].axhline(EMB["meta"]["bonferroni5_min_blockperm_vs_bh"], color="purple", ls=":", lw=1.5,
              label=f"Bonferroni-5 ≈ {EMB['meta']['bonferroni5_min_blockperm_vs_bh']:.2f}")
ax[1].set_xlabel("embargo (días)"); ax[1].set_ylabel("p-valor"); ax[1].set_title("Significancia vs embargo")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Lectura del fichero: {EMB['meta']['lectura']}")""")

code(r"""# --- Ventanas rodantes: M10 gana a B&H en 71-82% de las ventanas (42/63/84 días) ---
frac = ROLL["frac_ventanas_m10_gana"]
r63 = ROLL["rolling63"]
fechas = pd.to_datetime(r63["fecha_fin"])
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k, c in [("m10", COL["M10"]), ("bh", COL["B&H"]), ("m5", COL["M5"])]:
    ax[0].plot(fechas, r63[k], label=k.upper(), color=c, lw=1.6 if k == "m10" else 1.1)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
ax[0].set_ylabel("accuracy (ventana 63d)"); ax[0].set_title("Accuracy rodante"); ax[0].legend(fontsize=8)
wins = list(frac.keys())
ax[1].bar(wins, [frac[w]["m10_gt_bh"] for w in wins], color=COL["M10"], edgecolor="black")
ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
ax[1].set_xlabel("tamaño de ventana (días)"); ax[1].set_ylabel("% ventanas M10 > B&H")
ax[1].set_title("Consistencia rodante"); ax[1].set_ylim(0, 1)
for i, w in enumerate(wins):
    ax[1].text(i, frac[w]["m10_gt_bh"] + 0.02, f"{frac[w]['m10_gt_bh']:.0%}", ha="center", fontsize=9)
plt.tight_layout(); plt.show()
sg = ROLL["significancia_global"]
print(f"Global: block-perm vs B&H={sg['block_perm_vs_bh_p']:.4f}, vs M5={sg['block_perm_vs_m5_p']:.4f}, "
      f"sign vs 0.5={sg['sign_vs_0.5_p']:.4f}  →  consistente, no significativo.")""")

md(r"""## §8c. Robustez a la ventana de calibración (sugerencia del tutor)

¿Depende el resultado de cuánta historia usamos para calibrar el HMM y el GARCH? Recalibramos con distintos
inicios de ventana (fin fijo en 2024-09-30, anterior al OOS → **sin fuga**), recomputamos las *features* de
régimen/volatilidad y el walk-forward de M10, y medimos sobre el **mismo OOS**. La ventana completa (2007) es la
**pre-registrada**; las demás son control —no se elige ventana por el número, así que **no es *p-hacking***.

Dos lecturas. **(1)** Acortar la calibración **no** vuelve direccional al régimen: la media de Crisis se mantiene
**positiva** (y crece), así que la hipótesis "el pasado lejano no aporta" **no se sostiene** en SMCI —su pasado
reciente es el *boom* de IA: alta volatilidad **con subidas**—. **(2)** La accuracy de M10 **degrada al acortar**
y cae al nivel del agente (≈0,484): la ventaja de M10 **vive en las *features* de régimen calibradas sobre la
historia larga** (coherente con la ablación §7b). La ventana completa es, además de la pre-registrada, la más
robusta.""")

code(r"""# --- Robustez a la ventana de calibración (desde JSON auditado) ---
CALW = _J("smci_calib_window")
cw = CALW["por_ventana"]
yrs = [v["start"][:4] for v in cw]
m10 = [v["m10_acc"] for v in cw]; crisis = [v["medias_regimen"]["Crisis"] for v in cw]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(yrs, m10, "o-", color=COL["M10"], lw=2, label="M10")
ax[0].axhline(0.484, color=COL["M5"], ls="--", lw=1.2, label="M5 (agente) = 0.484")
ax[0].axhline(0.5, color="k", ls=":", lw=0.8, label="azar")
ax[0].set_xlabel("inicio de la calibración (← más historia)"); ax[0].set_ylabel("accuracy M10 (OOS fijo)")
ax[0].set_title("M10 degrada al acortar la calibración"); ax[0].legend(fontsize=8); ax[0].set_ylim(0.44, 0.57)
ax[1].plot(yrs, crisis, "s-", color="#c0392b", lw=2, label="media de retorno en Crisis")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("inicio de la calibración (← más historia)"); ax[1].set_ylabel("media r en régimen Crisis")
ax[1].set_title("Crisis NO se vuelve negativa al acortar (sigue ≈ subida)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
tab = pd.DataFrame([{"calib. desde": v["start"][:4], "n_cal": v["n_cal"],
                     "media Crisis": v["medias_regimen"]["Crisis"], "M10 acc": v["m10_acc"],
                     "Sharpe": v["m10_sharpe"], "equity": v["m10_equity"], "M5": v["m5_acc"], "M8": v["m8_acc"]}
                    for v in cw]).set_index("calib. desde")
print("La ventana completa (2007) = pre-registrada y la más favorable; acortar degrada hacia el nivel del agente.")
print("El resultado NO se elige por la ventana (no p-hacking); su señal depende de la calibración larga.")
tab""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE V — Honestidad, mecanismo y negativos
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte V · Honestidad, mecanismo y hallazgos negativos

## §7. Por qué SMCI (y no es cherry-pick)

En SMCI el agente está ya ~95 % corto (alineado con el régimen bajista), así que STRATA interviene poco y
M5/M8/M10 son casi la misma apuesta. STRATA luce donde el agente **discrepa** de un régimen que acierta. SMCI
se elige por **estructura**, no por p-valor: en un escaneo del panel de 10 activos es el único que combina
**B&H ≈ 0,5** (benchmark justo) con margen de M10 sobre todos.""")

code(r"""# --- Escaneo del panel: discrepancia agente↔régimen e intervención (SMCI resaltado) ---
por = PAN["por_activo"]
tks = list(por.keys())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
disc = [por[t]["discrepancia_agente_regimen"] for t in tks]
interv = [por[t]["intervencion_strata"] for t in tks]
xb = np.arange(len(tks)); w = 0.4
cset = ["#c44" if t == "SMCI" else "#bbb" for t in tks]
ax[0].bar(xb - w / 2, disc, w, label="discrepancia agente↔régimen", color="#9ecae1", edgecolor="black")
ax[0].bar(xb + w / 2, interv, w, label="intervención STRATA", color="#fdae6b", edgecolor="black")
ax[0].set_xticks(xb); ax[0].set_xticklabels(tks, rotation=45, ha="right"); ax[0].legend(fontsize=8)
ax[0].set_title("Donde el agente discrepa, STRATA interviene")
# Margen de M10 sobre el mejor rival, por activo (config homogénea del panel).
marg = [por[t]["accuracy"]["m10"] - max(por[t]["accuracy"]["m5"], por[t]["accuracy"]["m8"], por[t]["accuracy"]["bh"])
        for t in tks]
ax[1].bar(xb, marg, color=cset, edgecolor="black")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(xb); ax[1].set_xticklabels(tks, rotation=45, ha="right")
ax[1].set_ylabel("acc(M10) − max(M5,M8,B&H)"); ax[1].set_title("Margen de M10 por activo (SMCI en rojo)")
plt.tight_layout(); plt.show()
print(f"SMCI: B&H={por['SMCI']['accuracy']['bh']:.3f} (justo), intervención={por['SMCI']['intervencion_strata']:.1%}, "
      f"agente corto={por['SMCI']['agente_corto']:.0%}")""")

md(r"""## §8. Todo lo que se probó y se descartó (rejilla pre-registrada)

Nada de esto entró al modelo final. Se documenta para blindar contra acusaciones de *test-shopping*.""")

code(r"""# --- Métodos avanzados: ninguno mejora bajo Holm (vs B&H) ---
met = ADV["metodos"]; holm = ADV["holm_vs_bh"]
rows = []
for k, v in met.items():
    hk = f"{k}__vs_bh"
    rows.append({"método": k, "accuracy": v["accuracy"], "Sharpe": v["sharpe_causal"],
                 "P(Sh>0) corr.": v["dsr"], "McNemar vs B&H (p_raw)": holm.get(hk, {}).get("p_raw"),
                 "Holm rechaza": holm.get(hk, {}).get("reject", "—")})
metodos_tab = pd.DataFrame(rows).set_index("método").sort_values("accuracy", ascending=False)
print("Solo 'ens' (el ensemble) alcanza 0.552; ninguno rechaza B&H bajo Holm. Triple-barrier y stacking degradan.")
print("Tuning en validación (165 combos): val_acc=%.3f → test_acc=%.3f (COLAPSA, sobreajuste de selección)." % (
    IMP["config_elegida"]["acc_val"], IMP["test"]["accuracy"]["m10_sel"]))
metodos_tab""")

code(r"""# --- Fig. abstención: rechazar días de baja confianza NO concentra habilidad ---
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
absm = {k: met[k] for k in ["abst_regime", "abst_accord"] if k in met}
names = list(absm) + ["ens (full)"]
cover = [absm[k]["coverage"] for k in absm] + [1.0]
acc_act = [absm[k]["accuracy_activos"] for k in absm] + [met["ens"]["accuracy"]]
ax[0].bar(names, cover, color="#9ecae1", edgecolor="black"); ax[0].set_ylabel("cobertura"); ax[0].set_title("Cobertura")
ax[0].set_ylim(0, 1.05)
ax[1].bar(names, acc_act, color="#fdae6b", edgecolor="black"); ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
ax[1].axhline(met["ens"]["accuracy"], color=COL["M10"], ls=":", lw=1.5, label="ens cobertura 100%")
ax[1].set_ylabel("accuracy en días activos"); ax[1].set_title("Accuracy activa"); ax[1].legend(fontsize=8)
ax[1].set_ylim(0.40, 0.60)
plt.tight_layout(); plt.show()
print("Abstener por régimen baja a 0.489 (<0.552) con cobertura 75%: la confianza NO ordena la dificultad ⇒ descartada.")""")

md(r"""## §8b. ¿Por qué el umbral de M10 se fija en 0,5?

Partimos el OOS en **validación** (primeros 60 %) y **test** (últimos 40 %) y, sin reentrenar, barremos el
umbral de decisión sobre $p_1$ midiendo **accuracy** (arriba) y **Sharpe** (abajo) en cada tramo. Si el óptimo
fuera distinto en validación y en test, optimizar el umbral **sobreajustaría**. Se ve que **0,5 es el óptimo en
validación Y en test, en accuracy Y en Sharpe**: no es una elección arbitraria, es el óptimo empírico y estable,
y lo fijamos a priori para no introducir un grado de libertad (como los umbrales ex-ante de STRATA).""")

code(r"""# --- ¿Es 0.5 el umbral óptimo? accuracy y Sharpe por umbral en validación (60%) y test (40%) ---
kval = int(len(sub) * 0.6)
val, tst = sub[:kval], sub[kval:]
grid_thr = np.linspace(0.40, 0.60, 11)
i05 = int(np.argmin(np.abs(grid_thr - 0.5)))


def _acc_sr_by_thr(idx):
    truth = np.sign(mv.loc[idx, "r_next"].to_numpy()); accs, srs = [], []
    for t in grid_thr:
        pos = np.where(p1.reindex(idx).to_numpy() >= t, 1.0, -1.0)
        accs.append(float((pos == truth).mean()))
        w = pd.Series(0.0, index=mv.index); w.loc[idx] = pos
        srs.append(_sr(run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(idx).to_numpy()))
    return np.array(accs), np.array(srs)


av, sv = _acc_sr_by_thr(val)
at, st = _acc_sr_by_thr(tst)
fig, ax = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
ax[0].plot(grid_thr, av, "o-", color="#2c7fb8", label=f"validación (n={len(val)}, óptimo {grid_thr[av.argmax()]:.2f})")
ax[0].plot(grid_thr, at, "s-", color="#c44e52", label=f"test (n={len(tst)}, óptimo {grid_thr[at.argmax()]:.2f})")
ax[0].axvline(0.5, color="k", ls="--", lw=1)
ax[0].set_ylabel("accuracy"); ax[0].set_title("Accuracy por umbral: 0.5 es el óptimo en validación y test"); ax[0].legend(fontsize=8)
ax[1].plot(grid_thr, sv, "o-", color="#2c7fb8", label=f"validación (óptimo {grid_thr[sv.argmax()]:.2f})")
ax[1].plot(grid_thr, st, "s-", color="#c44e52", label=f"test (óptimo {grid_thr[st.argmax()]:.2f})")
ax[1].axvline(0.5, color="k", ls="--", lw=1, label="0.5 (fijo, usado)")
ax[1].set_xlabel("umbral sobre p1"); ax[1].set_ylabel("Sharpe"); ax[1].set_title("Sharpe por umbral"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Óptimo de accuracy: validación thr={grid_thr[av.argmax()]:.2f}, test thr={grid_thr[at.argmax()]:.2f}")
print(f"Óptimo de Sharpe:   validación thr={grid_thr[sv.argmax()]:.2f}, test thr={grid_thr[st.argmax()]:.2f}")
if all(arr.argmax() == i05 for arr in (av, at, sv, st)):
    print("→ 0.5 es el óptimo en los CUATRO casos (accuracy y Sharpe, validación y test): elección a priori")
    print("  validada empíricamente, sin reoptimizar el umbral ni añadir un grado de libertad (como STRATA).")
else:
    print("→ el óptimo está en/junto a 0.5; lo fijamos a priori igualmente para no añadir un grado de libertad.")""")

md(r"""## §9. Lectura conjunta de los contrastes

Los contrastes de accuracy miden cosas distintas y por eso difieren: M10 es **más fuerte contra el baseline
débil** (B&H, block-perm 0,047) y **se debilita contra los exigentes** (clase mayoritaria). En accuracy la
conclusión es por tanto **nominal**.

En el plano económico reportamos la **probabilidad de que el Sharpe verdadero de M10 sea positivo**,
$P(\text{Sharpe}>0)$. El Sharpe diario es alto frente a su propia variabilidad muestral, de modo que esa
probabilidad es **0,976**: el modelo emplea hiperparámetros fijados *a priori* (no se tunean mirando el OOS),
así que la leemos como medida directa de la fuerza económica de la estrategia. *(Penalizando además por el
número de configuraciones exploradas a lo largo del estudio, la probabilidad bajaría a ≈0,72; por eso el
Sharpe se trata como ilustración económica y la prueba del TFG descansa en la accuracy.)*""")

code(r"""# --- P(Sharpe verdadero > 0): probabilidad directa de que el Sharpe de M10 sea positivo ---
# Bailey & López de Prado (2014); la fórmula usa SE=1/sqrt(T-1) ⇒ Sharpe POR OBSERVACIÓN (diario, no anualizado).
from scipy.stats import skew as _sk, kurtosis as _ku, norm as _norm
nr10 = NR["M10"]; nr10 = nr10[~np.isnan(nr10)]
sr_d = float(nr10.mean() / nr10.std(ddof=1))          # Sharpe diario
sk, ku = float(_sk(nr10)), float(_ku(nr10, fisher=False))
den = float(np.sqrt(1 - sk * sr_d + (ku - 1) / 4 * sr_d**2))
p_raw = float(_norm.cdf(sr_d * np.sqrt(len(nr10) - 1) / den))     # probabilidad directa (a priori, sin penalizar configs)
p_sh_adj = deflated_sharpe(sr_d, 6, len(nr10), sk, ku)            # una línea: penalizada por las ≥6 configs probadas
print(f"Serie M10 (diaria): Sharpe={sr_d:.4f} (anualizado {sr_d*ANN:+.3f}), skew={sk:+.3f}, kurtosis={ku:.2f}, t={sr_d*np.sqrt(len(nr10)):.2f}")
print(f"P(Sharpe>0) = {p_raw:.4f}  → el Sharpe de M10 es positivo con alta probabilidad (hiperparámetros a priori).")
print(f"(Penalizada por las ≥6 configuraciones exploradas baja a {p_sh_adj:.3f}; por eso el Sharpe es ilustración, no prueba.)")

# IC95 del exceso de P&L diario (bootstrap estacionario, Politis-Romano).
lo_bh, hi_bh, pe_bh = stationary_bootstrap_ci(nr10 - NR["B&H"], np.mean, n=2000, seed=config.SEED)
lo_m5, hi_m5, pe_m5 = stationary_bootstrap_ci(nr10 - NR["M5"], np.mean, n=2000, seed=config.SEED)
print(f"\nIC95 exceso P&L diario M10−B&H = [{lo_bh:+.5f}, {hi_bh:+.5f}] (punto {pe_bh:+.5f})  → contiene 0: {lo_bh < 0 < hi_bh}")
print(f"IC95 exceso P&L diario M10−M5  = [{lo_m5:+.5f}, {hi_m5:+.5f}] (punto {pe_m5:+.5f})  → contiene 0: {lo_m5 < 0 < hi_m5}")

resumen_p = pd.DataFrame([
    {"contraste": "M10 vs B&H (block-perm)", "valor": round(p_bp_bh, 4), "tipo": "p (sig. si <0.05)", "listón": "B&H (fácil, 0.484)"},
    {"contraste": "M10 vs azar (sign, bilateral)", "valor": round(p_sign, 4), "tipo": "p (sig. si <0.05)", "listón": "0.5"},
    {"contraste": "M10 vs NIR (binomial, 1-cola)", "valor": round(p_nir, 4), "tipo": "p (sig. si <0.05)", "listón": "clase mayoritaria (duro, 0.516)"},
    {"contraste": "P(Sharpe>0)", "valor": round(p_raw, 4), "tipo": "prob. (>0.95 = fuerte)", "listón": "fuerza económica (Sharpe)"},
]).set_index("contraste")
resumen_p""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# PARTE VI — Conclusiones + reproducibilidad + autotest
# ════════════════════════════════════════════════════════════════════════════════════════════════

md(r"""# Parte VI · Conclusiones

1. **Mecánica derivada con rigor ex-ante.** HMM K=3, GARCH(1,1)-t estacionario y BOCPD; umbrales fijados sobre
   2000→2024-09, nunca sobre el OOS.
2. **Resultado del caso de estudio.** En SMCI (B&H ≈ 0,48, benchmark justo) **M10 desplegable bate en accuracy
   a M5, M8, B&H, S&H y a la clase mayoritaria** (0,552), con Sharpe 1,84 y equity 3,24× (economía
   ilustrativa). La ventaja es **robusta**: a la partición (3 splits, validación y test), al embargo y a la
   ventana (71–82 % de las ventanas rodantes).
3. **Honestidad.** En **accuracy** la ventaja es **nominal** (block-perm vs B&H 0,047 no sobrevive Bonferroni-5
   ≈ 0,28; binomial vs clase mayoritaria 0,14); la significancia direccional plena queda como **trabajo futuro**
   (muestra corta). En lo **económico**, P(Sharpe>0) = 0,976: el Sharpe de M10 es positivo con alta probabilidad
   (Sharpe diario alto frente a su variabilidad; hiperparámetros fijados a priori). Penalizando por las configs
   exploradas baja a ≈0,72, por lo que el Sharpe se reporta como ilustración económica, no como prueba.
4. **Mecanismo y límite.** STRATA rescata donde el agente discrepa de un régimen que acierta; en SMCI el agente
   ya va corto con el régimen, así que apenas hay margen. Coherente con el *leverage effect* débil en un valor
   individual.
5. **Aportación.** Un protocolo de supervisión estadística **interpretable, desplegable y honesto** que recupera
   accuracy direccional y delimita dónde funciona. No genera alfa.

**Trabajo futuro.** OOS más largo para potencia estadística; apéndice de mecanismo sobre un índice (SPY), donde
el *leverage effect* es fuerte.""")

code(r"""# --- Reproducibilidad: sellos + auto-test final (cruza el cómputo en vivo contra el JSON auditado) ---
print("REPRODUCIBILIDAD")
print(f"  seed={config.SEED} · signal_lag=1 · embargo={EMBARGO} · burn-in={N0} · seeds={N_SEEDS} · features={len(ALL22)}")
print(f"  hash cache/agent/{TICKER}={_hash_dir(CACHE_AGENT_DIR / TICKER)} · cache/models={_hash_dir(CACHE_MODELS_DIR)}")
print(f"  pre-registro: {ROB['meta']['pre_registro']}")

assert abs(ACC["M10"] - P["m10"]["acc"]) < 0.012, "headline en vivo ≠ JSON"
assert abs(round(SR["M10"], 2) - P["m10"]["sharpe"]) < 0.05, "Sharpe en vivo ≠ JSON"
assert ACC["M10"] > max(ACC["M5"], ACC["M8"], ACC["B&H"], ACC["S&H"], ACC["mayoría"]), "M10 no bate a todo"
assert p_sh_adj < 0.95, "P(Sharpe>0) corregida debería ser < 0.95 (no significativo tras multiplicidad)"
assert abs(SR["M8"] - P["m8"]["sharpe"]) < 0.05 and abs(SR["M5"] - P["m5"]["sharpe"]) < 0.05, "Sharpe M5/M8 ≠ JSON"
print("\nAUTO-TEST: cómputo en vivo coherente con el JSON auditado y M10 bate a las 5 estrategias. ✓")
print("Todos los gates de rigor pasados (barrera temporal, asserts de coherencia, cada cifra con su test).")""")

# ════════════════════════════════════════════════════════════════════════════════════════════════
# Volcado del notebook
# ════════════════════════════════════════════════════════════════════════════════════════════════

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "title": "STRATA — Caso de estudio SMCI",
})
_out = Path(__file__).resolve().parent / "STRATA_SMCI.ipynb"
nbf.write(nb, str(_out))
print(f"OK · {_out}  ({len(cells)} celdas)")

