"""Exporta las figuras del Capítulo 4 (SMCI) a tesis/figuras/*.pdf (vectoriales).

Recomputa en vivo el M10 definitivo (headline/equity/forest) y lee los JSON auditados para
ablación/SHAP y robustez. Cada figura con nombre estable para \\includegraphics.

Uso: python experiments/export_figs_cap4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import xgboost as xgb

import config
from config import CALIBRATION_END, STRATA_OOS_START, CACHE_MODELS_DIR
from core import data, features
from core.backtest import run_backtest
from core.metrics import equity_curve
from core.hmm import RegimeHMM
from core.garch import GARCHModel
import experiments.walkforward_robustez as wf

FIG = Path("tesis/figuras"); FIG.mkdir(parents=True, exist_ok=True)
TICKER = "SMCI"
STEP, EMBARGO, N0, N_SEEDS = 21, 1, 150, 10
SEEDS = [config.SEED + i for i in range(N_SEEDS)]
ANN = np.sqrt(252)
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist")
AGENT15 = [f"{nm}_{k}" for nm in wf.PERS for k in ("sign", "size", "conf")]
STRATA7 = ["ram_score", "psa_score", "gso_score", "calm_prob", "stress_prob", "crisis_prob", "garch_sigma"]
ALL22 = AGENT15 + STRATA7
COL = {"M10": "#2c7fb8", "M8": "#f0a830", "M5": "#9e9e9e", "B&H": "#4caf50", "S&H": "#c44e52", "mayoría": "#7b5cc4"}
REG = {"Calma": "#2e9e4f", "Estrés": "#e8a33d", "Crisis": "#c0392b"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "font.size": 10})


def _J(n):
    return json.load(open(f"outputs/experiments/{n}.json"))


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig); print(f"  ✓ {name}.pdf")


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


# ── estados + master + p1 (una vez) ──────────────────────────────────────────────
feat_df, ret = wf.load_features(TICKER)
import glob as _g
DATA_END = sorted(_g.glob(str(config.DATA_DIR / f"{TICKER}_{config.CALIBRATION_START}_*.parquet")))[-1].rsplit("_", 1)[1].replace(".parquet", "")
prices = data.load_market_data(TICKER, config.CALIBRATION_START, DATA_END)
calib = feat_df.loc[feat_df.index <= pd.Timestamp(CALIBRATION_END)]
hmm = RegimeHMM(n_states=3, seed=config.SEED).fit(calib.to_numpy())
garch = GARCHModel().fit(ret.loc[ret.index <= pd.Timestamp(CALIBRATION_END)])
oos_ret = ret[ret.index >= pd.Timestamp(STRATA_OOS_START)]
sigma = garch.forecast_path(oos_ret)
gamma = pd.DataFrame(hmm.predict_proba_filtered(feat_df.to_numpy()), index=feat_df.index, columns=["Calma", "Estrés", "Crisis"])
wf.reset_thresholds_cache()
m = wf.run_master(gamma, sigma, oos_ret, wf.load_agent(TICKER))
mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
y = (mv["r_next"] > 0).astype(int)
p1 = pd.Series(np.nan, index=mv.index)
for s in range(N0, len(mv), STEP):
    tr = s - EMBARGO
    if tr < 50:
        continue
    e = min(s + STEP, len(mv))
    p1.iloc[s:e] = np.mean([xgb.XGBClassifier(**PARAMS, random_state=sd).fit(mv[ALL22].iloc[:tr], y.iloc[:tr]).predict_proba(mv[ALL22].iloc[s:e])[:, 1] for sd in SEEDS], axis=0)
sub = mv.index[p1.notna().to_numpy()]
truth = np.sign(mv.loc[sub, "r_next"].to_numpy()); yt = (truth > 0).astype(int)
nir_dir = 1.0 if yt.mean() > 0.5 else -1.0
NIR = float(max(yt.mean(), 1 - yt.mean()))
_o = oos_ret.reindex(sub)
POS = {"M10": np.where(p1.dropna().to_numpy() >= 0.5, 1.0, -1.0),
       "M8": np.sign(mv.loc[sub, "final_size"].to_numpy()), "M5": np.sign(mv.loc[sub, "agent_size"].to_numpy()),
       "B&H": np.ones(len(sub)), "S&H": -np.ones(len(sub)), "mayoría": np.full(len(sub), nir_dir)}
ACC = {k: float((v == truth).mean()) for k, v in POS.items()}
NR = {"M10": run_backtest(oos_ret, pd.Series(POS["M10"], index=sub).reindex(mv.index).fillna(0.0), signal_lag=1)["net_return"].reindex(sub).to_numpy(),
      "M8": mv["nr_m8_causal"].reindex(sub).to_numpy(), "M5": mv["nr_m5_causal"].reindex(sub).to_numpy(),
      "B&H": _o.to_numpy()}
print(f"headline acc M10={ACC['M10']:.3f} (n={len(sub)})")

# ── Fig 1: headline accuracy 6 estrategias ───────────────────────────────────────
order = ["M5", "M8", "B&H", "S&H", "mayoría", "M10"]
fig, ax = plt.subplots(figsize=(8, 4))
b = ax.bar(order, [ACC[k] for k in order], color=[COL[k] for k in order], edgecolor="black", lw=0.8)
ax.axhline(0.5, color="k", ls="--", lw=1, label="azar = 0,500")
ax.axhline(NIR, color="#7b5cc4", ls=":", lw=1.6, label=f"clase mayoritaria = {NIR:.3f}")
for bb, k in zip(b, order):
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.004, f"{ACC[k]:.3f}", ha="center", fontsize=9,
            fontweight="bold" if k == "M10" else "normal")
ax.set_ylim(0.40, 0.60); ax.set_ylabel("accuracy direccional (OOS, n=250)")
ax.set_title("M10 supera a las cinco estrategias en accuracy (nominal)"); ax.legend(fontsize=8)
_save(fig, "cap4_headline_accuracy")

# ── Fig 2: curvas de equity ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
for k in ["M10", "M8", "M5", "B&H"]:
    eq = equity_curve(pd.Series(NR[k], index=sub).dropna())
    ax.plot(eq.index, eq.values, label=f"{k} (×{eq.iloc[-1]:.2f})", color=COL[k], lw=2 if k == "M10" else 1.3)
ax.axhline(1.0, color="k", lw=0.7); ax.set_ylabel("equity (€1 inicial)")
ax.set_title("Curvas de equity OOS (ilustración económica)"); ax.legend(fontsize=8)
_save(fig, "cap4_equity")

# ── Fig 3: forest plot de significancia (accuracy M10 + IC95, OOS + 3 splits) ─────
ROB = _J("m10_smci_valtest_robustez")
rows = [("OOS completo", ACC["M10"], len(sub))]
for s in ROB["robustez_splits"]:
    rows.append((f"test {int(s['frac_val']*100)}/{100-int(s['frac_val']*100)}", s["test"]["m10"]["acc"], s["test"]["n"]))
fig, ax = plt.subplots(figsize=(8, 3.6))
ys = np.arange(len(rows))[::-1]
for yy, (lbl, acc, n) in zip(ys, rows):
    lo, hi = _wilson(int(round(acc * n)), n)
    ax.errorbar(acc, yy, xerr=[[acc - lo], [hi - acc]], fmt="o", color=COL["M10"], capsize=4)
    ax.text(hi + 0.005, yy, f"{acc:.3f} (n={n})", va="center", fontsize=8)
ax.axvline(0.5, color="k", ls="--", lw=1, label="azar")
ax.axvline(NIR, color="#7b5cc4", ls=":", lw=1.6, label=f"clase mayoritaria ({NIR:.3f})")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel("accuracy de M10 con IC95 (Wilson)")
ax.set_title("M10: accuracy e IC95 en el OOS y en las particiones"); ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0.40, 0.75)
_save(fig, "cap4_forest_significancia")

# ── Fig 4: ablación + SHAP (desde JSON cap4_prep) ────────────────────────────────
PREP = _J("m10_smci_cap4_prep")
abl = PREP["ablacion"]["accuracy"]; shap_top = PREP["shap"]["top10"]; shap_share = PREP["shap"]["cuota_strata7"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ks = ["agente15", "strata7", "all22"]; labs = ["agente-15", "STRATA-7", "22 (todas)"]
ax[0].bar(labs, [abl[k] for k in ks], color=["#bdbdbd", "#2c7fb8", "#1a5276"], edgecolor="black")
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].axhline(NIR, color="#7b5cc4", ls=":", lw=1.4)
for i, k in enumerate(ks):
    ax[0].text(i, abl[k] + 0.004, f"{abl[k]:.3f}", ha="center", fontsize=10)
ax[0].set_ylim(0.40, 0.60); ax[0].set_ylabel("accuracy"); ax[0].set_title("Ablación: las 7 señales STRATA aportan la ventaja")
feats = list(shap_top)[::-1]; vals = [shap_top[f] for f in feats]
colors = ["#2c7fb8" if f in STRATA7 else "#bdbdbd" for f in feats]
ax[1].barh(feats, vals, color=colors, edgecolor="black", lw=0.4)
ax[1].set_title(f"SHAP: STRATA/régimen = {shap_share:.0%} de la importancia")
ax[1].legend(handles=[Patch(color="#2c7fb8", label="STRATA/régimen"), Patch(color="#bdbdbd", label="agente")], fontsize=8, loc="lower right")
_save(fig, "cap4_ablacion_shap")

# ── Fig 5: régimen sobre el precio (calibración + OOS) ────────────────────────────
_LBL = {0: "Calma", 1: "Estrés", 2: "Crisis"}


def _regplot(ax_top, ax_bot, feats_w, title):
    st = pd.Series(hmm.predict_states(feats_w.to_numpy()), index=feats_w.index)
    px = prices["Close"].reindex(st.index).ffill()
    lo, hi = float(px.min()), float(px.max())
    ax_top.plot(px.index, px.values, color="black", lw=0.9)
    for s in (0, 1, 2):
        ax_top.fill_between(st.index, lo, hi, where=(st == s).to_numpy(), color=REG[_LBL[s]], alpha=0.15, step="post")
    ax_top.set_ylim(lo * 0.99, hi * 1.01); ax_top.set_ylabel(f"{TICKER}"); ax_top.set_title(title)
    ax_top.legend(handles=[Patch(facecolor=REG[_LBL[s]], alpha=0.5, label=_LBL[s]) for s in (0, 1, 2)], fontsize=7, ncol=3, loc="upper left")
    for s in (0, 1, 2):
        ax_bot.fill_between(st.index, 0, (st == s).astype(int).to_numpy(), color=REG[_LBL[s]], alpha=0.9, step="post")
    ax_bot.set_ylim(0, 1); ax_bot.set_yticks([]); ax_bot.set_xlabel("Fecha")


oos_feat = feat_df.loc[feat_df.index >= pd.Timestamp(STRATA_OOS_START)]
fig, ax = plt.subplots(2, 2, figsize=(13, 5.2), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.15})
_regplot(ax[0, 0], ax[1, 0], calib, f"Régimen HMM · calibración ({calib.index[0].date()}→{calib.index[-1].date()})")
_regplot(ax[0, 1], ax[1, 1], oos_feat, f"Régimen HMM · OOS ({oos_feat.index[0].date()}→{oos_feat.index[-1].date()})")
_save(fig, "cap4_regimen_precio")

# ── Fig 6: matriz de transición ──────────────────────────────────────────────────
A = np.asarray(hmm.transition_matrix); labs = ["Calma", "Estrés", "Crisis"]
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white" if A[i, j] > 0.5 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs)
ax.set_xlabel("régimen en t+1"); ax.set_ylabel("régimen en t"); ax.set_title("Matriz de transición del HMM"); ax.grid(False)
fig.colorbar(im, fraction=0.046, pad=0.04)
_save(fig, "cap4_transicion")

# ── Fig 7: RAM-gate (bimodal + accuracy por τ) ───────────────────────────────────
pcalma = gamma.loc[gamma.index <= pd.Timestamp(CALIBRATION_END), "Calma"].to_numpy()
ycal = (ret.reindex(calib.index) > 0).astype(int).to_numpy()
base = float(ycal.mean()); grid = np.linspace(0.1, 0.95, 18)
accs = [float((ycal[pcalma >= t] == 1).mean()) if (pcalma >= t).sum() else np.nan for t in grid]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
ax[0].hist(pcalma, bins=30, color="#999"); ax[0].set_yscale("log"); ax[0].axvline(0.5, color="blue", lw=2, label="τ=0,5")
ax[0].set_xlabel("P(Calma) filtrada"); ax[0].set_title("RAM score bimodal"); ax[0].legend(fontsize=8)
ax[1].plot(grid, accs, "o-", color="#3a7"); ax[1].axhline(base, color="k", ls="--", lw=0.9, label=f"tasa base {base:.3f}")
ax[1].axvline(0.5, color="blue", lw=1.5); ax[1].set_xlabel("umbral τ"); ax[1].set_ylabel("accuracy 'largo'")
ax[1].set_title("Accuracy plana ⇒ τ no es un parámetro fino"); ax[1].legend(fontsize=8)
_save(fig, "cap4_ram_gate")

# ── Fig 8: robustez a la partición (desde JSON) ──────────────────────────────────
splits = ROB["robustez_splits"]; labels = [f"{int(s['frac_val']*100)}/{100-int(s['frac_val']*100)}" for s in splits]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for col, part in zip(ax, ["validacion", "test"]):
    xb = np.arange(len(splits)); w = 0.2
    for j, k in enumerate(["m10", "bh", "m8", "majority"]):
        col.bar(xb + (j - 1.5) * w, [s[part][k]["acc"] for s in splits], w, label=k.upper(),
                color={"m10": COL["M10"], "bh": COL["B&H"], "m8": COL["M8"], "majority": COL["mayoría"]}[k], edgecolor="black", lw=0.5)
    col.axhline(0.5, color="k", ls="--", lw=0.8); col.set_xticks(xb); col.set_xticklabels(labels)
    col.set_ylabel("accuracy"); col.set_title(part); col.legend(fontsize=7); col.set_ylim(0.40, 0.65)
fig.suptitle("Robustez a la partición (M10 gana en validación y test)", fontsize=11)
_save(fig, "cap4_robustez_particion")

# ── Fig 9: accuracy rodante + % ventanas (desde JSON; sin embargo, ver §4.5) ──────
ROLL = _J("m10_smci_rolling")
r63 = ROLL["rolling63"]; fechas = pd.to_datetime(r63["fecha_fin"])
frac = ROLL["frac_ventanas_m10_gana"]; wins = list(frac)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k, c in [("m10", COL["M10"]), ("bh", COL["B&H"]), ("m5", COL["M5"]), ("m8", COL["M8"])]:
    ax[0].plot(fechas, r63[k], label=k.upper(), color=c, lw=1.8 if k == "m10" else 1.1)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].set_ylabel("accuracy (ventana 63 d)")
ax[0].set_title("Accuracy rodante"); ax[0].legend(fontsize=8)
ax[1].bar(wins, [frac[w]["m10_gt_bh"] for w in wins], color=COL["M10"], edgecolor="black"); ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
for i, w in enumerate(wins):
    ax[1].text(i, frac[w]["m10_gt_bh"] + 0.02, f"{frac[w]['m10_gt_bh']:.0%}", ha="center", fontsize=9)
ax[1].set_xlabel("tamaño de ventana (días)"); ax[1].set_ylabel("% ventanas M10 > B&H"); ax[1].set_title("Consistencia rodante"); ax[1].set_ylim(0, 1)
_save(fig, "cap4_rolling")

# ── Fig 9b: un día de intervención por dentro (régimen + flip de posición) ────────
_interv = mv.index[mv["intervenido"] & mv["r_next"].notna()]
_t = _interv[len(_interv) // 2]; _r = mv.loc[_t]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(["Calma", "Estrés", "Crisis"], [_r["calm_prob"], _r["stress_prob"], _r["crisis_prob"]],
          color=[REG[k] for k in ["Calma", "Estrés", "Crisis"]], edgecolor="black")
ax[0].axhline(0.5, color="blue", ls="--", lw=1.2, label="τ = 0,5 (gate RAM)")
ax[0].set_ylim(0, 1); ax[0].set_ylabel("posterior filtrado del HMM"); ax[0].set_title(f"Régimen el {_t.date()}"); ax[0].legend(fontsize=8)
_ad = "corto" if _r["agent_size"] < 0 else "largo"; _fd = "corto" if _r["final_size"] < 0 else "largo"
ax[1].bar(["agente (M5)", "STRATA (M8)"], [_r["agent_size"], _r["final_size"]], color=["#9e9e9e", "#f0a830"], edgecolor="black")
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("posición $w_t$")
ax[1].set_title(f"RAM voltea {_ad}→{_fd} (r$_{{t+1}}$={_r['r_next']:+.3f})")
_save(fig, "cap4_dia_intervencion")

# ── Fig 10: robustez a la ventana de calibración (desde JSON) ────────────────────
CAL = _J("smci_calib_window")["por_ventana"]
yrs = [v["start"][:4] for v in CAL]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(yrs, [v["m10_acc"] for v in CAL], "o-", color=COL["M10"], lw=2, label="M10")
ax[0].axhline(0.484, color=COL["M5"], ls="--", lw=1.2, label="M5 = 0,484"); ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
ax[0].set_xlabel("inicio de calibración (← más historia)"); ax[0].set_ylabel("accuracy M10"); ax[0].set_title("M10 degrada al acortar la calibración"); ax[0].legend(fontsize=8); ax[0].set_ylim(0.44, 0.57)
ax[1].plot(yrs, [v["medias_regimen"]["Crisis"] for v in CAL], "s-", color="#c0392b", lw=2); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("inicio de calibración (← más historia)"); ax[1].set_ylabel("media r en Crisis"); ax[1].set_title("Crisis no se vuelve direccional al acortar")
_save(fig, "cap4_calibracion")

print(f"OK · {len(list(FIG.glob('cap4_*.pdf')))} figuras en {FIG}")
