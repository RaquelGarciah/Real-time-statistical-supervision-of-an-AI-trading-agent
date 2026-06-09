"""Figuras del experimento unificado de 9 configuraciones (M1–M9).

Plan: 14 figuras + 2 tablas auxiliares (CLAUDE.md §10), organizadas en cuatro
bloques. Cada figura se persiste en doble formato PNG/HTML en
``outputs/figures/<TICKER>/`` (p. ej. ``outputs/figures/SPY/``).

Ejecutar con ``python -m viz.comparison [--ticker SPY]`` tras tener todos los
``outputs/experiments/m{1..9}.json`` y ``statistical_tests.json`` del activo. Los
módulos que no encuentren datos suficientes emiten un aviso y siguen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import EXPERIMENTS_DIR, FIGURES_DIR
from core.metrics import equity_curve
from viz.shared import PALETTE, REGIME_COLOR, save_figure, setup_matplotlib

# Globals reasignables por ticker (ver generate_all): cada activo escribe en su carpeta.
OUT_DIR = FIGURES_DIR / "SPY"
EXP_DIR = EXPERIMENTS_DIR

# Etiquetas legibles de las 9 configuraciones del experimento unificado.
CONFIGS: tuple[str, ...] = (
    "m1_buy_and_hold",
    "m2_bh_garchhmm",
    "m3_ml_naive",
    "m4_ml_strata",
    "m5_agent_alone",
    "m6_strata_warn",
    "m7_strata_reduce",
    "m8_strata_override",
    "m9_ml_ai",
)
LABELS: dict[str, str] = {
    "m1_buy_and_hold": "M1 — B&H",
    "m2_bh_garchhmm": "M2 — B&H + GARCH×HMM",
    "m3_ml_naive": "M3 — H2O (KFold)",
    "m4_ml_strata": "M4 — H2O (CPCV) + sizing",
    "m5_agent_alone": "M5 — Agente solo",
    "m6_strata_warn": "M6 — STRATA warn",
    "m7_strata_reduce": "M7 — STRATA reduce",
    "m8_strata_override": "M8 — STRATA override",
    "m9_ml_ai": "M9 — ML + IA",
}


def _load(name: str) -> dict | None:
    p = EXP_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _net_returns_df() -> pd.DataFrame:
    """DataFrame de net_returns con intersección de fechas."""
    cols: dict[str, pd.Series] = {}
    for name in CONFIGS:
        d = _load(name)
        if d is None:
            continue
        s = pd.Series(
            {pd.Timestamp(k): v for k, v in d["net_returns"].items()}, name=name
        )
        cols[name] = s
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().dropna(how="any")


def _window_label(df: pd.DataFrame) -> str:
    if df.empty:
        return "OOS sin datos"
    return f"{len(df)} días ({df.index[0].date()} → {df.index[-1].date()})"


# =============================================================================
# Bloque comparativo principal (5 figuras)
# =============================================================================

def figura_01_equity():
    setup_matplotlib()
    nets = _net_returns_df()
    if nets.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_p = go.Figure()
    for i, name in enumerate(nets.columns):
        eq = equity_curve(nets[name])
        color = PALETTE[i % len(PALETTE)]
        lw = 2.6 if name == "m5_agent_alone" else 1.6
        ax.plot(eq.index, eq.values, color=color, label=LABELS[name], linewidth=lw)
        fig_p.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            name=LABELS[name], line={"color": color, "width": 3 if name == "m5_agent_alone" else 2},
        ))
    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":")
    ax.set_title(f"01 — Equity curves de las 9 configuraciones (SPY, {_window_label(nets)})")
    ax.set_ylabel("Equity (1.0 = capital inicial)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig_p.update_layout(
        title=f"01 — Equity curves (SPY, {_window_label(nets)})",
        template="simple_white", yaxis_title="Equity",
    )
    save_figure(fig, fig_p, "01_equity_curves", OUT_DIR)


def figura_02_drawdowns():
    setup_matplotlib()
    nets = _net_returns_df()
    if nets.empty:
        return
    n = len(nets.columns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.1 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]
    fig_p = go.Figure()
    for i, name in enumerate(nets.columns):
        eq = equity_curve(nets[name])
        dd = eq / eq.cummax() - 1
        color = PALETTE[i % len(PALETTE)]
        axes[i].fill_between(dd.index, dd.values, 0, color=color, alpha=0.6)
        axes[i].set_ylabel(LABELS[name].split(" — ")[0], fontsize=8)
        fig_p.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=LABELS[name],
            fill="tozeroy", line={"color": color},
        ))
    axes[-1].set_xlabel("Fecha")
    fig.suptitle(f"02 — Drawdowns apilados ({_window_label(nets)})")
    fig_p.update_layout(title="02 — Drawdowns apilados", template="simple_white")
    save_figure(fig, fig_p, "02_drawdowns", OUT_DIR)


def figura_03_tabla_metricas():
    """Tabla 9 × 7 con código de color (Sharpe, DSR, MaxDD, Calmar, Ret, Vol, Hit)."""
    setup_matplotlib()
    stats = _load("statistical_tests")
    configs = stats["configurations"] if stats else []
    if not configs:
        return

    rows = []
    for c in configs:
        if not c["available"]:
            rows.append([c["config"], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
        d = _load(c["config"])
        m = d["metrics"]
        rows.append([
            c["config"],
            m["sharpe"],
            c["dsr"],
            m["max_drawdown"],
            m.get("calmar", float("nan")),
            float(np.array(list(d["net_returns"].values()), dtype=float).sum()),
            m["hit_rate"],
        ])
    df = pd.DataFrame(rows, columns=[
        "Config", "Sharpe", "DSR", "MaxDD", "Calmar", "Ret total", "Hit rate",
    ])
    df["Config"] = df["Config"].map(LABELS).fillna(df["Config"])

    # Render: matplotlib heatmap por columna numérica.
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.axis("off")
    cell_text = []
    for _, r in df.iterrows():
        cell_text.append([
            r["Config"],
            f"{r['Sharpe']:+.3f}" if not pd.isna(r["Sharpe"]) else "—",
            f"{r['DSR']:+.3f}" if not pd.isna(r["DSR"]) else "—",
            f"{r['MaxDD']:+.3f}" if not pd.isna(r["MaxDD"]) else "—",
            f"{r['Calmar']:+.2f}" if not pd.isna(r["Calmar"]) else "—",
            f"{r['Ret total']:+.3f}" if not pd.isna(r["Ret total"]) else "—",
            f"{r['Hit rate']:.2f}" if not pd.isna(r["Hit rate"]) else "—",
        ])
    table = ax.table(
        cellText=cell_text,
        colLabels=list(df.columns),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    # Color por columna (verde si arriba, rojo si abajo).
    for col_idx, col_name in enumerate(df.columns):
        if col_name == "Config":
            continue
        vals = df[col_name].astype(float)
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmax == vmin:
            continue
        for row_idx in range(len(df)):
            v = vals.iloc[row_idx]
            if pd.isna(v):
                continue
            t = (v - vmin) / (vmax - vmin)
            # Para MaxDD, "menos negativo es mejor": invertimos.
            if col_name == "MaxDD":
                t = 1 - t
            color = plt.cm.RdYlGn(t)
            table[(row_idx + 1, col_idx)].set_facecolor(color)
    ax.set_title(f"03 — Tabla de métricas 9 × 7 ({_window_label(_net_returns_df())})")
    fig_p = go.Figure(data=[go.Table(
        header={"values": list(df.columns), "fill_color": "#0072B2", "font": {"color": "white"}},
        cells={"values": [df[c].tolist() for c in df.columns]},
    )])
    fig_p.update_layout(title="03 — Tabla de métricas")
    save_figure(fig, fig_p, "03_tabla_metricas", OUT_DIR)


def figura_04_pvalues_matrix():
    """Matriz 9 × 9 de p-valores Diebold-Mariano."""
    setup_matplotlib()
    stats = _load("statistical_tests")
    if stats is None:
        return
    matrix = stats["dm_p_matrix"]
    order = stats["config_order"]
    labels = [LABELS.get(o, o).split(" — ")[0] for o in order]

    arr = np.array([[np.nan if v is None else v for v in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(order)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), labels, fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title(f"04 — Diebold-Mariano p-values 9 × 9 ({_window_label(_net_returns_df())})")
    fig.colorbar(im, ax=ax, shrink=0.7, label="p-value (verde = signif.)")
    fig_p = go.Figure(go.Heatmap(z=arr.tolist(), x=labels, y=labels,
                                  colorscale="RdYlGn_r", zmin=0, zmax=0.5))
    fig_p.update_layout(title="04 — Matriz de p-values DM (fila vs columna)")
    save_figure(fig, fig_p, "04_pvalues_matrix", OUT_DIR)


def figura_05_sizing_series():
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_p = go.Figure()
    for i, name in enumerate(CONFIGS):
        d = _load(name)
        if d is None or "weights" not in d:
            continue
        s = pd.Series({pd.Timestamp(k): v for k, v in d["weights"].items()}).sort_index()
        color = PALETTE[i % len(PALETTE)]
        ax.plot(s.index, s.values, color=color, label=LABELS[name], alpha=0.85, linewidth=1.2)
        fig_p.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                   name=LABELS[name], line={"color": color}))
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"05 — Series temporales de sizing por configuración")
    ax.set_ylabel("Peso (sizing en [-1, 1])")
    ax.legend(fontsize=7, ncol=2)
    fig_p.update_layout(title="05 — Sizing por configuración", template="simple_white",
                        yaxis_title="Peso")
    save_figure(fig, fig_p, "05_sizing_series", OUT_DIR)


# =============================================================================
# Bloque explicativo (3 figuras)
# =============================================================================

def figura_06_regimenes_hmm():
    """Timeline dual-panel del régimen HMM sobre SPY (receta de
    ``replicar_regimen_mercado.md``):

    - Panel superior (3/4 alto): precio de SPY + bandas verticales por régimen.
    - Panel inferior (1/4 alto): banda compacta del régimen activo en cada fecha.
    """
    setup_matplotlib()
    d = _load("m2_bh_garchhmm")
    if d is None or "viterbi_states" not in d:
        return
    state_label = {0: "Calma", 1: "Estrés", 2: "Crisis"}
    states = pd.Series(
        {pd.Timestamp(k): int(v) for k, v in d["viterbi_states"].items()}
    ).sort_index()

    # Cargar precios de SPY sobre la ventana de regímenes para el panel superior.
    from core.data import load_market_data
    start = states.index.min().strftime("%Y-%m-%d")
    end = (states.index.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        spy = load_market_data("SPY", start, end)
        price = spy["Close"].reindex(states.index).ffill()
    except Exception:
        # Fallback: usa sigma_t como proxy si no hay precios.
        price = pd.Series(
            {pd.Timestamp(k): v for k, v in d["sigma_t"].items()}
        ).sort_index()

    # --- Matplotlib (dual panel, sharex) ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    # Panel superior: precio + bandas coloreadas por régimen
    ax_top.plot(price.index, price.values, color="black", linewidth=1.4, label="SPY Close")
    y_min, y_max = float(price.min()), float(price.max())
    for st_id in (0, 1, 2):
        mask = (states == st_id).astype(int)
        ax_top.fill_between(
            states.index, y_min, y_max,
            where=mask.astype(bool),
            color=REGIME_COLOR[state_label[st_id]],
            alpha=0.15, step="post",
        )
    ax_top.set_ylim(y_min * 0.99, y_max * 1.01)
    ax_top.set_ylabel("SPY Close")
    ax_top.set_title(f"06 — Régimen HMM sobre SPY ({_window_label(_net_returns_df())})")

    # Leyenda con parches de color (sin duplicados)
    from matplotlib.patches import Patch
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.4, label="SPY Close"),
    ]
    for st_id, label in state_label.items():
        handles.append(Patch(facecolor=REGIME_COLOR[label], alpha=0.5, label=label))
    ax_top.legend(handles=handles, fontsize=8, loc="upper left", ncol=4)

    # Panel inferior: banda compacta de régimen activo
    for st_id in (0, 1, 2):
        mask = (states == st_id).astype(int)
        ax_bot.fill_between(
            states.index, 0, mask,
            color=REGIME_COLOR[state_label[st_id]],
            alpha=0.9, step="post",
        )
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Régimen", fontsize=9)
    ax_bot.set_xlabel("Fecha")

    # --- Plotly equivalente (un trazo de precio + scatter por régimen) ---
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=price.index, y=price.values, mode="lines",
        name="SPY Close", line={"color": "black", "width": 2},
    ))
    for st_id, label in state_label.items():
        sel = states[states == st_id]
        fig_p.add_trace(go.Scatter(
            x=sel.index, y=price.reindex(sel.index).values,
            mode="markers", name=label,
            marker={"color": REGIME_COLOR[label], "size": 8, "opacity": 0.7},
        ))
    fig_p.update_layout(
        title="06 — Régimen HMM sobre SPY",
        template="simple_white", yaxis_title="SPY Close",
    )

    save_figure(fig, fig_p, "06_regimenes_hmm", OUT_DIR)


def figura_06b_regimenes_hmm_calibracion():
    """Misma figura dual-panel del régimen HMM pero sobre el periodo de
    calibración 2000-01-01 → 2024-09-30. Sirve para ilustrar el comportamiento
    del modelo sobre los datos con los que fue entrenado, complementando la
    figura 06 (OOS unificado).
    """
    import pickle

    setup_matplotlib()
    from config import CACHE_MODELS_DIR, CALIBRATION_END, CALIBRATION_START
    from core.data import load_sp500_and_vix
    from core.features import build_feature_matrix

    hmm_path = CACHE_MODELS_DIR / "hmm.pkl"
    if not hmm_path.exists():
        return
    with open(hmm_path, "rb") as f:
        hmm = pickle.load(f)

    market = load_sp500_and_vix(CALIBRATION_START, CALIBRATION_END)
    feats = build_feature_matrix(market).dropna()
    X = feats[["ret_log", "rv_21_ann"]].to_numpy()
    raw_states = hmm.predict_states(X)
    states = pd.Series(raw_states, index=feats.index)

    price = market["close_spx"].reindex(states.index).ffill()

    state_label = {0: "Calma", 1: "Estrés", 2: "Crisis"}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )
    ax_top.plot(price.index, price.values, color="black", linewidth=1.0, label="SPY Close")
    y_min, y_max = float(price.min()), float(price.max())
    for st_id in (0, 1, 2):
        mask = (states == st_id).astype(int)
        ax_top.fill_between(
            states.index, y_min, y_max,
            where=mask.astype(bool),
            color=REGIME_COLOR[state_label[st_id]],
            alpha=0.15, step="post",
        )
    ax_top.set_ylim(y_min * 0.99, y_max * 1.01)
    ax_top.set_ylabel("SPY Close")
    ax_top.set_title(
        f"06b — Régimen HMM sobre SPY · calibración "
        f"({states.index[0].date()} → {states.index[-1].date()}, {len(states)} días)"
    )
    from matplotlib.patches import Patch
    handles = [plt.Line2D([0], [0], color="black", linewidth=1.0, label="SPY Close")]
    for st_id, label in state_label.items():
        handles.append(Patch(facecolor=REGIME_COLOR[label], alpha=0.5, label=label))
    ax_top.legend(handles=handles, fontsize=8, loc="upper left", ncol=4)

    for st_id in (0, 1, 2):
        mask = (states == st_id).astype(int)
        ax_bot.fill_between(
            states.index, 0, mask,
            color=REGIME_COLOR[state_label[st_id]],
            alpha=0.9, step="post",
        )
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("Régimen", fontsize=9)
    ax_bot.set_xlabel("Fecha")

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=price.index, y=price.values, mode="lines",
        name="SPY Close", line={"color": "black", "width": 1.4},
    ))
    for st_id, label in state_label.items():
        sel = states[states == st_id]
        fig_p.add_trace(go.Scatter(
            x=sel.index, y=price.reindex(sel.index).values,
            mode="markers", name=label,
            marker={"color": REGIME_COLOR[label], "size": 4, "opacity": 0.55},
        ))
    fig_p.update_layout(
        title="06b — Régimen HMM sobre SPY · calibración",
        template="simple_white", yaxis_title="SPY Close",
    )

    save_figure(fig, fig_p, "06b_regimenes_hmm_calibracion", OUT_DIR)


def figura_07_garch_var():
    setup_matplotlib()
    d = _load("m2_bh_garchhmm")
    if d is None:
        return
    sigma = pd.Series({pd.Timestamp(k): v for k, v in d["sigma_t"].items()}).sort_index()
    # VaR 95 % paramétrico aproximado: 1.645 * sigma_daily.
    sigma_daily = sigma / np.sqrt(252)
    var95 = 1.645 * sigma_daily
    nets = _net_returns_df()
    rets = nets["m1_buy_and_hold"].reindex(sigma.index) if "m1_buy_and_hold" in nets.columns else None

    fig, ax = plt.subplots(figsize=(10, 4))
    fig_p = go.Figure()
    if rets is not None:
        ax.bar(rets.index, rets.values, color="grey", alpha=0.4, label="Retornos M1 (B&H)")
        fig_p.add_trace(go.Bar(x=rets.index, y=rets.values, name="Retornos M1",
                               marker={"color": "lightgrey"}))
    ax.plot(var95.index, var95.values, color=PALETTE[3], label="+VaR 95% diario")
    ax.plot(var95.index, -var95.values, color=PALETTE[3])
    ax.set_title(f"07 — σ_t GARCH y bandas VaR 95 % ({_window_label(_net_returns_df())})")
    ax.set_ylabel("Retorno / banda VaR")
    ax.legend(fontsize=8)
    fig_p.add_trace(go.Scatter(x=var95.index, y=var95.values, name="+VaR 95",
                               line={"color": PALETTE[3]}))
    fig_p.add_trace(go.Scatter(x=var95.index, y=-var95.values, name="-VaR 95",
                               line={"color": PALETTE[3]}))
    fig_p.update_layout(title="07 — GARCH + VaR", template="simple_white")
    save_figure(fig, fig_p, "07_garch_var", OUT_DIR)


def figura_08_kfold_vs_cpcv():
    """Compara la AUC del líder H2O en M3 (KFold) vs M4 (CPCV-purged)."""
    setup_matplotlib()
    m3 = _load("m3_ml_naive")
    m4 = _load("m4_ml_strata")
    m9 = _load("m9_ml_ai")
    if m3 is None or m4 is None:
        return

    # Métricas de calibración del líder vs Sharpe OOS.
    rows = []
    for label, d in [("M3 (KFold)", m3), ("M4 (CPCV)", m4), ("M9 (CPCV)", m9)]:
        if d is None:
            continue
        rows.append({
            "label": label,
            "cv_auc": d["h2o_leader"]["metric_value"],
            "oos_sharpe": d["metrics"]["sharpe"],
        })
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["cv_auc"] for r in rows], width=0.4,
           color=PALETTE[0], label="AUC CV (calibración)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["oos_sharpe"] for r in rows], width=0.4,
            color=PALETTE[1], label="Sharpe OOS")
    ax.set_xticks(x, [r["label"] for r in rows], fontsize=9)
    ax.set_ylabel("AUC validación cruzada")
    ax2.set_ylabel("Sharpe OOS")
    ax.set_title("08 — KFold vs CPCV: AUC CV vs Sharpe OOS")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=[r["label"] for r in rows],
                           y=[r["cv_auc"] for r in rows], name="AUC CV",
                           marker_color=PALETTE[0]))
    fig_p.add_trace(go.Bar(x=[r["label"] for r in rows],
                           y=[r["oos_sharpe"] for r in rows], name="Sharpe OOS",
                           marker_color=PALETTE[1], yaxis="y2"))
    fig_p.update_layout(
        title="08 — KFold vs CPCV", barmode="group",
        yaxis={"title": "AUC CV"},
        yaxis2={"title": "Sharpe OOS", "overlaying": "y", "side": "right"},
        template="simple_white",
    )
    save_figure(fig, fig_p, "08_kfold_vs_cpcv", OUT_DIR)


# =============================================================================
# Bloque STRATA (3 figuras)
# =============================================================================

def figura_09_detectores():
    """Scores RAM/PSA/GSO desde el JSON de M6 (warn) si está disponible."""
    setup_matplotlib()
    d = _load("m6_strata_warn")
    if d is None:
        return
    # severity_counts agrega; para series temporales detalladas haría falta
    # que el runner persistiera los scores diarios. Mostramos lo que hay:
    sev = d.get("severity_counts", {})
    detectors = ("ram", "psa", "gso")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig_p = go.Figure()
    levels = ["none", "low", "medium", "high"]
    width = 0.25
    x = np.arange(len(levels))
    for i, det in enumerate(detectors):
        counts = sev.get(det, {})
        vals = [counts.get(lv, 0) for lv in levels]
        ax.bar(x + (i - 1) * width, vals, width=width, color=PALETTE[i],
               label=det.upper())
        fig_p.add_trace(go.Bar(x=levels, y=vals, name=det.upper(),
                                marker_color=PALETTE[i]))
    ax.set_xticks(x, levels)
    ax.set_ylabel("nº días")
    ax.set_title(f"09 — Severidad de detectores (M6 warn, {_window_label(_net_returns_df())})")
    ax.legend(fontsize=9)
    fig_p.update_layout(title="09 — Severidad de detectores RAM/PSA/GSO",
                         barmode="group", template="simple_white")
    save_figure(fig, fig_p, "09_detectores", OUT_DIR)


def figura_10_ablacion():
    setup_matplotlib()
    abl = _load("ablation_strata")
    if abl is None:
        return
    configs = abl.get("configs", {})
    if not configs:
        return
    names = [k.replace("ablation_strata_", "") for k in configs]
    sharpe = [configs[k]["metrics"]["sharpe"] for k in configs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, sharpe, color=PALETTE[: len(names)])
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Sharpe (anualizado)")
    ax.set_title(f"10 — Ablación de detectores STRATA (reduce, {_window_label(_net_returns_df())})")
    fig_p = go.Figure(go.Bar(x=sharpe, y=names, orientation="h",
                              marker_color=PALETTE[: len(names)]))
    fig_p.update_layout(title="10 — Ablación STRATA", xaxis_title="Sharpe",
                        template="simple_white")
    save_figure(fig, fig_p, "10_ablacion", OUT_DIR)


def figura_11_estudio_caso():
    """4 paneles para un día representativo con intervención (si existe)."""
    setup_matplotlib()
    m5 = _load("m5_agent_alone")
    m7 = _load("m7_strata_reduce")
    m2 = _load("m2_bh_garchhmm")
    if m5 is None or m7 is None or m2 is None:
        return
    w5 = pd.Series({pd.Timestamp(k): v for k, v in m5["weights"].items()}).sort_index()
    w7 = pd.Series({pd.Timestamp(k): v for k, v in m7["weights"].items()}).sort_index()
    diff = (w5 - w7).abs()
    if diff.max() < 1e-6:
        # No interventions; el caso de estudio se queda como placeholder.
        return
    target_date = diff.idxmax()

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    fig.suptitle(f"11 — Estudio de caso: intervención el {target_date.date()}")
    # P1: precio
    states = pd.Series({pd.Timestamp(k): v for k, v in m2["viterbi_states"].items()}).sort_index()
    sigma = pd.Series({pd.Timestamp(k): v for k, v in m2["sigma_t"].items()}).sort_index()
    axes[0, 0].plot(sigma.index, sigma.values, color="black")
    axes[0, 0].axvline(target_date, color="red", linestyle="--")
    axes[0, 0].set_title("σ_t GARCH")
    # P2: régimen
    state_label = {0: "Calma", 1: "Estrés", 2: "Crisis"}
    for ts, st in states.items():
        axes[0, 1].axvspan(ts, ts + pd.Timedelta(days=1),
                           color=REGIME_COLOR[state_label[int(st)]], alpha=0.3)
    axes[0, 1].axvline(target_date, color="red", linestyle="--")
    axes[0, 1].set_title("Régimen HMM")
    # P3: sizing M5 vs M7
    axes[1, 0].plot(w5.index, w5.values, color=PALETTE[0], label="M5 agente")
    axes[1, 0].plot(w7.index, w7.values, color=PALETTE[3], label="M7 supervisado")
    axes[1, 0].axvline(target_date, color="red", linestyle="--")
    axes[1, 0].set_title("Sizing agente vs STRATA")
    axes[1, 0].legend(fontsize=8)
    # P4: net returns
    nets = _net_returns_df()
    if not nets.empty:
        for name in ("m5_agent_alone", "m7_strata_reduce"):
            if name in nets.columns:
                axes[1, 1].plot(nets.index, nets[name].cumsum(),
                                 label=LABELS[name])
        axes[1, 1].axvline(target_date, color="red", linestyle="--")
        axes[1, 1].set_title("Σ retornos netos")
        axes[1, 1].legend(fontsize=8)
    fig_p = go.Figure()
    fig_p.add_annotation(text=f"Día de intervención: {target_date.date()}",
                          x=0.5, y=0.5, showarrow=False)
    save_figure(fig, fig_p, "11_estudio_caso", OUT_DIR)


# =============================================================================
# Bloque arquitectural (3 figuras)
# =============================================================================

def figura_12_arquitectura():
    """Diagrama esquemático de STRATA (cajas y flechas)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    boxes = [
        ("Datos\nmercado\n(SPY, VIX,\nsectores)", (0.06, 0.55), PALETTE[0]),
        ("Macro\nsnapshot", (0.22, 0.78), PALETTE[5]),
        ("AI Hedge Fund\n5 personalidades\n+ DeepSeek V3", (0.40, 0.78), PALETTE[1]),
        ("HMM\n3 estados", (0.22, 0.40), PALETTE[2]),
        ("GARCH(1,1)\nStudent-t", (0.22, 0.18), PALETTE[3]),
        ("BOCPD", (0.40, 0.18), PALETTE[4]),
        ("RAM", (0.60, 0.65), PALETTE[2]),
        ("PSA", (0.60, 0.40), PALETTE[4]),
        ("GSO", (0.60, 0.18), PALETTE[3]),
        ("Capa de\nintervención\n(warn/reduce/override)", (0.80, 0.40), PALETTE[6]),
    ]
    for text, (x, y), color in boxes:
        ax.add_patch(plt.Rectangle((x - 0.07, y - 0.06), 0.14, 0.12,
                                     facecolor=color, alpha=0.6, edgecolor="black"))
        ax.text(x, y, text, ha="center", va="center", fontsize=8)
    # Flechas principales
    arrows = [
        ((0.13, 0.61), (0.20, 0.78)),
        ((0.29, 0.78), (0.40, 0.78)),
        ((0.13, 0.55), (0.20, 0.40)),
        ((0.13, 0.49), (0.20, 0.18)),
        ((0.29, 0.40), (0.55, 0.65)),
        ((0.47, 0.78), (0.55, 0.40)),
        ((0.47, 0.18), (0.55, 0.18)),
        ((0.67, 0.65), (0.75, 0.45)),
        ((0.67, 0.40), (0.75, 0.40)),
        ((0.67, 0.18), (0.75, 0.35)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.5})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("12 — Arquitectura STRATA")
    fig_p = go.Figure()
    fig_p.add_annotation(text="Diagrama arquitectural — ver PNG",
                          x=0.5, y=0.5, showarrow=False, font={"size": 16})
    save_figure(fig, fig_p, "12_arquitectura", OUT_DIR)


def figura_13_leaderboard_h2o():
    """Leaderboard H2O del líder de M3 y M9."""
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label, json_name in [
        (axes[0], "M3 (KFold)", "h2o_m3.json"),
        (axes[1], "M9 (CPCV + agente)", "h2o_m9.json"),
    ]:
        path = Path(__file__).resolve().parent.parent / "cache" / "models" / json_name
        if not path.exists():
            ax.text(0.5, 0.5, f"{label}\n(no entrenado)", ha="center", va="center")
            ax.axis("off")
            continue
        meta = json.loads(path.read_text())
        lb = meta["leaderboard"]
        models = [r["model_id"].split("_")[0] for r in lb[:10]]
        aucs = [r.get("auc", list(r.values())[1]) for r in lb[:10]]
        ax.barh(range(len(models)), aucs, color=PALETTE[0], alpha=0.7)
        ax.set_yticks(range(len(models)), models, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(meta["metric_name"].upper())
        ax.set_title(f"{label} — leader: {meta['leader_id'].split('_')[0]}\n"
                     f"{meta['metric_name']}={meta['leader_metric']:.3f}")
    fig.suptitle("13 — Leaderboard H2O AutoML (top 10)")
    fig_p = go.Figure()
    fig_p.add_annotation(text="Leaderboard H2O AutoML — ver PNG",
                          x=0.5, y=0.5, showarrow=False, font={"size": 16})
    save_figure(fig, fig_p, "13_leaderboard_h2o", OUT_DIR)


def figura_14_dashboard_placeholder():
    """Placeholder de la captura del dashboard live (pendiente)."""
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5,
            "14 — Captura del dashboard live\n(pendiente de despliegue)",
            ha="center", va="center", fontsize=14, style="italic")
    fig_p = go.Figure()
    fig_p.add_annotation(text="Dashboard live pendiente",
                          x=0.5, y=0.5, showarrow=False, font={"size": 16})
    save_figure(fig, fig_p, "14_dashboard", OUT_DIR)


# =============================================================================
# Tablas auxiliares
# =============================================================================

def tabla_a_sharpe_dsr_pvalor():
    """Tabla: Sharpe nominal vs deflactado vs p-valor (contra M1) por config."""
    setup_matplotlib()
    stats = _load("statistical_tests")
    if stats is None:
        return
    rows = []
    pairs = stats.get("pairs", {})
    for c in stats["configurations"]:
        if not c["available"]:
            continue
        p_vs_m1 = pairs.get(f"{c['config']}_vs_m1_buy_and_hold", {}).get(
            "diebold_mariano", {}).get("p_value", None)
        rows.append([
            LABELS.get(c["config"], c["config"]),
            f"{c['sharpe']:+.3f}",
            f"{c['dsr']:+.3f}",
            f"{p_vs_m1:.3f}" if p_vs_m1 is not None else "—",
        ])
    cols = ["Config", "Sharpe", "DSR", "p-valor DM vs M1"]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1))
    ax.axis("off")
    ax.table(cellText=rows, colLabels=cols, loc="center")
    ax.set_title("Tabla A — Sharpe nominal vs deflactado vs p-valor")
    fig_p = go.Figure(data=[go.Table(
        header={"values": cols, "fill_color": "#0072B2", "font": {"color": "white"}},
        cells={"values": list(map(list, zip(*rows)))},
    )])
    fig_p.update_layout(title="Tabla A — Sharpe / DSR / p-valor")
    save_figure(fig, fig_p, "tabla_a_sharpe_dsr_pvalor", OUT_DIR)


def tabla_b_momentos():
    """Tabla: momentos estadísticos (media, var, skew, kurt) de retornos diarios."""
    setup_matplotlib()
    nets = _net_returns_df()
    if nets.empty:
        return
    rows = []
    for name in nets.columns:
        x = nets[name].values
        x_mean = float(np.mean(x))
        x_var = float(np.var(x, ddof=1))
        # skew y kurt simples (sin pkg adicional).
        x_std = float(np.std(x, ddof=1))
        if x_std > 0:
            z = (x - x_mean) / x_std
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4) - 3)
        else:
            skew, kurt = 0.0, 0.0
        rows.append([
            LABELS.get(name, name),
            f"{x_mean:+.5f}",
            f"{x_var:.6f}",
            f"{skew:+.3f}",
            f"{kurt:+.3f}",
        ])
    cols = ["Config", "Media", "Varianza", "Skewness", "Kurt. exc."]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1))
    ax.axis("off")
    ax.table(cellText=rows, colLabels=cols, loc="center")
    ax.set_title("Tabla B — Momentos estadísticos de los retornos diarios")
    fig_p = go.Figure(data=[go.Table(
        header={"values": cols, "fill_color": "#0072B2", "font": {"color": "white"}},
        cells={"values": list(map(list, zip(*rows)))},
    )])
    fig_p.update_layout(title="Tabla B — Momentos")
    save_figure(fig, fig_p, "tabla_b_momentos", OUT_DIR)


def tabla_c_ablacion_sizing():
    """Tabla C — Ablación 2 × 2: efecto fold scheme vs sizing en M3/M4.

    Desambigua la comparación M3 vs M4: el script
    ``experiments/diagnose_m3_m4_sizing.py`` aísla dirección (KFold/CPCV)
    y sizing (off/on) sobre los mismos días OOS, mostrando que el delta
    Sharpe M3 − M4 está dominado por el ruido del fold scheme, no por
    el sizing.
    """
    setup_matplotlib()
    abl = _load("m3_m4_sizing_ablation")
    if abl is None:
        return
    cells = abl["cells"]
    deltas = abl["deltas"]
    rows = [
        [
            cells["m3_actual"]["label"],
            f"{cells['m3_actual']['sharpe']:+.3f}",
            f"{cells['m3_actual']['max_drawdown']:+.4f}",
            f"{cells['m3_actual']['abs_w_mean']:.4f}",
            f"{cells['m3_actual']['zero_weight_days']}",
        ],
        [
            cells["m3_plus_sizing"]["label"],
            f"{cells['m3_plus_sizing']['sharpe']:+.3f}",
            f"{cells['m3_plus_sizing']['max_drawdown']:+.4f}",
            f"{cells['m3_plus_sizing']['abs_w_mean']:.4f}",
            f"{cells['m3_plus_sizing']['zero_weight_days']}",
        ],
        [
            cells["m4_no_sizing"]["label"],
            f"{cells['m4_no_sizing']['sharpe']:+.3f}",
            f"{cells['m4_no_sizing']['max_drawdown']:+.4f}",
            f"{cells['m4_no_sizing']['abs_w_mean']:.4f}",
            f"{cells['m4_no_sizing']['zero_weight_days']}",
        ],
        [
            cells["m4_actual"]["label"],
            f"{cells['m4_actual']['sharpe']:+.3f}",
            f"{cells['m4_actual']['max_drawdown']:+.4f}",
            f"{cells['m4_actual']['abs_w_mean']:.4f}",
            f"{cells['m4_actual']['zero_weight_days']}",
        ],
    ]
    cols = ["Celda (dirección, sizing)", "Sharpe", "MaxDD", "|w|.mean", "Días w=0"]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    delta_lines = [
        f"Δ(fold scheme | sin sizing) = {deltas['fold_scheme_no_sizing']:+.3f}",
        f"Δ(sizing | dir KFold)        = {deltas['sizing_kfold_dir']:+.3f}",
        f"Δ(sizing | dir CPCV)         = {deltas['sizing_cpcv_dir']:+.3f}",
        f"Δ(regime Crisis→0 | dir CPCV)= {deltas['regime_crisis_cpcv']:+.3f}",
    ]
    ax.text(
        0.5, 0.18, "\n".join(delta_lines),
        ha="center", va="top", family="monospace", fontsize=9,
        transform=ax.transAxes,
    )
    ax.set_title(
        "Tabla C — Ablación 2 × 2: dirección (KFold / CPCV) × sizing (off / on)\n"
        "El delta M3 − M4 está dominado por el ruido del fold scheme, no por el sizing."
    )

    fig_p = go.Figure(data=[go.Table(
        header={"values": cols, "fill_color": "#0072B2", "font": {"color": "white"}},
        cells={"values": list(map(list, zip(*rows)))},
    )])
    fig_p.update_layout(
        title="Tabla C — Ablación 2×2 sizing M3/M4 | " + " ; ".join(delta_lines)
    )
    save_figure(fig, fig_p, "tabla_c_ablacion_sizing", OUT_DIR)


# =============================================================================
# Orquestador
# =============================================================================

def generate_all(ticker: str = "SPY") -> None:
    """Genera todas las figuras para ``ticker`` en ``outputs/figures/<ticker>/``.

    Los JSON de experimentos se leen de ``outputs/experiments/`` para SPY y de
    ``outputs/experiments/<ticker>/`` para el resto de activos.
    """
    ticker = ticker.upper()
    global OUT_DIR, EXP_DIR
    OUT_DIR = FIGURES_DIR / ticker
    EXP_DIR = EXPERIMENTS_DIR if ticker == "SPY" else EXPERIMENTS_DIR / ticker
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figura_01_equity()
    figura_02_drawdowns()
    figura_03_tabla_metricas()
    figura_04_pvalues_matrix()
    figura_05_sizing_series()
    figura_06_regimenes_hmm()
    figura_06b_regimenes_hmm_calibracion()
    figura_07_garch_var()
    figura_08_kfold_vs_cpcv()
    figura_09_detectores()
    figura_10_ablacion()
    figura_11_estudio_caso()
    figura_12_arquitectura()
    figura_13_leaderboard_h2o()
    figura_14_dashboard_placeholder()
    tabla_a_sharpe_dsr_pvalor()
    tabla_b_momentos()
    tabla_c_ablacion_sizing()
    print(f"Figuras → {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Figuras comparativas M1–M9 por ticker.")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()
    generate_all(args.ticker)
