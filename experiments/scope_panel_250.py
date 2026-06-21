"""Panel comparativo sobre la ventana común de ~250 días (post burn-in walk-forward de M10).

Para cada activo con caché de agente completa, todas las estrategias (M5, M8, M10, RAM crudo,
B&H, S&H, ZeroR) se evalúan sobre el MISMO tramo `sub` (los días donde M10 tiene predicción),
así son comparables entre sí y con M10. Reúne todo en una tabla única.

Reutiliza outputs/experiments/scope_oneoff_<TK>.json si existe; si no, lo genera.
Escribe outputs/experiments/scope_panel_250.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.analyze_one_ticker import analyze

# Orden del panel: 10 canónicos + ampliación de índices/ETF (leverage fuerte)
PANEL = ["SPY", "NVDA", "BAC", "TSLA", "XLE", "UNG", "MSTR", "SMCI", "ROKU", "MARA", "QQQ", "DIA"]


def _calmar(equity: float, max_dd: float, n: int) -> float:
    """Calmar = CAGR anualizado / |maxDD|, con 252 días bursátiles."""
    if max_dd == 0 or n == 0:
        return float("nan")
    cagr = equity ** (252.0 / n) - 1.0
    return cagr / abs(max_dd)


def main() -> None:
    out_dir = Path("outputs/experiments")
    rows = []
    for tk in PANEL:
        cache = out_dir / f"scope_oneoff_{tk}.json"
        res = json.loads(cache.read_text()) if cache.exists() else analyze(tk)
        if not cache.exists():
            cache.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        n = res["n_eval"]
        for nm, d in res["estrategias"].items():
            rows.append({
                "ticker": tk, "estrategia": nm, "n": n, "frac_up": res["frac_up"],
                "acc": d["accuracy"], "sharpe": d["sharpe"], "equity": d["equity"],
                "max_dd": d["max_dd"], "calmar": round(_calmar(d["equity"], d["max_dd"], n), 4),
            })

    panel = {
        "ventana": "OOS post burn-in walk-forward de M10 (~250 d), común a todas las estrategias por activo",
        "n_activos": len(PANEL),
        "filas": rows,
        "por_activo": {
            json.loads((out_dir / f"scope_oneoff_{tk}.json").read_text())["ticker"]: {
                k: json.loads((out_dir / f"scope_oneoff_{tk}.json").read_text())[k]
                for k in ("n_eval", "frac_up", "naturaleza", "canal_regimen", "canal_m10",
                          "lectura_canal_regimen")
            } for tk in PANEL
        },
    }
    (out_dir / "scope_panel_250.json").write_text(json.dumps(panel, indent=2, ensure_ascii=False))

    # --- Impresión: una tabla por activo + matriz resumen de accuracy ---
    estr_order = ["M5 (agente)", "M8 (STRATA)", "M10 (meta-learner)", "Régimen (RAM crudo)",
                  "B&H (siempre largo)", "S&H (siempre corto)", "Mayoría (ZeroR/NIR)"]
    by_tk: dict[str, dict] = {}
    for r in rows:
        by_tk.setdefault(r["ticker"], {})[r["estrategia"]] = r

    for tk in PANEL:
        d = json.loads((out_dir / f"scope_oneoff_{tk}.json").read_text())
        nt, cr, cm = d["naturaleza"], d["canal_regimen"], d["canal_m10"]
        print(f"\n=== {tk}  (n={d['n_eval']}, frac_up={d['frac_up']}) ===")
        print(f"  naturaleza: leverage={nt['leverage_corr']:+.3f} crisisOOS={nt['oos_crisis_frac']:.2f} "
              f"agente_corto={nt['agent_short_frac']:.2f} vol={nt['oos_vol_media']:.3f}")
        print(f"  canal M10: acc={cm['acc_m10']:.3f} skill_p={cm['skill_p_1cola']:.3f} "
              f"HAC_t={cm['hac_t']:+.2f} McNemar_vs_M5_p={cm['mcnemar_vs_m5_p']:.3f}")
        print(f"  {'estrategia':22s} {'acc':>6s} {'Sharpe':>8s} {'Calmar':>8s} {'equity':>8s} {'maxDD':>8s}")
        for nm in estr_order:
            r = by_tk[tk][nm]
            print(f"  {nm:22s} {r['acc']:6.3f} {r['sharpe']:8.2f} {r['calmar']:8.2f} "
                  f"{r['equity']:8.3f} {r['max_dd']:8.2%}")

    # Matriz accuracy: filas = activos, columnas = estrategias
    print("\n\n=== MATRIZ DE ACCURACY (todos los activos, ventana 250 d) ===")
    cols = ["M5", "M8", "M10", "RAM", "B&H", "ZeroR"]
    key = {"M5": "M5 (agente)", "M8": "M8 (STRATA)", "M10": "M10 (meta-learner)",
           "RAM": "Régimen (RAM crudo)", "B&H": "B&H (siempre largo)", "ZeroR": "Mayoría (ZeroR/NIR)"}
    print(f"  {'ticker':6s} " + " ".join(f"{c:>7s}" for c in cols) + f" {'frac_up':>8s}")
    for tk in PANEL:
        line = f"  {tk:6s} " + " ".join(f"{by_tk[tk][key[c]]['acc']:7.3f}" for c in cols)
        print(line + f" {by_tk[tk]['M5 (agente)']['frac_up']:8.3f}")


if __name__ == "__main__":
    main()
