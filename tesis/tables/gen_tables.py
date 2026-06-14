"""Genera tablas LaTeX de la memoria a partir de outputs/experiments/*.json.

Principio (instrucciones_redaccion.md): las CIFRAS NUNCA SE ESCRIBEN A MANO en la
memoria. Esta utilidad lee los JSON ejecutados y emite ficheros .tex que el documento
incluye con \\input{tables/...}. Si una cifra cambia al re-ejecutar un experimento, se
regenera la tabla y la memoria queda sincronizada.

Uso:  python tesis/tables/gen_tables.py
Salida: tesis/tables/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "experiments"
OUT = Path(__file__).resolve().parent


def _load(name: str) -> dict | None:
    fp = EXP / f"{name}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def tabla_m5_m8() -> str | None:
    """Tabla M5 (agente solo) vs M8 (STRATA, base P95/P99) desde el JSON de E4."""
    d = _load("psa_gso_threshold_sensitivity")
    if not d:
        return None
    m5, m8 = d["m5"], d["base"]
    rows = [
        ("Acierto direccional", _pct(m5["accuracy"]), _pct(m8["accuracy"])),
        ("Hit rate", _pct(m5["hit_rate"]), _pct(m8["hit_rate"])),
        ("Sharpe (causal)", f"{m5['sharpe_causal']:+.2f}", f"{m8['sharpe_causal']:+.2f}"),
        ("MCC", f"{m5['mcc']:+.3f}", f"{m8['mcc']:+.3f}"),
        ("\\euro{}1000 $\\to$", f"{m5['equity_final_1000']:.0f}", f"{m8['equity_final_1000']:.0f}"),
    ]
    body = "\n".join(f"        {c} & {a} & {b} \\\\" for c, a, b in rows)
    return (
        "% AUTOGENERADO por tesis/tables/gen_tables.py — no editar a mano.\n"
        "% Fuente: outputs/experiments/psa_gso_threshold_sensitivity.json (m5, base)\n"
        "\\begin{tabular}{lcc}\n    \\toprule\n"
        "    Métrica & Agente solo (M5) & STRATA (M8) \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n\\end{{tabular}}\n"
    )


def tabla_e4_sensibilidad() -> str | None:
    """Barrido de sensibilidad de los umbrales de PSA/GSO (experimento E4)."""
    d = _load("psa_gso_threshold_sensitivity")
    if not d:
        return None
    rows = []
    for p in d["sweep"]:
        rows.append(
            f"        {p['detector'].upper()} & {p['pctile']} & {p['n_intervenciones_detector']} & "
            f"{_pct(p['accuracy'])} & {p['delta_accuracy_vs_base']:+.3f} & {p['sharpe_causal']:+.2f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% AUTOGENERADO por tesis/tables/gen_tables.py — no editar a mano.\n"
        "% Fuente: outputs/experiments/psa_gso_threshold_sensitivity.json (sweep)\n"
        "\\begin{tabular}{llccccc}\n    \\toprule\n"
        "    Detector & Pctil & N interv. & Acierto & $\\Delta$acc & Sharpe \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n\\end{{tabular}}\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generadas = []
    for nombre, fn in (("tabla_m5_m8", tabla_m5_m8), ("tabla_e4_sensibilidad", tabla_e4_sensibilidad)):
        tex = fn()
        if tex is None:
            print(f"[skip] {nombre}: falta el JSON fuente")
            continue
        (OUT / f"{nombre}.tex").write_text(tex)
        generadas.append(nombre)
    print(f"OK · {len(generadas)} tablas generadas en {OUT}: {', '.join(generadas)}")


if __name__ == "__main__":
    main()
