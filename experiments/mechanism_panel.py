"""Diagnóstico mecánico por activo: ¿qué canal de STRATA rescata y por qué falla el otro? (sin H2O).

Tesis de dos supervisores: STRATA ofrece una REGLA transparente (M8, canal régimen) y un APRENDIZ flexible
(M10/AutoML, canal ML). Cuál rescata depende de si el signo direccional del régimen es fiable fuera de muestra,
medido por (a) `crisis_mean` —media del retorno del MISMO día en el régimen de alta vol: negativa = leverage
estándar (régimen direccional → la regla sirve); positiva = leverage invertido (régimen no direccional → hace
falta el aprendiz)— y (b) la tasa de acierto de M8 en los días que interviene (>0,5 corrige bien; <0,5 mete ruido).

Para cada activo del panel calcula: leverage_corr, crisis_mean, tasa de intervención de M8, acierto de M8 en
intervención, accuracy M5/M8/M10/AutoML, canal ganador, y una etiqueta de mecanismo. La accuracy de AutoML y M10
viene del panel canónico mm25; M5/M8 e intervención se recomputan con run_master (idéntico al panel). Cada "por
qué" del notebook se respalda con este JSON. Uso: python experiments/mechanism_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import experiments.automl_m10 as A

PANEL_FILE = ("outputs/experiments/automl_runs/"
              "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
LEV = json.load(open("outputs/experiments/leverage_screen.json"))["por_activo"]
PK = {"M5": "m5", "M8": "m8", "M10": "m10_xgb", "AutoML": "automl"}
PANEL = ["SPY", "QQQ", "DIA", "IWM", "XLE", "XLF", "XLK", "NVDA", "BAC", "TSLA",
         "MSTR", "SMCI", "ROKU", "MARA", "UNG"]
OUT = Path("outputs/experiments/mechanism_panel.json")


def _etiqueta(crisis_mean: float, interv: float, m8_hit: float, m8_ayuda: bool,
              canal: str, acc: dict) -> str:
    """Mecanismo en una frase, derivado de las condiciones medibles."""
    if interv < 0.03:
        return ("STRATA defiere: el agente ya está alineado con un régimen correcto (intervención ≈0); "
                f"M8≈M5. Si gana una STRATA es el ML ({canal}).") if acc["M10"] > acc["M5"] or acc["AutoML"] > acc["M5"] \
               else "STRATA defiere: el agente ya alineado; M8≈M5, sin margen de rescate."
    if crisis_mean > 0:
        return ("Leverage INVERTIDO (crisis_mean>0): el régimen no informa el signo → la regla M8 es ruido; "
                "el aprendiz (M10/AutoML) explota la condición (sesgo del agente × régimen) y rescata.")
    # crisis_mean<0 con leverage fuerte (régimen direccional): canal RÉGIMEN
    return ("Canal RÉGIMEN: leverage estándar (crisis_mean<0, régimen direccional) → la regla M8 tiene un signo "
            f"que explotar; acierta {m8_hit:.0%} al intervenir y rescata el RIESGO del agente. El ML puede afinar "
            "la accuracy, pero la regla ya disciplina el riesgo (decisión #18).")


def diagnostico(tk: str, pan: dict) -> dict:
    A.wf.TICKER = tk
    A.wf.reset_thresholds_cache()
    g, sig, oos = A.build_states_onthefly(tk)
    m = A.wf.run_master(g, sig, oos, A.wf.load_agent(tk))
    mv = m.loc[m["r_next"].notna() & (np.sign(m["r_next"]) != 0)].copy()
    y = np.sign(mv["r_next"].to_numpy())
    cM5 = (np.sign(mv["agent_size"].to_numpy()) == y)
    cM8 = (np.sign(mv["final_size"].to_numpy()) == y)
    iv = mv["intervenido"].astype(bool).to_numpy()
    t = pan[tk]["table"]
    acc = {s: t[PK[s]]["accuracy"] for s in PK}
    lc = LEV[tk]["leverage_corr"]; cm = LEV[tk]["crisis_mean"]
    interv = float(iv.mean()); m8_hit = float(cM8[iv].mean()) if iv.any() else float("nan")
    agent_short = float((np.sign(mv["agent_size"].to_numpy()) < 0).mean())
    reg_dir = bool(lc < -0.03 and cm < 0)
    m8_ayuda = acc["M8"] > acc["M5"]
    # canal_ganador POR MECANISMO (un solo discriminante coherente §5/§6), NO por accuracy bruta:
    #   canal RÉGIMEN  ⟺ el régimen es direccional fuera de muestra (leverage estándar: lc<-0.03 Y
    #                     crisis_mean<0) → la regla M8 tiene un signo que explotar.
    #   canal ML       ⟺ resto (leverage débil o invertido, crisis_mean>0) → la regla mete ruido y
    #                     el aprendiz M10/AutoML rescata aprendiendo la condición.
    # No se usa argmax de accuracy: M8 casi nunca gana en accuracy bruta aunque rescate el RIESGO
    # (decisión #18: el valor de M8 está en el riesgo, no en la accuracy). El discriminante
    # crisis_mean<0→régimen se cumple para todos los activos. La tasa de acierto de M8 al intervenir
    # (m8_hit) y M8_ayuda se reportan como EVIDENCIA de por qué la regla rescata (no como gate).
    canal_regimen = reg_dir
    if canal_regimen:
        canal = "régimen (M8)"
    else:
        best_ml = max(("M10", "AutoML"), key=lambda s: acc[s])
        canal = f"ML ({best_ml})"
    return {"leverage_corr": round(lc, 4), "crisis_mean": round(cm, 5),
            "regimen_direccional": reg_dir,
            "intervencion_M8": round(interv, 4), "M8_acierto_en_intervencion": round(m8_hit, 4),
            "agente_frac_corto": round(agent_short, 3),
            "acc": {s: round(acc[s], 4) for s in acc}, "acc_M5_intervenido": round(float(cM5[iv].mean()), 4) if iv.any() else None,
            "canal_ganador": canal, "M8_ayuda": m8_ayuda, "canal_regimen": canal_regimen,
            "mecanismo": _etiqueta(cm, interv, m8_hit, m8_ayuda, canal, acc)}


def main() -> None:
    pan = json.load(open(PANEL_FILE))["por_activo"]
    res = {"meta": {"fuente_accuracy": PANEL_FILE,
                    "tesis": "dos supervisores (regla M8 = canal régimen; aprendiz M10/AutoML = canal ML); "
                             "el canal que rescata depende de la fiabilidad direccional del régimen OOS, "
                             "medida por crisis_mean (signo) y la tasa de acierto de M8 en intervención.",
                    "variable_clave": "crisis_mean<0 → régimen direccional → regla; crisis_mean>0 (leverage "
                                      "invertido) → régimen no direccional → aprendiz."},
           "por_activo": {}}
    for tk in PANEL:
        try:
            res["por_activo"][tk] = diagnostico(tk, pan)
            r = res["por_activo"][tk]
            print(f"{tk:5s} lev={r['leverage_corr']:+.3f} crisisμ={r['crisis_mean']:+.4f} "
                  f"interv={r['intervencion_M8']:.0%} M8hit={r['M8_acierto_en_intervencion']} "
                  f"canal={r['canal_ganador']}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nOK · {OUT}")


if __name__ == "__main__":
    main()
