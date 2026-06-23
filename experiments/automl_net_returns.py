"""Serie diaria de retornos netos de AutoML por activo, reconstruida desde el panel canónico (sin H2O).

El panel canónico (`panel_mm25_inclGBM-XGB-SE_AUC_emb1_*.json`) guarda el acierto día a día de cada brazo
(`correct_by_arm`) pero no la serie de retornos, así que AutoML no podía entrar en las curvas de equity ni en
el bootstrap de riesgo. Como la posición de AutoML es direccional (±1), se recupera EXACTAMENTE desde el
acierto y la verdad: pos_t = sign(r_{t+1}) · (2·acierto_t − 1). Pasando esa posición por el mismo motor de
backtest (mismos costes de transacción) se obtiene la serie de retornos idéntica a la corrida canónica —se
valida activo a activo contra `table[automl]` (accuracy, Sharpe, equity, maxDD).

Esto evita re-ejecutar H2O (que en macOS no entrena XGBoost y daría un líder y una accuracy distintos del panel
canónico, generado con XGBoost): la serie sale de la propia configuración canónica, no de una nueva búsqueda.

Uso: python experiments/automl_net_returns.py
Salida: outputs/experiments/automl_net_returns.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import experiments.automl_m10 as A
from core import metrics
from core.backtest import run_backtest

PANEL = ("outputs/experiments/automl_runs/"
         "panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json")
OUT = Path("outputs/experiments/automl_net_returns.json")


def reconstruir(tk: str, entry: dict) -> dict:
    """Reconstruye la serie de retornos netos de AutoML para un activo y la valida contra la tabla canónica."""
    A.wf.TICKER = tk
    A.wf.reset_thresholds_cache()
    gamma_df, sigma, oos_ret = A.build_states_onthefly(tk)
    m = A.wf.run_master(gamma_df, sigma, oos_ret, A.wf.load_agent(tk))
    valid = m["r_next"].notna() & (np.sign(m["r_next"]) != 0)
    mv = m.loc[valid]
    td = mv.index[A.N0:]                                   # días con predicción del meta-learner (tras burn-in)
    corr = np.array(entry["correct_by_arm"]["automl"])
    if len(corr) != len(td):
        raise ValueError(f"{tk}: correct_by_arm.automl ({len(corr)}) ≠ td ({len(td)})")
    truth = np.sign(mv.loc[td, "r_next"].to_numpy())
    pos = truth * (2 * corr - 1)                           # recupera la posición ±1 exacta de AutoML
    w = pd.Series(0.0, index=m.index); w.loc[td] = pos
    net = run_backtest(oos_ret, w, signal_lag=1)["net_return"].reindex(td)
    nz = net.dropna(); eq = (1.0 + nz).cumprod()
    rec = {"accuracy": round(float(corr.mean()), 4), "sharpe": round(A._sr(nz.to_numpy()), 3),
           "equity_final": round(float(eq.iloc[-1]), 4), "max_dd": round(metrics.max_drawdown(eq), 4)}
    can = entry["table"]["automl"]
    # Validación: la reconstrucción tiene que coincidir con la cifra canónica (identidad, no aproximación).
    assert abs(rec["accuracy"] - can["accuracy"]) < 1e-3, f"{tk}: acc {rec} vs {can}"
    assert abs(rec["sharpe"] - can["sharpe"]) < 0.02, f"{tk}: sharpe {rec} vs {can}"
    assert abs(rec["equity_final"] - can["equity_final"]) < 0.01, f"{tk}: equity {rec} vs {can}"
    return {"dates": [str(d.date()) for d in td],
            "automl": [None if np.isnan(x) else round(float(x), 6) for x in net.to_numpy()],
            "validacion": {"reconstruido": rec, "canonico": {k: can[k] for k in rec}}}


def main() -> None:
    pan = json.load(open(PANEL))["por_activo"]
    res = {"meta": {"fuente": PANEL, "metodo": "pos_t = sign(r_{t+1})·(2·acierto_t−1) → run_backtest (mismos "
                    "costes); validado contra table[automl]. Sin H2O: serie consistente con la config canónica."},
           "por_activo": {}}
    for tk, entry in pan.items():
        try:
            res["por_activo"][tk] = reconstruir(tk, entry)
            v = res["por_activo"][tk]["validacion"]["reconstruido"]
            print(f"{tk:5s} OK · acc={v['accuracy']} sharpe={v['sharpe']} equity={v['equity_final']} maxDD={v['max_dd']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tk:5s} ERROR {type(e).__name__}: {e}", flush=True)
            res["por_activo"][tk] = {"error": f"{type(e).__name__}: {e}"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    ok = sum("error" not in v for v in res["por_activo"].values())
    print(f"\nOK · {OUT} · {ok}/{len(pan)} activos")


if __name__ == "__main__":
    main()
