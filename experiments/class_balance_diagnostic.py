"""Diagnóstico de balanceo de clases sube/baja en el OOS, por activo.

Motivación (reunión tutor 2026-06-16): el tribunal compara cualquier predicción
binaria contra el baseline trivial "clase mayoritaria". Si el OOS está muy
desbalanceado (SPY alcista), ese baseline acierta mucho sin modelo. Interesa
localizar el activo (o periodo) con proporción sube/baja más cercana a 50/50
para que batir al baseline signifique algo.

Métrica reportada por activo: fracción de días con r_{t+1} > 0 sobre las fechas
en que el agente decidió (clase = signo del retorno causal, signal_lag=1), y la
accuracy del baseline mayoritario = max(p_up, 1 - p_up).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CACHE_AGENT_DIR, CALIBRATION_START, STRATA_OOS_START
from core import data, features


def agent_dates(ticker: str) -> list[pd.Timestamp]:
    fechas = []
    for fp in sorted(glob.glob(str(CACHE_AGENT_DIR / ticker / f"{ticker}_*.json"))):
        fechas.append(pd.Timestamp(json.load(open(fp))["date"]))
    return sorted(fechas)


def balance(ticker: str) -> dict | None:
    fechas = agent_dates(ticker)
    if not fechas:
        return None
    data_end = (fechas[-1] + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    prices = data.load_market_data(ticker, CALIBRATION_START, data_end)
    ret = features.log_returns(prices["Close"])
    # retorno causal: la decisión de t se evalúa contra r_{t+1}
    r_next = ret.shift(-1)
    idx = pd.DatetimeIndex(fechas)
    oos = idx[idx >= pd.Timestamp(STRATA_OOS_START)]
    r = r_next.reindex(oos).dropna()
    if len(r) < 20:
        return None
    p_up = float((r > 0).mean())
    return {
        "ticker": ticker,
        "n_dias_oos": int(len(r)),
        "frac_sube": round(p_up, 4),
        "frac_baja": round(1 - p_up, 4),
        "baseline_mayoritario_acc": round(max(p_up, 1 - p_up), 4),
        "desbalanceo_abs": round(abs(p_up - 0.5), 4),
        "oos_ini": str(r.index[0].date()),
        "oos_fin": str(r.index[-1].date()),
    }


def main() -> None:
    tickers = sorted(p.name for p in CACHE_AGENT_DIR.iterdir() if p.is_dir())
    filas = [b for t in tickers if (b := balance(t)) is not None]
    filas.sort(key=lambda d: d["desbalanceo_abs"])

    print(f"{'ticker':<10}{'n_oos':>7}{'%sube':>9}{'%baja':>9}{'base_acc':>10}{'|desbal|':>10}")
    for f in filas:
        print(f"{f['ticker']:<10}{f['n_dias_oos']:>7}{f['frac_sube']:>9.3f}"
              f"{f['frac_baja']:>9.3f}{f['baseline_mayoritario_acc']:>10.3f}{f['desbalanceo_abs']:>10.3f}")

    out = Path("outputs/experiments/class_balance_diagnostic.json")
    out.write_text(json.dumps({
        "descripcion": "Balanceo sube/baja en OOS por activo; clase = signo de r_{t+1} (causal) "
                       "sobre fechas con decisión del agente. Ordenado por cercanía a 50/50.",
        "oos_start": STRATA_OOS_START,
        "activos": filas,
    }, indent=2, ensure_ascii=False))
    print(f"\n>>> {out}")
    mas = filas[0]
    print(f"Más balanceado: {mas['ticker']} ({mas['frac_sube']*100:.1f}% sube, "
          f"baseline mayoritario {mas['baseline_mayoritario_acc']*100:.1f}%)")


if __name__ == "__main__":
    main()
