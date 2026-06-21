"""Genera decisiones del agente (AI Hedge Fund) para tickers nuevos y las cachea.

Resumible: para cada (ticker, fecha) del calendario OOS de referencia (las fechas de
cache/agent/SPY), si el JSON ya existe lo salta; si no, llama a run_agent y lo escribe en
cache/agent/<TICKER>/<TICKER>_<fecha>.json en el formato que espera wf.load_agent.

Requiere OPENROUTER_API_KEY válida (el agente consulta el LLM). ~19 s por día.

Uso: python experiments/gen_agent_decisions.py --tickers QQQ DIA IWM XLF XLK
"""
from __future__ import annotations

import argparse
import glob
import json
import signal
import sys
import time
from pathlib import Path

DECISION_TIMEOUT = 240  # s; una decisión normal tarda ~10-80s. Más allá = llamada de red colgada.


class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from config import CACHE_AGENT_DIR
from agent.wrapper import run_agent

REF_TICKER = "SPY"  # calendario OOS de referencia (mismas fechas que el panel)


def _ref_dates() -> list[str]:
    fps = sorted(glob.glob(str(CACHE_AGENT_DIR / REF_TICKER / f"{REF_TICKER}_*.json")))
    return [Path(f).stem.split("_", 1)[1] for f in fps]


def _looks_like_failure(out) -> bool:
    """Detecta el patrón de fallo del LLM (todo 'hold' a confianza 0): NO se cachea.

    Cuando una llamada al LLM agota reintentos (401/429), el agente devuelve un default
    'hold' con size y confianza 0 en TODAS las personalidades y en el PM. Una decisión
    legítima rara vez tiene todo a cero exacto; rechazarla evita meter basura al caché y
    fuerza el reintento en la siguiente pasada (resumible).
    """
    pm_default = out.action == "hold" and float(out.size) == 0.0 and float(out.confidence) == 0.0
    pers_default = all(po.action == "hold" and float(po.confidence) == 0.0
                       for po in out.personalities.values()) and len(out.personalities) > 0
    return pm_default and pers_default


def _serialize(out) -> dict:
    return {
        "date": out.date, "ticker": out.ticker, "action": out.action,
        "size": float(out.size), "confidence": float(out.confidence),
        "reasoning": getattr(out, "reasoning", ""),
        "personalities": {
            nm: {"action": po.action, "size": float(po.size),
                 "confidence": float(po.confidence), "reasoning": getattr(po, "reasoning", "")}
            for nm, po in out.personalities.items()},
    }


def generate(ticker: str, dates: list[str]) -> tuple[int, int, int]:
    outdir = CACHE_AGENT_DIR / ticker
    outdir.mkdir(parents=True, exist_ok=True)
    hechas = nuevas = errores = 0
    for i, d in enumerate(dates):
        fp = outdir / f"{ticker}_{d}.json"
        if fp.exists():
            hechas += 1
            continue
        try:
            t0 = time.time()
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(DECISION_TIMEOUT)  # corta llamadas de red colgadas
            try:
                out = run_agent(ticker, d)
            finally:
                signal.alarm(0)
            if _looks_like_failure(out):
                errores += 1
                print(f"  {ticker} {d} FALLO-LLM (todo hold/conf0) → no se cachea, se reintenta", flush=True)
                continue
            fp.write_text(json.dumps(_serialize(out), ensure_ascii=False, indent=1))
            nuevas += 1
            print(f"  {ticker} {d} [{i+1}/{len(dates)}] {out.action} size={out.size:+.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"  {ticker} {d} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
    return hechas, nuevas, errores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    args = ap.parse_args()
    dates = _ref_dates()
    print(f"Calendario OOS de referencia ({REF_TICKER}): {len(dates)} días "
          f"[{dates[0]} → {dates[-1]}]", flush=True)
    for tk in args.tickers:
        print(f"\n=== {tk} ===", flush=True)
        h, n, e = generate(tk, dates)
        print(f"{tk}: {h} ya estaban · {n} nuevas · {e} errores · total caché "
              f"{len(list((CACHE_AGENT_DIR / tk).glob(f'{tk}_*.json')))}", flush=True)
    print("\nOK · decisiones generadas", flush=True)


if __name__ == "__main__":
    config.set_seeds(config.SEED)
    main()
