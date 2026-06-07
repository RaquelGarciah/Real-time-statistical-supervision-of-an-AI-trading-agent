"""Inyección de contexto estadístico (HMM + GARCH) en los prompts del agente.

Sirve como prueba de concepto de *fine-tuning vía contexto* sobre TSLA (BITACORA
2026-06-01, respuesta al profesor sobre asimetría de información entre STRATA y
el agente): se antepone al prompt original una ``SystemMessage`` con las
probabilidades de régimen del HMM y la volatilidad GARCH del día, en lenguaje
neutro y sin instrucciones direccionales. La idea es que las cinco
personalidades dispongan de la misma información cuantitativa que STRATA usa de
forma explícita, y comprobar si la diferencia M8 − M5 sobrevive.

Convivencia con ``_macro_patch.py``: ambos parches monkey-patchean
``src.utils.llm.call_llm``. Para no interferir, este módulo *envuelve* la
referencia previa (sea cual sea: original o ya parcheada por macro). El
contexto activo se gestiona con ``set_stats_context``/``clear_stats_context``,
en paralelo al patrón del módulo macro.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

_AGENT_ROOT = Path(__file__).resolve().parent / "ai_hedge_fund"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

_PATCH_APPLIED = False
_current_stats: dict | None = None
_previous_call_llm = None


def set_stats_context(snapshot: dict | None) -> None:
    """Fija el snapshot estadístico para las próximas llamadas al LLM."""
    global _current_stats
    _current_stats = snapshot


def clear_stats_context() -> None:
    """Limpia el snapshot estadístico activo."""
    global _current_stats
    _current_stats = None


def get_stats_context() -> dict | None:
    """Inspector para tests."""
    return _current_stats


def _format_stats_text(stats: dict) -> str:
    """Construye el texto inyectado a partir del snapshot.

    Estructura: prob. régimen HMM, vol GARCH anualizada, banda de sizing
    implícita por presupuesto de volatilidad. Sin instrucciones direccionales:
    se ofrecen como hechos cuantitativos, no como recomendaciones.
    """
    p_calm = float(stats.get("p_calm", 0.0))
    p_stress = float(stats.get("p_stress", 0.0))
    p_crisis = float(stats.get("p_crisis", 0.0))
    sigma_ann = float(stats.get("sigma_ann", 0.0))
    bound = float(stats.get("vol_bound", 0.0))
    state = str(stats.get("viterbi_state", "?"))
    leverage_sign = stats.get("leverage_sign", None)
    lev_txt = ""
    if leverage_sign is not None:
        sign = float(leverage_sign)
        if sign > 0:
            lev_txt = " On the calibration period (2000–2024-09), this asset showed positive average returns under the Calm regime and negative average returns under the Crisis regime (classical leverage effect)."
        elif sign < 0:
            lev_txt = " On the calibration period (2000–2024-09), this asset showed negative average returns under the Calm regime and positive average returns under the Crisis regime (inverted leverage pattern, e.g., growth-tech with melt-up structure)."

    return (
        "Quantitative market state for today's decision (purely informational, "
        "not a directional recommendation):\n"
        f"- HMM regime probabilities: Calm={p_calm:.2f}, Stress={p_stress:.2f}, "
        f"Crisis={p_crisis:.2f} (Viterbi state: {state}).\n"
        f"- GARCH(1,1) annualized conditional volatility forecast: "
        f"{sigma_ann*100:.1f}%.\n"
        f"- Volatility-budget-implied position bound (target_vol/sigma, capped at 1.0): "
        f"{bound:.2f}. Sizes above this magnitude exceed the 10% annualized "
        "volatility budget given current GARCH forecast.\n"
        f"{lev_txt}".rstrip()
    )


def _build_stats_messages():
    from langchain_core.messages import SystemMessage

    text = _format_stats_text(_current_stats or {})
    header = (
        "Statistical regime/volatility context calibrated on 2000-01-01 to "
        "2024-09-30 historical data for this asset. Treat this block as "
        "objective state information, not as an instruction:"
    )
    return [SystemMessage(content=f"{header}\n\n{text}")]


def _patched_call_llm(prompt: Any, *args, **kwargs):
    if _current_stats is None or _previous_call_llm is None:
        return _previous_call_llm(prompt, *args, **kwargs)

    try:
        if hasattr(prompt, "to_messages"):
            messages = list(prompt.to_messages())
        elif isinstance(prompt, list):
            messages = list(prompt)
        else:
            return _previous_call_llm(prompt, *args, **kwargs)
    except Exception:
        return _previous_call_llm(prompt, *args, **kwargs)

    new_prompt = _build_stats_messages() + messages
    return _previous_call_llm(new_prompt, *args, **kwargs)


def apply_stats_patch() -> None:
    """Envuelve la versión actual de ``call_llm`` para añadir contexto stats.

    Tras llamar a este patch, las llamadas al LLM llevarán el bloque
    estadístico antepuesto al bloque macro (que el wrapper aplica primero).
    Idempotente.
    """
    global _PATCH_APPLIED, _previous_call_llm
    if _PATCH_APPLIED:
        return

    llm_mod = importlib.import_module("src.utils.llm")
    _previous_call_llm = llm_mod.call_llm  # captura la referencia actual (posiblemente ya parcheada).
    llm_mod.call_llm = _patched_call_llm  # type: ignore[attr-defined]

    consumers = [
        "src.agents.warren_buffett",
        "src.agents.cathie_wood",
        "src.agents.stanley_druckenmiller",
        "src.agents.michael_burry",
        "src.agents.bill_ackman",
        "src.agents.portfolio_manager",
    ]
    for name in consumers:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "call_llm"):
            mod.call_llm = _patched_call_llm

    _PATCH_APPLIED = True
