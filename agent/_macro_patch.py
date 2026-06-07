"""Inyección de contexto macro/sentimiento en los prompts de las personalidades.

SPY es un ETF agregado: las funciones ``get_financial_metrics``,
``search_line_items``, ``get_insider_trades`` y ``get_company_news`` del
submódulo devuelven listas vacías cuando se interrogan con ticker ``SPY``,
y las cinco personalidades acaban respondiendo *"insufficient data on
fundamentals"* (BITACORA 2026-05-16). Como AI Hedge Fund no se modifica
directamente (regla CLAUDE.md §12), aquí monkey-patcheamos
``src.utils.llm.call_llm`` para anteponer al prompt una ``SystemMessage`` con
el snapshot macro construido por ``core/macro_features.py``.

El patch es idempotente y se aplica una sola vez por proceso. El contexto
activo se gestiona con ``set_macro_context``/``clear_macro_context``: el
wrapper lo fija antes de cada llamada al agente y lo limpia al terminar.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

# El submódulo debe estar accesible en sys.path para importar ``src.*``.
_AGENT_ROOT = Path(__file__).resolve().parent / "ai_hedge_fund"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

_PATCH_APPLIED = False
_current_context: dict | None = None
_original_call_llm = None


def set_macro_context(snapshot: dict | None) -> None:
    """Fija el snapshot macro activo para las próximas llamadas al LLM."""
    global _current_context
    _current_context = snapshot


def clear_macro_context() -> None:
    """Limpia el snapshot macro activo (no se inyecta hasta el próximo set)."""
    global _current_context
    _current_context = None


def get_macro_context() -> dict | None:
    """Inspector útil para tests."""
    return _current_context


def _build_macro_messages():
    """Construye la lista de mensajes a anteponer al prompt original."""
    from core.macro_features import format_macro_text  # import diferido
    from langchain_core.messages import SystemMessage

    text = format_macro_text(_current_context)
    header = (
        "El subyacente es un ETF agregado (SPY o equivalente), no una empresa individual. "
        "Las fuentes de fundamentales empresariales (FCF, márgenes, deuda, insider trading) "
        "no aplican y devolverán listas vacías. Razona sobre el mercado agregado usando el "
        "siguiente contexto macro y de sentimiento como sustituto de los fundamentales:"
    )
    return [SystemMessage(content=f"{header}\n\n{text}")]


def _patched_call_llm(prompt: Any, *args, **kwargs):
    """Reemplazo de ``src.utils.llm.call_llm`` que añade contexto macro.

    Si no hay snapshot activo, la llamada pasa al original sin tocar nada.
    """
    if _current_context is None or _original_call_llm is None:
        return _original_call_llm(prompt, *args, **kwargs)

    # ``prompt`` puede ser un ChatPromptValue (resultado de .invoke) o una
    # lista de BaseMessage. Cualquiera de los dos casos lo aceptamos.
    try:
        if hasattr(prompt, "to_messages"):
            messages = list(prompt.to_messages())
        elif isinstance(prompt, list):
            messages = list(prompt)
        else:
            # Tipos inesperados: cae al original sin inyectar para no romper.
            return _original_call_llm(prompt, *args, **kwargs)
    except Exception:
        return _original_call_llm(prompt, *args, **kwargs)

    new_prompt = _build_macro_messages() + messages
    return _original_call_llm(new_prompt, *args, **kwargs)


def apply_macro_patch() -> None:
    """Reemplaza ``call_llm`` en el módulo origen y en sus consumidores.

    Idempotente: aplicada más de una vez no tiene efecto adicional.
    """
    global _PATCH_APPLIED, _original_call_llm
    if _PATCH_APPLIED:
        return

    llm_mod = importlib.import_module("src.utils.llm")
    _original_call_llm = llm_mod.call_llm  # type: ignore[attr-defined]
    llm_mod.call_llm = _patched_call_llm  # type: ignore[attr-defined]

    # Las personalidades importan ``call_llm`` por ``from src.utils.llm import
    # call_llm``, así que mantienen una referencia local que también debemos
    # reemplazar. Mismo patrón que ``_price_patch._apply``.
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
