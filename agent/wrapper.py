"""Adaptador entre AI Hedge Fund y el formato canónico de STRATA.

Hace de única superficie de contacto con el submódulo ``agent/ai_hedge_fund``.
Resto del código nunca importa nada de ``src.*`` directamente; usa
``run_agent`` y recibe un ``AgentOutput`` listo para los detectores.

Reglas de configuración:

- LLM por defecto: **DeepSeek V3 vía OpenRouter** (decisión pivot BITACORA
  2026-05-19; el ID concreto se lee de ``config.LLM_MODEL``).
- Personalidades habilitadas (5): Buffett, Wood, Druckenmiller, Burry, Ackman.
  Bill Ackman sustituye a Howard Marks porque éste no existe en el submódulo;
  Ackman aporta dimensión activista/posiciones concentradas, ortogonal a los
  otros cuatro perfiles.
- **Inyección de contexto macro/sentimiento.** Como SPY es un ETF agregado y
  Financial Datasets API devuelve listas vacías para sus fundamentales, antes
  de cada llamada al LLM se inyecta un snapshot macro construido por
  ``core/macro_features.py``. La inyección se hace vía ``agent/_macro_patch.py``,
  que parchea ``src.utils.llm.call_llm`` para anteponer una ``SystemMessage``
  con el snapshot al prompt original de cada personalidad.
"""

from __future__ import annotations

import sys
from pathlib import Path

# El submódulo importa como ``from src.* import ...``; añadimos su raíz al path.
_AGENT_ROOT = Path(__file__).resolve().parent / "ai_hedge_fund"
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from config import LLM_MODEL  # noqa: E402
from strata.types import AgentOutput, PersonalityOutput  # noqa: E402

# Mapeo personalidades STRATA → keys de AI Hedge Fund.
PERSONALITIES_STRATA: tuple[str, ...] = (
    "warren_buffett",
    "cathie_wood",
    "stanley_druckenmiller",
    "michael_burry",
    "bill_ackman",
)

# Provider literal exigido por src.llm.models.ModelProvider.
LLM_PROVIDER = "OpenRouter"


def _ensure_cache_enabled() -> None:
    """Activa la caché JSON global de LangChain si aún no lo está."""
    from langchain_core.globals import get_llm_cache

    if get_llm_cache() is None:
        from agent.llm_client import enable_global_cache

        enable_global_cache()


def _ensure_price_patch() -> None:
    """Reemplaza el proveedor de precios del submódulo por yfinance."""
    from agent._price_patch import apply_price_patch

    apply_price_patch()


def _ensure_macro_patch() -> None:
    """Activa la inyección de contexto macro en los prompts de las personalidades."""
    from agent._macro_patch import apply_macro_patch

    apply_macro_patch()


def _portfolio_for_single_ticker(ticker: str, cash: float = 100_000.0) -> dict:
    """Construye el dict de portfolio mínimo que ``run_hedge_fund`` espera."""
    return {
        "cash": cash,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0},
        },
    }


def _quantity_to_size(quantity: int, action: str, cash: float, price: float) -> float:
    """Normaliza la decisión a un tamaño en ``[-1, 1]``.

    Si la acción es *hold*, el tamaño es 0. Para *buy/short/cover/sell*,
    se calcula el peso sobre el cash disponible como
    ``(quantity * price) / cash``, con signo positivo si es long y negativo
    si es short.
    """
    if action in ("hold",) or quantity <= 0 or cash <= 0 or price <= 0:
        return 0.0
    raw = (quantity * price) / cash
    raw = float(max(-1.0, min(1.0, raw)))
    if action in ("short", "sell"):
        return -raw
    return raw


def _action_to_strata(action: str) -> str:
    """Traduce el verbo del Portfolio Manager al espacio de acciones de STRATA.

    ``buy``/``cover`` → ``long``; ``short``/``sell`` → ``short``; ``hold`` → ``hold``.
    """
    if action in ("buy", "cover"):
        return "long"
    if action in ("short", "sell"):
        return "short"
    return "hold"


def run_agent(
    ticker: str,
    date: str,
    portfolio_cash: float = 100_000.0,
    *,
    model_name: str = LLM_MODEL,
    show_reasoning: bool = False,
) -> AgentOutput:
    """Ejecuta AI Hedge Fund para ``(ticker, date)`` y devuelve un ``AgentOutput``.

    Args:
        ticker: símbolo (por ejemplo ``"SPY"``).
        date: fecha de la decisión en formato ``YYYY-MM-DD``.
        portfolio_cash: cash disponible, define la escala del sizing normalizado.
        model_name: identificador OpenRouter (default DeepSeek V3 vía ``config.LLM_MODEL``).
        show_reasoning: si ``True``, deja que el agente imprima su razonamiento.
    """
    _ensure_cache_enabled()
    from src.main import run_hedge_fund  # type: ignore

    _ensure_price_patch()
    _ensure_macro_patch()

    # Construye e inyecta el snapshot macro para esta (ticker, date).
    from agent._macro_patch import clear_macro_context, set_macro_context
    from core.macro_features import build_macro_snapshot

    snapshot = build_macro_snapshot(date, ticker=ticker)
    set_macro_context(snapshot)

    portfolio = _portfolio_for_single_ticker(ticker, cash=portfolio_cash)
    # AI Hedge Fund usa ``start_date`` para datos históricos y ``end_date`` como
    # fecha de decisión. Le damos una ventana de 1 año hacia atrás.
    end_date = date
    start_date = (
        __import__("datetime")
        .datetime.fromisoformat(date)
        .replace(year=int(date[:4]) - 1)
        .strftime("%Y-%m-%d")
    )

    try:
        result = run_hedge_fund(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=show_reasoning,
            selected_analysts=list(PERSONALITIES_STRATA),
            model_name=model_name,
            model_provider=LLM_PROVIDER,
        )
    finally:
        # El contexto macro no debe leakear entre llamadas a fechas distintas.
        clear_macro_context()

    decisions = (result or {}).get("decisions") or {}
    decision = decisions.get(ticker, {})
    raw_action = str(decision.get("action", "hold"))
    quantity = int(decision.get("quantity", 0))
    confidence = float(decision.get("confidence", 0.0)) / 100.0
    reasoning = str(decision.get("reasoning", ""))

    # Precio de cierre del día de decisión, requerido para normalizar quantity.
    signals = (result or {}).get("analyst_signals") or {}
    price = float(_extract_close_price(signals, ticker)) or 0.0
    size = _quantity_to_size(quantity, raw_action, portfolio_cash, price)
    action = _action_to_strata(raw_action)

    # Detalle por personalidad para el detector RAM y la trazabilidad.
    personalities: dict[str, PersonalityOutput] = {}
    for key in PERSONALITIES_STRATA:
        sig = signals.get(f"{key}_agent", {}).get(ticker, {})
        if not sig:
            continue
        per_action = str(sig.get("signal", "hold"))
        per_action_strata = _action_to_strata(
            "buy" if per_action == "bullish" else ("short" if per_action == "bearish" else "hold")
        )
        personalities[key] = PersonalityOutput(
            name=key,
            action=per_action_strata,
            size=0.0,  # las personalidades emiten signal/confidence, no quantity
            confidence=float(sig.get("confidence", 0.0)) / 100.0,
            reasoning=str(sig.get("reasoning", "")),
        )

    return AgentOutput(
        date=date,
        ticker=ticker,
        action=action,
        size=size,
        confidence=confidence,
        reasoning=reasoning,
        personalities=personalities,
    )


def _extract_close_price(signals: dict, ticker: str) -> float:
    """Recupera el último precio de cierre que cualquier nodo haya cacheado.

    El ``risk_management_agent`` de AI Hedge Fund publica
    ``signals[risk_management_agent][ticker]['current_price']``. Se intenta esa
    ruta primero; con fallback a una llamada directa a yfinance si no aparece
    (caso defensivo).
    """
    risk = signals.get("risk_management_agent", {}).get(ticker, {})
    if isinstance(risk, dict) and risk.get("current_price"):
        return float(risk["current_price"])
    # Fallback: leer del último cierre en yfinance.
    try:
        import yfinance as yf

        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0
