"""Capa de intervención de STRATA: decide qué hacer con la decisión del agente
una vez los detectores han hablado.

Modos:

- ``warn``: pasa la decisión sin modificarla; sólo registra los detectores.
- ``reduce``: atenúa el tamaño multiplicativamente. Según ``reduce_mode``:
  ``"bucket"`` por la severidad discreta del peor detector, ``"continuous"`` por
  ``(1 - max_score)`` de los tres, o ``"ram_continuous"`` (control M7) por
  ``(1 - RAM)`` pero solo cuando RAM dispararía el override (severidad
  medium/high, i.e. score ≥ τ): es el análogo "encoger" de override-C, que
  "voltea". Aísla cuánto del rescate de M8 viene de reducir exposición frente a
  cuánto de corregir la dirección.
- ``override``: si algún detector marca *medium* o *high*, el tamaño se
  sustituye. La respuesta a la desalineación direccional de RAM tiene tres
  variantes (``override_variant``):

  - ``"A"`` (neutralizar): el sizing va a cero (comportamiento histórico).
  - ``"B"`` (inversión parcial): se reorienta hacia el régimen, proporcional a
    su probabilidad: ``0.5 * regime_sign * bound * p_dominant``.
  - ``"C"`` (inversión total): se sustituye por el sizing GARCH con el signo del
    régimen: ``regime_sign * bound``.
  - ``"D"`` (corrección de signo a escala del agente): conserva la magnitud de
    convicción del agente y solo corrige la dirección: ``regime_sign * |size|``.
    Aísla el efecto dirección del efecto escala que confunden B y C.

  Es el único modo que actúa como interventor de verdad.

La acción cualitativa (long/short/hold) se ajusta cuando el tamaño final cae a
cero (queda como ``hold``).
"""

from __future__ import annotations

from strata.types import (
    AgentOutput,
    DetectorResult,
    InterventionMode,
    Severity,
    SupervisedDecision,
)

_SEVERITY_MULTIPLIER: dict[Severity, float] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.6,
    "high": 1.0,
}


def _max_severity(detectors: dict[str, DetectorResult]) -> Severity:
    """Devuelve la severidad más alta entre los detectores."""
    levels: list[Severity] = ["none", "low", "medium", "high"]
    order = {lvl: i for i, lvl in enumerate(levels)}
    worst: Severity = "none"
    for det in detectors.values():
        if order[det.severity] > order[worst]:
            worst = det.severity
    return worst


def _final_action(size: float) -> str:
    if abs(size) < 1e-9:
        return "hold"
    return "long" if size > 0 else "short"


def supervise(
    agent: AgentOutput,
    detectors: dict[str, DetectorResult],
    mode: InterventionMode,
    override_variant: str = "A",
    reduce_mode: str = "bucket",
) -> SupervisedDecision:
    """Aplica el modo de intervención y devuelve una decisión supervisada.

    Args:
        agent: salida cruda del Portfolio Manager del agente.
        detectors: resultados de RAM, PSA y GSO.
        mode: ``"warn"``, ``"reduce"`` o ``"override"``.
        override_variant: respuesta de RAM en modo override (``"A"``/``"B"``/``"C"``).
        reduce_mode: en modo reduce, ``"bucket"`` (atenúa por severidad discreta) o
            ``"continuous"`` (atenúa ∝ score máximo de los detectores, ``1 - clip(score)``).
    """
    if mode == "warn":
        return SupervisedDecision(
            date=agent.date,
            ticker=agent.ticker,
            mode=mode,
            final_action=agent.action,
            final_size=agent.size,
            was_intervened=False,
            detectors=detectors,
            raw_agent=agent,
        )

    if mode == "reduce":
        if reduce_mode == "ram_continuous":
            # Reduce *gated* en RAM (control M7 de la escalera de intervención).
            # Solo atenúa cuando el régimen contradice la acción con confianza
            # suficiente, es decir cuando RAM ya dispararía el override
            # (severidad medium/high; el corte 'medium' es el umbral τ calibrado
            # vía ram_thresholds). El factor es 1 - RAM_t: encoge la posición del
            # agente hacia cash en proporción a la confianza de incoherencia. Es
            # el análogo "reduce" de override-C: en lugar de voltear al régimen,
            # solo reduce la exposición del agente sin imponer dirección.
            ram = detectors.get("ram")
            if ram is not None and ram.severity in {"medium", "high"}:
                factor = 1.0 - min(1.0, max(0.0, ram.score))
                was_intervened = True
            else:
                factor = 1.0
                was_intervened = False
        elif reduce_mode == "continuous":
            # Atenuación continua: factor 1 - max(score) en [0,1]. Más fino que
            # los buckets de severidad; el score de RAM ya es una probabilidad.
            max_score = max((d.score for d in detectors.values()), default=0.0)
            factor = 1.0 - min(1.0, max(0.0, max_score))
            was_intervened = max_score > 0.0
        else:
            worst = _max_severity(detectors)
            factor = 1.0 - _SEVERITY_MULTIPLIER[worst]
            was_intervened = worst != "none"
        final_size = float(agent.size * factor)
        return SupervisedDecision(
            date=agent.date,
            ticker=agent.ticker,
            mode=mode,
            final_action=_final_action(final_size),
            final_size=final_size,
            was_intervened=was_intervened,
            detectors=detectors,
            raw_agent=agent,
        )

    if mode == "override":
        gso = detectors.get("gso")
        ram = detectors.get("ram")
        psa = detectors.get("psa")
        final_size = agent.size
        was_intervened = False

        # Paso 1: GSO restringe magnitud al límite permitido por volatilidad.
        if gso is not None and gso.severity in {"medium", "high"}:
            final_size = float(gso.extra.get("bounded_size", final_size))
            was_intervened = True

        # Paso 2: RAM impone alineamiento de régimen. Según la variante, el
        # sizing se neutraliza (A) o se reorienta hacia la dirección implícita
        # del régimen con la banda GARCH (B parcial, C total).
        if ram is not None and ram.severity in {"medium", "high"}:
            regime_sign = float(ram.extra.get("regime_sign", 0.0))
            p_dom = float(ram.extra.get("p_dominant", 0.0))
            bound = float(gso.extra.get("bound", 1.0)) if gso is not None else 1.0
            if override_variant == "B":
                final_size = 0.5 * regime_sign * bound * p_dom
            elif override_variant == "C":
                final_size = regime_sign * bound
            elif override_variant == "D":
                final_size = regime_sign * abs(agent.size)
            else:  # "A" (default): neutralizar a cash.
                final_size = 0.0
            was_intervened = True

        # Paso 3: PSA con severidad alta indica que el sizing del agente acaba
        # de cambiar bruscamente; aplicamos un freno temporal (a la mitad)
        # para reducir el ruido de transición.
        if psa is not None and psa.severity == "high":
            final_size = float(final_size * 0.5)
            was_intervened = True

        return SupervisedDecision(
            date=agent.date,
            ticker=agent.ticker,
            mode=mode,
            final_action=_final_action(final_size),
            final_size=final_size,
            was_intervened=was_intervened,
            detectors=detectors,
            raw_agent=agent,
        )

    raise ValueError(f"Modo de intervención desconocido: {mode!r}")
