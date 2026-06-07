"""Tipos de datos canónicos de STRATA.

Estos dataclasses fijan el contrato entre la capa del agente (que produce las
decisiones), los detectores estadísticos (RAM/PSA/GSO) y la capa de
intervención (warn/reduce/override). Aislan el resto del código de los detalles
internos de AI Hedge Fund.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Action = Literal["long", "short", "hold"]
Severity = Literal["none", "low", "medium", "high"]
InterventionMode = Literal["warn", "reduce", "override"]


@dataclass
class PersonalityOutput:
    """Salida individual de una de las cinco personalidades de AI Hedge Fund."""

    name: str
    action: Action
    size: float  # tamaño normalizado en [-1, 1] (negativo = short)
    confidence: float  # confianza en [0, 1]
    reasoning: str = ""


@dataclass
class AgentOutput:
    """Decisión agregada del Portfolio Manager, lista para supervisión.

    ``size`` está en escala normalizada por nominal (no número de acciones):
    ``size = +1.0`` significa toda la exposición permitida al alza; ``-1.0`` al
    bajo; ``0.0`` neutral.
    """

    date: str  # YYYY-MM-DD
    ticker: str
    action: Action
    size: float
    confidence: float
    reasoning: str = ""
    personalities: dict[str, PersonalityOutput] = field(default_factory=dict)


@dataclass
class DetectorResult:
    """Resultado de uno de los tres detectores de STRATA."""

    name: Literal["ram", "psa", "gso"]
    score: float
    flag: bool
    severity: Severity = "none"
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class SupervisedDecision:
    """Decisión final tras la capa de intervención."""

    date: str
    ticker: str
    mode: InterventionMode
    final_action: Action
    final_size: float
    was_intervened: bool
    detectors: dict[str, DetectorResult] = field(default_factory=dict)
    raw_agent: AgentOutput | None = None
