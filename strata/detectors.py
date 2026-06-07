"""Los tres detectores estadísticos de STRATA.

Cada detector recibe la decisión del agente y el estado actual del mercado
(régimen HMM, volatilidad GARCH, historia de sizing) y emite un
``DetectorResult`` con un score continuo, un flag binario y una severidad
cualitativa.

- **RAM** (Regime-Action Mismatch): coherencia con el régimen de mercado
  detectado por el HMM de 3 estados. Política simétrica con el leverage
  effect: Calma → short penalizado, Crisis → long penalizado, Estrés sin
  penalización (refinamiento BITACORA 2026-05-19).
- **PSA** (Position Sizing Anomaly): coherencia temporal del sizing del propio
  agente, medida con BOCPD sobre su historial.
- **GSO** (GARCH-bounded Sizing Override): coherencia con la volatilidad
  condicional, expresada como banda permitida ``target_vol/sigma_t``.

Los tres son ortogonales por diseño.

**Umbrales por detector.** Los umbrales de severidad pueden ser:

1. Los defaults del diseño preliminar ``(0.7, 0.4, 0.2)`` (compartidos por los
   tres detectores), o
2. Los recalibrados por percentiles que produce
   ``experiments/recalibrate_strata_thresholds.py`` y guarda en
   ``cache/models/strata_thresholds.json``. Cuando ese fichero existe, los
   detectores PSA y GSO usan ``(max, P99, P95)`` como umbrales de
   ``high/medium/low`` (≈0,1 % / 1 % / 5 % de activación). RAM mantiene
   siempre sus defaults porque su score es una masa de probabilidad sobre
   regímenes (no depende de calibración de datos).

La carga del JSON es perezosa y se cachea en memoria del proceso.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from config import AGENT_MAX_SIZE, CACHE_MODELS_DIR
from core.bocpd import bocpd
from strata.types import DetectorResult, Severity

# Volatilidad objetivo anualizada (mismo valor que OS1).
TARGET_VOL: float = 0.10

# Defaults: severidad uniforme heredada del diseño preliminar.
_DEFAULT_THRESHOLDS: tuple[tuple[float, Severity], ...] = (
    (0.7, "high"),
    (0.4, "medium"),
    (0.2, "low"),
    (0.0, "none"),
)

_THRESHOLDS_FILE = CACHE_MODELS_DIR / "strata_thresholds.json"
_CACHE: dict[str, tuple[tuple[float, Severity], ...]] | None = None


# Percentil de flag por defecto para RAM si el JSON no lo especifica.
_RAM_DEFAULT_FLAG_PCT: int = 90


def _load_thresholds() -> dict[str, tuple[tuple[float, Severity], ...]]:
    """Devuelve un dict ``detector → tabla de umbrales``.

    Si el JSON de recalibración existe, devuelve umbrales por percentil para
    los detectores que aparezcan en él. Para PSA y GSO el mapeo es
    ``low=P95 / medium=P99 / high=max`` (≈5 % / 1 % / 0,1 % de activación). Para
    RAM el umbral de *flag* (severidad ``medium``, que es donde actúa el
    override) se fija en el percentil ``flag_percentile`` calibrado sobre la
    ventana 2000-2024; ``low=P75`` (disparador suave del modo reduce) y
    ``high=P99``. Cualquier detector ausente del JSON cae a los defaults.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    table: dict[str, tuple[tuple[float, Severity], ...]] = {
        "ram": _DEFAULT_THRESHOLDS,
        "psa": _DEFAULT_THRESHOLDS,
        "gso": _DEFAULT_THRESHOLDS,
    }
    if _THRESHOLDS_FILE.exists():
        try:
            data = json.loads(_THRESHOLDS_FILE.read_text())
            for name in ("psa", "gso"):
                if name not in data:
                    continue
                dist = data[name]["score_distribution"]
                table[name] = (
                    (dist["max"], "high"),
                    (dist["p99"], "medium"),
                    (dist["p95"], "low"),
                    (0.0, "none"),
                )
            if "ram" in data:
                dist = data["ram"]["score_distribution"]
                pf = int(data["ram"].get("flag_percentile", _RAM_DEFAULT_FLAG_PCT))
                medium = float(dist[f"p{pf}"])
                high = max(float(dist["p99"]), medium)
                low = float(dist["p75"])
                table["ram"] = (
                    (high, "high"),
                    (medium, "medium"),
                    (low, "low"),
                    (0.0, "none"),
                )
        except Exception:
            pass  # Falla silenciosa: mantenemos defaults.
    _CACHE = table
    return table


def reset_thresholds_cache() -> None:
    """Limpia el caché de umbrales (útil tras regenerar el JSON o en tests)."""
    global _CACHE
    _CACHE = None


def _classify_severity_for(detector: str, score: float) -> Severity:
    """Traduce un score continuo en severidad usando los umbrales del detector."""
    table = _load_thresholds().get(detector, _DEFAULT_THRESHOLDS)
    s = max(0.0, float(score))
    for thresh, level in table:
        if s >= thresh:
            return level
    return "none"


def _classify_severity(score: float) -> Severity:
    """Alias retro-compatible: severidad con los defaults uniformes."""
    s = max(0.0, min(1.0, score))
    for thresh, level in _DEFAULT_THRESHOLDS:
        if s >= thresh:
            return level
    return "none"


def ram_detector(
    agent_size: float,
    regime_probs: dict[str, float],
) -> DetectorResult:
    """RAM — Regime-Action Mismatch.

    Asigna un *sentido permitido* por régimen y mide la masa de probabilidad
    sobre regímenes en los que la acción del agente es inconsistente. La
    política es **simétrica con el leverage effect** documentado para índices
    agregados (correlación negativa retorno/volatilidad; Black 1976; Christie
    1982): Calma ≈ drift alcista, Crisis ≈ drift bajista, Estrés indefinido.

    - Calma  → permitido long; short es inconsistente.
    - Estrés → cualquier sentido permitido, sin penalización.
    - Crisis → permitido short; long es inconsistente.

    El refinamiento *Crisis → short permitido* (frente a la regla preliminar
    "Crisis → flat") se introdujo el 2026-05-19 tras el diagnóstico del
    backtest de 21 días sobre SPY: ver entrada BITACORA correspondiente. La
    asunción implícita de RAM (régimen ≈ dirección por proxy) implica que la
    acción coherente en Crisis es el opuesto direccional de Calma, no flat.

    ``regime_probs`` es ``{"Calma": p1, "Estrés": p2, "Crisis": p3}``.
    """
    agent_sign = 0 if abs(agent_size) < 1e-9 else (1 if agent_size > 0 else -1)

    calm_prob = float(regime_probs.get("Calma", 0.0))
    crisis_prob = float(regime_probs.get("Crisis", 0.0))

    inconsistency = 0.0
    if agent_sign < 0:
        inconsistency += calm_prob
    if agent_sign > 0:
        inconsistency += crisis_prob

    score = float(min(1.0, inconsistency))
    severity = _classify_severity_for("ram", score)
    # Dirección implícita del régimen (leverage effect): Calma → long, Crisis →
    # short. ``regime_sign`` y ``p_dominant`` los consume la capa de override
    # para reorientar el sizing del agente hacia el régimen (variantes B/C).
    regime_sign = 1.0 if calm_prob >= crisis_prob else -1.0
    p_dominant = calm_prob if regime_sign > 0 else crisis_prob
    return DetectorResult(
        name="ram",
        score=score,
        flag=severity in ("medium", "high"),
        severity=severity,
        extra={
            "agent_sign": float(agent_sign),
            "calm_prob": calm_prob,
            "crisis_prob": crisis_prob,
            "regime_sign": regime_sign,
            "p_dominant": p_dominant,
        },
    )


def _psa_severity_from_runlength(run_length: int, short_window: int) -> Severity:
    """Severidad threshold-free de PSA inversa a la longitud de run del MAP.

    Un cambio reciente deja la run-length del MAP en valores bajos; la
    severidad escala inversamente: ``<=1 → high``, ``<=3 → medium``,
    ``<=short_window → low``, en otro caso ``none``. No depende de calibración.
    """
    if run_length <= 1:
        return "high"
    if run_length <= 3:
        return "medium"
    if run_length <= short_window:
        return "low"
    return "none"


def psa_detector(
    sizing_history: Sequence[float],
    short_window: int = 5,
    hazard: float = 1 / 250,
    signal: str = "cp_prob",
) -> DetectorResult:
    """PSA — Position Sizing Anomaly vía BOCPD sobre el historial de sizing.

    Señales de detección seleccionables con ``signal``:

    - ``"cp_prob"`` (por defecto): masa posterior de run-length ``r_T`` en
      ``[0, short_window]`` del BOCPD sobre los **niveles** de sizing; sube cuando
      el agente cambia de régimen interno de sizing. Severidad por umbrales del JSON.
    - ``"cp_prob_delta"``: igual que ``cp_prob`` pero el BOCPD se aplica a los
      **incrementos** ``diff(sizing)``; detecta saltos de sizing aunque el nivel
      oscile. Severidad por umbrales del JSON.
    - ``"map_runlength"``: longitud de run del MAP en ``t = T-1``;
      *threshold-free*, la severidad escala inversa a la run-length (ver
      ``_psa_severity_from_runlength``). El flag se activa si la run-length es
      ``<= short_window`` (cambio reciente).

    ``hazard`` es la tasa esperada de cambio del BOCPD (mayor = más sensible).
    Necesita al menos ``short_window + 2`` observaciones (``+3`` para la variante
    de incrementos); en caso contrario se emite *none*.
    """
    min_len = short_window + 2 + (1 if signal == "cp_prob_delta" else 0)
    if len(sizing_history) < min_len:
        return DetectorResult(
            name="psa",
            score=0.0,
            flag=False,
            severity="none",
            extra={"history_len": float(len(sizing_history))},
        )

    obs = np.asarray(sizing_history, dtype=float)
    if signal == "cp_prob_delta":
        obs = np.diff(obs)
    res = bocpd(obs, hazard=hazard, short_window=short_window)
    map_rl = int(res.map_run_length[-1])

    if signal == "map_runlength":
        severity = _psa_severity_from_runlength(map_rl, short_window)
        # Score informativo en [0, 1]: 1 cuando la run-length es 0, 0 al llegar
        # a short_window. No interviene en la lógica, solo en la traza.
        score = float(max(0.0, (short_window - map_rl) / short_window))
        flag = map_rl <= short_window
    else:
        score = float(res.cp_prob[-1])
        severity = _classify_severity_for("psa", score)
        # Con umbrales recalibrados el "flag" se alinea con severidad ``medium``
        # o superior (en defaults equivale a score ≥ 0.4).
        flag = severity in ("medium", "high")

    return DetectorResult(
        name="psa",
        score=score,
        flag=flag,
        severity=severity,
        extra={
            "history_len": float(len(sizing_history)),
            "map_run_length": float(map_rl),
        },
    )


def _gso_severity_from_ratio(ratio: float) -> Severity:
    """Severidad threshold-free de GSO a partir del ratio de riesgo ``r``.

    ``r = |size| / bound`` (1 = exactamente en el objetivo de volatilidad). La
    desviación multiplicativa de dos colas ``dev = |log2 r|`` mide a cuántos
    factores 2 está la posición del objetivo, sea por exceso o por defecto:
    ``<0.3 (≈±1.23×) → none``, ``<0.585 (≤1.5×) → low``, ``<1 (≤2×) → medium``,
    ``>=1 (>2×) → high``. No depende de calibración.
    """
    if ratio <= 0.0:
        return "high"
    dev = abs(np.log2(ratio))
    if dev < 0.3:
        return "none"
    if dev < 0.585:
        return "low"
    if dev < 1.0:
        return "medium"
    return "high"


def gso_detector(
    agent_size: float,
    sigma_t_annualized: float,
    target_vol: float = TARGET_VOL,
    gso_mode: str = "absolute",
) -> DetectorResult:
    """GSO — GARCH-bounded Sizing Override.

    Banda de volatilidad ``bound = clip(target_vol / sigma_t, 0, 1)``. Tres modos:

    - ``"absolute"`` (por defecto): detector de **sobreexposición**. Score
      ``(|size| - bound) / max(bound, eps)`` (>0 solo si el agente excede la
      banda); severidad por los umbrales calibrados del JSON; ``bounded_size =
      sign·min(|size|, bound)`` (solo capa hacia abajo). Comportamiento histórico.
    - ``"relative"``: detector de **desviación de vol-target de dos colas**
      (threshold-free, ver ``_gso_severity_from_ratio``). ``bounded_size =
      sign(agent)·bound`` (reescala al objetivo de vol conservando la dirección).
    - ``"relative_conviction"``: como ``relative`` pero conservando el gradiente
      de convicción: ``bounded_size = sign(agent)·clip(|size|/AGENT_MAX_SIZE,0,1)·bound``.

    En los modos relativos el detector también dispara ante **infra-exposición**,
    de modo que en *override* reescala la posición del agente al objetivo de vol.
    """
    if sigma_t_annualized <= 0:
        bound = 1.0
    else:
        bound = float(min(1.0, target_vol / sigma_t_annualized))

    abs_size = abs(float(agent_size))
    sign = 0.0 if abs_size < 1e-12 else float(np.sign(agent_size))

    if gso_mode in ("relative", "relative_conviction"):
        ratio = abs_size / max(bound, 1e-3)
        severity = _gso_severity_from_ratio(ratio)
        score = float(abs(np.log2(ratio)) if ratio > 0 else 3.0)
        if gso_mode == "relative_conviction":
            conviction = min(1.0, abs_size / AGENT_MAX_SIZE)
            bounded_size = float(sign * conviction * bound)
        else:
            bounded_size = float(sign * bound)
        flag = severity != "none"
    else:  # "absolute": comportamiento histórico de sobreexposición.
        exceso = max(0.0, abs_size - bound)
        # El score GSO ya no se clipea a 1.0: el wrapper compara contra los
        # umbrales recalibrados que pueden estar muy por encima de 1 (P95 ≈ 2.4).
        score = float(exceso / max(bound, 1e-3))
        severity = _classify_severity_for("gso", score)
        bounded_size = float(sign * min(abs_size, bound))
        flag = score > 0

    return DetectorResult(
        name="gso",
        score=score,
        flag=flag,
        severity=severity,
        extra={
            "bound": bound,
            "bounded_size": bounded_size,
            "sigma_t": float(sigma_t_annualized),
        },
    )
