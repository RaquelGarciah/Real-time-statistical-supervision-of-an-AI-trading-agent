"""Hiperparámetros centralizados, semillas y fechas del proyecto STRATA.

Cualquier constante reproducible vive aquí. El resto del código importa de este
módulo y no redefine valores. Las semillas se fijan en una sola función
``set_seeds`` que debe llamarse al principio de cada script de experimento.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# Semilla global. Todo lo aleatorio del proyecto se siembra desde aquí.
SEED: int = 42

# Rutas base
ROOT_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = ROOT_DIR / "cache"
CACHE_LLM_DIR: Path = CACHE_DIR / "llm"
CACHE_AGENT_DIR: Path = CACHE_DIR / "agent"
CACHE_MODELS_DIR: Path = CACHE_DIR / "models"
DATA_DIR: Path = ROOT_DIR / "data"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
EXPERIMENTS_DIR: Path = OUTPUTS_DIR / "experiments"
LIVE_DIR: Path = OUTPUTS_DIR / "live"

# Ventanas temporales (pivot BITACORA 2026-05-19: calibración extendida hasta
# 2024-09-30 para alinearla con el OOS unificado; el antiguo OOS de OS1 queda
# absorbido en el experimento comparativo de 9 configuraciones).
CALIBRATION_START: str = "2000-01-01"
CALIBRATION_END: str = "2024-09-30"
STRATA_OOS_START: str = "2024-10-01"

# Activos de mercado usados como fuente de régimen y volatilidad.
# TICKER_PRIMARY es el subyacente del experimento unificado (SPY, justificado
# por el leverage effect; ver CLAUDE.md sección 1). TICKER_INDEX se conserva
# para series de régimen calculadas sobre el índice no tradeable.
TICKER_PRIMARY: str = "SPY"
TICKER_INDEX: str = "^GSPC"
TICKER_VIX: str = "^VIX"

# HMM gaussiano (núcleo del detector RAM)
HMM_N_STATES: int = 3  # Calma, Estrés, Crisis
HMM_COVARIANCE_TYPE: str = "full"
HMM_N_ITER: int = 1000

# GARCH(1,1) Student-t (núcleo del detector GSO)
GARCH_P: int = 1
GARCH_Q: int = 1
GARCH_DIST: str = "t"

# Tope empírico del |size| del agente (risk manager de AI Hedge Fund). Lo usa el
# modo GSO 'relative_conviction' para normalizar la convicción al presupuesto de
# volatilidad (ver strata/detectors.py y BITACORA 2026-05-20).
AGENT_MAX_SIZE: float = 0.25

# BOCPD (núcleo del detector PSA)
BOCPD_HAZARD: float = 1 / 250  # tasa esperada de un cambio cada año bursátil

# Costes de transacción del backtest
COST_BPS: float = 1.0  # 1 punto básico por operación

# LLM (DeepSeek V3 vía OpenRouter; decisión pivot BITACORA 2026-05-19). El
# benchmark del 2026-05-14 priorizó gpt-oss-120b por menos errores de parsing,
# pero la migración a DeepSeek V3 es parte del pivot al experimento unificado.
LLM_MODEL: str = "deepseek/deepseek-chat"
LLM_PROVIDER: str = "openrouter"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 2048
LLM_SEED: int = SEED

# Personalidades activas de AI Hedge Fund. Bill Ackman sustituye a Howard
# Marks porque este último no está implementado en el submódulo; ver entrada
# BITACORA 2026-05-14 con la justificación.
ACTIVE_PERSONALITIES: tuple[str, ...] = (
    "warren_buffett",
    "cathie_wood",
    "stanley_druckenmiller",
    "michael_burry",
    "bill_ackman",
)


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de numpy, random y la variable de entorno PYTHONHASHSEED.

    Llamar al inicio de cada script de experimento para garantizar determinismo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
