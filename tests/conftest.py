"""Configuración de pytest.

Omite en la fase de colección los tests que requieren dependencias no siempre
presentes: el runtime del agente (langchain + submódulo ``agent/``), los datos
locales de ``data/`` (gitignored) o módulos de ``experiments/`` aún no creados.
Así la CI ligera corre verde sin instalar el stack pesado, y en local —con el
entorno completo— esos tests se ejecutan con normalidad.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
collect_ignore: list[str] = []


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


# Dependen del runtime del agente (langchain + agent/): se omiten si no está.
if not _has("langchain_core"):
    collect_ignore += ["test_common.py", "test_llm_client.py", "test_wrapper.py"]

# Necesita los parquet de data/ (gitignored, ausentes en la CI).
if not (_ROOT / "data").exists():
    collect_ignore.append("test_data.py")

# Importa experiments/decision_level_analysis.py (aún no presente en el repo).
if not (_ROOT / "experiments" / "decision_level_analysis.py").exists():
    collect_ignore.append("test_decision_level.py")
