"""Caché en disco de inferencias LLM como ``BaseCache`` de LangChain.

LangChain expone ``langchain_core.caches.BaseCache`` como interfaz para
persistir respuestas. Aquí implementamos uno con un fichero JSON por
``(prompt, llm_string)`` nombrado por SHA-256, según la convención de
``CLAUDE.md`` sección 7.1. La estructura de cada fichero JSON es:

```
{
  "model": "<llm_string opaco que LangChain pasa>",
  "messages_hash": "sha256:...",
  "prompt": "...",
  "response": "<contenido devuelto por el LLM>",
  "timestamp_utc": "2026-...",
  "parameters": {"temperature": 0.0, "seed": 42}
}
```

Para activarlo globalmente: ``langchain_core.globals.set_llm_cache(JSONFileCache(...))``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.outputs import Generation

from config import CACHE_LLM_DIR, LLM_SEED, LLM_TEMPERATURE


def _hash_key(prompt: str, llm_string: str) -> str:
    """Hash SHA-256 de ``(llm_string + prompt)``. ``llm_string`` ya codifica
    modelo, temperatura, etc., así que basta concatenar.
    """
    h = hashlib.sha256()
    h.update(llm_string.encode("utf-8"))
    h.update(b"\n----\n")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class JSONFileCache(BaseCache):
    """Caché de inferencias LLM persistida como JSON por hash.

    Cada entrada se almacena en ``{cache_dir}/{sha256}.json``. ``lookup``
    devuelve la lista de ``Generation`` reconstruida; ``update`` la persiste.
    Pensado para uso single-process (no se protege con locks); apto para los
    backtests y el modo live de STRATA, que son secuenciales.
    """

    def __init__(self, cache_dir: Path = CACHE_LLM_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, prompt: str, llm_string: str) -> Path:
        return self.cache_dir / f"{_hash_key(prompt, llm_string)}.json"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            return None
        with open(path) as f:
            payload = json.load(f)
        return [Generation(text=payload["response"])]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        path = self._path(prompt, llm_string)
        text = "\n---\n".join(getattr(g, "text", "") for g in return_val)
        payload = {
            "llm_string": llm_string,
            "messages_hash": _hash_key(prompt, llm_string),
            "prompt": prompt,
            "response": text,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "parameters": {"temperature": LLM_TEMPERATURE, "seed": LLM_SEED},
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def clear(self, **kwargs) -> None:
        """Borra todas las entradas de la caché en disco."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


def enable_global_cache() -> JSONFileCache:
    """Activa la caché global de LangChain con persistencia en ``cache/llm/``."""
    from langchain_core.globals import set_llm_cache

    cache = JSONFileCache()
    set_llm_cache(cache)
    return cache
