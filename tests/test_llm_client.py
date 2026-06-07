"""Tests de ``agent.llm_client``."""

from __future__ import annotations

from langchain_core.outputs import Generation

from agent.llm_client import JSONFileCache, _hash_key


def test_hash_es_estable():
    a = _hash_key("hola", "openrouter:nemotron")
    b = _hash_key("hola", "openrouter:nemotron")
    assert a == b
    assert len(a) == 64


def test_hash_distingue_prompts():
    a = _hash_key("hola", "X")
    b = _hash_key("adios", "X")
    assert a != b


def test_lookup_y_update_persisten(tmp_path):
    cache = JSONFileCache(cache_dir=tmp_path)
    assert cache.lookup("p1", "modelo:x") is None
    cache.update("p1", "modelo:x", [Generation(text="respuesta-de-prueba")])
    rec = cache.lookup("p1", "modelo:x")
    assert rec is not None
    assert rec[0].text == "respuesta-de-prueba"


def test_clear_borra_entradas(tmp_path):
    cache = JSONFileCache(cache_dir=tmp_path)
    cache.update("p1", "x", [Generation(text="r1")])
    cache.update("p2", "x", [Generation(text="r2")])
    assert len(list(tmp_path.glob("*.json"))) == 2
    cache.clear()
    assert len(list(tmp_path.glob("*.json"))) == 0


def test_misma_clave_distinto_directorio_no_solapa(tmp_path):
    """Dos cachés en directorios distintos no se confunden."""
    a = JSONFileCache(cache_dir=tmp_path / "a")
    b = JSONFileCache(cache_dir=tmp_path / "b")
    a.update("p", "x", [Generation(text="va")])
    assert b.lookup("p", "x") is None
    b.update("p", "x", [Generation(text="vb")])
    assert a.lookup("p", "x")[0].text == "va"
    assert b.lookup("p", "x")[0].text == "vb"
