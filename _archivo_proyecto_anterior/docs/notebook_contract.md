# Contrato del notebook atómico

Define qué debe hacer el notebook end-to-end que se construye en el siguiente paso. El principio
es **atomicidad**: un único notebook autocontenido que se ejecuta de arriba a abajo y reproduce
todos los resultados de la memoria, **parametrizado por un solo `TICKER`** al inicio. Cambiando
`TICKER = "SPY"` por `TICKER = "NVDA"` (o cualquier activo con caché de agente) se reproduce la
extensión correspondiente sin tocar nada más.

## Qué se LEE de caché (no se regenera)

- **Decisiones del agente:** `cache/agent/<TICKER>/<TICKER>_<YYYY-MM-DD>.json`. Son las salidas
  de las cinco personalidades + Portfolio Manager, irreplazables (cuestan cuota de API/LLM). El
  notebook **no llama nunca** a OpenRouter ni a la API de datos. SPY (~401 días) y NVDA (~409
  días) están cacheados al completo → corren 100 % offline.
- **Metadatos H2O AutoML:** `cache/models/h2o_m3.json`, `h2o_m4.json`, `h2o_m9.json` (algoritmo
  ganador, métrica, semilla). El leaderboard no se reentrena en el notebook.

## Qué se RECALCULA en el notebook (determinista, semilla fija)

- HMM de 3 estados y GARCH(1,1) Student-t sobre la ventana de calibración (2000-01-01 →
  2024-09-30). Se pueden cargar de `cache/models/*.pkl` o reajustar inline (verificado determinista).
- Features (`ret_log`, `rv_21_ann`, RSI, SMA, momentum, lags).
- Backtests M1–M9 sobre el OOS, con `signal_lag=1` (causal) — ver [known_issues.md](known_issues.md).
- Los tres detectores (RAM, PSA, GSO), la capa de intervención y la ablación.
- Tests estadísticos (DSR, bootstrap, Diebold-Mariano) y la matriz pareada 9 × 9.
- Todas las figuras (se generan dentro del notebook, no se importan de `viz/`).

## Qué se DESCARGA (red, gratis, regenerable)

- Datos de mercado vía yfinance a `data/*.parquet` (precio de `<TICKER>`, `^GSPC`, `^VIX`). Si el
  parquet existe se lee de disco; si no, se descarga. Es la única dependencia de red.

## Frontera para un activo nuevo

Para un activo **sin** caché de agente, solo la parte de agente (M5–M9) requeriría ejecutar el
AI Hedge Fund (API + LLM). Todo lo demás (M1–M4, detectores sobre una decisión dada, tests) es
reproducible sin red más allá de yfinance. Por eso la caché de agente se versiona por activo:
añadir `cache/agent/<NUEVO>/` habilita el notebook para ese activo.

## Determinismo

Semilla única `SEED = 42` fijada al inicio (numpy, random, PYTHONHASHSEED). HMM prueba 10
semillas y elige por log-score (selección determinista). GARCH se ajusta una vez y se congela en
el OOS. Misma entrada → misma salida.
