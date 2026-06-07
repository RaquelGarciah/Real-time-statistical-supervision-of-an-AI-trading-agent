# Incidencias conocidas

## Look-ahead de 1 día en el backtest — RESUELTO (2026-05-20)

**Síntoma.** El motor de backtest aplicaba `peso_d × retorno_d` sin desfase. Pero la decisión
para el día *d* usa información hasta el cierre de *d*, así que multiplicarla por el retorno del
mismo día introduce un look-ahead de 1 día: el sistema "sabía" el retorno que iba a aprovechar.

**Corrección.** Parámetro `signal_lag: int = 1` (por defecto) en `core/backtest.run_backtest`:
`w = w.shift(signal_lag)` antes de `gross = w * r`. La decisión en *t* se aplica al retorno *t+1*.
Un único punto arregla los ≥7 llamantes. `tests/test_backtest.py` actualizado; los harness duales
pasan `signal_lag` explícito (0 = same-day, 1 = causal). M1–M2, M5–M8 y la ablación se
re-ejecutaron; M3/M4/M9 se re-evaluaron desde los pesos almacenados (deterministas, sin
reentrenar H2O).

**Impacto en los resultados (Sharpe causal neto, definitivos):**

| Config | Causal | Nota |
|---|---|---|
| M1 (B&H) | +1,01 | |
| **M2 (B&H + GARCH×HMM)** | **+0,77** | benchmark cuantitativo |
| M3 (H2O KFold) | −0,44 | |
| **M4 (H2O CPCV + sizing)** | **+0,48** | el same-day era −2,28: el look-ahead **invertía** el ranking ML |
| M5 (agente solo) | −1,83 | sin edge causal |
| M6 (STRATA warn) | −1,77 | |
| M7 (STRATA reduce) | −0,95 | |
| **M8 (STRATA override C + filtered)** | **+0,66** | el overlay de régimen causal rescata a la IA |
| M9 (ML + IA) | −1,13 | |

**Jerarquía causal:** la estadística (M1/M2) y el ML purgado (M4) tienen edge; STRATA override C
(M8) convierte la IA en positiva; la IA cruda (M5) y la supervisión por atenuación (M6/M7/M9)
quedan negativas.

**Aviso para la memoria.** Todas las cifras *same-day* previas quedan obsoletas (eran artefacto
del look-ahead). Cualquier "mejor Sharpe" alto debe validarse en **causal neto** antes de creerlo;
varias mejoras aparentes (GSO relativo, régimen *smoothed*) eran fugas de información. Detalle en
`BITACORA.md`, entrada `[Milestone] 2026-05-20`. Relación con el techo de supervisión en
[hallazgos_strata.md](hallazgos_strata.md).
