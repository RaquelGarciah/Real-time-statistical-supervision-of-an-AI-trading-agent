# Backtest largo del canal de régimen (sin agente), 2020–2024 — EXPLORATORIO

> **Estado: EXPLORATORIO.** No canónico. Ver `[[trabajo-exploratorio-aislado]]`.

## Motivación

Régimen / B&H / ZeroR **no dependen del agente LLM** → se pueden backtestear en ventanas largas con
crisis reales (el OOS del TFG, post-2024-10, no tiene crash). Pregunta: **¿el timing de régimen bate a
comprar-y-mantener cuando SÍ hay crisis que esquivar?**

## Diseño (causal, sin look-ahead)

- Ventana común a los 15 activos (ROKU es el más joven, IPO 2017-09).
- Calibración por activo con su propia historia **≤2019-12-31** (HMM K=3 + GARCH).
- Test común **2020-01-01 → 2024-09-30** (incluye crash COVID-2020 y bear-2022).
- `s_dom` expansible y causal (sembrado con calibración ≤2019, actualizado hasta t-1). `signal_lag=1`.
- Régimen/B&H/ZeroR con el MISMO sizing vol-target (comparable); `B&H_1x` = comprar y mantener real (1x).
- Fuente: `experiments/regime_largo.py` → `outputs/experiments/regime_largo.json`.

## Resultado — [verificado]

Medias del panel (15 activos, ~1190 días cada uno):

| Estrategia | acc | Sharpe | maxDD | Calmar | equity |
|---|---:|---:|---:|---:|---:|
| Régimen | 0,526 | **0,41** | −20,3% | 0,27 | 1,221 |
| B&H (vol-target) | 0,524 | **0,50** | −20,4% | 0,29 | 1,278 |
| ZeroR | 0,528 | 0,50 | −20,7% | 0,30 | 1,278 |
| B&H_1x (real) | 0,524 | 0,44 | **−64,4%** | 0,17 | 2,154 |

**Régimen bate a B&H:** Sharpe 5/15 · Calmar 5/15 · acc 4/15. **A ZeroR:** 4/15 · 4/15 · 2/15.

## Conclusión honesta (corrige una hipótesis previa)

1. **NO. Aun con crisis dentro (2020-2024), el timing de régimen NO bate a B&H/ZeroR de media** (Sharpe
   0,41 vs 0,50; gana solo en 5/15). Esto **corrige** lo que aventuré antes ("la limitación era solo la
   ventana sin crisis"): el problema es más de fondo — el régimen no añade ventaja neta sobre B&H a lo
   largo de un ciclo completo, porque la protección en la caída se devuelve en la recuperación.

2. **PERO el régimen SÍ amortigua las crisis** (su valor es real pero localizado):
   - **bear-2022:** Régimen ret **−3,1%** / maxDD −12,2% vs B&H **−8,1%** / −13,0%.
   - **COVID-2020:** Régimen ret −5,9% / maxDD −9,8% vs B&H −6,8% / −10,8%.
   En ambas perdió menos. La señal de régimen (RAM) **es real** y reduce la pérdida en el crash; lo que
   no consigue es batir a B&H sobre el ciclo entero.

3. **El control de drawdown viene del vol-targeting (GARCH/GSO), no de la dirección del régimen:** B&H_1x
   tiene maxDD **−64%** frente a **−20%** de las versiones vol-target. Resultado limpio y defendible para
   el detector **GSO**: la gestión de riesgo la hace el sizing por volatilidad.

4. Régimen sigue fallando donde **prior-flip** (MSTR Sharpe −0,19, MARA −0,20): se pone corto cuando no
   debe. Coherente con la falsación pre-registrada.

## Lectura para el TFG

Separa dos cosas y ambas son honestas: (a) **el detector GSO/vol-target controla el riesgo** (−20% vs
−64% maxDD) — robusto y universal; (b) **el detector RAM/régimen amortigua los crashes** (2022: −3% vs
−8%) pero **no bate a comprar-y-mantener** a lo largo del ciclo. Ninguna de las dos contradice la tesis
de supervisión (que vive en el OOS post-2024 con agente); la matizan: el valor estadístico está en el
**riesgo**, no en superar al mercado en retorno.
