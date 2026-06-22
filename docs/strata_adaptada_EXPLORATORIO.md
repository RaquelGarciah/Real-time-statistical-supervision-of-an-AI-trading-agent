# STRATA adaptada — ¿se alcanza STRATA-U desde el supervisor real? (EXPLORATORIO)

> **Estado: EXPLORATORIO.** Trabajo en pruebas (rama `feat/quant-validation-panel`). No es referencia
> de la memoria ni de la BITACORA canónica hasta que Raquel lo valide. Ver
> `[[trabajo-exploratorio-aislado]]`.

## La pregunta

¿Se puede llevar **M8** (la STRATA actual: override-C, supervisa al agente) hasta **STRATA-U** (régimen
al mando + vol-target) **moviendo umbrales y haciendo que RAM dispare más**, sin cambiar la lógica de
intervención, para presentar **una sola** estrategia parametrizada en lugar de dos deterministas casi
iguales?

## Diagnóstico (verificado en el código)

El RAM histórico (modo **mismatch**) solo dispara cuando el agente **contradice** al régimen: su score
es la masa de probabilidad del régimen incoherente (`calm_prob` si el agente va corto, `crisis_prob` si
va largo). Por eso **bajar el umbral no basta**: solo intensifica los overrides de contradicción y nunca
aprovecha el régimen cuando el agente coincide o se abstiene.

**Cambio implementado** (retrocompatible, tests verdes): `ram_detector` y `StrataSupervisor` admiten
`ram_mode`:
- `"mismatch"` (por defecto, = M8 histórico).
- `"regime"` (nuevo): score = confianza del régimen direccional dominante `max(P(Calma),P(Crisis))`.
  RAM dispara con la **confianza** del régimen, con independencia del signo del agente; el override-C
  —idéntico— impone `regime_sign · bound`. Aprovecha el régimen también cuando el agente coincide o se
  abstiene.

Y `regime_sign_map` para inyectar el **signo data-driven** del régimen por activo (no el leverage
hardcoded; CLAUDE.md §9).

## Método

`experiments/strata_adaptada.py` barre configuraciones del **mismo** `StrataSupervisor` (override-C)
sobre los 13 activos con caché completa, ventana `mv[150:]` (≡ ventana de M10), sizing **justo** (mismo
vol-target para todas → Sharpe/maxDD comparables, aíslan la dirección). El override-C se calcula
vectorizado (GSO paso 1 + RAM paso 2), omitiendo solo el freno PSA (×0.5 en transición, no cambia
dirección ni vol-target). **Sanity verificado**: la config `mismatch/absolute/τ=0.5` reproduce la
accuracy de M8 de `fair_sizing_compare.json` con diferencia 0,000 en los 13 activos. M10 se lee de
`fair_sizing_compare.json` (misma ventana; no se recomputa).

Configs (todas = el mismo supervisor; solo cambian estos campos): modo RAM (mismatch/regime), umbral τ,
GSO (absolute/relative) y fuente del signo (leverage / data-driven estático de calibración / **causal
expansible `s_dom`**, el que usa STRATA-U).

## Resultados (medias del panel, sizing justo) — [verificado]

| Estrategia | acc | Sharpe | maxDD | Calmar |
|---|---:|---:|---:|---:|
| ZeroR (trivial) | **0,550** | **1,17** | −7,2% | **1,67** |
| Régimen | 0,537 | 0,77 | −9,2% | 1,33 |
| **A_reg_sdom_τ00** (adaptada) | **0,537** | **0,77** | −9,2% | 1,33 |
| B&H | 0,533 | 0,80 | −9,2% | 1,31 |
| STRATA-U | 0,530 | 0,73 | −8,2% | 1,11 |
| A_reg_lev_τ50 (regime, leverage) | 0,511 | 0,42 | −9,1% | 0,61 |
| A_reg_dd_τ50 (regime, calib. estático) | 0,503 | 0,29 | −9,3% | 0,49 |
| M8 / A_mm_τ50_abs | 0,503 | 0,48 | −8,5% | 0,67 |
| A_mm_τ30_abs (solo bajar umbral) | 0,503 | 0,46 | −8,7% | 0,67 |
| M10 | 0,501 | −0,12 | −11,1% | 0,58 |
| M5 (agente) | 0,464 | −0,86 | −12,7% | −0,37 |

Cobertura de `A_reg_sdom_τ00`: **acc≥STRATA-U 11/13, Sharpe≥STRATA-U 11/13**; acc≥M8 11/13; pero
**acc>ZeroR 0/13** y Sh>ZeroR 0/13.

## Por qué M8 (actuar solo en contradicción) < seguir el régimen — [verificado]

`experiments/m8_nonintervention_diag.py` → `m8_nonintervention_diag.json`. RAM (mismatch) **no comprueba
que el agente esté alineado**, solo que no haya contradicción fuerte (masa ≥ τ=0,5): **silencio ≠
aprobación**. Descomposición de la ventana de M8 (media panel):

| | media |
|---|---:|
| % días que M8 NO interviene | 63% |
| coincidencia agente↔régimen en esos días | **48%** (no ~100%) |
| acc del agente en esos días | **0,465** (bajo el azar) |
| acc que habría dado el régimen | **0,534** |
| M8 total / Régimen total | 0,503 / 0,537 |

M8 entrega la "zona gris" (contradicciones débiles, régimen poco confiado, agente plano) a un predictor
malo → ahí está el gap. **No universal:** en prior-flip (MSTR 0,556 > régimen 0,452) M8 gana por seguir
al agente; por eso llegar a STRATA-U exigió el signo causal `s_dom` (baja τ donde el régimen es bueno y
**se apaga** donde es malo; el leverage hardcoded no se apaga).

## Conclusión

1. **Sí se unifican M8 y STRATA-U en UNA estrategia parametrizada** del mismo supervisor (override-C
   intacto). El **dial es τ** = tasa de intervención: M8 ≈ extremo conservador (agente por defecto,
   ~34 % de intervención), STRATA-U ≈ extremo agresivo (régimen al mando, ~78 %). Elimina la redundancia
   de presentar dos cosas casi iguales.

2. **Para alcanzar STRATA-U NO basta mover umbrales.** Hacen falta, a la vez: (a) RAM en modo **regime**,
   (b) **signo causal-expansible `s_dom`** (el estático de calibración / leverage **no llega**: falla en
   los activos con *prior-flip*), (c) **τ→0** + GSO `relative`. Con las tres, la adaptada **iguala
   exactamente a Régimen y supera a STRATA-U en 11/13**. A τ=0 con signo `s_dom`, el override-C ES seguir
   el régimen cada día con vol-target (de ahí que `A_reg_sdom_τ00 ≡ Régimen` en accuracy, exacto).

3. **Caveat (no negociable).** A τ=0 el agente queda **fuera** → deja de ser *supervisión del agente* y
   pasa a ser **timing de régimen**. Y **nadie bate a ZeroR**. La elección de marco para la memoria
   sigue: **supervisión → M8** (bate al agente, pooled-significativo); **timing → régimen / STRATA
   adaptada con τ→0** (mejor señal no trivial, no supera a las triviales).

4. **Implementación.** `ram_mode` + `regime_sign_map` (estático) ya están en el supervisor. Para que
   produzca *nativamente* STRATA-U falta alimentarle el signo `s_dom` **causal por día** (hoy se
   demuestra en el harness `experiments/strata_adaptada.py`); extensión pequeña, no método nuevo.

## Ficheros

- Código: `strata/detectors.py` (`ram_detector` con `mode`/`regime_sign_override`), `strata/strata.py`
  (`ram_mode`/`regime_sign_map`).
- Experimento: `experiments/strata_adaptada.py` → `outputs/experiments/strata_adaptada.json`.
- Notebook: `notebooks/STRATA_adaptada.ipynb` (builder `_build_strata_adaptada.py`).
