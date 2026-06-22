# ¿Una búsqueda AutoML bate a M10-XGBoost (o al techo ZeroR)? (EXPLORATORIO)

> **Estado: EXPLORATORIO.** No canónico. Un solo activo (SPY), una sola ventana OOS, n_test=251.
> Budget de búsqueda modesto (30 s/reentreno). Descriptivo, no confirmatorio. Ver `[[trabajo-exploratorio-aislado]]`.

## Pregunta

M10 fija **un único estimador** (XGBoost) por pre-registro. ¿Esa restricción deja rendimiento sobre la mesa?
Si se afloja y se deja que un **AutoML** busque sobre muchas familias de modelos, ¿encuentra algo que
XGBoost no encontró? ¿Bate al techo **ZeroR causal** que ninguna señal direccional ha superado en el OOS
(memoria `[[accuracy-techo-zeror-oos]]`)?

Encaje en la hipótesis del TFG: **§2 nivel 3 (universalidad)**. Se espera que AutoML **NO** bata a M10/ZeroR;
si lo hiciera, refutaría que XGBoost ya captura toda la señal. Un AutoML que no mejora es la confirmación
fuerte de la universalidad, no un resultado negativo.

## Método

H2O AutoML (3.46) busca sobre **GLM, GBM, Random Forest, Deep Learning, XGBoost y StackedEnsembles** en
**exactamente el mismo pipeline causal que M10**:

- Mismas **ALL22** features (15 personalidades sign/size/conf + 7 STRATA/régimen), mismo target `signo(r_{t+1})`.
- **Walk-forward expandible** anclado, reentreno mensual (STEP=21, N0=150), **embargo=5**, idéntico a
  `walkforward_m10_causal.py`.
- Validación interna de AutoML por **Purged K-Fold** con embargo (`core/h2o_automl.py`, López de Prado 2018
  sec. 7.4) → sin fuga temporal en la selección del leader.
- Comparación contra **M5, M8, M10-XGB, ZeroR causal y B&H** con McNemar pareado, sign test y ΔSharpe
  bootstrap estacionario.

`experiments/automl_m10.py` · `outputs/experiments/automl_m10.json`. Pre-registro en el docstring del script.

## Resultado — [verificado] (exploratorio)

SPY, OOS 2025-05-09 → 2026-05-11, n_test=251, 12 reentrenos, budget=30 s/reentreno.

| Brazo | Accuracy | Sharpe causal |
|---|---:|---:|
| M5 (agente) | 0,367 | −3,07 |
| M8 (STRATA) | 0,442 | −0,46 |
| M10-XGB | 0,534 | 0,50 |
| **AutoML** | **0,546** | 1,27 |
| ZeroR causal / B&H | 0,566 | 2,21 |

McNemar pareado:

| Comparación | p | b (ref solo) | c (AutoML solo) |
|---|---:|---:|---:|
| AutoML vs M10-XGB | **0,836** | 45 | 48 |
| AutoML vs ZeroR | 0,625 | 36 | 31 |
| AutoML vs M5 | 0,0008 | 63 | 108 |
| AutoML vs M8 | 0,027 | 51 | 77 |

Sign test AutoML vs azar: p=0,165 (k=137/251, IC95 [0,482, 0,609]). Familias ganadoras por reentreno:
**DeepLearning, GBM, StackedEnsemble** (ni siquiera gana XGBoost de forma consistente).

## Conclusión

1. **AutoML ≈ M10-XGBoost** (McNemar p=0,84, accuracy 0,546 vs 0,534). Aflojar "un solo estimador" y dejar
   que la búsqueda automática explore 6 familias **no encuentra nada que XGBoost no encontrara**.
2. **AutoML NO bate al techo ZeroR** (0,546 < 0,566, p=0,63) ni al azar direccional (sign test p=0,17).
   Coherente con `[[accuracy-techo-zeror-oos]]`: ninguna señal direccional supera a ZeroR causal en el OOS.
3. **Rescata al agente igual que M10** (vs M5 p=0,0008, vs M8 p=0,027) — misma historia: el valor está en
   corregir al agente cuando pierde, no en accuracy absoluta.
4. **Refuerza la universalidad (§2 nivel 3):** la búsqueda de modelos **redescubre**, no bate. Es el
   resultado que conviene a la tesis.

## Limitaciones (objeción del tutor)

- **Budget modesto** (30 s/reentreno). Mitiga la objeción que AutoML ≈ XGBoost (p=0,84) y que con 6 familias
  no se acerca a ZeroR, pero una corrida de mayor presupuesto la blindaría del todo. Pendiente.
- Un activo (SPY), una ventana OOS. No multi-activo.
- **Multiple testing:** AutoML prueba decenas de modelos; cualquier p crudo favorable debería descontarse.
  No aplica aquí (no hubo p favorable), pero queda registrado en `leaders_por_reentreno`.
