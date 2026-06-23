# Revisión iterada del notebook definitivo del marco práctico

Bucle constructor ↔ revisora (`raquel-quant`, quant senior + matemática) sobre
`notebooks/STRATA_marco_practico.ipynb`, hasta declararlo definitivo contra el gate G1–G6.
Cierre: **APROBADO sin condiciones en la ronda 2** (2026-06-23).

## Ronda 1 — BLOQUEADO (3 fixes de honestidad)

La revisora aprobó G1 (estructura), G2 (rigor) y G4–G6, pero **bloqueó por G3 (honestidad)** al detectar un
error de signo en §4.2:

1. **§4.2 leverage effect — error de signo (FALLA G3).** La celda imprimía *"Signo del retorno medio en Crisis:
   +1 → negativo = leverage effect"*: el valor era **positivo** (+0.00138 en el OOS) pero el texto afirmaba
   "negativo". Además medía el retorno **contemporáneo** (mismo día), no el del día siguiente, así que ni
   siquiera justificaba `signal_lag=1`. Empíricamente el régimen separa por **volatilidad**, no por dirección
   del día siguiente (en calibración, SPY Crisis `ret_dia_sig`=+0.000147, `frac_sube`=0.5246).
2. **§4.2 vs §4.6 — auto-contradicción.** §4.2 usaba la media de Crisis como "leverage direccional" y §4.6 la
   usaba (en SMCI) como "régimen no direccional". El mismo hecho sostenía dos conclusiones opuestas.
3. **§4.1 — rótulo de ventana temporal.** Imprimía *"OOS SPY: 2025-05-09 → 2026-05-11"* como si fuera el OOS,
   cuando es la **ventana de evaluación post-burn-in**; el OOS empieza en 2024-10-01.

## Ronda 1 — construcción (fixes aplicados)

El constructor reescribió el builder `_build_STRATA_marco_practico.py`:
- **Fix 1/2:** §4.2 ya no imprime el signo contradictorio. Presenta el leverage effect como **relación
  contemporánea** leyendo `regime_direction_table.json` (calib SPY): `ret_mismo_dia` baja monótono
  Calma +0.000536 > Estrés +0.000167 > Crisis −0.000002; y muestra que el régimen **no predice** el día
  siguiente (`frac_sube_sig` ≈ 0.52). §4.2 y §4.6 usan ya el mismo criterio: el HMM separa por volatilidad; la
  utilidad direccional es **condicional al leverage** (fuerte en SPY, débil en SMCI).
- **Fix 3:** §4.1 distingue *"Inicio del OOS: 2024-10-01"* de *"Ventana de evaluación post-burn-in:
  2025-05-09 → 2026-05-11"*.

## Ronda 2 — APROBADO (sin condiciones)

La revisora verificó que los tres fixes están genuinamente cerrados contra el notebook reejecutado (0 errores,
auto-test verde) y el builder, y **cotejó toda cifra headline contra su JSON**:

- SPY (panel mm25): M5 0.3665 / M8 0.4422 / M10 0.4940 / **AutoML 0.5737** / ZeroR 0.5657; McNemar vs M5
  0.0002/0.0074/0.0509, vs ZeroR **0.902/0.133** (nominal).
- Pooled (decision_automl_prep): M8 vs M5 ΔSharpe **+0.664 IC[0.225,1.157]** sig, ΔmaxDD **+0.242
  IC[0.017,0.445]** sig.
- Universalidad: cuota SHAP STRATA media **0.6629**. Clustering: Rand kmeans~ward **1.0**. SPY ablación sobre
  momentum 0.521→0.582 (Δ+0.061, 3/5 sig). SMCI binomial vs NIR 0.141 (nominal).

**Veredicto final:** los seis gates PASA, O1–O6 ✓. La línea roja de honestidad se respeta (las únicas menciones
a "batir al mercado"/"genera alfa" son negaciones). El cuaderno convence (G6) de que supervisar
estadísticamente a un agente IA aporta **valor diferencial probable** (rescate en accuracy vs M5 + riesgo
pooled, universalidad SHAP, patrón clustering), no solo de que la IA pierde. **Entregable defendible.**

## Lo que un tribunal cazaría primero (anotado por la revisora)

1. **Potencia estadística:** McNemar vs ZeroR p=0.90 con n=251 — ausencia de significancia ≠ equivalencia. El
   cuaderno ya lo etiqueta como nominal / línea futura.
2. **Crisis OOS de SPY (+0.00138, n=29):** signo opuesto a la calibración. El cuaderno lo desactiva (n pequeño)
   y usa la calibración (n grande) para el claim del leverage; niega direccionalidad de r_{t+1}.
3. **Clustering (n=15):** Rand=1.0 parece demasiado limpio y Spectral discrepa (0.401). Se presenta como
   hipótesis exploratoria, no como prueba.
