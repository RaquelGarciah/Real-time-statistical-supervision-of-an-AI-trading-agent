# Análisis de alcance y validación de producción — EXPLORATORIO (NO canónico)

> ⚠️ **Esto es trabajo en pruebas, NO documentación válida de la tesis.** Nada de aquí debe tomarse
> como referencia para la memoria del TFG mientras no se valide y Raquel lo apruebe. No tocar
> `MANUAL.md`, `RESULTADOS_OBJETIVO.md`, los capítulos de `tesis/` ni la BITACORA canónica con estas
> cifras. Todo lo relacionado con el panel de 10 activos, el screen de leverage, el análisis de patrones
> y la "hipótesis de los dos canales" vive AQUÍ hasta nuevo aviso.
>
> Rama: `feat/quant-validation-panel`. Fecha de inicio: 2026-06-20/21.

---

## 0. Qué es esto

Una **prueba de concepto ampliada**: someter a M10 a la batería de *due-diligence* de un comité de
inversión (habilidad-vs-suerte con control de multiplicidad + mérito económico) y, a partir de ahí,
investigar de forma **inductiva** la hipótesis de alcance ("¿cuándo funciona STRATA?") cruzando la
naturaleza de los activos con los resultados. Caso SMCI + panel de 10, y candidatos de leverage fuerte
(QQQ, DIA, IWM, XLF, XLK) en generación.

Artefactos (todos reproducibles, cifras desde JSON auditado):
- `core/validation.py` — batería quant nueva (HAC, Lo, FDR, haircut HLZ, PBO/CSCV, RC/SPA, FF, borrow).
- `experiments/quant_validation_panel.py` → `outputs/experiments/quant_validation_panel.json`.
- `experiments/leverage_screen.py` → `outputs/experiments/leverage_screen.json`.
- `experiments/scope_analysis.py` → `outputs/experiments/scope_analysis.json`.
- `experiments/gen_agent_decisions.py` — genera decisiones del agente para tickers nuevos (resumible).
- `notebooks/STRATA_quant_validation.ipynb` — el informe ejecutable.
- `tests/test_validation.py` — sanidad del módulo nuevo.

Auditado por `@rigor-matematico` y `@experto-inferencia` (APROBADO CON CAMBIOS; aplicados).

---

## 1. Validación de producción de M10 (SMCI + panel de 10)

**Veredicto: NO-GO (condicional).** Prometedor en nominal, no apto para producción bajo multiplicidad.

- **SMCI nominal**: accuracy 0,552 (B&H 0,484), Sharpe 1,84, HAC t=2,09 (p=0,037), permutación por
  bloques vs B&H p=0,047, sign test 1-cola p=0,057, alpha factorial t=2,47 (β mercado −2,05).
- **Bajo multiplicidad**: FDR (BH y BY) rechaza **0/10**; haircut HLZ del mejor (best-of-10) lleva el
  Sharpe a ~0; PBO=0,38; alpha tras Bonferroni p=0,135; DSR 0,72/0,57/0,43 (6/12/24 pruebas);
  MinBTL≈2,5 años vs ~1 de OOS. White RC vs cash 0,024 (laxo) / vs B&H propio 0,069 (no rechaza).
- **Borrow despreciable** (1,84→1,81 a 500 pb): M10 no está permanentemente corto.
- El IC del Sharpe de Lo cruza 0 y HAC p=0,037 **no se contradicen** (ambos t≈1,8–2,1).

Detalle completo en `outputs/experiments/quant_validation_panel.json` y en el notebook.

---

## 2. Screen ex-ante del leverage effect (15 activos)

Medido SOLO en calibración (pre-registrable, sin tocar OOS). Dos métricas:
- **Correlación de Black** `corr(r_t, Δvol_{t+1})` — la métrica limpia. Separa nítidamente:
  - **Leverage fuerte** (índices/ETF amplios): DIA −0,112, SPY −0,110, IWM −0,102, QQQ −0,092,
    XLF −0,091, XLE −0,089, XLK −0,086.
  - **Débil/nulo** (acciones/cripto): BAC −0,066, MARA −0,059, TSLA −0,021, SMCI −0,004, NVDA +0,002,
    ROKU +0,003, MSTR +0,017, UNG +0,041.
- **Media del régimen Crisis** — ruidosa (contaminada por crisis idiosincrásicas: BAC-2008, etc.).
  Se descarta como criterio principal a favor de la correlación de Black.

Los 5 candidatos (QQQ, DIA, IWM, XLF, XLK) son leverage fuerte → buenos para el grupo "debería
funcionar". Detalle en `outputs/experiments/leverage_screen.json`.

---

## 3. Análisis inductivo de patrones (naturaleza → resultado, n=10)

> Correlaciones con n=10: **ILUSTRATIVAS, no significativas**. Marcan dirección, no prueban.

| Relación | corr |
|---|---|
| leverage (Black) ~ accuracy direccional del régimen | **−0,64** |
| leverage (Black) ~ valor añadido de STRATA (M8−M5) | **−0,55** |
| sesgo corto del agente ~ accuracy M10 | **+0,75** |
| volatilidad OOS ~ accuracy M10 | **+0,67** |
| accuracy direccional del régimen ~ accuracy M10 | −0,19 |
| leverage (Black) ~ accuracy M10 | +0,33 |

Grupos por leverage de Black (fuerte = corr < −0,05):
- **Fuerte** [SPY, BAC, XLE, MARA]: régimen direccional 0,534 · ΔM8(STRATA) **+0,055** · M10 0,490.
- **Débil** [NVDA, TSLA, UNG, MSTR, SMCI, ROKU]: régimen direccional 0,493 · ΔM8 **+0,025** · M10 0,519.

Detalle en `outputs/experiments/scope_analysis.json`.

### Hipótesis inducida: STRATA tiene DOS canales de valor distintos

1. **Canal régimen (RAM/M8) — el de la tesis.** El régimen del HMM es **direccional donde el leverage
   effect es fuerte** (índices/ETF amplios): leverage ~ régimen-direccional = −0,64, y STRATA (M8) añade
   ~2× más accuracy sobre el agente ahí (ΔM8 0,055 vs 0,025). En leverage débil/inverso (acciones,
   commodities, cripto) el régimen no es direccional y este canal no aporta.
2. **Canal meta-aprendiz (M10) — distinto.** La accuracy de M10 la manda el **sesgo corto del agente en
   nombres muy volátiles y bajistas** (sesgo-corto ~ M10 = +0,75; vol ~ M10 = +0,67), NO el régimen
   (régimen ~ M10 = −0,19). Por eso SMCI (leverage nulo) lidera M10: explota que el agente estaba corto
   en una acción que se desplomó, no habilidad de régimen.

**Confundido a vigilar**: en este panel los activos de mucha Crisis-OOS son justo los de leverage débil
(SMCI 70 %, UNG 51 %), así que "fracción de Crisis" sale correlacionada en negativo con el régimen
direccional; manda la *calidad* del leverage, no la *cantidad* de crisis.

---

## 4. Estado y bloqueos

- **Agente**: submódulo `agent/ai_hedge_fund/` presente; `OPENROUTER_API_KEY` actualizada y válida
  (~19 s/decisión). El tier free aplica **rate limits (429)** con backoff de 60 s → el batch va lento.
- **En generación** (`experiments/gen_agent_decisions.py`, resumible): QQQ, DIA, IWM, XLF, XLK sobre el
  calendario OOS de SPY (~401 días × 5). Al terminar: re-correr `quant_validation_panel.py` y
  `scope_analysis.py` sobre los 15 para **confirmar el canal 1** (predicción: régimen direccional alto y
  ΔM8 alto en los índices/ETF).

---

## 5. Preguntas abiertas / próximos pasos

1. Confirmar el canal 1 con los 5 índices/ETF (¿régimen direccional > 0,5 y ΔM8 > 0 robusto?).
2. ¿El canal 1 se ve más en **tramos de Crisis** del OOS? (condicionar por régimen).
3. ¿Conviene reescribir la sección de alcance de la tesis con los dos canales? **Solo si se valida** —
   por ahora se queda aquí.
4. Pre-registro: ver `docs/preregistro_quant_validation_EXPLORATORIO.md` (borrador).
