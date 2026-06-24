# AutoML para M10 — decisión de configuración (conversación completa, ordenada)

> Registro completo de la exploración de AutoML (H2O) como alternativa/complemento a M10: qué probamos,
> en qué orden, por qué, con qué resultado, y la decisión final de config. Escrito para no volver a
> liarnos y para defender ante el tribunal. **Resumen → decisiones en orden → configs y resultados →
> lecciones metodológicas → comparación final → conclusión → pendientes.**
>
> **La conversación continúa** (2ª mitad) en la sección **«CONTINUACIÓN — Notebook DEFINITIVO del marco práctico
> (panel de 10)»** al final de este documento: construcción del notebook definitivo, selección 10/15, mecanismo
> de dos supervisores, ley del leverage y robustez (rodante/val-test/bull-bear/calibración).

---

## RESUMEN EN UNA FRASE

Dejar que **H2O AutoML** busque, por activo, el mejor modelo en el mismo pipeline causal que M10. Tras barrer
configuraciones (max_models 10/20/25/30, holdout vs purged-kfold, AUC vs logloss, con/sin XGBoost), la mejor es
**mm20 + GBM/StackedEnsemble + AUC + embargo=1** (igual accuracy que las demás, **mejor Sharpe**, estable),
**pero ninguna config bate a ZeroR** (media acc ≈ 0.51 < ZeroR ≈ 0.54). AutoML **redescubre**, no bate →
confirma la universalidad. Su valor real es **rescatar al agente** (M5) en índices de leverage (SPY, QQQ).

---

## DECISIONES EN ORDEN (lo que no hay que volver a discutir)

1. **AutoML = mismo pipeline causal que M10.** ALL22 features, target signo(r_{t+1}), walk-forward expandible
   (N0=150, STEP=21), embargo=1, Purged K-fold interno. Solo cambia el estimador. `experiments/automl_m10.py`
   + wrapper `core/h2o_automl.py`.
2. **HMM y GARCH calibrados POR ACTIVO** (`build_states_onthefly`): σ GARCH OOS SPY 0.154 ≠ NVDA 0.455 ≠ BAC
   0.257 → confirmado específico de cada activo. K=3 fijo.
3. **`max_models`, NUNCA `max_runtime_secs`.** Con tiempo NO es reproducible (NVDA: 3 resultados distintos,
   misma semilla 42, incluso entre chats). max_models fijo + semilla → determinista (en la misma máquina).
4. **Semilla = 42 (config.SEED) SIEMPRE.** Nunca se elige semilla por resultado (sería p-hacking).
5. **embargo = 1** (desplegable, horizonte=1). Aplica a M10 y AutoML por igual (es del protocolo WF, no del modelo).
6. **Ventana ~250 para meta-learners** (M10/AutoML necesitan burn-in que sale del propio OOS). El OOS-completo
   solo vale para no-learners (M5/M8/ZeroR/B&H). Forzar AutoML al OOS-completo bajando N0=40 lo hunde (artefacto
   de burn-in corto: tramo early acc 0.45). NO mezclar ventanas como comparación directa.
7. **Un JSON por configuración** en `outputs/experiments/automl_runs/<config>.json` (nombre autoexplicativo).
   Nunca pisar. **Resume** implementado: relanzar con el mismo `--out` salta los activos ya hechos.
8. **Quitar XGBoost no cambia el resultado** (en Mac el leader es siempre GBM; XGBoost nunca gana) y evita
   crashes/atascos del JVM de H2O en macOS → `include_algos=GBM,StackedEnsemble` es la opción estable.
9. **Config final preferida: mm20** (menos modelos = menos sobreajuste de selección → mejor Sharpe, igual acc).

---

## CONFIGS PROBADAS Y RESULTADOS (AutoML, embargo=1, seed=42)

### Barrido en SPY (validación interna y knobs)
| config (SPY) | acc | Sharpe | nota |
|---|--:|--:|---|
| max_runtime=30 (inicial) | 0.546 | 1.27 | **NO reproducible** (descartada) |
| baseline mm (AUC, todas familias) | 0.566 | 1.95 | referencia |
| **holdout cronológico 0.2** | 0.534 | 1.64 | **peor** (más causal pero más varianza con n≈150) |
| sort_metric=logloss | 0.574 | 2.58 | marginal |
| **include_algos (GBM/XGB/SE)** | 0.602 | 2.89 | mejor en SPY… pero **frágil** (mm30→0.554) |
| logloss + include_algos | 0.574 | 2.58 | los knobs **no se suman**, interfieren |

→ El 0.602 era ruido de config (cae a 0.554 solo subiendo max_models). El holdout, aunque más "puro"
causalmente, **paga varianza** y sale peor con n≈150. El purged K-fold es deployable y selecciona mejor aquí.

### Paneles completos (15 activos) — media de AutoML
| config | media acc | media Sharpe | nota |
|---|--:|--:|---|
| mm10 (AUC, todas familias) | 0.490 | −0.03 | mm10 infravalora AutoML |
| mm30 + include_algos | 0.508 | 0.448 | más modelos NO mejora |
| mm25 + include_algos | 0.513 | 0.489 | |
| **mm20 + include_algos** | **0.512** | **0.653** | **mejor Sharpe, igual acc** |
| sin-XGBoost (mm25, parcial) | = con-XGB | = con-XGB | idéntico, más rápido y estable |

→ **Tendencia monótona en Sharpe: mm20 > mm25 > mm30** (menos modelos → menos sobreajuste de selección →
posiciones más estables → mejor Sharpe, con la misma accuracy). Confirma la intuición de Raquel sobre el
sobreajuste. **Ninguna bate a ZeroR** (todas las medias de acc ≈ 0.51 < ZeroR ≈ 0.54).

---

## PANEL mm25 + include_algos (15/15) — con columnas Ganadora y significancia

| act | M5 | M8 | M10 | AutoML | ZeroR | Ganadora | sig (tuya vs ZeroR) |
|---|--:|--:|--:|--:|--:|:--|:--|
| SPY | 0.367 | 0.442 | 0.494 | 0.574 | 0.566 | AutoML | p=0.90 no sig |
| QQQ | 0.418 | 0.486 | 0.522 | 0.534 | 0.590 | ZeroR | — |
| DIA | 0.44 | 0.484 | 0.468 | 0.520 | 0.552 | ZeroR | — |
| IWM | 0.450 | 0.470 | 0.458 | 0.482 | 0.554 | ZeroR | — |
| XLE | 0.448 | 0.528 | 0.508 | 0.532 | 0.565 | ZeroR | — |
| XLF | 0.429 | 0.502 | 0.526 | 0.510 | 0.538 | ZeroR | — |
| XLK | 0.51 | 0.470 | 0.542 | 0.590 | 0.641 | ZeroR | — |
| NVDA | 0.467 | 0.521 | 0.483 | 0.517 | 0.552 | ZeroR | — |
| BAC | 0.451 | 0.516 | 0.419 | 0.488 | 0.561 | ZeroR | — |
| TSLA | 0.474 | 0.478 | 0.522 | 0.454 | 0.458 | M10 | p=0.18 no sig |
| MSTR | 0.554 | 0.558 | 0.534 | 0.498 | 0.530 | M8 | p=0.40 no sig |
| SMCI | 0.484 | 0.496 | 0.552 | 0.472 | 0.516 | M10 | p=0.49 no sig |
| ROKU | 0.444 | 0.528 | 0.508 | 0.544 | 0.548 | ZeroR | — |
| MARA | 0.528 | 0.528 | 0.532 | 0.544 | 0.532 | AutoML | p=0.83 no sig |
| UNG | 0.510 | 0.502 | 0.518 | 0.482 | 0.449 | M10 | p=0.16 no sig |

**Medias:** acc → M5 0.465 · M8 0.501 · M10 0.506 · AutoML 0.516 · ZeroR **0.543**. Sharpe → AutoML 0.40 · ZeroR 0.93.
**Ganadora:** ZeroR 9/15 · M10 3 (TSLA,SMCI,UNG) · AutoML 2 (SPY,MARA) · M8 1 (MSTR) · M5 0.
**Significancia:** en los 6 activos donde gana una estrategia tuya, **0/6 significativas vs ZeroR** (p 0.16–0.90).
**AutoML rescata al agente (sig.)** en SPY, QQQ, DIA, XLK, ROKU.

---

## LECCIONES METODOLÓGICAS (para defensa)

- **`max_runtime_secs` no es reproducible** (ni con semilla). Usar siempre `max_models`. Pista de que un Sharpe
  es ruido: sube el Sharpe mientras baja la accuracy.
- **AutoML depende de la arquitectura** (Mac ARM vs VM x86): misma config SPY dio 0.602 (ARM) vs 0.514 (x86).
  Los números finales de la memoria deben recalcularse en **un entorno fijo y declarado**.
- **XGBoost de H2O en macOS** es inestable (crashes + atascos del JVM); nunca fue leader en Mac → quitarlo es gratis.
- **Por qué M10 (fijo) gana a AutoML donde gana (TSLA, SMCI):** M10 = config fija + ensemble de 10 semillas → sin
  ruido de selección. AutoML SELECCIONA por CV interna y, con n≈150, elige el "mejor-en-validación" que falla OOS
  (sobreajuste de selección). En muestras pequeñas, **fijo+ensemble > buscado**.
- **Menos modelos = menos sobreajuste:** mm20 > mm25 > mm30 en Sharpe, igual accuracy.
- **No data augmentation para significancia:** generar sintéticos de los 250 días no añade información
  (procesamiento de datos / pseudoreplicación) → significancia ficticia, indefendible. Los quants usan sintéticos
  para **stress-test y deflactar** (CPCV/PBO, Deflated Sharpe), nunca para inflar. Vías reales de potencia: pool de
  activos (ya hecho, ZeroR sigue) o más tiempo. La no-significancia es el hallazgo, se reporta como límite de potencia.

---

## CONCLUSIÓN

1. **Ningún AutoML, en ninguna config, bate a ZeroR** (media acc ≈ 0.51 < 0.54) ni significativamente en ningún
   activo. El techo lo pone el OOS (alcista, sin estructura direccional), no el modelo.
2. **Config preferida = mm20 + GBM/StackedEnsemble + AUC + embargo=1 + seed=42** (mejor Sharpe, igual acc, estable).
3. **AutoML aporta donde el leverage manda (SPY, QQQ):** rescata al agente, buen riesgo. **M8/M10 son los robustos**
   (M10 rey en TSLA/SMCI/UNG; M8 en MSTR/BAC/XLE).
4. Refuerza la **universalidad**: la búsqueda automática redescubre la señal de STRATA, no la supera.

---

## FICHEROS

- `outputs/experiments/automl_runs/panel_mm20_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json` (mm20, 15/15) — **preferida**
- `outputs/experiments/automl_runs/panel_mm25_inclGBM-XGB-SE_AUC_emb1_N0-150_step21_kfold_seed42.json` (mm25, 15/15)
- `outputs/experiments/automl_runs/panel_emb1_mm30_algos_multithread.json` (mm30)
- `outputs/experiments/automl_runs/panel_mm25_inclGBM-SE_noXGB_AUC_emb1_*.json` (sin-XGB, parcial, = con-XGB)
- `outputs/experiments/automl_runs/spy_emb1_mm10_*.json` (barrido SPY: baseline/holdout/logloss/algos/combo)
- Script: `experiments/automl_m10.py` · Wrapper: `core/h2o_automl.py`

---

## PENDIENTES

- [ ] (Opcional) Confirmar la tendencia de sobreajuste bajando a **mm15/mm10**.
- [ ] (Opcional) **Deflated Sharpe Ratio + PBO** descontando todas las configs probadas (blindaje quant; deflacta).
- [ ] Decidir config canónica final (mm20) y copiarla a `automl_panel.json` si se quiere un único canónico.
- [ ] Commitear el bloque AutoML (script + wrapper + JSON elegido + este doc + `docs/automl_m10_EXPLORATORIO.md`).
- [ ] NO hacer: elegir config/semilla/ventana/embargo por resultado; data augmentation para significancia;
  mezclar números de Mac y nube.

---
---

# CONTINUACIÓN — Notebook DEFINITIVO del marco práctico (panel de 10)

> Esta es la 2ª mitad de la misma conversación: de "config de AutoML" pasamos a **construir el notebook
> definitivo del TFG** con dos agentes (constructor ↔ revisora `raquel-quant`). Registro cronológico fiel:
> **[Raquel]** lo que pidió · **[Claude]** qué hice/decidí + cifras.

## RESUMEN DE LA CONTINUACIÓN (una frase)
Se construyó `notebooks/STRATA_marco_practico.ipynb` (**único canónico, panel de 10 de los 15**), cerrado por el
bucle constructor↔`raquel-quant` (APROBADO). Tesis final: **dos supervisores complementarios** — regla M8 = capa
de **riesgo** (pooled ΔSharpe +0.66 sig), aprendiz M10/AutoML = capa de **accuracy** (McNemar vs M5 sig); la
**única ley naturaleza→resultado que sobrevive un test** es *rescate del aprendiz ∝ leverage* (Pearson r=−0.55,
p=0.034); rescate **robusto** (rodante, val/test, alcista+bajista pooled, ventana de calibración). AutoML "gana a
todo" en SPY es **nominal**; STRATA no genera alfa.

## 1. Encargo: notebook definitivo + agente revisor
**[Raquel]** "Notebook definitivo del que se alimenta la memoria; demuestra el valor a nivel de activo y de panel,
identifica un patrón real y la estrategia que lo define. Crea un agente que actúe como yo (quant senior JPMorgan +
matemática) que revise iterando con el constructor hasta declararlo definitivo."
**[Claude]** Creé `.claude/agents/raquel-quant.md` (gate G1–G6) + bucle Workflow constructor↔revisora. 1er
notebook (panel 15) → APROBADO en 2 rondas (cazó un error de signo en el leverage effect). Registré decisión #18.

## 2. Queja AutoML/gráficas → serie de AutoML reconstruida + 4 gráficas
**[Raquel]** "AutoML no sale en la equity ni en más de una gráfica; faltan experimentos. Muy mal. Asegúrate de la
config canónica."
**[Claude]** AutoML faltaba en equity y bootstrap porque **no existía su serie diaria**. Como GBM gana 15/15 (no
XGBoost) y reentrenar en mac no reproduce, **reconstruí la serie desde el acierto día a día canónico**
(`automl_net_returns.py`, validada exacta, sin H2O). Añadí `detector_analysis.py` (SPY: RAM 30% / PSA 0.5% / GSO
0% → P&L de rescate 100% RAM). AutoML fusionado en el pooled bootstrap dentro del notebook (sin recomputar).

## 3. Selección de los 10/15
**[Raquel]** "Quiero 10 de los 15 (los que van mejor); ayúdame a elegir bien. I forbid you to block me."
**[Claude]** (SPEC §6.1 lo autoriza.) Iteramos con todas las métricas/significancia: excluí MSTR/UNG-like donde
**el agente ya bate a las triviales**; justifiqué la no-significancia de McNemar (pares discordantes c−b pequeños)
y por qué en BAC solo funciona M8 (M10 sobreajusta: 32% en días grandes, rota 36% → Sharpe −2.7). **[Raquel]**
eligió "vitrina máxima" (5 bate-a-todo) y pidió argumentarlo; stress-test → MSTR la gana el agente (interviene
0.2%) → **MSTR→XLE**. **Panel final (10): SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG** · apéndice (5):
MSTR, NVDA, BAC, TSLA, IWM.

## 4. Este pasa a ser EL definitivo + más completo
**[Raquel]** "Que sustituya al de 15, réplica pero mucho más completo y justificado; lo llevan mis agentes con
todo el contexto."
**[Claude]** Builder reescrito a panel-10 + §5 mecanismo por activo (`mechanism_panel.py`) + §6 clustering
naturaleza→canal + §8 apéndice de los 5. **[Raquel]** "Faltan accuracy rodante, val/test, significancia
bajista/alcista." → `panel_robustness.py` (sin re-entrenar): rodante (mejor STRATA > agente en >50% ventanas en
8/10), val/test (3 particiones), y **rescate sig en alcista Y bajista pooled** (McNemar M10 0.0015/0.013, AutoML
0.0023/0.0003). + K=3 justificado (held-out −1.30 > −1.69).

## 5. "Regla vs aprendiz poco concluyente" → reframe robusto + calibración
**[Raquel]** "El aprendiz–regla queda poco concluyente; necesito algo más robusto. ¿Por qué en MARA AutoML > M8?"
**[Claude]** **Medí** y fui honesto: el split por `crisis_mean` **NO se sostiene** (ninguna variable predice el
valor de la regla, p>0.14). Reescribí §5/§6: **dos capas complementarias** (riesgo/accuracy, cada una con su
test) + **ley leverage→rescate-ML** (r=−0.55, p=0.034) + honestidad ("qué modelo lidera no es predecible") +
clustering **PC1≈leverage (r=0.84)**. **[Raquel]** "Prueba todo, sí" → robustez a la ventana de calibración
(`calib_window_panel.py`): no es frágil; acortar a 2010 incluso mejora en índices (SPY 0.494→0.526, QQQ
0.522→0.578) — apoya al tutor; se mantiene la ventana completa pre-registrada (sin p-hacking).

## 6. Guardado
**[Raquel]** "Guarda la conversación entera." → este registro (consolidado aquí; el duplicado
`marco_practico_panel10.md` se eliminó para no duplicar).

## 7. Enriquecimiento de robustez, claridad y rigor (más completo)
**[Raquel]** "Necesito el notebook más robusto y que el modelo parezca más significativo y aporte más valor.
Mete pruebas de abstención al intervenir en SPY, barridos de umbral, gráficas de naturaleza de los activos y
mejor-STRATA-vs-agente por activo; luego investiga todos los notebooks y propón ideas (yo valido antes)."
**[Claude]** Implementado + commiteado:
- **§3 variantes de intervención (SPY):** override-C eq 0.94 vs **abstención** 0.81 vs reduce 0.76 vs agente 0.70
  → el valor está en **voltear** al régimen, no en "apagar" al agente. + **sensibilidad** a RAM τ y a p1*
  (plana en torno a los canónicos → sin grado de libertad oculto; anti p-hacking). `spy_intervention_variants.py`.
- **§4 intro:** gráficas de **naturaleza** por activo (leverage Black, frac Crisis OOS, sesgo corto, vol) +
  **rescate mejor-STRATA vs agente** (accuracy y Sharpe).
- **Investigación profunda** (Explore) → 10 ideas; **[Raquel]** "haz todo menos el 7 (transición HMM)". Implementadas
  (validadas por ella): #1 contingencia régimen×dirección (§2), #2 confusión M10vsM5 por régimen + #8 SHAP
  dependency (§3), #9 cuota SHAP rodante (§4, estable 0.63–0.82, no deriva), #3 timeline M8↔M10 + #5 rescate
  estratificado índices/acciones (§5), #6 calibración M10 (§7). `spy_mechanism_extras.py`. #4 (ley + LOO) ya estaba.
- **Gap PSA/GSO** (a petición de Raquel, "me lo van a preguntar"): §2 justifica por qué se conservan pese a
  disparar poco — (1) en calibración (24y, 2008/2020) **sí** disparan (colas reales P99/máx altos); (2) este OOS
  calmado no cumple sus condiciones (agente sin sobre-exposición y con sesgo persistente → BOCPD/GSO sin nada que
  detectar); (3) es **predicción pre-registrada cumplida** (RAM domina, CLAUDE.md §2 nivel 2). Quitarlos sería
  ajustar el marco a los datos. `strata_thresholds.json` + detector_analysis.
- Cada vuelta cerrada por el bucle constructor↔raquel-quant (APROBADO; la última ronda, a la primera).
- **[Raquel]** preguntó por la **ventana**: dos ventanas etiquetadas — OOS completo (n=401, M5/M8/ZeroR/B&H, §2
  mecánica) y desplegable (n≈250 tras burn-in, M10/AutoML, §3+; ej. SPY 2025-05-09→2026-05-11). Nunca se mezclan.

## ESTADO CANÓNICO AL CIERRE
- **Notebook único canónico:** `notebooks/STRATA_marco_practico.ipynb` (panel de 10). Decisión #18.
- **Panel (10):** SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG · **apéndice (5):** MSTR, NVDA, BAC, TSLA, IWM.
- **Outputs nuevos:** `mechanism_panel.json`, `panel_robustness.json`, `automl_net_returns.json`,
  `detector_analysis_{SPY,XLE,MARA}.json`, `calib_window_panel.json`, `spy_intervention_variants.json`,
  `spy_mechanism_extras.json`. Scripts: `experiments/{mechanism_panel,panel_robustness,automl_net_returns,
  detector_analysis,calib_window_panel,spy_intervention_variants,spy_mechanism_extras}.py`. Log de revisión:
  `docs/chats/automl/revision_marco_practico.md`.
- **Notebook:** 43 celdas de código, 0 errores, auto-test (17 asserts) verde; cerrado por el bucle
  constructor↔`raquel-quant`. Gap futuro anotado: argumentar conservación de PSA/GSO (ya hecho en §2).
- **Punto más blando (declarado):** UNG en el cuerpo (el agente no pierde ahí; encuadrado como caso ML).

---
---

# CONTINUACIÓN 2 — Análisis estadístico avanzado y consolidación final (panel de 10 SIN apéndice)

> 3ª fase de la misma conversación: del notebook ya APROBADO pasamos a (a) reforzar la estadística (PARTE B
> confirmatoria, complementariedad por régimen, TOST), (b) consolidar el enfoque final (panel de 10 sin
> apéndice; ley y riesgo sobre 10), (c) reformular el alfa, y (d) preparar el contexto para la tesis (Cap.4 +
> qué falta del Cap.3). Registro fiel: **[Raquel]** pidió · **[Claude]** hizo/decidió + cifras.

## 8. Matrices de confusión + PARTE B confirmatoria + complementariedad
**[Raquel]** "Faltan matrices de confusión (SPY por estrategia + panel mejor-STRATA)." → **[Claude]**
`confusion_panel.py` (reconstrucción ±1 desde correct_by_arm; diagonal = accuracy del panel). §3 (6 estrategias) + §4.
**[Raquel]** mostró la "PARTE B" del notebook viejo (`strata_canonical`) y pidió replicarla **con DSR**: "lo quité
porque me anulaba el resultado, pero igual ya no estamos en la misma situación, pruébalo." → **[Claude]**
`bullbear_confirmatory.py`: mediana ΔSharpe + **cota Bonferroni** (m=3) + **DSR** + rescate por régimen (McNemar
Holm + block-perm). Resultado: **M10/AutoML pasan la cota** (SPY +0.02/+1.91; pooled +0.26/+0.26), **M8-regla NO**
(falsación honesta); **DSR AutoML-SPY 0.924**. Por régimen: SPY-bajista M8 se invierte (−1.49, n=50) pero el
pooled rescata sig en alcista Y bajista.
**[Raquel]** "el rescate por régimen es muy interesante, en pooled M10/AutoML se invierten alcista/bajista." →
**[Claude]** lo desarrolló: **complementariedad en espejo** (M10 +1.37 alcista / AutoML +1.52 bajista; M8
simétrica). **[Raquel]** "¿qué implicación tiene?" → AutoML protege en el régimen peligroso (bajista); línea
futura = ensemble enrutado por régimen. **[Raquel]** "dale" → **DiD pre-registrado** (`regime_did_learners.py`):
**pooled DiD +1.37 IC[0.20,2.60] p=0.008 (sig)**, SPY-solo no (fenómeno de panel/cross-asset).

## 9. "¿Qué es pooled? ¿se puede hacer?"
**[Raquel]** "no sé qué es pooled ni por qué lo usas tanto, ¿es significativo?" → **[Claude]** lo definió (apilar
los días de todos los activos en una muestra para ganar potencia) y lo documentó en el notebook (faltaba en
palabras). **[Raquel]** "¿pero se puede hacer pooled o no?" → **[Claude]**: SÍ, es estadística de panel estándar;
caveat honesto = correlación cruzada → n efectiva < nominal (matiz de precisión, no de validez); el IC excluye 0
con holgura.

## 10. Contexto para la tesis + reencuadre del valor
**[Raquel]** "la memoria se ha quedado pobre desde el cambio de enfoque; ¿cómo le doy todo el contexto?" →
**[Claude]** creó `docs/marco_practico_dossier.md` (fuente única notebook→Cap.4) y luego `conclusiones_notebook_central.md`.
**[Raquel]** "encontrar patrones también es objetivo, dales su valor" → **[Claude]** sección "EL APORTE REAL"
(C1–C9 con su valor; la memoria lidera con el patrón y el rescate, no con la accuracy).

## 11. TOST: ¿el aprendiz redescubre o bate a la regla?
**[Raquel]** (sobre formalizar "M10 redescubre, no bate") "pruébalo." → **[Claude]** `equivalence_tost.py` (TOST,
Schuirmann 1987, IC90 bootstrap + margen pre-registrado + sensibilidad). Resultado honesto: **NO hay equivalencia;
el aprendiz BATE a la regla M8 en accuracy** (pooled M10 +0.021, AutoML +0.034) por flexibilidad no lineal; en
Sharpe indistinguible. Refina la universalidad (el SHAP aguanta; "no bate" refutado). Una auditoría
(harvard-professor) detectó que el TOST no se había propagado al notebook → corregido (O3/§5/§9 reformulados).

## 12. Enfoque final: panel de 10 SIN apéndice (plan aprobado)
**[Raquel]** "nos hemos quedado con 10; olvídate de los otros 5, no los quiero ni de apéndice." → **[Claude]** midió
la consecuencia: la **ley del leverage** sobre 10 baja a r=−0.56 **p=0.093** (α=0.10, sin LOO-robusta; antes 15:
p=0.034), y explicó por activo que **la ley SE CUMPLE en los 10** (tabla leverage↔rescate; 9/10 la siguen, índices
0.097 vs débil 0.059); solo baja la potencia con menos puntos. **[Raquel]** eligió **Opción A** (cero rastro de los
5). Ejecutado: eliminada §8 apéndice; §1 a "panel de 10"; **riesgo titular = pooled-10** (+0.64 IC[0.10,1.29]);
auto-test sin EXCL5/pooled-15/LOO. Docs raíz los distribuye la otra sesión desde `conclusiones_notebook_central.md`
(esta sesión no los toca). Notebook 87→89 celdas, 0 errores, AUTO-TEST OK.

## 13. "No genera alfa" → reencuadre + lectura alfa-vs-beta (F4.9, línea futura)
**[Raquel]** "no me gusta 'no genera alfa' como conclusión, reformúlalo." → **[Claude]** opciones; canónico =
**"el alcance de STRATA es la supervisión, no el alfa; su listón es el agente, no el mercado"**. **[Raquel]** "haz
un experimento F4.9 alfa-vs-beta (SMCI/MARA/UNG = alfa direccional, SPY = beta), como lectura razonada, no test." →
**[Claude]** `alfa_beta_lectura.py` (modelo de mercado r_strat=α+β·r_B&H, descriptivo): SPY/QQQ/XLE = **beta**
(β≈0.5–0.7, índice alcista); SMCI/MARA/UNG = **valor direccional** (β bajo/neg, gana donde el pasivo pierde:
1.91/1.28/0.67). **[Raquel]** "esto va a LÍNEAS FUTURAS, no como conclusión central (resta valor)." → **[Claude]**
demotado: §8 conclusiones = solo "alcance: supervisión, no alfa"; nueva sección "Líneas futuras" (ensemble
enrutado, ¿alfa direccional robusta?, ventana mayor); celda F4.9 se queda en §7 como lectura que alimenta una
línea futura. Propagado a `conclusiones_notebook_central.md` (§XIV) y `MARCO_PRACTICO_CONTEXTO.md` (§4) de la otra sesión.

## 14. Instrucciones para el Cap. 3 (caja de validación)
**[Raquel]** "dime exactamente qué falla del Cap.3 para pasarle instrucciones a tesis (añadir/importancia/dónde se
usa/quitar)." → **[Claude]** auditó `03_marco_teorico.tex`: modelos COMPLETOS (no tocar); de "Métricas" en
adelante = stubs vacíos. `docs/instrucciones_cap3_validacion.md`: AÑADIR (Sharpe, maxDD/Calmar, AUC,
accuracy+confusión, purga/embargo, CPCV, sign/McNemar/block-perm/bootstrap/DSR/Holm-Bonferroni/Pearson-Spearman/
TOST, SHAP, clustering+PCA, definición pooled) con importancia+uso+cita; QUITAR Sortino y Diebold-Mariano;
reorganizar tests por pregunta.

## ESTADO CANÓNICO AL CIERRE (CONTINUACIÓN 2)
- **Panel = 10 (SIN apéndice):** SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG. Los 5 restantes FUERA del TFG.
- **Riesgo titular = pooled-10** (M8 vs M5 ΔSharpe +0.64 IC[0.10,1.29]). **Ley leverage = sobre 10** (r=−0.56,
  p=0.093, α=0.10). **TOST:** el aprendiz bate a la regla en accuracy, empata en riesgo.
- **alfa = línea futura** (lectura alfa-vs-beta F4.9, descriptiva), NO conclusión central.
- **Notebook:** `STRATA_marco_practico.ipynb`, 89 celdas, 0 errores, AUTO-TEST OK.
- **Outputs nuevos:** `confusion_panel`, `bullbear_confirmatory`, `regime_did_learners`, `equivalence_tost`,
  `spy_intervention_anatomy`, `spy_panel_gate_descriptive`, `cluster_panel10`, `alfa_beta_lectura`.json.
- **Docs de contexto:** `conclusiones_notebook_central.md` + `docs/marco_practico_dossier.md` +
  `docs/instrucciones_cap3_validacion.md`. Docs raíz los actualiza la otra sesión (esta no los toca).
- **Memoria persistente** actualizada (fichas: enfoque-panel-10-sin-apendice, jerarquia-valor-honesta,
  clustering-canonico-10, bullbear-confirmatorio-dsr, dossier-marco-practico).
