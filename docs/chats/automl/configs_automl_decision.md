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

## ESTADO CANÓNICO AL CIERRE
- **Notebook único canónico:** `notebooks/STRATA_marco_practico.ipynb` (panel de 10). Decisión #18.
- **Panel (10):** SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG · **apéndice (5):** MSTR, NVDA, BAC, TSLA, IWM.
- **Outputs nuevos:** `mechanism_panel.json`, `panel_robustness.json`, `automl_net_returns.json`,
  `detector_analysis_{SPY,XLE,MARA}.json`, `calib_window_panel.json`. Log de revisión:
  `docs/chats/automl/revision_marco_practico.md`.
- **Punto más blando (declarado):** UNG en el cuerpo (el agente no pierde ahí; encuadrado como caso ML).
