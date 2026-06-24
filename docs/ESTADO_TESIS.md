# ESTADO DE LA TESIS — arranque para una sesión nueva

> Lee esto + `CLAUDE.md` + `MARCO_PRACTICO_CONTEXTO.md` y estarás al día. Rama de trabajo: `feat/quant-validation-panel`.

## Hecho (commiteado y con todas las gates en verde)
- **Cap. 2** (`tesis/chapters/02_estado_arte.tex`): añadida la descripción de cómo funciona el agente
  (5 personalidades Buffett/Wood/Druckenmiller/Burry/Ackman + risk manager + portfolio manager LLM que integra);
  cifras alineadas a SPY/panel-10 (ya NO SMCI).
- **Cap. 3** (`tesis/chapters/03_marco_teorico.tex`): la **caja de validación** completa (métricas, validación
  causal, contraste por pregunta). Modelos (HMM/GARCH/BOCPD/detectores/aprendizaje) ya estaban y NO se tocan.
- **Cap. 4** (`tesis/chapters/04_marco_practico.tex` + `cap4_parts/s1_spy,s2_panel,s3_patrones,s4_limites.tex`):
  reescrito al canónico de 10 activos.
- **Cap. 5** (`tesis/chapters/05_conclusiones.tex`): conclusiones, hipótesis resuelta nivel por nivel.

## Pendiente
- **Figuras del cap. 4** (15): las trae la sesión del notebook. Handoff en `tesis/figuras/REQUERIDAS_cap4.md`.
  Hasta que existan, `main.tex` no compila entero.
- **Q&A de defensa** → `docs/questions_and_answers.md` (sin empezar; `defensa-tutor` ya tiene la munición).

## Cifras canónicas (NO reintroducir las viejas)
- **Pooled-10 riesgo**: fuente = `outputs/experiments/bullbear_confirmatory.json` bloque **POOLED10**
  (M8 +0,60 [0,05,1,22] NO pasa Bonferroni · M10 +1,12 · AutoML +1,08). El +0,64/+0,93/+0,97 era **pooled-15
  obsoleto** (`decision_automl_prep.json /pooled`); NO usarlo.
- **Ley del leverage**: `outputs/experiments/leverage_law_panel10.json` (r=−0,56, p=0,093, α=0,10).
  El `leverage_law_panel.json` (sin 10) lleva metadatos de 15.
- SPY M5 acc 0,366 / equity 0,70. McNemar vs M5: AutoML 0,0002 / M10 0,0074 / M8 0,051. TOST M10 +0,021 /
  AutoML +0,034. SHAP STRATA 0,66 (>0,5 en 10/10). DiD +1,37 p=0,008. Clustering Rand=1,0, PC1≈leverage 0,84.

## Reglas duras (de CLAUDE.md y CONTEXTO)
- Panel de **10 activos**, ex ante, **SIN apéndice**. PROHIBIDO "quince"/"10 de N". SMCI es uno de los 10.
- **Alfa** solo como límite de alcance + línea futura; nunca bandera negativa.
- **override** al comparar variantes / **intervención** para el mecanismo. Sin costes en la narrativa.
- Toda cifra trazada a su JSON. No tocar el notebook (lo lleva la otra sesión).
- Protocolo de redacción: coordinador-redaccion (estructura, valida Raquel) → redactor-tesis → gates
  (rigor-matematico, experto-citas, detector-plagio, estilo-raquel, detector-ia; +harvard/defensa donde aplique).
- Commits atómicos + push a la rama feature. Nunca `.env`.
