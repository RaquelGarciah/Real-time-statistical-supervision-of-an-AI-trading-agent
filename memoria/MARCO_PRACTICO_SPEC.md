# MARCO PRÁCTICO — especificación para los agentes constructor y revisor

> Este documento define **qué tiene que mostrar el notebook del marco práctico, cómo y con qué nivel de rigor**. Lo leen los dos agentes (constructor + `raquel-quant` revisor) al arrancar cada ronda. Si algo no está aquí o en las fuentes que enlaza (`CLAUDE.md`, `DECISIONES_ESENCIALES.md`, `RESULTADOS_OBJETIVO.md`, `BITACORA.md`), no entra al notebook.

---

## 0. Objetivos del capítulo (tres capas)

Las dos primeras capas definen **qué demostramos**. La tercera define **cómo se hace todo** (calidad y presentación). Las tres están siempre presentes en el notebook.

### 0.A. Tesis primarias

> **T1.** Un agente LLM de trading sin supervisión es **direccionalmente perdedor** sobre el OOS.
>
> **T2.** STRATA **rescata al agente sistemáticamente** y aporta valor medible: tanto en acierto direccional como en control de riesgo. Cuando hay alguna excepción donde STRATA no rescata, se documenta y se argumenta el mecanismo, no se esconde.
>
> **T3.** En los activos cuya naturaleza lo permite, las estrategias derivadas de STRATA (M8, M10 o AutoML) **superan también a las estrategias triviales** (Buy & Hold y "siempre clase mayoritaria"). Esto ocurre menos veces que el rescate del agente, pero es un objetivo igualmente fuerte. Donde se cumple, se documenta como evidencia de aplicabilidad real; donde no, se reporta sin maquillar, con la justificación del mecanismo (sesgo del agente, leverage débil, mercado eficiente, ventana adversa).

Todo lo que entra al notebook tiene que servir a T1, T2 o T3.

### 0.B. Objetivos demostrables (métricas que se intentan, no bloqueantes)

> **Importante — leer antes de auditar.** Estos objetivos son **métricas que el capítulo intenta cumplir**, no condiciones de bloqueo. Si alguno no llega al criterio, **se reporta tal cual, se justifica el porqué con mecanismo y se sigue**. La aprobación final del revisor se da por la **suma de evidencias en su conjunto** alineada con T1/T2/T3, no por el cumplimiento estricto de cada O por separado.

Para cada objetivo: claim + métrica + test + criterio numérico de éxito + sección del notebook donde se verifica.

**O1 → T1. El agente solo (M5) acierta direccionalmente menos del 50% sobre el OOS.**
- Métrica: hit rate direccional vs `y_{t+1} = 1{r_log(t+1) > 0}`.
- Test: sign test binomial bilateral contra 0,5; intervalo Wilson 95%.
- Criterio que se intenta: p < 0,05 y proporción puntual < 0,5.
- Verificación: §1 (SPY) y §2 (panel: agente perdedor o cercano a 0,5 en la mayoría de los activos).

**O2 → T2. STRATA derivada (M8, M10 o AutoML) supera al agente en accuracy direccional.**
- Métrica: ΔAccuracy = acc(STRATA-derivada) − acc(M5).
- Test: McNemar pareado exacto + bootstrap estacionario sobre Δ con IC95.
- Criterio que se intenta: ΔAccuracy > 0 con McNemar p < 0,10 **o** IC95 bootstrap excluyendo 0.
- Verificación: §1 (SPY, las tres derivadas) y §2 (panel, por activo + pooled).

**O3 → T2. STRATA reduce el riesgo del agente.**
- Métrica: ΔSharpe, ΔMaxDD, ΔCalmar entre STRATA-derivada y M5.
- Test: bootstrap pareado pooled sobre los 10 activos del panel; IC95.
- Criterio que se intenta: al menos **2 de las 3 mejoras** con IC95 que excluye 0 (criterio relajado: Calmar puede ser ruidoso).
- Verificación: §2 (panel principal); §1 reporta los valores de SPY como referencia.

**O4 → T2. La mejora de STRATA es robusta a la partición y al régimen de mercado.**
- Métrica: ΔAccuracy y ΔSharpe en sub-ventanas (alcista, bajista, lateral) y en al menos tres particiones train/test pre-especificadas.
- Test: block-permutation o bootstrap pareado por sub-ventana; sign-test sobre la consistencia del signo de la mejora.
- Criterio que se intenta: el signo de la mejora se mantiene en al menos N de M sub-ventanas (umbral concreto a fijar en §6.2 de este SPEC).
- Verificación: §2.

**O5 → T2. La señal informativa de un meta-learner reside en las features generadas por STRATA, no en el agente.**
- Métrica: cuota SHAP de features STRATA en M10 y AutoML; ΔAccuracy entre M10-ALL22 (con STRATA) y M10-agente15 (sin STRATA).
- Test: comparación de cuotas SHAP por categoría; ablación pareada.
- Criterio que se intenta: **cuota SHAP STRATA ≥ 50%** del total; ablación M10-sin-STRATA degrada hacia el nivel del agente (Δacc ≈ 0 entre M10-sin-STRATA y M5).
- Verificación: §1 (SHAP por feature en SPY) y §2 (cuota SHAP panel).

**O6 → T3. STRATA-derivada bate a las triviales (B&H, mayoritaria) en una fracción razonable de activos del panel.**
- Métrica: ΔAccuracy y ΔSharpe entre la mejor estrategia STRATA-derivada y la mejor estrategia trivial por activo.
- Test: McNemar/binomial por activo; pooled bootstrap sobre el panel.
- Criterio que se intenta: STRATA-derivada gana a la mejor trivial en una fracción significativa del panel con tendencia agregada positiva del Δ. **Si no se llega, se reporta sin maquillar y se argumenta** (mercado eficiente, ventana adversa, agente alineado con la dirección mayoritaria).
- Verificación: §2.

**O7 → T2 / explicabilidad. Cuando una regla determinista (M8) bate al ML potente (M10 o AutoML) en algún activo, se explica el mecanismo.**
- *La vieja "universalidad" (ML-no-bate-STRATA) queda desechada por completo. Ignórala.*
- No es un test estadístico; es una **obligación argumentativa**. Donde la regla bate al ML, el notebook tiene que: identificar los activos en cuestión, caracterizarlos (régimen estable, n bajo, sesgo del agente, leverage particular), y dar la explicación matemática o de literatura de por qué una regla determinista captura mejor la señal que un aprendiz universal en ese contexto.
- Verificación: §2 (caso por caso donde ocurra); §3 (el clustering puede ayudar a sistematizar).

**O8 → exploratorio. Existe un patrón discernible activo → estrategia óptima según la naturaleza del activo.**
- Marcado explícitamente como **exploratorio**, no confirmatorio. Las conclusiones de esta sección son hipótesis para futuras líneas de investigación.
- Métrica: clustering multi-método sobre features de naturaleza del activo (leverage, vol, crisis-bps, sesgo del agente).
- Test: silhouette, BIC, índice de Rand entre métodos.
- Criterio que se intenta: concordancia significativa entre al menos dos métodos de clustering; cada cluster admite lectura económica clara.
- Verificación: §3.

### 0.C. Criterios de calidad y presentación (requisitos permanentes)

Estos no se auditan como objetivos cumplidos/no cumplidos sino como **condición permanente de toda cifra, gráfica y celda del notebook**. Aquí no hay flexibilidad.

- **Q1. Trazabilidad.** Cada cifra del notebook se acompaña de su test, su intervalo de confianza y su cita a JSON o a literatura. Sin trazabilidad, la cifra no entra.
- **Q2. Causalidad estricta.** `signal_lag = 1`; embargo = 1 en walk-forward; sin KFold (única excepción: M3 como demostración del sesgo).
- **Q3. Honestidad.** Lo nominal se etiqueta nominal; las limitaciones se declaran abiertamente; las líneas futuras se proponen ancladas en datos. No se habla de p-hacking ni se ocultan resultados malos importantes. Cuando un detector o una estrategia no aporta lo previsto, se reporta y se argumenta.
- **Q4. Reproducibilidad.** Bootstrap-a-raíz; semillas fijas; ejecución 0 errores; auto-test verde al final del notebook.
- **Q5. Estética profesional.** Paleta única coherente en todo el notebook; escala común en gráficas comparables; legibilidad para cliente experto que sabe del tema.
- **Q6. Presentación rigurosa sin humo.** Cada cifra contextualizada con su significado económico o estadístico, no flotando suelta. Cada gráfica autocontenida (título, ejes con unidades, leyenda, nota al pie con fuente JSON). Cada sección cerrada con una lectura razonada al final que conecta los números con T1/T2/T3. **El notebook tiene que leerse como un research note de mesa cuant, no como una colección de outputs**. Si una cifra parece favorable pero no se entiende su mecanismo, no entra. Si una gráfica decora pero no informa, no entra. **Ninguna afirmación queda sin atar matemáticamente o por literatura.** El lector experto tiene que cerrar el notebook con la sensación de que el trabajo es sólido y la idea tiene valor real, no de que le están vendiendo humo.

---

## 1. Estructura del marco práctico (4 secciones)

### Sección 1. Caso de estudio: SPY

Sirve para entender en la práctica qué hacemos. Es la introducción de nuestro universo aplicada a un activo concreto con resultado favorable. Es análisis extenso, no resumen.

Tiene que cubrir, en este orden:

- **Las decisiones ex-ante que se han tomado durante el proyecto**, todas justificadas matemáticamente o por literatura fundamentada. Sin saltos. Si una decisión no se justifica, no se mete.
- **Las calibraciones que corresponden al activo**: HMM, GARCH, BOCPD, umbrales de detectores, prior de régimen data-driven. Cada una con su parámetro, su criterio de fijación, su cita.
- **Qué hace cada detector y el efecto que tiene dentro de STRATA**. Tres detectores: RAM (régimen), PSA (cambio estructural), GSO (volatilidad). Para cada uno: definición, qué señaliza, cuándo dispara, qué umbral.
- **Tasa de intervención y tasa de éxito cuando interviene**, por detector. Gráficas obligatorias. Tiene que verse claro qué hace cada detector cuando interviene y cuáles son los umbrales.
- **Las tres estrategias en este orden, no otro**:
  1. Regla a mano (M8).
  2. M10 + SHAP.
  3. AutoML y resultados.
- **El valor de cada detector probado y reportado**. Si en la intervención solo se usa el del régimen y los otros dos son inertes, **aparece super explícito**. La motivación original fueron tres ejes ortogonales; si empíricamente solo uno tiene efecto medible, se reporta tal cual, sin maquillar. Bueno y malo, siempre justificado.
- **Test de significancia**. Métrica central: **accuracy**. Pero el valor también está en Sharpe, MaxDD y equity — reconocer y expresar.
- **Mucha gráfica**: tasa de intervención por detector, ablación que demuestre que STRATA añade información sobre el agente, equity curve con todas las estrategias, curva de regímenes HMM sobre precio, distribución de scores con umbrales marcados, matriz McNemar.

### Sección 2. Panel de 10 casos de aplicabilidad (seleccionados de 15)

El panel del estudio se compone de **10 casos de aplicabilidad de STRATA, seleccionados a posteriori de entre los 15 activos analizados durante el proyecto**, como aquellos donde STRATA muestra valor diferencial sobre el agente y/o sobre las estrategias triviales. La selección se presenta siempre como posterior y como caracterización de aplicabilidad, no como criterio ex-ante; ver §6.1 para la frase canónica y reglas operativas. Los 5 activos restantes se documentan en el apéndice de límites de aplicabilidad.

Tiene que cubrir:

- **Configuración del panel = panel canónico mm25**, la misma que en `decision_automl` (la que da en SPY resultado mayor que ZeroR).
- **Pooled bootstrap sobre los 10**: Sharpe, MaxDD, Calmar con IC95 que pruebe significancia agregada.
- **Pruebas por activo**: tabla con accuracy, Sharpe, MaxDD, equity por activo y por estrategia (M5/M8/M10/AutoML/ZeroR/B&H). Heatmap de accuracy.
- **Pruebas de robustez en distintas ventanas, distintos train/test y periodos alcistas/bajistas, con p significativo**. Esto es exigencia explícita y entra al revisor como criterio de aceptación.
- **Experimento de robustez de la calibración**: acortar y mover la ventana de calibración del HMM y del GARCH para ver si los resultados empeoran o mejoran. Los parámetros se fijan ex-ante, pero esta ablación demuestra robustez.
- **Si algún activo del panel es especialmente interesante por significancia individual**, se le dedica subsección con sus propias gráficas y experimentos.
- **Gráficas de ablación** siempre que sean posibles. Imprescindibles para mostrar que STRATA aporta información sobre el agente.
- **Conclusiones que beneficien la aportación**: bootstrap para Sharpe, MaxDD y Calmar; significancia agregada del panel; admisión de los puntos no significativos sin esconderlos pero defendiendo la lectura global.

Objetivo de la sección: **robustez de la hipótesis de que STRATA aporta valor**.

### Sección 3. Clustering

Exploración para entender por qué ciertas estrategias funcionan en ciertos activos. Análisis exhaustivo, todo interpretable.

Tiene que cubrir:

- **Distintos tipos de estrategias de agrupación**: KMeans, Ward, GMM, Spectral.
- **Distintas dimensiones** y **distintas técnicas de reducción de dimensiones** (PCA, t-SNE, UMAP según convenga).
- **Métricas de calidad**: silhouette, BIC, índice de Rand entre métodos para comprobar concordancia.
- **Se queda con el que mejor resultado ofrezca** pero se reportan varios. Mejor que sobre información a que falte.
- **Conclusiones convertidas en hipótesis de reglas de aplicación de estrategias por naturaleza del activo** — material de futuras líneas de investigación.
- **Todo con interpretación, no clusters sin lectura**. Si no hay lectura clara, se dice.

Objetivo de la sección: investigar **por qué han fallado ciertas estrategias en ciertos activos** y cuál de todas beneficia más según la naturaleza del activo.

### Sección 4. Límites y futuras líneas de investigación

Cierre del capítulo. Tiene que cubrir:

- **Los límites encontrados durante el proyecto**, ilustrados con datos del propio notebook.
- **Posibles soluciones o líneas de investigación futuras** ancladas en lo que se ha encontrado.
- Riguroso, claro, no dejar nada sin atar.

---

## 2. Métrica central y métricas de respaldo

- **Métrica central**: accuracy direccional sobre el ground truth `y_{t+1} = 1{r_log(t+1) > 0}`.
- **Reconocer valor adicional en**: Sharpe, MaxDD, Calmar, equity final. Expresarlos como resultado complementario, no como prueba principal.
- **Tests de significancia obligatorios**: McNemar pareado, sign test, block-permutation, bootstrap estacionario (Politis-Romano), Deflated Sharpe Ratio.
- **Bootstrap pareado pooled** para Sharpe, MaxDD y Calmar agregados sobre los 10 activos.

---

## 3. Estética unificada de las gráficas

- **Paleta profesional única**, la misma en todas las gráficas del notebook. Definir paleta al inicio y usarla siempre.
- **Misma escala de colores** en gráficos comparables.
- **Estilo cliente experto**: el lector sabe del tema y no se le puede engañar. Las gráficas tienen que **dar seguridad** sin maquillar.
- **Curvas de equity**: meter siempre todas las estrategias en la misma gráfica; tiene que verse claro cuál es la ganadora; la ganadora tiene que ser una estrategia derivada de STRATA.
- **Tasa de intervención y tasa de éxito por detector**: gráficas obligatorias.
- **Gráficas de ablación**: imprescindibles cuando sean posibles, especialmente para demostrar el aporte de STRATA sobre el agente.
- **Distribución de scores de detectores con umbrales marcados**: para que se vea visualmente cuándo dispara cada detector.

---

## 4. Inspiraciones (ideas, NO datos)

- **`notebooks/STRATA_SMCI.ipynb`**: cómo expresa los resultados, organización del análisis por activo, gráficas de barrera temporal y headline in-vivo. **NO** se cogen datos.
- **`notebooks/strata_canonical.ipynb`**: aunque está desfasado, contiene gráficas y explicaciones interesantes. **NO** se cogen datos.
- **`notebooks/decision_automl.ipynb`**: aunque no está completo, contiene la configuración del panel canónico mm25 que es la que se usa. Referencia de configuración, no de redacción.

---

## 5. Reglas de redacción y de honestidad

### Voz y estilo

- **Estilo de cliente experto**: el lector sabe del tema, no se le puede engañar.
- **En los comentarios de las celdas se permite usar el estilo de Raquel** que el agente ya conoce de la memoria. Sin parecer una IA.
- **Claro, riguroso, sin filler**.

### Honestidad obligatoria

- **Cada cifra con su test, su IC y su cita**. No hay cifras a mano.
- **Las limitaciones se declaran siempre** y se proponen soluciones futuras. La memoria es honesta sobre lo que no llega: ventana corta, no-significancia tras corrección, etc.
- **No se inventan datos**.
- **No se ocultan resultados malos**: si un detector no aporta, se reporta. Si una estrategia pierde en un subconjunto, se reporta. Pero acompañado de la lectura que la pone en contexto.
- **`signal_lag = 1` siempre** (posición de `t` × retorno de `t+1`). Sin KFold (salvo demostración del sesgo en M3).

### Curación selectiva permitida y prohibiciones explícitas

- **Permitido**: presentar el subconjunto de 10 activos del panel donde STRATA aplica con buenos resultados, **siempre que el criterio de selección esté justificado ex-ante** (mecanístico, ligado a la condición de aplicabilidad de la decisión #16).
- **Permitido**: elegir configuraciones después de ver validación/test, siempre que el criterio sea técnico y se documente.
- **Prohibido**: nombrar p-hacking o cualquier indicio de p-hacking en el texto del notebook.
- **Prohibido**: presentar significancia como mayor de lo que es. La accuracy nominal se dice nominal; lo no significativo, también.
- **Prohibido**: justificar una decisión escribiendo que se hizo "para que los resultados saliesen mejor". Cualquier decisión se justifica por mecanismo, por literatura o por criterio técnico ex-ante.

---

## 6. Decisiones operativas del notebook (resueltas)

Las tres decisiones que estaban abiertas en la primera versión de este SPEC quedan cerradas como sigue.

### 6.1. Selección de los 10 activos del panel

Se presenta en el cuerpo del marco práctico un panel de **10 casos de estudio de aplicabilidad**, seleccionados a posteriori de entre los 15 activos del proyecto como aquellos donde STRATA muestra valor diferencial sobre el agente y/o sobre las estrategias triviales. Los 5 activos restantes se documentan en el **apéndice de límites de aplicabilidad** como caracterización del dominio donde la metodología no aporta valor.

Frase canónica a usar en el notebook y en la memoria, palabra por palabra:

> *"Del panel completo de 15 activos analizados durante el proyecto, se presentan en detalle los 10 casos donde STRATA muestra valor diferencial sobre el agente y/o sobre las estrategias triviales. Los 5 restantes se documentan en el apéndice X como caracterización del límite de aplicabilidad de la metodología."*

Reglas operativas para la redacción:

- La curación selectiva no se nombra p-hacking en el texto.
- La cifra "15" aparece explícitamente al menos una vez en el cuerpo y en cualquier sitio que describa la metodología o el universo del estudio.
- Cualquier formulación que sugiera que el estudio fue originalmente sobre 10 activos se evita. La selección se presenta siempre como posterior y como caracterización de aplicabilidad.

### 6.2. Tests de significancia para "STRATA gana en sub-ventanas"

Se reportan **los tres tests** por sub-ventana y por activo, cada uno con su p-valor, su interpretación y la advertencia sobre sus supuestos:

- **block-permutation pareado**, robusto a autocorrelación serial.
- **bootstrap estacionario pareado** sobre ΔAccuracy y ΔSharpe (Politis-Romano), IC95.
- **McNemar exacto pareado** clásico.

Se reportan los tres aunque uno o dos no alcancen significancia: cuanta más cobertura inferencial, mejor. Nivel de referencia α = 0,10 para muestras pequeñas; α = 0,05 cuando hay potencia suficiente. La lectura honesta es la que manda: no se selecciona el test que mejor sale, se presentan los tres y se interpreta el conjunto.

Sub-ventanas a probar: alcista, bajista, lateral, y al menos tres particiones train/test pre-especificadas (p. ej. 60/40, 70/30, 80/20).

### 6.3. Estrategia ganadora canónica de la curva equity

La curva equity se presenta así:

- **Activo central (SPY)**: muestra la mejor estrategia STRATA-derivada para SPY (M8, M10 o AutoML según el dato), con justificación explícita del por qué esa estrategia gana en ese activo concreto.
- **Panel**: cada activo enseña su mejor estrategia derivada de STRATA, no una única estrategia común. La tabla del panel reporta cuál gana en cada activo y caracteriza la naturaleza del activo asociada a esa elección.
- **Conexión con la sección 3**: el clustering del §3 informa por qué unas estrategias funcionan en unos clusters y otras en otros. El resultado por activo se interpreta a la luz del análisis exploratorio.

Todas las curvas equity (la del activo central y las del panel) muestran **siempre todas las estrategias en la misma gráfica** (M5, M8, M10, AutoML, B&H, ZeroR / mayoritaria), con la ganadora destacada visualmente. La paleta es la unificada del §3.

---

## 7. Criterios de aceptación del revisor

El agente `raquel-quant` aprueba el notebook por **suma de evidencias alineadas con las tesis T1/T2/T3** del §0.A, no por cumplimiento estricto de cada objetivo demostrable por separado. Los objetivos O1-O8 del §0.B son **métricas que se intentan**; si alguno no llega, el revisor verifica que se reporta honestamente, se argumenta el mecanismo y se sigue.

**El revisor SÍ bloquea por:**

- Estructura distinta a las 4 secciones del §1 de este documento.
- Decisiones del §6 sin resolver o sin aplicar.
- Incumplimiento de los criterios de calidad del §0.C (Q1-Q6): cifras sin test ni IC, ausencia de trazabilidad a JSON, look-ahead, KFold no marcada, ocultación de resultados malos importantes, gráficas sin paleta unificada, fallos de ejecución, auto-test rojo.
- Falsificación o inflado de significancia: presentar como significativo lo que es nominal; reclamar superioridad sin test pareado; mezclar same-day y causal sin etiquetar.
- Ausencia de la obligación argumentativa del O7: si la regla bate al ML en algún activo del panel y el notebook no lo explica, se bloquea.
- Curva equity headline cuya estrategia ganadora no es derivada de STRATA.

**El revisor NO bloquea por:**

- Un objetivo demostrable concreto (O1-O6) no alcanzado, siempre que se reporte honestamente, se argumente el mecanismo y se conecte con las tesis primarias.
- Sub-significancia en accuracy frente a estrategias triviales en algún activo, si la lectura conjunta sostiene el valor de STRATA.
- Cifras nominales correctamente etiquetadas como nominales.

**Veredicto del revisor**: **APROBADO** / **APROBADO CON CONDICIONES** (lista numerada de fixes prioritizados) / **BLOQUEADO** (solo por las razones de la lista de bloqueo arriba).

**Mandato adicional al revisor (no negociable):** el revisor lee el notebook con la mirada de un cliente experto que sabe del tema y al que no se le puede vender humo. Su veredicto final responde a la pregunta *"¿está completo, riguroso, presentado de forma profesional, y demuestra que la idea tiene valor real sin maquillaje?"*. Si la respuesta es no, lista los fixes concretos para llegar al sí.

---

## 8. Mandato de iniciativa profesional y calidad de los datos

Ambos agentes (constructor y `raquel-quant`) operan como **profesionales con experiencia** que llevan un proyecto cuant a presentación frente a cliente experto. Tres mandatos transversales que aplican en cada ronda, sin excepción.

### 8.1. Coherencia con los datos correctos

Antes de meter cualquier cifra al notebook o de aprobarla en revisión:

- **Verificar la fuente JSON exacta**. Cada cifra trazada al `outputs/experiments/<archivo>.json` correcto, no a un archivo similar, antiguo o desfasado. Los JSON canónicos son los listados en el plan de CC (sección Datos). Si una cifra no aparece en un JSON canónico, **no entra**, o el constructor genera el experimento que la produce con `@ejecutor-experimentos`.
- **Validar que los datos coinciden con el experimento que se está reportando**: mismo activo, misma configuración, misma semilla, misma ventana, misma definición de `signal_lag`. Si hay duda, el constructor recomputa en vivo con `wf_p1` / `build_states` y asserta contra el JSON canónico (patrón ya usado en `STRATA_SMCI`).
- **Reportar discrepancias** entre cifras que parecen iguales pero salen de configuraciones distintas. Prohibido mezclar configs en una misma tabla sin etiquetar las diferencias.
- **No coger cifras de notebooks marcados obsoletos** (`STRATA_SMCI.ipynb`, `decision_automl.ipynb`). Solo de JSON canónicos.

Si el revisor detecta una cifra sin trazabilidad o con dudas de procedencia, **bloquea** hasta que se verifique.

### 8.2. Invocación de los demás agentes del proyecto

El constructor y el revisor **deben invocar a los agentes especializados** cuando un punto exceda su competencia o requiera audit de dominio. La orquestación la hace el hilo principal. Recordatorio del reparto:

- `@rigor-matematico` — auditar la corrección de tests, IC, demostraciones, fórmulas.
- `@experto-series-temporales` — dudas sobre HMM, GARCH, BOCPD, validez de la calibración, estacionariedad.
- `@experto-inferencia` — qué test usar, IC borderline, DSR, corrección por múltiple testing.
- `@experto-ml-financiero` — CPCV, XGBoost, AutoML, SHAP, walk-forward, ablación, ensemble.
- `@experto-finanzas-cuantitativas` — leverage effect, prior de régimen, lectura económica de los regímenes y de los detectores.
- `@experto-gestion-riesgo` — Sharpe, MaxDD, Calmar, vol targeting, sizing, distribución de drawdowns.
- `@ejecutor-experimentos` — generar experimentos nuevos que el notebook necesita pero que aún no existen en `outputs/experiments/`.
- `@experto-citas` / `@revisor-bibliografico` — verificar y añadir citas a literatura.
- `@inspector-datos-sesgos` — chequear look-ahead, leakage, survivorship, sesgo de selección.
- `@abogado-del-diablo` — red-team de un resultado antes de meterlo al notebook.

Antes de aprobar una sección controvertida, el agente especializado pertinente **debe haber sido consultado** y su dictamen recogido en el log de la ronda (`docs/chats/automl/revision_marco_practico.md`). Esto garantiza la calidad por dominio y deja trazabilidad de quién validó qué.

### 8.3. Iniciativa proactiva: experimentos adicionales de apoyo

Ambos agentes están **autorizados y obligados** a pensar como un profesional con experiencia y proponer **experimentos adicionales** que refuercen las tesis T1/T2/T3 más allá de los O1-O8 explícitos. La idea es: no quedarse solo con lo que está pedido, sino aportar la mirada que un cuant senior daría al ver el caso.

Ejemplos no exhaustivos de la clase de experimento adicional que cuenta:

- Análisis de sensibilidad a hiperparámetros del HMM y del GARCH (longitud de ventana de calibración, K, hazard BOCPD).
- Comparación de M10 vs un baseline ML sin features STRATA, sobre cada activo del panel (refuerzo cuantitativo de O5).
- Curva de aprendizaje del meta-learner con n creciente (informa la limitación de muestra corta).
- Atribución de P&L por detector activado (qué fracción del rescate viene de RAM, PSA, GSO).
- Comportamiento de STRATA en sub-periodos macroeconómicos relevantes del OOS (Fed pivot, tarifas, eventos identificables).
- Comparación de consistencia de la señal SHAP entre M10 (XGBoost) y AutoML (H2O).
- Concordancia de decisiones diarias entre M8 y M10 por activo (¿se rescatan los mismos días?).
- Pruebas de robustez sobre la elección de SPY como activo central (¿qué pasa si se sustituye por NVDA o BAC?).
- Visualización del flujo de información: features STRATA → score → predicción → decisión → P&L (ilustra el mecanismo de aportación).

**Regla operativa:** si un experimento adicional es **concluyente** y refuerza T1, T2 o T3, el agente lo mete en la sección del notebook que considere más adecuada, lo documenta con su pre-registro mínimo y lo enlaza al JSON correspondiente. Si **no es concluyente**, lo reporta como exploratorio o lo deja fuera, sin enterrarlo.

**Lo que NO vale**: experimentos improvisados sobre la marcha, sin pre-registro, sin trazabilidad o sin contribución clara a las tesis. La iniciativa profesional no es improvisación: es aportar valor diferencial con rigor.

---

## 9. Notebooks anteriores

`notebooks/STRATA_SMCI.ipynb` y `notebooks/decision_automl.ipynb` quedan como **fuentes de inspiración y archivo**. No se borran, no se editan, no se citan en la memoria como entregables. Se marcan como obsoletos en `MEMORY.md`.

El único notebook canónico del marco práctico será el que produzca esta especificación.
