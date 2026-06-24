# Correcciones de la memoria — registro vivo

Documento de trabajo para ir acumulando **todo lo que descuadra** en la memoria del TFG
(`tesis/chapters/`), detectado en la auditoría del 2026-06-24 (lectura de catedrático +
agentes de rigor matemático, trazabilidad de cifras contra JSON, y red-team de tribunal).

**Cómo usarlo.** Un apartado por bloque grande (un capítulo, más uno transversal de defensa).
Cada ítem lleva: severidad · dónde (fichero:línea) · **qué descuadra** · **por qué importa** ·
**qué hacer** · estado. Marca el estado al resolverlo: `ABIERTO` → `HECHO` / `DESCARTADO`.

Severidad: 🔴 crítico (un tribunal lo caza enseguida / contradicción literal) ·
🟠 importante (rigor o trazabilidad) · 🟡 menor (estilo, equilibrio, recorte).

Estado de la verificación: `[verificado]` = comprobado contra el .tex o el JSON real ·
`[provisional]` = criterio de tribunal o pendiente de comprobar en código/notebook.

---

## BLOQUE A — Cap.1 Introducción (`01_introduccion.tex`)

### A1 · 🟡 · El capítulo está sin redactar · ABIERTO
- **Dónde:** `01_introduccion.tex` (solo comentarios de estructura).
- **Qué descuadra:** es el único capítulo vacío. [verificado]
- **Por qué:** la memoria no puede entregarse sin introducción; además fija el gancho y la
  hipótesis que el resto desarrolla.
- **Qué hacer:** redactar con el pipeline `/redaccion-capitulo 1`. Debe abrir con SPY como caso
  central (ver B1) y enunciar la hipótesis falsable y la jerarquía de valor honesta.

---

## BLOQUE B — Cap.2 Estado del arte (`02_estado_arte.tex`)

### B1 · 🔴 · Contradicción SMCI vs SPY como caso de estudio · ABIERTO
- **Dónde:** `02_estado_arte.tex` líneas 10-12 vs todo el Cap.4 (`cap4_parts/s1_spy.tex` línea 7).
- **Qué descuadra:** el Cap.2 dice literalmente *"SMCI, **el activo del caso de estudio**"*, con
  el agente perdiendo a accuracy **0,484** y €1000→€978. El Cap.4 construye "toda la cadena"
  sobre **SPY**, donde el agente pierde a **0,366**. [verificado]
- **Por qué:** es la grieta más letal y no requiere saber estadística para verla. Es un resto del
  pivote del 2026-06-23 (decisión #18: caso central = SPY; SMCI pasa a ser uno más del panel de
  10). El lector concluye "tesis reescrita a medias" y desconfía del resto.
- **Qué hacer:** reescribir 02 líneas 10-12 para que el caso motivador sea **SPY**, o presentar
  SMCI explícitamente como "el activo que originó el proyecto, generalizado después a SPY y al
  panel". Unificar el número del agente perdedor (decidir si se cita el 0,366 de SPY o se quita
  la cifra concreta del estado del arte). Revisar que no quede ninguna otra mención a SMCI como
  caso central en ningún capítulo.

### B2 · 🟡 · Capítulo corto y con erratas de prosa · ABIERTO
- **Dónde:** `02_estado_arte.tex` (31 líneas frente a 906 del Cap.3); sección "El hueco que llena
  STRATA", línea 28 en adelante.
- **Qué descuadra:** desequilibrio de extensión; y la frase de la línea 31 está mal puntuada
  (*"Estas tres líneas de investigación se han tocado, por separado, nuestra motivación es crear
  un sistema que las una…"* — falta conector/punto) y tiene "corrigirá" (debe ser "corregirá").
  [verificado]
- **Por qué:** un estado del arte de TFG suele desarrollar más la posición frente a la literatura;
  y las erratas restan en la primera impresión.
- **Qué hacer:** pulir prosa y, si procede, ampliar el posicionamiento frente a trabajos previos.

---

## BLOQUE C — Cap.3 Marco teórico (`03_marco_teorico.tex`)

> Las seis demostraciones (forward, monotonía EM, estacionariedad GARCH, pseudo-residuos,
> pesos XGBoost, recursión BOCPD) están **verificadas y son correctas**. En particular, la
> dirección de necesidad del Teorema de estacionariedad GARCH **no es circular**. [verificado]
>
> **Re-auditoría 2026-06-24 (v2):** el fichero creció con una sección nueva §2 "Agentes de
> trading basados en modelos de lenguaje" (líneas 44-97, con figura TikZ del bucle del agente).
> Verificado: sus 9 citas existen en el `.bib` (sin alucinaciones); el preámbulo carga
> `tikz` + `arrows.meta` (la figura compila); labels y refs cruzadas correctas. La sección es
> una buena adición. Confirmado además que **C1-C12 siguen sin resolver** (no se ha tocado
> nada de lo anterior). Nuevos ítems de la sección: C13, C14.

### C1 · 🔴 · Colisión de notación en λ (tres significados) · ABIERTO
- **Dónde:** λ = parámetros del HMM (líneas 103-283); λ = inverso del hazard BOCPD (λ=250,
  línea 564); λ = regularización L2 de XGBoost (λ=1, línea 699). [verificado]
- **Qué descuadra:** un mismo símbolo para tres objetos sin relación; los dos numéricos (250 y 1)
  conviven a ~130 líneas.
- **Por qué:** el Cap.3 presume de "fijar la notación que usa el resto de la memoria" (línea 9).
  Un tribunal lo marca.
- **Qué hacer:** renombrar el hazard BOCPD (p.ej. `ℓ` o `τ₀`). λ para parámetros HMM y para L2
  son convenciones fuertes (Rabiner / Chen-Guestrin) y se mantienen.

### C2 · 🟠 · Viterbi: uso real no atado a su figura + descuadre calibración/evaluación · ABIERTO
- **Dónde:** `03_marco_teorico.tex` línea 274 vs `cap4_parts/s1_spy.tex` línea 43 (`fig:mp-regimenes-spy`).
- **Qué descuadra:** el Cap.3 dice que Viterbi se usa "para la lectura interpretativa de la
  secuencia de regímenes sobre el periodo de **calibración**", pero (a) no lo ata a ninguna figura,
  y (b) la única figura de regímenes pinta el **periodo de evaluación** (OOS) y no declara si usa
  Viterbi o el filtrado. [verificado el texto; pendiente verificar en notebook qué usa la figura]
- **Por qué:** si la figura OOS usa Viterbi, es **no causal** (mira toda la serie) → un tribunal
  pregunta por el look-ahead aunque sea "solo una figura". Si usa el filtrado, entonces Viterbi no
  ilustra nada y su desarrollo completo (líneas 262-274) sobra.
- **Qué hacer:** (1) verificar en el notebook cómo se pinta `fig:mp-regimenes-spy`. (2) Si es
  filtrado, decirlo en el caption y reducir Viterbi a un párrafo (enunciado + cita Viterbi 1967 +
  "difiere de encadenar filtrados" + "uso auxiliar"); la recursión δ puede ir a nota. (3) Si es
  Viterbi sobre OOS, cambiar a filtrado para evitar el look-ahead, o mover la figura a calibración.

### C3 · 🟠 · BIC usado sin definir y con criterio cruzado · ABIERTO
- **Dónde:** línea 900 (clustering) vs línea 281 (selección de K del HMM por log-verosimilitud
  held-out).
- **Qué descuadra:** el BIC aparece por primera vez en la última subsección sin fórmula ni cita, y
  la frase sugiere que el BIC fija también "los estados", cuando K lo fijó la held-out. [verificado]
- **Por qué:** incoherencia de criterio (¿qué elige K, held-out o BIC?) y concepto sin introducir.
- **Qué hacer:** aclarar que BIC gobierna el nº de grupos del clustering del panel (y los estados de
  las mezclas gaussianas del clustering), mientras la held-out fija K del HMM. Dar fórmula + cita.

### C4 · 🟠 · "d-separación" invocada sin definir · ABIERTO
- **Dónde:** línea 155 (prueba de la recursión forward).
- **Qué descuadra:** se cita "la d-separación en el grafo del modelo" sin introducir el concepto
  de redes bayesianas. [verificado]
- **Por qué:** la prueba se sostiene solo con la independencia condicional (∗) y la markovianidad;
  d-separación es andamiaje innecesario.
- **Qué hacer:** eliminar la frase (recomendado) o citarla (Bishop 2006, cap. 8).

### C5 · 🟠 · Modelo de observación de PSA: familia conjugada nunca nombrada · ABIERTO
- **Dónde:** líneas 440-511 (predictiva conjugada de BOCPD) y 511, 564 (se difiere al "capítulo de
  metodología").
- **Qué descuadra:** RAM y GSO quedan totalmente especificados (Defs. de líneas 532, 547); PSA deja
  la "predictiva conjugada" en abstracto sin nombrar la familia exponencial (¿Normal-Gamma?). [verificado]
- **Por qué:** el lector no puede evaluar una "predictiva conjugada" sin saber la familia.
- **Qué hacer:** nombrar la familia exponencial concreta aquí (aunque los hiperparámetros vayan al
  Cap.4).

### C6 · 🟠 · GSO score sin declarar rango (no acotado) · ABIERTO
- **Dónde:** línea 552 (definición de GSO).
- **Qué descuadra:** RAM y PSA declaran soporte [0,1]; GSO no acota y diverge si σ_t→∞ (b_t→0).
  [verificado]
- **Por qué:** asimetría que el tribunal nota (dos scores en [0,1] y uno no acotado umbralizado por
  percentiles).
- **Qué hacer:** declarar GSO_t ∈ [0,∞) y justificar que por eso se umbraliza por percentiles.

### C7 · 🟠 · Orden de composición de la capa M8 no justificado · ABIERTO
- **Dónde:** líneas 584-595 (Def. capa M8, composición C∘R∘G).
- **Qué descuadra:** R_RAM(w)=ρ_t·b_t **descarta** su argumento (reorienta), así que al disparar
  **anula** el recorte que hizo G_GSO; los operadores no conmutan. La prosa describe los efectos en
  orden inverso al de aplicación y no avisa de esto. [verificado]
- **Por qué:** el tribunal preguntará por el orden y por qué RAM borra lo de GSO.
- **Qué hacer:** justificar el orden en una frase y señalar explícitamente la no conmutatividad.

### C8 · 🟡 · ℓ_val(K) impreciso sobre qué condiciona · ABIERTO
- **Dónde:** línea 281.
- **Qué descuadra:** la suma de log-predictivas un-paso presupone forward propagado desde el final
  del entrenamiento; tal como está, x_{1:t-1} mezcla train y val sin aclararlo. [verificado]
- **Por qué:** no es error de fórmula, es imprecisión; un examinador cuidadoso lo nota.
- **Qué hacer:** aclarar que la suma usa el filtrado propagado desde el tramo de entrenamiento.

### C9 · 🟡 · sign(0) y Sharpe sin r_f sin convención explícita · ABIERTO
- **Dónde:** líneas 530-540 (d_k, sign(w_t)); línea 762 (Sharpe = ḡ/σ̂_g, cita Sharpe 1994 que
  sí incluye exceso sobre r_f).
- **Qué descuadra:** no se fija la convención de sign(0) en preliminares; el Sharpe no resta r_f.
  [verificado]
- **Por qué:** detalles que un catedrático señala.
- **Qué hacer:** fijar sign(0) en preliminares; añadir una línea justificando r_f≈0 (long-short).

### C10 · 🟡 · Colisión T = nº de hojas vs T = horizonte temporal · ABIERTO
- **Dónde:** T = longitud de la serie (línea 103 y passim) vs T = nº de hojas del árbol (línea 668,
  Ω(f)=γT+…). [verificado]
- **Qué hacer:** nota aclaratoria o cambiar el nº de hojas a |hojas|/otro símbolo.

### C11 · 🟡 · Frase-mapa de apertura incompleta · PARCIAL
- **Dónde:** línea 9.
- **Qué descuadra:** la frase se editó (v2) y ahora sí cita la sección nueva del agente, pero
  **sigue anunciando "cuatro bloques" y omite la §"Estrategias supervisadas"** (M10/AutoML/
  boosting, `sec:estrategias`), que es extensa y central. [verificado v2]
- **Qué hacer:** añadir ese bloque al mapa (es el que aprende sobre las señales).

### C13 · 🟡 · Coherencia de la salida del agente: ¿lleva confianza la decisión final? · ABIERTO
- **Dónde:** línea 90 (el gestor de cartera integra en *"una dirección y un tamaño"*) vs líneas
  95 y 102 (*"una dirección, un tamaño y la confianza con que los emite"* / terna signo-tamaño-
  confianza). [verificado v2]
- **Qué descuadra:** la decisión orquestada se describe sin confianza en un sitio y con confianza
  en otros dos. Las 15 features del meta-learner son la confianza **por personalidad** (línea
  660), no necesariamente una confianza de la decisión final.
- **Qué hacer:** aclarar si la decisión final del fondo lleva un escalar de confianza propio o si
  la confianza es solo por personalidad. Unificar la descripción de la salida en las tres líneas.

### C14 · 🟢 · Oportunidad: el gestor de riesgo interno explica por qué GSO casi no dispara · ABIERTO
- **Dónde:** línea 90 (*"Un gestor de riesgo fija el límite de la posición a partir de la
  volatilidad reciente"* — interno al agente) vs s1_spy/GSO inerte (F5, D-GSO 0% disparo).
- **Qué descuadra:** no es un error, es una conexión sin explotar. El agente **ya dimensiona por
  volatilidad internamente**; por eso su tamaño rara vez viola la banda de GSO y el detector
  externo casi nunca dispara. [verificado v2 — coherente con que GSO es inerte]
- **Por qué importa:** convierte una debilidad ("GSO no hace nada") en un hallazgo defendible
  ("GSO es redundante con el risk-manager interno del agente, que ya hace vol-targeting"). Buena
  munición frente a la objeción F5.
- **Qué hacer:** añadir una frase en §GSO (o en límites del Cap.4) ligando la inactividad de GSO
  al risk-manager interno del agente. Refuerza, no debilita.

### C12 · 🟡 · Recortar pasajes de nivel doctoral · ABIERTO
- **Dónde:** monotonía EM (243-258); construcción de la solución estacionaria GARCH en L¹
  (370-389); recursión completa de Viterbi (262-274, ver C2).
- **Qué descuadra:** tres pasajes largos por encima del nivel de TFG. [provisional, es criterio]
- **Por qué:** desproporción; Viterbi además ni se usa operativamente.
- **Qué hacer:** conservar **una** de las dos pruebas largas como vitrina y reducir la otra a
  enunciado + esquema + cita; recortar Viterbi (ver C2).

---

## BLOQUE D — Cap.4 Marco práctico (`04_marco_practico.tex` + `cap4_parts/`)

> Verificación cifra a cifra contra los JSON de `outputs/experiments/`: la inmensa mayoría
> coincide al decimal, sin cifras inventadas en las tablas principales. [verificado]

### D1 · 🔴 · La figura de casos (XLE) contradice su propio JSON · ABIERTO
- **Dónde:** `cap4_parts/s3_patrones.tex` línea 97 y caption de `fig:mp-casos` (línea 135).
- **Qué descuadra:** etiqueta XLE como *"leverage fuerte: el régimen es direccional y la regla
  rescata"*, pero `mechanism_panel.json` → XLE da `crisis_mean=+0.00007` (positivo), canal ganador
  = **AutoML**, mecanismo literal *"leverage INVERTIDO… la regla M8 es ruido; el aprendiz rescata"*.
  En `tab:mp-accuracy` XLE lo lidera AutoML (0,532), no M8 (0,528). [verificado]
- **Por qué:** estás ilustrando el mecanismo de la **regla** con el único índice del grupo que el
  dato asigna al **aprendiz**. Quien abra el JSON lo ve en 30 s.
- **Qué hacer:** cambiar el activo de ejemplo del panel izquierdo a uno con `crisis_mean<0`
  (SPY, QQQ, XLF, DIA o XLK) y regenerar la figura.

### D2 · 🟠 · p-valor huérfano sin JSON ("todas con p>0.14") · ABIERTO
- **Dónde:** `cap4_parts/s3_patrones.tex` línea 53.
- **Qué descuadra:** *"ninguna propiedad correlaciona con el rescate de la regla… todas con
  p>0.14"* no existe en ningún JSON de `outputs/experiments/` (solo hay correlaciones del rescate
  del *aprendiz*). [verificado: el dato no está en disco]
- **Por qué:** en una tesis cuyo lema es la trazabilidad ("toda cifra trazada a su JSON"), un
  p-valor sin fuente es munición directa.
- **Qué hacer:** volcar ese cálculo a un JSON propio y citarlo, o suavizar la frase a cualitativo.

### D3 · 🟠 · La ablación de M10 contradice el argumento y el texto lo oculta · ABIERTO
- **Dónde:** `cap4_parts/s1_spy.tex` línea 147 y `tab:mp-ablacion` (157-158).
- **Qué descuadra:** al quitar PSA+GSO, **M10 sube** 0,494→0,514 (correcto en la tabla), mientras
  AutoML baja 0,574→0,550. La prosa solo razona sobre AutoML ("degrada → usa sus scores") y calla
  que M10 mejora sin ellos. [verificado contra `detector_ablation_panel.json` y
  `automl_ablation_detectors.json`]
- **Por qué:** el tribunal preguntará "si la ablación prueba que el modelo usa PSA+GSO, ¿por qué
  M10 mejora al quitarlos?". El apoyo en los detectores se sostiene sobre AutoML y el SHAP, no
  sobre la ablación de M10.
- **Qué hacer:** añadir una frase reconociéndolo y reapoyar la conclusión en SHAP + AutoML.

### D4 · 🟠 · r(PC1, leverage) = 0,84 vs 0,83 recalculado · ABIERTO
- **Dónde:** `cap4_parts/s3_patrones.tex` línea 96 y caption `fig:mp-pca` (línea 128).
- **Qué descuadra:** reconstruido desde `cluster_panel10.json` (PC1 vs leverage_corr) da r=0,8309
  → 0,83, no 0,84. No está guardado como clave en el JSON. [verificado el recálculo]
- **Por qué:** desajuste pequeño pero no trazable, en una tesis de trazabilidad estricta.
- **Qué hacer:** recalcular en el notebook con datos sin redondear y reportar el exacto.

### D5 · 🟡 · Etiqueta imprecisa de la cuota SHAP 0,71 · ABIERTO
- **Dónde:** `cap4_parts/s1_spy.tex` línea 147.
- **Qué descuadra:** se llama "cuota del **ensemble** M10" (0,71), pero la fuente
  (`decision_automl_prep.json`) es TreeSHAP **sobre un árbol** (el JSON avisa: "ensemble sin
  atribución exacta → SHAP sobre mejor XGBoost"). La cifra (0,4791+0,1017+0,134=0,71) es correcta;
  la etiqueta no. Las otras dos cuotas (0,565 árbol / 0,564 permutación, n=401) sí coinciden
  exactas. [verificado]
- **Qué hacer:** matizar la etiqueta (SHAP sobre el mejor XGBoost del ensemble, no sobre el
  ensemble).

### D6 · 🟡 · Nombre del agente: "HedgeFoundAI" vs "AI Hedge Fund" · ABIERTO
- **Dónde:** `04_marco_practico.tex` línea 9 vs CLAUDE.md y resto de la memoria.
- **Qué descuadra:** dos nombres para el mismo agente. [verificado]
- **Qué hacer:** unificar a "AI Hedge Fund".

### D7 · 🟡 · BITÁCORA con pooled-15 obsoleto (no es el .tex, es defensa) · ABIERTO
- **Dónde:** `BITACORA.md` entrada [2026-06-23] (M8 +0,64 [0,10,1,29] / M10 +0,93 / AutoML +0,97).
- **Qué descuadra:** el .tex usa el pooled-10 correcto (M8 +0,60 [0,05,1,22] / M10 +1,12 / AutoML
  +1,08, = `bullbear_confirmatory.json` POOLED10). Las cifras de la BITÁCORA son del pooled-15
  obsoleto. El .tex está bien. [verificado]
- **Por qué:** dos números distintos en tus propios entregables el día de la defensa.
- **Qué hacer:** añadir entrada de corrección en BITÁCORA apuntando al pooled-10 canónico.

---

## BLOQUE E — Cap.5 Conclusiones (`05_conclusiones.tex`)

### E1 · 🟠 · Overclaiming: el alcance no viaja pegado a la afirmación · ABIERTO
- **Dónde:** línea 4 (*"un valor que sobrevive a un test"*, singular); línea 6 (*"le saca un punto
  de acierto a la regla"*, es +0,021 con IC90 que roza +0,001); s4 línea 59 (*"No es exposición
  pasiva capturada por casualidad"*, sin contraste detrás). [verificado]
- **Qué descuadra:** el cuerpo (s4) es escrupuloso, pero la prosa de conclusiones generaliza más
  que las tablas. El único test duro (pooled de riesgo) tiene el problema F1, y la regla no lo pasa.
- **Por qué:** un tribunal hostil explota el gradiente de optimismo entre tabla y narrativa.
- **Qué hacer:** matizar para que el alcance (aprendices / agregado / contra-agente) acompañe cada
  afirmación, como ya se hace en s4. "no es casualidad" → "es compatible con". Es edición.

---

## BLOQUE F — Objeciones de fondo / defensa (transversal)

> No son errores de cifra (las cifras están bien): son construcciones cuya **defensa** puede no
> bastar ante un tribunal. Cada una trae el experimento que la zanja. [provisional]

### F1 · 🟠 · El pooled-10 corrige la dependencia equivocada · ABIERTO
- **Dónde:** `cap4_parts/s2_panel.tex` líneas 90-118; `s4_limites.tex` línea 43.
- **Qué descuadra:** el pooled-10 (n=2493) es el resultado central y el único donde el riesgo cruza
  a significativo, pero apila 6 índices con correlación cruzada >0,9. El remedio citado —bootstrap
  de bloque √n— corrige la autocorrelación **temporal dentro de cada serie**, NO la correlación
  **transversal** entre activos el mismo día, que aquí domina. El n efectivo está más cerca de
  ~800-1000 que de 2493. [provisional]
- **Por qué:** es la objeción técnica más afilada; la defensa actual es cosmética (reconoce el
  problema pero ataca la dimensión equivocada).
- **Qué hacer:** re-bootstrap remuestreando **fechas completas** (todos los activos del mismo día
  juntos) o cluster-bootstrap por activo; reportar el nuevo IC y el n efectivo. Si M10/AutoML
  (+1,08/+1,12) siguen excluyendo el cero → la objeción muere y la tesis sale reforzada. La regla
  M8 (+0,60) probablemente se hunde más (coherente con que ya no pasa Bonferroni). **Prioridad 1**:
  sostiene el resultado central.

### F2 · 🟠 · max(M10, AutoML) es un estimador sesgado al alza · ABIERTO
- **Dónde:** `cap4_parts/s3_patrones.tex` línea 45 (ley del leverage) y clustering.
- **Qué descuadra:** la variable dependiente de la ley es `max(M10,AutoML)−M5`, elegida activo a
  activo. El máximo introduce sesgo de selección que crece con la varianza del activo — justo los
  volátiles de leverage débil (SMCI/MARA/ROKU/UNG) que sostienen la pendiente; ROKU "la excepción"
  es donde más mordería. [provisional]
- **Por qué:** contamina una correlación de n=10 ya al filo (p=0,093).
- **Qué hacer:** repetir la ley con la **media** de (M10−M5, AutoML−M5) o pre-registrando un solo
  modelo. Si r=−0,56 aguanta, la ley sobrevive limpia. Reportar también el clustering quitando
  leverage de los 5 rasgos (ver si los grupos persisten o eran el leverage disfrazado).

### F3 · 🟠 · Calibración del agente-LLM nunca verificada (SHAP quizá trivial) · ABIERTO
- **Dónde:** nivel de universalidad, `cap4_parts/s2_panel.tex` (cuota SHAP 0,66) y `05` línea 34.
- **Qué descuadra:** el agente está corto el 95% de los días → sus features son cuasi-constantes.
  SHAP reparte importancia hacia lo que **varía**; si el agente es casi constante, *cualquier*
  feature variable (STRATA) lo supera por construcción, no por mérito. [provisional]
- **Por qué:** podría vaciar "el aprendiz redescubre STRATA" → "el aprendiz redescubre lo único que
  se mueve".
- **Qué hacer:** reliability diagram + Brier score del agente; o ablación que sustituya las features
  del agente por su media constante (si el aprendiz pierde poco, el SHAP>0,5 era trivial).

### F4 · 🟠 · El modo override se eligió mirando el OOS · ABIERTO
- **Dónde:** `cap4_parts/s1_spy.tex` líneas 78-97 (`tab:mp-variantes`).
- **Qué descuadra:** se comparan override/abstención/reducir sobre el mismo OOS de SPY, gana
  override (0,478) y se adopta como canónico — luego se testa en ese mismo OOS. Son 3 modos × 3
  modelos de grados de libertad que Bonferroni (m=3) no cuenta. [provisional]
- **Por qué:** la tesis presume rigor de pre-registro; esto es un grado de libertad ex post.
- **Qué hacer:** si hay pre-registro del modo override en BITÁCORA **anterior** a esa tabla,
  citarlo (lo zanja). Si no, declararlo como exploración honesta y comprobar si M10/AutoML
  sobreviven con m=9.

### F5 · 🟡 · GSO inerte: ablación sin test de significancia · ABIERTO
- **Dónde:** `cap4_parts/s1_spy.tex` líneas 52, 147 (GSO 0% disparo; ablación 0,574→0,550).
- **Qué descuadra:** la única evidencia de que GSO/PSA aportan es un Δ=0,024 sin IC, sobre un
  AutoML que la propia memoria dice no reproducible con max_runtime. [provisional]
- **Por qué:** "tres detectores ortogonales" se queda en uno activo (RAM) + dos scores de aporte
  no testado.
- **Qué hacer:** IC bootstrap o McNemar sobre ALL22 vs sin-PSA+GSO, con el AutoML reproducible
  (max_models, sin DeepLearning). Si el Δ excluye cero, quedan justificados.

---

## Prioridad sugerida de resolución

1. **B1** (SMCI→SPY) — crítico, trivial, primera impresión.
2. **F1** (pooled cross-sectional) — sostiene el resultado central; el que más refuerza si sale bien.
3. **C1, D1** (λ, figura XLE) — críticos editables.
4. **F2, F3, F4** (max, calibración, override) — experimentos de blindaje para defensa.
5. **C3-C7, D2-D4** (gaps de rigor y trazabilidad).
6. **E1, C12, B2** (overclaiming, recortes, prosa).
7. **A1** (redactar introducción).
