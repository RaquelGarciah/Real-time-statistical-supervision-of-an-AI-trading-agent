ESCRIBIR LA MEMORIA DEL TFG — agentes especializados, LaTeX, anti-IA-detection

CONTEXTO
Soy Raquel García, autora de un TFG en Matemáticas y Ciencia de Datos en la UCM
sobre supervisión estadística de un agente LLM de trading. El proyecto
está en /Users/Raquel/Desktop/STRATA_kit. La constitución, decisiones, lecciones
y resultados objetivo viven en CLAUDE.md, DECISIONES_ESENCIALES.md,
LECCIONES_APRENDIDAS.md, RESULTADOS_OBJETIVO.md, CONOCIMIENTO_ACUMULADO.md,
ENTREGABLES.md, BITACORA.md y AGENTES_SUGERIDOS.md + CONSEJO_ASESOR.md. Léelos
antes de tocar nada. Resultados ejecutados están en outputs/experiments/ y en el
notebook canónico notebooks/strata_canonical.ipynb.

OBJETIVO
Escribir la memoria del TFG en LaTeX, capítulo a capítulo, con rigor de profesor
de series temporales, citas bibliográficas correctas en cada afirmación no
original, voz de estudiante (no de IA), y blindada contra Turnitin (plagio + AI
score). La memoria NO es la presentación: es el documento académico extenso.

ESTRUCTURA OBLIGATORIA
1. Introducción (2 páginas) — problema, motivación, pregunta del TFG, esquema.
2. Estado del arte (2 páginas) — agentes LLM en trading actualmente, literatura
   sobre supervisión runtime, hueco que rellena este TFG. Citas obligatorias.
3. Marco teórico (extenso, 20-25 páginas) — todos los conceptos y la base
   matemática usada. Se muy ordenado, claro y sobre todo SE RIGUROSO. Tengo que controlar todo lo que digo, todo tiene que estar justificado y debes recordar que es un tfg de matematicas. Además si usas tests para lo que sea deberías de desarrollar la teoria de estos tests también aquí. Además de todo lo matemático y todas las decisiones, deberias de explicar conceptos economicos clave como sharpe, etc porque mi tribunal no sabe nada de economia y hay que poner contexto de todo.Cierra
   con una sección que explique el proyecto y que ponga STRATA dentro del marco.
4. Marco práctico. Resultados de cada parte del trabajo y su lectura — nunca pongas cifras copiadas a mano, asegurate que lo que dices es lo que es siempr.
5. Conclusiones generales + trabajo futuro.
6. Bibliografía (BibTeX, formato APA o estilo serie temporal según convenga).

CARPETAS QUE TIENES QUE CREAR
tesis/
  main.tex                  ← documento maestro
  chapters/
    01_introduccion.tex
    02_estado_arte.tex
    03_marco_teorico.tex
    04_marco_practico.tex
    05_conclusiones.tex
  bibliography.bib          ← TODAS las referencias citadas
  figures/                  ← PNG/PDF de outputs/figures/
  tables/                   ← .tex de tablas autogeneradas desde JSON
  preamble.tex              ← paquetes LaTeX, comandos, hyphenation español
tesis_assets/
  estilo_raquel/            ← textos de Raquel (entrada manual obligatoria)
  fuentes_bibliograficas/   ← PDFs de los papers citados (para verificación)
  glosario.md               ← terminología consistente cross-capítulo

AGENTES QUE DEBES CREAR (en .claude/agents/)
Cada agente con su prompt en formato standard del proyecto. Sin subagente
llamando a subagente — la orquestación la hace el hilo principal (lección del
CONSEJO_ASESOR.md). Cada agente devuelve dictamen estructurado.

  @redactor-tesis           opus  Produce prosa académica matemática en español
                                  de estudiante UCM. Entrada: outline + datos
                                  + carpeta tesis_assets/estilo_raquel/. Salida:
                                  borrador .tex de la sección. Prohibido usar
                                  vocabulario AI-typical (lista abajo).
  @estilo-raquel            sonnet Audita match estilístico contra
                                  tesis_assets/estilo_raquel/. Reporta
                                  desviaciones: longitud media de frase,
                                  conectores raros para Raquel, uso de primera
                                  persona, ratio voz pasiva/activa. Bloquea si
                                  desviación > umbral.
  @experto-citas            opus  Verifica que cada afirmación no original
                                  tiene cita. Comprueba que la cita está en
                                  bibliography.bib y que el .bib es válido.
                                  Marca claims sin cita. Audita paráfrasis vs
                                  cita literal (>40 palabras → quote, no
                                  paráfrasis).
  @matematico-formal        opus  Revisa demostraciones, definiciones, notación
                                  matemática. Verifica consistencia entre
                                  símbolos del código y de la memoria. Detecta
                                  errores tipo "alpha+beta<1 sin definir alpha".
  @detector-ia              opus  Simula detector de IA. Busca tells: frases de
                                  longitud uniforme (desv. típica < 5 palabras
                                  → sospechoso), uso de "delve", "moreover",
                                  "in essence", "es importante destacar", "cabe
                                  mencionar", estructura tri-partita repetitiva
                                  (cada párrafo 3 frases). Reporta score
                                  estimado de IA-likeness y devuelve secciones
                                  a reescribir.
  @detector-plagio          sonnet Para cada párrafo, busca similitud semántica
                                  con fuentes citadas y con literatura común.
                                  Marca pasajes que parafrasean demasiado
                                  cerca. Sugiere reformulación.
  @latex-experto            sonnet Compila main.tex, resuelve errores BibTeX,
                                  ajusta tablas/figuras, verifica que el PDF
                                  sale limpio. Optimiza tipografía (escalado de
                                  fórmulas, microtype, hyphenation español).
  @narrativa-coherencia     opus  Verifica que la notación, los símbolos y las
                                  cifras son consistentes cross-capítulo. Si
                                  marco teórico llama sigma_t a la vol GARCH,
                                  marco práctico no puede llamarla v_t.
  @coordinador-redaccion    opus  Orquesta el pipeline. Decide qué agente
                                  invocar, recibe dictámenes, decide si pasa o
                                  retorna a redacción. Lleva el changelog.

VOCABULARIO PROHIBIDO (lista no exhaustiva — añadir más)
"delve", "delving", "moreover", "furthermore", "in essence", "it's worth
noting", "cabe destacar", "es importante mencionar", "en esencia", "por otro
lado" cuando no hay contraste real, "en resumen" al final de cada sección,
"podemos observar que", "se puede afirmar que", "vale la pena señalar",
"abordaremos", "exploraremos" en tono manual. Variar conectores: usar conexión
implícita por orden lógico, no etiquetas explícitas en cada frase.

PROTOCOLO POR CAPÍTULO (workflow obligatorio, sin atajos)
1. @asesor-historico lee BITACORA + outputs/experiments + decisiones para el
   capítulo en cuestión. Reporta material disponible.
2. @coordinador-redaccion produce outline detallado: secciones, subsecciones,
   citas necesarias por sección, figuras/tablas a incluir.
3. @redactor-tesis escribe sección por sección consultando
   tesis_assets/estilo_raquel/ continuamente. Salida: draft .tex.
4. @matematico-formal audita rigor matemático. Bloquea si hay error o
   imprecisión.
5. @experto-citas añade citas, verifica .bib, marca claims huérfanos.
6. @estilo-raquel audita estilo. Bloquea si desviación de patrón.
7. @detector-ia ejecuta scan AI-likeness. Bloquea si score > umbral.
8. @detector-plagio scan similitud. Bloquea si match > 15% sobre cualquier
   fuente.
9. @narrativa-coherencia verifica consistencia con capítulos previos.
10. @latex-experto compila, devuelve PDF limpio.
11. Si todos los gates pasan, capítulo a main.tex.
12. Entrada en BITACORA cerrando el capítulo.

REGLAS DE ESCRITURA (todas obligatorias)
- Español académico de UCM, no traducido del inglés.
- Primera persona singular ocasional ("durante la calibración observé que...")
  cuando refleja experiencia real de Raquel — coherente con el ejercicio
  académico de TFG. No "we" plural mayestático constante.
- Longitud de frase variable: 8-30 palabras, distribución no uniforme.
- Cada cita con su contexto. Citar "Black (1976)" no es lo mismo que "Black
  (1976) demostró que la correlación negativa entre retornos y volatilidad en
  índices agregados...".
- Cifras siempre desde JSON, nunca a mano. Tabla LaTeX autogenerada con script
  que lee outputs/experiments/.
- Fórmulas con notación coherente con el código (sigma_t, mu_k, alpha, beta).
- No "wikipedizar": no listas de viñetas en cuerpo de memoria salvo en el
  apéndice. Prosa académica.
- Cada afirmación cuantitativa con su fuente (JSON o BITACORA o cita).

CRITERIO DE ÉXITO POR CAPÍTULO
- 0 claims sin cita (auditado por @experto-citas).
- @estilo-raquel devuelve "match aceptable" (definir umbral concreto en su
  prompt: e.g., desviación típica de longitud de frase entre 8 y 14, ratio
  primera persona consistente con muestras de Raquel).
- @detector-ia devuelve score < 30% AI-likeness estimado.
- @detector-plagio: 0 matches > 15% con cualquier fuente; 0 paráfrasis cercana.
- @matematico-formal: 0 errores, 0 imprecisiones críticas.
- LaTeX compila sin warnings críticos.
- Páginas dentro del rango fijado por capítulo (Introducción 2, Estado arte 2,
  Marco teórico 15-25, etc.).

PROTOCOLO ANTI-DATA-SNOOPING NARRATIVO
La narrativa no puede inventar resultados. Cada cifra reportada se trazea a
outputs/experiments/<exp>.json o a BITACORA con timestamp. Si una cifra no se
puede trazar, no entra. @narrativa-coherencia bloquea cualquier número sin
fuente.

PRE-REGISTRO Y BITACORA
Cada capítulo se pre-registra en BITACORA antes de escribirse:
"## [YYYY-MM-DD] [Pre-registro redacción] - Capítulo N
 Outline: ...
 Material disponible: ...
 Gates a cumplir: ...
 Tiempo estimado: ..."
Al cerrar, entrada [Cierre redacción] con resultado de gates.

PROTOCOLO
- Rama nueva por capítulo: docs/cap-NN-titulo.
- Commits atómicos: outline / draft / citas / estilo / cierre.
- PR a main solo cuando los 8 gates están verdes.
- NO mergees a main tú. Para y espera revisión de Raquel.

ORDEN DE EJECUCIÓN
Capítulo 3 (Marco teórico) PRIMERO — es el más extenso y todo lo demás se
apoya en su notación. Después Capítulo 4 (Marco práctico). Después Capítulo 1
(Introducción) y 2 (Estado del arte) en paralelo. Conclusiones al final.

DEPENDENCIAS PREVIAS QUE RAQUEL DEBE ENTREGAR ANTES DE EMPEZAR
- tesis_assets/estilo_raquel/ con al menos 10 textos suyos.
- Confirmación del estilo de cita preferido (APA / Vancouver / Chicago).
- Confirmación de la plantilla LaTeX exigida por UCM (si la hay).

DEPENDENCIAS QUE DEBES CREAR
- tesis_assets/fuentes_bibliograficas/ con los PDFs de los papers que se citarán en la tesis. Necesito que aquí crees un documento md con todo los papers DEBEN SER ACADEMICOS (puedes buscar en google academico) que se hagan referencia en el trabajo.

Si alguna dependencia no está, PARA y pídela explícitamente.