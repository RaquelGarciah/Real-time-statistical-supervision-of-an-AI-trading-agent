# Correcciones aprendidas — preferencias de Raquel (memoria viva)

> **Fichero canónico de reglas.** Lo leen SIEMPRE `redactor-tesis` y los gates antes de redactar/auditar, para
> no repetir fallos que Raquel ya corrigió. Una regla entra aquí **solo tras la aprobación de Raquel**
> (las propone el agente `aprendiz-correcciones` a partir de `memoria/correcciones/capN.md`).
>
> **Cómo se usa:** el redactor aplica estas reglas al escribir; cada gate aplica las de su categoría. Si una
> regla nueva contradice a una vieja, gana la más reciente (y se actualiza la vieja).
>
> **Aplicación hacia atrás:** cuando entra una regla nueva, el agente `barrido-retroactivo` revisa **toda la
> prosa ya escrita** y propone arreglar dónde más se incumple, para que la corrección no valga solo de ahí en
> adelante sino también para lo ya redactado.
>
> **Formato de cada regla:**
> ```
> - **Regla:** <enunciado operativo, imperativo>
>   - Categoría · Prioridad (alta/media/tentativa) · Visto Nx · 2026-..
>   - Ejemplo: «antes» → «después»
>   - Afecta a: redactor-tesis | estilo-raquel | rigor-matematico | narrativa-coherencia | arquitecto-estructura
> ```
> Prioridad: **alta** = cumplir siempre (exclusiones, énfasis, rigor, cifras); **media** = patrón confirmado
> (visto ≥2-3); **tentativa** = visto 1 vez en estilo (aplicar con criterio, se confirmará con más datos).

---

## NO mencionar (exclusiones) — prioridad ALTA siempre
*Cosas que Raquel NO quiere que aparezcan, o encuadres que prohíbe. Una sola señal basta para que entren.*

- **No uses «(hechos) estilizados».** A Raquel le suena a IA. Sustituye por «regularidades empíricas» (fiel y ya usado en cap. 3) o, en su defecto, «regularidades observadas» / «hechos empíricos». (Nota: «hechos estilizados»/*stylized facts* es término estándar de econometría —Cont 2001—, pero queda excluido por preferencia de la autora.)
  - Exclusión · Prioridad alta · Visto 1x · 2026-06-18
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia

---

## Énfasis y prioridades
*Qué pesa más o menos, con qué liderar, qué desarrollar y qué acortar.*

_(vacío)_

---

## Estilo y voz
*Longitud de frase, guiones, conectores, primera persona, tono. Se refuerzan con la repetición.*

- **Regla:** No uses la construcción "no es X sino Y" (ni "no es A sino B, y se reporta/lee con esa lectura"). Afirma directamente lo que la cosa ES; si es una limitación o un hallazgo, dilo en positivo y en primera persona.
  - Estilo · Prioridad alta · Visto 2x · 2026-06-18
  - Ejemplo: «No es un fallo de implementación sino un hallazgo en sí mismo, y se reporta en los resultados con esa lectura.» → «Es una limitación del detector sobre este activo, que recojo en los resultados.»
  - Ejemplo (cap. 2): «una capa de auditoría que no reemplaza al agente, sino que lo corrige…» → «una capa de auditoría que corrige al agente cuando…»
  - Afecta a: redactor-tesis · estilo-raquel

- **Regla:** Elimina los meta-comentarios sobre la propia redacción ("Insistimos en esta distinción a lo largo de la memoria", "conviene subrayar/recordar/enunciar que", "cabe destacar/mencionar", "es importante señalar", "es esencial que", "como veremos"). El énfasis va DENTRO de la frase ("lo decisivo es", "esto subraya", "merece atención"), nunca anunciándolo. Enuncia la idea una sola vez, directa; si se va a repetir, que se repita sin anunciarlo. El andamiaje no es solo verbal: quita también los cross-refs `\ref` puestos de pasada ("se justifica en detalle en la Sección~X") cuando no aportan en ese punto.
  - Estilo · Prioridad alta (orden de exclusión: no metas estas frases) · Visto 6x · 2026-06-18
  - Ejemplo: «Insistimos en esta distinción a lo largo de la memoria: STRATA no predice el retorno, sino que mide…» → «STRATA no predice el retorno: mide en qué grado…»
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia
  - Lista completa de prohibidos y sus reformulaciones: `tesis_assets/conectores_raquel.md` (sección «Prohibidos»).

- **Regla:** No abras un bloque con una frase de marco grandilocuente ("Las tres líneas anteriores se rozan, pero no se han combinado del modo que aquí se propone", "Esa es la propuesta del trabajo") ni lo cierres con un meta-comentario sobre el estatus de lo que vas a demostrar ("Hasta qué punto se consigue no se adelanta aquí como propiedad demostrada"). Entra directo a la idea, con ritmo de frase variado (corta + larga), y deja que los resultados hablen en su sitio.
  - Estilo / anti-IA · Prioridad media · Visto 1x · 2026-06-18
  - Ejemplo: «Las tres líneas anteriores se rozan, pero no se han combinado del modo que aquí se propone. … Esa es la propuesta del trabajo: … Hasta qué punto se consigue … no se adelanta aquí como propiedad demostrada.» → «Ninguno de los tres enfoques combina supervisión de régimen, coherencia temporal y volatilidad sobre un mismo agente. STRATA añade esa capa de auditoría…»
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia

- **Regla:** No repitas la coletilla de causalidad ("en ningún paso entra información posterior a $t$", "no usa observaciones futuras", "para no incurrir en fuga") cada vez que aparece un detector o una variante. La causalidad se establece UNA vez en su sitio (la prueba del filtrado y la sección de validación sin fuga); después se da por sentada salvo que aporte algo nuevo.
  - Estilo · Prioridad media · Visto 4x · 2026-06-18
  - Ejemplo: «…trabaja solo con el posterior filtrado y con la convención causal $w_t\cdot r_{t+1}$; en ningún paso entra información posterior a $t$.» → «…trabaja solo con el posterior filtrado del régimen.»
  - Afecta a: redactor-tesis · estilo-raquel · rigor-matematico

- **Regla:** Variedad de conectores. Ningún conector se repite dos veces seguidas en el mismo párrafo; rota sinónimos del banco `tesis_assets/conectores_raquel.md`. Si quieres "además" otra vez, la segunda usa "asimismo", "junto con" o "a esto se añade".
  - Estilo · Prioridad alta · Visto 1x (directiva de Raquel) · 2026-06-18
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia

- **Regla:** Conexión implícita siempre que se pueda. El orden lógico bien construido no necesita etiqueta; no antepongas "por tanto/además/en consecuencia" si la relación ya se entiende.
  - Estilo · Prioridad alta · Visto 1x (directiva de Raquel) · 2026-06-18
  - Ejemplo: «…implica correlación negativa entre retornos y volatilidad. Por tanto, STRATA explota esta asunción.» → «…implica correlación negativa entre retornos y volatilidad. STRATA explota esta asunción.»
  - Afecta a: redactor-tesis · estilo-raquel

- **Regla:** Densidad razonable de conectores. Ni etiqueta en cada frase (relleno) ni saltos sin puente lógico; conecta donde el lector lo necesita.
  - Estilo · Prioridad media · Visto 1x (directiva de Raquel) · 2026-06-18
  - Afecta a: redactor-tesis · estilo-raquel

- **Regla:** Registro académico en primera persona del **plural** por defecto ("calibramos", "denotamos", "fijamos", "consideremos", "tomemos"), coherente en todos los capítulos. El **impersonal con "se"** ("se obtiene", "se calcula", "se verá") está permitido, pero **solo en las ocasiones en que de verdad lo pida la coherencia o suene mejor**, no como alternativa libre ni por defecto: ante la duda, plural. Evita la primera persona del **singular** (a lo sumo una admisión personal muy puntual) y el futuro burocrático vacío. (Actualiza 2026-06-18: la decisión previa era "plural uniforme, sin se"; Raquel admite el "se" con moderación, donde quede más coherente.)
  - Estilo / voz · Prioridad alta · Visto 2x (directiva de Raquel) · 2026-06-18
  - Ejemplo: «Aquí basta con dejarla fijada» → «Y así nos aseguramos…» (plural por defecto); el "se" se reserva para cuando el sujeto plural cargaría la frase o rompería el hilo.
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia
  - Pendiente de propagación: el "recojo" de `03_marco_teorico.tex` (§GSO) y de `tesis_assets/estilo_raquel/cap3_frases_corregidas.md` → plural.

- **Regla:** Frase corta y en registro directo. Cuando dos frases consecutivas establecen una consecuencia ("esto garantiza X. A partir de ahora, Y"), fúndelas en una sola más corta; recorta el andamiaje que solo administra al lector ("aquí basta con dejarla fijada", "salvo que se indique lo contrario"). Prefiere lo económico y hablado a lo completo y formal.
  - Estilo / voz · Prioridad tentativa · Visto 1x · 2026-06-18
  - Ejemplo: «Esta convención se justifica en detalle en la Sección~X… Aquí basta con dejarla fijada: cuando más adelante se escriba un producto entre una posición y un retorno, el retorno será $\rnext$ salvo que se indique lo contrario.» → «Y así nos aseguramos la ausencia de fuga temporal; de aquí en adelante, todo producto entre una posición y un retorno usará $\rnext$.»
  - Afecta a: redactor-tesis · estilo-raquel
  - Nota: el arranque coloquial ("Y así…") y la coma de empalme son licencias de ritmo PUNTUALES de la autora, no normas a propagar; en muestras se normaliza a punto y coma.

- **Regla:** Vocabulario llano, el que usaría Raquel. Evita palabras eruditas o rebuscadas aunque sean técnicamente correctas; prefiere el término simple. Si una palabra culta no aporta precisión que el lector necesite, cámbiala. **Al reescribir, baja también el registro del vocabulario, no solo reordenes la frase.**
  - Estilo / voz · Prioridad alta (directiva de Raquel: «ten cuidado con eso») · Visto 1x · 2026-06-18
  - Palabra vetada: «parsimoniosa» (usar «sencilla» / «con menos parámetros»). Ejemplo: «la especificación más parsimoniosa» → «la versión más sencilla, la que consigue esa persistencia con menos parámetros».
  - Afecta a: redactor-tesis · estilo-raquel · detector-ia

---

## Rigor matemático
*Imprecisiones o afirmaciones más fuertes de lo sostenible que Raquel corrigió.*

_(vacío)_

---

## Cifras y datos
*Errores de cifras o trazabilidad a JSON.*

_(vacío)_

---

## Estructura y claridad
*Orden, dependencias, exposición.*

- **Regla:** No digas lo mismo dos veces seguidas. Cubre dos casos: (a) definir el mismo objeto dos veces con otras palabras (definición + "es decir" + reformulación que no añade nada) — da una sola definición operativa; (b) dos frases consecutivas que dicen lo mismo — borra la peor y conserva la más natural. Si hace falta la lectura intuitiva, intégrala en la frase, no en una segunda paralela. (Es **redundancia local**.)
  - Claridad · Prioridad alta · Visto 3x · 2026-06-18
  - Ejemplo: «…$\gammaf(k)$ condiciona solo sobre $x_{1:t}$… No usa ninguna observación posterior a $t$. La recursión… sin que intervenga $x_{t+1},\dots$» → (quitar "No usa ninguna observación posterior a $t$.").
  - Ejemplo: «el RAM score es la masa de probabilidad… El score vive en $[0,1]$…: es la probabilidad… de que el mercado esté en dirección que contradice la apuesta.» → «El score vive en $[0,1]$: cuanto más alto, más probabilidad pone el HMM en que el agente apuesta contra el régimen.»
  - Afecta a: redactor-tesis · narrativa-coherencia

- **Regla:** Nombra el objeto concreto y su parámetro en vez de una etiqueta vaga. Si STRATA devuelve tres scores, escríbelos $(\mathrm{RAM}_t, \mathrm{PSA}_t, \mathrm{GSO}_t)$, no "señales de supervisión"; si usas una volatilidad, di "volatilidad realizada $\mathrm{RV}_t$ a 21 días", no "una medida de volatilidad". La etiqueta genérica solo vale cuando el detalle aún no se ha introducido y nombrarlo distraería.
  - Claridad / concreción · Prioridad media · Visto 2x · 2026-06-18
  - Ejemplo: «$\longmapsto$ señales de supervisión» → «$\longmapsto (\mathrm{RAM}_t,\ \mathrm{PSA}_t,\ \mathrm{GSO}_t)^\top$».
  - Ejemplo: «el log-retorno y una medida de volatilidad» → «el log-retorno y la volatilidad realizada $\mathrm{RV}_t$ con ventana de 21 días».
  - Afecta a: redactor-tesis · estilo-raquel · rigor-matematico

---

## Citas
*Preferencias de citación.*

- **Regla:** Antes de descartar una cita por "no fiable", **verifícala** (arXiv/DOI/venue): varias que sonaban dudosas resultaron reales. Si tras verificar la fuente NO se sostiene (no existe, sin venue creíble, no localizable), **no la cites**: degrada la afirmación a una mención genérica ("existen métodos de *runtime enforcement* para agentes", "se han propuesto marcos de fiabilidad") sin atribuir nada concreto a nadie. Antes una mención sin cita que una cita frágil que el tribunal pueda tumbar. Esto **no exime** de citar lo que sí tiene fuente sólida (regla `ESTILO_Y_ANTIIA.md §3`): solo regula qué hacer cuando la fuente es dudosa.
  - Citas · Prioridad alta · Visto 1x · 2026-06-18
  - Ejemplo: «AgentSpec propone… \cite{wang2025agentspec}. …marcos de fiabilidad… \cite{flehmig2025reliability}.» → «Existen métodos de *runtime enforcement* y marcos de fiabilidad para agentes, todavía dispersos y sin un estándar consolidado.» (caso real: estas dos resultaron verificables; el criterio es verificar primero.)
  - Afecta a: redactor-tesis · experto-citas · narrativa-coherencia

---

### Histórico de cambios de este fichero
- 2026-06-17 · creación (estructura vacía; sistema de aprendizaje montado).
- 2026-06-18 · cap. 2: nueva regla de Citas (verificar antes de descartar; mención genérica si la fuente no se sostiene); nueva regla anti-IA (apertura grandilocuente / cierre meta); refuerzos: «no es X sino Y» media→alta (2x), redundancia local media→alta (3x, enunciado ampliado), meta-comentarios 4x→5x.
- 2026-06-18 · cap. 3 (ediciones directas de Raquel): nueva regla de concreción (nombrar el objeto, no etiqueta vaga; media, 2x); nueva regla de primera persona del plural uniforme (decisión de Raquel; supersede el "recojo" singular); nueva regla de frase corta/hablada (tentativa); refuerzo meta-comentarios 5x→6x (incluye quitar cross-refs `\ref` puestos de pasada). Arranque "Y así" y coma de empalme = licencias puntuales, no reglas.
