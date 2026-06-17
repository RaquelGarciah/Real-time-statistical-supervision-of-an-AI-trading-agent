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

_(vacío — se llenará con las correcciones)_

---

## Énfasis y prioridades
*Qué pesa más o menos, con qué liderar, qué desarrollar y qué acortar.*

_(vacío)_

---

## Estilo y voz
*Longitud de frase, guiones, conectores, primera persona, tono. Se refuerzan con la repetición.*

- **Regla:** No uses la construcción "no es X sino Y" (ni "no es A sino B, y se reporta/lee con esa lectura"). Afirma directamente lo que la cosa ES; si es una limitación o un hallazgo, dilo en positivo y en primera persona.
  - Estilo · Prioridad media · Visto 1x · 2026-06-18
  - Ejemplo: «No es un fallo de implementación sino un hallazgo en sí mismo, y se reporta en los resultados con esa lectura.» → «Es una limitación del detector sobre este activo, que recojo en los resultados.»
  - Afecta a: redactor-tesis · estilo-raquel

- **Regla:** Elimina los meta-comentarios sobre la propia redacción ("Insistimos en esta distinción a lo largo de la memoria", "conviene subrayar/recordar/enunciar que", "como veremos"). Enuncia la idea una sola vez, directa; si se va a repetir, que se repita sin anunciarlo.
  - Estilo · Prioridad alta (orden de exclusión: no metas estas frases) · Visto 3x · 2026-06-18
  - Ejemplo: «Insistimos en esta distinción a lo largo de la memoria: STRATA no predice el retorno, sino que mide…» → «STRATA no predice el retorno: mide en qué grado…»
  - Afecta a: redactor-tesis · estilo-raquel

- **Regla:** No repitas la coletilla de causalidad ("en ningún paso entra información posterior a $t$", "no usa observaciones futuras", "para no incurrir en fuga") cada vez que aparece un detector o una variante. La causalidad se establece UNA vez en su sitio (la prueba del filtrado y la sección de validación sin fuga); después se da por sentada salvo que aporte algo nuevo.
  - Estilo · Prioridad media · Visto 4x · 2026-06-18
  - Ejemplo: «…trabaja solo con el posterior filtrado y con la convención causal $w_t\cdot r_{t+1}$; en ningún paso entra información posterior a $t$.» → «…trabaja solo con el posterior filtrado del régimen.»
  - Afecta a: redactor-tesis · estilo-raquel · rigor-matematico

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

- **Regla:** No definas el mismo objeto dos veces seguidas con otras palabras (definición + "es decir" + reformulación que no añade nada). Da una sola definición operativa; si hace falta la lectura intuitiva, intégrala en esa frase, no en una segunda paralela. (Es **redundancia local**: no decir lo mismo dos veces seguidas.)
  - Claridad · Prioridad media · Visto 2x · 2026-06-18
  - Ejemplo: «…$\gammaf(k)$ condiciona solo sobre $x_{1:t}$… No usa ninguna observación posterior a $t$. La recursión… sin que intervenga $x_{t+1},\dots$» → (quitar "No usa ninguna observación posterior a $t$.").
  - Ejemplo: «el RAM score es la masa de probabilidad… El score vive en $[0,1]$…: es la probabilidad… de que el mercado esté en dirección que contradice la apuesta.» → «El score vive en $[0,1]$: cuanto más alto, más probabilidad pone el HMM en que el agente apuesta contra el régimen.»
  - Afecta a: redactor-tesis · narrativa-coherencia

---

## Citas
*Preferencias de citación.*

_(vacío)_

---

### Histórico de cambios de este fichero
- 2026-06-17 · creación (estructura vacía; sistema de aprendizaje montado).
