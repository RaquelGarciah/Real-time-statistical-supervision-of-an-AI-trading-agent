# Correcciones de Raquel — Capítulo 3 (Marco teórico)

Registro de las correcciones para el agente `aprendiz-correcciones`. Sesión 2026-06-18 00:38.

---

### [cap3 · §3.5 capa de intervención] estilo
- Original:  "La variante que implementamos, \emph{override-C}, trabaja solo con el posterior filtrado y con la convención causal $w_t \cdot \rnext$; en ningún paso entra información posterior a $t$."
- Corregido: "La variante que implementamos, \emph{override-C}, trabaja solo con el posterior filtrado del régimen."
- Razón:     "El final suena a IA: lo de la causalidad ('en ningún paso entra información posterior a t') lo has repetido muchas veces ya."
- Categoría: estilo

### [cap3 · §3.3 GSO] estilo
- Original:  "No es un fallo de implementación sino un hallazgo en sí mismo, y se reporta en los resultados con esa lectura."
- Corregido: "Es una limitación del detector sobre este activo, que recojo en los resultados."
- Razón:     "Este tipo de frase suena a IA ('no es X sino Y, y se reporta con esa lectura'). No me gusta nada."
- Categoría: estilo

### [cap3 · §2 filtrado] claridad
- Original:  "…condiciona solo sobre $x_{1:t}$, es decir, sobre el pasado y el presente. No usa ninguna observación posterior a $t$. La recursión… $\alpha_t(k)$ se construye… sin que en ningún momento intervenga $x_{t+1}, x_{t+2}, \dots$"
- Corregido: (quitar la frase "No usa ninguna observación posterior a $t$.") "…condiciona solo sobre $x_{1:t}$, es decir, sobre el pasado y el presente. La recursión… $\alpha_t(k)$ se construye… sin que en ningún momento intervenga $x_{t+1}, x_{t+2}, \dots$"
- Razón:     "Dices lo mismo dos veces seguidas con otras palabras: 'no usa observación posterior a t' y 'sin que intervenga x_{t+1}…' son lo mismo."
- Categoría: claridad

### [cap3 · §3.2 RAM score] claridad
- Original:  "el \emph{RAM score} es la masa de probabilidad filtrada sobre los regímenes en los que la acción del agente es incoherente… El score vive en $[0,1]$ y tiene una interpretación probabilística directa: es la probabilidad, según el HMM, de que el mercado esté en una dirección que contradice la apuesta."
- Corregido: (no redefinir el score) "El score vive en $[0,1]$: cuanto más alto, más probabilidad pone el HMM en que el agente apuesta contra el régimen."
- Razón:     "Defines el score dos veces seguidas con otras palabras."
- Categoría: claridad

### [cap3 · §1] estilo
- Original:  "Insistimos en esta distinción a lo largo de la memoria: STRATA no predice el retorno, sino que mide en qué grado la decisión del agente es coherente con el estado del mercado."
- Corregido: "STRATA no predice el retorno: mide en qué grado la decisión del agente es coherente con el estado del mercado."
- Razón:     "'Insistimos en esta distinción a lo largo de la memoria' es repetir que voy a repetir; suena a IA y además repite el 'supervisa' de antes."
- Categoría: estilo

---

## Ediciones directas de Raquel (autoría suya, sesión 2026-06-18 16:40) — muestra de voz

> No son correcciones a mi prosa: son frases que Raquel reescribió ella misma. Las guardo para imitar su voz.

### [cap3 · §convención temporal] estilo / voz
- Original (versión previa):  "Esta convención (la posición de $t$ multiplica al retorno de $t+1$) se justifica en detalle en la Sección~\ref{sec:cpcv} como condición de ausencia de fuga temporal. Aquí basta con dejarla fijada: cuando más adelante se escriba un producto entre una posición y un retorno, el retorno será $\rnext$ salvo que se indique lo contrario."
- Reescrito por Raquel: "Y así nos aseguramos la ausencia de fuga temporal, de aquí en adelante, todo producto entre una posición y un retorno usará $\rnext$."
- Señales de voz: (a) **mucho más corta**, funde dos frases en una; (b) abre con conjunción **"Y así"** (arranque conversacional, no formal); (c) **primera persona del plural** ("nos aseguramos"); (d) **elimina el andamiaje meta** ("Aquí basta con dejarla fijada", "salvo que se indique lo contrario") y el cross-ref \ref que no necesitaba ahí; (e) tolera la **coma de empalme** por ritmo. Prefiere directo y hablado sobre completo y formal.
- Categoría: estilo / voz

### [cap3 · §3.1 mapa de STRATA] énfasis / concreción
- Original:  salida del mapa como "$\longmapsto \text{señales de supervisión}$".
- Reescrito por Raquel: salida explícita como vector columna de los tres scores, $\longmapsto (\mathrm{RAM}_t,\ \mathrm{PSA}_t,\ \mathrm{GSO}_t)^\top$.
- Señales de voz: prefiere **mostrar el objeto concreto** (los tres scores que devuelve STRATA) antes que una etiqueta vaga ("señales de supervisión").
- Categoría: énfasis / concreción
- NOTA TÉCNICA (no de estilo): la sintaxis LaTeX que tecleó está rota y NO compila — escribió "[mathrm{…}" en vez de "\[\mathrm{…}" y "\ " en vez de "\\" entre filas del pmatrix. Pendiente de arreglar cuando ella lo pida (no toco ahora).

### [cap3 · §2 emisiones] énfasis / concreción
- Original:  "el log-retorno y una medida de volatilidad".
- Reescrito por Raquel: "el log-retorno y la volatilidad realizada $\mathrm{RV}_t$ con ventana de $21$ días".
- Señales de voz: nombra la cantidad concreta y su parámetro (RV, 21 días) en lugar de la perífrasis vaga. Coherente con la edición del mapa: concreción sobre etiqueta.
- Categoría: énfasis / concreción
