# Sesión con el consejero — 2026-06-24

<!-- sesión: ff5c8ffd-69dd-47e8-9801-e6fc5a53fc28 · registro automático (hook) + hitos (consejero) -->

Comenta con `> COMENTARIO:`, `> AYUDA:` o `> DUDA:` donde quieras (ver `../README.md`).

### Hitos

_(el consejero anota aquí decisiones, anotaciones y agentes llamados)_

---

## Conversación

#### [1717] Consejero

I'll load the context now, starting with the live sources.

#### [1718] Consejero

The output-style spec is itself outdated — it points at SMCI as the case. The live MANUAL (updated today) and the newest notebook (`STRATA_marco_practico.ipynb`, built today 06:28) override it: the case is now **SPY + panel de 10**. Per the spec's own rule (gana el notebook/MANUAL), I'll align with that. Let me ground on the current source of truth.

#### [1718] Consejero

Contexto cargado. Aviso importante de entrada: **la spec del consejero (`consejero-tesis.md`) y el comando de arranque siguen anclados a SMCI, pero ya no es el caso central.** El MANUAL (actualizado hoy) y el notebook vivo `STRATA_marco_practico.ipynb` (regenerado hoy 06:28) mandan, y dicen: caso central **SPY** + **panel de 10** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG); SMCI es uno de los diez. Fuente de verdad de la redacción: `MARCO_PRACTICO_CONTEXTO.md`. Me alineo con eso.

Cifras ancla que confirman que estoy en la página correcta: en SPY el agente solo (M5) acierta **0,366** (Sharpe −3,07, equity 0,70×) y STRATA lo **rescata** —ΔSharpe pooled-10 **M8 +0,60 / M10 +1,12 / AutoML +1,08**, cuota SHAP STRATA **0,66**, ley del leverage sobre los 10 (p=0,093).

Puedo: **revisar** tus `.tex` sección a sección con anotaciones `% >>> CONSEJERO` (tú decides qué aplicas), **responder dudas** de contenido/decisiones citando la fuente exacta, y **orquestar los subagentes** (siempre pidiéndote luz verde antes). Las cifras solo salen de los JSON / el notebook / `RESULTADOS_OBJETIVO §1`, nunca de memoria.

¿Por dónde empezamos? Si quieres, reviso un capítulo concreto del `main_v2` (veo que estás trabajando la v2: intro, estado del arte, cap.3, cap.4 SPY, panel, conclusiones) o resuelvo una duda puntual.

#### [1723] Raquel

Arquitectónicamente, STRATA instancia la lógica del \emph{meta-learner} \cite{lopezdeprado2018}: un modelo secundario que filtra o atenúa las señales del primario. Lo nuevo es trasladar esa lógica a un agente LLM multipersona y hacerla falsable con contrastes explícitos. Se diferencia así del equipo de riesgo de TradingAgents \cite{xiao2024tradingagents}, un agente LLM deliberativo dentro del propio sistema que comparte su información y sus sesgos. STRATA es exógeno: su intervención depende de los detectores estadísticos, no de lo que el agente diga. REESCRIBE ESTO NO SE SI ESTÁ ENTENDIENDO BIEN LO QUE ES STRATA

#### [1723] Consejero

Raquel, tu instinto es correcto: ese párrafo **confunde STRATA con su derivada**. Déjame ver el contexto exacto antes de reescribir.

#### [1724] Consejero

Diagnóstico (y por qué tu instinto acierta):

El párrafo dice *«STRATA instancia la lógica del meta-learner: un modelo secundario que filtra o atenúa»*. Eso **confunde STRATA con su derivada** y cruza la línea roja del MANUAL §8: *«STRATA supervisa; las derivadas (M8/M10/AutoML) predicen. No confundir la capa de señales con su explotación.»* STRATA no es *un modelo secundario* — es la **capa de tres detectores** (RAM/PSA/GSO) que produce señales. El *meta-learner* propiamente dicho es **M10**, que **consume** esas señales.

Hay además una **incoherencia interna**: cuatro líneas más abajo (línea 25) escribes *«el valor de un meta-learner libre debe residir en las señales de STRATA»* — ahí el meta-learner **usa** STRATA. La línea 23 dice que STRATA **es** el meta-learner. No pueden ser las dos.

El paralelismo correcto con López de Prado existe, pero es a nivel de **arquitectura**: una etapa secundaria que decide si actuar sobre un primario. STRATA recoge esa idea y la lleva más lejos (su intervención fuerte llega a voltear el signo, más allá del meta-labeling, que solo modula el tamaño).

Lo aplico:

#### [1725] Consejero

Hecho. Qué cambió y por qué:

- **STRATA = la capa de detectores**, no un meta-learner. El meta-labeling de López de Prado entra ahora como **paralelismo arquitectónico** (una etapa secundaria sobre un primario), no como identidad.
- **Resuelve la incoherencia** con la línea 25: aquí STRATA es el supervisor; allí el meta-learner libre consume sus señales. Ya no se contradicen.
- *«la lleva más lejos»* recoge que la intervención fuerte voltea el signo, cosa que el meta-labeling clásico no hace (solo modula tamaño) — defendible ante el tutor.
- Evité el patrón prohibido *«no es X sino Y»* y dejé los conectores variados, en tu voz.

Un aviso aparte, no lo toco sin que me lo pidas: los párrafos de **motivación (líneas 7–9)** tienen erratas que sí conviene corregir — *«inperceptibles»*, *«todo lo que viviendo ha pasado»*, *«no dejan de sers istemas no determinista»*. ¿Te los reviso también, o seguimos con otra cosa?

#### [1726] Raquel

La hipótesis que ponemos a prueba es falsable: filtrar o atenuar las decisiones del agente con detectores estadísticos clásicos lo rescata de forma significativa cuando pierde dinero y acierta la dirección menos de la mitad de las veces. La falsamos en tres niveles. El rescate debe ser significativo en un test de McNemar pareado; la atribución del P\&L a cada detector debe identificar de dónde viene la corrección; y el valor de un meta-learner libre debe residir en las señales de STRATA, lo que se comprueba si se apoya en ellas y se degrada al quitarlas. La regla \texttt{prior-flip}, fijada antes de mirar los resultados, marca de antemano cuándo concluiríamos que la técnica no funciona.

El rigor matemático es la aportación del trabajo. El rescate se prueba en acierto direccional y en riesgo, y el aprendiz redescubre las señales de STRATA como las informativas. STRATA rescata al agente que pierde y deja ver la mecánica de cada corrección. No supera de forma significativa a una posición pasiva: la rentabilidad absoluta frente al mercado queda como línea futura.  REDUCE LO MAXIMO QUE PUEDAS, ME PARECE UN POCO RELLENO NO PIERDAS MI ESTILO

#### [1727] Consejero

Tienes razón, la segunda mitad repite lo que ya dicen la hipótesis y los objetivos. Recorto y dejo solo lo que aporta: los tres niveles, el `prior-flip`, y los dos remates canónicos (rigor = la aportación; no bate al pasivo). Conservo tu frase de apertura tal cual.

#### [1727] Consejero

Recortado: de ~135 a ~85 palabras. Qué quité y qué dejé:

- **Fuera (relleno):** *«El rescate se prueba en acierto direccional y en riesgo, y el aprendiz redescubre las señales»* (es ya el nivel 1 y 3 de la falsación) y *«STRATA rescata al agente y deja ver la mecánica»* (lo dicen la hipótesis y los objetivos).
- **Dentro:** los tres niveles (ahora en una sola frase con dos puntos, tu ritmo), el `prior-flip`, y los dos remates canónicos — **rigor = la aportación** (tu frase original intacta) y **no bate al pasivo / alfa = línea futura** (el caveat honesto del MANUAL, que no se toca).
- Comprimí *«ser significativo en un test de McNemar»* → *«salir significativo en un McNemar»* y *«concluiríamos que la técnica no funciona»* → *«diríamos que no funciona»*. Sin perder nada defendible.

¿Sigo con las erratas de la motivación (líneas 7–9), o lo dejamos aquí?

#### [1731] Raquel

La falsamos en tres niveles: el rescate debe salir significativo en un McNemar pareado; la atribución del P\&L debe decir de qué detector viene la corrección; y un meta-learner debe apoyarse en las señales de STRATA y degradarse al quitarlas.  no puedes dar las conclusiones en la introducción

#### [1731] Consejero

Cierto. Tal y como está, *«debe salir significativo / debe apoyarse / degradarse»* anuncia el resultado esperado: eso es conclusión, no introducción. En la intro toca decir **qué contrastamos**, no **qué saldrá**. Confirmo el estado exacto de la línea (veo que quitaste «libre») y lo reescribo neutro.

#### [1731] Consejero

Vale, también convertiste la hipótesis en pregunta. Toco solo los tres niveles, dejando lo demás como lo tienes:

#### [1732] Consejero

Hecho. Ahora los tres niveles nombran **los instrumentos de contraste**, no su desenlace:

- *«el rescate debe salir significativo»* → *«un McNemar pareado entre el agente supervisado y el agente solo»* (la prueba, no el veredicto).
- *«debe decir de qué detector viene»* → *«la atribución del P\&L a cada detector»*.
- *«debe apoyarse… y degradarse al quitarlas»* → *«el papel de las señales de STRATA dentro de un meta-learner»* (deja la pregunta abierta; el resultado va al cap. 4).

El `prior-flip` se queda: es pre-registro, no conclusión. Y encaja con tu hipótesis ya en forma de pregunta — planteas la duda y dices con qué la resolverás, sin adelantar la respuesta.

#### [1959] Raquel

como acorto esto deja solo lo importante, pasamelo por aqui
\subsubsection{Estimación por máxima verosimilitud}
\label{sec:garch-mv}

Los parámetros $\theta = (\omega, \alpha, \beta, \nu)$ se estiman por máxima verosimilitud condicional. Sea $f_\nu$ la densidad de la $t$ de Student estandarizada de $\nu$ grados de libertad. La log-verosimilitud condicional de la muestra $\epsilon_{1:T}$, dado un valor inicial $\sigma_1^2$, es la suma de las contribuciones diarias
\[
\ell(\theta) \;=\; \sum_{t=1}^{T} \left[\, \log f_\nu\!\left(\frac{\epsilon_t}{\sigmat}\right) - \log \sigmat \,\right],
\]
donde cada $\sigmat$ se obtiene recorriendo la recursión \eqref{eq:garch11} con los parámetros $\theta$ en evaluación y el término $-\log\sigmat$ es el jacobiano del cambio de variable de $\eta_t$ a $\epsilon_t$. La derivación de la verosimilitud, a partir de que $\epsilon_t = \sigmat\eta_t$ es un cambio de escala de $\eta_t$ por el factor $\sigmat$ conocido en $t-1$, está en el Anexo~\ref{anx:garch-mv}. No existe solución cerrada para el máximo, así que se maximiza $\ell(\theta)$ numéricamente sobre la región admisible $\omega>0$, $\alpha,\beta\ge 0$, $\alpha+\beta<1$, $\nu>2$.

El estimador resultante hereda las buenas propiedades de la máxima verosimilitud bajo condiciones de regularidad. Bajo el modelo correctamente especificado, la condición de estacionariedad del Teorema~\ref{teo:garch-estac} y condiciones adicionales de regularidad (en particular, momentos suficientes de las innovaciones, $\nu>4$ para el cuarto momento, y un óptimo interior a la región admisible), el estimador máximo-verosímil $\hat\theta$ es consistente y asintóticamente normal, con matriz de covarianzas el inverso de la información de Fisher. La prueba de estas propiedades para el GARCH no figura en los trabajos que introducen el modelo, sino en su tratamiento posterior, al que se remite \cite{francqzakoian2019}. Una cautela: la normalidad asintótica de $\hat\alpha$ y $\hat\beta$ se degrada cuando $\alpha+\beta$ se aproxima a la frontera $1$, justo el régimen IGARCH discutido arriba, lo que tenemos presente al interpretar los intervalos de confianza de los parámetros estimados.

#### [2000] Consejero

No lo toco en el fichero; te lo dejo aquí para que lo pegues tú. Recorto de ~230 a ~115 palabras, conservando lo que el tutor (series temporales) querrá ver: el objetivo de verosimilitud, que se maximiza numéricamente en la región admisible, las propiedades asintóticas con su cita, y la cautela IGARCH.

```latex
\subsubsection{Estimación por máxima verosimilitud}
\label{sec:garch-mv}

Los parámetros $\theta = (\omega, \alpha, \beta, \nu)$ se estiman por máxima verosimilitud condicional. Con $f_\nu$ la densidad de la $t$ de Student estandarizada de $\nu$ grados de libertad, la log-verosimilitud de la muestra $\epsilon_{1:T}$, dado un valor inicial $\sigma_1^2$, suma las contribuciones diarias
\[
\ell(\theta) \;=\; \sum_{t=1}^{T} \left[\, \log f_\nu\!\left(\frac{\epsilon_t}{\sigmat}\right) - \log \sigmat \,\right],
\]
donde cada $\sigmat$ sale de la recursión \eqref{eq:garch11} y el término $-\log\sigmat$ es el jacobiano del cambio de variable $\eta_t \mapsto \epsilon_t = \sigmat\eta_t$ (derivación en el Anexo~\ref{anx:garch-mv}). No hay solución cerrada: se maximiza $\ell(\theta)$ numéricamente sobre $\omega>0$, $\alpha,\beta\ge 0$, $\alpha+\beta<1$, $\nu>2$.

Bajo especificación correcta, la estacionariedad del Teorema~\ref{teo:garch-estac} y condiciones de regularidad ($\nu>4$ para el cuarto momento y un óptimo interior), $\hat\theta$ es consistente y asintóticamente normal, con matriz de covarianzas el inverso de la información de Fisher \cite{francqzakoian2019}. Esa normalidad se degrada cuando $\alpha+\beta\to 1$ —el régimen IGARCH—, lo que tenemos presente al interpretar los intervalos de confianza de $\hat\alpha$ y $\hat\beta$.
```

Qué quité:
- *«hereda las buenas propiedades de la máxima verosimilitud bajo condiciones de regularidad»* — lo dice ya el «consistente y asintóticamente normal».
- *«La prueba… no figura en los trabajos que introducen el modelo, sino en su tratamiento posterior»* — meta-comentario sobre dónde vive la prueba; basta con citar a Francq-Zakoián. (Además quitaba el patrón *no… sino*.)
- *«momentos suficientes de las innovaciones»* → directamente $\nu>4$.

Mantengo tu *«tenemos presente»* y la mecánica del jacobiano, que es lo que da rigor. Si lo quieres aún más corto puedo fundir las dos frases del segundo párrafo en una.

#### [2031] Raquel

El agente solo (M5) acierta sobre SPY el $0{,}366$ de las direcciones y pierde dinero: ese es el punto de partida. Usaremos este ETF del S\&P~500\ con un emph{leverage effect} de índice amplio que da al régimen contenido introduce en una linea SPY

#### [2031] Consejero

Te lo aplico. Añado la línea que define SPY y de paso arreglo dos erratas de LaTeX que tenías ahí: `emph{...}` sin barra (no iba a renderizar la cursiva) y el `\` suelto tras `S\&P~500`.

#### [2031] Consejero

Hecho. Cambios:

- **Línea nueva de SPY:** *«SPY es el ETF que replica el S\&P~500, el índice de las quinientas mayores empresas cotizadas de Estados Unidos.»* — definición en una frase, antes de invocar el *leverage effect*.
- **Erratas LaTeX:** `emph{leverage effect}` → `\emph{...}`; quitado el `\` suelto tras `S\&P~500`.
- Fundí *«para mostrar el funcionamiento de nuestro sistema»* en *«Lo usamos como caso porque…»*, que ya carga esa función sin repetir.

El resto del párrafo queda igual. ¿Sigo en este capítulo o vuelves a la intro?

#### [2036] Raquel

Tiene un \emph{leverage effect} de índice amplio da al régimen contenido direccional (Sección~\ref{sec:strata-ram}) y nos permite comprobar si nuestra hipotesis sobre el\emph{leverage effect} : --- se cumple en este caso marginal. tenemos definida en algun lado la hipotesis con el leverage effect?

#### [2036] Consejero

Buena pregunta. Déjame ver dónde está formulada la hipótesis del *leverage effect* en la memoria, para que la referencies en vez de redefinirla aquí.

#### [2036] Consejero

Sí, está definida — y conviene distinguir **dos cosas** que tu frase está mezclando:

**1. El *leverage effect* (el fenómeno).** Definido en el **estado del arte**, [02_estado_arte_v2.tex:27](tesis/chapters/02_estado_arte_v2.tex#L27), con su `\label{sec:leverage}` — Black \cite{black1976} y Christie \cite{christie1982}, retorno y volatilidad correlados negativamente, fuerte en índices y débil en valores aislados. Se reexplica en el marco teórico, [03_marco_teorico.tex:583](tesis/chapters/03_marco_teorico.tex#L583).

**2. La *ley del leverage* (la hipótesis).** *«El rescate del aprendiz en accuracy escala con el leverage effect del activo.»* Formulada y contrastada en el **panel**, [05_panel_v2.tex:227](tesis/chapters/05_panel_v2.tex#L227) (`\subsection{La ley del leverage}`): Pearson $r=-0{,}56$, $p=0{,}093$ — tendencia al $\alpha=0{,}10$ pre-registrado. Fuente: `leverage_law_panel10.json`.

**Mi recomendación para tu frase de cap. 4 — dos avisos:**

- **La ley del leverage no se puede comprobar en SPY solo.** Es una correlación **entre activos** (rescate vs. leverage de cada uno); necesita los diez. En un único activo no hay con qué contrastarla — es el trabajo del cap. 5. Aquí solo puedes *anunciar* que SPY es el extremo de leverage fuerte y remitir al panel.
- **SPY no es un "caso marginal".** Es justo lo contrario: el **índice amplio con leverage fuerte**, el caso donde el régimen tiene más contenido direccional. "Marginal" lo serían SMCI/UNG (leverage invertido) o ROKU (leverage casi nulo).

Si quieres, te dejo la frase reescrita en esa línea: SPY como ancla de leverage fuerte, con un `(Sección~\ref{sec:panel-ley})` que apunte a la ley en el panel. ¿La aplico o te la paso por aquí?

#### [2037] Raquel

pasala

#### [2037] Consejero

Aquí la tienes, lista para pegar (sustituye tu frase con el `---`):

```latex
Tiene un \emph{leverage effect} de índice amplio, el extremo fuerte del panel, y por eso el régimen adquiere contenido direccional (Sección~\ref{sec:strata-ram}). Que ese rescate escale con el \emph{leverage} de un activo a otro es una hipótesis entre activos —la ley que contrastamos en el panel (Sección~\ref{sec:panel-ley})—; con un solo activo no hay con qué medir esa relación, así que en SPY enseñamos la mecánica.
```

Qué arregla respecto a tu borrador:

- **SPY = extremo fuerte**, no *«caso marginal»* (eso describe a SMCI/UNG/ROKU).
- **La ley del leverage no se contrasta aquí.** La frase remite a `sec:panel-ley` (el sitio donde sí se mide, con los diez) en lugar de prometer comprobarla en SPY solo.
- Quita el `\emph{leverage effect}` repetido a dos palabras de distancia y el `---` suelto.

Verifica el `\ref`: la subsección de la ley está en [05_panel_v2.tex:227](tesis/chapters/05_panel_v2.tex#L227) y su label es `sec:panel-ley` (lo usa la propia línea 279). Si al compilar sale `??`, dímelo y reviso la etiqueta exacta. ¿Quieres que la deje aplicada en el `.tex` o la pegas tú?

#### [2049] Raquel

La Tabla~\ref{tab:mp-calib} recoge cada decisión de calibración con su criterio. El número de estados del HMM, $K = 3$, sale de una selección de modelo ex ante: la log-verosimilitud held-out y el BIC mejoran al pasar de dos a tres estados y la tabla anota los dos valores. El orden de los tres estados justifica su uso sobre SPY: el más volátil tiene el retorno medio de calibración más bajo y el más tranquilo el más alto. Esa gradación negativa entre volatilidad y retorno vuelve el régimen un proxy de la dirección en un índice amplio (Rabiner 1989 \cite{rabiner1989}); los etiquetamos Calma, Estrés y Crisis. La volatilidad la modelamos con un GARCH(1,1) de innovaciones $t$ de Student, porque los retornos diarios de renta variable tienen más movimientos extremos de los que admite una normal (Bollerslev 1987 \cite{bollerslev1987}). El cambio estructural lo seguimos con BOCPD: el detector mantiene una distribución sobre cuántos días lleva el agente sin cambiar de conducta y la actualiza con cada observación nueva, sin mirar el futuro (Adams y MacKay 2007 \cite{adams2007}). Los tres umbrales son cuantiles de la calibración, que la tabla detalla.

\begin{table}[htbp]
\centering
\caption{Decisiones de calibración de STRATA sobre SPY, con su criterio de fijación. Todas se cierran sobre 2000-01-01 a 2024-09-30, antes de la evaluación. Fuente: \texttt{k\_selection.json}, \texttt{cache/models/strata\_thresholds.json}.}
\label{tab:mp-calib}
\begin{tabular}{lll}
\toprule
Decisión & Valor & Criterio (cita) \\
\midrule
Estados del HMM & $K = 3$ & held-out $-1{,}30 > -1{,}69$; BIC $18\,775 < 24\,131$ \cite{rabiner1989} \\
Volatilidad & GARCH(1,1)-$t$ & colas pesadas de los retornos \cite{bollerslev1987} \\
Cambio estructural & BOCPD \emph{online} & causalidad de la alarma \cite{adams2007} \\
Umbral RAM & $\tau = 0{,}50$ & voto de mayoría del régimen \\
Umbral PSA & $P_{95} = 0{,}023$ & cuantil de calibración \\
Umbral GSO & $P_{95} = 2{,}371$ & cuantil de calibración \\
\bottomrule
\end{tabular}
\end{table}
 ESTO NO ME GUSTA PARECE MUY CHERRY PICKY

#### [2050] Consejero

Te entiendo, y sé exactamente qué huele a cherry-picking aquí. No es la tabla (una tabla de decisiones pre-registradas está bien): es **la prosa**, que presenta dos cosas como si las hubieras elegido *porque* salieron bonitas:

1. *«El orden de los tres estados **justifica su uso sobre SPY**»* — suena circular y post-hoc: como si miraras el resultado (la gradación negativa) y lo usaras para justificar la elección. SPY se eligió **ex ante** por su naturaleza; la gradación es una **propiedad observada** que reportas, no una justificación.
2. *«held-out $-1{,}30 > -1{,}69$»* enseña solo **K=2 vs K=3**. Un tribunal pregunta: *¿y K=4? ¿K=5?* Si solo enseñas la comparación que gana, parece elegida.

Déjame ver si en el JSON tienes más valores de K, porque si probaste 4 y 5 y no mejoran, eso **mata** la acusación de raíz.

#### [2051] Consejero

El JSON solo contiene **K=2 y K=3** (`ks: [2, 3]`), así que no puedo apoyarme en "probé 4 y 5". Pero el propio criterio del JSON es *«held-out + ocupación + interpretabilidad»*, y ahí está la salida honesta: **K=3 no es un óptimo que rebuscaste, es la elección interpretable estándar** (Calma/Estrés/Crisis) confirmada frente al baseline de 2 — y hay un dato que desactiva el cherry-picking: **ningún estado queda casi vacío** (ocupación mínima 0,16), o sea, el tercer régimen es real, no un artefacto.

Reframe del párrafo (la tabla queda igual). Te lo paso:

```latex
La Tabla~\ref{tab:mp-calib} reúne las decisiones de calibración con su criterio, todas cerradas sobre 2000-01-01 a 2024-09-30 antes de ver la evaluación. Fijamos el HMM en $K = 3$ estados (Rabiner 1989 \cite{rabiner1989}) por interpretabilidad: tres regímenes admiten la lectura Calma--Estrés--Crisis habitual en la modelización de cambio de régimen, y la selección de modelo ex ante lo respalda frente al baseline de dos estados, con mejor log-verosimilitud held-out ($-1{,}30$ frente a $-1{,}69$) y mejor BIC ($18\,775$ frente a $24\,131$), sin que ningún estado quede casi vacío (ocupación mínima $0{,}16$). Una vez estimados, los tres estados ordenan por volatilidad, y su retorno medio de calibración cae según sube la volatilidad: el más volátil es el de retorno más bajo y el más tranquilo el más alto. Esa gradación negativa es la firma del \emph{leverage effect} en un índice amplio, y es la que da al régimen contenido direccional. La volatilidad la modelamos con un GARCH(1,1) de innovaciones $t$ de Student, porque los retornos diarios de renta variable tienen más movimientos extremos de los que admite una normal (Bollerslev 1987 \cite{bollerslev1987}). El cambio estructural lo seguimos con BOCPD: mantiene una distribución sobre cuántos días lleva el agente sin cambiar de conducta y la actualiza con cada observación nueva, sin mirar el futuro (Adams y MacKay 2007 \cite{adams2007}). Los tres umbrales son cuantiles de la calibración, que la tabla detalla.
```

Qué cambia y por qué deja de sonar cherry-picky:

- *«El orden de los tres estados justifica su uso sobre SPY»* → la gradación pasa a ser **propiedad observada** que da contenido direccional, no justificación post-hoc de la elección.
- **K=3 se lidera con interpretabilidad** (estándar Calma/Estrés/Crisis), y held-out + BIC quedan como **confirmación**, no como el motivo de la elección.
- Añado **ocupación mínima 0,16** (de `k_selection.json`): prueba que el tercer régimen no es un estado degenerado — el argumento más fuerte contra el cherry-picking, y trazable.
- Moví `\cite{rabiner1989}` a la elección del HMM (donde corresponde) y quité esa cita de la frase del leverage, que se apoya en Black/Christie ya citados en los caps. 2 y 3.

Un aviso honesto: como solo evaluaste K∈{2,3}, si el tribunal insiste en *«¿por qué no 4?»*, la respuesta es *«3 es el número interpretable pre-registrado, y domina al modelo más simple»* — defendible, pero no digas ni sugieras que barriste más K. ¿Lo aplico al `.tex`?

#### [2052] Raquel

La Figura~\ref{fig:mp-regimenes-spy} muestra los tres regímenes pintados sobre el precio de SPY VALE CREO QUE VOY A MOVER DE LA SECCION 3 A ANEXOS LO DEL ALGORITMO DE VITERNI PORQUE SOLO SE USABA PARA PINTAR ESTO Y LO VOY A MENCIONAR AQUÍ

#### [2053] Consejero

Antes de darte el visto bueno, déjame confirmar una cosa clave: **qué decodificación usa RAM de verdad**. Si Viterbi solo pinta la figura y el detector en vivo usa la probabilidad filtrada causal, moverlo a anexos es buena idea. Si RAM usara Viterbi, sería otra historia (Viterbi mira toda la secuencia → look-ahead). Lo verifico en el código.

#### [2053] Consejero

Ojo, Raquel — antes de que muevas: Viterbi **no se usa solo para pintar la figura**. Déjame confirmar el segundo uso, que es el importante.

#### [2054] Consejero

Confirmado, y el matiz importa. Esto es lo que hace el código:

- **RAM en vivo** usa la **probabilidad filtrada causal** `predict_proba_filtered` ([core/hmm.py:178](core/hmm.py#L178)) — forward, fila $t$ depende solo del pasado. **No usa Viterbi.**
- **El prior régimen→dirección** sale del **signo de la media de emisión $\mu_k$** del HMM (lo dice tu propio cap. 3, [03_marco_teorico.tex:583](tesis/chapters/03_marco_teorico.tex#L583)), no de etiquetar días con Viterbi.
- **Viterbi** (`predict_states`, [core/hmm.py:157](core/hmm.py#L157)) aparece en el pipeline canónico para **pintar el camino de regímenes** de la figura ([export_figs_cap4.py:162](experiments/export_figs_cap4.py#L162)).

Tu instinto es correcto: **en el entregable, Viterbi es descriptivo**. Moverlo a anexos y mencionarlo aquí es buena estructura — el cuerpo del cap. 3 se queda con la maquinaria causal (forward), y Viterbi vive donde de verdad se usa.

**Pero aprovecha el movimiento para blindar un flanco de look-ahead.** El trazo de la figura es una decodificación **retrospectiva sobre toda la serie** (Viterbi mira el futuro). Si no lo dices, un tribunal ve "régimen vigente cada día" y supone que el detector conoce el régimen con visión de futuro. Conviértelo en fortaleza diciéndolo explícito. Frase para insertar aquí:

```latex
La Figura~\ref{fig:mp-regimenes-spy} muestra los tres regímenes pintados sobre el precio de SPY. El trazo emplea el camino de máxima verosimilitud de Viterbi (Anexo~\ref{anx:viterbi}), una decodificación retrospectiva sobre toda la serie; el detector RAM no lo usa: en tiempo real lee la probabilidad filtrada causal del estado, que nunca mira el futuro.
```

Dos avisos:
- Crea el `\label{anx:viterbi}` en el anexo cuando muevas el contenido; si no, saldrá `??` al compilar.
- Verifica si `regime_direction_table.json` (fuente de la figura y de las cifras del párrafo siguiente) se construye con `predict_states`. Si es así, di que Viterbi es la **decodificación descriptiva** (figura + tabla de regímenes), no *solo* la figura — para no quedarte corta. ¿Te confirmo cómo se genera ese JSON?

#### [0354] Raquel

con todos los cambios que he hecho ya tienes que saber todo de mi, ya tienes que poder corregir cosas que suenan a ia por ejemplo ( Es el caballo de batalla) expresiones asi nunca las usaria, esas cosas suenan a ia. asi que pofavor hazme una revisión en todos los documentos _v2 para buscar cosas que yo no pondría

#### [0354] Consejero

Sí. Llevas razón: a estas alturas ya tengo tu voz. *«El caballo de batalla»*, las metáforas de manual, los meta-comentarios — fuera. Hago la pasada por todos los `_v2`. Primero cargo tus reglas para no marcar a ojo, luego barro.

#### [0355] Consejero

Tengo tus reglas. Cazo los tells idiomáticos en los siete `_v2` de una pasada — metáforas de manual, meta-comentarios, *«no es X sino Y»*, cierres floridos:

#### [0355] Consejero

Dos confirmados en esta pasada (los de `apilar` son falsos positivos — la `r` de "a-pilar"):
- **03:659** *«Es el caballo de batalla del riesgo en el panel»* — el que señalaste.
- **04:48** *«Que RAM acierte al intervenir no es casualidad»* — mismo tipo de muletilla.

Segunda pasada, ahora metáforas-verbo, *« sino »* y cierres floridos:

#### [0356] Consejero

Una más, para cerrar el barrido (metáforas de maquinaria/paisaje y conectores-etiqueta en arranque):

#### [0357] Consejero

Barrido completo de los siete `_v2`. Lo agrupo por confianza. **No toco nada todavía** — tú eliges qué cae.

### 🔴 Firmes: expresiones que no son tu voz

| # | Sitio | Tu texto | Cambio |
|---|---|---|---|
| 1 | [03:659](tesis/chapters/03_como_funciona_strata_v2.tex#L659) | «Es el **caballo de batalla** del riesgo en el panel» | «Es el contraste que usamos para todo el riesgo del panel: pooled-10, estratos por régimen y diferencia en diferencias.» |
| 2 | [04:48](tesis/chapters/04_strata_spy_v2.tex#L48) | «Que RAM acierte al intervenir **no es casualidad**» | «La Figura~\ref{...} muestra por qué RAM acierta al intervenir.» |

### 🟠 Patrón repetido: «dibuja» (7 veces) — tu propia regla lo veta

Es el tell más extendido. *«Nada de metáforas literarias para algo técnico: di la acción plana»* (`correcciones_aprendidas`, 2026-06-25). Lo usas como verbo-comodín en cap. 5 sobre todo:

- [05:272](tesis/chapters/05_panel_v2.tex#L272) (título) «la naturaleza **dibuja** los grupos» → «la naturaleza **separa** los grupos»
- [05:112](tesis/chapters/05_panel_v2.tex#L112) «La Figura **dibuja** los intervalos» → «**muestra**»
- [05:207](tesis/chapters/05_panel_v2.tex#L207) «La Figura la **dibuja**» → «lo **muestra**»
- [05:225](tesis/chapters/05_panel_v2.tex#L225) «que esa propiedad **dibuja**» → «que esa propiedad **marca**»
- [05:275](tesis/chapters/05_panel_v2.tex#L275) «los bloques que el rescate **dibuja**» → «que el rescate **define**»
- [06:24](tesis/chapters/06_conclusiones_v2.tex#L24) «**dibuja** la misma estructura» → «**reproduce** la misma estructura»
- [03:711](tesis/chapters/03_como_funciona_strata_v2.tex#L711) «la naturaleza **dibuja** grupos» → «**forma** grupos»

### 🟡 Verbo florido: «descansa sobre»

- [04:145](tesis/chapters/04_strata_spy_v2.tex#L145) «**descansa sobre** una meseta estable»
- [04:155](tesis/chapters/04_strata_spy_v2.tex#L155) «todo **descansa sobre** una meseta robusta»

«Meseta» se queda (es el término estándar de un barrido plano); lo florido es *descansa sobre* → «se mantiene en». 

### ⚪ Mecánico (no es IA, pero está roto) — [04:20](tesis/chapters/04_strata_spy_v2.tex#L20)

Al pegar mi frase de Viterbi quedó: *«Esto es **escriptivo**, empleamos…»* (typo → «descriptivo»), `look ahead` sin formato, y dos frases pegadas sin punto (*«…del estado.Una precisión…»*). Lo dejo limpio si quieres.

---

### Dos cosas que considero y **NO marco** (para que conste que las miré):

- **«compuerta»** (RAM como gate, ~6 veces): es término técnico **preciso** y consistente con tu variable `gate RAM`. No es tell de IA. Lo dejaría salvo que a ti te canse.
- **«no es X **sino** Y»** (en 03:12, 03:64, 03:72, 03:236, 03:381, 02:23): tu regla los veta, pero casi todos son contrastes definitorios limpios («no el precio sino su variación»), no el «no es un fallo sino un hallazgo» que de verdad te molesta. El único que reescribiría por torpe es [03:64](tesis/chapters/03_como_funciona_strata_v2.tex#L64) («no aplica umbrales fijos sino que es un modelo…»). El resto, tu decisión.

---

¿Aplico el **lote firme + el de «dibuja»** (9 cambios, todos seguros) y te dejo lo amarillo/sino para que decidas? ¿O prefieres que para una pasada más fina —longitud de frase, ratio de conectores— lance a **`estilo-raquel`** contra los `_v2` (te pido luz verde antes)?

#### [0401] Raquel

si.

#### [0401] Consejero

Aplico los 9 cambios seguros (los dos idiomas + el lote «dibuja»).

#### [0401] Consejero

Necesito abrir los ficheros con la tool de lectura antes de editarlos. Los leo y aplico.

#### [0401] Consejero

Veo que ya quitaste tú misma el «caballo de batalla» (03:659) y el «no es casualidad» (04:48) — perfecto, esos dos ya están. Me quedan los «dibuja». Aplico 03:711 y 06:24, y leo el cap. 5 para los otros cinco.

#### [0402] Consejero

Título 05:272 ya lo arreglaste tú. Aplico los cuatro «dibuja» restantes del cap. 5:

#### [0402] Consejero

Listo. **Lote «dibuja» completo** — los 7 fuera (3 los habías quitado tú: el «caballo de batalla», el «no es casualidad» y el título de §clustering; los otros 4 los acabo de aplicar). Ahora cap. 5 dice *muestra / marca*, cap. 6 *reproduce*, cap. 3 *forma*.

Nota menor: en cap. 5 quedan dos «marca» a ~50 líneas (225 y 275); distinta subsección, no chirría, pero si quieres varío uno a «define».

**Queda pendiente de tu decisión** (te lo dejé sin tocar):
- 🟡 **«descansa sobre»** en [04:145](tesis/chapters/04_strata_spy_v2.tex#L145) y [04:155](tesis/chapters/04_strata_spy_v2.tex#L155) → «se mantiene en».
- ⚪ **Lío mecánico de [04:20](tesis/chapters/04_strata_spy_v2.tex#L20)** (typo «escriptivo», `look ahead` sin formato, dos frases pegadas) — del Viterbi que pegaste.
- **«no es X sino Y»**: solo reescribiría [03:64](tesis/chapters/03_como_funciona_strata_v2.tex#L64) por torpe; el resto son contrastes limpios, tú decides.

¿Te limpio el 04:20 y el «descansa sobre» también, o lo dejas para ti? Y sigue en pie lo de lanzar **`estilo-raquel`** para la pasada fina (longitud de frase, ratio de conectores) si la quieres — pido luz verde antes.

#### [0403] Raquel

La purga no agota la fuga, anetdótico, estratos: todas las csi siempre las palabras raras son ia

#### [0404] Consejero

Captado: **palabra rara = IA**, y los aforismos ingeniosos tipo *«La purga no agota la fuga»* (juego purga/fuga) son lo más IA de todo. Tu regla de vocabulario llano es prioridad alta. Leo el contexto de esa frase y barro vocabulario rebuscado en todos los `_v2`.

#### [0404] Consejero

La frase que seleccionaste (line 622) ya no está ahí — el fichero se ha movido. La busco, y rehago el grep de vocabulario sin que el filtro me oculte hits:

#### [0405] Consejero

Casi todo limpio: el único «raro» de mi lista que aparece es **anecdótica** (03:202). Tu *«purga no agota la fuga»* ya no está (alguien la quitó). Busco «estratos» y un par más que encajan en tu queja:

#### [0405] Consejero

«estratos» ya no aparece en los `_v2` (debía de ser del 03:659 viejo que tú editaste). Lo que sí queda:

**Aplico ya** (lo nombraste tú):
- [03:202](tesis/chapters/03_como_funciona_strata_v2.tex#L202) «no **anecdótica**» → «no un detalle menor».

#### [0406] Consejero

Hecho. Resumen del barrido de vocabulario:

- ✅ **anecdótica** (03:202) → «no es un detalle menor».
- ✅ **«La purga no agota la fuga»** y **«estratos»** — ya no existen en los `_v2`.

**Quedan dos, tu decisión** (no son latinajos, por eso no los toco sin preguntarte):
- **«contundente»** — [05:53](tesis/chapters/05_panel_v2.tex#L53) «La primera lectura es contundente» → «clara» / «rotunda». Suena un poco a titular de prensa.
- **«holgura»** — aparece 5 veces ([04:155](tesis/chapters/04_strata_spy_v2.tex#L155), [05:112](tesis/chapters/05_panel_v2.tex#L112), [05:117](tesis/chapters/05_panel_v2.tex#L117), [05:348](tesis/chapters/05_panel_v2.tex#L348), [06:13](tesis/chapters/06_conclusiones_v2.tex#L13)), siempre «con holgura / con más holgura». La palabra es correcta, pero repetirla 5 veces canta; podría variar a «con margen / de sobra» en alguna.

Un apunte de método: tu criterio *«palabra rara = IA»* ya vive como regla de vocabulario llano en `correcciones_aprendidas`, pero puedo afinarla con estos ejemplos concretos (anecdótica, contundente) vía `aprendiz-correcciones` para que la próxima redacción ni las proponga — ¿lo lanzo? Y dime si quieres que cambie «contundente» y varíe algún «holgura».

#### [0548] Raquel

Lectura sin contraste asociado. BUSCA FRASES SIMILARES A ESTA, QUE ME QUEDEN EN EVIDENCIA

#### [0549] Consejero

Entiendo: *«Lectura sin contraste asociado»* te deja expuesta — es un fragmento telegráfico, sin verbo, que además admite a pelo que algo no tiene test detrás. Justo lo que tu regla prohíbe: no le regales munición cruda al tribunal. Veo el contexto exacto y busco las hermanas de esa frase.

#### [0549] Consejero

Confirmado lo que pasa en [05:130](tesis/chapters/05_panel_v2.tex#L130): *«Lectura sin contraste asociado.»* va **pegada al final del caption** (sin espacio: «…alcista.Lectura…»), sin verbo, y **repite** lo que el cuerpo ya dijo en [05:125](tesis/chapters/05_panel_v2.tex#L125) («queda como lectura, no como afirmación»). Es un cartelito de IA que además grita la debilidad. Busco sus hermanas — fragmentos telegráficos y admisiones crudas:

#### [0552] Consejero

Encontradas. Distingo dos cosas, porque no todas se arreglan igual.

### Las hermanas gemelas de tu frase (fragmento sin verbo, cartel de IA)

Son las que de verdad te dejan en evidencia: etiquetas telegráficas pegadas a un caption.

| Sitio | Frase | Problema |
|---|---|---|
| [05:130](tesis/chapters/05_panel_v2.tex#L130) | «…periodo alcista.**Lectura sin contraste asociado.**» | Sin verbo, sin espacio, y **repite** el cuerpo (05:125 ya lo dice framed). |
| [05:84](tesis/chapters/05_panel_v2.tex#L84) | «**Victorias nominales (n≈250, a posteriori).**» | Mismo patrón: coletilla sin verbo al final del caption, soltando la debilidad a pelo. |

**Arreglo:** un caption no lleva veredictos crudos. En 05:130 **borro el cartelito** (el matiz «sin contraste» ya vive, bien encuadrado, en el cuerpo 05:125). En 05:84 lo convierto en frase: «Las cuatro victorias son nominales: la ventana ronda los 250 días y la ganadora se elige a posteriori.»

### El párrafo que grita el límite — [05:79](tesis/chapters/05_panel_v2.tex#L79)

No es telegráfico, pero es el que más munición regala, y encima **en negrita**:

> «…son nominales. **Ninguna bate a ZeroR con significancia**… el límite es que no sobrevive a un test.»

Tu regla ALTA: honesta sí, pero el límite **no se titula ni se pone en negrita**. La negrita sobre la debilidad es justo lo contrario. Propongo **quitar la negrita** y encuadrar: liderar con el valor (la victoria en punto) y dejar el «no llega a significativa» como cláusula secundaria dentro de la frase, no como remate en bold.

### Lo que NO marco (para que conste)

La **tabla de veredictos** (05:329–345) está llena de fragmentos («Tendencia al α=0,10; no p<0,05»), pero ahí son **celdas de tabla**: el formato pide fragmento, no es tell. Y las admisiones del cuerpo (05:123, 05:125, 06:47) son honestidad legítima bien encuadrada — se quedan.

---

¿Aplico lo seguro ya — **borrar el cartel de 05:130** (y arreglar el espacio) y **quitar la negrita de 05:79**— y te dejo propuesto el reframe de 05:84 y del párrafo 79 para que lo apruebes? También puedo seguir el barrido del mismo patrón en cap. 4 y 6 si quieres la pasada completa.

#### [0613] Raquel

busca la palabra anécdota o anecdotico  y remplazala

#### [0613] Consejero

Solo queda una (la de 03:202 ya la cambié antes): [05:227](tesis/chapters/05_panel_v2.tex#L227) «el rescate deja de ser **una anécdota** de SPY». La leo y la sustituyo.

#### [0614] Consejero

Hecho. «una anécdota de SPY» → «un resultado aislado de SPY». 

Ya no queda ninguna «anécdota / anecdótico» en toda la tesis (las dos que había: esta y la de 03:202, las dos fuera).

