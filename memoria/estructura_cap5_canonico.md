# Estructura validada del Capítulo 5 (Conclusiones)

> **Fuente de verdad:** [`MARCO_PRACTICO_CONTEXTO.md`](../MARCO_PRACTICO_CONTEXTO.md) + hipótesis falsable de
> `CLAUDE.md` §2. **Estado:** VALIDADA por Raquel (2026-06-24). El cap. 5 SINTETIZA y ELEVA el cap. 4: referencia
> con `\ref`, NO re-tabula ni re-deriva cifras. Cada cifra de cabecera aparece una sola vez con remisión.

## Reglas duras
- Panel de **diez activos**, ex ante, SIN apéndice. PROHIBIDO "quince"/"10 de N". SMCI uno de los 10.
- Override (variantes) / intervención (mecanismo). Sin costes. El **alfa** solo en 5.5 como límite de alcance +
  línea futura; NUNCA "STRATA no genera alfa" como bandera (§VII: destacar lo que STRATA SÍ hace).
- Honestidad sin diluir: los tres "no" (nominal vs trivial, ley α=0,10, M8 no Bonferroni) van en 5.4, no
  contaminan el titular en positivo de 5.1.4 ni del cierre.
- Voz Raquel: frases cortas, dos puntos, primera persona plural, sin guiones largos muletilla, sin "no es X
  sino Y", sin meta-comentarios ("conviene", "cabe destacar"), sin honestidad performativa ("lo decimos sin
  maquillar"), sin enumeraciones ordinales rígidas ("El primero/segundo..."), sin andamiaje de `\ref` en cadena.
- NO introducir símbolos ni citas nuevas: todo viene definido del cap. 4.

## Arco (~5–6 pág)

**Intro (~0,4 pág):** reabre problema + pregunta del TFG; veredicto en positivo; enuncia la jerarquía de valor.

**5.1 Respuesta a la hipótesis: veredicto en tres niveles** `\label{sec:conc-hipotesis}` (~1,5 pág, núcleo)
- 5.1.0 punto de partida: agente perdedor direccional transversal (ref `\ref{sec:mp-spy}`, `\ref{sec:mp-panel-agente}`; SPY M5 0,366, sign test p<0,001).
- 5.1.1 **Nivel estadístico → CONFIRMADO.** McNemar sig (AutoML 0,0002/M10 0,0074/M8 0,051; ref `\ref{tab:mp-spy}`, `\ref{tab:mp-regimen}`) + pooled-10 riesgo excluye 0 (M8 +0,60/M10 +1,12/AutoML +1,08; ref `\ref{tab:mp-pooled}`, `\ref{fig:mp-forest}`). Grieta nombrada: **M8 no pasa Bonferroni** (cota −0,047).
- 5.1.2 **Nivel mecánico → CONFIRMADO.** RAM domina (100% P&L rescate; ref `\ref{sec:mp-spy-detectores}`); dos capas disjuntas (`\ref{tab:mp-capas}`: regla 0/10 accuracy, aprendiz 10/10); gate RAM 6/10, r=0,93 (`\ref{fig:mp-gate-ram}`); ablación AutoML 0,574→0,550 (`\ref{tab:mp-ablacion}`).
- 5.1.3 **Nivel universalidad → MATIZADO** (la más delicada). SHAP confirma apoyo en STRATA (>0,5 en 10/10, media 0,66; `\ref{sec:mp-panel-tost}`); el aprendiz SÍ bate modestamente a M8 en accuracy (TOST M10 +0,021/AutoML +0,034; `\ref{tab:mp-tost}`) pero NO en riesgo (no concluyente). Redactar como **PRECISIÓN del enunciado** original ("no debe batir significativamente"), NO como refutación: el espíritu (valor en STRATA) se sostiene (lo confirman SHAP + ablación).
- 5.1.4 Veredicto consolidado: la hipótesis se sostiene, con su alcance medido.

**5.2 El patrón que ordena el rescate: la ley del leverage** `\label{sec:conc-ley}` (~0,7 pág)
- El rescate del aprendiz escala con el leverage (r=−0,56, p=0,093, α=0,10; 9/10, ROKU excepción; ref `\ref{sec:mp-mec-ley}`); clustering Rand=1,0, PC1≈leverage r=0,84 (exploratorio); complementariedad DiD +1,37 p=0,008. Como **hipótesis a futuro**, no ley cerrada. La naturaleza ordena el aprendiz, NO la regla (crisis_mean descriptivo).

**5.3 Aportaciones** `\label{sec:conc-aportaciones}` (~1 pág)
- 5.3.1 **Metodológica** (el rigor = la contribución, CLAUDE.md §4): protocolo interpretable, calibración ex ante, tests pareados+IC, causalidad estricta (signal_lag=1, embargo=1, dos ventanas), pre-registro de falsaciones (SPY-bajista −1,49 resuelto por panel; ref `\ref{sec:mp-spy-protocolo}`, `\ref{sec:mp-panel-regimen}`).
- 5.3.2 **Empírica** (doble): la capa clásica rescata con significancia pareada (remite a 5.1) + la ley naturaleza→canal (remite a 5.2). NO repetir cifras.

**5.4 Limitaciones** `\label{sec:conc-limitaciones}` (~0,7 pág)
- Síntesis elevada (NO la tabla del cap. 4; ref `\ref{sec:mp-limites}`): muestra corta → nominal vs trivial; riesgo sig en agregado no por activo; alcance del rescate se estrecha con leverage débil; un solo agente/ventana; M8 no Bonferroni; ley α=0,10; clustering exploratorio. Cierre: reconocer el alcance no diluye lo demostrado.

**5.5 Líneas futuras** `\label{sec:conc-futuro}` (~0,7 pág)
- Cada límite ancla una continuación (variar conectores, NO "La primera/segunda..."): muestra mayor; ensemble enrutado por régimen (ancla DiD); otros agentes/LLMs/mercados (¿la ley aguanta fuera?); modos intermedios de intervención; despliegue en vivo (regla día 1, aprendices tras burn-in); **alfa direccional robusto** (solo aquí, ancla SMCI/MARA/UNG de `\ref{fig:mp-equity-panel}`, con contraste pre-registrado, no a posteriori; punto de partida, no conclusión).

**Cierre del TFG** (~0,3 pág, párrafo final o `\label{sec:conc-cierre}`)
- Síntesis sobria en positivo: la supervisión vuelve fiable a un agente que solo no lo era; valor probado = rescate + interpretabilidad; batir al mercado no era el objetivo. Devuelve a la pregunta de apertura.

## Riesgos para los gates
- No repetir cap. 4 (referenciar, no re-tabular).
- Nivel 3: precisión del enunciado, no auto-refutación (crítico para `defensa-tutor`/`harvard-professor`).
- Alfa solo como línea futura; ni bandera negativa ni logro.
- Universo de diez, frase canónica; cero "quince".
