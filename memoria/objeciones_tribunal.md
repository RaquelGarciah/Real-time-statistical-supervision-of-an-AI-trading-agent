# Objeciones del tribunal y respuestas (consolidado)

> Respuestas honestas y defendibles a lo previsible. Detalle en `docs/defensa_walkforward.md` y
> `docs/chats/questions_and_answers.md`. **Regla de oro: no inflar. Lo nominal se dice nominal.**

| # | Objeción | Respuesta |
|---|---|---|
| 1 | *"¿Ganaste porque el mercado subió?"* | No: el caso de estudio es **SMCI con B&H ≈ 0,484** (benchmark justo, casi una moneda), no un activo alcista. Ahí M10 0,552 > B&H 0,484. |
| 2 | *"Una estrategia trivial gana."* | En SMCI las clases están casi balanceadas → la mayoritaria ("siempre corto") saca ≈ 0,516; **M10 0,552 la supera**. Por eso se eligió un activo balanceado, no uno alcista donde "siempre largo" ganaría por construcción. |
| 3 | *"Tu ventaja no es significativa / el IC roza el cero."* | Cierto y se reporta: **nominal, no significativa** (DSR 0,72; block-perm 0,047 no sobrevive multiplicidad). La barra del tutor es *batir a lo trivial + trabajo futuro*, no significancia (imposible a N≈250). Evidencia de que **no es azar**: robustez a la partición (60/40, 70/30, 80/20) y al rolling (71–82 %). |
| 4 | *"Un XGBoost debería batir tu regla a mano."* | Sí, y es justo lo que muestro: **M10 (XGBoost) supera a la regla a mano M8** (0,552 vs 0,496) porque extrae más de las **mismas señales de STRATA**. Por eso M10 es el modelo del caso de estudio; STRATA aporta las features informativas que M10 explota. |
| 5 | *"¿Por qué K=3 y no otro?"* | Interpretable (Calma/Estrés/Crisis) y validado: K-por-activo **no generaliza** (falsación, `falsacion/k_y_signo/`); `k_ablation_panel` confirma K=3. |
| 6 | *"¿Hay look-ahead?"* | No: walk-forward expandible, solo pasado, **embargo=1**, `signal_lag=1` (posición de t × retorno de t+1). El CPCV (que ve el futuro) se reporta **solo como contraste** y en SMCI da peor (0,448). |
| 7 | *"Si la regla a mano M8 apenas supera al agente, ¿qué aporta STRATA?"* | En SMCI el agente ya va casi siempre alineado con el régimen, así que la regla rígida M8 aporta poco. El valor está en que **M10, sobre las mismas señales de STRATA, extrae más** y bate a agente, regla y trivial. STRATA aporta como **conjunto de señales informativas**, no como regla rígida. |
| 8 | *"STRATA es predicción disfrazada."* | No: **STRATA supervisa** (produce señales sobre la decisión del agente); **quien predice la dirección es M10** (un clasificador sobre esas señales). La distinción se mantiene en toda la memoria. |
| 9 | *"Un solo activo es poco para una tesis."* | Es un **caso de estudio / prueba de concepto**. El objetivo de fondo —una estrategia de **supervisión desplegable en tiempo real para cualquier agente y activo**— es la **línea principal** y se desarrolla como **trabajo futuro** (cap. 5). El tiempo del TFG permite validar un activo con rigor; la generalización está **declarada**, no escondida. Se elige un activo de **clases balanceadas** para que la comparación con lo trivial sea **justa** y la accuracy, informativa. |

**Frase de cierre defendible:** *"El meta-learner desplegable M10 bate en accuracy al agente, a la regla a mano
y a lo trivial en un benchmark justo (0,552 vs 0,484–0,516), de forma robusta a la partición y al rolling; la
significancia plena requiere más muestra. La aportación no es batir al mercado: es una estrategia de supervisión
estadística interpretable y desplegable, demostrada aquí en un caso de estudio y generalizable a cualquier
agente y activo como trabajo futuro."*
