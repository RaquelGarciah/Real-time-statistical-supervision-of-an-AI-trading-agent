# Objeciones del tribunal y respuestas (consolidado)

> Respuestas honestas y defendibles a lo previsible. Detalle en `docs/defensa_walkforward.md` y
> `docs/chats/questions_and_answers.md`. **Regla de oro: no inflar. Lo nominal se dice nominal.**

| # | Objeción | Respuesta |
|---|---|---|
| 1 | *"¿Ganaste porque el mercado subió?"* | No: el caso central es **SMCI con B&H ≈ 0,484** (benchmark justo, casi una moneda), no un activo alcista. Ahí M10 0,552 > B&H 0,484. |
| 2 | *"Una estrategia trivial (siempre +1) gana."* | En SMCI las clases están casi balanceadas → la mayoritaria ("siempre corto") saca ≈ 0,516; **M10 0,552 la supera**. Por eso se eligió SMCI, no SPY (donde "siempre largo" da 0,569). |
| 3 | *"Tu ventaja no es significativa / el IC roza el cero."* | Cierto y se reporta: **nominal, no significativa** (DSR 0,72; block-perm 0,047 no sobrevive multiplicidad). La barra del tutor es *batir a lo trivial + trabajo futuro*, no significancia (imposible a N≈250). Evidencia de que **no es azar**: robustez a la partición (60/40, 70/30, 80/20) y al rolling (71–82 %). |
| 4 | *"Un XGBoost debería batir tu regla a mano."* | M10 (XGBoost) y M8 (regla) son **equivalentes en P&L** (DM); **SHAP** señala las features STRATA como las informativas. El XGBoost **redescubre** STRATA, no la bate → confirma la universalidad (CLAUDE.md §2, nivel 3). |
| 5 | *"¿Por qué K=3 y no otro?"* | Interpretable (Calma/Estrés/Crisis) y validado: K-por-activo **no generaliza** (falsación, `falsacion/k_y_signo/`); `k_ablation_panel` confirma K=3. |
| 6 | *"¿Hay look-ahead?"* | No: walk-forward expandible, solo pasado, **embargo=1**, `signal_lag=1` (posición de t × retorno de t+1). El CPCV (que ve el futuro) se reporta **solo como contraste** y en SMCI da peor (0,448). |
| 7 | *"Si M8/M10 no baten a B&H en SPY, ¿qué rescataste?"* | "Rescate" = **recuperar la dirección del agente**, no batir al mercado. Es **significativo en SPY** (M10 vs M5 p=0,0041) porque ahí el agente discrepa de un régimen que acierta (leverage effect). En SMCI no hay margen porque el agente ya va alineado con el régimen. |
| 8 | *"STRATA es predicción disfrazada."* | No: STRATA **supervisa** (`f: tupla_agente × estado → tupla_supervisada`). Quien predice es el agente y el XGBoost. Distinción mantenida en toda la memoria. |

**Frase de cierre defendible:** *"El M10 desplegable bate al pasivo en un benchmark justo (0,552 vs 0,484), de
forma robusta a la partición y al rolling; la significancia plena requiere más muestra. La aportación no es
batir al mercado: es un protocolo de supervisión estadística interpretable, desplegable y honesto."*
