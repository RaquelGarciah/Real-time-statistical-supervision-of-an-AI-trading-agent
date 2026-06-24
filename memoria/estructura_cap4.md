# Estructura aprobada del Capítulo 4 (Marco práctico — caso de estudio SMCI) — OBSOLETO

> **⚠️ OBSOLETO (2026-06-24).** Esta estructura es de la era **SMCI** y NO se usa. La estructura del cap. 4 del
> enfoque actual (SPY + panel de 10 + patrones) la define `coordinador-redaccion` sobre la fuente de verdad
> [`MARCO_PRACTICO_CONTEXTO.md`](../MARCO_PRACTICO_CONTEXTO.md) (§XII: estructura y figuras/tablas imprescindibles).
> Se conserva abajo solo como registro histórico.

> **Fuente de verdad [HISTÓRICA]** que custodiaba el agente `arquitecto-estructura`. Cualquier redacción del cap. 4 debe
> seguir EXACTAMENTE estos bloques y este orden. Decisiones de Raquel (2026-06-18): **todo SMCI** (SPY no entra
> al cuerpo), **mecánica de STRATA en el cuerpo** (no apéndice), **compacto y directo** (subsecciones de 1-2
> párrafos). Las cifras vienen de JSON / `RESULTADOS_OBJETIVO.md §1bis`, nunca a mano.

## Idea que vertebra el capítulo (honesta, todo interno a SMCI)

A n≈250 ninguna pieza prueba habilidad por sí sola, pero **cuatro evidencias independientes convergen**, y se
presentan en positivo (no repetir "trabajo futuro"):
1. **Dominancia 2×2 estricta**: M10 (0,552) es el único que bate a M5, M8, B&H **y** a la clase mayoritaria/NIR
   (0,516) → descarta "solo es estar corto".
2. **Invarianza a la partición**: gana en validación Y test en 60/40, 70/30, 80/20.
3. **Consistencia temporal**: gana a B&H en 71–82 % de ventanas rodantes.
4. **Mecanismo identificado (ablación)**: sin las 7 señales STRATA cae a 0,468 ≈ agente; con ellas 0,552.
La no-significancia es **límite de potencia (n≈250), no de signo**.

## Principio: 7 bloques

**§4.0 Introducción-mapa** (breve). La pregunta operativa sobre SMCI y el recorrido del capítulo. Sin adelantar
todas las cifras.

**§4.1 El problema y el banco de pruebas justo**
- 4.1.1 SMCI y por qué es un *benchmark justo*: intro breve del activo (resuelve el `>>>RAQUEL` actual); clases
  ≈balanceadas (B&H 0,484), accuracy informativa, clase mayoritaria/ZeroR 0,516 = "siempre corto"
  (`witten2016datamining`, `kuhn2008caret`); criterio a-priori que justifica elegir SMCI, **sin tabla de panel**.
- 4.1.2 Protocolo: calibración 2000→2024-09 congelada; OOS n=250 (= 400 − 150 burn-in); walk-forward causal,
  `signal_lag=1`, embargo=1. Referencia a cap. 3 **una sola vez** (no repetir la coletilla de causalidad).
- 4.1.3 **M5** sin supervisar: el punto de partida. Acierta 0,484 (< azar), 95 % corto. Test: sign/binomial vs 0,5.

**§4.2 STRATA sobre SMCI: régimen, detectores y umbrales** (mecánica en el cuerpo, compacta)
- Régimen del HMM sobre el precio (calibración + OOS) y **matriz de transición** (persistencia).
- Los tres detectores en este activo: **RAM** es el que actúa; **PSA y GSO no disparan** (agente pasivo,
  tamaño ≈0,10; umbrales ex-ante P95/P99; la decisión de no tunearlos se valida en val/test, sin mejora).
- Un día de intervención por dentro (RAM/PSA/GSO en un caso concreto).

**§4.3 Dos formas de explotar STRATA**
- 4.3.1 **M8**, la regla a mano: override-C, gate τ=0,5; 0,496 ≈ M5 porque agente y régimen coinciden (interviene
  ~3 %); referencia interpretable.
- 4.3.2 **M10**, el meta-learner: XGBoost (`chen2016`); w=signo(p1−0,5); 22 features = 15 agente + 7 STRATA
  (nombrar el objeto concreto); walk-forward burn-in 150 / reentreno 21 d / embargo 1; ensemble 10 semillas =
  bagging (`breiman1996`, `dietterich2000`); **justificar** XGBoost (no-linealidad e interacciones régimen×agente)
  y embargo=1 (horizonte de etiqueta); qué NO lleva (CPCV, momentum, abstención). Es un meta-learner en el
  sentido de *stacking* (`wolpert1992`): modelo de segundo nivel sobre las salidas del agente y de STRATA.

**§4.4 Resultado principal**
- 4.4.1 La comparación (todo el OOS): tabla de 6 estrategias + **figura headline**. M10 0,552 único que bate a
  todos. Criterio de éxito enunciado aquí (superar a lo trivial nominal y consistentemente).
- 4.4.2 La economía como ilustración: equity 3,24×, Sharpe +1,84, **P(Sharpe>0) 0,976 cruda / 0,72 corregida**
  (`bailey2014`; NUNCA llamarlo "DSR" suelto). Con n≈250 el 3,24× es compatible con suerte.
- 4.4.3 Contraste estadístico, **jerarquía clara**: primario = binomial vs NIR=0,516 (p=0,141); secundarios =
  block-perm vs B&H 0,047 **con su Bonferroni-5≈0,28 en la misma frase**, McNemar vs M5 0,16; **forest plot**;
  lectura: ventaja nominal.
- 4.4.4 ¿Aporta el ML sobre la regla? **M8 vs M10** (McNemar + Diebold-Mariano sobre P&L): la **hipótesis nivel 3**
  del trabajo (un ML no debe batir significativamente a la regla). Fuente: JSON de prep (DM nuevo).

**§4.5 Robustez** (un eje por subsección, compacto)
- A la partición (3 splits, val + test). Tabla `tab:smci-robustez`.
- En el tiempo (rolling-window 71–82 %).
- Al protocolo (embargo: pico aislado en emb=1 → sensibilidad, no confirmatorio).
- A la ventana de calibración (la completa es la más robusta; `smci_calib_window`).

**§4.6 Interpretabilidad y honestidad**
- 4.6.1 M10 inseparable de STRATA: **ablación** (15 → 22: 0,468 → 0,552, McNemar 0,053) + **SHAP** real
  (TreeExplainer): las 7 señales STRATA/régimen = **64,7 %** de la importancia (los 5 primeros features son todos
  de STRATA: garch_sigma, ram, stress_prob, psa, crisis_prob). `lundberg2017`. (Nota: 41,4 % era *gain* de
  XGBoost, fallback cuando `shap` no estaba instalado; el valor citable es el TreeSHAP = 64,7 %.)
- 4.6.2 Por qué la ventaja es pequeña en SMCI: agente 95 % corto + régimen corto → los tres modelos apuestan
  igual; STRATA rescata donde el agente discrepa de un régimen que acierta. El drawdown del verano 2025 como
  ejemplo breve del límite (régimen no direccional en SMCI).

**§4.7 Límites y trabajo futuro**. Muestra corta (n≈250, por el corte del LLM) → significancia plena = futuro;
generalización multi-agente / multi-activo / despliegue vivo (coste cuantificado, cap. 5); **SPY como activo
donde el mecanismo debería ser significativo** (trabajo futuro / apéndice), sin desarrollarlo aquí.

## Reglas de colocación (lo que vigila el guardián)
- La **economía** (Sharpe, equity, P(Sharpe>0)) vive en §4.4.2, marcada como ilustración, nunca como prueba.
- Cada **test** va con su cifra y su **fuente JSON**; jerarquía explícita (primario vs secundarios) en §4.4.3.
- La **causalidad/embargo** se referencia a cap. 3 **una sola vez** (§4.1.2).
- El **"por qué SMCI"** se argumenta a-priori (benchmark justo), sin arrastrar otros activos al cuerpo.
- **SPY** no aparece en el cuerpo; solo como trabajo futuro en §4.7.
- **Nada inventado**: toda afirmación no original lleva cita; toda cifra, fuente JSON. Estilo: reglas vivas de
  `memoria/ESTILO_Y_ANTIIA.md` y `memoria/correcciones_aprendidas.md` (sin guion-muletilla, sin meta-comentarios,
  sin "no es X sino Y", conectores variados, primera persona plural).
