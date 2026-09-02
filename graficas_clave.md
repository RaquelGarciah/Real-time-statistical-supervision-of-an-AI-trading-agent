# Gráficas clave del notebook final de SMCI — qué enseñar y por qué

> Documento único, al grano. Todas las gráficas que pueden ir al notebook final del caso de estudio (SMCI),
> de dónde salen, qué muestran y por qué importan. Supuesto: **todo se lee como si fuera sobre SMCI** (las del
> canónico están hoy sobre SPY → habría que replicarlas sobre SMCI). Marcadas: ✅ ya existe · 🟡 existe pero
> sobre SPY (replicar) · 🆕 propuesta nueva (no existe, la construyo si quieres).
>
> **Resultado que cuenta:** M10-WF ensemble (embargo=1) sobre todo el OOS = **accuracy 0.552 > M5 0.484 /
> M8 0.496 / B&H 0.484**, respaldado por robustez (partición, embargo, ventanas). Nominal, no significativo.

---

## ESTADO (2026-06-18): notebook construido → `notebooks/STRATA_SMCI.ipynb`

El notebook definitivo **ya está construido y ejecutado** (builder `_build_STRATA_SMCI.py`; 56 celdas, **23
figuras**, 0 errores; sustituye a `strata_canonical`). Este documento queda como **mapa/justificación** de las
gráficas; el inventario real ejecutado es:

**Parte II (mecánica/calibración):** régimen sobre precio (calibración + OOS), LL held-out vs K + scatter
régimen, **matriz de transición del HMM** 🆕, percentiles PSA/GSO, RAM-gate (bimodal + accuracy-por-τ).
**Parte III (resultado):** barra headline 6 estrategias, curvas de equity, **análisis del drawdown verano 2025**
🆕, **rezago del régimen (precio→RV²¹→régimen + posición M10)** 🆕, SHAP, ablación agente-15/STRATA-7/22,
matrices de confusión M5/M10, descriptivo 3×3, gate RAM, **día de intervención por dentro (RAM/PSA/GSO)** 🆕,
alcista/bajista. **Parte IV (robustez):** particiones 60/40·70/30·80/20, embargo, rolling, **umbral 0.5 en
val/test** 🆕, **robustez a la ventana de calibración** 🆕. **Parte V (honestidad):** panel intervención +
margen por activo, métodos avanzados, abstención. **Nota:** el Sharpe se reporta como **P(Sharpe>0)** (0.976
sin corregir / 0.72 corregida), no como "DSR".

---

## Estrategias / baselines a comparar (en TODA gráfica y tabla)

Toda comparación de accuracy enfrenta **6 estrategias**: 3 modelos + 3 baselines triviales.

| Estrategia | Qué es | Accuracy en SMCI (OOS, n=250) |
|---|---|---|
| **M5** | el agente LLM (posición del agente) | 0.484 |
| **M8** | STRATA regla (override-C) | 0.496 |
| **M10** | meta-learner (XGBoost ensemble, embargo=1) — **nuestro modelo** | **0.552** |
| **B&H** | *buy-and-hold* = **siempre largo** (baseline económico) | 0.484 |
| **S&H** | *short-and-hold* = **siempre corto** (el espejo de B&H) | 0.516 |
| **Clase mayoritaria** | regla ZeroR / NIR = **siempre la dirección dominante** (baseline de no-habilidad) | 0.516 |

**Aviso clave (que el tribunal puede preguntar):** en SMCI **S&H = clase mayoritaria = 0.516**, porque bajan
más días de los que suben → la dirección dominante ES "corto". No son la misma idea (S&H es una estrategia
constante fija; la clase mayoritaria es "la mejor de las dos constantes", `max(B&H, S&H)`), pero **en este
activo coinciden numéricamente**. Si se reportan los dos, dejar la nota: *"en SMCI la clase mayoritaria se
materializa como S&H (siempre corto)"*.

**Por qué importan los dos baselines triviales (B&H y S&H/mayoría):** M10 (0.552) **bate a los dos** → su
ventaja no es un sesgo a un lado. El contraste honesto de significancia es el **binomial vs NIR=0.516**
(clase mayoritaria), no vs 0.5 (ver `decisiones_respaldadas_literatura.md` §12 y `logic_esential` §14e).

---

## ⭐ LAS 6 IMPRESCINDIBLES (el hilo del caso de estudio, en orden)

| # | Gráfica | Qué muestra | Por qué | Fuente |
|---|---|---|---|---|
| 1 | **Headline: accuracy M10 vs M5/M8/B&H** (todo el OOS) | M10 0.552 por encima de los tres; líneas azar (0.5) y B&H | Es **el resultado**: M10 bate a todo en un benchmark justo | 🆕 (versión limpia; hoy mezclada en §C de `m10_better_smci`) |
| 2 | **Curvas de equity** (M10 ens, base, M5, M8, B&H) | M10 ens 3.24× vs B&H 0.71× | Enriquecimiento económico (Sharpe/equity), con DSR a la vista | ✅ `m10_better_smci` §C.2 |
| 3 | **Robustez a la partición val/test** (60/40, 70/30, 80/20) | M10 > B&H en validación Y test en los 3 splits | **Robustez**: la conclusión no depende del corte (la que pediste) | ✅ `m10_better_smci` §E.2 |
| 4 | **Robustez al embargo** (accuracy y p por embargo 0–10) | pico no monótono; emb=1 elegido por principio | Honestidad + justifica el protocolo (emb=1) | ✅ `m10_better_smci` §0bis |
| 5 | **Rolling-window** (accuracy rodante + % ventanas ganadas) | M10 > B&H 71–82 % y > agente 67–76 % de ventanas | **Consistencia** intra-OOS, no suerte de un tramo | ✅ `m10_better_smci` §G |
| 6 | **Por qué SMCI** (margen acc(M10)−máx(M5,M8,B&H) por activo) | solo SMCI tiene barra positiva en el panel | Justifica la **elección del activo** (único que bate a todo) | ✅ `m10_better_smci` §F.2 |

Con estas 6 cuentas toda la historia: **resultado → economía → robusto al split → robusto al embargo →
consistente en el tiempo → por qué SMCI.**

---

## 🔍 DE APOYO (mecanismo y honestidad — para defender, apéndice o cuerpo)

| Gráfica | Qué muestra | Por qué | Fuente |
|---|---|---|---|
| **Agente 95 % corto + intervención STRATA** (panel: discrepancia agente↔régimen e intervención) | el agente ya está corto → STRATA interviene 3 % → no hay rescate | explica **por qué M5/M8/M10 no se separan** en SMCI | ✅ `m10_better_smci` §F.1 |
| **Rescate de M10 sobre el agente por activo** (verde = sig.) | solo SPY significativo (p=0.0041) | sitúa SMCI vs SPY (dónde STRATA sí rescata) | ✅ `m10_better_smci` §F.1 |
| **Abstención: cobertura vs accuracy** (ensemble/régimen/acuerdo) | abstener no sube accuracy (régimen baja a 0.489) y recorta cobertura | por qué se **descarta** la abstención (literatura §11) | ✅ `m10_better_smci` §C.3 |
| **Tuning en validación fracasa** (val ganador vs test) | la config elegida se desploma en test | demuestra **sobreajuste de selección** (validación≠test) | ✅ `m10_better_smci` §A |
| **Configs fijas a priori** (accuracy + Sharpe por config) | techo 0.552 (ensemble); señal real/recencia no suben | mapa de lo probado; el ensemble destaca | ✅ `m10_better_smci` §B |
| **Métodos avanzados** (accuracy de triple-barrier/régimen/stacking/voting) | ninguno supera al ensemble | exhaustividad: se probó la literatura | ✅ `m10_better_smci` §C |

---

## 🧩 MECÁNICA DE STRATA (del notebook canónico — replicar sobre SMCI)

Hoy están sobre SPY (`strata_canonical.ipynb`); para el notebook final de SMCI habría que regenerarlas con
los datos de SMCI. Son las que explican **cómo funciona** STRATA por dentro.

| Gráfica | Qué muestra | Por qué | Fuente |
|---|---|---|---|
| **Los 3 regímenes (vol y signo)** | Calma/Estrés/Crisis distintos en volatilidad y media de retorno | base del HMM (RAM); leverage effect | 🟡 canónico §3 |
| **LL fuera de muestra vs K** | K=3 generaliza mejor que 2/4 | justifica **K=3** (decisión por literatura) | 🟡 canónico §3 |
| **RAM como gate** (RAM alto → agente falla; seguir régimen acierta más) | en los días que RAM dispara, el régimen acierta más que el agente | el mecanismo del "rescate" de STRATA | 🟡 canónico §4–§6 |
| **Matriz de confusión dirección + sign test** | acierto direccional del agente vs azar | baseline: el agente como predictor | 🟡 canónico §7 |
| **Día representativo de intervención** | cómo override-C voltea el signo | mecánica de M8, didáctico | 🟡 canónico §5 |
| **SHAP / importancia de features de M10** | régimen+STRATA arriba, personalidades del agente abajo | la señal informativa es la de STRATA, no el agente | 🟡 (ablation NVDA / canónico) |

---

## 🆕 PROPUESTAS NUEVAS (para visualizar mejor lo robusto — las construyo si quieres)

1. **Scorecard headline** (una tarjeta): accuracy, Sharpe, equity y DSR de **M10 vs B&H/M5/M8** en una sola
   figura compacta, con ✓/✗ de "bate a". *Por qué:* resumen de 1 vistazo para abrir el notebook/la defensa.
2. **Accuracy estratificada por régimen en SMCI** (Calma/Estrés/Crisis): barras M10 vs B&H por régimen.
   *Por qué:* muestra **dónde** acierta M10 y conecta con el leverage effect (análogo al condicional del
   canónico, pero para SMCI).
3. **Equity con sombreado de régimen / tendencia** (tramos alcista/bajista sombreados sobre la curva de
   equity de M10 vs B&H). *Por qué:* hace visible la historia honesta de §F (M10 corto pierde en el rally de
   mediados de 2025, gana en la caída) — defiende el límite sin que te lo saquen.
4. **Forest plot de significancia**: M10 vs B&H, vs M5 y vs azar, con p-valor/IC, en el OOS completo **y** en
   los 3 splits, en una sola figura. *Por qué:* enseña a la vez la **robustez** (gana siempre) y la
   **honestidad** (significancia borderline) — es la figura que blinda contra el tribunal.
5. **Distribución de posiciones de M10** (% largo/corto) vs % días alcistas reales. *Por qué:* responde de
   antemano a "¿no es solo estar corto?" mostrando que en el OOS completo M10 está ~47 % corto (equilibrado).

**Mi recomendación de prioridad para construir:** #1 (scorecard) y #4 (forest plot) son las que más suman a
la narrativa robusta; #3 (equity sombreada) es la mejor para defender el límite honesto.

---

## TABLA RÁPIDA: gráfica → fuente → estado

| Gráfica | Fuente actual | Estado |
|---|---|---|
| Headline accuracy M10 vs todo | `m10_better_smci` §C (mezclada) | 🆕 versión limpia |
| Curvas de equity | `m10_better_smci` §C.2 | ✅ |
| Robustez partición (3 splits) | `m10_better_smci` §E.2 | ✅ |
| Robustez embargo | `m10_better_smci` §0bis | ✅ |
| Rolling-window | `m10_better_smci` §G | ✅ |
| Por qué SMCI (margen panel) | `m10_better_smci` §F.2 | ✅ |
| Agente corto + intervención | `m10_better_smci` §F.1 | ✅ |
| Abstención cobertura/accuracy | `m10_better_smci` §C.3 | ✅ |
| Tuning fracasa | `m10_better_smci` §A | ✅ |
| Configs / métodos avanzados | `m10_better_smci` §B, §C | ✅ |
| 3 regímenes / LL vs K / RAM gate / confusión / SHAP | `strata_canonical` (SPY) | 🟡 replicar SMCI |
| Scorecard · estratificada régimen · equity sombreada · forest plot · posiciones | — | 🆕 propuestas |

*Fuente de cifras: `RESULTADOS_OBJETIVO.md` §1bis (SMCI) · recorrido: `docs/chats/decision_activo/smci.md`.*

---

# 🎯 VEREDICTO: estructura del notebook final de SMCI

Mi recomendación de qué incluir, **en este orden**, y por qué. La idea: contar la historia de arriba a abajo
—**resultado → por qué SMCI → robustez → honestidad → cierre**— con ~10 figuras, **cortando lo redundante**.
Cada figura cuenta algo nuevo; si una no aporta argumento, fuera.

## El orden (de lo que abre a lo que cierra)

**§0 · Apertura — el resultado en una tarjeta**
- 🆕 **Scorecard** (accuracy/Sharpe/equity/DSR de M10 vs M5/M8/B&H, con ✓/✗).
- *Por qué primero:* el tribunal ve la respuesta en 5 segundos; el resto del notebook la defiende.

**§1 · Resultado principal (todo el OOS, n=250)**
- 🆕 **Headline accuracy** M10 vs M5/M8/B&H (0.552, líneas azar y B&H).
- ✅ **Curvas de equity** (§C.2).
- *Por qué:* el qué (accuracy, lo que pide el tutor) + el enriquecimiento económico, juntos. Es el núcleo.

**§2 · Por qué SMCI y por qué no es trivial**
- ✅ **Margen por activo "solo SMCI bate a todo"** (§F.2) → justifica la elección del activo.
- ✅ **Agente 95 % corto + intervención STRATA** (§F.1) → explica por qué M5/M8/M10 no se separan y por qué
  el benchmark es justo.
- *Por qué:* adelanta y desactiva la primera objeción ("¿por qué este activo?", "¿no es suerte?").

**§3 · Robustez (la defensa fuerte)**
- ✅ **Robustez a la partición** — 3 splits val/test (§E.2). ← la que pediste.
- ✅ **Robustez al embargo** (§0bis).
- ✅ **Rolling-window** — consistencia temporal (§G).
- *Por qué:* el corazón de la defensa. Demuestra que el resultado **no depende** del corte, del embargo ni
  del sub-periodo. Tres ángulos de robustez seguidos = muy difícil de tumbar.

**§4 · Honestidad: qué se probó y NO mejoró**
- ✅ **Tuning en validación fracasa** (§A) → demuestra disciplina validación≠test.
- ✅ **Abstención: cobertura vs accuracy** (§C.3) → método de literatura probado y descartado.
- ✅ *(condensar en UNA figura)* **"lo que se probó"**: accuracy de configs fijas + métodos avanzados
  (triple-barrier/régimen/stacking) en una sola barra (fusionar §B y §C).
- *Por qué:* enseña rigor y exhaustividad; convierte los negativos en argumento ("probamos todo, honestos").

**§5 · Cierre — significancia honesta**
- 🆕 **Forest plot de significancia**: M10 vs B&H / vs M5 / vs azar, en el OOS completo **y** en los 3 splits,
  con p/IC en una figura.
- *Por qué:* cierra mostrando a la vez la **robustez** (gana siempre) y la **honestidad** (significancia
  borderline, no plena). Es la figura que te blinda: lo dices tú antes que el tribunal.

**Apéndice · Mecánica de STRATA (replicado sobre SMCI)** — 🟡
- 3 regímenes (vol y signo) · LL vs K (justifica K=3) · RAM como gate · SHAP de M10.
- *Por qué apéndice:* explica *cómo* funciona por dentro; no es el resultado, pero sostiene la
  interpretabilidad. Va al final para no romper el hilo argumental.

## Qué CORTAR del notebook actual (para no saturar)

- ❌ **§E (selección de burn-in, split 70d desequilibrado)** y **§E.1 (test 0.587 en ventana bajista)**:
  el split era malo y lo reemplaza §E.2 (robustez a la partición). Si acaso, una línea de texto, sin figura.
- ❌ **Figuras separadas de §B y §C**: fusionar en UNA ("lo que se probó y no mejoró").
- ❌ Las didácticas del ensemble/embargo (ilustración de varianza, indexado): **se quedan en
  `logic_esential.ipynb`**, no en el notebook de resultados.

## Qué CONSTRUIR (no existe aún)

1. 🆕 **Scorecard** (§0) — apertura.
2. 🆕 **Headline accuracy limpia** (§1) — hoy mezclada en §C.
3. 🆕 **Forest plot de significancia** (§5) — el cierre que blinda.

## Resumen del veredicto

**10 figuras, 6 secciones + apéndice.** Orden: *tarjeta → resultado → por qué SMCI → robustez (×3) →
honestidad → forest plot de cierre → (apéndice mecánica)*. Construir 3 nuevas (scorecard, headline, forest
plot), reutilizar 7 que ya existen, cortar las redundantes. Con eso el notebook **cuenta la historia robusta
sin ruido** y resiste el escrutinio del tribunal.
