# Revisar al volver — dudas y decisiones que dejo abiertas

> Raquel: 2026-06-23. Me pediste que ejecutara sin esperar tu respuesta y que dejara aquí todo lo que dudara.
> Esto es eso. Cada punto trae **mi default** (lo que aplicaré para no bloquearme) y **por qué**, para que solo
> tengas que confirmar o corregir. Prioridad: 🔴 alta (cambia el cap. 4), 🟡 media, ⚪ housekeeping.

## Lo que ya he hecho (no requiere tu acción, pero revísalo)

1. **MANUAL.md reescrito** al nuevo enfoque: SPY central + panel 15 (10 beneficiados) + clustering; tesis
   T1/T2/T3 del SPEC; SMCI marcado como histórico; las tres derivadas de STRATA (M8 / M10 / AutoML-M10). El
   brief que manda para el cap. 4 es `MARCO_PRACTICO_SPEC.md`; el MANUAL solo lo resume.
2. **Hook + watcher del notebook**: `.claude/hooks/marco_practico_watch.py` (registrado en `settings.json` en
   `SessionStart` y `UserPromptSubmit`) + un watcher en background. En cuanto aparezca
   `notebooks/STRATA_marco_practico.ipynb` (≥20 KB y estable), me pongo a reescribir el cap. 4 según el SPEC sin
   pedirte permiso. Idempotente: al terminar marco `cap4_written=true` en
   `.claude/hooks/.marco_practico_state.json` para no repetirlo.
   - ⚠️ **Limitación honesta del disparador:** el watcher en background vive mientras la sesión/app siga abierta.
     Si cierras todo y otro de tus Claude construye el notebook con todo cerrado, la escritura automática no
     arranca sola hasta que **vuelvas y escribas un mensaje** (ahí salta el hook de `settings.json`, que es
     persistente, y retomo). Es decir: como muy tarde se escribe en tu primer mensaje al volver. Si quieres
     garantía total aunque cierres el portátil, dímelo y lo paso a un mecanismo de SO (launchd/cron).

## 🔴 Decisiones del SPEC §6 que tú tenías que fijar (aplico un default para no parar)

3. **§6.1 — Criterio de inclusión de los 10 activos del panel.** El SPEC ofrece A (discrepancia agente↔régimen,
   decisión #16), B (leverage effect ex-ante) o C (ambos).
   - **Mi default: C (ambos), reportando también A y B por separado.** Razón: es el más defendible ante tribunal
     (criterio mecanístico ex-ante doble) y el análisis de panel ya sugiere que el régimen es direccional donde el
     leverage es fuerte. **Riesgo que te señalo:** con leverage fuerte solo hay ~7 activos (DIA, SPY, IWM, QQQ,
     XLF, XLE, XLK); si el criterio C deja menos de 10, el «10 de 15» no cuadra y habrá que relajar a B o a A. Si
     pasa, lo resuelvo dejando el criterio que dé un panel coherente y lo documento; **confírmame si prefieres
     fijar A, B o C de entrada.**
4. **§6.2 — «Ganar con p significativo» en ventanas/particiones/régimen.**
   - **Mi default:** α = **0,10** (defendible en muestra corta); tests = **block-permutation pareado** +
     **bootstrap estacionario pareado**; «ganar» = la mejor derivada de STRATA supera **a la vez al agente (M5) y
     a la mejor trivial (B&H/ZeroR)**; signo de la mejora consistente en **≥ ⌈2/3⌉ de las sub-ventanas**.
   - **Duda real:** ¿los resultados de robustez por ventana/partición **ya están** en algún JSON canónico del
     panel mm25, o el notebook tiene que generarlos? Si hay que generarlos, lo encargo a `@ejecutor-experimentos`
     con pre-registro; dímelo si ya existen para no recomputar.
5. **§6.3 — Estrategia ganadora canónica de la curva de equity headline.**
   - **Mi default:** la curva headline muestra **la mejor derivada de STRATA por activo**; para el headline de SPY
     uso **M10** (es el meta-learner canónico y el que el SPEC pone como eje). Si en algún activo gana M8 o AutoML,
     lo digo y la curva de ese activo enseña la que toque. Confírmame si prefieres una única derivada fija para
     todo el capítulo.

## 🟡 Coherencia del resto de documentos (SMCI sigue vivo en varios sitios)

6. Solo me autorizaste a tocar el **MANUAL**. Estos siguen con el enfoque viejo (SMCI) y **chocan** con el nuevo;
   los dejo intactos hasta que me digas, pero los señalo porque los gates los leen:
   - `memoria/estructura_cap4.md` — es íntegramente SMCI y **queda superado por `MARCO_PRACTICO_SPEC.md`**. Para
     el cap. 4 seguiré el SPEC, no este fichero. **Propongo retirarlo o reescribirlo apuntando al SPEC.**
   - `DECISIONES_ESENCIALES.md #13–17` — pivot a SMCI; habría que añadir una decisión nueva «#18: caso central =
     SPY + panel; SMCI = histórico» para no dejar contradicción.
   - `RESULTADOS_OBJETIVO.md §1bis` — cifras canónicas de SMCI; las dejo como histórico (el §1 de SPY sigue
     válido como método). El cap. 4 tomará cifras del notebook nuevo, no de aquí.
   - **No los toco sin tu OK** (no es lo que me pediste). Dime si quieres que propague el cambio a estos tres.

7. **`tesis/chapters/04_marco_practico.tex` actual es 100 % SMCI.** Cuando se dispare, lo **reescribo entero**
   según el SPEC (SPY/panel/clustering/límites). Guardaré el viejo en `memoria/historico_redacciones/cap4/` antes
   de sobrescribir, por si quieres recuperarlo.

## 🟡 Cifras de SPY para el cap. 4

8. Las cifras de SPY que tengo documentadas (`RESULTADOS_OBJETIVO.md §1`: M5 0,384 / M8 0,436 / **M10 0,539**
   / B&H 0,569) son **CPCV** (contraste, mira el futuro), **no el desplegable walk-forward**. El SPEC pide el
   resultado desplegable. **Default:** tomaré como canónicas las cifras de SPY **walk-forward / config mm25 que
   produzca el notebook nuevo**, y usaré las CPCV solo como contraste etiquetado. Si el notebook no recalcula SPY
   en walk-forward, lo marco como hueco y lo encargo. Avísame si el número de SPY que quieres de titular es otro.

## ⚪ Housekeeping (haré salvo que digas lo contrario)

9. Al escribir el cap. 4: lo paso por el pipeline de gates (redactor-tesis → rigor-matematico/harvard →
   experto-citas → estilo-raquel → detector-ia/plagio → narrativa-coherencia → latex-experto) y respeto
   `ESTILO_Y_ANTIIA.md` + `correcciones_aprendidas.md` (sin guion-muletilla, sin meta-comentarios, sin «no es X
   sino Y», conectores variados, primera persona plural, cifras con coma decimal y trazadas a JSON).
10. Marcaré `STRATA_SMCI.ipynb`, `strata_canonical.ipynb` y `decision_automl.ipynb` como **obsoletos/archivo** en
    `MEMORY.md` (lo manda el SPEC §9) y actualizaré la memoria que decía que STRATA_SMCI era el entregable.
11. **Rama y commit:** estás en `feat/quant-validation-panel`. Commiteo el cap. 4 de forma atómica y hago push a
    esa rama (como tienes por costumbre), nunca a main ni con `.env`. Si quieres otra rama para el cap. 4, dímelo.

## ⚪ Una observación para que la tengas presente

12. El tutor te sugirió meter el nombre del activo en el título del TFG para blindarte del «¿cuántas veces lo
    probaste?». Eso era por el caso único SMCI. Con **SPY central + panel de 15 con criterio de inclusión
    mecanístico ex-ante**, esa objeción se responde por el diseño (no por el título). Quizá el título ya no necesite
    el nombre del activo; lo dejo a tu criterio, no es bloqueante para el cap. 4.

---

# ACTUALIZACIÓN 2026-06-23 — el cap. 4 YA ESTÁ ESCRITO

El notebook `STRATA_marco_practico.ipynb` apareció, el disparador funcionó y he reescrito el cap. 4 entero
sin esperarte, siguiendo el SPEC. Resumen de lo que tienes que mirar:

**Qué he escrito.** `tesis/chapters/04_marco_practico.tex` completo, estructura del notebook (§4.1 datos ·
§4.2 mecánica · §4.3 SPY · §4.4 panel+universalidad · §4.5 clustering · §4.6 robustez · §4.7 conclusiones).
Caso central SPY, panel de 15. **Todas las cifras trazadas a los JSON canónicos** (`decision_automl_prep.json`,
`panel_mm25_…_seed42.json`, `automl_importance.json`, `strategy_clustering15.json`, `spy_ablation_robustness.json`,
`m10_smci_valtest_robustez.json`). **Compila limpio** (latexmk exit 0, PDF regenerado, sin refs/cites rotas).
El cap. 4 viejo (SMCI) está guardado en `memoria/historico_redacciones/cap4/cap4_SMCI_pre-pivot_2026-06-23.tex`.

**Gates pasados (los he corrido como pediste):**
- `rigor-matematico`: **APROBADO CON CONDICIONES**. Todas las cifras cuadran con los JSON, sin look-ahead. Apliqué
  sus condiciones: nota al pie de que n varía por activo (246–259), nomenclatura unificada del bootstrap
  (estacionario pareado por bloques), «silueta máxima en k=3 para los tres métodos concordantes», y aviso de que
  la figura SHAP §4.4 usa la media del panel (no SPY) al exportarla.
- `detector-ia`: **PASS** (22%, umbral 30%).
- `estilo-raquel`: BLOQUEÓ por molde tripartito repetido, dos «no es X sino Y» y cross-refs de pasada. **Lo
  corregí todo** (disolví los «La primera/La segunda/La tercera», quité las negaciones, los «por tanto/por último»
  y la pregunta retórica, y saqué cifras encadenadas de la prosa).

**Lo que SÍ necesito que revises:**
- 🔴 **Figuras por exportar.** El .tex tiene 4 cajas-placeholder donde van las figuras (régimen HMM sobre SPY,
  equity curves, heatmap+SHAP, PCA clustering). Hay que **exportarlas del notebook a `tesis/figures/`** y
  sustituir las cajas por `\includegraphics`. No lo hago yo sin tus PNG; te lo dejo señalado en el .tex con
  comentarios `% [Figura por exportar — …]` y la celda de origen.
- 🟡 **El §6.1 (criterio de los 10 activos) quedó resuelto solo:** el notebook **muestra los 15**, no una
  selección de 10. Mejor así (responde el cherry-pick por diseño). El cap. 4 es honesto con los 15: en índices
  ganan las triviales (alcista), y donde una derivada de STRATA bate a las dos triviales es en MSTR (M8), SMCI y
  UNG (M10) y MARA (AutoML). Si tú querías la curación a 10, dímelo y la aplico.
- 🟡 **Encuadre del claim:** el notebook (y por tanto el cap. 4) lidera con «rescate del agente (SÍ significativo,
  accuracy y riesgo) + universalidad (el ML redescubre STRATA) + patrón», y deja como **nominal** el batir a las
  triviales. Esto es coherente con tu SPEC. Si quieres más peso en el canal de riesgo (drawdown/vol-targeting) que
  vi en los docs exploratorios, lo reforzamos.
- 🟡 La tabla maestra del panel y la pooled están en el cuerpo; revisa que el nivel de detalle te encaja para una
  memoria (vs. llevar algo a apéndice).

**Pendiente que NO toqué (sigue como en los puntos 6–8 de arriba):** `estructura_cap4.md` (SMCI, superado por el
SPEC), `DECISIONES_ESENCIALES` y `RESULTADOS_OBJETIVO §1bis`. Dime si propago el pivot a esos tres.
