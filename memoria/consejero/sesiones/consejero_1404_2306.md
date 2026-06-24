# Sesión con el consejero — 2026-06-23

<!-- sesión: 1c7f52c4-0310-40db-b4e2-3b894ae539a1 · registro automático (hook) + hitos (consejero) -->

Comenta con `> COMENTARIO:`, `> AYUDA:` o `> DUDA:` donde quieras (ver `../README.md`).

### Hitos

_(el consejero anota aquí decisiones, anotaciones y agentes llamados)_

---

## Conversación

#### [1404] Consejero

Contexto cargado. Antes de presentarme, una nota importante que no puedo callar: **mi propia ficha de persona (`consejero-tesis.md`) está calibrada al encuadre viejo (SMCI como caso central)**, pero el `MANUAL.md` se actualizó el **2026-06-23** y da un giro: el caso central pasa a ser **SPY**, con panel de 15 (10 beneficiados) y clustering; SMCI queda archivado. La regla del propio estilo es clara —*manda el MANUAL/notebook*—, así que me alineo con el encuadre nuevo y trabajo el **cap. 4** según `MARCO_PRACTICO_SPEC.md`.

---

Soy tu **consejera-revisora**. Tengo cargado el contexto vivo: MANUAL (giro a SPY), el SPEC del marco práctico (4 secciones: SPY · panel 10/15 · clustering · límites) y las cifras de referencia. Encuadre canónico actual: **caso central SPY** (M5 0,366 → AutoML 0,574, RAM = 100 % del P&L, McNemar AutoML vs M5 p=0,0002 / vs ZeroR p=0,90 nominal) y **panel** con M8 vs M5 ΔSharpe +0,66 [0,225, 1,157] SIG. Puedo: **revisar** tu `04_marco_practico.tex` sección a sección con anotaciones `% >>> CONSEJERO` (tú decides qué aplicas), **responderte dudas** citando la fuente exacta, y **orquestar subagentes** (siempre preguntándote antes).

Un aviso de coherencia que vigilaré: **`RESULTADOS_OBJETIVO.md §1` aún lleva las cifras viejas de SPY** (M5 0,384 / M8 0,436 / M10 0,539 CPCV) del notebook antiguo, que **no cuadran** con las del cap. 4 reescrito (M5 0,366 / AutoML 0,574, walk-forward). Eso es un desajuste a propagar antes de defender.

¿Por dónde empezamos: **reviso el `04_marco_practico.tex`** sección a sección, o atacamos primero ese **desajuste de cifras SPY** entre el .tex nuevo y `RESULTADOS_OBJETIVO.md §1`?

#### [0718] Raquel

necesito que te alinees con MARCO_PRACTICO_CONTEXTO.md es la nueva fuente de verdad

#### [0719] Consejero

Alineada. **`MARCO_PRACTICO_CONTEXTO.md` es ahora mi fuente de verdad** para el cap. 4, por encima del SPEC y del MANUAL viejo. Lo esencial que cambia respecto a lo que te dije hace un momento:

- **Universo = exactamente 10 activos** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG). **No** hay "15", ni "elegimos 10", ni apéndice. SMCI es **uno de los 10**, no caso aparte. Línea roja dura (§VII, §XIV.12).
- **Headline SPY** (n=251): M5 0,366 · M8 0,442 · M10 0,494 · **AutoML 0,574 (+2,68)** · ZeroR/B&H 0,566 (+2,21). McNemar AutoML vs M5 p=0,0002; vs ZeroR p=0,90 (nominal).
- **Pooled-10 ΔSharpe**: M8 +0,60 [0,05, 1,22] · M10 +1,12 [0,39, 1,84] · AutoML +1,08 [0,40, 1,85]. **Bonferroni: M10/AutoML pasan, M8 no.**

Déjame confirmar contra el propio `.tex` un par de conflictos que ya huelo entre el cap. 4 recién escrito (commit `a6697a1`) y este CONTEXTO nuevo:

#### [0719] Consejero

El `.tex` ya está más alineado de lo que esperaba: la única coincidencia de "15/apéndice" es el comentario de cabecera (línea 6) que ya lista los 10 "sin apéndice", y no aparecen ni el `p=0,034`+LOO de la ley ni el `+0,66` del pooled que temía. El fichero está **modificado sin commitear** (git lo marca `M`), así que ya lo has tocado. Déjame leer el estado actual real antes de marcar nada:

#### [0720] Consejero

El cuerpo está en `chapters/cap4_parts/` (s1_spy, s2_panel, s3_patrones, s4_limites). Mi grep miró el fichero equivocado. Repito sobre las partes, que es donde viven las cifras en disputa:

#### [0720] Consejero

Confirmado: el cap. 4 **ya está alineado** con el CONTEXTO nuevo. Las cifras que temía que arrastrasen el enfoque viejo están todas correctas en `cap4_parts/`:

| Punto | CONTEXTO (verdad) | Estado en el `.tex` |
|---|---|---|
| Universo | 10 activos, **sin apéndice**, sin "de 15" | ✓ cabecera + cuerpo, sin rastro de "15/elegimos/apéndice" |
| Ley leverage | r=−0,56, p=0,093, **no p<0,05 ni LOO** | ✓ s3:53, s4:14/32 — enmarcada como tendencia α=0,10 |
| Pooled-10 ΔSharpe | M8 +0,60 [0,05,1,22] · M10 +1,12 · AutoML +1,08; **M8 no pasa Bonferroni** (−0,047) | ✓ s2:92/109 exacto |
| Headline SPY | M5 0,366 → AutoML 0,574 (+2,68); McNemar vs M5 0,0002/0,0074/0,051 | ✓ s1:131 |
| TOST / DiD | accuracy superior, Sharpe no concluyente; DiD +1,37 [0,20,2,60] p=0,008 | ✓ s2:134/180 |

El commit `a6697a1` era el enfoque viejo (15→10, ley p=0,034+LOO); el trabajo de hoy —los cuatro `s*.tex`, **sin commitear** aún— ya lo reescribió a este CONTEXTO. Estoy operando con `MARCO_PRACTICO_CONTEXTO.md` como fuente única para el cap. 4 de aquí en adelante.

Una cosa que sí conviene propagar pronto, no en el `.tex` sino en los docs raíz: **`RESULTADOS_OBJETIVO.md §1` sigue con las cifras viejas de SPY** (M5 0,384 / M10 0,539 CPCV) que el CONTEXTO §X manda actualizar al headline §V. Eso es trabajo de `narrativa-coherencia`.

Mi recomendación para empezar: que **revise sección a sección los cuatro `s*.tex`** con anotaciones `% >>> CONSEJERO` —ahora que las cifras cuadran, el riesgo está en prosa, estilo y huecos de cita, no en los números. ¿Arranco por `s1_spy.tex`, o prefieres que primero dispare la actualización de `RESULTADOS_OBJETIVO.md §1` con `narrativa-coherencia`?

#### [0734] Raquel

STRATA rescata al agente que pierde, disciplina su riesgo y dibuja con honestidad dónde funciona.  reescribe

#### [0735] Consejero

Esa frase tiene dos problemas: **repite** lo que ya dice la frase anterior (rescate + riesgo aparecen dos veces seguidas) y el ritmo es una **tripleta uniforme** de verbos ("rescata… disciplina… dibuja…"), que es justo el patrón que delata IA. Además "dibuja con honestidad" es metáfora floja para un cierre de intro.

Mi recomendación es no reescribirla en redundante, sino **subir el encuadre de valor** que la frase previa no da (el listón es el agente, no el mercado — CONTEXTO §I), que de paso elimina la repetición:

**Opción A (recomendada) — añade el plano de valor:**
> Su listón es el propio agente, no el mercado: rescata a un decisor que pierde y le acota el riesgo, con la honestidad de señalar dónde no llega.

**Opción B (conservadora) — mantiene tu idea, solo limpia el estilo:**
> STRATA rescata al agente que pierde y le pone disciplina al riesgo, sin esconder dónde no llega.

A introduce el "listón = el agente" (claim del CONTEXTO, fuerte para desactivar el "¿y esto gana dinero?" del tribunal); B se queda pegada a tu original. Las dos rompen la tripleta y matan "dibuja".

¿Aplico la **A**, la **B**, o ajustas alguna palabra antes?

#### [0735] Raquel

no me gusta ninguna

#### [0736] Consejero

Vale — entonces el problema no es el arranque sino lo que estoy metiendo yo. Las dos te sonaban porque cargaban el "no alfa / dónde no llega" en negativo, y a ti eso no te beneficia subirlo a bandera en la intro.

Le doy la vuelta: **tu primera mitad se queda** ("STRATA rescata al agente que pierde y disciplina su riesgo") —es tu voz y son dos verbos limpios— y solo cambio la cola floja, sin tripleta y sin "dibuja", en tono afirmativo:

- **C:** …y disciplina su riesgo, honesta sobre dónde funciona y dónde no.
- **D:** …y disciplina su riesgo, con la frontera de su alcance medida, no escondida.
- **E:** …y disciplina su riesgo. Dónde funciona y dónde no, lo medimos.

Antes de seguir disparando variantes, dime el rumbo para no fallar otra vez: ¿qué quieres que pese esta frase de cierre?

#### [0736] Consejero

Punch afirmativo, con la honestidad como nota breve. Esa es la **C**. La aplico:

#### [0737] Consejero

Aplicado en [04_marco_practico.tex:13](tesis/chapters/04_marco_practico.tex#L13):

> STRATA rescata al agente que pierde y disciplina su riesgo, **honesta sobre dónde funciona y dónde no.**

Cierra en lo que sí hace, la honestidad queda como nota breve, y fuera la tripleta y el "dibuja".

De paso, dos cosas que vi en esa misma frase y la anterior (no las toco sin tu OK):
- Línea 13 abre raro: *"La respuesta es, en orden, el aprendiz rescata…"* — pediría dos puntos: *"La respuesta es, en orden: el aprendiz rescata…"*.
- Línea 11: *"AI Hedge **Found**"* — es **Fund** (aparece también en la portada del cap, conviene unificar).

¿Las arreglo, o sigo y te dejo revisar a tu ritmo?

#### [0737] Raquel

FATAL

#### [0737] Consejero

Compila pero **no genera PDF**: dos fallos. El FATAL son las **figuras que aún no existen** (la fase de acabado que quedaba pendiente) — esas dan `LaTeX Error: File not found` y tumban `pdflatex`. Detrás hay **10 citas sin definir** en el `.bib`. Déjame sacar la lista completa de ambas:

#### [0737] Consejero

Diagnóstico claro. **Las 10 citas son falsa alarma**: están todas en `bibliography.bib`, salían "undefined" solo porque era la primera pasada antes de `biber`. Se resuelven solas al recompilar.

**El FATAL real: 15 figuras del cap. 4 que aún no existen.** El `.tex` ya las referencia, pero ningún `figuras/cap4_*` está generado:

```
cap4_equity_spy        cap4_equity_panel        cap4_forest_pooled
cap4_regimenes_spy     cap4_scores_detectores   cap4_confusion_spy
cap4_atribucion_capas  cap4_sensibilidad_umbrales  cap4_heatmap_accuracy
cap4_tost_2x2          cap4_gate_ram            cap4_did_regimen
cap4_scatter_leverage  cap4_pca_clusters        cap4_casos
```

Es la fase de acabado que quedaba pendiente. Déjame ver si existe el script de export y cómo de caro es generarlas:

#### [0738] Consejero

Ojo — hay un **desajuste de nombres**, no solo figuras ausentes. En `tesis/figuras/` **sí hay 12 figuras** del cap. 4, pero con los **nombres viejos** (`cap4_equity`, `cap4_forest_significancia`, `cap4_ram_gate`, `cap4_regimen_precio`…). El `.tex` reescrito hoy referencia **nombres nuevos** (`cap4_equity_spy`, `cap4_forest_pooled`, `cap4_gate_ram`…). Hay un `REQUERIDAS_cap4.md` que parece el contrato. Déjame leerlo y encontrar el script generador:

#### [0738] Consejero

Ya tengo el cuadro completo. Esto es lo que pasa:

- **`REQUERIDAS_cap4.md` es el contrato**: 15 figuras nuevas (F4.1–F4.15), cada una con su JSON. Las 12 que hay en `tesis/figuras/` son **de la era SMCI y el propio contrato dice que NO sirven**.
- Existe el generador **`experiments/export_figs_cap4.py`**. Déjame ver si ya produce los nombres nuevos o sigue en los viejos:

