# Conversación — organización y expansión del notebook STRATA

Hilo de trabajo sobre `notebooks/strata_tfg.ipynb` y la documentación del TFG, **2026-05-21 → 2026-05-27**, rama `feat/panel-multiactivo`. Mensajes del usuario verbatim entre comillas; resúmenes de mis respuestas en prosa.

---

## Resumen rápido

1. **Prior direccional de RAM re-signado por activo** (data-driven sin look-ahead). NVDA M8 +0,66 → **+0,95**; SPY/BAC intactos. Documentado en `docs/decisiones.md` §9 y `docs/hallazgos_strata.md` §3.
2. **Dos documentos para la memoria**: `docs/marco_teorico.md` (guion del marco teórico) y `docs/explicacion_notebook.md` (paso a paso del notebook).
3. **Sección §14 TSLA y §15 XLE** añadidas al notebook (HMM+leverage, M1–M9, equity/drawdowns, sizing, **matriz Diebold-Mariano**). DM matrices también en NVDA y BAC. TSLA M8 **+1,137** — el único activo donde STRATA bate a la mejor clásica.
4. **Aclaración "¿por qué M8 rinde menos que B&H?"** + columna **`Ret@σBH`** (retorno a riesgo común). TSLA M8 +15 % bruto → **+122 %** a vol de B&H.
5. **Decisión diferida** sobre presentación del retorno (vol-targeting). Documentada en `docs/decisiones.md` §10 y BITACORA.
6. **Screening de candidatos UNG/XBI/MSTR** con `experiments/tuning/preview_candidates.py` — **refutó la hipótesis "leverage invertido"** (los 3 salen clásicos). Usuario eligió cachear **UNG y MSTR** para probar la vía complementaria ("STRATA bate a B&H sobre activo en caída libre").
7. **Generación de caché del agente** vía `gen_agent_cache.py` con **watchdog + `caffeinate`**. UNG 401/401, MSTR 401/401 (~5 h con auto-recuperación tras varios cuelgues del proveedor y sueño del Mac).
8. **§16 UNG y §17 MSTR** añadidas. UNG = caso *cuchillo cayendo* (M8 supera a B&H y a la mejor clásica vía leverage clásico + corto en Crisis). MSTR = **contraejemplo** (cambio estructural software→bitcoin-proxy no recogido en calibración). Panel pasa a §18, repro a §19. Correlación cross-seccional reforzada a **−0,54**.

---

## Cronología

### 1. RAM-resigned priors (2026-05-21)

> *"haz una prueba con los priores de ram reasignados por activo, muestrame los resultados y decidimos si lo modificamos en el notebook"*

Construí `experiments/tuning/diagnose_ram_resigned.py`: para SPY/NVDA/BAC, computa el signo del retorno medio por régimen en calibración (sin look-ahead) y deriva un prior data-driven `regime_dir = (Calma, 0, Crisis)`. Resultados:

| Activo | Crisis calib. | Prior derivado | M8 default → re-signado |
|---|---|---|---|
| SPY | −4,2 bps | Crisis ⇒ short (= default) | +0,62 → +0,62 (idéntico) |
| NVDA | **+17,3 bps** | Crisis ⇒ **long** | **+0,66 → +0,95** |
| BAC | −5,8 bps | Crisis ⇒ short (= default) | +0,86 → +0,86 (idéntico) |

Solo NVDA cambia (leverage invertido). SPY cross-check se mantiene verde (Δ 3,3e-7).

> *"Sí, como default"* (a la pregunta sobre adopción)

Lo adopté como default en el notebook: nuevo helper `regime_dir_from_calib()`, función `ram_detector` parametrizada por `regime_dir`, propagación a `supervised_sizes`. Actualicé `docs/decisiones.md` §9, `docs/hallazgos_strata.md` §3, `docs/README.md`, memoria `strata-nvda-per-asset-hmm` (descripción → "todo por activo"), y una nueva entrada [2026-05-21][Decisión] en `BITACORA.md`.

### 2. Documentos didácticos del TFG

> *"create marco_teorico.md file with the key points i should relate in the theorical framework and why."*

Creé [`docs/marco_teorico.md`](docs/marco_teorico.md): cuatro bloques (modelado del mercado, detección/supervisión, validación honesta, gestión de riesgo) con cada concepto anclado a la sección del notebook que lo usa, más referencias clásicas (Black 1976, Hamilton 1989, López de Prado 2018, Bailey & López de Prado 2014, Diebold-Mariano 1995, etc.).

> *"hazme otro documento explicacion_notebook.md explicandome paso a paso que se hace en el notebook y como lo he programado para explicarselo a mi profesor de manera facil y clara"*

Creé [`docs/explicacion_notebook.md`](docs/explicacion_notebook.md): guía pedagógica con la idea en una frase, las cinco decisiones de programación defendibles (atómico, *inline*, dos fronteras agente+H2O, determinismo, por activo, cross-check de fidelidad) y un recorrido §0→§14 + las cinco ideas a destacar.

> *"ahora explicame brevemente cada funcion o codigo que aparece en el notebook de manera ordenada tengo que entenderlo bien"*

**Aquí entré en un fallo de modo (loop)** que el usuario interrumpió:

> *"esta en bucle"*

Lo reconocí, expliqué que había generado intentos repetidos de "voy a leer el código" sin actuar, y pedí permiso para reintentar — pero no llegué a producir un `funciones_notebook.md` limpio en esa pasada. El usuario pasó a otra tarea.

### 3. Sección TSLA y XLE + DM matrices "en todos"

> *"tengo el cache de TSLA y XLE cargado, replica el analisis para cada activo y añade las secciones al notebook, además quiero que añadas en todos el test de mariano y la matriz de p valores"*

Verifiqué cachés (TSLA 401 ficheros, XLE 398) y parquets de precio. La estructura del panel §14 ya tenía `run_asset_pipeline()` con auto-inclusión por caché, pero el usuario quería **secciones de inmersión propias** (como NVDA/BAC), no solo filas en el panel.

Plan en modo plan, aprobado. Implementación:

- Helper reutilizable `plot_dm_matrix(nr_dict, common, asset)` (réplica del bloque DM de §8).
- **Matriz DM 9×9 añadida a NVDA, BAC, TSLA, XLE** (SPY ya tenía la suya en §8).
- **Sección §14 TSLA** y **§15 XLE** (9 celdas cada una, mirror de BAC: header, HMM+leverage, §X.1, M1–M9, tabla 9×7, equity+drawdowns, sizing, DM, *Lectura*).
- Panel renumerado §14 → §16; sembrado de `PANEL` con TSLA/XLE; trim de `dm_tickers` para evitar duplicar matrices ya presentes en las inmersiones.

Resultados (causal neto, OOS ~401 sesiones):

| | SPY | NVDA | BAC | TSLA | XLE |
|---|---|---|---|---|---|
| Crisis (bps) | −4,2 | +17,3 | −5,8 | **+39,9** | +0,7 |
| M2 GARCH×HMM | +0,810 | +0,992 | +0,875 | **+0,027** | +0,925 |
| M8 override | +0,621 | +0,945 | +0,855 | **+1,137** | +0,298 |

**TSLA** es el único activo donde M8 supera a B&H **y** a la mejor clásica — caso "alfa" complementario al techo de supervisión. Cross-check SPY verde, pytest 106 verde, panel a 5 activos con correlación calidad-B&H vs ventaja-STRATA = **−0,51**.

### 4. Vol-targeting y `Ret@σBH`

> *"TSLA: ... PORQUE EL RETORNO ES MENOR EN M8 QUE CON B&H?"*

(Note: la tabla pegada era en realidad la de XLE — se lo señalé.) Expliqué que **el retorno bruto no es comparable** entre estrategias a vol distinta. M2/M4/M8 operan a *vol-target* ~10 % (sin apalancamiento por `clip(target_vol/σ, 0, 1)`); B&H va al 100 % (vol real del activo). Menos exposición → menos retorno bruto, **pero el Sharpe es invariante al escalado**: la comparación justa es Sharpe o retorno reescalado a riesgo común.

> *"1. POR QUÉ NO VAMOS AL 100%?"*

Detallé los tres motivos: (1) comparabilidad cross-activo, (2) STRATA = disciplina de riesgo (no maximizar retorno), (3) tope a 1,0 = sin apalancamiento, decisión conservadora. Y la clave teórica: el nivel de exposición es una palanca **aparte** de la calidad de la estrategia.

> *"no entiendo el problema, no entiendo porque no se puede escalar, si no no podemos comparar las metricas"*

Aclaré que **no estoy diciendo que no se pueda escalar** — la afirmación es la opuesta: escalar es legítimo y preserva el Sharpe; lo que no es comparable es el retorno **bruto**, no el escalado.

> *"Si"* (a la oferta de añadir una columna escalada al notebook)

Añadí el helper `ret_at_vol(r, target_vol)` (escalado lineal, preserva Sharpe) y la columna **`Ret@σBH`** en las 5 tablas de métricas, escalada a la vol del propio B&H del activo. Plus una nota explicándola una sola vez tras la tabla de SPY (aplica a todas).

| Activo | M8 Retorno bruto | M8 `Ret@σBH` | B&H Retorno | Sharpe ratio |
|---|---|---|---|---|
| TSLA | +15,1 % | **+122,1 %** | +53,7 % | M8 1,14 > B&H 0,75 |
| SPY | +6,0 % | +15,7 % | +31,7 % | M8 0,62 < B&H 1,09 |
| XLE | +3,7 % | +6,9 % | +27,2 % | M8 0,30 < B&H 0,75 |

> *"vamos a dejar esta decision para otro momento documentala en decisiones y bitacora es muy importante"*

(El "decisión" diferida: si en la entrega final se **sustituye** la columna "Retorno" bruta por `Ret@σBH` o se mantienen ambas.) Documentada como **§10** en `docs/decisiones.md` (Qué/Por qué/Cómo/Qué cambia/Decisión abierta/Cómo se defiende + fila en el resumen de una página) y como entrada `[2026-05-22] [Decisión]` en `BITACORA.md`.

### 5. UNG, XBI, MSTR — screening y caché

> *"replica el analisis del notebook para los siguientes activos que creo que el agente va a superar a b&h por el leverage invertido : UNG, XBI, MSTR"*

Comprobación previa: **ninguno** tiene caché del agente ni parquets de precio. Generar la caché es la tarea lenta y rate-limited (~400 LLM-decisiones/activo, ~días). En modo plan ofrecí tres caminos:

> *"Preview sin agente primero"* (respuesta del usuario)

Construí `experiments/tuning/preview_candidates.py`: descarga precio (yfinance), entrena HMM/GARCH propios sobre calibración, mide el signo del leverage (Crisis bps en calibración) y computa M1–M4 sin agente. Hallazgo:

| Ticker | Crisis (bps calib.) | Prior | B&H Sharpe | B&H Ret | Veredicto |
|---|---|---|---|---|---|
| UNG | −19,8 | clásico | −0,37 | **−50,9 %** | candidato débil (premisa refutada) |
| XBI | −4,1 | clásico | +0,74 | +30,0 % | candidato débil |
| MSTR | −10,4 | clásico | +0,14 | **−31,9 %** | candidato débil |

**La hipótesis "leverage invertido" se refuta a nivel de premisa** para los tres. MSTR sale clásico porque su calibración 2000–2024 está dominada por la era software/puntocom (carácter bitcoin-proxy solo desde 2020).

Pero hay un ángulo alternativo: **UNG y MSTR tienen B&H catastrófico** → STRATA podría batirlo por la otra vía (corto en Crisis con leverage clásico bien signado), no por melt-up. Pregunté qué hacer.

> *"Cachear UNG + MSTR"*

XBI descartado (B&H sólido → sería techo de supervisión).

### 6. Generación de caché — la saga del watchdog

Lanzamiento inicial en background: `gen_agent_cache.py` para UNG luego MSTR, ~32 s/decisión nominal.

> *"como va?"* (×varias veces)

Reportes intermedios: UNG 81 → 121 → 145 → 155… luego **se atascó en 155 con el log frozen 4h45**. Diagnosis: el proceso vivo a 0 % CPU, estado `S` (bloqueado en una llamada de red sin timeout) — **colgado**. Plus el Mac probablemente se durmió por la noche (pmset confirmó DarkWakes).

Solución: maté el job vía `TaskStop` y relancé con **watchdog** (kill+restart si pasan 2 min sin avance) más **`caffeinate -dimsu -w $$ &`** (impide que el Mac se duerma). Reanudó desde lo cacheado (155), pasó del punto atascado (162→163→…). Lento (~70 s/decisión por proveedor degradado de noche) pero **autónomo y resiliente**.

> *"si"* (a "lo dejo corriendo y te aviso al terminar")

Madrugada del 2026-05-26 a las 05:12, el watchdog reportó **ALL DONE: UNG=401, MSTR=401**.

### 7. Sección UNG (§16) y MSTR (§17) — resultados

Inserté las 18 celdas nuevas (9 por activo, mismo template que TSLA/XLE incluida la columna `Ret@σBH`), seedeé `PANEL`, añadí UNG/MSTR a `DEEP_DIVE`, renumeré panel §16→§18 y repro §17→§19. Compile OK; re-ejecuté el notebook (~20 min).

**UNG (gas natural):**

- B&H **−50,5 %**, MaxDD **−66 %** (catastrófico, como esperado por *contango*).
- **M5 agente +0,33** — único activo del panel con M5 positivo (el agente acierta direccionalmente sobre un activo en caída libre).
- **M7 reduce +0,40** (mejor del activo), **M8 override +0,18**.
- A `Ret@σBH`: M7 **+8,8 %** vs B&H **−50,5 %**.
- **UNG entra como segundo activo donde M8 supera a la mejor clásica**, vía clásica: leverage clásico bien signado + corto en Crisis sobre activo que se desploma. Es la **otra cara** de TSLA (cuchillo cayendo vs melt-up).

**MSTR (proxy de bitcoin):**

- B&H paradójico: Sharpe **+0,25** pero Ret **−19,9 %**, MaxDD **−81,5 %** (vol 84 % erosiona el compuesto).
- **M2 colapsa: −1,40** — el HMM calibrado en la era software/puntocom genera un *sizing* completamente fuera de fase en la era bitcoin-proxy.
- M8 override **−0,04** (flat). El prior mis-signado para la era nueva neutraliza el rescate.
- M9 ML+IA Sharpe +1,88 (DSR 1) pero a vol 0,7 % → `Ret@σBH +614 %` es **artefacto del escalado lineal** (k≈120 sobre serie casi-flat). El Sharpe es real, el retorno reescalado pierde significado.
- **MSTR = contraejemplo defendible**: cuando los regímenes calibrados no representan el carácter actual del activo (cambio estructural), STRATA no puede arreglarlo. Límite del marco (estacionariedad del régimen).

**Panel (ahora 7 activos):**

- Correlación calidad-B&H vs ventaja-STRATA: **−0,54** (más fuerte que −0,51 con 5).
- M8 supera a B&H en **4/7** (UNG, TSLA, BAC, NVDA); supera a la mejor clásica en **2/7** (UNG, TSLA).
- Las **dos vías** del mecanismo condicional quedan visibles: TSLA = melt-up con leverage invertido; UNG = cuchillo cayendo con leverage clásico.

Cross-check SPY verde (Δ 3,27e-7), pytest 106 verde. BITACORA actualizada `[2026-05-26][Hallazgo]`; memoria `panel-multiactivo-status` reescrita (5→7 activos, dos vías, MSTR como contraejemplo).

### 8. Vista del notebook en el IDE

> *"no veo lo ultimo en el notebook"*

Verifiqué que el `.ipynb` en disco estaba al día (mtime 08:05, headers §16/§17/§18/§19 presentes, 96 celdas). El IDE tenía la versión cacheada. Indiqué cerrar y reabrir la pestaña o "Revert File".

### 9. Guardar la conversación

> *"guarda toda la conversacion y almacenala en chats/organize_notebook.md"*

(Este documento.) En este punto el disco del Mac está al **100 %** (205 MB libres de 460 GB) — `Bash` falla al crear su fichero de output en `/private/tmp`. Estoy escribiendo el chat con `Write` directamente al destino.

---

## Continuación posterior (referencia, fuera del hilo principal)

Según `memory/panel-multiactivo-status.md` (actualizada por ti / linter mientras tanto), tras este hilo se ha empezado una **nueva ronda de candidatos growth-invertido** con `experiments/tuning/screen_candidates.py`: elegidos **SMCI, ROKU, MARA** (Crisis +16,1 / +31,8 / +55,8 bps respectivamente). El caching está en curso pero **bloqueado por disco lleno** (SMCI 146/403, ROKU 0, MARA 0). En cuanto liberes disco, la memoria documenta el procedimiento de reanudación (driver con `caffeinate` + watchdog 2-min).

---

## Ficheros creados o modificados en este hilo

- `notebooks/strata_tfg.ipynb` — RAM re-signed, helper `plot_dm_matrix`, DM en NVDA/BAC, sección TSLA §14, XLE §15, UNG §16, MSTR §17, panel §18, repro §19, columna `Ret@σBH`. Pasó de 56 a **96 celdas**.
- `docs/marco_teorico.md` (nuevo).
- `docs/explicacion_notebook.md` (nuevo).
- `docs/decisiones.md` — añadidas §9 (NVDA priores) y §10 (vol-targeting + decisión diferida); fila 10 en el resumen.
- `docs/hallazgos_strata.md` — §3 reescrita con la tabla de 3 variantes (rég. S&P / rég. propio leverage / rég. propio re-signed).
- `docs/README.md` — entradas nuevas para marco_teorico y explicacion_notebook; mapping line actualizada.
- `BITACORA.md` — entradas `[2026-05-21][Decisión]` (prior RAM por activo), `[2026-05-22][Decisión]` (vol-targeting diferido), `[2026-05-22][Hallazgo]` (TSLA/XLE), `[2026-05-26][Hallazgo]` (UNG/MSTR).
- `experiments/tuning/diagnose_ram_resigned.py` (nuevo).
- `experiments/tuning/preview_candidates.py` (nuevo).
- Memoria: `strata-nvda-per-asset-hmm` (descripción ampliada) y `panel-multiactivo-status` (varias revisiones).

Sin commitear; sin merge a `main`. Rama `feat/panel-multiactivo`.
