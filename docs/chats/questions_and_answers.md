# Preguntas y respuestas — banco de defensa (STRATA)

Recopilación de las dudas de Raquel y sus respuestas, pensadas para la defensa ante el
tutor/tribunal. Cada entrada: la pregunta, la respuesta defendible con cifras, y —cuando
aplica— la frase lista para usar oralmente y el talón de Aquiles. Las cifras provienen del
notebook `notebooks/strata_canonical.ipynb` (pipeline causal: HMM filtrado + σ causal,
`signal_lag=1`); donde una decisión está pendiente de re-ejecución se marca.

Conversación origen: 2026-06-08. Decisiones metodológicas asociadas en `BITACORA.md`.

---

## Q01 — ¿El `size` que dice el agente pesa en STRATA, o todo se va a long/short entero?

**Respuesta.** Depende de si STRATA interviene, y es "todo o nada":
- **74% de los días (297/401): STRATA no toca nada** → la posición es exactamente el size
  del agente (~0.1–0.25). Ahí el size del agente es lo único que manda.
- **26% de los días (104/401): interviene** (los 104 por RAM; GSO y PSA nunca disparan
  solos en SPY) → en override-C **descarta signo y magnitud del agente** y los sustituye
  por `regime_sign · bound`, con `bound = min(1, target_vol/σ_t)`, `target_vol = 0.10`.

En los días intervenidos el `|size|` medio del agente es 0.227 y STRATA lo reemplaza por
una posición de magnitud media 0.900 (×4) con el signo volteado. **No siempre es ±1**:
solo en 34/401 días `|final_size|=1.0` (cuando σ≤10%); en 70 de los 104 intervenidos
`bound<1` y la posición queda en 0.65–0.99. El `bound` modula la magnitud según la
volatilidad; el signo siempre va al régimen.

**Matiz para el tribunal.** Como M8 usa la variante C (descarta el size), su mejora viene
del **signo**, no de la escala. La ablación que lo aislaría es la variante D (conserva
`|size|` del agente). Conviene tenerla a mano.

**Fuente.** `strata/intervention.py` (override C, líneas 116-161); notebook §5–§6.

---

## Q02 — ¿Cuáles son las features de M10? ¿Son las mismas que en el proyecto anterior? ¿Por qué?

**Respuesta.** Son **exactamente las mismas 22**, idénticas en número, nombres y
agrupación (confirmado en `_archivo_proyecto_anterior/outputs_canonicos/m10_ml_meta.json`
y BITACORA del proyecto anterior 2026-06-02):
- 15 del agente: por cada una de las 5 personalidades (Buffett, Wood, Druckenmiller,
  Burry, Ackman), su `{signo de acción, size, confianza}`.
- 3 de STRATA: `ram_score, psa_score, gso_score`.
- 4 de régimen: `calm_prob, stress_prob, crisis_prob, garch_sigma`.

**Por qué esas.** Principio de **paridad de información**: el XGBoost recibe exactamente lo
que tienen el agente y STRATA juntos, y nada más. Si le quitas o añades algo, el test
"¿el boosting bate a tu regla?" sería asimétrico e indefendible. Por eso se incluyen
incluso features que el modelo luego ignora (las 5 `*_size` y `gso_score` no se usaron
como split en el proyecto anterior): dárselas y que las descarte es un *hallazgo*;
ocultárselas sería objetable.

---

## Q03 — ¿Por qué M10 no bate a M8? ¿Y en el proyecto anterior sí?

**Respuesta.** En el proyecto anterior M10 (+0.69) **tampoco batía** a M8 (+0.66): eran
indistinguibles (DM p≈0.75), le sacaba 3 milésimas de Sharpe (ruido). La tesis siempre fue
"son indistinguibles", no "M10 gana".

Auditoría de mi M10 (no había fuga; cobertura OOF = 5 por día, correcta). El gap aparente
venía de **mi mapeo de posición**: usé `w = 2·p1 − 1` (continuo) → Sharpe 0.51; la posición
natural de un clasificador direccional es `w = sign(p1 − 0.5)` (el "1/−1" del problema) →
Sharpe **0.71 ≈ M8 (0.75)**. Corregido a direccional:
- **M10 = +0.71 ≈ M8 = +0.75**, indistinguibles (DM p=0.61; ΔSharpe −0.04, IC [−1.93,+1.94]
  contiene 0). TOST **no** declara equivalencia (p=0.45): con N≈400 "no hay diferencia
  detectable" ≠ "son iguales", pero el meta-learner **no bate** la regla.
- Log-loss OOF mediana 0.912 ≈ el 0.914 del proyecto anterior (coherencia de validación).

**Por qué un meta-learner con todas las features no bate a la regla a mano:** M8 no se
"entrena", no paga gap de sobreajuste; M10 sí lo paga al validarse out-of-fold con N≈400.
Es justo lo que predice López de Prado. **El XGBoost no mejora la regla: la redescubre**
(SHAP pone arriba las features de STRATA; la ablación sin STRATA hunde a M10 de +0.71 a
+0.21). Decisión del mapeo documentada en `BITACORA.md` (para que no parezca cherry-picking:
es el mapeo canónico del clasificador, el del proyecto anterior).

---

## Q04 — ¿Cómo veo los cortes de las variables a los que llega XGBoost en M10?

**Respuesta.** Con `booster.trees_to_dataframe()`. XGBoost **no elige un corte**, hace
cientos por variable repartidos por todo el rango (modelo sobre 401 días):

| feature | nº splits | corte mediano | rango |
|---|---|---|---|
| psa_score | 538 | 0.005 | 0.005–0.012 |
| garch_sigma | 486 | **0.130** | 0.085–0.545 |
| stress_prob | 398 | 0.933 | 0–0.999 |
| ram_score | 357 | 0.002 | 0–0.9999 (bimodal: corta cerca de 0 y de 1) |

Dos lecturas: en `ram_score` corta en los dos extremos (confirma que el régimen es
cuasi-binario); en `garch_sigma` su corte mediano (0.13) está pegado al `target_vol=0.10`
de STRATA → el boosting redescubre la banda. "Umbral aprendido" en M10 no es un número, es
una nube de cientos de cortes; M8 hace el mismo trabajo con uno fijo.

---

## Q05 — El umbral de RAM, explicado bien (el tutor va a preguntar). ¿Cambia con M10?

**Qué es el RAM score.** El HMM reparte cada día 100% de probabilidad entre Calma/Estrés/
Crisis. La política (leverage effect): Calma→largo coherente, Crisis→corto coherente,
Estrés→ambos. **El score = la probabilidad que el mercado pone en un régimen donde el
agente va al revés**: agente short → score=P(Calma); agente long → score=P(Crisis). 0 = no
choca; 1 = choca de lleno. (Ej. 3-oct-2025: agente short, P(Calma)=1.0 → score=1.0.)

**Por qué vive en 0 o 1 (cuasi-binario).** El posterior **filtrado** del HMM es
cuasi-determinista: casi todos los días concentra la masa en un régimen. En el OOS solo
**11 de 401 días** caen en la zona intermedia [0.2,0.7).

**¿Cambia con M10?** No — **en M10 el umbral de RAM no existe.** En M8 (regla) comparas
`ram_score ≥ umbral`. En M10 el `ram_score` es **una feature más entre 22** y XGBoost
aprende sus propios cortes vía CPCV; **no usa** los umbrales de M8.

**Frase para el tutor ("ese umbral te lo has inventado").**
> *"El 0.40 era un default y lo digo en §4. Pero el resultado no depende de él: el RAM
> score es una masa de probabilidad del HMM filtrado, cuasi-determinista; mover el corte de
> 0.30 a 0.50 cambia 6 días de 401. Y en M10 dejo que el XGBoost elija el corte óptimo
> sobre `ram_score` y aun así no bate a mi regla: si el corte exacto importara, M10 ganaría.
> Lo que importa es que el régimen del HMM es informativo, y SHAP lo confirma."*

---

## Q06 — Metodología rigurosa para fijar los umbrales de M8 desde el histórico

**Contexto.** El percentil (que usan PSA/GSO) es **degenerado** para RAM porque su
distribución es bimodal (P50=0.001, P75=0.996). El percentil naíf empuja el corte a 0.996 y
**empeora** M8 (Sharpe −0.49).

**El "aha": hay DOS variables y se estaban mezclando.**
- **Eje A — confianza en la incoherencia** = el score de RAM (P(régimen incompatible)).
  **Monótono** por construcción → debe gobernar *cuánto* se atenúa (el *gain*).
- **Eje B — fiabilidad direccional del régimen** = `P(régimen acierta el signo de r_{t+1} |
  confianza)`, medida sobre 24 años. **No monótona**: ~0.48 (<0.2, ruido), 0.58–0.62
  (0.2–0.9, informativo), 0.52 (>0.9, deep-regime) → debe gobernar *dónde* empieza a actuar
  (el *gate*).

**Metodología recomendada (sólida, mínimos grados de libertad):** un único umbral
calibrado por activo, el **gate `τ`**:
1. Estimar la curva de fiabilidad B con **regresión isotónica** (no bins arbitrarios); `τ`
   = donde cruza 0.5. IC por **bootstrap estacionario** (Politis-Romano, en `core/stats.py`).
   En SPY `τ ≈ 0.20–0.25`, con ~242 días de calibración en la zona intermedia.
2. **`reduce` continuo y gated**: atenuación `factor = 1 − score` solo si `score ≥ τ`. El
   score ya es una probabilidad → no hacen falta umbrales low/medium/high.
3. **override (M8)** usa el mismo `τ` como gate.
4. **Por activo**: re-estimar la misma receta (nunca elegir el τ que maximiza el Sharpe OOS).
5. **Fallback honesto "no calibrable"**: si la fiabilidad nunca cruza 0.5 de forma
   significativa (leverage débil), RAM se declara inaplicable y se reporta como limitación
   (como la regla prior-flip).

**Caveats a reportar.** La fiabilidad es modesta (~0.58–0.62): hay que dar también su IC; si
incluyera 0.5, `τ` no sería distinguible de ruido. La curva B se construye sobre
`conf=[P(Calma),P(Crisis)]` pero el gate dispara sobre `score=incoherencia`; coinciden
cuando el agente va contra el régimen dominante (el caso de interés) — hay que escribirlo.
Pre-registrar antes de mirar OOS; contar este grado de libertad en el `n_trials` del DSR.

(Dictamen conjunto de `experto-inferencia` y `experto-series-temporales`.)

---

## Q07 — ¿Para qué medium y high si hacen lo mismo? ¿Por qué no solo low/high? ¿Los 4 niveles?

**Respuesta.** Los 4 niveles (none/low/medium/high) son una **escala general de STRATA para
sus 3 modos**, no específicos de M8:
- En modo **`reduce`** cada nivel atenúa distinto: none 0%, low 25%, medium 60%, high 100%.
  Ahí los cuatro hacen cosas distintas.
- En modo **`override`** (M8) la regla es **binaria**: interviene si severidad ≥ medium, y
  medium y high hacen exactamente lo mismo. low y none no intervienen.

**Conclusión:** en M8 solo hay **una frontera operativa** (no-intervenir vs intervenir) =
el umbral medium. Los cuatro niveles son herencia del marco general; en override colapsan a
dos resultados → un solo umbral.

---

## Q08 — ¿Por qué se descartó `reduce`? ¿Qué hacía?

**Qué hacía (M7).** Encoge el tamaño de la apuesta del agente según la severidad, **sin
cambiar la dirección** ("short 0.20" → "short 0.08").

**Resultado (proyecto anterior, a re-verificar).** SPY OOS causal: M5 −1.83 → **M7 −0.95**
→ M8 +0.66. M7 pierde la mitad que el agente, pero **sigue perdiendo**.

**Por qué override y no reduce.** El fallo del agente es **direccional** (corto en mercado
alcista, acierta 41%). Encoger una apuesta en el sentido equivocado pierde menos pero sigue
perdiendo; **voltear** la dirección al régimen inyecta señal y la lleva a positivo.
> *"Escalar no arregla un error de dirección; reemplazar la dirección sí."*

**Por qué se queda fuera / dentro.** Quedó fuera por la decisión de alcance ("M5/M8/M10 +
baselines", sin la escalera M1–M9). **Recomendación revisada: añadir M7 como control**,
porque completa el mecanismo en tres pasos (M5→M7→M8) y demuestra que el rescate viene de
**cambiar la dirección**, no de arriesgar menos. Conecta con SMCI (panel), donde al revés
reduce sería preferible (agente con señal). (Citas: `DECISIONES_ESENCIALES.md` #9;
`_archivo_proyecto_anterior/BITACORA.md` 2026-05-20.)

---

## Q09 — ¿Cómo se calcula `reduce` ahora (método nuevo)?

**Respuesta.** Solo necesita el mismo `τ`, ningún número extra. Por día:
- **Si score < τ** (régimen no fiable) → no toca: `posición = size del agente`.
- **Si score ≥ τ** → encoge proporcional al score, sin cambiar signo:
  `posición = size · (1 − score)`.

La atenuación **no se calibra**: sale del propio score (una probabilidad). score=1 → factor
0 → cash; score=0.5 → mitad. Nunca voltea el signo.

| día: agente short −0.20, P(Calma) | M5 | M7 (reduce) | M8 (override) |
|---|---|---|---|
| score 1.00 | −0.20 | 0.00 (cash) | +1.0 (voltea) |
| score 0.50 | −0.20 | −0.10 (mitad) | +1.0 (voltea) |
| score 0.10 (<τ) | −0.20 | −0.20 (no toca) | −0.20 (no toca) |

Frente al reduce viejo (buckets 25/60/100% con 3 umbrales arbitrarios), el nuevo los
elimina: atenuación continua y un solo número calibrado, `τ`, compartido con override.

---

## Q10 — ¿Qué es el "score" y qué es la "función de fiabilidad direccional"?

(Versión conceptual, ver también Q05 y Q06.)

- **El score** (de HOY): cuánto va el agente contra el régimen. Score = P(régimen
  incompatible con su acción). 0 = no choca; 1 = choca de lleno.
- **La fiabilidad direccional** (de la HISTORIA): mirando 24 años, "de los días en que el
  régimen estaba seguro al nivel X, ¿cuántas veces acertó la dirección que predecía?".
  Confianza <0.2 → 48% (ruido); 0.2–0.9 → 58–62% (sabe); >0.9 → 52% (flojea).

**Conexión.** El score dice cuánto choca el agente *hoy*; la fiabilidad dice cuándo el
régimen es de fiar *según la historia*; el umbral `τ` (~0.2) es el punto donde la historia
dice "a partir de aquí, fíate". No se inventa: sale de dónde el régimen empieza a acertar
más que una moneda.

---

## Q11 — Si el score es cuasi-binario, ¿sirve tener 3 estados o binario daría igual?

**Respuesta (agentes de rigor): no daría igual; binario sería PEOR.** El razonamiento
"score cuasi-binario ⇒ régimen binario" mezcla dos cosas: que el **score** sea cuasi-binario
(un número ~0/1) y que el **régimen** sea binario (2 estados). Son independientes; la
cuasi-determinación del posterior es señal de estados **bien separados**, no de que sobre uno.

- **Calma y Crisis ya son 2 estados de signo opuesto** (el score lee P(Calma) o P(Crisis)
  según la dirección del agente).
- **Estrés es el estado de abstención** ("no sé, no actúo"): alta vol **sin dirección
  fiable** (su media de retorno tiene IC que incluye 0), y es **~36% de los días** (2222 de
  6204). RAM no interviene ahí por construcción.

Si colapsas a 2 estados, esos ~36% de días de Estrés caen en "alta vol" = Crisis y RAM
**penalizaría los largos en días ambiguos** → **sobre-intervención**. Además, el score
cuasi-binario **presupone** los 3 estados: sin Estrés que absorba lo ambiguo, esa masa
inflaría falsamente el score.

**Pero ahora está asumido, no demostrado** (`config.py: HMM_N_STATES = 3` hardcoded). La
forma rigurosa de zanjarlo es una **ablación K=2 vs K=3 (vs K=4)**:
1. Selección de modelo en calibración: **BIC** (`−2·logL + k·log T`); se espera K=3 cerca
   del mínimo, K=2 peor, K=4 con estado degenerado.
2. Aguas abajo (OOS): M8(K=2) vs M8(K=3) con **McNemar** y **Diebold-Mariano** pareados,
   nº de intervenciones (K=2 ~36% más), Sharpe causal + DSR, días de Estrés reclasificados.
3. Pre-registro con criterio de fracaso honesto: si K=2 iguala/bate a K=3, el tercer estado
   es ornamental y se documenta.

**Frase para el tutor.** *"3 estados no es estético: Estrés es el estado de abstención que
evita intervenir en el 36% de días ambiguos; lo justifico con BIC y con la ablación K=2 vs
K=3, que sobre-interviene y no mejora. Y es falsable: si K=2 ganara, lo reportaría."*

(Dictamen de `rigor-matematico` y `experto-series-temporales`.)

---

<!-- Bloque añadido en la sesión 2026-06-08 (entrevista de comprensión I/O + descriptivos).
     Cifras de la ejecución actual de `notebooks/strata_canonical.ipynb` (§6 y §11). -->

## Q12 — ¿STRATA predice el retorno de mañana? ¿Qué devuelve exactamente?

**No predice nada.** STRATA es una **función determinista**
`f: (decisión_agente_t, estado_mercado_t) → posición_t`, sin variable objetivo. Devuelve un
`SupervisedDecision` cuyo campo operativo es **`final_size = w_t ∈ [−1,+1]`** (el peso para
hoy); el resto (`final_action`, `was_intervened`, los 3 `DetectorResult`) es **traza**.

El retorno de mañana **solo** aparece **fuera** de STRATA, en la contabilidad del backtest:
`PnL_t = w_t · r_{t+1}` con `signal_lag=1`. STRATA **nunca ve** `r_{t+1}` → blindaje anti
look-ahead.

**Frase para el tutor.** *"STRATA no es un modelo direccional, es un supervisor: decide una
posición con información de hoy; el retorno futuro solo entra en el P&L con un día de desfase."*

**Fuente.** `strata/strata.py:46`, `strata/types.py:59`, `core/backtest.py:62`.

---

## Q13 — Si M8 devuelve una posición y M10 una probabilidad, ¿no comparamos cosas distintas?

**No: M5, M8 y M10 devuelven los tres el MISMO objeto, una posición `w_t ∈ [−1,+1]`.** Lo
único que cambia es **cómo** la calculan:
- **M5**: copia al agente.
- **M8**: corrige al agente con la regla estadística (override-C).
- **M10**: calcula `p1 = P(r_{t+1}>0)` y la convierte en posición con `w = sign(p1 − 0.5)`.

La probabilidad `p1` de M10 es un **paso intermedio** ("el borrador"), no su salida: se
descarta tras producir el signo, igual que M8 usa el régimen HMM por dentro. Se compara
**posición contra posición** — McNemar sobre el signo (§10), Diebold-Mariano sobre el P&L
diario (§11). Nunca "probabilidad de M10 contra posición de M8".

**Matiz.** A M10, por ser clasificador, SÍ se le miden métricas de clasificación (log-loss,
etc.) que a M8 no tienen sentido; eso es **diagnóstico de M10**, no la comparación entre
estrategias.

**Frase para el tutor.** *"Las tres entregan la misma hoja de respuestas, una posición; la
probabilidad de M10 es su borrador. Comparo hojas entregadas, no borradores."*

**Fuente.** notebook §11 (`w = sign(p1 − 0.5)`), §10 (McNemar/DM).

---

## Q14 — ¿Qué me pidió exactamente el tutor? (los deberes)

Transcripción 2026-06-07 (`docs/tutor_transcripts/`). **Cuatro entregables concretos:**
1. **Matriz de confusión de la IA** (agente solo, M5).
2. **Descriptivo de las variables**: histograma de cada variable continua **coloreado por
   +1/−1**, con su corte de árbol.
3. **Boosting con TODAS las variables** que prediga el +1/−1 (= M10).
4. **Matriz de confusión de tu modelo** (M8).

**Condiciones de rigor:** target **retardado** `r_{t+1}` (no contemporáneo); umbrales por
**árbol/histograma**, no a mano; **contraste pareado** entre métodos (McNemar); validación en
**varios periodos/inicios**; centrar la defensa en el **acierto del signo** (matriz de
confusión), **no en el Sharpe** (*"el Sharpe no lo he visto en mi vida"*); no vender humo.

**Estado (notebook):** 1, 3, 4 hechos (§10/§11); 2 hecho (§6, panel de todas las variables +
descriptivo condicional de RAM); **validación multi-periodo pendiente**.

**Cita del tutor.** *"Lo importante era acertar el 1 o el menos 1."*

---

## Q15 — ¿Qué significan los descriptivos y cómo se leen?

Cada recuadro: eje X = valor de la variable, eje Y = nº de días; barra **apilada** en verde
(SPY sube mañana, `r_{t+1}>0`) / rojo (baja). La pregunta: *"cuando la variable vale X,
¿mañana tiende a subir o bajar?"* → mirar la **proporción verde/rojo** y cómo cambia de
izquierda a derecha. La **línea azul** = corte de un árbol de profundidad 1 (lo que pidió el
tutor). `acc` = acierto direccional usando **solo** esa variable; listón **trivial = 0.569**
(proporción de días al alza en el OOS).

**Resultado (§6):** **ninguna variable bate al trivial con holgura** (la mejor, `size`
agente, llega a 0.599). La única separación nítida y con sentido: **el agente va a
contramano** — cuando se pone corto (`size ≤ −0.012`) SPY sube el **61%**; cuando largo, sube
solo el **44%**. El resto ≈ moneda, o el corte cae en muestra mínima (`PSA` n=7, `RAM` n=9,
`GSO` n=1 en el lado alto → no concluyentes).

**Frase para el tutor.** *"El descriptivo demuestra, variable a variable, que la dirección a
un día es casi impredecible; la única señal limpia es que el propio agente se equivoca de
forma sistemática."*

**Fuente.** notebook §6 (panel de todas las variables).

---

## Q16 — En el descriptivo RAM sale plano. ¿RAM no sirve?

**Sí sirve — el descriptivo estándar mide la pregunta equivocada para RAM.** RAM no es un
predictor del mercado, es un **gate condicionado al agente**: marca los días en que el régimen
**contradice** la apuesta del agente. `GSO` (magnitud) y `PSA` (estabilidad temporal) tampoco
predicen el signo. Por eso los tres salen planos contra `r_{t+1}`; esa planura **es el
resultado** (la dirección no es predecible univariante), no un defecto.

El **descriptivo correcto para RAM** condiciona al agente (§6, "descriptivo correcto para
RAM"):

| RAM | n días | acierto seguir AGENTE | acierto seguir RÉGIMEN |
|---|---|---|---|
| `< τ` (no dispara) | 258 | 40.3% | 47.3% |
| **`≥ τ` (dispara)** | 121 | **41.3%** | **58.7%** |

Cuando RAM dispara, **seguir al régimen acierta 58.7% vs 41.3% del agente (+17 pp)**: el
rescate, dibujado. Es la versión visual del McNemar M8 vs M5.

**Talón de Aquiles.** Usa la misma ventana OOS → es **descriptivo**, no prueba OOS
independiente; la prueba formal sigue siendo el McNemar (§10) + robustez (panel multi-activo
y validación multi-periodo).

**Fuente.** notebook §6 (descriptivo condicional de RAM).

---

## Q17 — ¿Cómo se traduce ese gate de RAM a M10?

En **M8** el gate es **explícito** (`si RAM ≥ τ → voltea a regime_sign`). En **M10 no hay
gate**: el XGBoost recibe `ram_score` + las probabilidades de régimen como features (entre las
22) y **aprende solo** a combinarlas con splits de árbol. El "descriptivo equivalente" de M10
es el **SHAP**.

**SHAP pooled out-of-fold (§11), top por |SHAP| medio:**
`ram_score` 0.43 · `garch_sigma` 0.43 · `psa_score` 0.38 · `crisis_prob` 0.36 ·
`stress_prob` 0.31 · `calm_prob` 0.31 → **las 6 primeras son STRATA + régimen.** La primera
personalidad (`cathie_wood_conf`) entra en el puesto **7** (0.21, la mitad que `ram_score`).
**Ablación**: quitando las features de STRATA/régimen, Sharpe de M10 **+0.64 → +0.21**.

**Lectura.** M10 **redescubre** el gate de M8 desde los datos: le da el máximo peso a
`ram_score` y al régimen, y releva a las personalidades. Por eso M8 y M10 **empatan**
(Diebold-Mariano p=0.61): explotan el **mismo fenómeno** (el régimen contradice al agente),
uno con regla a mano y otro con splits aprendidos.

**Frase para el tutor.** *"En M8 el gate lo escribo yo; en M10 nadie lo escribe, pero el SHAP
demuestra que el XGBoost le da el máximo peso justo a `ram_score` y al régimen — redescubre el
gate. El agente queda relegado."*

**Fuente.** notebook §11 (SHAP pooled OOF, ablación, DM/TOST).

---

## Q18 — La frase "Estrés = abstención hace de STRATA un supervisor" ¿está demostrada o asumida?

**Categoría:** metodología

**Pregunta del tribunal (literal probable):**
> "Tu §12 dice que la abstención del estado Estrés es lo que convierte STRATA en supervisor y no en cabalga-drifts. ¿Qué experimento prueba eso? Enséñame el test que falsa esa frase con K=3."

**Respuesta defendible (60–120 palabras):**
Hoy NO está demostrada: está asumida. El único test falsable del mecanismo (`experiments/m10_vs_m8_drift.py`) instancia `RegimeHMM(n_states=2)` en la línea 71, no K=3. Demuestra que el M8 **binario** cabalga el drift (corr drift↔(M10−M8) ρ=−0.70), que es justo lo que ya sé. NO toca el modelo canónico K=3, así que no puede probar que K=3 NO cabalgue. La afirmación se apoya solo en ocupación (~49% largo K=3 vs ~75% K=2). Hasta re-correr el drift-test con K=3, la frase es una conjetura interpretativa, no un resultado.

**Evidencia anclada:**
- Código del autogol: `experiments/m10_vs_m8_drift.py:71` (`n_states=2`)
- Afirmación expuesta: `notebooks/_build.py:1238-1240` (§12 del canónico)
- Salida: `outputs/experiments/m10_vs_m8_drift.json` (ρ=−0.70, p=0.188)

**Talón de Aquiles:**
El tribunal puede pedir el resultado K=3 en directo. Si al re-correrlo K=3 también monta el drift en alcistas, la tesis del "supervisor disciplinado" se cae. Conceder honestamente: "el test que tenía mide el binario; el de K=3 está pendiente y es la pieza que cierra §12".

---

## Q19 — "La verosimilitud fuera de muestra prefiere K=3 sin ambigüedad": ¿probaste K≥4?

**Categoría:** metodología

**Pregunta del tribunal (literal probable):**
> "Dices que la held-out likelihood prefiere K=3 sin ambigüedad. ¿Comparaste con K=4, K=5? Una gaussiana mal especificada casi siempre pide más estados. ¿Cómo sabes que el máximo está en 3 y no que 3 simplemente bate a 2?"

**Respuesta defendible (60–120 palabras):**
La frase "sin ambigüedad" está sobre-afirmada. `heldout_ll` (`_build.py:1153`) y `k_selection.json` solo evalúan K∈{2,3}. La held-out LL es monótona creciente esperable bajo mala especificación gaussiana, así que con K≥4 muy probablemente seguiría subiendo. Lo honesto: K=3 se elige porque (a) bate a K=2 en held-out LL, (b) los tres estados son económicamente distintos en vol y signo, (c) es interpretable (Calma/Estrés/Crisis) y (d) aporta abstención. NO porque la verosimilitud tenga un máximo en 3. Debo reescribir "sin ambigüedad" → "K=3 bate a K=2; K≥4 fragmenta sin ganancia interpretable".

**Evidencia anclada:**
- `notebooks/_build.py:1227-1229` (texto "sin ambigüedad")
- `notebooks/_build.py:1153-1161` (`heldout_ll`, folds solo K=2/K=3)
- `outputs/experiments/k_selection.json` (`"ks": [2, 3]`; held-out K3 −1.301 vs K2 −1.693)

**Talón de Aquiles:**
Si corres K=4 y la LL sube, el tribunal dirá "entonces tu criterio elige ∞". Rebate anclando la elección en parsimonia + interpretabilidad + significado económico, no en argmax LL. Reportar la curva LL(K) para K∈{2,3,4,5} y mostrar que el salto grande es 2→3 y luego se aplana.

---

## Q20 — ¿El "rescate" de M8 es detección de régimen o solo deshacer el sesgo-corto del agente en un mercado alcista?

**Categoría:** resultados

**Pregunta del tribunal (literal probable):**
> "El agente va corto, el mercado sube 54%, tú lo volteas a largo y ganas. ¿Eso es tu HMM detectando régimen, o cualquier regla que neutralice el sesgo bajista del agente habría ganado igual en esta ventana?"

**Respuesta defendible (60–120 palabras):**
Es el confound más serio y hoy no está cerrado. M8-K3 va largo ~49% de los días; en una ventana alcista (SPY +54%) parte del rescate puede ser "demean del sesgo-corto del agente", no destreza de régimen. Falta el control: un modelo `M_drift` de exposición larga neta fija SIN régimen, o el agente demeaned, contra el cual medir cuánto P&L añade el HMM por encima de "estar largo". El McNemar pareado (0.069) mide acierto direccional, que es más robusto que el P&L al confound, pero no lo elimina. Hasta tener `M_drift` como baseline, la atribución "régimen vs drift" no es separable.

**Evidencia anclada:**
- Confound descrito en `BITACORA.md` (entrada FINAL 2026-06-08, PRUEBA 2 y caveat K=2)
- Ocupación K=3 ~49% largo: `notebooks/_build.py:1219`
- McNemar M8 vs M5 = 0.069: `BITACORA.md` (cifras canónicas finales)

**Talón de Aquiles:**
El tribunal puede construir mentalmente `M_drift` y notar que probablemente gana en esta ventana. Conceder: "el McNemar separa parte del confound vía acierto pareado, pero el baseline largo-fijo es el control que me falta; es lo primero de mi lista de arreglos".

---

## Q21 — Deflated Sharpe con n_trials=32: ¿cuántas configuraciones probaste de verdad?

**Categoría:** metodología

**Pregunta del tribunal (literal probable):**
> "Pones n_trials=32 en el Deflated Sharpe. Pero esta sesión barriste τ por 4-5 métodos, K por 3 criterios sobre {2,3,4}, K-por-activo, paneles... ¿No son más de 100 configuraciones? ¿Tu DSR no está inflado?"

**Respuesta defendible (60–120 palabras):**
n_trials=32 (`_build.py:846`) cuenta el barrido de τ y la elección de K, pero infra-cuenta la exploración real de la sesión (τ por isotónica/logística/histograma/árbol, K∈{2,3,4} por LL/accuracy/panel, K-por-activo). Con el conteo honesto (>100) el DSR cae de 0.106 a ~0.04-0.07. Esto NO hunde la tesis porque ya declaro que el Sharpe NO es robusto a multiplicidad: la evidencia del rescate es el **McNemar pareado** (0.069) y la **permutación por bloques** (0.044), no el Sharpe. El DSR honesto refuerza esa jerarquía: el Sharpe es ilustrativo, no prueba.

**Evidencia anclada:**
- `notebooks/_build.py:843-851` (`N_TRIALS = 32`, comentario "n_trials honesto")
- DSR 0.106 reportado: `BITACORA.md` (cifras canónicas finales)
- Jerarquía evidencia: `CLAUDE.md` §4 (Sharpe ≠ prueba; McNemar/permutación primarios)

**Talón de Aquiles:**
Si el tribunal exige el n_trials exacto y no sabes contarlo, pierdes credibilidad. Documenta la cuenta (lista de configuraciones probadas) y reporta DSR con n_trials conservador (≥100). Conceder que el Sharpe positivo de M8 "podría ser ruido" y que por eso la tesis se ancla en el acierto pareado.

---

## Q22 — El test del drift tiene n=5 y p=0.19. ¿Lo presentas como "confirmado"?

**Categoría:** resultados

**Pregunta del tribunal (literal probable):**
> "Tu test de que la ventaja de K=2 es condicional al alza usa 5 activos y da p=0.19. Eso no es significativo. ¿Por qué lo presentas como confirmación?"

**Respuesta defendible (60–120 palabras):**
Tiene razón: con n=5, ρ=−0.70 da p=0.188 (`m10_vs_m8_drift.json`), NO significativo a α=0.10. Con n=5 ni una ordenación perfecta (ρ=−1) baja de p≈0.05. Presentarlo como "confirmado" (`_build.py:1135`, "demuestra") es sobre-afirmar. Lo defendible es: "patrón **sugestivo y consistente** con la hipótesis (signo correcto, magnitud grande), pero **sin potencia estadística** por n=5". Para confirmarlo de verdad necesito ampliar a los 10 activos del panel y, sobre todo, validación cross-TIEMPO (walk-forward), no solo cross-activo. Hasta entonces es evidencia direccional, no prueba.

**Evidencia anclada:**
- `outputs/experiments/m10_vs_m8_drift.json` (ρ=−0.70, p=0.188, n=5)
- Verbo "demuestra"/"confirmado": `notebooks/_build.py:1135` y BITACORA FINAL punto 3
- Lista de tickers n=5: `experiments/m10_vs_m8_drift.py:34`

**Talón de Aquiles:**
El tribunal puede pedir el test con n=10 en directo; si pierde el signo, la narrativa se debilita. Conceder de antemano que n=5 no da potencia y degradar el lenguaje de "confirmado" a "consistente, pendiente de potencia".

---

## Q23 — Una sola ventana OOS alcista: el tutor pidió "lánzalo en diferentes años". ¿Dónde está?

**Categoría:** limitaciones

**Pregunta del tribunal (literal probable):**
> "Todo esto es una ventana, 2024-10 a 2026-05, y el mercado subió. Te pedí que lo lanzaras en distintos inicios, distintos años. ¿Cómo sé que no tuviste suerte con el periodo? Tu panel de 10 activos comparte el MISMO OOS alcista."

**Respuesta defendible (60–120 palabras):**
Es la limitación central y el tutor ya la planteó literalmente ("yo necesito que tu método lo lances en diferentes años, en diferentes momentos"). El panel de 10 activos es cross-ASSET pero NO cross-TIEMPO: comparten el mismo OOS 2024-10→2026-05, y en un crash sus correlaciones tienden a 1, así que no es diversificación temporal real. Falta walk-forward histórico re-calibrando antes de 2008, 2020 y 2022 (regímenes bajistas/volátiles) y, como mínimo, estratificar el OOS actual por régimen para ver si el rescate sobrevive en los subtramos no alcistas. Hasta tenerlo, el resultado es "válido en esta ventana", no general.

**Evidencia anclada:**
- Petición literal del tutor: `docs/tutor_transcripts/Calle de Hilarión Eslava, 46 2.md` ("lánzalo en diferentes años / en diferentes momentos")
- OOS único: `CLAUDE.md` §3 (2024-10-01 → cierre TFG)
- Panel cross-asset mismo OOS: `BITACORA.md` (entrada panel K-ablation, "OOS 2024-10-01 → cierre")

**Talón de Aquiles:**
Walk-forward histórico es el experimento que más puede tumbar la tesis (si el rescate desaparece en 2008/2020/2022). Pero NO hacerlo es peor: el tutor ya lo pidió y su ausencia es lo primero que cazará. Hacerlo y reportarlo honestamente —aunque el rescate se debilite fuera del alza— es más defendible que esconderlo.

---

## Mejoras de rigor pendientes de implementar (salidas de esta conversación)

Para añadir y re-ejecutar la Parte IV en bloque, pre-registradas en `BITACORA.md`:
1. Umbral `τ` calibrado por isotónica + IC (gate), compartido por M8 y M7.
2. **M7 (reduce continuo, gated en τ)** como fila de control → progresión M5→M7→M8.
3. **Ablación K=2 vs K=3** para justificar los 3 estados del HMM.
