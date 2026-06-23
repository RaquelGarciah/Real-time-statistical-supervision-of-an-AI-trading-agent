# Dossier del marco práctico (Cap. 4) — fuente única para la memoria

**Qué es esto.** El puente entre el notebook canónico `notebooks/STRATA_marco_practico.ipynb` y el capítulo 4 de
la memoria (`tesis/chapters/04_marco_practico.tex`). Reúne, en prosa y por secciones, **toda** la mecánica,
cifras, tests, decisiones, justificaciones y límites del marco práctico — incluido lo añadido en la última fase
(PARTE B/DSR, complementariedad por régimen, anatomía de la intervención, gate RAM, clustering de los 10, gráficas
por grupo, definición de *pooled*). Sirve para que la redacción del capítulo tenga **todo el contexto** sin
reconstruirlo del código.

**Cómo usarlo.** (1) Es la fuente única del Cap. 4: cada afirmación del `.tex` debe poder rastrearse a una entrada
de aquí. (2) La redacción (agentes `redactor-tesis` + gates de estilo/citas) consume este dossier, no el código.
(3) Se mantiene sincronizado con el notebook, `BITACORA.md`, `DECISIONES_ESENCIALES.md` y `RESULTADOS_OBJETIVO.md`
(si una cifra cambia en el notebook, se actualiza aquí). (4) El §9 del notebook es el *resumen*; este dossier es
el *desarrollo*.

**Regla de honestidad transversal (no negociable).** STRATA **no genera alfa**. Su valor probado es: (a) **rescate
del agente** (accuracy, McNemar, significativo); (b) **rescate de riesgo** (ΔSharpe pooled, significativo);
(c) **universalidad** (el ML redescubre STRATA, cuota SHAP); (d) **patrón** naturaleza→estrategia. La superioridad
en accuracy sobre las triviales (ZeroR/B&H) es **nominal**, no significativa: se etiqueta siempre como tal.

---

## EL APORTE REAL — las conclusiones reveladoras y su valor

> **Reencuadre del objetivo (importante para la memoria).** El objetivo inicial era **vencer en accuracy**. El
> resultado honesto es que, en este OOS corto, **ninguna estrategia bate a las triviales en accuracy de forma
> significativa** (techo ZeroR). Pero forzar ese marco sería medir el trabajo por lo que NO es. Lo que esta
> investigación SÍ produce —y es **más valioso y más científico** que un punto de accuracy— es un **mapa
> falsable de cuándo y cómo la supervisión estadística de un agente LLM aporta valor**: qué canal funciona según
> la **naturaleza** del activo, en qué **régimen**, con qué **modelo**, y con qué **mecanismo**. Encontrar
> patrones contrastables NO es el premio de consolación: **es la contribución**. Una técnica de supervisión cuyo
> dominio de aplicabilidad está caracterizado y falsado es más fiable —y más sólida para **seguir la línea de
> investigación**— que un número de accuracy sin teoría detrás. El capítulo debe **liderar con esto**, no
> esconderlo tras la accuracy.

Las conclusiones de esta fase, con la configuración actual (panel-10, M8/M10/AutoML canónicos, OOS desplegable),
y **el valor que tienen**:

**C1. El rescate de riesgo del agente es real y sobrevive el test más estricto.**
*Resultado:* pooled, M8 vs M5 ΔSharpe **+0.66 [0.225, 1.157]** (excluye 0); en el confirmatorio con **cota
Bonferroni**, **M10 y AutoML pasan** (SPY +0.02 / +1.91; pooled +0.26 / +0.26) y el **DSR de AutoML-SPY = 0.924**.
*Valor:* la afirmación "supervisar reduce el riesgo del agente" no es retórica — aguanta bootstrap pareado,
corrección por multiplicidad y deflación. Es el resultado **duro** que sostiene la tesis.

**C2. La regla pura (M8) NO sobrevive el confirmatorio en Sharpe — y eso es un resultado, no un fracaso.**
*Resultado:* M8 sola no pasa la cota Bonferroni (SPY −0.66; pooled −0.05); el meta-learner sí.
*Valor:* es una **falsación pre-registrada cumplida**, escrita por honestidad antes de mirar. Delimita
exactamente qué parte de STRATA es robusta (el aprendiz) y cuál es un rescate de riesgo estable pero no
"confirmatorio" (la regla). Un tribunal premia esto: sabes dónde está el límite de tu propia técnica.

**C3. El valor de la supervisión NO es de un solo régimen de mercado.**
*Resultado:* el rescate (accuracy y Sharpe) es significativo en **alcista Y bajista** en el pooled (McNemar
p_Holm<0.10 en los 6 contrastes; block-perm<0.07). A nivel SPY-solo se concentraba en alcista y la regla se
invertía en bajista (n=50) — falsación que la **agregación resuelve**.
*Valor:* responde de frente la objeción nº1 ("solo funciona en el mercado alcista que muestreaste"). La
supervisión aporta también cuando el mercado cae, que es cuando más importa.

**C4. [PATRÓN] Los dos aprendices son complementarios por régimen — y es significativo.**
*Resultado:* **M10 rescata más en alcista** (ΔSharpe +1.37 vs +0.72), **AutoML más en bajista** (+1.52 vs +0.81),
**M8 simétrica**. Test DiD pre-registrado: **significativo en el pooled** (DiD +1.37 [0.20, 2.60], p=0.008), **no
en SPY-solo** → es un **fenómeno cross-asset**. *Valor:* es de los hallazgos más reveladores. (a) Explica una
**estructura interna** de la capa de accuracy que nadie había nombrado. (b) Tiene **implicación de despliegue
directa**: AutoML —que modela la interacción condicional— protege mejor en el régimen **peligroso** (bajista), así
que es el candidato a capa de accuracy desplegable. (c) Abre una **línea futura concreta y pre-registrable**: un
**ensemble enrutado por régimen** (M10 en tendencia alcista / AutoML en bajista) usando la señal de RAM. Esto es
exactamente "seguir esta línea de investigación con confianza".

**C5. [PATRÓN] La naturaleza del activo gobierna qué canal funciona — y se mide.**
*Resultado:* clustering de los 10 con **consenso unánime** (KMeans/Ward/GMM/Spectral, Rand=1.0, silhouette 0.55) →
tres grupos por naturaleza: **índices de leverage fuerte** (canal régimen/riesgo), **leverage invertido** (canal
aprendiz), **volátiles** (canal aprendiz). El eje principal del clustering **es el leverage** (PC1≈leverage,
r≈0.84). Y la **única ley que sobrevive un test**: el rescate del aprendiz en accuracy **escala con el leverage
effect** (Pearson r=−0.55, p=0.034; Spearman −0.54, p=0.038; robusta a leave-one-out, peor caso p≈0.095).
*Valor:* convierte "a veces gana uno y a veces otro" —la duda que te angustiaba— en una **regla mecánica
falsable**: *donde el régimen es coherente con la dirección (leverage fuerte), funciona la regla; donde el régimen
miente sobre el signo (leverage invertido), solo el aprendiz rescata*. Eso es **entender el fenómeno**, no solo
medirlo.

**C6. [MECANISMO] Las dos capas tienen trabajos distintos y visibles.**
*Resultado:* por grupo, **M8 levanta el Sharpe del agente** (ΔSharpe ≈ +1.3 en índices y volátiles; ≈0 en
leverage invertido, donde la regla no puede y gana M10); la **cuota SHAP de STRATA es >0.5 en 10/10** (universal).
Gate RAM: cuando el detector dispara, seguir el **régimen** bate a seguir al agente en **6/10**; y la
**intervención crece con la discrepancia agente↔régimen** (Pearson **r=0.93, p<0.001**). *Valor:* el sistema hace
**lo que dice que hace** — interviene donde el agente se aparta del régimen, y cada capa (riesgo/accuracy) cumple
su función. Mecanismo interpretable = se entiende **por qué** actúa (base para confiar en la técnica).

**C7. [MECANISMO] El que las features de STRATA ayuden depende del modelo — y el ganador las usa.**
*Resultado:* **AutoML** alcanza su máximo con las 22 features (0.574) y **degrada** al quitar PSA+GSO (0.550) → sí
extrae su señal; **M10-XGBoost (params fijos) se sobreajusta** con 22 (0.494 < agente-15 0.542). *Valor:* zanja el
*gap* de "PSA/GSO no disparan": como **reglas** casi no actúan, pero como **features continuas** informan a un
aprendiz capaz. Justifica conservarlos sin inflar su papel.

**C8. La intervención, hecha tangible y honesta.**
*Resultado:* de **121 intervenciones** en SPY, M8 acierta 58.7% vs 41.3% del agente (**71 aciertan, 50 fallan**,
P&L +0.312); dos días reales con el mismo mecanismo y desenlace opuesto. *Valor:* baja la mecánica a un caso
concreto y **no esconde los fallos** — la regla es favorable en el agregado, no infalible. Credibilidad.

**C9. [JERARQUÍA HONESTA] El aprendiz redescubre la regla y la supera con flexibilidad — contrastado con un test
de equivalencia.**
*Resultado:* el SHAP confirma que el aprendiz **usa las señales de STRATA** (cuota 0.66, parte de universalidad
que aguanta). Y un **test de equivalencia (TOST, Schuirmann 1987)** —no un test de diferencia no significativo—
zanja el "¿redescubre o bate?": **NO hay equivalencia; el aprendiz BATE a la regla M8 en accuracy** (pooled M10
Δacc=+0.021 IC90[+0.001,+0.039], AutoML +0.034 [+0.010,+0.056]; SPY AutoML fuerte, +0.132). En **Sharpe es no
concluyente** → regla y aprendiz **indistinguibles en riesgo**. El batir a la regla **se explica por el mecanismo
(C5)**: el aprendiz modela **interacciones no lineales que la regla determinista no puede** (leverage invertido,
donde M8 falla). De aquí sale el **orden honesto y nítido** que estructura todo el capítulo:

> **El aprendiz RESCATA al agente (significativo) > BATE modestamente a la regla en accuracy (TOST: superior, no
> equivalente, por flexibilidad no lineal) > EMPATA con la regla en riesgo (Sharpe indistinguible) > NO BATE a lo
> trivial (ZeroR, accuracy nominal).**

*Valor:* es la **escala de valor de STRATA escrita con rigor** — cada peldaño con su test (McNemar/bootstrap →
rescate; TOST → vs regla; ceiling ZeroR → vs trivial). Responde de una sola frase, falsable, "¿cuánto y frente a
qué aporta supervisar?". Y es **honesta hasta el final**: no infla (no bate a lo trivial) ni esconde (sí bate a la
regla, y se explica por qué). Esta jerarquía es lo que la memoria debe llevar como tesis del marco práctico.

**Síntesis del valor.** El trabajo aporta (i) un **resultado duro contrastado** (rescate de riesgo, C1–C3); (ii)
un **mapa de patrones falsables** sobre dónde y cómo supervisar (C4–C5), que es el corazón científico; (iii) un
**mecanismo interpretable** que hace la técnica entendible y fiable (C6–C8); y (iv) una **jerarquía de valor honesta y
contrastada peldaño a peldaño** (C9), que es la frase-tesis del capítulo. La accuracy frente a lo trivial es
nominal y se declara; el valor está en **entender el fenómeno de la supervisión estadística de agentes LLM** —
cuánto aporta, frente a qué, y por qué—, no en un punto de acierto.

---

## 0. Glosario (la jerga, en una línea cada una)

**Estrategias** (todas devuelven una posición diaria $w_t=\pm1$):
- **M5** — el **agente** LLM solo (AI Hedge Fund, 5 personalidades). Es el sujeto a supervisar.
- **M8** — la **regla determinista** de STRATA: override-C por **régimen** (si RAM detecta que el agente es
  incoherente con el régimen, voltea su posición hacia la dirección del régimen). Es la **capa de riesgo**.
- **M10** — el **meta-learner**: ensemble de 10 XGBoost (semillas 42–51), walk-forward (N0=150, paso=21,
  embargo=1), con las 22 features. Capa de **accuracy**.
- **AutoML** — búsqueda automática de modelo (H2O: GBM/XGBoost/StackedEnsemble, AUC, Purged K-Fold, `max_models=25`,
  `seed=42` → determinista). El **modelo ganador** en accuracy.
- **Régimen** — usar solo la dirección del régimen HMM (RAM crudo), sin agente.
- **ZeroR / B&H** — triviales: ZeroR predice siempre la clase mayoritaria; B&H = comprar y mantener (siempre largo).

**Detectores** (los tres ejes ortogonales de STRATA):
- **RAM** — coherencia con el **régimen discreto** (HMM gaussiano 3 estados). Es el que actúa.
- **PSA** — coherencia **temporal** del agente (cambio estructural de opinión; BOCPD).
- **GSO** — coherencia con la **volatilidad** continua (banda GARCH(1,1)-t).

**Features:** AGENT15 (5 personalidades × {signo, tamaño, confianza}) + STRATA7 (ram/psa/gso_score,
calm/stress/crisis_prob, garch_sigma) = **ALL22**.

**Ventanas (NUNCA se mezclan, siempre etiquetadas):**
- **OOS completo** (n≈401, desde 2024-10-01): para M5/M8/ZeroR/B&H, que no necesitan burn-in.
- **Desplegable** (n≈250, tras burn-in de 150): para M10/AutoML (necesitan histórico para entrenar). SPY: n=251.

**Conceptos estadísticos:**
- **`signal_lag=1`** — la posición del día $t$ multiplica al retorno del día $t{+}1$ (causal; el único válido).
- **pooled** — *ver §nota metodológica abajo*. Apilar los días de todos los activos en una sola muestra.
- **ΔSharpe / ΔmaxDD** — diferencia de Sharpe / de máximo drawdown entre dos estrategias.
- **McNemar** — test pareado de aciertos correlacionados (¿una estrategia acierta en días distintos que otra?).
- **block-permutation** — variante de McNemar robusta a la autocorrelación temporal.
- **bootstrap estacionario pareado** (Politis-Romano 1994) — IC de una métrica respetando la dependencia temporal.
- **DSR (Deflated Sharpe Ratio)** — $P(\text{Sharpe verdadero}>0)$ tras descontar la esperanza del máximo de
  $n_{\text{trials}}$ Sharpes bajo $H_0$ (haircut por haber explorado varias configuraciones; Bailey-LdP 2014).
- **Holm / Bonferroni** — correcciones por comparaciones múltiples (controlan el error de familia, FWER).
- **DiD (difference-in-differences)** — contraste de si una diferencia (M10 vs AutoML) **cambia** entre regímenes.
- **leverage effect** (Black 1976) — correlación negativa retorno↔volatilidad: en índices, alta vol ↔ caídas.

---

## NOTA METODOLÓGICA — ¿Se puede hacer *pooled*? (sí, con lectura correcta)

**Qué es.** En lugar de analizar cada activo por separado, se **apilan los retornos diarios de todos los activos
en una sola muestra** y se hace el test sobre ella. 10 activos × ~250 días ≈ **2 493** (pooled-10); 15 × ~250 ≈
**3 751** (pooled-15). El bootstrap remuestrea de esa serie combinada.

**¿Es legítimo? SÍ.** Es **estadística de panel** estándar (econometría de panel; backtests multi-instrumento del
mundo López de Prado). Apilar unidades para ganar potencia es ortodoxo. En este proyecto es además **necesario**:
con n≈250 por activo no hay potencia y el McNemar per-activo del rescate es >0.12 en todo el cuerpo; renunciar al
pooled sería renunciar a concluir, no ser más riguroso.

**Por qué se usa tanto.** La significancia del rescate **vive en el agregado, no por activo**. Frase canónica del
notebook: *"la significancia vive en el pooled, no per-activo"*.

**Cómo se hace bien (ya implementado).** No se usa un t-test naíf (que asumiría independencia), sino **block
bootstrap**, que respeta la dependencia temporal **dentro** de cada activo.

**Límite honesto (declarado en §4 del notebook).** Apilar trata cada *día-activo* como independiente, pero el
mismo día los índices se mueven juntos (correlación **cruzada**, sobre todo en crisis) → la **n efectiva es menor
que la nominal**; el block bootstrap corrige la autocorrelación dentro de un activo, no entre activos, así que la
precisión del IC está algo **sobreestimada**. Es un matiz de **precisión, no de validez**: el resultado canónico
(M8 vs M5 ΔSharpe +0.66, IC95 [0.225, 1.157]) excluye 0 **con holgura**, con margen para ese fleco. **Robustez
pendiente/opcional:** un bootstrap **por fechas** (remuestrear días de calendario llevándose todos los activos de
cada día juntos) preservaría la correlación cruzada y convertiría el caveat de "declarado" a "contestado con un
test". (Pre-registrable; no ejecutado aún.)

**Lectura correcta de la afirmación:** *"en agregado / a nivel de panel el rescate es significativo"* — **no**
*"en cualquier activo individual"*.

---

## §1 Datos, universo y protocolo

- **Universo:** 15 activos analizados; **cuerpo = 10** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG),
  **apéndice = 5** (MSTR, NVDA, BAC, TSLA, IWM).
- **Decisión (cohorte 10).** La selección de los 10 es **ex-ante por naturaleza** (clases de activo), no por
  resultado. *Honestidad:* se subraya que **no** se usa la significancia per-activo como criterio de selección
  (con n≈250 casi nunca hay p<0.10); el criterio es **ilustrativo del mecanismo**, y la significancia vive en el
  pooled (§4) y los estratos (§7). Los 5 del apéndice (§8) delimitan dónde STRATA **no** aporta.
- **Periodos.** Calibración 2000-01-01 → 2024-09-30 (24 años, HMM/GARCH/BOCPD entrenados una vez). OOS unificado
  2024-10-01 → cierre (inicio posterior al cutoff de DeepSeek V3 → sin contaminación look-ahead del LLM).
- **Rigor:** `signal_lag=1`; embargo=1 en el walk-forward desplegable (horizonte de etiqueta=1, rolling-origin,
  justificado en DECISIONES #15); barrera temporal con asserts anti-fuga.

## §2 Mecánica ex-ante (los tres detectores)

- **HMM K=3.** Elegido por **verosimilitud held-out** (K=3 LL/obs −1.30 > K=2 −1.69); no por el OOS.
- **GARCH(1,1)-t** (α+β<1, estacionario) para la banda de volatilidad; **BOCPD** (Adams-MacKay 2007) para PSA.
- **Umbrales ex-ante** (RAM τ=0.5; PSA/GSO P95/P99 fijados en calibración).
- **¿Qué detector actúa?** En override-C **solo RAM interviene**; PSA/GSO casi inertes en este OOS. Atribución del
  P&L de rescate: **100% al canal régimen (RAM)**; PSA/GSO = 0. **Se enseña, no se esconde.**
- **Activación en el panel (10):** RAM dispara entre **2% y 71%** según activo; PSA ≤ 2.5%; **GSO = 0%** en los 10.
- **¿Por qué se conservan PSA/GSO?** (a) En calibración (con 2008/2020) **sí** disparan (colas reales); este OOS,
  calmado, no llega a esa cola. (b) Sus **scores continuos** sí informan al aprendiz aunque no disparen como
  reglas (ablación abajo). (c) Es una **predicción pre-registrada cumplida** (CLAUDE.md §2: "RAM domina la
  atribución"); quitarlos a posteriori sería ajustar el marco a los datos.
- **[NUEVO] Anatomía de un día de intervención.** Dos días reales, mismo mecanismo (agente corto en Calma → RAM lo
  voltea a largo, RAM≈0.99), desenlace opuesto: **ACIERTO 2024-11-05** (+2.46%, rally post-electoral; M8 acierta,
  M5 falla) vs **FALLO 2024-10-30** (−1.98%; M8 falla, M5 acierta). **Balance:** de **121 intervenciones**, M8
  acierta 58.7% vs agente 41.3% → **71 aciertan, 50 fallan**, P&L de rescate +0.312. *La regla no es infalible; es
  favorable en el agregado.* Diagrama de flujo visual agente→STRATA→resultado.
- **[NUEVO] Ablación de detectores en el meta-learner (SPY, misma config).** Para M8 no hay ablación (M8 *es* el
  detector; sin él colapsa a M5). **AutoML (el ganador)** alcanza su máximo con las 22 (acc 0.574) y **degrada al
  quitar PSA+GSO** (0.550) → **sí extrae valor de los scores de los detectores**. **M10-XGBoost (params fijos) se
  sobreajusta** con 22 (0.494 < agente-15 0.542): que STRATA ayude **depende del modelo**. Conclusión que zanja el
  *gap*: PSA/GSO rara vez disparan como reglas, pero sus scores continuos llevan información que un aprendiz capaz
  aprovecha.
- **[NUEVO] Descriptivo SPY (corte de árbol depth-1).** Cada variable vs el signo de $r_{t+1}$: **ninguna separa
  bien sola** (la mejor, crisis_prob, acc univariante 0.594) → la dirección **no es univariante**, justifica el
  meta-learner que combina las 22.
- **Régimen × dirección.** El régimen baja con el retorno del **mismo día** (leverage), pero la fracción que sube
  **al día siguiente** ronda 0.5 en todos los regímenes → el régimen separa por **volatilidad**, no anticipa el
  signo (es contemporáneo, no predictivo).

## §3 Caso de estudio: SPY (el gancho)

- **Tabla 6 estrategias (ventana desplegable, n=251):** M5 0.366 (Sharpe −3.07) · M8 0.442 · M10 0.494 ·
  **AutoML 0.574** · ZeroR/B&H ≈ 0.566. AutoML gana a todas **en punto**.
- **Honestidad clave:** ese "AutoML gana a todo" es **NOMINAL** — McNemar AutoML vs ZeroR **p=0.90**.
- **Rescate del agente (SÍ significativo):** McNemar vs M5 → AutoML **0.0002**, M10 **0.0074**, M8 **0.051**.
- **[NUEVO] Override vs abstención + umbrales:** override (eq 0.94) > abstención (0.81) > reduce (0.76) > agente
  (0.70); barridos de τ_RAM y p1* planos → no hay grados de libertad ocultos.
- **[NUEVO] 6 matrices de confusión SPY** (predicho ±1 vs real): el agente falla sobre todo en largos (FN alto);
  M8/M10/AutoML reequilibran; ZeroR/B&H siempre largo (FN=TN=0).
- **M10 vs M5 por régimen + SHAP dependency:** el aprendiz corrige al agente en Calma y Estrés; su SHAP varía de
  forma estructurada con cada señal STRATA (la usa).
- **[NUEVO] Calibración (reliability) de M10:** comprueba que su probabilidad p1 es honesta (consistencia interna),
  no superioridad. Lectura descriptiva (n=251).

## §4 Generalización — panel de 10 (universalidad y riesgo)

- **[NUEVO] Ablación de features (barra) M10-XGB vs AutoML:** ver §2.
- **SHAP:** cuota de las features de STRATA en el aprendiz **>0.5 en 10/10 activos, media ≈0.66** → el ML
  **redescubre las señales de STRATA**, no inventa otra señal (la parte de universalidad que se sostiene).
- **[NUEVO] ¿Redescubre o BATE a la regla? Test de equivalencia (TOST, Schuirmann 1987).** La afirmación "no
  bate" se contrasta con un TOST (no con un test de diferencia no significativo). **Resultado honesto: el TOST NO
  confirma equivalencia; el aprendiz BATE a la regla M8 en accuracy** — pooled M10 Δacc=+0.021 IC90[+0.001,+0.039]
  y AutoML +0.034 IC90[+0.010,+0.056] (modesto, 2–3 pp); SPY AutoML bate con fuerza (Δacc +0.132, Sharpe +4.25).
  En **Sharpe es no concluyente** (regla y aprendiz **indistinguibles en riesgo**). **Reencuadre:** el aprendiz
  redescubre las señales de STRATA (SHAP) y **extrae algo MÁS de accuracy que la regla fija** porque modela
  **interacciones no lineales que la regla determinista no puede** (los activos de leverage invertido donde M8
  falla, §5) — no es "otra señal", es la misma combinada con más flexibilidad. Orden honesto: **rescata al agente
  (sig) > bate modestamente a la regla en accuracy > empata con la regla en riesgo > no bate a ZeroR (nominal).**
  Fuente: `equivalence_tost.json`.
- **[NUEVO] Activación de detectores en el panel:** RAM actúa, PSA/GSO dormidos (gráfica de barras 10 activos).
- **[NUEVO] Gate RAM por activo:** cuando RAM dispara, seguir el **régimen** (override) bate a seguir al **agente**
  en **6/10** activos → ahí aporta M8; en el resto manda el canal ML. Y la **intervención de M8 crece con la
  discrepancia agente↔régimen** (Pearson **r=0.93, p<0.001**) → STRATA actúa donde el agente se aparta del régimen.
- **RESULTADO DURO — rescate de riesgo (pooled bootstrap):** canónico **pooled-15** M8 vs M5 **ΔSharpe +0.66,
  IC95 [0.225, 1.157]** (excluye 0); ΔmaxDD +0.24, IC95 [0.017, 0.445]. **pooled-10** del cuerpo consistente (mismo
  signo, IC también excluye 0), incluyendo AutoML. *(Ver nota metodológica sobre el pooled.)*

## §5 Mecanismo — dos supervisores con trabajos distintos

- **Encuadre de dos capas:** **M8 = capa de RIESGO** (rescate de Sharpe pooled significativo, interpretable, rota
  poco); **M10/AutoML = capa de ACCURACY** (McNemar vs M5 significativo). No compiten; cada función pasa su test.
- **La única ley que sobrevive un test:** el rescate del **aprendiz** en accuracy **escala con el leverage effect**
  — Pearson **r=−0.55 (p=0.034)**, Spearman ρ=−0.54 (p=0.038), **robusta a leave-one-out** (peor caso drop-MSTR
  p≈0.095 < 0.10). Ninguna variable de naturaleza predice el valor de M8 (todas p>0.10): el discriminante
  `crisis_mean` es **descriptivo, no una ley**.
- **Casos trabajados (uno por canal):** índices (régimen coherente → M8) vs MARA/cripto (leverage invertido: el
  régimen "miente" sobre la dirección → solo el aprendiz, que modela la condición, rescata).
- **[NUEVO] Timeline M8↔M10 (SPY):** coinciden ~X% de los días; en los de desacuerdo M10 acierta más → son capas
  distintas, no la misma señal.

## §6 Clustering por naturaleza — el eje que importa es el leverage

- **[NUEVO — CANÓNICO sobre los 10]** Clustering re-ejecutado **sobre los 10** del cuerpo (la versión de 15 se
  conserva en `strategy_clustering15.json` como respaldo). **Consenso unánime** de KMeans/Ward/GMM **y spectral**
  (Rand ajustado = **1.0**), silhouette **0.55** (más limpio que sobre los 15).
- **Tres grupos por naturaleza:**
  - **C0 — índices de leverage fuerte** (SPY, QQQ, XLF, DIA, XLK, XLE): lev −0.097; mejor no-trivial AutoML.
  - **C1 — leverage invertido** (SMCI, UNG): lev +0.018, agente 97% corto; mejor M10.
  - **C2 — volátiles** (ROKU, MARA): lev −0.028, vol 0.83; mejor AutoML.
- **PC1 ≈ leverage** (Pearson r≈0.84): el eje principal del clustering ES el leverage, justo el que predice el
  rescate del aprendiz (§5). Cadena cerrada: naturaleza (leverage) → eje del clustering → rescate.
- **[NUEVO] Cuatro gráficas por grupo** (réplica de `exploracion_estrategias`): (1) accuracy media por estrategia;
  (1b) **Sharpe media por estrategia** — hace visible que **M8 rescata el RIESGO** del agente (ΔSharpe ≈ +1.3 en
  C0/C2; en C1 apenas, y ahí gana M10); (2) accuracy según coincida con el **drift**; (2b) **Sharpe según coincida
  con el drift** (ir con el régimen/tendencia es mucho menos arriesgado); (3) cuota SHAP por activo.
- **Honestidad:** n=10 → exploratorio/descriptivo, no confirmatorio. Qué MODELO concreto gana por activo **no es
  predecible** por el cluster (no se infla); lo que sí se sostiene es la ley leverage→rescate.

## §7 Robustez y honestidad

- **Rodante (W=63):** la mejor STRATA supera al agente en >50% de ventanas en **8/10** activos.
- **Val/test** (60/40, 70/30, 80/20): gana al agente en val Y test en las tres en varios activos.
- **Rescate alcista vs bajista (McNemar pooled):** significativo en **ambos** regímenes (M10/AutoML p<0.02).
- **[NUEVO] PARTE B confirmatoria — ΔSharpe con cota Bonferroni + DSR (SPY y pooled-10):**
  - Confirmatorio (veredicto por **cota Bonferroni**, m=3): M10 y AutoML **SÍ** pasan (SPY +0.02 / +1.91; pooled
    +0.26 / +0.26); **M8-regla sola NO** (SPY −0.66; pooled −0.05) → **falsación honesta de la regla pura** en el
    plano riesgo: la regla rescata en accuracy pero no sobrevive el confirmatorio en Sharpe; el meta-learner sí.
  - **DSR reintroducido** (n_trials=6, aplicado por igual a los 4 brazos, 3 reprueban → no es elegir el test):
    **AutoML-SPY 0.924**; M5/M8/M10 bajos. *Matiz:* M10-SPY pasa Bonferroni por **rescate** vs un M5 pésimo, no por
    skill absoluta (su DSR=0.048, Sharpe negativo). Se quitó el DSR en el estudio SPY-solo previo (daba ~azar); se
    reintroduce porque con el meta-learner sobre panel "ya no es la misma situación".
- **[NUEVO] Rescate en Sharpe por régimen + complementariedad:**
  - SPY-solo: el rescate se concentra en **alcista** y la **regla M8 se invierte en bajista** (ΔSharpe −1.49, n=50)
    → **falsación pre-registrada** cumplida a nivel de un activo; la agregación la resuelve.
  - Pooled-10: los seis contrastes significativos en **ambos** regímenes → el rescate **no es de un solo régimen**.
  - **Complementariedad en espejo:** **M10 rescata más en alcista** (ΔSharpe +1.37 vs +0.72), **AutoML más en
    bajista** (+1.52 vs +0.81), **M8 simétrica** (+0.63 / +0.55).
  - **[NUEVO] Test DiD (pre-registrado):** la complementariedad es **significativa en el pooled** (DiD +1.37, IC95
    [+0.20, +2.60] excluye 0, p one-sided **0.008**) pero **NO en SPY-solo** (IC cruza 0; AutoML domina ambos
    regímenes en SPY) → es un **fenómeno de PANEL/cross-asset**, no de un activo. **Implicación de despliegue:**
    AutoML protege mejor en el régimen **peligroso** (bajista) → argumento para que sea la capa de accuracy
    desplegable. **Línea futura pre-registrable:** ensemble **enrutado por régimen** (M10 alcista / AutoML bajista)
    usando la señal de RAM (no validado; enrutar post-hoc sería p-hacking).
- **Robustez a la ventana de calibración** (sugerencia del tutor): acortar a 2010 no daña (incluso mejora en
  índices); **se mantiene la ventana completa pre-registrada** (cambiarla por la que maximiza el OOS sería
  p-hacking).
- **Techo ZeroR:** ninguna señal direccional bate a ZeroR causal en el OOS de forma significativa → la accuracy es
  **nominal** (ventana corta, línea futura); el valor está en **riesgo** y en el **rescate**, no en batir lo trivial.

## §8 Apéndice — límite de aplicabilidad (los 5 excluidos)

- **MSTR:** el agente ya bate a las triviales → no hay nada que rescatar (STRATA defiere).
- **BAC/NVDA/TSLA:** el agente pierde pero el rescate no alcanza significancia per-activo (n≈250) y/o es redundante
  con casos del cuerpo.
- **IWM:** caso de **borde** del discriminante. leverage_corr=−0.1022 (el más negativo del apéndice) lo haría
  "canal régimen", pero crisis_mean≈0 → el discriminante por signo es **ambiguo** (no se etiqueta como "leverage
  invertido", sería contradecir su propio leverage). Redundante con SPY/QQQ.
- **Lectura:** delimitar dónde NO aporta **refuerza** la tesis (sabemos cuándo no usarla).

## §9 Conclusiones (resumen; ver notebook §9 para la lista numerada O1–O7)

Supervisar estadísticamente a un agente LLM **aporta valor diferencial medible**, y ese valor se ordena en una
**jerarquía honesta contrastada peldaño a peldaño** (C9): el aprendiz **rescata al agente** (significativo) >
**bate modestamente a la regla en accuracy** (TOST: superior, no equivalente, por flexibilidad no lineal) >
**empata con la regla en riesgo** (Sharpe indistinguible) > **no bate a lo trivial** (ZeroR, accuracy nominal).
A esto se añaden las dos capas complementarias cuyo uso se relaciona con la naturaleza del activo (ley
leverage→rescate, complementariedad por régimen), la universalidad por SHAP (el ML redescubre las señales de
STRATA) y un mecanismo interpretable. Los límites (apéndice, leverage débil, accuracy nominal) se declaran.
**STRATA rescata, ordena el valor con rigor y acota; no genera alfa.**

---

## Trazabilidad: experimento → JSON → sección

| Resultado | Experimento | JSON | Sección |
|---|---|---|---|
| Tabla 6 estrategias + McNemar | panel canónico (automl_m10) | `automl_runs/panel_mm25_*.json` | §3 |
| Rescate riesgo pooled-15 (canónico) | decision_automl_prep | `decision_automl_prep.json` | §4, §5 |
| Series netas AutoML (reconstruidas) | automl_net_returns | `automl_net_returns.json` | §3, §4, §7 |
| Ley leverage→rescate + LOO | (en decision_automl_prep/mechanism) | `mechanism_panel.json` | §5 |
| Clustering 10 + gráficas por grupo | **cluster_panel10** | `cluster_panel10.json` | §6 |
| Activación detectores + ablación M10 | detector_ablation_panel | `detector_ablation_panel.json` | §2, §4 |
| Ablación AutoML-H2O | automl_ablation_detectors | `automl_ablation_detectors.json` | §2 |
| Matrices de confusión | confusion_panel | `confusion_panel.json` | §3, §4 |
| Anatomía de la intervención | **spy_intervention_anatomy** | `spy_intervention_anatomy.json` | §2 |
| Gate RAM + descriptivo SPY | **spy_panel_gate_descriptive** | `spy_panel_gate_descriptive.json` | §2, §4 |
| PARTE B confirmatoria + DSR + régimen | **bullbear_confirmatory** | `bullbear_confirmatory.json` | §7 |
| Test DiD complementariedad | **regime_did_learners** | `regime_did_learners.json` | §7 |
| Equivalencia/superioridad aprendiz vs regla (TOST) | **equivalence_tost** | `equivalence_tost.json` | §4 |
| Variantes de intervención SPY | spy_intervention_variants | `spy_intervention_variants.json` | §3 |
| Rodante / val-test / bull-bear panel | panel_robustness | `panel_robustness.json` | §7 |
| Robustez calibración | calib_window_panel | `calib_window_panel.json` | §7 |

(En **negrita**, los experimentos nuevos de la última fase.)

## Caveats globales declarados (para que el tribunal no los "encuentre")

1. **Accuracy nominal:** no se bate a ZeroR/B&H con significancia (ventana corta, n≈250 → línea futura).
2. **pooled:** n efectiva < nominal por correlación cruzada entre activos (ver nota metodológica); robustez por
   bootstrap-por-fechas pendiente/opcional.
3. **Clustering n=10:** exploratorio/descriptivo; qué modelo gana por activo no es predecible por el cluster.
4. **Complementariedad por régimen:** significativa en pooled, **no** en SPY-solo (fenómeno cross-asset).
5. **Selección de la cohorte 10:** ex-ante por naturaleza, no por significancia; los 5 del apéndice son el límite.
6. **STRATA no genera alfa:** todos los Sharpe absolutos siguen mayormente negativos; se mide el rescate **relativo**
   al agente, que es la tesis.
