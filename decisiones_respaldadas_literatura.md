# Decisiones metodológicas respaldadas por literatura

Decisiones de STRATA que se apoyan en **literatura verificada** (existencia y contenido comprobados por
`@experto-citas`, sin alucinaciones). Cada una con: **Decisión + Respaldo + Citas + Matiz de atribución +
Honestidad (qué NO implica)**. Sirve como munición directa para la defensa oral. Las citas están en
`tesis/bibliography.bib`.

> Regla de oro: la literatura respalda la **elección metodológica**, no convierte un resultado frágil en
> significativo. Por eso cada decisión lleva su nota de honestidad.

---

## 1. Embargo = 1 en el walk-forward desplegable (no 5) — [2026-06-17]

**Decisión.** En la validación walk-forward de M10 el embargo es **1 día**, no 5.

**Respaldo en literatura.**
- **Purga vs embargo** (López de Prado 2018, cap. 7, §7.4): la **purga** elimina del entrenamiento las
  observaciones cuya **etiqueta se solapa en el tiempo** con la del test → su tamaño = **horizonte de la
  etiqueta**. El **embargo** elimina además unas pocas observaciones *posteriores* al test por
  autocorrelación residual (lo fija como fracción pequeña: *"A small value h ≈ 0.01·T often suffices"*).
  Ambos existen porque en K-fold/CPCV los folds tienen entrenamiento **antes y después** del test
  (interleaved, bidireccional).
- **Walk-forward rolling-origin** (Tashman 2000): el test es **siempre futuro** respecto al entrenamiento →
  **no existe** el solape bidireccional que motiva el embargo grande de CPCV. El único solape es el de la
  **etiqueta de horizonte 1** (`y_t = 1[r_{t+1}>0]`) → purga = **1**.
- **Validez con hueco mínimo** (Bergmeir, Hyndman & Koo 2018): para predictores con **residuos no
  correlados**, la validación cruzada es válida sin necesidad de un hueco grande.
- **Apoyo sobre el tamaño del hueco en datos dependientes:** *h-block* (Burman, Chow & Nolan 1994) introduce
  eliminar h vecinos; la idea de ligar h a la **estructura de dependencia** (y *hv-block*) es de Racine
  (2000); Bergmeir & Benítez (2012) respalda empíricamente el buen comportamiento de la CV en series.

**Frase de defensa.** *"El embargo ≥ 5 es una recomendación calibrada para Purged/Combinatorial K-Fold con
folds interleaved y etiquetas multi-día (López de Prado 2018, §7.4), no para evaluación walk-forward de
origen móvil con etiqueta de horizonte 1. En rolling-origin (Tashman 2000) el test es siempre futuro respecto
al entrenamiento, lo que elimina por construcción el solape bidireccional que motiva el embargo; el único
solape residual —la etiqueta `y_t = 1[r_{t+1}>0]`— se purga con embargo = 1. La validez de la validación con
hueco mínimo bajo residuos no correlados está en Bergmeir, Hyndman & Koo (2018)."*

**Matiz de atribución (respetar).** Separar **purga** (cubre el horizonte de la etiqueta) de **embargo**
(serialidad residual). La idea "h ∝ dependencia serial" se atribuye a Racine/literatura posterior, **no** a
Burman et al. (que lo dan como fracción del tamaño muestral).

**Honestidad (qué NO implica).** embargo = 1 se elige **por principio** (horizonte = 1), no por su p-valor.
Sube la accuracy de SMCI 0.524 → 0.552 (nominal), pero la significancia **no sobrevive**: el único p<0.05 vs
B&H (block-perm 0.047) es un pico aislado al embargo=1 (embargo 0 y 2 dan p≈0.12–0.13); corregido por el
barrido (Bonferroni-5 ≈ 0.28) → no significativo. Se reporta como sensibilidad.

**Citas:** `lopezdeprado2018` (§7.4), `tashman2000`, `bergmeir2018`, `burman1994`, `racine2000`,
`bergmeir2012`. Documentado en `logic_esential.ipynb` §14b, BITACORA 2026-06-17, DECISIONES_ESENCIALES #15.

---

## 2. Ensemble de semillas en M10 (promediar 10 XGBoost) — [2026-06-17]

**Decisión.** El M10 desplegable promedia las probabilidades de **10 XGBoost** que difieren solo en la
**semilla aleatoria** (`p1_ens = media(p1₁,…,p1₁₀)`); posición = `signo(p1_ens − 0.5)`. Apuesta el 100 % de
los días (cobertura completa).

**Respaldo en literatura.**
- **Principio de bagging** (Breiman 1996, *Bagging Predictors*): *"The vital element is the instability of the
  prediction method. If perturbing the learning set can cause significant changes in the predictor
  constructed, then bagging can improve accuracy."* Promediar versiones **inestables** de un predictor reduce
  el **componente de varianza** del agregado, sin sesgo añadido. XGBoost con `subsample`/`colsample_bytree <
  1` es inestable respecto a su semilla → encaja en el principio.
- **Aleatorización del aprendiz** (Dietterich 2000, *Ensemble Methods in Machine Learning*): trata
  explícitamente la **aleatorización interna del algoritmo** (no solo de los datos) como mecanismo de
  ensemble — que es exactamente la fuente de variabilidad aquí (la semilla del submuestreo).

**Frase de defensa.** *"El promediado de las probabilidades de 10 XGBoost reduce la varianza del predictor
agregado siguiendo el principio del bagging: promediar versiones inestables de un predictor disminuye su
componente de varianza sin sesgo añadido [Breiman 1996]. A diferencia del bagging clásico, que remuestrea el
conjunto de entrenamiento (bootstrap), aquí la única fuente de variabilidad entre versiones es la semilla
aleatoria que controla el submuestreo interno de cada árbol —seed averaging—, un caso de ensemble por
aleatorización del aprendiz [Dietterich 2000]. El promediado se realiza sobre modelos ya entrenados con el
mismo conjunto, por lo que no introduce look-ahead."*

**Matiz de atribución (respetar).** El *bagging* clásico de Breiman remuestrea los **datos** (bootstrap);
aquí promedio modelos que difieren solo en la **semilla** (mismo dato, distinto submuestreo interno) →
*seed averaging*. Es legítimo citar Breiman 1996 como respaldo del **principio**, pero NO presentar la
variante como "bagging" sin matizar; Dietterich 2000 cubre la aleatorización del aprendiz.

**Honestidad (qué NO implica).** El ensemble **reduce ruido, no crea señal**: mejora accuracy de forma
modesta (0.524 → 0.552) y, sobre todo, Sharpe (0.85 → 1.84) y equity (1.45× → 3.24×), pero NO es cherry-pick
(se promedian **las 10** semillas, no se elige la mejor) ni look-ahead. La significancia sigue sin
sobrevivir (DSR = 0.72 < 0.95). Es la **única** palanca probada que mejora sin romper la cobertura ni la
comparación con B&H.

**Citas:** `breiman1996`, `dietterich2000`. Documentado en BITACORA 2026-06-17, DECISIONES_ESENCIALES #14.

---

## 3. Modelo de régimen (RAM): HMM gaussiano **K=3 estados**, **filtrado** (causal)

**Decisión.** El detector RAM usa un **HMM gaussiano de 3 estados** (Calma/Estrés/Crisis) sobre
`[log-retorno, volatilidad realizada 21d]`, y se usa la posterior **filtrada** `γ^f_t = P(s_t | x_{1:t})`
(solo pasado), no la suavizada (que mira todo el histórico).

**Respaldo en literatura.**
- **Regime-switching y HMM** (Hamilton 1989; Rabiner 1989): los modelos de cambio de régimen son el marco
  estándar para series con estados latentes; K=3 (calma/estrés/crisis) es el número canónico e interpretable.
- **Filtrado vs suavizado** (Rabiner 1989, algoritmo *forward*, ecs. 18–20): la posterior filtrada usa solo
  información hasta `t` → **causal, sin look-ahead** (el suavizado *forward-backward* usaría el futuro).
- Ajuste por Baum-Welch/EM (Baum et al. 1970); estado más probable por Viterbi (1967).

**[tutor] — dicho literal en la reunión 2026-06-16:** *"el K es igual a tres ya solo por la literatura y
porque es más interpretable […] K igual a dos da peor accuracy"*. Es decir: K=3 **elegido por literatura +
interpretabilidad**, con K=2 descartado empíricamente.

**Honestidad.** El régimen captura **volatilidad**, no dirección; funciona como proxy direccional **solo**
donde se cumple el leverage effect (ver decisión #4). La feature es **volatilidad realizada** (observada), no
VIX implícito (forward-looking), por coherencia con el HMM gaussiano.

**Citas:** `hamilton1989`, `rabiner1989`, `baum1970`, `viterbi1967`. (Ya en `bibliography.bib`.)

---

## 4. Leverage effect → **SPY como caso central** y régimen como proxy direccional

**Decisión.** El caso central del método es **SPY** (índice), y la política RAM (Calma→penaliza short,
Crisis→penaliza long) se justifica porque el régimen de volatilidad sirve de **proxy de dirección**.

**Respaldo en literatura.**
- **Leverage effect** (Black 1976; Christie 1982): en índices agregados hay correlación negativa fuerte
  (~−0.7) entre retorno y volatilidad → la alta volatilidad coincide con caídas. Por eso el HMM de régimen
  actúa **implícitamente como detector direccional** en SPY.

**[tutor]:** en la reunión 2026-06-16 Raquel apoya la elección de los regímenes en que *"hay mucha literatura
de esto de los regímenes"*.

**Honestidad.** El leverage effect es propiedad de **índices**, **débil en valores individuales** → es la
razón estructural de por qué STRATA rescata en SPY (significativo) y **no** en SMCI (ver §F de
`m10_better_smci.ipynb` y DECISIONES_ESENCIALES #1, #16).

**Citas:** `black1976`, `christie1982`. (Ya en `bibliography.bib`.)

---

## 5. Volatilidad condicional (GSO): **GARCH(1,1) Student-t**

**Decisión.** El detector GSO usa la previsión de volatilidad de un **GARCH(1,1) con innovaciones t de
Student** para acotar el tamaño de la posición.

**Respaldo en literatura.** ARCH (Engle 1982) → GARCH (Bollerslev 1986) → GARCH-t para colas pesadas de los
retornos financieros (Bollerslev 1987). Es el modelo estándar de volatilidad condicional; la t de Student
captura las colas que la normal infravalora.

**Citas:** `engle1982`, `bollerslev1986`, `bollerslev1987`. (Ya en `bibliography.bib`.)

---

## 6. Coherencia temporal del agente (PSA): **BOCPD**

**Decisión.** El detector PSA usa *Bayesian Online Changepoint Detection* sobre el historial de sizing del
agente para detectar cambios estructurales de opinión (hazard 1/60).

**Respaldo en literatura.** Adams & MacKay (2007) — algoritmo BOCPD online; Fearnhead (2006) — inferencia
exacta para múltiples puntos de cambio. Es el marco bayesiano canónico para detección de rupturas en tiempo
real, sin mirar el futuro.

**Citas:** `adams2007`, `fearnhead2006`. (Ya en `bibliography.bib`.)

---

## 7. Validación cruzada de contraste: **CPCV / Purged CV**

**Decisión.** La versión **no desplegable** de M10 (contraste) usa **Combinatorial Purged Cross-Validation**
(`n_splits=6, n_test_splits=2`, embargo=5, `t1=índice.shift(-1)`). Es el contrapunto a la versión desplegable
walk-forward (decisiones #1 y #2).

**Respaldo en literatura.** López de Prado (2018), cap. 7 (purga/embargo) y cap. 12 (CPCV): metodología
diseñada para series financieras con muestra pequeña, que evita el sesgo del KFold y el desperdicio de un
walk-forward simple. **Aquí su embargo=5 es coherente** porque CPCV sí tiene folds bidireccionales (a
diferencia del walk-forward, decisión #1).

**Honestidad.** CPCV **ve bloques cronológicamente futuros** → da una estimación de backtest insesgada pero
**no simula despliegue**; por eso M10-CPCV se reporta solo como contraste (en SMCI da 0.448, peor que el WF).

**Citas:** `lopezdeprado2018`. (Ya en `bibliography.bib`.)

---

## 8. Batería de contraste estadístico (la usada en TODO este trabajo)

**Decisión.** Toda cifra se contrasta con tests apropiados, no con diferencias simples:
- **McNemar pareado** (corrección de continuidad; binomial exacto si b+c<25) para comparar accuracy de dos
  estrategias sobre los mismos días → McNemar (1947), Edwards (1948).
- **Diebold-Mariano** para comparar precisión predictiva / P&L → Diebold & Mariano (1995).
- **Sign test** (binomial exacto) contra 0.5 → Conover (1999).
- **Bootstrap estacionario / permutación por bloques** (bloque medio √N) para IC y p-valores robustos a
  autocorrelación → Politis & Romano (1994).
- **TOST** (two one-sided tests) para contrastes de **equivalencia** ("P&L indistinguible") → Schuirmann (1987).

**Respaldo en literatura.** Cada test es el estándar para su pregunta; la elección de versiones
**autocorr-robustas** (bootstrap estacionario, block-permutation) es deliberada porque los retornos diarios
están serialmente correlados.

**Honestidad.** Son los tests con los que se concluye que en SMCI **nada es significativo** tras corrección;
se aplican igual aunque el resultado sea negativo (anti-p-hacking).

**Citas:** `mcnemar1947`, `edwards1948`, `diebold1995`, `conover1999`, `politis1994`, `schuirmann1987`.
(Ya en `bibliography.bib`.)

---

## 9. **Deflated Sharpe Ratio** (corrección por multiplicidad)

**Decisión.** Todo Sharpe se reporta con su **Deflated Sharpe Ratio**, que corrige por el número de
configuraciones probadas (`n_trials`).

**Respaldo en literatura.** Bailey & López de Prado (2014): bajo `n_trials` pruebas sobre la misma serie, la
esperanza del **máximo** Sharpe muestral es positiva aunque todos los Sharpes verdaderos sean cero; el DSR
deflacta esa selección. Es la defensa formal contra el *backtest overfitting*.

**Honestidad.** Es justo lo que hace que el Sharpe del ensemble (1.84) se quede en **DSR=0.72 < 0.95** →
ilustrativo, no significativo. CLAUDE.md §4 prohíbe reportar Sharpe sin DSR.

**Citas:** `bailey2014`. (Ya en `bibliography.bib`.)

---

## 10. Sizing por **volatility targeting** (GSO)

**Decisión.** La magnitud de la posición escala inversamente a la volatilidad: `peso ∝ target_vol/σ_t`.

**Respaldo en literatura.** Moreira & Muir (2017), *volatility-managed portfolios*: gestionar la exposición
por volatilidad mejora el perfil riesgo-retorno. Es el fundamento del componente de **magnitud** del GSO
(separado del **signo**, que lo fija el régimen).

**Citas:** `moreira2017`. (Ya en `bibliography.bib`.)

---

## 11. Abstención selectiva: propuesta por la literatura, **evaluada y descartada** en SMCI

> Esta NO es una decisión adoptada, sino un método de la literatura que **probamos por rigor y descartamos**
> porque en nuestros datos no funciona. Es un resultado **negativo defendible** y muy útil para la memoria:
> "la literatura lo propone, lo implementamos, y no mejora — y entendemos por qué".

**Qué propone la literatura.** La *clasificación selectiva* / *aprendizaje con rechazo* dice que un modelo
puede **abstenerse** de predecir en los casos de **baja confianza** y así **reducir el error en los casos que
sí predice** — el clásico *trade-off error–rechazo* (Chow 1970) o *riesgo–cobertura* (El-Yaniv & Wiener 2010).
La regla óptima de Chow rechaza donde la **probabilidad a posteriori (la confianza) es baja**.

**La condición que la hace funcionar.** El beneficio existe **solo si la señal de rechazo está alineada con
dónde se equivoca el modelo** — es decir, si la confianza **ordena bien la dificultad** (correlaciona con el
acierto). Cortes, DeSalvo & Mohri (2016) lo formalizan: una función de rechazo **desacoplada** de dónde yerra
el clasificador (p.ej. un umbral de confianza fijo) es **subóptima**. Esa condición es la que falla en SMCI.

**Qué probamos (nuestro experimento).** Implementamos tres formas de abstención sobre el M10 ensemble
(`m10_smci_advanced.py`, §C.3 del notebook):
- **abst_regime**: abstenerse menos cuando el régimen HMM es decisivo (Calma/Crisis claras), más en Estrés.
- **abst_accord**: abstenerse menos cuando las 5 personalidades del agente coinciden en dirección.
- **vote_m5_m10**: actuar solo cuando M10 y el agente (M5) coinciden.

**Qué salió (embargo=1, OOS SMCI).** A **cobertura completa** son idénticas al ensemble (0.552, porque no
cambian la posición los días que sí actúan). Pero en los **días activos** —donde se supone que debería subir
el acierto— **no sube; incluso baja**:

| Método | accuracy completa | cobertura | **accuracy en días activos** |
|---|---|---|---|
| ensemble (sin abstención) | 0.552 | 100 % | 0.552 |
| abst_regime | 0.552 | 75 % | **0.489** ← más baja que la completa |
| abst_accord | 0.552 | 77 % | 0.513 |
| vote_m5_m10 | 0.552 | 46 % | 0.557 (≈ completa) |

**Conexión con la literatura (lo importante).** El resultado es **exactamente lo que predice la teoría cuando
su premisa no se cumple**: en SMCI la confianza del modelo, el régimen y el acuerdo del agente **no
discriminan** qué días son más fáciles → la función de rechazo no está alineada con el error (condición de
Cortes et al. 2016) → abstenerse no reduce el error, y la abstención por régimen incluso **descarta días
buenos** (0.489 < 0.552). Además, abstenerse baja la **cobertura**, rompiendo la comparación justa con B&H
(que apuesta el 100 %).

**Frase para la memoria.** *"Evaluamos la abstención selectiva (Chow 1970; Cortes, DeSalvo & Mohri 2016):
abstenerse en los días de baja confianza mejora la accuracy únicamente si la señal de rechazo está alineada
con el error del modelo. En SMCI esa condición no se cumple —la confianza, el régimen y el acuerdo del agente
no ordenan la dificultad—, de modo que la abstención no mejora la accuracy en los días operados (e incluso la
reduce: 0.489 vs 0.552 en la variante por régimen) y además rompe la comparabilidad con el pasivo al reducir
la cobertura. Por ello se descarta y el modelo final opera a cobertura completa."*

**Citas:** `chow1970`, `cortes2016rejection`, `elyaniv2010`. (Añadidas a `bibliography.bib` [2026-06-17].)

---

## 12. Baseline de no-habilidad = **clase mayoritaria (ZeroR)** / *no-information rate* — [2026-06-17]

**Decisión.** Se compara la accuracy direccional contra **seis estrategias**: los 3 modelos (M5 agente, M8
regla, M10 meta-learner) y **3 baselines triviales** — **B&H** (*buy-and-hold* = siempre largo), **S&H**
(*short-and-hold* = siempre corto) y la **clase mayoritaria** (regla **ZeroR**: predecir siempre la dirección
dominante; accuracy = *no-information rate*, NIR). B&H y S&H son las dos estrategias constantes; la clase
mayoritaria es la mejor de las dos = `max(B&H, S&H)`. **En SMCI, donde predominan los días bajistas, la clase
mayoritaria se materializa como S&H ("siempre corto"), NIR=0.516** (coinciden numéricamente en este activo).

**Respaldo en literatura.**
- **ZeroR / clase mayoritaria** (Witten, Frank, Hall & Pal 2016, *Data Mining*): el baseline de no-habilidad
  estándar en clasificación; predecir siempre la clase más frecuente, contra el que se evalúan los modelos.
- **No-information rate + test** (Kuhn 2008, *caret*): define el NIR = "the largest class percentage in the
  data" y propone el **test binomial unilateral** (accuracy del modelo > NIR). Es el contraste correcto:
  comparar la accuracy contra la **clase mayoritaria**, no contra el 0.5 ingenuo.

**Por qué importa (y por qué es más fuerte que B&H).** B&H = "siempre largo" solo es buen baseline si suben
más días. En un activo bajista/lateral como SMCI, la clase mayoritaria ("siempre corto", NIR=0.516) es un
**listón más duro** que B&H (0.484). M10 (0.552) **bate a los dos** → tiene discriminación direccional real,
no un simple sesgo a un lado. Es el argumento que **mata la objeción "M10 solo gana por estar corto"**: en
los tramos bajistas la clase mayoritaria ES "siempre corto", y M10 también la supera.

**Frase para la memoria.** *"El baseline de no-habilidad es el clasificador de clase mayoritaria (regla ZeroR
[Witten et al. 2016]), cuya accuracy iguala la frecuencia de la clase predominante —el no-information rate
[Kuhn 2008]—; en SMCI, donde predomina la clase 'baja', equivale a una posición permanentemente corta. M10 la
supera (0.552 > 0.516), además de superar a B&H, lo que descarta que su ventaja sea un mero sesgo direccional."*

**Honestidad.** El margen sobre la clase mayoritaria (+3.6 pts en el OOS) es **más estrecho** que sobre B&H
(+6.8) — como debe ser, es un baseline más exigente. El contraste honesto de significancia es el **binomial
vs NIR** (Kuhn 2008), no vs 0.5.

**Citas:** `witten2016datamining`, `kuhn2008caret`. (Añadidas a `bibliography.bib` [2026-06-17].)

---

## Referencias (verificadas; APA-español)

*Todas las claves están en `tesis/bibliography.bib`. Las del embargo y el ensemble (decisiones #1–#2) fueron
verificadas explícitamente por `@experto-citas` [2026-06-17]; el resto provienen del marco metodológico ya
establecido del proyecto (docstrings de `core/` y `strata/`).*

- Breiman, L. (1996). Bagging predictors. *Machine Learning, 24*(2), 123–140. https://doi.org/10.1007/BF00058655
- Bergmeir, C. y Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation.
  *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028
- Bergmeir, C., Hyndman, R. J. y Koo, B. (2018). A note on the validity of cross-validation for evaluating
  autoregressive time series prediction. *Computational Statistics & Data Analysis, 120*, 70–83.
  https://doi.org/10.1016/j.csda.2017.11.003
- Burman, P., Chow, E. y Nolan, D. (1994). A cross-validatory method for dependent data. *Biometrika, 81*(2),
  351–358. https://doi.org/10.1093/biomet/81.2.351
- Dietterich, T. G. (2000). Ensemble methods in machine learning. En *Multiple Classifier Systems (MCS 2000)*,
  LNCS (Vol. 1857, pp. 1–15). Springer. https://doi.org/10.1007/3-540-45014-9_1
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (cap. 7, §7.4). John Wiley & Sons.
- Racine, J. S. (2000). Consistent cross-validatory model-selection for dependent data: hv-block
  cross-validation. *Journal of Econometrics, 99*(1), 39–61. https://doi.org/10.1016/S0304-4076(00)00030-0
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review.
  *International Journal of Forecasting, 16*(4), 437–450. https://doi.org/10.1016/S0169-2070(00)00065-0

*Marco metodológico ya establecido del proyecto (claves preexistentes en `bibliography.bib`):*

- Adams, R. P. y MacKay, D. J. C. (2007). *Bayesian online changepoint detection*. arXiv:0710.3742. `[adams2007]`
- Bailey, D. H. y López de Prado, M. (2014). The Deflated Sharpe Ratio: correcting for selection bias,
  backtest overfitting, and non-normality. *Journal of Portfolio Management, 40*(5). `[bailey2014]`
- Baum, L. E., Petrie, T., Soules, G. y Weiss, N. (1970). A maximization technique occurring in the
  statistical analysis of probabilistic functions of Markov chains. *Annals of Mathematical Statistics,
  41*(1), 164–171. `[baum1970]`
- Black, F. (1976). Studies of stock price volatility changes. *Proceedings of the American Statistical
  Association, Business and Economic Statistics Section*, 177–181. `[black1976]`
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of
  Econometrics, 31*(3), 307–327. `[bollerslev1986]`
- Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates
  of return. *Review of Economics and Statistics, 69*(3), 542–547. `[bollerslev1987]`
- Christie, A. A. (1982). The stochastic behavior of common stock variances. *Journal of Financial
  Economics, 10*(4), 407–432. `[christie1982]`
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information
  Theory, 16*(1), 41–46. https://doi.org/10.1109/TIT.1970.1054406 `[chow1970]`
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3.ª ed.). Wiley. `[conover1999]`
- Cortes, C., DeSalvo, G. y Mohri, M. (2016). Learning with rejection. En *Algorithmic Learning Theory (ALT
  2016)*, LNCS (Vol. 9925, pp. 67–82). Cham: Springer. https://doi.org/10.1007/978-3-319-46379-7_5
  `[cortes2016rejection]`
- El-Yaniv, R. y Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of
  Machine Learning Research, 11*, 1605–1641. `[elyaniv2010]`
- Diebold, F. X. y Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic
  Statistics, 13*(3), 253–263. `[diebold1995]`
- Edwards, A. L. (1948). Note on the "correction for continuity" in testing the significance of the
  difference between correlated proportions. *Psychometrika, 13*(3), 185–187. `[edwards1948]`
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United
  Kingdom inflation. *Econometrica, 50*(4), 987–1008. `[engle1982]`
- Fearnhead, P. (2006). Exact and efficient Bayesian inference for multiple changepoint problems.
  *Statistics and Computing, 16*(2), 203–213. `[fearnhead2006]`
- Kuhn, M. (2008). Building predictive models in R using the caret package. *Journal of Statistical
  Software, 28*(5), 1–26. https://doi.org/10.18637/jss.v028.i05 `[kuhn2008caret]`
- Witten, I. H., Frank, E., Hall, M. A. y Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools
  and Techniques* (4.ª ed.). Burlington, MA: Morgan Kaufmann. `[witten2016datamining]`
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the
  business cycle. *Econometrica, 57*(2), 357–384. `[hamilton1989]`
- López de Prado, M. (2018). *Advances in Financial Machine Learning* (cap. 7 y 12). John Wiley & Sons.
  `[lopezdeprado2018]`
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or
  percentages. *Psychometrika, 12*(2), 153–157. `[mcnemar1947]`
- Moreira, A. y Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance, 72*(4), 1611–1644.
  `[moreira2017]`
- Politis, D. N. y Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical
  Association, 89*(428), 1303–1313. `[politis1994]`
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
  *Proceedings of the IEEE, 77*(2), 257–286. `[rabiner1989]`
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for
  assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics,
  15*(6), 657–680. `[schuirmann1987]`
- Viterbi, A. J. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding
  algorithm. *IEEE Transactions on Information Theory, 13*(2), 260–269. `[viterbi1967]`

*Todas las claves están en `tesis/bibliography.bib`. Para añadir una decisión nueva: verifica la cita con
`@experto-citas` antes de escribirla aquí, y amplía `bibliography.bib` si la clave no existe.*
