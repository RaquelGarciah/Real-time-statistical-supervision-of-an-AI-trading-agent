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
modesta (0.52 → 0.552) y, sobre todo, Sharpe (0.85 → 1.84) y equity (1.45× → 3.24×), pero NO es cherry-pick
(se promedian **las 10** semillas, no se elige la mejor) ni look-ahead. La significancia sigue sin
sobrevivir (DSR = 0.72 < 0.95). Es la **única** palanca probada que mejora sin romper la cobertura ni la
comparación con B&H.

**Citas:** `breiman1996`, `dietterich2000`. Documentado en BITACORA 2026-06-17, DECISIONES_ESENCIALES #14.

---

## Referencias (verificadas por @experto-citas; APA-español)

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

*Todas las claves están en `tesis/bibliography.bib`. Para añadir una decisión nueva: verifica la cita con
`@experto-citas` antes de escribirla aquí.*
