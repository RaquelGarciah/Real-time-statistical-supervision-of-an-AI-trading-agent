# Marco teórico — guion de puntos clave para la memoria del TFG

Este documento lista **qué fundamentos teóricos** debe establecer el marco teórico de la memoria y
**por qué** cada uno: a qué componente de STRATA o a qué resultado del experimento da soporte. Está
alineado 1-a-1 con el notebook `notebooks/strata_tfg.ipynb`. Regla: en el marco teórico solo entra lo
que luego se **usa**; cada bloque debe poder señalar la sección del notebook que lo aplica.

El hilo conductor de la memoria es: **un agente de IA (LLM) toma decisiones de trading, y una capa
estadística ortogonal (STRATA) las audita en tiempo de ejecución.** El marco teórico debe armar las
dos mitades de esa frase — la estadística que sostiene la auditoría y la metodología que hace que las
conclusiones sean honestas.

---

## Bloque A — Modelado estadístico del mercado

Los detectores no inventan estructura: la leen de tres modelos clásicos del comportamiento de los
precios. Hay que presentarlos como las "lentes" con las que STRATA mira el mercado.

### A1. Regímenes de mercado y modelos ocultos de Markov (HMM)
- **Qué.** El mercado alterna estados latentes no observables (Calma / Estrés / Crisis) con dinámicas
  distintas. Un HMM gaussiano de 3 estados los modela: matriz de transición, emisiones gaussianas,
  estimación por Baum-Welch (EM) y decodificación por Viterbi / probabilidades *forward-backward*.
- **Por qué en STRATA.** Es la base del detector **RAM** y del *regime conditioning* del sizing
  (M2, M4, M8). El estado discreto es la variable contra la que se mide la coherencia de la acción del
  agente. Conviene justificar la elección de *features* `(retorno log, volatilidad realizada 21d)` y por
  qué la **volatilidad realizada** y no `log(VIX)` (el VIX incorpora prima de riesgo, contamina el
  régimen). → Notebook §2, §3.
- **Referencias.** Hamilton (1989, *regime switching*); Baum et al. (1970); Rabiner (1989); Viterbi (1967).

### A2. Volatilidad condicional: GARCH(1,1) con innovaciones t de Student
- **Qué.** La volatilidad no es constante: se agrupa (*volatility clustering*). GARCH modela
  σ²_t como función de los shocks y la varianza pasados; la t de Student captura las colas pesadas de
  los retornos diarios. Condición de estacionariedad α+β<1 ⇒ varianza incondicional finita.
- **Por qué en STRATA.** Sostiene el detector **GSO** (banda de tamaño `clip(target_vol/σ_t,0,1)`),
  el *sizing* por volatilidad de M2/M4 y las bandas de VaR. La calibración por activo (cada ticker su
  GARCH) es lo que hace el sistema multi-activo. → Notebook §2, §3.
- **Referencias.** Engle (1982, ARCH); Bollerslev (1986, GARCH); Bollerslev (1987, GARCH-t).

### A3. *Leverage effect* (correlación negativa retorno–volatilidad)
- **Qué.** En índices agregados existe correlación fuerte y negativa (≈ −0,7) entre retornos y
  volatilidad: la volatilidad sube cuando el mercado cae. Por eso, en un índice, **alta volatilidad ≈
  régimen bajista**.
- **Por qué en STRATA.** Es la **justificación teórica central** de dos decisiones: (1) elegir **SPY**
  como activo principal (el régimen funciona como proxy de la dirección, lo que da sentido a los priores
  de RAM «Calma ⇒ long, Crisis ⇒ short»); (2) el hallazgo multi-activo: en un *growth stock* como NVDA
  el efecto se **invierte** (la alta volatilidad es alcista, *melt-ups*), lo que obliga a **derivar el
  prior direccional de RAM del signo empírico del leverage effect de cada activo**, no a fijarlo. Es el
  puente entre la teoría y el resultado de §12–§13. → Notebook §0, §12, §13.
- **Referencias.** Black (1976); Christie (1982).

---

## Bloque B — Detección estadística y supervisión en tiempo de ejecución

El núcleo conceptual del TFG: vigilar a un decisor opaco con estadística pura.

### B1. Detección bayesiana de puntos de cambio en línea (BOCPD)
- **Qué.** Estima en tiempo real la distribución posterior de la *run-length* (tiempo desde el último
  cambio estructural) de una serie, con una *hazard rate* a priori y verosimilitud conjugada. Da una
  probabilidad de cambio actualizable observación a observación.
- **Por qué en STRATA.** Es el motor del detector **PSA**: aplicado al historial de *sizing* del
  agente, detecta si cambia de convicción de forma estructural e injustificada (coherencia temporal con
  su propia historia). → Notebook §3, §8.
- **Referencias.** Adams & MacKay (2007); Fearnhead (2006).

### B2. Supervisión en tiempo de ejecución y detección de anomalías
- **Qué.** Marco conceptual de un **monitor ortogonal** que observa las salidas de un sistema sin
  intervenir en su lógica interna, y que escala su respuesta (avisar / atenuar / sustituir) según la
  severidad de la desviación detectada.
- **Por qué en STRATA.** Es la tesis arquitectónica: los tres detectores son **ortogonales** (régimen
  discreto, coherencia temporal, volatilidad continua) y la capa de intervención tiene tres modos
  (warn/reduce/override). Justifica por qué una capa estadística *externa* al LLM es preferible a
  reentrenar o a confiar en la introspección del propio agente. Conecta con el resultado del **techo de
  supervisión**: STRATA aporta *disciplina de riesgo*, no alfa. → Notebook §3, §9, §11.
- **Referencias.** Marco de *runtime verification* / *anomaly detection* (Chandola, Banerjee & Kumar,
  2009, como encuadre); apóyate sobre todo en la ortogonalidad de los tres ejes definidos en el TFG.

---

## Bloque C — Validación honesta: la mitad metodológica del TFG

Una parte sustancial del trabajo **denuncia** un sesgo metodológico. El marco teórico tiene que dejar
claras las reglas de una evaluación causalmente correcta para que la denuncia (M3 vs M4) tenga base.

### C1. Causalidad temporal y sesgo de *look-ahead*
- **Qué.** En una serie temporal, una decisión tomada con información hasta `t` solo puede aplicarse al
  retorno de `t+1`. Usar información del futuro (incluso un desfase de un día mal alineado) infla
  artificialmente los resultados.
- **Por qué en STRATA.** Justifica el `signal_lag=1` del motor de backtest (`w.shift(1)`) y explica por
  qué varias "mejoras" tempranas resultaron ser *look-ahead* y se revirtieron. Es el estándar contra el
  que se miden las nueve configuraciones. → Notebook §5, §6; `docs/known_issues.md`.
- **Referencias.** López de Prado (2018, cap. 11); Bailey et al. (2014, *backtest overfitting*).

### C2. Validación cruzada en series temporales financieras: CPCV, *purging* y *embargo*
- **Qué.** El KFold convencional **fuga información** en series temporales (mezcla pasado y futuro, y
  solapa etiquetas que dependen de ventanas temporales). La **Combinatorial Purged Cross-Validation**
  (CPCV) purga del train las muestras solapadas con el test y aplica un *embargo* temporal.
- **Por qué en STRATA.** Es el corazón del contraste **M3 (KFold, sesgado a propósito) vs M4 (CPCV)**:
  el KFold invierte las conclusiones (parece que el ML acierta cuando no). Sostiene también el test
  crítico `test_no_leakage` (`max(train)+embargo ≤ min(test)`). → Notebook §7; `tests/test_no_leakage.py`.
- **Referencias.** López de Prado (2018, sec. 7.4 *purging/embargo*, cap. 12 CPCV).

### C3. Métricas de rendimiento e inferencia estadística sobre ellas
- **Qué.** El Sharpe ratio resume rentabilidad ajustada por riesgo, pero **un Sharpe alto puede ser
  suerte** cuando se prueban muchas estrategias (problema de comparaciones múltiples). El **Deflated
  Sharpe Ratio (DSR)** corrige por el número de pruebas y la no-normalidad; el test de **Diebold-Mariano**
  compara la precisión predictiva de dos estrategias; el **bootstrap** da intervalos de confianza sin
  asumir normalidad. Hace falta también caracterizar las distribuciones de retornos por sus **momentos**
  (media, varianza, asimetría, curtosis).
- **Por qué en STRATA.** Da rigor a la comparación: la **matriz de significancia pareada 9×9** (DM
  p-values + DSR) evita declarar ganadores por azar. Es lo que separa "esta config es mejor" de "esta
  config parece mejor en esta muestra". → Notebook §8.
- **Referencias.** Sharpe (1966, 1994); Bailey & López de Prado (2014, DSR); Diebold & Mariano (1995);
  Harvey, Liu & Zhu (2016, comparaciones múltiples en finanzas); Politis & Romano (1994, *stationary
  bootstrap*).

### C4. Estabilidad estructural del periodo de calibración
- **Qué.** Antes de calibrar un modelo sobre 24 años de datos conviene verificar que no hay cambios
  estructurales que invaliden una única calibración (test de rupturas tipo Bai-Perron).
- **Por qué en STRATA.** Respalda la decisión de **una sola calibración 2000→2024-09** para las nueve
  configuraciones y documenta sus límites. → Notebook §2.
- **Referencias.** Bai & Perron (1998, 2003).

---

## Bloque D — El objeto supervisado y su gestión de riesgo

### D1. Agentes de decisión basados en LLM
- **Qué.** Un LLM puede razonar sobre información financiera y emitir decisiones (acción + tamaño +
  confianza), pero es un **decisor opaco y mal calibrado**: su "confianza" no es una probabilidad
  fiable y puede ser sistemáticamente sesgado (en este OOS, *short* en un mercado alcista el 76 % del
  tiempo). No ofrece garantías estadísticas sobre sus salidas.
- **Por qué en STRATA.** Es la **motivación** de todo el sistema: si el agente tuviera garantías
  internas, no haría falta supervisarlo. El punto de interceptación es la salida del *Portfolio
  Manager* (no las opiniones individuales). Justifica también la inyección de contexto macro/sentimiento
  (SPY carece de fundamentales empresariales). → Notebook §4.
- **Referencias.** Encuádralo como *estado del arte* (agentes LLM multipersonalidad, p. ej. el proyecto
  AI Hedge Fund) más la literatura sobre **calibración y exceso de confianza de los LLM**; el peso
  teórico recae en B2 (por qué una guardarraíl estadística externa).

### D2. Gestión de riesgo: dimensionamiento por volatilidad (*volatility targeting*)
- **Qué.** Escalar la posición inversamente a la volatilidad estimada (`peso ∝ target_vol/σ_t`)
  estabiliza el riesgo de la cartera y mejora el rendimiento ajustado por riesgo frente a una exposición
  fija.
- **Por qué en STRATA.** Sostiene el *sizing* de M2/M4 y la banda del detector GSO; es el mecanismo por
  el que la estadística clásica (M2) reduce drásticamente el *drawdown* frente a B&H (M1). Aclara además
  que **el sizing no arregla una dirección equivocada** (el Sharpe es invariante al escalado uniforme):
  matiz importante en la ablación 2×2. → Notebook §2, §6, §7.
- **Referencias.** Moreira & Muir (2017, *volatility-managed portfolios*); Harvey et al. (2018,
  *the impact of volatility targeting*).

### D3. (Encuadre) Eficiencia de mercado como vara de medir
- **Qué.** La hipótesis de eficiencia del mercado explica por qué **batir al Buy & Hold es difícil** y
  por qué el benchmark correcto no es "ganar dinero" sino "ganar a B&H ajustando por riesgo".
- **Por qué en STRATA.** Encuadra la jerarquía de resultados (M1/M2 fuertes; la IA cruda no tiene
  *edge* a horizonte diario) y evita sobrevender los resultados. Basta una mención breve. → Notebook §6, §11.
- **Referencias.** Fama (1970); Fama & French (en lo justo).

---

## Cómo encajan (orden sugerido del capítulo y dependencias)

1. **Encuadre** (D3 eficiencia, D1 agentes LLM): el problema y por qué supervisarlo.
2. **Lentes del mercado** (A1 HMM, A2 GARCH, A3 leverage effect): la estadística que define "régimen" y
   "volatilidad".
3. **Teoría de la auditoría** (B1 BOCPD, B2 supervisión ortogonal): cómo se construyen los tres
   detectores y la capa de intervención.
4. **Metodología honesta** (C1 causalidad, C2 CPCV, C3 inferencia sobre métricas, C4 estabilidad): las
   reglas que hacen creíbles las conclusiones y que sostienen la denuncia del sesgo ML.
5. **Gestión de riesgo** (D2 vol targeting): el mecanismo de *sizing* común a las configuraciones.

**Mapa de dependencias rápido:** RAM ← {A1, A3}; PSA ← B1; GSO ← {A2, D2}; M3 vs M4 ← {C1, C2};
matriz 9×9 ← C3; elección de SPY y el caso NVDA/BAC ← A3; "STRATA es disciplina de riesgo, no alfa"
← {B2, C3, D3}.

**Qué NO sobre-desarrollar.** La implementación interna del LLM, frameworks de software, o derivaciones
largas de Baum-Welch/EM: cítalas y remite a la referencia. El marco teórico justifica decisiones, no
reproduce manuales.
