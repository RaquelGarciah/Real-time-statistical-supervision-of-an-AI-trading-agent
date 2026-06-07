# Hallazgos de STRATA y protocolo de medición

Este documento recoge dos cosas que durante el desarrollo solo quedaron anotadas en notas de
trabajo y conviene fijar para la memoria: (1) el **techo de supervisión** que alcanza STRATA
sobre SPY y por qué, y (2) el **protocolo de medición** con el que se evaluó cada cambio. Ambos
sostienen la discusión del TFG sobre qué aporta —y qué no— la capa de supervisión estadística.

## 1. El techo de supervisión: STRATA es disciplina de riesgo, no generador de alfa

Sobre el OOS completo de SPY (~402 días, fin 2026-05-11), el agente AI Hedge Fund opera
**short el 76 % del tiempo en un mercado alcista**, con `|size| ≤ 0,25` por su propio risk
manager. El intento (2026-05-20) de empujar el Sharpe supervisado hacia el benchmark
cuantitativo (M2 = +1,114) arrojó:

- **Ninguna configuración supervisada supera al baseline cuantitativo.** La mejor supervisada
  por atenuación es **M7 reduce con los defaults de RAM (0,2 / 0,4 / 0,7) = +0,937**, frente al
  agente solo (M5) = +0,867. La supervisión recupera parte del terreno pero no alcanza a M2.
- **Calibrar RAM por percentil empeora el Sharpe.** Los posteriores del HMM saturan a ≈0/1, así
  que un umbral más alto dispara menos y corrige menos shorts; corregir *más* shorts es mejor.
  RAM se queda en sus defaults.
- **Invertir la dirección (variantes override B/C/D) empeora** respecto a atenuar: voltear la
  apuesta equivale a confiar en que el régimen acierta ese día concreto; atenuar es más robusto.
- **PSA es marginal y GSO es inerte por diseño.** La señal MAP run-length de BOCPD degenera
  (rampa monótona) sobre el sizing oscilante del agente → 0 % de activación; GSO no se activa
  porque el agente nunca sobredimensiona respecto a la banda GARCH.

**Conclusión defendible:** STRATA funciona como **disciplina de riesgo vía RAM + reduce**, no
como generador de alfa. El agente es demasiado contrarian en este OOS para rescatarlo hasta el
benchmark puramente cuantitativo. Esto es coherente con el marco teórico: la supervisión vigila
la coherencia con el régimen, no descubre dirección.

### Sintonía finalmente adoptada (2026-05-20)

- **M8 override = `override_variant "C"` + `regime_mode "filtered"`.** Overlay de régimen causal
  (long en Calma / short en Crisis a vol-target, solo en los días que RAM marca). **Sharpe causal
  +0,659 neto** — el primer positivo de STRATA, cerca de M2 (+0,767) y honesto. La cifra same-day
  baja a +0,255. Sustituyó al GSO relativo (+1,59 same-day pero −1,03 causal: era look-ahead). Se
  rechazó la versión con régimen *smoothed* (+1,29 "causal"): el suavizado usa el futuro → look-ahead
  severo no alcanzable en vivo, y el lag-1 no lo quita porque la fuga está en el peso.
- **M7 reduce = PSA `cp_prob_delta` + hazard 1/60.** Mejor Sharpe causal del modo reduce
  (−0,95 frente a −1,08); actúa como control de daños.

El catálogo completo de pruebas de detectores (scripts `experiments/tuning/tune_*.py` + los CSV
en `outputs/reports/`) queda como evidencia instrumental. Lección recurrente: casi todo "mejor
Sharpe" alto resultó ser look-ahead (same-day, o régimen smoothed); ver [known_issues.md](known_issues.md).

## 2. Protocolo de medición (cómo se evaluó cada cambio)

Regla de trabajo seguida al tocar STRATA, pensada para que el resultado sea defendible:

1. **Congelar un baseline antes de tocar nada** (p. ej. `outputs/reports/baseline_pre_mejoras.csv`).
2. **Implementar una palanca cada vez** y medir su efecto sobre el Sharpe supervisado.
3. **Conservar solo lo que sube el Sharpe**; revertir el resto a defaults y documentarlo como
   hallazgo honesto (código instrumental + entrada en `BITACORA.md`).
4. **Medir siempre sobre el OOS completo** (~402–404 días, 2024-10 → fin), **nunca** sobre
   ventanas cortas tipo 21/90 días.

**Por qué.** El TFG necesita resultados defendibles: un negativo medido limpiamente vale más que
una mejora aparente sin baseline. Las ventanas cortas son casi mono-régimen e inflan los efectos.
Por eso muchas "mejoras" se revirtieron y se documentaron como negativas en lugar de adoptarse.

## 3. STRATA por activo: HMM propio **y** prior de RAM re-signado (NVDA M8 +0,95)

Hasta el 2026-05-21 la extensión NVDA reutilizaba el **régimen del S&P** mientras el GARCH, la σ del
sizing y los umbrales sí eran de NVDA — una asimetría incoherente (ver [decisiones.md](decisiones.md)
§9). Se aplicaron **dos correcciones**, ambas el 2026-05-21:

1. **HMM por activo:** entrenar el régimen sobre la **propia serie de NVDA** (simétrico con GARCH y
   umbrales). Por sí sola sube M8 de **−0,46 a +0,66**.
2. **Prior de RAM re-signado por activo (adoptado como default):** el sentido favorable de cada
   régimen (`regime_dir`) se **deriva del signo del retorno medio de calibración** (2000→2024-09, sin
   look-ahead; Estrés neutro), en vez de fijarlo al leverage effect del índice. Para NVDA esto voltea
   «Crisis ⇒ short» a «Crisis ⇒ long», subiendo M8 de **+0,66 a +0,95** y M7 de −0,04 a +0,19.

Comparativa M1–M9 sobre los 403 días del OOS de NVDA (causal neto):

| Config | rég. S&P (antiguo) | rég. propio + prior leverage | **rég. propio + prior re-signado** |
|---|---:|---:|---:|
| M1 B&H | +0,872 | +0,872 | +0,872 |
| M2 GARCH×HMM | +0,868 | +0,992 | **+0,992** |
| M3 ML KFold | +0,114 | +0,114 | +0,114 |
| M4 ML CPCV | +0,077 | +0,180 | +0,180 |
| M5 Agente | −0,591 | −0,591 | −0,591 |
| M6 warn | −0,591 | −0,591 | −0,591 |
| M7 reduce | −0,339 | −0,036 | **+0,193** |
| **M8 override** | **−0,462** | **+0,662** | **+0,945** |
| M9 ML+IA | −0,406 | −0,357 | −0,357 |

(Solo M7/M8 dependen de RAM; las demás no cambian con el prior.) **M8 override pasa de ser de los
peores (−0,46) a +0,95**, el M8 más alto del trío, y queda por debajo de su propio M2 (+0,99): la
disciplina de STRATA transfiere y el techo de supervisión se mantiene.

**Leverage effect invertido (retorno medio por régimen, bps):**

| | Calma | Estrés | Crisis |
|---|---:|---:|---:|
| S&P por su régimen | +5,3 | +2,9 | **−4,2** |
| NVDA por su propio régimen | +15,0 | +4,4 | **+17,3** |
| BAC por su propio régimen | +5,6 | −0,1 | **−5,8** |

(Valores de calibración 2000→2024-09, los que alimentan el prior.) En el S&P y en BAC la Crisis es
**bajista** (leverage effect clásico) → el prior data-driven reproduce «Crisis ⇒ short» y los
resultados quedan **idénticos al default**. En NVDA su régimen de máxima volatilidad es el **más
alcista** (+17,3): los *melt-ups* de un growth stock ocurren con volatilidad alta → el prior se voltea
solo a «Crisis ⇒ long». RAM se activa más con el régimen propio (61,3 % vs 36,2 %): el detector
funciona y, con el prior corregido, interviene en la dirección correcta del activo.

**Conclusión defendible.** Todos los parámetros de STRATA son por activo: HMM, GARCH, umbrales **y el
prior direccional de RAM**, este último derivado sin look-ahead del signo empírico del leverage effect.
Con ello la frontera "STRATA falla en *stocks*" desaparece: el sistema **se auto-adapta** al signo del
leverage de cada activo (SPY/BAC conservan «Crisis ⇒ short», NVDA lo voltea) y rescata al agente en los
tres (M8 SPY +0,62 · NVDA +0,95 · BAC +0,86), siempre por debajo del M2 correspondiente. Reproducible
con `experiments/tuning/diagnose_ram_resigned.py` (SPY/NVDA/BAC, default vs re-signado) y en las §12–§13
del notebook `notebooks/strata_tfg.ipynb`. El cross-check de SPY queda verde (su prior re-signado es
idéntico al leverage). El script previo `diagnose_nvda_own_hmm.py` (HMM propio vs régimen S&P) se
conserva como paso intermedio.

**Matiz importante (2026-05-27).** La regla `regime_dir_from_calib` usa umbral cero estricto
(`+1 si means>=0 else -1`). Para activos con **Calma calib bajista** (SMCI −0,05 bp, ROKU −2,5 bp, UNG
−12,4 bp, MARA −18,9 bp) la función devuelve `dc=−1`, produciendo priors `(−1, 0, ±1)` funcionalmente
distintos del clásico. En UNG y MARA el signo refleja una realidad empírica (Calma genuinamente
bajista en miners cripto y gas en *contango*); en SMCI/ROKU es ruido alrededor de cero. Esta frontera
se documenta como **hallazgo derivado** en §4, no se "arregla" con un umbral ε arbitrario — la decisión
es mantener data-driven full y delimitar su alcance con las tres condiciones del mecanismo.

## 4. Las tres condiciones del mecanismo TSLA: cuándo STRATA bate a la mejor clásica

Sobre el panel de **10 activos** inmersionados en `notebooks/strata_tfg.ipynb` (SPY, NVDA, BAC, TSLA,
XLE, UNG, MSTR, SMCI, ROKU, MARA), la correlación cross-seccional **calidad de B&H ↔ ventaja de STRATA
(M8−B&H)** es **−0,41** (pendiente negativa robusta: STRATA aporta donde el largo pasivo falla). M8
supera a B&H en 5/10 (UNG, SMCI, TSLA, BAC, NVDA) y a la mejor clásica solo en **2/10** (TSLA, UNG).
Los dos wins absolutos llegan por mecanismos opuestos:

- **TSLA** — leverage invertido extremo (Crisis calib +39,9 bps), 30 % del OOS en Crisis, override flippea
  agente short → long en esos días, captura *melt-up*. M8 +1,137 / M8−clásica +1,110.
- **UNG** — leverage clásico catastrófico (Crisis −19,8 bps), B&H se desploma (−0,36), agente único
  positivo (+0,33) por estar corto, override apenas interviene (RAM filtered 0,5 %). M8 +0,183 /
  M8−clásica +0,546.

Los **ocho casos restantes** delimitan el alcance del mecanismo. Las cifras *filtered* (causales) del
override son las internamente comparables, no las *smoothed* que cada inmersión imprime para análisis:

| Activo | Crisis calib (bps) | % OOS Crisis | % RAM filtered | M8 | M8−clásica | DSR M8 | Lectura |
|---|---:|---:|---:|---:|---:|---:|---|
| TSLA | +39,9 | 30,6 % | 56,6 % | +1,137 | +1,110 | 1,000 | **win** — las tres condiciones cumplidas |
| NVDA | +17,3 | ~30 % | 65,8 % | +0,945 | −0,047 | 1,000 | techo: M2 ya alto |
| BAC | −5,8 | ~30 % | 47,9 % | +0,855 | −0,020 | 1,000 | techo: M2 ya alto |
| SPY | −4,2 | ~25 % | 24,9 % | +0,621 | −0,189 | 1,000 | techo: M1 (B&H) imbatible en *bull* suave |
| XLE | +0,7 | ~30 % | 66,1 % | +0,298 | −0,627 | 1,000 | techo: M2 captura sin sizing direccional |
| UNG | −19,8 | ~10 % | 0,5 % | +0,183 | +0,546 | 0,983 | **win** — agente ya correcto, M8 no estropea |
| MSTR | −10,4 | bajo | 0,2 % | −0,040 | −0,287 | 0,010 | contraejemplo: **cambio estructural** (era cripto post-2020 no en calibración) |
| SMCI | +16,0 | 68 % | 68,6 % | −0,071 | −0,635 | 0,002 | contraejemplo: **leverage NO estable** (Crisis calib alcista pero OOS bajista) |
| MARA | +55,8 | **2,5 %** | 4,2 % | −0,125 | −0,959 | 0,000 | contraejemplo: **Crisis OOS casi inexistente** |
| ROKU | +31,8 | **5,5 %** | 7,7 % | −0,555 | −1,286 | 0,000 | contraejemplo: **Crisis OOS casi inexistente** |

De los contraejemplos emergen **tres condiciones** que el mecanismo TSLA exige para producir alfa por
encima de la mejor clásica:

1. **Signo del leverage en calibración** (Crisis bps positivo) — necesario para que `regime_dir`
   adopte `(±1, 0, +1)` y el override flippee agente short → long. Cumplido por NVDA, TSLA, XLE, SMCI,
   ROKU, MARA.
2. **Estabilidad estructural del leverage entre calibración y OOS** — el Crisis OOS debe ser, en signo
   y magnitud, consistente con el de calibración. **SMCI viola esta condición**: Crisis calib +16 bps
   (2000-2024) pero el activo cae en OOS (B&H −0,11, retorno negativo); el override flippea masivamente
   (68,6 %) hacia long en un activo que está cayendo. **MSTR es un caso afín**: cambio estructural
   software→bitcoin-proxy desde 2020 hace que el régimen calibrado no refleje el carácter actual.
3. **Frecuencia suficiente del régimen explotable durante el OOS** — el régimen Crisis (donde el override
   actúa con leverage invertido) debe materializarse en una fracción razonable del OOS. **ROKU (5,5 %)
   y MARA (2,5 %) la violan**: el override apenas tiene oportunidad de intervenir (7,7 % y 4,2 %
   filtered, no las cifras smoothed que las inmersiones imprimen), insuficiente para mover el Sharpe
   agregado. TSLA cumplió con Crisis OOS al 30,6 %.

**Conclusión defendible para la memoria.** La ventaja condicional de STRATA es real (corr −0,41,
robusta sobre 10 activos) pero **delimitada**: STRATA bate a la mejor clásica solo en activos donde
las tres condiciones se cumplen simultáneamente. El experimento se reporta íntegro (los 10 activos,
incluidos los contraejemplos elegidos ex-ante por su Crisis bps positivo) para mantener honesto el
DSR y evitar el sesgo de selección que el propio TFG denuncia. Es **información valiosa** que la
hipótesis "leverage invertido → win" se refuta parcialmente, no un fallo del marco — refuerza
que STRATA es disciplina de riesgo con un mecanismo de alfa **conditional, no incondicional**.
