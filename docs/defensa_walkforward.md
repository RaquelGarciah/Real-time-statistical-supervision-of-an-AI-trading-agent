# Defensa oral — Walk-forward de STRATA (robustez multi-ventana)

Cifras de `outputs/experiments/walkforward_robustez.json` y BITACORA [2026-06-09] [Hallazgo].
Preparado para defender ante el tutor/tribunal. Listo para decir en voz alta.

## Posición global (una frase)

> "Hice exactamente lo que me pidió: lancé el método en muchas ventanas distintas. El resultado
> honesto es que el modelo de régimen sí aguanta a través de 24 años incluyendo crisis, pero el
> rescate del agente solo es robusto cuando el mercado sube. Lo sé porque puse una regla por
> escrito ANTES de mirar los datos para cazar justo eso, y saltó."

No defiendo un rescate universal. Defiendo un **sistema que sabe dónde no funciona** (§4f).

## Lo que SÍ defiendo vs lo que CONCEDO

**SÍ (con datos):**
- El HMM K=3 generaliza inter-época: gana a K=2 en 15/16 orígenes (incl. 2008/2020/2022).
- La validación es honesta y pre-registrada: la regla de falsificación se escribió antes y se respetó.
- En el agregado, M8 mejora a M5 en TODAS las métricas (acc, MCC, Sharpe).

**CONCEDO (sin pelear):**
- El rescate en Sharpe NO es robusto multi-ventana: las cotas Bonferroni (M8−M5=−0.49; M10−M5=−0.48)
  ambas <0 → H1_b no se sostiene. El IC95 crudo de M10−M5=[−0.02,+5.79] roza el cero pero el
  criterio pre-registrado es la cota Bonferroni, no el IC95: no hay base para afirmar robustez.
- El ΔSharpe se INVIERTE en bajista: M8=−3.92, M10=−1.06. La falsificación pre-registrada se
  dispara para AMBOS en el plano Sharpe.
- La accuracy de M8 en alcista no sobrevive Holm (p_adj=0.15); en bajista es nula (p=1.0).
- Ni M8 ni M5 ganan direccionalmente contra B&H (acc < 0.566).

**DISTINGO (importante para no conceder más de lo que toca):**
- En el plano **accuracy**, M10 rescata al agente en AMBOS regímenes: alcista Holm p_adj=0.005
  (block-perm p=0.000) y bajista Holm p_adj=0.075 (block-perm p=0.061, robusto a autocorrelación).
  Lo que se invierte en bajista es el **Sharpe** (plano económico), no el **acierto de dirección**.
  Son ejes distintos: el primero cuenta signos, el segundo pondera por magnitud del retorno.

---

## OBJ 1 (la más dura) — "Tu método no funciona / solo ganas porque el mercado subió"
- **Concedo (plano Sharpe):** el rescate en P&L en SPY es condicional a que el mercado suba. Lo digo
  yo primero. El ΔSharpe se invierte en bajista tanto para M8 (−3.92) como para M10 (−1.06). La
  falsificación pre-registrada saltó para ambos.
- **Distingo (plano accuracy):** en el plano de **acierto de dirección** la historia es distinta.
  M10 rescata accuracy al agente en **ambos** regímenes: alcista p_adj Holm=0.005 (block-perm=0.000)
  y bajista p_adj Holm=0.075 (block-perm=0.061, robusto a autocorrelación). El fallo en bajista es
  económico (Sharpe), no de dirección. Acertar la dirección y ganar dinero son ejes distintos: el
  Sharpe pondera por la magnitud de los retornos, y en bajista las caídas son mayores en magnitud.
- **Giro general:** no me pilló por sorpresa — pre-registré la regla de falsificación antes de mirar
  datos ("si el ΔSharpe se invierte en bajista con ≥60 días, no es universal"). El bajista tuvo 123
  días. La regla saltó. No oculto el fallo: lo cacé con una trampa que yo misma puse.
- **Implicación:** no invalida el TFG; lo convierte en ciencia falsable. "STRATA-SPY recupera
  accuracy direccional de forma robusta (M10, ambos regímenes); su ventaja económica (Sharpe) es
  condicional al alza. Que el sistema distinga dónde funciona y dónde no es la aportación."
- **Back-up:** separo "el modelo generaliza" (sí, Parte A) de "el rescate en Sharpe es robusto" (no).
- Cifras M8: `drift.bajista.delta_sharpe=−3.92 (n=123)`, `alcista=+8.45 (n=278)`.
  Cifras M10: `drift.bajista.delta_sharpe=−1.06 (n=123)`, accuracy bajista Holm=0.075, falsif_spy_m10_bajista=True (plano Sharpe).

## OBJ 2 — "¿Por qué K=3 y no K=4, si K=4 da mejor verosimilitud?"
- **Concedo:** K=4 mejora el LL held-out a K=3 en 14/16. Por verosimilitud pura ganaría K=4.
- **Giro:** elijo K=3 por (1) ya bate a K=2 en 15/16 incl. crisis → el 3er estado es régimen real;
  (2) mapea a Calma/Estrés/Crisis (legible); (3) el estado Estrés = abstención (el supervisor no
  interviene ahí). K=2→K=3 capta estructura; K=3→K=4 es ajuste fino con coste de interpretabilidad.
- **Implicación:** criterio de información (pediría K≥4) vs criterio funcional (interpretabilidad +
  abstención). Elijo funcional: es un supervisor, no un ajustador de curvas.
- Cifras: `k3_domina_frac=0.938`; K=4>K=3 en 14/16 (`per_origin_K`).

## OBJ 3 — "El IC de tu ΔSharpe incluye el cero. ¿Qué demuestras?"
- **Concedo:** el test confirmatorio (único pre-declarado) da mediana +2.45 pero IC95 [−0.21, +5.71]
  → no puedo afirmar robustez al re-muestreo.
- **Giro:** "incluye el cero" = no concluyente, no nulo. Punto en +2.45; con N≈400 y efecto pequeño la
  potencia es baja (lo pre-registré). Ausencia de evidencia ≠ evidencia de ausencia.
- **Implicación:** por eso la evidencia fuerte no es el Sharpe (Deflated Sharpe=0.50 ≈ azar) sino el
  McNemar pareado del acierto direccional en el agregado.
- **Back-up (la curva que el tutor entiende):** €1000 → agente €903, STRATA €1069; por encima en el
  global, pero no garantizado en cada sub-periodo.
- Cifras: `median_delta_sharpe=2.45`, `ci95=[−0.21,5.71]`, `dsr=0.50`, equity M5=0.9035/M8=1.0689.

## OBJ 4 — "Todo es una sola ventana alcista de 18 meses; el panel comparte periodo"
- **Concedo:** la Parte B (rescate con agente) vive entera en un único OOS alcista. No es robustez
  inter-época. No lo finjo.
- **Giro:** por eso partí la validación en dos. El agente LLM no puede correr en 2008/2020 (cutoff
  2024 → sería look-ahead). La robustez inter-época recae en la **Parte A** (24 años, sin agente),
  y ahí K=3 aguanta (15/16).
- **Implicación:** robustez temporal del modelo, sí; del rescate, solo intra-OOS (límite de diseño).
- **Back-up:** el panel da robustez transversal (no temporal): ΔSharpe>0 en 9/10 activos.

## OBJ 5 (trampa) — "El 73.7% de ventanas positivas, ¿no es evidencia?"
- **Concedo el número, niego que sea evidencia.**
- **Giro:** las 56 ventanas se solapan (comparten 115/120 días), ρ_lag1=0.98 → N efectivo de Bartlett
  ≈ 0.6. Contar 56 como independientes sería inflar la muestra — el error que el rigor me prohibió.
- **Implicación:** lo degradé a descriptivo; el confirmatorio es el bootstrap sobre la serie diaria
  pareada (re-muestrea días, no ventanas) y da inconcluso.
- Cifras: `frac_positive=0.737`, `rho_lag1=0.979`, `n_eff_bartlett=0.6`.

## OBJ 6 — "Si M8 sigue siendo perdedor direccional (acc<0.5), ¿qué has rescatado?"
- **Concedo (para M8):** M8 acierta 0.436 (walk-forward agregado: 0.454), por debajo del azar y del
  0.566 de B&H. No vuelvo ganador al agente con la regla a mano.
- **Amplío con M10:** M10 acierta 0.539, y es el único modelo con MCC positivo (+0.068). Sigue por
  debajo de B&H pero predice la dirección, no solo cabalga el drift. Es el mejor decodificador
  de la señal STRATA y rescata accuracy en alcista Y bajista (§13 walk-forward).
- **Giro general:** el objeto no era predecir el mercado, era **rescatar** = reducir daño medible y
  pareado. M8 mejora a M5 en: acc 0.436 vs 0.384, Sharpe +0.67 vs −1.82. M10 mejora más: acc 0.539.
- **Implicación:** "STRATA reduce el daño; M10 sobre features STRATA alcanza casi la accuracy de B&H
  (0.539 vs 0.569) prediciendo, no cabalga un drift."
- **Back-up:** la aportación es el protocolo, no el P&L — la señal es de STRATA (ablación+SHAP) y
  equivale a lo que el XGBoost redescubre.

## OBJ 7 — "El sanity same-day/causal se invierte. ¿No es look-ahead?"
- **Concedo:** entiendo la sospecha — es el síntoma del bug peso_t×retorno_t del proyecto anterior.
- **Giro:** aquí va en sentido CONTRARIO al bug. El look-ahead INFLA el causal; en mis datos el causal
  está PENALIZADO (M5 causal −1.82 vs same-day +0.88). Si hubiera leak, el causal saldría mejor, no peor.
- **Implicación:** es propiedad del agente perdedor (correlaciona + con r_t, − con r_{t+1}); por eso
  usar r_{t+1} (signal_lag=1, lo que usted exigió) lo penaliza. El dato válido es el causal.
- **Back-up:** `test_no_leakage.py` en CI verifica que perturbar r_t no altera posiciones de t ni
  anteriores. Si hubiera look-ahead, fallaría.

---

## Cierre — "¿y entonces qué conclusión defiendes?"

> "Defiendo dos cosas separadas. Una: el modelo de régimen de STRATA generaliza a través de 24 años
> con crisis incluidas — eso es robusto. Dos: el rescate del agente solo funciona cuando el mercado
> sube, y lo sé porque puse una regla por escrito antes de mirar los datos para cazar justo esa
> condicionalidad, y saltó. No vendo un sistema que bate al mercado. Presento un protocolo de
> supervisión estadística que rescata a un agente perdedor en condiciones que el propio sistema
> delimita. Que sepa decir dónde no funciona es la aportación, no el defecto."

## NO citar mal (exigencias de rigor)
- NO afirmar que K=3 es óptimo de LL (K=4 lo supera; argumento = parsimonia + función).
- NO citar el p=0.0 del esquema disjoint: artefacto de Bartlett con ρ=−0.999 y n=3, no un resultado.
- NO afirmar que "el rescate se invierte en bajista" sin distinguir plano Sharpe de plano accuracy ni
  distinguir M8 de M10. La inversión es del **Sharpe** (ambos), no de la **accuracy** (M10 rescata
  accuracy en bajista; M8 no).
- NO vender el IC95 [−0.02, +5.79] de M10−M5 como "casi significativo": el criterio pre-registrado
  es la **cota Bonferroni** (cuantil 0.0125), que da −0.48 < 0 → H1_b False.
