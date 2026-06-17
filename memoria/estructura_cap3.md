# Estructura aprobada del Capítulo 3 (Marco teórico)

> **Fuente de verdad** que custodia el agente `arquitecto-estructura`. Cualquier redacción del cap. 3 debe
> seguir EXACTAMENTE estos 4 bloques y este orden. Las demostraciones ya escritas se **conservan** y se
> **re-secuencian**; no se reescriben. Estructura elegida por Raquel: **por disciplina**.

## Principio: 4 bloques

**§0 Preliminares y notación** (breve). Precio `P_t`, log-retorno `r_t^log` y aditividad, convención causal
(posición de `t` × `r_{t+1}`), volatilidad realizada `RV_t`, hechos estilizados, posición `w_t∈[−1,1]`, la
tupla de decisión del agente (acción, size, confianza).

**§1 STRATA a nivel técnico** (visión de conjunto — el "qué" antes del "cómo"). La función de supervisión
`f: tupla_agente × estado_mercado → tupla_supervisada`; los 3 detectores ortogonales (RAM/PSA/GSO) y sus ejes;
los 3 modos (warn/reduce/override). **Sin** la matemática todavía.

**§2 Teoría matemática de los detectores** (el grueso, con demostraciones):
- 2.1 **HMM gaussiano** (base de RAM): definición, recursión forward (prueba), filtrado vs suavizado (prueba +
  causalidad), Baum-Welch/EM, Viterbi, selección de K.
- 2.2 **GARCH(1,1)-t** (base de GSO): ARCH→GARCH, teorema de estacionariedad (prueba completa), Student-t, MV.
- 2.3 **BOCPD** (base de PSA): run-length, recursión de Adams–MacKay (prueba), hazard y predictiva conjugada.

**§3 STRATA aplicado: construcción, calibración y umbrales** (máximo rigor + justificación económica + citas):
- 3.1 Datos y calibración (2000→2024-09, una vez; estabilidad estructural — cita Bai–Perron).
- 3.2 **RAM**: del HMM al `RAM_score`. Leverage effect (Black 1976; Christie 1982). Prior **data-driven** por
  activo (signo de μ_k). Umbrales 0,25 / **τ=0,50** / 0,70; por qué τ=0,5.
- 3.3 **GSO**: del GARCH a la banda. Volatility targeting (Moreira–Muir 2017). Umbrales P95/P99. Limitación:
  GSO no dispara (hallazgo negativo).
- 3.4 **PSA**: del BOCPD al score. `cp_prob_delta`, hazard 1/60, umbrales P95/P99.
- 3.5 **Capa de intervención**: warn / reduce / override-C, gate τ, sustitución de `w_t`.

**§4 Validación: métricas, tests y ausencia de fuga** (teoría + demostraciones):
- 4.1 **Métricas**: Sharpe (+ fragilidad), Sortino, MaxDD, Calmar.
- 4.2 **Validación sin fuga**: `signal_lag=1`, purga y embargo, CPCV (López de Prado 2018).
- 4.3 **Contrastes**: sign test binomial, McNemar, Diebold–Mariano, TOST, bootstrap estacionario, DSR. Cada uno:
  H0, estadístico, distribución bajo H0, uso.

## Reglas de colocación (lo que vigila el guardián)
- La **economía** se reparte: *leverage effect* y *volatility targeting* → §3 (justifican RAM/GSO); Sharpe y
  métricas de riesgo → §4.
- Un **test** va en §4, nunca en §2. Un **umbral/calibración** va en §3, nunca en §2.
- La **notación** se introduce antes de usarse (§0).
- **Nada inventado:** toda afirmación no original lleva cita; toda cifra, fuente JSON.
