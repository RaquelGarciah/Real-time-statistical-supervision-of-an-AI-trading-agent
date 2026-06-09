# STRATA — Supervisión estadística en tiempo real de agentes de trading con IA

[![CI](https://github.com/RaquelGarciah/strata-tfg/actions/workflows/ci.yml/badge.svg)](https://github.com/RaquelGarciah/strata-tfg/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Estado](https://img.shields.io/badge/estado-investigaci%C3%B3n%20activa-success.svg)

> **STRATA** (*Statistical Trading Real-time Audit*) es una capa de **supervisión estadística** que
> audita y corrige, decisión a decisión, lo que hace un agente de trading basado en LLMs. Convierte
> un agente caja-negra que **pierde dinero** en un sistema **disciplinado, interpretable y
> validado estadísticamente** — sin convertirse en otra caja negra.

Trabajo de Fin de Grado · Doble Grado en **Matemáticas y Ciencia de Datos**, Universidad Complutense
de Madrid · Autora: **Raquel García**.

---

## El problema

Los agentes de trading basados en LLMs se venden como la nueva frontera de la inversión
automática, pero son **cajas negras poco fiables**. El agente que estudiamos —*AI Hedge Fund*, un
sistema open-source con cinco personalidades inversoras (Buffett, Wood, Druckenmiller, Burry,
Ackman)— sobre el ETF **SPY** y en datos fuera de muestra (oct 2024 – jun 2026, 401 sesiones):

- **Pierde dinero**: €1 000 → **€903**.
- **Acierta la dirección del mercado menos del 50 %**: 38,4 % de los días (*sign test* p < 0,001).
  Peor que una moneda.

La pregunta del proyecto: **¿cómo se hace utilizable un agente LLM que, por sí solo, no es viable —
sin sustituir una caja negra por otra?**

---

## La solución

STRATA **no predice el mercado**. Es una **función determinista** que se interpone entre el agente
y el mercado y solo usa información disponible *hoy*:

```
f : (decisión del agente,  estado del mercado hoy)  ⟶  posición supervisada  w ∈ [−1, +1]
```

Tres **detectores estadísticos clásicos y ortogonales** auditan cada decisión diaria:

| Detector | Eje que vigila | Modelo subyacente | Pregunta |
|---|---|---|---|
| **RAM** | Régimen de mercado | HMM gaussiano de 3 estados | ¿La dirección del agente es coherente con el régimen (calma/estrés/crisis)? |
| **PSA** | Coherencia del agente | BOCPD (Adams & MacKay, 2007) | ¿El agente acaba de cambiar de opinión de forma anómala? |
| **GSO** | Volatilidad del mercado | GARCH(1,1)-t | ¿El tamaño de la apuesta es compatible con la volatilidad? |

Cuando el régimen contradice con confianza al agente, STRATA **voltea** la posición hacia la
dirección que sugiere el régimen, dimensionada por volatilidad. Tres modos de intervención
(`warn` / `reduce` / `override`) cubren desde el registro pasivo hasta la corrección activa.

---

## Resultados clave

Sobre SPY, fuera de muestra (oct 2024 – jun 2026, 401 sesiones), evaluación **causal estricta**
(la posición de hoy gana el retorno de mañana, `signal_lag = 1`):

| Estrategia | Acierto direccional | Sharpe | €1 000 → |
|---|:--:|:--:|:--:|
| Agente LLM solo (sin supervisar) | 38,4 % | −1,82 | €903 |
| **STRATA (supervisión estadística)** | **43,6 %** | **+0,67** | **€1 069** |
| Meta-learner XGBoost (referencia ML) | 53,9 % | +0,64 | €1 035 |
| Buy & Hold (mercado pasivo) | 56,9 % | +1,09 | €1 317 |

**Tres conclusiones, todas con su test estadístico:**

1. **STRATA rescata al agente.** El acierto direccional sube de 38,4 % a 43,6 % y la cuenta pasa de
   perder a recuperar. La mejora es **significativa en el contraste pareado** que de verdad importa:
   *McNemar* STRATA vs agente, **p ≈ 0,07** (de 121 días en que difieren, STRATA arregla 71 y
   estropea 50). Honestamente: significativo a α = 0,10, *borderline* a α = 0,05.

2. **Un meta-learner con "todo dentro" no lo bate.** Un XGBoost validado con *Combinatorial Purged
   CV* sobre 22 features (las 5 personalidades + los 3 detectores + 4 de régimen) **iguala** a la
   regla a mano pero no la supera (*Diebold-Mariano* p = 0,61). Y el **SHAP** revela que las features
   informativas son justo las de STRATA y el régimen —no las del agente—: **el ML redescubre la
   regla, no la mejora.**

3. **Honestidad científica.** Ningún sistema bate al mercado pasivo (Buy & Hold, €1 317). La
   aportación **no es "ganar al mercado"**: es **rescatar a un agente perdedor con un protocolo
   estadístico defendible**.

---

## La aportación — por qué este proyecto vale

- **Rigor por encima de la curva bonita.** Ninguna cifra se reporta sin su test: contrastes
  pareados (*McNemar*, *Diebold-Mariano*), *Deflated Sharpe Ratio*, *bootstrap* estacionario,
  validación CPCV **sin fuga temporal**, y **pre-registro** de cada experimento (hipótesis y
  criterio fijados *antes* de mirar resultados) como blindaje anti *p-hacking*.

- **Interpretabilidad frente a caja negra.** STRATA está hecho de estadística clásica (HMM, GARCH,
  BOCPD) que se puede **explicar y defender** ante un tribunal, no de un modelo opaco. Cada
  intervención es trazable paso a paso.

- **Disciplina estadística > complejidad de ML.** El resultado central —que una regla a mano bien
  fundamentada es *estadísticamente indistinguible* de un XGBoost universal— es una lección
  contraintuitiva y valiosa: en un problema con señal débil y muestra pequeña, **la complejidad no
  compra ventaja; la disciplina sí.**

- **Ciencia falsable.** El proyecto documenta explícitamente **cuándo NO funciona** (la regla
  *prior-flip*: si el signo calibrado del régimen difiere del signo fuera de muestra, se reporta como
  fallo). Reportar los límites es parte del resultado.

---

## Cómo funciona — un día concreto

El motor de backtest es contabilidad pura: `P&L = posición · retorno_mañana`. Lo único que cambia
entre estrategias es **cómo se calcula la posición**. Ejemplo de día con intervención:

| Paso | Cálculo | Resultado |
|---|---|---|
| 1. El agente decide | long, *size* = +0,30 | tupla del agente |
| 2. HMM + GARCH | régimen = **Crisis** (P = 0,80), σ = 23 % | estado del mercado |
| 3. RAM detecta incoherencia | long en Crisis ⇒ score 0,80 (*high*) | dispara |
| 4. *override* hacia el régimen | signo_régimen · banda_vol = −1 · 0,43 | **posición = −0,43** |

El agente quería comprar en plena crisis; STRATA lo reorienta a corto. Sobre 401 días, ese tipo de
corrección es lo que convierte la pérdida en recuperación.

---

## Stack técnico

**Modelos:** HMM gaussiano (regímenes) · GARCH(1,1)-Student-t (volatilidad) · BOCPD (puntos de
cambio) · XGBoost + SHAP (meta-learner de referencia).
**Inferencia:** McNemar · Diebold-Mariano · *sign test* · Deflated Sharpe · *bootstrap* estacionario
(Politis-Romano) · Combinatorial Purged CV (López de Prado).
**Ingeniería:** Python 3.11, `numpy` / `pandas` / `scipy` / `scikit-learn` / `hmmlearn` / `arch` /
`xgboost`; tests con `pytest`; CI en GitHub Actions; determinismo con semilla fijada.

---

## Estructura del repositorio

```
core/         Primitivas matemáticas testeadas (HMM, GARCH, BOCPD, CPCV, métricas, contrastes)
strata/       Los tres detectores (RAM/PSA/GSO) + la capa de intervención
experiments/  Experimentos reproducibles, cada uno con su pre-registro y su salida JSON
notebooks/    Cuaderno canónico del TFG (strata_canonical) + cuaderno de experimentos
tests/        Suite de tests (incluye verificación de ausencia de look-ahead)
cache/        Modelos calibrados (HMM/GARCH/thresholds) y decisiones del agente por activo
BITACORA.md   Cuaderno de campo: decisiones metodológicas, hallazgos y pre-registros
```

---

## Reproducibilidad

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # suite de tests (incl. no-leakage)
jupyter notebook notebooks/strata_canonical.ipynb   # análisis completo del TFG
```

Calibración de modelos: 2000–2024 (una sola vez). Evaluación fuera de muestra: oct 2024 en adelante,
con inicio posterior al *cutoff* del LLM para eliminar contaminación por *look-ahead*.

---

## Alcance y limitaciones (declaradas)

- **Caso central: SPY.** Funciona porque en índices agregados el *leverage effect* (Black 1976;
  Christie 1982) hace que la alta volatilidad coincida con caídas, y el régimen sirve de *proxy*
  direccional. La asunción se debilita en acciones individuales — limitación documentada, con un
  panel de robustez de 10 activos como apéndice.
- **Una única ventana fuera de muestra** (alcista). La validación multi-ventana / *walk-forward* es
  trabajo en curso.
- **No bate al mercado pasivo.** El objetivo es supervisar al agente, no superar a Buy & Hold.

---

*STRATA — Statistical Trading Real-time Audit. Raquel García, Universidad Complutense de Madrid.*
