# STRATA — Constitución del proyecto

Trabajo Fin de Grado en Matemáticas y Ciencia de Datos, Universidad Complutense de Madrid. **Autora: Raquel García.** Tutor especializado en series temporales. Todo el código y la memoria deben parecer producidos por una estudiante de matemáticas, no por una IA.

Este fichero es la referencia única para el nuevo proyecto. **Lee este fichero entero antes de tocar nada.** Después: `CONOCIMIENTO_ACUMULADO.md` → `DECISIONES_ESENCIALES.md` → `LECCIONES_APRENDIDAS.md` → `RESULTADOS_OBJETIVO.md` → `AGENTES_SUGERIDOS.md`.

---

## 1. Qué es STRATA y qué NO es

**STRATA** (*Statistical Trading Real-time Audit*) es un sistema de **supervisión estadística** en tiempo de ejecución sobre las decisiones de un agente de trading basado en LLMs (AI Hedge Fund con 5 personalidades: Buffett, Wood, Druckenmiller, Burry, Ackman).

**Tres detectores ortogonales:**

| Detector | Eje | Pregunta | Modelo |
|---|---|---|---|
| **RAM** | Régimen discreto | ¿La acción es coherente con el régimen? | HMM gaussiano 3 estados |
| **PSA** | Coherencia temporal del agente | ¿Está cambiando de opinión estructuralmente? | BOCPD (Adams & MacKay 2007) |
| **GSO** | Volatilidad continua | ¿El sizing es compatible con la vol? | GARCH(1,1) Student-t |

**Tres modos de intervención:** `warn` (solo registra), `reduce` (atenúa size), `override` (sustituye).

**STRATA NO predice retornos.** Es una función `f: tupla_agente × estado_mercado → tupla_supervisada`. Su valor se mide por: (a) corrección del agente cuando éste pierde, (b) significancia estadística pareada de esa corrección, (c) interpretabilidad de la mecánica.

---

## 2. Hipótesis falsable del TFG

> *Filtrar/atenuar decisiones de un agente LLM con detectores estadísticos clásicos rescata significativamente al agente cuando éste pierde dinero y acierta direccionalmente menos del 50%.*

Falsable en tres niveles:

1. **Estadístico:** McNemar pareado M8 vs M5 con p < 0.10.
2. **Mecánico:** atribución de P&L a las intervenciones de cada detector (espera: RAM domina).
3. **Universalidad:** un meta-learner XGBoost-CPCV (M10) no debe batir significativamente a M8 (DM p > 0.10), y SHAP debe identificar las features STRATA como las informativas.

Cualquier resultado contrario invalida la hipótesis y debe reportarse honestamente. La regla `prior-flip` (signo de la calibración por régimen ≠ signo en OOS) es un **mecanismo de falsificación pre-registrado** que documenta cuándo NO funciona la técnica.

---

## 3. Universo experimental

- **Caso central del TFG:** SPY (ETF del S&P 500). Justificación: el **leverage effect** (Black 1976; Christie 1982) hace que en SPY el régimen de alta volatilidad coincida con bajista; el HMM funciona como proxy direccional. Esta asunción NO se cumple en stocks individuales con leverage débil — limitación documentada.
- **Panel de robustez:** 10 tickers en `cache/agent/` (SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR, SMCI, ROKU, MARA). Decision-level analysis sobre el panel = apéndice de la memoria.
- **Periodos:**
  - Calibración: **2000-01-01 → 2024-09-30** (24 años; HMM/GARCH/BOCPD entrenados una sola vez).
  - OOS unificado: **2024-10-01 → cierre del TFG**. Inicio posterior al cutoff de DeepSeek V3 (jul-oct 2024) para eliminar contaminación por look-ahead específico del LLM.

---

## 4. Filosofía de rigor

**El rigor matemático es la aportación.** La economía es enriquecimiento ilustrativo.

Toda cifra reportada debe ir acompañada de:

1. **Test estadístico apropiado.** No reportar Sharpe sin Diebold-Mariano + Deflated Sharpe Ratio (López de Prado). No reportar accuracy sin sign test contra 0.5 o McNemar pareado. No reportar IC sin método explícito (bootstrap estacionario, Politis-Romano 1994, bloque medio `sqrt(N)`).
2. **Verificación de causalidad temporal.** `signal_lag=1`: la posición del día *t* multiplica al retorno del día *t+1*. En CPCV: `t1 = índice.shift(-1)`, `embargo ≥ 5`, purge correctamente aplicado. Test `test_no_leakage.py` corre en CI.
3. **Pre-registro en BITACORA antes de mirar resultados.** Hipótesis nula explícita, estadístico de contraste, criterio de éxito numérico, criterio de fracaso (regla `prior-flip` o equivalente). Esto blinda contra acusaciones de p-hacking.
4. **Cita bibliográfica** en el docstring de la función que implementa el método. Ejemplo: `"""Combinatorial Purged Cross-Validation (López de Prado 2018, sec. 7.4)."""`.
5. **Doble protocolo de medición** cuando aplique: `same-day` (peso_t × retorno_t, sirve para sanity check) y `causal` (peso_t × retorno_{t+1}, el único válido para reportar).

**La economía** (€1000 → equity final, equity curve, Sharpe ilustrativo) **es enriquecimiento**, no prueba.

---

## 5. Workflow obligatorio para experimentos nuevos

Ningún experimento se ejecuta saltándose este workflow. Cada paso deja traza.

```
nueva pregunta de investigación
         ↓
1.  @asesor-historico        ← ¿se intentó algo parecido? cita BITACORA
         ↓
2.  @disenador-experimentos  ← pre-registra en BITACORA (hipótesis, test, criterio)
         ↓
3.  @rigor-matematico        ← audita diseño antes de ejecutar
         ↓
4.  @ejecutor-experimentos   ← corre y guarda outputs JSON
         ↓
5.  @rigor-matematico        ← audita resultados
         ↓
6.  @bitacora                ← decide si entra (decisión metodológica vs ruido)
         ↓
7.  @narrativa-coherencia    ← propaga al notebook + decisiones + memoria
         ↓
8.  @defensa-tutor           ← anticipa objeciones del tribunal
```

Esto es lo que en el proyecto anterior no se hizo y produjo 12 scripts de tuning huérfanos, 4 notebooks duplicados y una BITACORA con ruido. Ver `LECCIONES_APRENDIDAS.md` §8 y §10.

---

## 6. Estándares de código

- **Python 3.11+.**
- **Código en inglés. Comentarios y docstrings en español.**
- **Nombres matemáticos cortos** que mapean 1-a-1 a la memoria. `sigma_t`, `mu_k`, `alpha`, `beta`, no `volatility_at_time_t`.
- **Type hints** en toda función pública.
- **Docstring con cita** cuando hay referencia académica.
- **Determinismo riguroso.** Toda fuente de aleatoriedad lleva semilla fijada en `config.py`.
- **Sin APIs de pago.** Solo OpenRouter free (DeepSeek/gpt-oss). Cero euros gastados en inferencia.
- **Sin sobreabstracción.** Sin factories, builders, services. El código es de investigación.
- **Sin comentarios redundantes.** Comenta el *porqué*, no el *qué*.
- **Sin manejo defensivo excesivo.** Asume contratos razonables internos; valida solo en boundaries.

Cómo NO debe verse el código (patrones IA típicos a evitar):

- Docstrings idénticas con bloques `Args:`/`Returns:`/`Raises:` para todas las funciones.
- `if x is not None and isinstance(x, ...) and len(x) > 0` defensivo en código interno.
- Nombres tipo `calculate_garch_volatility_value_at_time_t` cuando `garch_vol(t)` basta.
- Logging excesivo (`logger.info("Entering function X")` en cada función).
- Try/except envolviendo cosas que no pueden fallar.

---

## 7. BITACORA — cuaderno de campo

Fichero versionado, parte del entregable, defendible ante el tribunal. Documenta el **proceso** de investigación, no solo el resultado.

**Cuándo actualizarla** (filtrado por `@bitacora` antes de escribir):

- Al cerrar un milestone (entrada principal).
- Al tomar una decisión metodológica relevante (umbrales, calibraciones, exclusiones, cambios de modelo).
- Al resolver un error que generó tiempo perdido o cambio de plan.
- Al descubrir algo del fenómeno estudiado que conviene reflejar en la memoria.

**Cuándo NO actualizarla:**

- Después de cada commit pequeño.
- Para anotar progreso trivial.
- Para mensajes tipo "he completado X y voy a empezar Y".

**Formato de cada entrada:**

```markdown
## [YYYY-MM-DD] [Milestone | Decisión | Error | Hallazgo] - Título

**Contexto.** Una o dos frases.

**Detalle.** Lo ocurrido. Suficiente para entenderlo en tres semanas sin reconstruir contexto.

**Implicaciones para el TFG.** Si aplica.

**Referencias.** Commits, ficheros, papers.
```

**Pre-registro de experimento** (obligatorio antes de ejecutar):

```markdown
## [YYYY-MM-DD] [Pre-registro] - Experimento <nombre>

**Hipótesis.** Falsable, explícita.
**H0.** Hipótesis nula.
**Estadístico.** Qué test (DM, McNemar, sign test, bootstrap).
**Criterio de éxito.** p<α=?, IC contiene/no contiene cero, Δ métrica > umbral concreto.
**Criterio de fracaso.** Regla prior-flip o equivalente. Qué resultado refuta la hipótesis.
**Datos.** OOS exacto. Embargo. Splits.
**Output esperado.** `outputs/experiments/<nombre>.json` con qué claves.
```

---

## 8. Caché y reproducibilidad

Cuatro caches independientes:

| Caché | Tamaño | En git | Cuándo invalidar |
|---|---:|---|---|
| `cache/agent/` | 16M | **SÍ** | Cambio en wrapper, en patches macro/price/stats, en personalidades habilitadas |
| `cache/models/` | 44K | **SÍ** | Cambio en periodo de calibración o en hiperparámetros HMM/GARCH/BOCPD |
| `cache/llm/` | 156M | NO | Cambio en prompt template, modelo o temperatura |
| `data/` | 170M | NO | Cambio de proveedor de datos |

**Protocolo de invalidación:** cuando un cambio invalida un caché, el código lanza error explícito en lugar de regenerar silenciosamente. Forzar regeneración: borrar la carpeta y volver a correr.

**Cuando un activo nuevo necesita decisiones del agente** (extender el panel a un nuevo ticker): correr `live/daily_run.py` o el experimento M5 con `--ticker NEW --end-date YYYY-MM-DD`. El runner reusa cache donde existe y solo paga inferencias para fechas nuevas.

---

## 9. Lo que NO se hace

- **KFold convencional en series temporales.** CPCV o Walk-Forward. Única excepción documentada: M3 deliberadamente KFold para denunciar el sesgo. Cualquier otro uso es un error y `@rigor-matematico` lo detecta.
- **APIs de pago.** OpenAI, Anthropic directos, Google directo. Solo OpenRouter free.
- **Look-ahead de cualquier tipo.** `peso_t × retorno_t` está prohibido (es el bug que infectó M8 durante semanas en el proyecto anterior). Siempre `peso_t × retorno_{t+1}` con `signal_lag=1`.
- **Mover o regenerar caché silenciosamente.** Error explícito + intervención humana.
- **Commits gigantes** que mezclan funcionalidad + refactor + estilo. Atómicos.
- **Saltarse el pre-registro.** Cualquier experimento sin pre-registro en BITACORA no entra a la memoria.
- **BITACORA con ruido** ("completé X y voy a Y"). Filtro `@bitacora`.
- **Notebooks duplicados.** Un solo notebook canónico. Las variantes viven en branches, no en `notebooks/`.
- **Tuning huérfano.** Carpeta `experiments/tuning/` prohibida. Toda exploración tiene BITACORA o se elimina.
- **Reglas hardcoded sin calibración.** Prior RAM data-driven por activo, nunca "Crisis ⇒ short" hardcoded.

---

## 10. Punto de entrada al nuevo proyecto

> *"Empieza preguntando a `@asesor-historico` qué decisión es relevante para tu tarea."*

El asesor lee de `_archivo_proyecto_anterior/` (BITACORA + decisiones + chats + transcripciones del tutor) y te dice qué se intentó antes, por qué se descartó y a qué cita atender. Ese es el atajo para no repetir los errores del proyecto anterior.
