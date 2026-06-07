# Entregables — qué va al tribunal

Lista exhaustiva, basada en la transcripción del tutor del 2026-06-07, los chats de defensa y las decisiones consolidadas. Marcada cada pieza como **Memoria**, **Notebook**, **Apéndice**, **Defensa oral** según su destino.

---

## A. Memoria del TFG (documento LaTeX)

Fuera del kit (vive en otro repo), pero referencia los outputs que produce el código del nuevo proyecto.

**Estructura mínima exigida por el tutor:**

1. **Introducción y motivación** — el problema de los agentes LLM en trading. Por qué la supervisión estadística importa.
2. **Marco teórico** — leverage effect (Black 1976, Christie 1982); HMM gaussiano; GARCH(1,1) Student-t; BOCPD (Adams & MacKay 2007); CPCV (López de Prado 2018); SHAP (Lundberg et al. 2020); McNemar, Diebold-Mariano, sign test, bootstrap estacionario (Politis-Romano 1994).
3. **Metodología** — descripción de los 3 detectores STRATA; los 3 modos de intervención; calibración 2000–2024-09; OOS 2024-10 → cierre; protocolo de validación causal.
4. **Diseño experimental** — pre-registro de M5 / M8 / M10 con hipótesis nula y criterios de éxito.
5. **Resultados** — tabla maestra + análisis condicional + SHAP + panel multi-activo (apéndice).
6. **Discusión** — qué se ha demostrado, qué no; límites de generalización; cuando STRATA NO funciona (prior-flip, SMCI).
7. **Conclusión** — la narrativa final (ver `CONOCIMIENTO_ACUMULADO.md` última sección).
8. **Bibliografía** — citas mínimas en el código deben aparecer aquí.
9. **Apéndices** — panel multi-activo, NVDA per-asset, limitación GSO.

---

## B. Notebook canónico ejecutado limpio

Un solo `.ipynb` en `notebooks/strata_canonical.ipynb`. Math-first. SPY-only en cuerpo, panel multi-activo en apéndice. Estructura:

| § | Contenido |
|---|---|
| §0 | Preámbulo: imports, seed, paths |
| §1 | Marco matemático con LaTeX explícito de HMM, GARCH, BOCPD, CPCV, RAM/PSA/GSO, override C, XGBoost, SHAP, métricas |
| §2 | Datos: descarga, alineación de calendarios, validación de barrera temporal |
| §3 | Calibración explícita: parámetros HMM/GARCH/BOCPD + tablas + check α+β<1 + prior RAM por régimen |
| §4 | Detectores OOS día a día (RAM, PSA, GSO con sus scores y severidades) |
| §5 | Ground truth y baselines triviales (always long, always short, coin flip) |
| §6 | M5 vs ground truth: accuracy + sign test |
| §7 | M8 vs ground truth: accuracy + McNemar pareado M8 vs M5 |
| §8 | M10 vs ground truth: CPCV explícito + SHAP + gain de XGBoost |
| §9 | Comparativa cara a cara: análisis condicional por régimen, severidad RAM, quintil de \|r\| + Diebold-Mariano |
| §10 | Tangibles económicos: equity, €1000, drawdown, Sharpe |
| §11 | Hipótesis cumplidas: lectura para defensa |
| §12 | Reproducibilidad: hashes, versiones, fechas |

---

## C. Tablas exigidas explícitamente por el tutor (van a Memoria + Defensa oral)

### C.1. Tabla maestra de métricas matemáticas

| Estrategia | Accuracy | AUC | Log-loss | Brier | MCC | Sharpe | €1000→ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline "always long" | objetivo ≈ 0.566 | — | — | — | — | ≈ B&H | ≈ 1323 |
| M5 (agente solo) | objetivo ≈ 0.407 | 0.481 | 0.756 | 0.281 | −0.106 | −1.83 | 903 |
| M8 (STRATA override C) | objetivo ≈ 0.460 | 0.471 | 1.640 | 0.312 | −0.090 | +0.66 | 1064 |
| M10 (XGBoost CPCV) | objetivo ≈ 0.530 | 0.504 | 0.785 | 0.284 | +0.022 | +0.69 | 1035 |

Cifras de Sharpe y €1000 = de `_archivo_proyecto_anterior/outputs_canonicos/m{5,8,10}*.json` (canónico al 2026-06-07).
Cifras de accuracy/AUC/log-loss/Brier/MCC = del notebook canónico (calcular en nuevo proyecto).

### C.2. Tests estadísticos pareados

| Test | Comparación | Estadístico | p-valor objetivo | Lectura |
|---|---|---|---|---|
| McNemar pareado | M8 vs M5 | χ² | ≈ 0.088 | STRATA rescata al agente con significancia borderline |
| Diebold-Mariano | Sharpe(M10) − Sharpe(M8) | DM | ≈ 0.75 | M10 y M8 indistinguibles |
| Bootstrap estacionario (Politis-Romano) | Δ Sharpe(M10−M8) | IC95% | ≈ [−1.80, +1.15] | El intervalo contiene cero |
| Sign test direccional | M5 vs 0.5 | binomial | < 0.001 | El agente solo es peor que el azar |
| Bootstrap | P(M10 > M8) | bootstrap | ≈ 0.543 | Moneda |

### C.3. SHAP global de M10

Top 5 features del XGBoost meta-learner ordenadas por |SHAP| medio:

| # | Feature | SHAP medio objetivo |
|---|---|---:|
| 1 | `ram_score` | ≈ 0.527 |
| 2 | `psa_score` | ≈ 0.428 |
| 3 | `garch_sigma` | ≈ 0.346 |
| 4 | `stress_prob` | ≈ 0.342 |
| 5 | `calm_prob` | ≈ 0.324 |

**Las 3 features STRATA + 2 de régimen ocupan el top 5. Ninguna personalidad del agente llega.** Este es el golpe a la objeción del tutor.

### C.4. Análisis condicional

| Subset | N | Métrica | Estrategia ganadora |
|---|---:|---|---|
| Régimen Crisis | ~85 | accuracy | M10 ≈ 60.7% |
| Régimen Calma | ~250 | accuracy | M8 ≈ 57.0% |
| RAM-flag high | 107 | accuracy M8 − M5 | **+17.8 pp** |

### C.5. Estabilidad temporal del umbral XGBoost

| Umbral `p1` | Sharpe mitad-1 (train) | Sharpe mitad-2 (test) |
|---:|---:|---:|
| 0.42 | +0.20 | **+1.07** |
| 0.50 | +0.41 | +0.09 |
| **0.565 (óptimo train)** | **+0.76** | +0.14 |
| 0.60 | +0.40 | −0.04 |

**Conclusión:** el umbral aprendido por XGBoost no es estable. Los umbrales calibrados de STRATA sí lo son por construcción. Este es el argumento de defendibilidad por interpretabilidad.

### C.6. Atribución de P&L por detector (panel multi-activo, apéndice)

| Detector | % P&L atribuible | Activos donde domina |
|---|---:|---|
| RAM | **98%** (+9218 bps) | SPY, NVDA, BAC, TSLA, XLE, UNG, MSTR |
| PSA | 2% (+185 bps) | Marginal |
| GSO | 0% | No dispara medium+ en NINGÚN activo |

**RAM es el driver del aporte. Reorganizar la presentación de STRATA poniendo RAM en el centro.**

---

## D. Walkthrough día-a-día (Memoria + Defensa oral)

Tabla paso a paso para una fecha concreta (sugerencia: 12-mar-2025, día donde STRATA y XGBoost discrepan visiblemente). Va al notebook §9 o §10.

| Paso | Cálculo | M5 | M8 | M10 |
|---|---|---:|---:|---:|
| Agente Portfolio Manager | tupla `(action, size, conf)` | `short, −0.197, 0.85` | misma | misma |
| HMM | `P(Calma/Estrés/Crisis)` | — | `(0%, 99.9%, 0%)` | `(0%, 99.9%, 0%)` |
| GARCH | `σ_t` | — | `23.3%` | `23.3%` |
| Detectores STRATA | `(RAM, PSA, GSO)` | — | scores | scores |
| Decisión final | posición | `−0.197` | `override C` aplicado | `2·0.8073−1 = +0.615` |
| Sizing risk parity | — | — | — | `0.10/0.233 = 0.43` |
| Factor régimen | — | — | — | `×0.5` |
| **Posición final** | — | `−0.197` | `?` | `+0.132` |
| Retorno SPY mañana | — | `−1.34%` | `−1.34%` | `−1.34%` |
| **P&L** | — | `+0.264%` | `?` | `−0.18%` |

Lecciones a destacar en la defensa oral:
1. Ese día el agente acertó (short) y XGBoost falló (long).
2. STRATA queda en medio: depende de qué detector dispare con qué severidad.
3. **Por eso se mide sobre 400 días, no sobre uno.** Un día no demuestra nada.

---

## E. Apéndices de la memoria

### E.1. Panel multi-activo (decision-level analysis)

- Tabla `hit_rate.csv` con accuracy M5/M8 por activo + McNemar p-valor.
- Tabla `pnl_intervention.csv` con P&L atribuible + IC95% bootstrap estacionario + sign test.
- Tabla `attribution_proportional.csv` y `attribution_exclusive.csv` con % atribución por detector.

Datos en `_archivo_proyecto_anterior/outputs_canonicos/decision_level/` como referencia.

### E.2. NVDA per-asset HMM + prior RAM re-signado

Caso documentado donde el prior RAM data-driven cambia de signo respecto a SPY. Justifica la decisión #6.

### E.3. Limitación GSO

Nota a pie + media página: GSO no dispara con severidad medium+ en ningún activo del panel OOS. Calibrado demasiado laxo o el sizing del agente cae sistemáticamente dentro de la banda admisible. **Reportable como hallazgo metodológico negativo, no como fallo de implementación.**

### E.4. Casos donde STRATA NO funciona

- **MSTR** — `prior-flip` clásico (medias calibradas vs OOS con signo opuesto).
- **SMCI** — agente con información direccional complementaria al prior.

Ambos refuerzan el rigor: el TFG documenta cuándo NO funciona la técnica.

---

## F. Defensa oral — preparada con `@defensa-tutor`

8 objeciones anticipables que el tutor (o el tribunal) puede plantear, con bullet points listos para responder. Generables con `@defensa-tutor` leyendo `_archivo_proyecto_anterior/docs/chats/need_mathematic_rigor.md` y la transcripción 2026-06-07.

Objeciones anticipadas (esqueleto):

1. *"Un XGBoost con todo dentro debería batir tu regla a mano."* → M10 + SHAP zanjan empíricamente.
2. *"¿Por qué SPY y no un stock individual?"* → Leverage effect + asunción RAM.
3. *"Tu STRATA tiene 6 hiperparámetros calibrados, el XGBoost también tiene hiperparámetros — no es una ventaja real."* → Estabilidad temporal del umbral.
4. *"40% de accuracy es peor que tirar una moneda, ¿cómo defiendes eso?"* → Confusión Sharpe/accuracy + asimetría de magnitudes.
5. *"Tu OOS son 400 días, no es suficiente."* → Pre-registro + cutoff DeepSeek + IC bootstrap.
6. *"GSO no aporta en el panel — ¿por qué lo mantienes?"* → Hallazgo metodológico negativo, no fallo.
7. *"SMCI rompe tu hipótesis."* → Sí, y lo documentamos como límite. La hipótesis es probabilística, no determinista.
8. *"Estás haciendo data-snooping al haber visto los resultados y ajustado el diseño."* → Pre-registro en BITACORA con timestamps anteriores a la ejecución.

---

## G. Reproducibilidad

Una sección al final del notebook + sección de la memoria:

- Versión de Python, librerías clave (numpy, pandas, hmmlearn, arch, xgboost) con hashes de `requirements.txt`.
- Hash de `cache/agent/` y `cache/models/`.
- Semillas (config.py).
- Fechas de calibración y OOS (literal).
- Comando exacto para reproducir cada figura.
- Acreditar que se cumple: cualquier máquina con el kit + `pip install -r requirements.txt` + `python -m experiments.m5 + m8 + m10` debe producir los mismos JSON byte-a-byte.

---

## Checklist final antes de entregar

- ☐ Notebook canónico ejecutado limpio, todas las celdas verdes
- ☐ Memoria LaTeX referencia outputs/figures del nuevo proyecto
- ☐ Tabla maestra con cifras desde JSON canónicos (no copiadas de chats)
- ☐ Tests pareados con p-valores reportados con tres decimales
- ☐ SHAP top-5 con justificación de por qué es "el golpe"
- ☐ Walkthrough día-a-día listo
- ☐ Apéndices con casos de fallo documentados
- ☐ Defensa oral preparada con `@defensa-tutor` para las 8 objeciones
- ☐ BITACORA versionada con todos los pre-registros
- ☐ Tests pytest verdes
- ☐ README.md del proyecto explica cómo reproducir todo
