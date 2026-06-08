# Consejo Asesor — el segundo anillo de agentes de STRATA

Los 9 agentes de `AGENTES_SUGERIDOS.md` son la **capa de proceso**: gatekeepers que auditan, ejecutan y mantienen el orden (pass/fail). El **Consejo Asesor** es la **capa de criterio**: un equipo de expertos de dominio que aportan visión, exploran posibilidades y **debaten entre sí** la sustancia matemática y económica, al nivel de sofisticación del tribunal.

Filosofía: el tutor no pregunta solo *si* las reglas se aplican bien (eso lo cubre `@rigor-matematico`), sino *si son las reglas correctas*, *si los supuestos se cumplen* y *si las conclusiones generalizan*. Para eso hace falta un consejo, no un control de calidad.

---

## Los 10 miembros

| Agente | Eje | Modelo | Qué aporta |
|---|---|---|---|
| `@experto-series-temporales` | Matemática | opus | HMM, GARCH, BOCPD, Bai-Perron. La especialidad del tutor. |
| `@experto-inferencia` | Matemática | opus | Tests, IC, DSR y sus supuestos finos. Enseña, no bloquea. |
| `@experto-ml-financiero` | Matemática | opus | CPCV, XGBoost, SHAP. La objeción central del tutor. |
| `@experto-finanzas-cuantitativas` | Economía | opus | Leverage effect, régimen↔dirección, eficiencia, prior RAM. |
| `@experto-gestion-riesgo` | Economía | sonnet | Vol targeting, banda GARCH, drawdown, sizing, Kelly. |
| `@coordinador-consejo` | Coordinación | opus | Reconcilia dictámenes, marca discrepancias, dispara 2ª ronda. |
| `@abogado-del-diablo` | Visión extra | opus | Red-team proactivo: objeciones nuevas, ataca supuestos. |
| `@revisor-bibliografico` | Visión extra | sonnet | Estado del arte, literatura ausente, novedad real. |
| `@redactor-academico` | Visión extra | opus | Prosa matemática que parezca de estudiante, no de IA. |
| `@inspector-datos-sesgos` | Visión extra | sonnet | Look-ahead, calibración del LLM, survivorship, huecos. |

---

## Restricción técnica (importante)

En Claude Code **un subagente no puede invocar a otro subagente**. Por tanto "que se comuniquen entre ellos" lo orquesta **el hilo principal** (Claude en la conversación): es quien convoca a los expertos, recoge sus dictámenes y se los pasa al coordinador. La "mesa redonda" es real, pero el cartero es el hilo principal, no los propios agentes.

---

## Protocolo: "convoca al consejo sobre X"

Cuando Raquel dice *"convoca al consejo sobre <pregunta>"*, el hilo principal ejecuta:

1. **Selección.** Elige qué expertos son relevantes a la pregunta (ver tabla de abajo). No se convoca a los 10 siempre.
2. **Ronda 1 (en paralelo).** Invoca a cada experto elegido. Cada uno responde en el **formato de dictamen fijo**:
   ```
   POSTURA: <1-2 líneas>
   FUNDAMENTO: <con cita: paper o fichero core/strata:línea>
   RIESGOS / SUPUESTOS QUE PODRÍAN ROMPERSE:
   POSIBILIDADES ALTERNATIVAS:
   GRADO DE CONFIANZA: alto | medio | bajo
   ```
3. **Síntesis.** Pasa todos los dictámenes a `@coordinador-consejo`, que devuelve consenso + tabla de discrepancias + qué exige 2ª ronda.
4. **Ronda 2 (solo si hay conflicto).** El hilo principal re-consulta SOLO a los expertos en desacuerdo con la pregunta afilada que formuló el coordinador. El coordinador cierra con la recomendación final y los votos disidentes.
5. **Salida.** Una recomendación única y defendible, con la minoría documentada (no enterrada).

---

## Qué experto para qué pregunta

| Si la pregunta es sobre… | Convoca a… |
|---|---|
| HMM, regímenes, GARCH, BOCPD, etiquetado | `@experto-series-temporales` |
| qué test usar, IC, p-valor borderline, DSR | `@experto-inferencia` |
| CPCV, sobreajuste, XGBoost vs STRATA, SHAP | `@experto-ml-financiero` |
| leverage effect, dirección, eficiencia, prior | `@experto-finanzas-cuantitativas` |
| tamaño de posición, vol targeting, drawdown | `@experto-gestion-riesgo` |
| "¿qué nos estamos perdiendo / qué falla?" | `@abogado-del-diablo` |
| citas, novedad, estado del arte | `@revisor-bibliografico` |
| redacción de la memoria, notación, demostraciones | `@redactor-academico` |
| ¿los datos mienten? sesgo, look-ahead, huecos | `@inspector-datos-sesgos` |
| reconciliar varias opiniones | `@coordinador-consejo` |

Una pregunta típica de método mezcla ejes: p. ej. *"¿reportamos McNemar p=0.088 como evidencia?"* convoca a `@experto-inferencia` + `@abogado-del-diablo` (+ `@experto-finanzas-cuantitativas` si toca el relato).

---

## Cómo encaja con el workflow de proceso (los 9)

El Consejo asesora **antes** de que `@disenador-experimentos` escriba el pre-registro, y **en paralelo** a `@rigor-matematico`:

```
nueva pregunta de investigación
        ↓
  @asesor-historico        (¿se intentó antes?)
        ↓
  CONSEJO ASESOR           (¿es buena idea? ¿qué riesgos/posibilidades?)  ← criterio
        ↓
  @disenador-experimentos  (pre-registra)
        ↓
  @rigor-matematico        (audita pass/fail)                             ← proceso
        ↓
  @ejecutor-experimentos → @rigor-matematico → @bitacora → @narrativa-coherencia → @defensa-tutor
```

El Consejo da **criterio** (qué hacer y por qué); la capa de proceso da **garantías** (que se hace bien). No se solapan: cada miembro del Consejo declara en su prompt a su contraparte de proceso para no pisarla.

---

## No-solapamiento (resumen)

| Capa de proceso | ↔ | Consejo asesor |
|---|---|---|
| `@rigor-matematico` (bloquea) | ↔ | `@experto-inferencia` (enseña/propone) |
| `@defensa-tutor` (reactivo) | ↔ | `@abogado-del-diablo` (proactivo) |
| `@cache-doctor` (integridad) | ↔ | `@inspector-datos-sesgos` (sesgo estadístico) |
| `@narrativa-coherencia` (cifras) | ↔ | `@redactor-academico` (prosa matemática) |
| `@panel-multiactivo` (ejecuta panel) | ↔ | `@experto-finanzas-cuantitativas` (relato económico) |
