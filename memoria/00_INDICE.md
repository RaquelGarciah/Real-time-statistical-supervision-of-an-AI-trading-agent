# memoria/ — carpeta de trabajo de la memoria

Todo lo relacionado con escribir la memoria, en un sitio. **Empieza por `MANUAL.md`** (la estrella polar).
Esta carpeta **no duplica** los documentos canónicos: los **indexa**.

## Qué hay aquí
| Fichero | Para qué |
|---|---|
| **`MANUAL.md`** | Lo esencial: qué demostramos, objetivos, claim canónico, estructura, tabla mínima, dónde está cada cosa. **Léelo en cada sesión.** |
| `estructura_cap3.md` | Estructura aprobada del cap. 3 (4 bloques). Verdad del agente `arquitecto-estructura`. |
| `ESTILO_Y_ANTIIA.md` | Reglas de estilo de Raquel + lista de lo prohibido (anti-IA, anti-plagio). |
| `objeciones_tribunal.md` | Las objeciones del tutor/tribunal con su respuesta honesta, consolidadas. |

## Fuentes canónicas (fuera de esta carpeta, no duplicar)
- **Decisiones:** `DECISIONES_ESENCIALES.md` (16, #13–16 = pivot SMCI).
- **Cifras:** `RESULTADOS_OBJETIVO.md` (§1 SPY método · §1bis SMCI).
- **Recorrido SMCI:** `docs/chats/decision_activo/smci.md`. **SPY mecanismo:** `…/spy_understandStrata.md`.
- **Citas/literatura:** `decisiones_respaldadas_literatura.md`, `tesis/bibliography.bib`,
  `tesis_assets/fuentes_bibliograficas/papers.md`.
- **Negativos (lo que probé y descarté):** `falsacion/INDICE.md`.
- **Figuras:** `graficas_clave.md`, `tesis/figures/`.
- **LaTeX (lo que se compila):** `tesis/` (capítulos, `bibliography.bib`, `tables/`, `figures/`).
- **Constitución y método:** `CLAUDE.md`, `LECCIONES_APRENDIDAS.md`.

## Flujo de redacción (pipeline)
`asesor-historico` → `arquitecto-estructura` (outline contra `estructura_cap3.md`) → `redactor-tesis` →
`rigor-matematico`/`harvard-professor` → `experto-citas`/`revisor-bibliografico` → `estilo-raquel` →
`detector-ia`/`detector-plagio` → `narrativa-coherencia` → `latex-experto`. Orquesta el hilo principal; ningún
subagente llama a otro.
