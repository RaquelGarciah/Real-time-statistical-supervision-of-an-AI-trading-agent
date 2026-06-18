# Consejero-revisor de la tesis — ficha de uso

Compañero persistente para redactar y revisar la memoria. Revisa el `main.tex` *contigo* (te marca qué
cambiaría y una reescritura, tú decides), responde dudas de contenido/decisiones citando la fuente, y llama al
especialista que haga falta. Implementado como **output style** (persona del hilo principal), porque eso es lo
único que **persiste toda la sesión** y **puede invocar a otros agentes** — un subagente normal no.

## Cómo se enciende y se apaga

```
/output-style consejero-tesis     ← lo activas (una vez, al abrir el chat). No desaparece en toda la sesión.
/output-style default             ← vuelves al modo normal.
```

Sobrevive a la compactación de contexto: aunque el chat se resuma, sigue siendo el consejero.

## La fuente viva: el notebook + el MANUAL

Al arrancar lee primero `notebooks/_build_STRATA_SMCI.py` (el builder del notebook entregable
`STRATA_SMCI.ipynb`) y `memoria/MANUAL.md`, y los toma como **la verdad de lo que estás haciendo ahora**. Si el
texto de un capítulo —sobre todo el cap. 4— arrastra una decisión, un parámetro o una **fecha que el notebook ya
cambió**, te lo marca como anotación de coherencia con el valor correcto. Ante conflicto, gana el notebook.

## Qué hace

- **Revisa** sección a sección. Te marca cada cambio dentro del `.tex`, en el sitio, como un comentario de Word:
  ```
  % >>> CONSEJERO [cap3-04 · estilo]
  %   CAMBIARÍA: ...
  %   POR QUÉ:   ...   (regla aprendida / fuente del repo)
  %   REESCRITURA: «...»
  % <<< CONSEJERO
  ```
  No toca tu texto. Tú decides: "aplica la cap3-04" (la aplica y borra el comentario) o "descarta la 05".
- **Asesora**: "¿meto/quito X?", "¿por qué está Y aquí?", "¿esto se sostiene?" → responde citando la fuente y
  recomienda.
- **Consultorio**: detecta algo dudoso (cifra sin test, cita sin verificar, olor a IA) y **te pregunta antes**
  de lanzar al especialista (`rigor-matematico`, `estilo-raquel`, `experto-citas`, `defensa-tutor`, …).

## Gestionar las anotaciones

```
/consejero-anotaciones          ← lista todas las anotaciones >>> CONSEJERO vivas (id, categoría, fichero:línea).
/consejero-anotaciones clear    ← borra las que ya resolviste (pregunta antes cuáles).
```

Los comentarios `% >>> CONSEJERO` no afectan a la compilación; puedes dejarlos mientras decides.

## Qué NO hace

- No reescribe capítulos enteros solo: revisa y propone, tú eliges.
- No aplica cambios sin que se lo pidas, ni lanza subagentes sin avisar.
- No inventa cifras: todo sale del JSON / `RESULTADOS_OBJETIVO.md §1bis` / las tablas.
- No commitea ni mergea a `main` sin permiso; nunca toca `.env`.

## Ficheros

- Persona: `.claude/output-styles/consejero-tesis.md`
- Comando de anotaciones: `.claude/commands/consejero-anotaciones.md`
- Contexto que carga al arrancar: `CLAUDE.md`, `DECISIONES_ESENCIALES.md`, `memoria/MANUAL.md`,
  `memoria/estructura_cap3.md`, `memoria/ESTILO_Y_ANTIIA.md`, `memoria/correcciones_aprendidas.md`,
  `RESULTADOS_OBJETIVO.md` (§1bis), `docs/chats/decision_activo/smci.md`, `tesis_assets/glosario.md`.
