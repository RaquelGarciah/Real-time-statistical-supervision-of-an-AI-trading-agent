# Histórico del PDF de la tesis

Versiones compiladas de la memoria, una por hito. La **versión viva** se compila siempre en `tesis/main.pdf`
(artefacto de build, se sobrescribe); aquí quedan las copias fechadas que no se pisan.

**Convención de nombre:** `tesis_AAAA-MM-DD_HHMM_<etiqueta>.pdf` (fecha y hora de la compilación + qué hito).
Orden cronológico ascendente.

**Cómo añadir una versión:** comando `/tesis-snapshot [etiqueta]` — compila `main.tex` y guarda aquí una copia
fechada, añadiendo su línea a este índice.

| Fecha | Hora | Fichero | Hito |
|---|---|---|---|
| 2026-06-18 | 05:20 | [`tesis_2026-06-18_0520_borrador-cap3-4.pdf`](tesis_2026-06-18_0520_borrador-cap3-4.pdf) | Borrador con cap. 3 y cap. 4 ya redactados (estado temprano del día). |
| 2026-06-18 | 09:21 | [`tesis_2026-06-18_0921_cap4-compila-fix-percent.pdf`](tesis_2026-06-18_0921_cap4-compila-fix-percent.pdf) | Cap. 4 compila tras arreglar `\,\%` (la versión que construyó la Action en verde). |
| 2026-06-18 | 11:20 | [`tesis_2026-06-18_1120_override-intervencion-dsr.pdf`](tesis_2026-06-18_1120_override-intervencion-dsr.pdf) | Renombrado override→intervención + reencuadre del DSR (cap. 3 y 4). **Versión actual.** |

---

*Nota:* el historial completo de git guarda además cada autosave de `tesis/main.pdf` (recuperable con
`git show <commit>:tesis/main.pdf > salida.pdf`). Esta carpeta es la **vista humana ordenada** de los hitos, no
de cada autosave.
| 2026-06-18 | 12:39 | [`tesis_2026-06-18_1239_conectores-ritmo.pdf`](tesis_2026-06-18_1239_conectores-ritmo.pdf) | Pasada de conectores y ritmo (voz de Raquel) en cap. 3 y 4. |
