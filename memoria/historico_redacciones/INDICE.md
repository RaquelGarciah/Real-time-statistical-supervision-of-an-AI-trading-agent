# Histórico de redacciones — índice

Archivo ordenado de **todas las versiones de prosa entregadas** para la memoria, para comparar y ver la
evolución (borrador → revisado por gates → corregido por Raquel). Es la **vista humana** del historial; git
guarda además todo commit a commit.

**Cuándo se guarda un snapshot:** en cada hito — (a) primer borrador, (b) tras pasar los gates de calidad,
(c) tras las correcciones de Raquel. El fichero se nombra `capN/capN_vM_AAAA-MM-DD_<hito>.tex`.

---

## Capítulo 3 — Marco teórico

| Versión | Fecha | Hito | Fichero | Qué cambió / origen |
|---|---|---|---|---|
| v1 | 2026-06-17 | tras gates | `cap3/cap3_v1_2026-06-17_tras-gates.tex` | Reestructura en 4 bloques (§0 preliminares · §1 STRATA técnico · §2 detectores · §3 STRATA aplicado · §4 validación), demostraciones preservadas. Pasados y corregidos los 6 gates: estilo (plantillas rotas, sin guiones-muletilla), rigor (regla de mayoría con K=3, hazard 1/60, override-C), citas, anti-IA, plagio (reformulada prosa forward/run-length/hazard), coherencia (STRATA≠M8, μ_k aclarado). |

*Nota:* el borrador inicial y los estados intermedios existen en el historial de git (commits `wip(tesis):
autosave` del 2026-06-17 ~20:31 → ~21:59). A partir de aquí se guardan snapshots limpios por hito. La próxima
versión será **v2** cuando Raquel corrija el capítulo (origen: `memoria/correcciones/cap3.md`).

---

## Cómo se relaciona con el aprendizaje
Cada versión nueva que nace de una corrección de Raquel apunta, en la columna "origen", al
`memoria/correcciones/capN.md` que la motivó. Así queda la cadena completa: **borrador → gates → corrección →
reglas aprendidas** (`memoria/correcciones_aprendidas.md`).
