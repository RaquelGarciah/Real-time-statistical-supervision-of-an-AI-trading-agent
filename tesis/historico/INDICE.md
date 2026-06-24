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
| 2026-06-18 | 11:20 | [`tesis_2026-06-18_1120_override-intervencion-dsr.pdf`](tesis_2026-06-18_1120_override-intervencion-dsr.pdf) | Renombrado override→intervención + reencuadre del DSR (cap. 3 y 4). |
| 2026-06-18 | 12:39 | [`tesis_2026-06-18_1239_conectores-ritmo.pdf`](tesis_2026-06-18_1239_conectores-ritmo.pdf) | Pasada de conectores y ritmo (voz de Raquel) en cap. 3 y 4. |
| 2026-06-18 | 12:43 | [`tesis_2026-06-18_1243_conectores-redaccion.pdf`](tesis_2026-06-18_1243_conectores-redaccion.pdf) | Snapshot manual tras la pasada de conectores/redacción (mismo contenido que las 12:39, recompilado). |
| 2026-06-18 | 18:24 | [`tesis_2026-06-18_1824_cap3-rigor-citas.pdf`](tesis_2026-06-18_1824_cap3-rigor-citas.pdf) | Cap. 3 cerrado: rigor matemático (teorema GARCH no circular, lema EM, d-separación, BOCPD) + 4 definiciones formales (RAM/GSO/PSA/M8) + citas en cada demostración. |
| 2026-06-18 | 18:49 | [`tesis_2026-06-18_1849_cap3-intro-series-temporales.pdf`](tesis_2026-06-18_1849_cap3-intro-series-temporales.pdf) | Cap. 3: intro a series temporales y ARCH en §3.4 (revisada por agentes) + 4 refs. |
| 2026-06-18 | 19:47 | [`tesis_2026-06-18_1947_cap4-estilo.pdf`](tesis_2026-06-18_1947_cap4-estilo.pdf) | Cap. 4: pasada de estilo (gates estilo-raquel + detector-ia) sobre la versión reescrita por Raquel; registro plural/impersonal, fuera meta-comentarios y aperturas calcadas. |
| 2026-06-18 | 20:52 | [`tesis_2026-06-18_2052_primer-borrador.pdf`](tesis_2026-06-18_2052_primer-borrador.pdf) | Primer borrador completo (44 págs.): cap. 3 con correcciones de redacción aplicadas. |
| 2026-06-23 | 01:41 | [`tesis_2026-06-23_0141_cap3-estrategias-supervisadas.pdf`](tesis_2026-06-23_0141_cap3-estrategias-supervisadas.pdf) | Cap. 3: nueva §Estrategias supervisadas (49 págs.) con rigor máximo (árboles→GBM→XGBoost→stacking→AutoML→catálogo M5/M8/M10/AutoML/ZeroR/B&H), 2 proposiciones demostradas (pseudo-residuo logístico, score XGBoost) + 4 citas nuevas (Friedman 2001, van der Laan 2007, LeDell 2020, Hastie 2009). Gates rigor (PASS) + estilo (7 fixes). |
| 2026-06-23 | 07:10 | [`tesis_2026-06-23_0710_nuevo_enfoque.pdf`](tesis_2026-06-23_0710_nuevo_enfoque.pdf) | Snapshot del nuevo enfoque (caso SPY + panel de robustez): registro del agente revisor `raquel-quant` y registro de decisiones/resultados del marco práctico definitivo. |
| 2026-06-23 | 15:49 | [`tesis_2026-06-23_1549_marco_practico.pdf`](tesis_2026-06-23_1549_marco_practico.pdf) | Marco práctico (cap. 4) reescrito a la estructura del SPEC (4 secciones): ablación AutoML-H2O de detectores (§2), matrices de confusión SPY (6 estrategias, §3) y panel (mejor STRATA por activo, §4), fixes del bucle revisor `raquel-quant` (rescate estratificado, cuota SHAP SPY, IWM apéndice). APROBADO ronda 2. 47 págs. |
| 2026-06-23 | 17:07 | [`tesis_2026-06-23_1707_marco-practico-v2.pdf`](tesis_2026-06-23_1707_marco-practico-v2.pdf) | Marco práctico v2 (cap. 4) expandido a research note completo: 13 tablas (detectores+atribución, variantes de intervención override/abstención/reduce, McNemar, ablación+SHAP, bull/bear, costes/turnover, silueta, perfiles de cluster). Gates: rigor APROBADO (3 fixes: denominador detectores n=401, PSA 0,5%, rolling 9/10), detector-ia 22% PASS, estilo BLOCK→15 fixes (patrón "no es X", meta-comentarios). Cifras trazadas a JSON; intro de Raquel intacta. 49 págs. |
| 2026-06-24 | 04:07 | [`tesis_2026-06-24_0407_prueba.pdf`](tesis_2026-06-24_0407_prueba.pdf) | Snapshot de prueba del flujo `/tesis-snapshot`. Estado del PDF tras commit `aa50ad5` (cap. 4 + cap. 5 reescritos al enfoque canónico de 10 activos). 44 págs. |
| 2026-06-24 | 07:40 | [`tesis_2026-06-24_0740_CAP4_BIEN.pdf`](tesis_2026-06-24_0740_CAP4_BIEN.pdf) | Cap. 4 en estado «bien» (caso SPY: leverage effect, tres detectores, atribución RAM, McNemar, panel de robustez). 59 págs. *Incidencia conocida:* la Figura de regímenes (`s1_spy.tex:42`) referencia `figuras/cap4_regimenes_spy`, que no existe; el fichero real es el huérfano `cap4_regimen_precio.pdf` → sale una caja vacía en esa figura. Archivado tal cual a petición. |
| 2026-06-24 | 09:30 | [`tesis_2026-06-24_0930_cap3,4,5_v1.pdf`](tesis_2026-06-24_0930_cap3,4,5_v1.pdf) | Snapshot v1 con cap. 3 + cap. 4 + cap. 5 (68 págs.). Compilación limpia: sin avisos de figuras faltantes ni referencias/citas indefinidas — resuelta la incidencia de la figura de regímenes del snapshot de las 07:40. Contexto: commit `9d38111` (cap. 2 explica el agente + cifras alineadas a SPY/panel-10). **Versión actual.** |

---

*Nota:* el historial completo de git guarda además cada autosave de `tesis/main.pdf` (recuperable con
`git show <commit>:tesis/main.pdf > salida.pdf`). Esta carpeta es la **vista humana ordenada** de los hitos, no
de cada autosave.
