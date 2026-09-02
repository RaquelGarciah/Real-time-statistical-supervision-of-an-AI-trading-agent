# LIMPIEZA_MANIFIESTO — inventario para aprobación

**Fecha:** 2026-06-17. **Política:** manifiesto con aprobación previa + **archivar** (mover, no borrar).
**Raquel aprueba esta tabla ANTES de mover nada.** Tras la aprobación y la auditoría de coherencia (Fase 0.5),
se hace `git mv` a `_obsoleto/` (obsoleto) y a `falsacion/` (negativos pre-registrados). Cero `rm` duro.

**Nunca se toca:** `BITACORA.md`, `cache/agent/`, `cache/models/`, `tests/`, `core/`, `strata/`, `agent/`,
`config.py`, todo `tesis/` y `tesis_assets/`.

**Cuatro destinos:** CONSERVAR · ACTUALIZAR (vigente, corregir cifras) · FALSACIÓN (`falsacion/`, negativos
honestos) · ARCHIVAR (`_obsoleto/`).

> **Hallazgo importante (buena noticia).** Los documentos canónicos **ya están actualizados a SMCI**:
> `DECISIONES_ESENCIALES.md` (#13–16, vivas a 2026-06-17), `RESULTADOS_OBJETIVO.md` (§1bis SMCI, embargo=1,
> 0,552), `decisiones_respaldadas_literatura.md`, `graficas_clave.md`, `docs/chats/decision_activo/smci.md`. No
> hay que reescribirlos: hay que **indexarlos** y arreglar los pocos rezagados. El caos es de dispersión, no de
> contenido.

---

## 1. CONSERVAR (canónico/vigente)

### Documentos raíz
| Fichero | Motivo |
|---|---|
| `CLAUDE.md`, `BITACORA.md`, `LECCIONES_APRENDIDAS.md`, `AGENTES_SUGERIDOS.md`, `CONSEJO_ASESOR.md` | Constitución y registro; no envejecen. |
| `DECISIONES_ESENCIALES.md` | 16 decisiones vivas a 2026-06-17, #13–16 = pivot SMCI. **Es la verdad de decisiones.** |
| `RESULTADOS_OBJETIVO.md` | §1bis SMCI canónico (0,552, embargo=1). *Coherencia: §1 SPY aún cita 0,539 CPCV → revisar en 0.5.* |
| `decisiones_respaldadas_literatura.md` | Decisiones con cita verificada (embargo=1, ensemble, tests). Base para el cap. 3/4. |
| `graficas_clave.md` | Figuras imprescindibles del caso SMCI con su fuente. |
| `ENTREGABLES.md`, `instrucciones_redaccion.md` | Checklist de entrega y guía de redacción, vigentes. |

### Notebooks y su builder
| Fichero | Motivo |
|---|---|
| `notebooks/strata_canonical.ipynb` (+ `_build.py`) | Notebook canónico del método (SPY). |
| `notebooks/m10_better_smci.ipynb` (+ `_build_m10_better_smci.py`) | **Entregable del caso SMCI** (§A–§G). |
| `notebooks/logic_esential.ipynb` (+ `_build_logic_esential.py`) | Didáctico (embargo §14b, ensemble §14d). |
| `notebooks/validacion_live_backtest.ipynb` | Protocolo de validación pre-deployment. |

### docs/
| Fichero | Motivo |
|---|---|
| `docs/chats/decision_activo/smci.md` | Recorrido canónico de la elección de SMCI (auditado). |
| `docs/chats/decision_activo/spy_understandStrata.md` | SPY = caso-mecanismo (rescate significativo). |
| `docs/chats/questions_and_answers.md`, `docs/defensa_walkforward.md` | Banco de defensa ante el tribunal. |
| `docs/tutor_transcripts/Reunion_Dani_2026-06-16.md` + `README.md` | Última reunión del tutor (pivot). |
| `docs/tutor_transcripts/Calle de Hilarión Eslava, 46 2.md` | **Es la transcripción de la reunión 1 del tutor** (la dirección es el lugar). Renombrar a `Reunion_2026-06-07.md`. |

### experiments/ + outputs/ (canónicos y diagnósticos pre-registrados)
| Familia | Ficheros | Motivo |
|---|---|---|
| Caso SMCI (final) | `m10_smci_deep`, `m10_smci_rolling`, `m10_smci_embargo`, `m10_smci_valtest_robustez` | Configs a priori, rolling, embargo y robustez a la partición (60/40, 70/30, 80/20). |
| SPY método | `spy_m10_full_report`, `spy_ablation_robustness`, `spy_momentum_ablation`, `walkforward_robustez` | M10 SPY, ablaciones, walk-forward (falsación de robustez Sharpe ya registrada). |
| Régimen / detectores | `k_ablation_panel`, `psa_gso_threshold_sensitivity`, `panel_intervention_scan`, `regime_direction_table`, `drift_test_k2k3`, `embargo_sweep`, `class_balance_diagnostic` | Selección K=3, umbrales, discrepancia agente↔régimen, balance de clases. |
| Panel desplegable | `walkforward_m10_causal`, `m10_pivot_scan` | Barrido panel que selecciona SMCI. |
| Dominio (límite) | `recalibrate_nvda` (+ `walkforward_robustez_nvda.json`) | NVDA: STRATA no rescata (leverage débil). Delimita el alcance. |

---

## 2. ACTUALIZAR (vigente, pero con cifras pre-pivot a corregir)
| Fichero | Qué corregir |
|---|---|
| `CONOCIMIENTO_ACUMULADO.md` | Aún se apoya en SPY 0,539 (CPCV) como cifra estrella; añadir el marco SMCI/embargo=1 y marcar 0,539 como CPCV-contraste. |
| `README.md` | Sin mención a SMCI (pre-pivot). Añadir el caso de estudio SMCI. |
| `INVESTIGACION_VALIDACION_TIEMPO_REAL.md` | Cita embargo=5 y variantes M10-v5/v6/v7 descartadas; alinear con embargo=1 y M10-WF ensemble. |
| `docs/chats/chat_m10.md`, `docs/chats/pasos_m10.md` | Cifras CPCV (0,539) y embargo=5; anotar que son históricas/no desplegables. |

---

## 3. FALSACIÓN → `falsacion/` (todo lo que probé y descarté, GUARDADO y ORGANIZADO)
Decisión de Raquel: que **todas las configuraciones probadas** queden bien guardadas aquí, por si el tribunal o
ella preguntan. Con `falsacion/INDICE.md` que explique **qué refutó cada una**. Subcarpetas:

| Subcarpeta | Ficheros | Qué se probó / qué refutó |
|---|---|---|
| `falsacion/momentum/` | `momentum_decision_rule`, `momentum_exante_rule`, `momentum_exante_battery`, `momentum_horizon_rule`, `momentum_conditional_calib_oos`, `momentum_rule_robustness`, `momentum_tsmom_monthly` (+ json) | Momentum **como señal**: mejora la accuracy en **SPY** (0,590, p=0,005) pero **no en SMCI**; ninguna regla a-priori para decidir si meterlo es robusta → se decide **no incluir momentum** (universalidad). Contexto en `spy_understandStrata.md` §4–§7. |
| `falsacion/m10_configs/` | `m10_smci_advanced`, `m10_improve_smci`, `m10_smci_select`, `m10_v3_causal_panel`, `m10_causal_panel_recon` (+ json) **+** `M10_V3_GUIA.md`, `M10_V7_GUIA.md` | Configs de M10 exploradas: tuning que sobreajusta; métodos avanzados (triple-barrier, stacking, voting, abstención) que no mejoran; v3/v7 que resultaron **look-ahead** (CPCV). Las guías documentan esas configs. |
| `falsacion/casos_y_horizontes/` | `m10_valtest_casestudy` (corte 60/40 con config **antigua** → 0,150, falló), `m10_weekly_horizon` | Corte único 60/40 con la config vieja falla (SMCI no estacionario); horizonte semanal no da significancia. **OJO:** el 60/40 con el **ensemble actual SÍ funciona** y está en CONSERVAR (`m10_smci_valtest_robustez`). |
| `falsacion/k_y_signo/` | `k_per_asset`, `k_per_asset_directional`, `m8_datadriven_sign` (+ json) | K-por-activo no generaliza; el arreglo del **bug del signo** (hardcode→data-driven) corrige un error real pero **no** produce un caso ganador (el bug ya es decisión viva #6). |
| `falsacion/smci_estudios/` | `smci_config_study`, `smci_protocol_study` (+ json) | Estudios de config/protocolo SMCI, superados por `m10_smci_deep`/`m10_smci_valtest_robustez`. |
| `falsacion/nvda/` | `m10_shap_ablation_nvda` (+ json) | SHAP/ablación en NVDA (activo de límite, leverage débil). |

`VALIDACION_VAL_TEST.md` → su contenido (las configs M10-v2…v7 y su consistencia val/test) se referencia desde
`falsacion/INDICE.md`; el documento en sí va a `_obsoleto/` (ver §4).

---

## 4. ARCHIVAR → `_obsoleto/` (clutter sin valor probatorio: duplicados/superado)
| Fichero | Subcarpeta | Motivo |
|---|---|---|
| `strata_tesis_overleaf.zip`, `strata_tesis_overleaf_raiz.zip` | `zips/` | Backups de Overleaf duplicados; la memoria vive en `tesis/`. |
| `notebooks/esqueleto.ipynb` (+ `_build_esqueleto.py`) | `notebooks/` | Plantilla sin análisis. |
| `notebooks/experimentos.ipynb` (+ `_build_experimentos.py`) | `notebooks/` | Exploración superada por los notebooks canónicos. |
| `notebooks/decision_activo.ipynb` (+ `_build_decision_activo.py`) | `notebooks/` | Registro gráfico (Fase 2); superado por `smci.md` + `m10_better_smci.ipynb`. |
| `docs/chats/decision_activo/decision_activo.md` | `docs/` | Conversación previa; superada por `smci.md`. |
| `VALIDACION_VAL_TEST.md` | `docs/` | Recomienda M10-v4/v6, que chocan con el M10-WF ensemble canónico (configs ya guardadas en `falsacion/`). |
| `experiments/k_selection*.py` (3) + `.json` | `experiments/` | Selección de K isotónica antigua; superada por `k_ablation_panel`. |
| `experiments/m10_vs_m8_drift.py`, `hmm_seed_stability_spy.py` | `experiments/` | Reejecutados/superados (drift→`drift_test_k2k3`; τ=0,5 congelada). |
| `experiments/umbral_psa_gso_valtest.py` | `experiments/` | Superado por `psa_gso_threshold_sensitivity`. |
| `experiments/strata_activo_balanceado.py` | `experiments/` | Exploración de "activo balanceado" ya resuelta con SMCI. |
| `docs/tutor_transcripts/Reunión Dani 2.m4a` | `audio/` | Audio fuente; redundante con la transcripción `.md`. |

**Se quedan como CONSERVAR (tooling, NO mover):** `experiments/recalibrate_hmm.py` y `experiments/m10_k2.py`
son **utilidades de código** que importan scripts canónicos (`recalibrate_nvda`, `walkforward_robustez`);
moverlas rompería esos scripts. `docs/referencia_viz_comparison.py` se mantiene como utilidad de figuras.

---

## 5. DUDA — RESUELTO (decisiones de Raquel, 2026-06-17)
Todas las dudas quedan resueltas en §3 y §4: momentum→`falsacion/momentum/`; `m8_datadriven_sign`→`falsacion/`;
`VALIDACION_VAL_TEST.md`→`_obsoleto/` (configs guardadas en `falsacion/`); `smci_config/protocol_study`,
`m10_causal_panel_recon`, `m10_shap_ablation_nvda`→`falsacion/`; helpers de código→CONSERVAR.

---

## 6. Incongruencias a resolver en la Fase 0.5 (antes de archivar)
- **SPY: 0,539 vs 0,534.** Decisión #14 dice "desplegable = walk-forward; CPCV solo como contraste", pero
  `RESULTADOS_OBJETIVO §1`, `CONOCIMIENTO_ACUMULADO` y `chat_m10` aún usan **0,539 (CPCV)** como cifra SPY. El
  número desplegable es **0,534 (WF)** (`smci.md` §7). Fijar uno y anotar el otro como contraste.
- **Método M10:** que la spec única (ensemble 300×4, embargo=1) sea idéntica en todos los docs vivos (hay restos
  de "embargo=5" en `pasos_m10`/`INVESTIGACION_*`).
- **Periodos:** OOS SMCI = 250 d (post burn-in 150); OOS método SPY = 2024-10→2026; calibración 2000→2024-09.
  Verificar que no conviven dos OOS distintos en lo conservado.
- **Objetivos:** el claim (SMCI nominal + robustez partición/rolling; significancia=futuro; SPY=mecanismo) debe
  ser idéntico en `DECISIONES`, `RESULTADOS §1bis`, `smci.md`, `questions_and_answers`, `defensa_walkforward`.

---

## 7. Estructura propuesta tras archivar
```
STRATA_kit/
  memoria/            ← NUEVO: MANUAL.md + guion + índice (Fase 1)
  falsacion/          ← NUEVO: negativos pre-registrados + INDICE.md
  _obsoleto/          ← NUEVO: archivado (guias_m10/ zips/ notebooks/ experiments/ docs/)
  tesis/ tesis_assets/ ← LaTeX (intacto)
  experiments/ outputs/ notebooks/ ← solo lo CONSERVADO
  cache/ core/ strata/ agent/ tests/ config.py ← intactos
  + docs canónicos raíz (CLAUDE, BITACORA, DECISIONES_ESENCIALES, RESULTADOS_OBJETIVO, ...)
```

**Siguiente paso:** revisa esta tabla, resuelve la sección 5 (DUDA) y dime correcciones. Con tu visto bueno
ejecuto la Fase 0.5 (coherencia) y luego el archivado.
