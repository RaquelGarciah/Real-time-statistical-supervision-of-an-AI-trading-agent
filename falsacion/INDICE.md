# Falsación — registro de lo que probé y descarté (honesto)

Estos experimentos **no son basura**: son el mecanismo de falsación del TFG (CLAUDE.md §2). Cada uno probó una
hipótesis y la refutó o quedó descartado tras una prueba honesta. Se guardan aquí, organizados, por si el
tribunal (o yo) pregunta. **Los resultados (.json) siguen en `outputs/experiments/`** (los lee el notebook
`m10_better_smci.ipynb`); aquí están los **scripts** y la explicación.

| Carpeta / script | Qué probé | Qué refutó / por qué se descartó | Resultado (JSON) |
|---|---|---|---|
| **momentum/** (7 scripts) | Momentum como señal y reglas a-priori para decidir si meterlo al XGBoost | Mejora la accuracy en **SPY** (0,590, p=0,005) pero **no en SMCI**; ninguna regla a-priori es robusta → se decide **no incluir momentum** (más universal). Contexto: `docs/chats/decision_activo/spy_understandStrata.md` §4–§7 | `outputs/experiments/momentum_*.json` |
| **m10_configs/** `m10_improve_smci` | Tuning de 165 configs eligiendo la mejor en validación | Se desploma en test (−0,10): sobreajuste de selección | `m10_improve_smci.json` |
| **m10_configs/** `m10_smci_advanced` | Triple-barrier, modelos por régimen, stacking, voting, abstención | Ninguno mejora la accuracy de SMCI; varios la degradan | `m10_smci_advanced.json` |
| **m10_configs/** `m10_v3_causal_panel`, `m10_causal_panel_recon` | M10-v3 (isotónica/abstención/P95) en causal sobre el panel | Negativo en los 10 → confirma que las cifras buenas de las guías eran **look-ahead** (CPCV) | `m10_v3_causal_panel.json`, `m10_causal_panel_recon.json` |
| **m10_configs/** `M10_V3_GUIA.md`, `M10_V7_GUIA.md` | Documentación de las variantes M10-v3 y v7 | Variantes CPCV/look-ahead descartadas; sustituidas por el M10-WF ensemble desplegable | — |
| **casos_y_horizontes/** `m10_valtest_casestudy` | Corte único 60/40 en SMCI (config antigua, 1 semilla) | Dio 0,150 (anti-predictivo): SMCI es no estacionario → exige walk-forward. **OJO:** el 60/40 con el ensemble actual SÍ funciona (`experiments/m10_smci_valtest_robustez.py`, conservado) | `m10_valtest_casestudy.json` |
| **casos_y_horizontes/** `m10_weekly_horizon` | Horizonte semanal (5 días) | No da significancia en el panel | `m10_weekly_horizon.json` |
| **k_y_signo/** `k_per_asset`, `k_per_asset_directional` | Elegir K por activo (maximizando accuracy de calibración) | No generaliza (concordancia calib↔OOS ≈ azar) → se fija **K=3** para todos | `k_per_asset*.json` |
| **k_y_signo/** `m8_datadriven_sign` | Corregir el **bug del signo** de RAM (hardcode → data-driven por activo) | Corrige un error real (`strata/detectors.py:209`) pero el arreglo **no** produce un caso ganador significativo. El bug ya es decisión viva #6 | `m8_datadriven_sign.json` |
| **smci_estudios/** `smci_config_study`, `smci_protocol_study` | Configs y protocolo en SMCI | Superados por `m10_smci_deep` / `m10_smci_valtest_robustez` (conservados) | `smci_*_study.json` |
| **nvda/** `m10_shap_ablation_nvda` | SHAP/ablación en NVDA | NVDA es activo de límite (leverage débil); diagnóstico, no caso central | `m10_shap_ablation_nvda.json` |

**Resumen de la lección transversal:** la dirección diaria de un activo individual es casi un paseo aleatorio a
este tamaño de muestra; ninguna palanca (tuning, métodos avanzados, momentum, horizonte, signo) produce un
B&H-beat **significativo**. Lo que sí es robusto y desplegable es el M10-WF ensemble, que **bate a todo
nominalmente** en SMCI (benchmark justo). Ver `RESULTADOS_OBJETIVO.md` §1bis y `DECISIONES_ESENCIALES.md` #13–16.
