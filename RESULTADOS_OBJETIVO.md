# Resultados objetivo — cifras de referencia (no objetivo único)

> **⚠️ ENFOQUE ACTUAL (2026-06-24) — fuente de verdad práctica: [`MARCO_PRACTICO_CONTEXTO.md`](MARCO_PRACTICO_CONTEXTO.md).**
> Caso central = **SPY + panel de 10** (SPY, QQQ, XLF, DIA, XLK, XLE, ROKU, SMCI, MARA, UNG), **sin apéndice**;
> SMCI es **uno de los 10**, no el caso central. Headline: SPY AutoML 0,574 (McNemar vs M5 0,0002; vs ZeroR 0,90
> nominal); **pooled-10 riesgo M8/M10/AutoML vs M5 +0,60/+1,12/+1,08** (`bullbear_confirmatory.json` POOLED10; Bonferroni: M10/AutoML sí, M8 no); **TOST**
> (aprendiz bate la regla en accuracy, empata en riesgo); **DiD** +1,37 (p=0,008); **ley leverage sobre 10** r=−0,56
> p=0,093 (α=0,10). Alcance = rescate + riesgo (alfa = línea futura). **Lo que abajo diga "SMCI caso de estudio /
> 0,539 CPCV / pooled-15 / M10 0,552" está obsoleto.**
>
> _[Coherencia 2026-06-17 — HISTÓRICO, superado]_ caso de estudio = §1bis (SMCI, M10 0,552); §1 (SPY) 0,539 es CPCV.

**ESTRUCTURA:** §1 = cifras canónicas del nuevo proyecto (fuente única de verdad para la memoria LaTeX). §2 = tabla del proyecto anterior (referencia histórica, no canónica).

---

## §1. Cifras canónicas — nuevo proyecto (notebook strata_canonical.ipynb, 2026-06-08)

**Fuente:** `notebooks/strata_canonical.ipynb` (K=3, τ=0.5, OOS 2024-10-01→2026-06, N=401). Estas son las cifras que van a la memoria LaTeX.

### Tabla maestra canónica (SPY OOS, N=401 días)

| Estrategia | Accuracy | Sharpe | McNemar vs M5 | Lectura |
|---|---:|---:|---:|---|
| B&H (referencia pasiva) | 0.569 | ≈ B&H | — | Techo del problema |
| M5 (agente solo) | 0.384 | −1.82 | — | Perdedor direccional (sign test p=4·10⁻⁶) |
| M7 (reduce) | 0.384 | −1.41 | trivial (b=c=0) | Reduce daño en P&L (DM p=0.095), no accuracy |
| M8 (STRATA override C) | 0.436 | +0.67 | **p=0.069 (τ=0.5) / 0.088 (τ=0.40)** | Rescata accuracy; Sharpe frágil (P(Sharpe>0) corr.≈0.10) |
| M10 (XGBoost CPCV) | **0.539** | +0.64 | DM M10 vs M8: p=0.67 | Mejor accuracy; equivalente a M8 en P&L |
| M2 (régimen×GARCH, sin agente) | — | +~0 | DM M8 vs M2: p=0.44 | Ablación sin agente |

**Métrica primaria: accuracy direccional** (el Sharpe es ilustración frágil; P(Sharpe>0) corregida M8 ≈ 0.10).
**Dos planos (walk-forward §13):**
- *Plano accuracy:* M10 rescata accuracy en AMBOS regímenes (bajista Holm p_adj=0.075, block-perm p=0.061). M8 solo en alcista (Holm p_adj=0.15) y nulo en bajista (p=1.0). MCC M10=+0.068 (único positivo).
- *Plano Sharpe:* rescate NO robusto. Criterio confirmatorio = cota Bonferroni: M8−M5=−0.49; M10−M5=−0.48; ambas <0 → H1_b False. IC95 crudo M10−M5=[−0.02,+5.79] NO es el criterio. ΔSharpe se invierte en bajista: M8=−3.92, M10=−1.06 (n=123 ≥ 60) — falsificación pre-registrada disparada para ambos.
**Veredicto formal:** `robustez_no_sostenida` (plano Sharpe). "STRATA-SPY recupera accuracy cross-régimen para M10; su ventaja económica es condicional al alza."

### Tests pareados canónicos

| Test | Comparación | p-valor | Lectura |
|---|---|---:|---|
| McNemar (τ=0.5) | M8 vs M5 | **0.069** | Rechaza H0 a α=0.10 |
| McNemar (τ=0.40 default) | M8 vs M5 | **0.088** | Rechaza H0 a α=0.10 (blindaje dual) |
| Block permutation | M8 vs M5 | **0.044** | Controla autocorrelación |
| Diebold-Mariano | M10 vs M8 P&L | **0.67** | Equivalentes en P&L |
| Walk-forward B-conf M8−M5 | cota Bonferroni (DECIDE) | −0.49 | H1_b False — rescate en Sharpe no robusto |
| Walk-forward B-conf M10−M5 | cota Bonferroni (DECIDE) | −0.48 | H1_b False — ídem; IC95=[−0.02,+5.79] NO es el criterio |
| McNemar M10 vs M5 bajista | Holm p_adj | **0.075** | M10 rescata accuracy en bajista (block-perm 0.061) |
| McNemar M10 vs M5 alcista | Holm p_adj | **0.005** | M10 rescata accuracy en alcista (block-perm 0.000) |

---

## §1bis. Caso de estudio SMCI (activo del tutor) — cifras canónicas (embargo=1, 2026-06-17)

SPY (§1) es el **caso central del método**. **SMCI es el activo del CASO DE ESTUDIO** que pide el tutor:
un activo con **B&H ≈ 50 %** (benchmark justo) donde el M10 **desplegable** bate a todo en accuracy. Fuente:
`notebooks/m10_better_smci.ipynb` + JSON de `experiments/m10_smci_*`. Recorrido completo:
`docs/chats/decision_activo/smci.md`.

| Estrategia (OOS SMCI, n=250, walk-forward, embargo=1) | Accuracy | Sharpe | Equity | Lectura |
|---|---|---|---|---|
| M5 (agente) | 0.484 | −0.24 | — | agente 95 % corto |
| M8 (regla) | 0.496 | +0.33 | — | STRATA interviene solo 3 % → ≈ M5 |
| B&H (trivial, siempre largo) | 0.484 | +0.03 | 0.71× | benchmark económico (≈ moneda) |
| S&H (siempre corto, espejo de B&H) | 0.516 | — | — | la otra estrategia constante |
| Clase mayoritaria (ZeroR / NIR) | 0.516 | — | — | no-habilidad = máx(B&H, S&H); en SMCI **= S&H** ("siempre corto") |
| **M10-WF ensemble** (10 semillas, 22 features) | **0.552** | **+1.84** | **3.24×** | bate a **todo** (incl. mayoría) **nominal** |

- **Baselines:** se compara contra B&H (económico, siempre largo) **y la clase mayoritaria** (ZeroR /
  no-information rate = siempre la dirección dominante; en SMCI "siempre corto", 0.516 — Witten et al. 2016;
  Kuhn 2008). M10 (0.552) bate a **ambos** → su ventaja **no es un mero sesgo a corto**.
- **Significancia:** **nominal, no plena.** Test correcto = **binomial M10 vs NIR** (clase mayoritaria) = 0.141
  (no sig); block-perm vs B&H 0.047 (no sobrevive Bonferroni-5 ≈ 0.28); sign vs 0.5 p=0.057 (binomial 1-cola;
  0.114 sign bilateral), full OOS n=250; no bate al agente (McNemar 0.16). En lo económico, **P(Sharpe>0) =
  0.976** (Sharpe positivo con alta prob., hiperparámetros a priori; penalizada por las ≥6 configs exploradas
  baja a ≈0.72, por eso el Sharpe es ilustración, no prueba; método Bailey-LdP 2014).
  Ablación: las 7 features STRATA suben la accuracy de
  M10 de 0.468 (solo-agente) a 0.552 (McNemar 0.053, casi sig.) → el meta-aprendiz sí usa la señal de STRATA.
- **Robustez a la partición (respaldo):** con 3 splits estándar (60/40, 70/30, 80/20; burn-in 150), M10 bate a
  M5/M8/B&H **y a la clase mayoritaria** en validación Y test en los tres (val 0.52–0.535, test 0.60–0.62) → la
  conclusión no depende del corte. Fuente: `experiments/m10_smci_valtest_robustez.py`. (Al achicar el test la
  accuracy sube pero pierde potencia: binom vs NIR 0.183→0.060; por eso el headline es el de todo el OOS.)
- **Robustez a la ventana de calibración (sugerencia del tutor):** recalibrando HMM+GARCH con ventanas más
  cortas (2007..2022, fin fijo, sin fuga), M10 **degrada** (0.552 con la completa → ~0.48, nivel del agente) y la
  media de Crisis **sigue positiva** (no se vuelve direccional). La ventana completa (pre-registrada, dec. #3) es
  la más robusta; la ventaja de M10 depende de las features de régimen calibradas sobre la historia larga.
  Fuente: `experiments/smci_calib_window.py`.
- **Umbral 0.5 validado:** barriendo el umbral por validación/test (60/40), 0.5 es el óptimo en accuracy Y Sharpe
  en ambos tramos → fijado a priori, sin grado de libertad extra.
- **El régimen no es direccional en SMCI:** separa por **volatilidad** (std 0.019<0.034<0.066) pero la media por
  régimen solo es significativa en Estrés (positiva); **Crisis tiene media positiva** (leverage débil). En el
  drawdown de verano 2025 (−34 %) el régimen **capta** la crisis (Crisis dominante el 81 %, rezago RV²¹ ~6 d)
  pero M10 sigue largo porque aprendió "Crisis≈subida": la causa es la **no-direccionalidad**, no el rezago.
- **Por qué no es significativo:** en SMCI el agente ya está 95 % corto (alineado con el régimen) → M5/M8/M10
  son la misma apuesta corta; STRATA rescata solo donde el agente discrepa de un régimen que acierta (SPY,
  M10 vs M5 p=0.0041). SMCI es el único activo del panel donde M10 > M5, M8 y B&H (muro estructural 2×2).
- **Protocolo:** WF expandible, burn-in 150, reentreno 21 d, **embargo 1** (horizonte de etiqueta=1; Tashman
  2000 / López de Prado 2018 §7.4 — ver DECISIONES_ESENCIALES #15). Ensemble = bagging (Breiman 1996).
- **Frase de cierre SMCI:** "El M10 desplegable bate al pasivo en un benchmark justo (0.552 vs 0.484), de
  forma nominal; la significancia plena requiere más muestra (trabajo futuro)."

---

## §1ter. Caso central SPY con AutoML — cifras canónicas del marco práctico (panel mm25, 2026-06-23)

Caso **central** del notebook definitivo `STRATA_marco_practico.ipynb` (decisión #18). Con la config canónica
(AutoML-H2O max_models=25, GBM/XGBoost/StackedEnsemble, AUC, Purged K-Fold emb=1, WF N0=150/step=21), en SPY
**AutoML gana en punto a TODAS** las estrategias. Fuente: `outputs/experiments/automl_runs/panel_mm25_*.json`
(`por_activo.SPY`), `spy_m10_full_report.json`, `decision_automl_prep.json`.

| Estrategia (SPY OOS desplegable, n=251) | Accuracy | Sharpe | maxDD | Equity | Lectura |
|---|---|---|---|---|---|
| M5 (agente) | 0.3665 | −3.07 | −0.302 | 0.699× | agente direccionalmente malo, se arruina |
| M8 (regla STRATA) | 0.4422 | −0.464 | −0.152 | 0.941× | rescata al agente (riesgo) |
| M10 (XGBoost canónico) | 0.4940 | −0.604 | −0.161 | 0.920× | meta-learner |
| **AutoML (H2O)** | **0.5737** | **2.681** | **−0.055** | **1.380×** | **gana en punto a todas (nominal)** |
| ZeroR (clase mayoritaria) | 0.5657 | 2.206 | −0.098 | 1.303× | baseline trivial (techo) |
| B&H | 0.5657 | 2.206 | −0.098 | 1.303× | mercado alcista |

- **Honestidad (clave).** "Gana a todo" en **accuracy** es **NOMINAL**: McNemar **AutoML vs ZeroR p=0.902**,
  M10 vs ZeroR p=0.133 (n=251 → sin potencia; significancia de accuracy = línea futura). No se afirma batir al
  baseline/mercado.
- **Rescate del agente — SÍ significativo.** En accuracy: McNemar **AutoML vs M5 p=0.0002**, **M10 vs M5
  p=0.0074**, **M8 vs M5 p=0.0509**. En riesgo (bootstrap pareado **pooled**, 15 activos, n=3751): **M8 vs M5
  ΔSharpe +0.66 IC95[0.225,1.157]** y **ΔmaxDD +0.24 IC95[0.017,0.445]** (ambos excluyen 0). A nivel SPY el IC de
  riesgo aún cruza 0 (poca potencia); la significancia llega en el pooled.
- **Universalidad.** Cuota STRATA en SHAP: panel media 0.66; SPY 0.565 (mejor árbol) / 0.564 (permutation del
  ensemble). Top features SPY: garch_sigma, psa_score, ram_score, stress_prob → el ML redescubre STRATA.
- **STRATA sobre baseline simple (SPY).** Ablación por bloques de semillas: momentum solo acc 0.521 → STRATA7+mom
  0.582 (Δ +0.061; 3/5 bloques McNemar sig. 0.10). Fuente: `spy_ablation_robustness.json`.
- **Frase de cierre SPY:** "En SPY, AutoML supera en punto a todas las estrategias (0.574 vs 0.566), de forma
  nominal; el valor que sí sobrevive a un test es el rescate del agente (accuracy vs M5 y riesgo pooled) y la
  universalidad (el ML redescubre STRATA)."

---

## §2. Tabla del proyecto anterior (referencia histórica — NO usar en memoria LaTeX)

Las cifras que el proyecto anterior produjo al cierre (2026-06-07). El nuevo proyecto las ha replicado con mejor rigor (K=3 fijo, τ=0.5, walk-forward, accuracy-first). Si en la memoria aparece una cifra, debe venir de §1.

**Fuente primaria:** `_archivo_proyecto_anterior/outputs_canonicos/m{5,8,10}*.json` y `statistical_tests.json`.

| Estrategia | Accuracy | AUC | Log-loss | Brier | MCC | Sharpe | €1000→ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline "always long" | 0.566 | — | — | — | — | ≈ B&H | 1323 |
| M5 (agente solo) | 0.407 | 0.481 | 0.756 | 0.281 | −0.106 | **−1.83** | **903** |
| M8 (STRATA override C) | 0.460 | 0.471 | 1.640 | 0.312 | −0.090 | **+0.66** | **1064** |
| M10 (XGBoost CPCV) | 0.530 | 0.504 | 0.785 | 0.284 | +0.022 | **+0.69** | **1035** |

*Diferencias respecto al canónico actual: K no fijo vs K=3; τ=0.40 vs τ=0.5; N=400 vs 401; M10 accuracy 0.530 vs 0.539; M8 Sharpe 0.66 vs 0.67; DM p=0.75 vs 0.67. Las diferencias son menores y esperadas por el rediseño.*

### Métricas auxiliares del backtest (JSON canónico)

| Métrica | M5 | M8 | M10 |
|---|---:|---:|---:|
| Sortino | −2.37 | +0.99 | +1.09 |
| Max drawdown | −9.7% | −6.8% | −4.4% |
| Calmar | −0.64 | +0.59 | +0.48 |
| Profit factor | 0.70 | 1.15 | 1.17 |
| Hit rate (backtest) | 0.375 | 0.418 | 0.441 |
| Turnover | 0.139 | 0.224 | 0.174 |
| `n_obs` | 400 | 402 | 417 |

**Nota.** Los `n_obs` difieren ligeramente porque M10 usa CPCV-within-OOS y tiene splits que reciclan ciertos días; M5/M8 son secuenciales. Documentar en el notebook.

---

## Tests pareados

| Test | Comparación | Estadístico | p-valor | Lectura |
|---|---|---|---:|---|
| **McNemar** | M8 vs M5 (decisiones) | χ² pareado | **≈ 0.088** | STRATA rescata al agente con significancia borderline |
| **Diebold-Mariano** | Sharpe M10 vs M8 | DM | **≈ 0.75** | M10 y M8 indistinguibles |
| **Sign test** | M5 contra 0.5 | binomial | **< 0.001** | Agente solo peor que el azar |
| **Bootstrap estacionario** | Δ Sharpe(M10−M8) | IC95% Politis-Romano | **[−1.80, +1.15]** | El intervalo contiene cero |
| **Bootstrap** | P(M10 > M8) | porcentil | **≈ 0.543** | Moneda |

---

## SHAP global de M10 — top 5 features

Calculado con TreeSHAP nativo de XGBoost (`booster.predict(X, pred_contribs=True)`). Eficiencia de Shapley verificada: `max |sum(SHAP_j) + base − logit(p)| < 1e-6`.

| # | Feature | SHAP medio |
|---|---|---:|
| 1 | `ram_score` | **0.527** |
| 2 | `psa_score` | **0.428** |
| 3 | `garch_sigma` | 0.346 |
| 4 | `stress_prob` | 0.342 |
| 5 | `calm_prob` | 0.324 |

**Las 3 features STRATA + 2 de régimen ocupan el top 5. Ninguna personalidad del agente llega.**

Top features de cada personalidad (Buffett, Wood, Druckenmiller, Burry, Ackman) caen por debajo del puesto 10 en SHAP. Esto es el **argumento empírico** de que la regla a mano de STRATA captura la señal informativa.

---

## Análisis condicional (SPY OOS)

| Subset | N | M5 acc | M8 acc | M10 acc | Lectura |
|---|---:|---:|---:|---:|---|
| Régimen Crisis | ~85 | 35.7% | 35.7% | **60.7%** | XGBoost útil en crisis |
| Régimen Calma | ~250 | 41.1% | **57.0%** | 51.9% | STRATA-override óptimo en calma |
| RAM-flag high | 107 | 41.1% | **58.9%** | — | **M8 − M5 = +17.8 pp** cuando RAM dispara high |

Este último número (RAM-high → +17.8 pp) es **el dato más limpio para defensa**.

---

## Estabilidad temporal del umbral XGBoost

| Umbral `p1` | Sharpe mitad-1 (train) | Sharpe mitad-2 (test) |
|---:|---:|---:|
| 0.42 | +0.20 | **+1.07** |
| 0.50 | +0.41 | +0.09 |
| **0.565 (óptimo train)** | **+0.76** | +0.14 |
| 0.60 | +0.40 | −0.04 |

**Conclusión:** umbral aprendido por XGBoost no estable. STRATA con umbrales fijos calibrados ex-ante (RAM 0.20/0.40/0.70, PSA/GSO P95/P99) es estable por construcción. Argumento de **interpretabilidad como ventaja defendible** frente a meta-learners.

---

## Panel multi-activo (decision-level, 10 tickers, ~1406 intervenciones)

Datos en `_archivo_proyecto_anterior/outputs_canonicos/decision_level/`.

### Atribución de P&L por detector

| Detector | % P&L | bps totales |
|---|---:|---:|
| **RAM** | **98%** | +9218 |
| PSA | 2% | +185 |
| GSO | 0% | 0 |

### P&L atribuible por activo (sign test sobre mediana 0)

| Ticker | P&L (bps) | sign test p | McNemar p vs M5 |
|---|---:|---:|---:|
| SPY | +1740 | < 0.10 | ≈ 0.088 |
| XLE | +1840 | < 0.10 | — |
| NVDA | (+) | — | — |
| BAC | (+) | — | < 0.05 |
| TSLA | (+) | — | — |
| UNG | (+) | — | — |
| MSTR | (−) | — | (contraejemplo) |
| **SMCI** | (−) | — | **0.011 (contra M8)** |
| ROKU | (+) | — | — |
| MARA | (+) | — | — |

Hit rate M5 vs M8 **mejora en 8/10 activos**. Sign test panel `p ≈ 0.109` (borderline).

### Hallazgos del panel

- **GSO no dispara con severidad medium+ en NINGÚN activo del panel.** Hallazgo metodológico negativo.
- **SMCI = contraejemplo McNemar contra M8** (agente con información direccional complementaria al prior).
  *[Actualizado 2026-06-17: lectura del proyecto anterior, con signo de RAM hardcodeado (corregido). En el
  proyecto actual SMCI es el ACTIVO DEL CASO DE ESTUDIO del tutor — cifras canónicas en §1bis.]*
- **MSTR = `prior-flip` clásico** (signo calibración ≠ signo OOS).

---

## La narrativa de cierre (frase canónica — usar §14 del notebook)

> *"Un agente LLM perdedor direccional (38.4%, < azar, sign test p<0.001) es rescatado por supervisión estadística clásica: la accuracy sube 0.384 → 0.436 (regla M8) → 0.539 (XGBoost M10 sobre features STRATA), y regla a mano y caja negra son equivalentes en P&L (DM p=0.67). La señal informativa es la de STRATA: ablación sin features de régimen/RAM/PSA/GSO cae a Sharpe +0.21. Ningún sistema bate B&H pasivo (0.569 accuracy) — STRATA reduce el daño, no genera alfa. En el plano accuracy, M10 rescata al agente en ambos regímenes (bajista Holm p_adj=0.075, block-perm p=0.061; alcista Holm p_adj=0.005); M8 solo en alcista y nulo en bajista. En el plano Sharpe el rescate es condicional al alza para ambos modelos (walk-forward §13: ΔSharpe se invierte en bajista, M8=−3.92 / M10=−1.06, n=123 ≥ 60; cota Bonferroni M8−M5=−0.49 y M10−M5=−0.48 → H1_b False; falsificación pre-registrada disparada); el modelo K=3 sí generaliza inter-época (15/16 orígenes). La aportación es un protocolo de supervisión estadística interpretable que recupera accuracy direccional de un agente perdedor —robusto en ambos regímenes para M10— y delimita honestamente dónde funciona y dónde no."*

---

## Cómo verificar estas cifras desde el kit

```bash
cd /Users/Raquel/Desktop/STRATA_kit/
python3 -c "
import json
for f in ['m5_agent_alone', 'm8_strata_override', 'm10_ml_meta']:
    d = json.load(open(f'_archivo_proyecto_anterior/outputs_canonicos/{f}.json'))
    m = d['metrics']
    print(f'{f}: Sharpe={m[\"sharpe\"]:.3f}, equity={d[\"equity_final\"]:.4f}, hit={m[\"hit_rate\"]:.3f}, n={d[\"n_obs\"]}')
"
```

Output esperado:
```
m5_agent_alone:  Sharpe=-1.831, equity=0.9031, hit=0.375, n=400
m8_strata_override: Sharpe=+0.659, equity=1.0639, hit=0.418, n=402
m10_ml_meta: Sharpe=+0.692, equity=1.0353, hit=0.441, n=417
```

Si el nuevo proyecto produce cifras **dentro de ±10%** de estas, valida el resultado. Si difieren más, hay que entender por qué.
